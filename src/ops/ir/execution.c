#include "ops/ir/ir.h"
#include "ops/ir/execution.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/ir/graph_cache.h"
#include "ops/ir/schedule.h"
#include "core/logging.h"
#ifdef CML_HAS_LLVM_BACKEND
#include "ops/ir/llvm/llvm_backend.h"
#endif
#include "backend/blas.h"
#include "backend/threadpool.h"
#include "backend/device.h"
#include "ops/simd_utils.h"
#include "ops/simd_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __SSE__
#include <xmmintrin.h>
#include <emmintrin.h>
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

static CMLBlasContext* g_exec_blas_ctx = NULL;

#ifdef CML_HAS_LLVM_BACKEND
static CMLLLVMBackend* g_llvm_backend = NULL;

CMLLLVMBackend* cml_get_llvm_backend(void) {
    if (!g_llvm_backend) {
        g_llvm_backend = cml_llvm_backend_init();
    }
    return g_llvm_backend;
}
#endif

#define BUFFER_CACHE_MIN_BUCKET 6  // 64 bytes (2^6)
#define BUFFER_CACHE_MAX_BUCKET 26 // 64 MB (2^26)
#define BUFFER_CACHE_NUM_BUCKETS (BUFFER_CACHE_MAX_BUCKET - BUFFER_CACHE_MIN_BUCKET + 1)
#define BUFFER_CACHE_MAX_PER_BUCKET 8 // Max cached buffers per size bucket

typedef struct CachedBuffer {
    void* data;
    struct CachedBuffer* next;
} CachedBuffer;

typedef struct {
    CachedBuffer* free_list; // Linked list of free buffers
    int count;               // Number of buffers in list
    size_t bucket_size;      // Size of buffers in this bucket
} BufferBucket;

typedef struct {
    BufferBucket buckets[BUFFER_CACHE_NUM_BUCKETS];
    size_t cache_hits;
    size_t cache_misses;
    size_t bytes_cached;
    size_t bytes_allocated;
    bool initialized;
} BufferCache;

static BufferCache g_buffer_cache = {0};

static int get_bucket_index(size_t size) {
    if (size == 0)
        return -1;

    int bucket = 0;
    size_t s   = size - 1;
    while (s > 0) {
        s >>= 1;
        bucket++;
    }

    if (bucket < BUFFER_CACHE_MIN_BUCKET)
        bucket = BUFFER_CACHE_MIN_BUCKET;
    if (bucket > BUFFER_CACHE_MAX_BUCKET)
        return -1; // Too large

    return bucket - BUFFER_CACHE_MIN_BUCKET;
}

static void init_buffer_cache(void) {
    if (g_buffer_cache.initialized)
        return;

    for (int i = 0; i < BUFFER_CACHE_NUM_BUCKETS; i++) {
        g_buffer_cache.buckets[i].free_list   = NULL;
        g_buffer_cache.buckets[i].count       = 0;
        g_buffer_cache.buckets[i].bucket_size = (size_t)1 << (i + BUFFER_CACHE_MIN_BUCKET);
    }
    g_buffer_cache.initialized = true;
}

void* cml_buffer_cache_alloc(size_t size) {
    if (size == 0)
        return NULL;

    init_buffer_cache();

    int bucket_idx = get_bucket_index(size);
    if (bucket_idx < 0) {
        g_buffer_cache.cache_misses++;
        g_buffer_cache.bytes_allocated += size;
        return calloc(1, size);
    }

    BufferBucket* bucket = &g_buffer_cache.buckets[bucket_idx];

    if (bucket->free_list) {
        CachedBuffer* cached = bucket->free_list;
        bucket->free_list    = cached->next;
        bucket->count--;

        void* data = cached->data;
        free(cached);

        g_buffer_cache.cache_hits++;
        g_buffer_cache.bytes_cached -= bucket->bucket_size;

        memset(data, 0, size);
        return data;
    }

    g_buffer_cache.cache_misses++;
    g_buffer_cache.bytes_allocated += bucket->bucket_size;
    return calloc(1, bucket->bucket_size);
}

void cml_buffer_cache_free(void* ptr, size_t size) {
    if (!ptr)
        return;

    init_buffer_cache();

    int bucket_idx = get_bucket_index(size);
    if (bucket_idx < 0) {
        free(ptr);
        return;
    }

    BufferBucket* bucket = &g_buffer_cache.buckets[bucket_idx];

    if (bucket->count >= BUFFER_CACHE_MAX_PER_BUCKET) {
        free(ptr);
        return;
    }

    CachedBuffer* cached = (CachedBuffer*)malloc(sizeof(CachedBuffer));
    if (!cached) {
        free(ptr);
        return;
    }

    cached->data      = ptr;
    cached->next      = bucket->free_list;
    bucket->free_list = cached;
    bucket->count++;
    g_buffer_cache.bytes_cached += bucket->bucket_size;
}

void cml_cleanup_buffer_cache(void) {
    if (!g_buffer_cache.initialized)
        return;

    for (int i = 0; i < BUFFER_CACHE_NUM_BUCKETS; i++) {
        BufferBucket* bucket  = &g_buffer_cache.buckets[i];
        CachedBuffer* current = bucket->free_list;
        while (current) {
            CachedBuffer* next = current->next;
            free(current->data);
            free(current);
            current = next;
        }
        bucket->free_list = NULL;
        bucket->count     = 0;
    }

    g_buffer_cache.bytes_cached = 0;
    g_buffer_cache.initialized  = false;
}

void cml_print_buffer_cache_stats(void) {
    printf("Buffer Cache Stats:\n");
    if (!g_buffer_cache.initialized) {
        printf("  (Not initialized)\n");
        return;
    }

    size_t total_cached = 0;
    int total_buffers   = 0;
    for (int i = 0; i < BUFFER_CACHE_NUM_BUCKETS; i++) {
        if (g_buffer_cache.buckets[i].count > 0) {
            total_buffers += g_buffer_cache.buckets[i].count;
            total_cached += g_buffer_cache.buckets[i].count * g_buffer_cache.buckets[i].bucket_size;
        }
    }

    size_t total_requests = g_buffer_cache.cache_hits + g_buffer_cache.cache_misses;
    float hit_rate = total_requests > 0 ? (100.0f * g_buffer_cache.cache_hits / total_requests) : 0;

    printf("  Cache hits:    %zu\n", g_buffer_cache.cache_hits);
    printf("  Cache misses:  %zu\n", g_buffer_cache.cache_misses);
    printf("  Hit rate:      %.1f%%\n", hit_rate);
    printf("  Cached now:    %d buffers (%.2f KB)\n", total_buffers, total_cached / 1024.0f);
    printf("  Total alloc:   %.2f KB\n", g_buffer_cache.bytes_allocated / 1024.0f);
}

CMLBlasContext* get_blas_context(void) {
    if (!g_exec_blas_ctx) {
        g_exec_blas_ctx = cml_blas_init();
        if (g_exec_blas_ctx) {
            LOG_INFO("BLAS acceleration enabled: %s", cml_blas_get_library_name(g_exec_blas_ctx));
        }
    }
    return g_exec_blas_ctx;
}

/**
 * @file execution.c
 * @brief IR execution with CPU interpreter and optional LLVM JIT
 */

// Numpy-style broadcast index: given a flat index in the output tensor,
// compute the corresponding flat index in a (possibly smaller) input tensor.
// Handles cases like [N,M] op [N,1] or [N,M] op [1,M] correctly.
static inline size_t _broadcast_idx(Tensor* inp, Tensor* out_t, size_t flat_i) {
    if (inp->numel == 1)
        return 0;
    if (inp->numel == out_t->numel)
        return flat_i;

    // Decompose flat_i into multi-dim indices using output shape,
    // then map to input using min(dim, 1) for broadcast dims
    size_t idx       = 0;
    size_t remaining = flat_i;

    // Precompute output strides
    int ndim = out_t->ndim;
    size_t out_strides[8]; // max 8 dims
    out_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--)
        out_strides[d] = out_strides[d + 1] * (size_t)out_t->shape[d + 1];

    // Compute input strides (only for non-broadcast dims)
    size_t inp_strides[8];
    int inp_ndim = inp->ndim;
    if (inp_ndim > 0) {
        inp_strides[inp_ndim - 1] = 1;
        for (int d = inp_ndim - 2; d >= 0; d--)
            inp_strides[d] = inp_strides[d + 1] * (size_t)inp->shape[d + 1];
    }

    // Right-align dimensions (numpy broadcasting rules)
    idx = 0;
    for (int d = 0; d < ndim; d++) {
        size_t coord = remaining / out_strides[d];
        remaining %= out_strides[d];

        // Map to input dimension (right-aligned)
        int inp_d = d - (ndim - inp_ndim);
        if (inp_d >= 0 && inp_d < inp_ndim) {
            if (inp->shape[inp_d] > 1) {
                idx += coord * inp_strides[inp_d];
            }
            // If inp->shape[inp_d] == 1, this dim is broadcast -- contributes 0
        }
    }
    return idx;
}

int cpu_execute_node(struct IRNode* node) {
    if (!node || !node->output) {
        return -1;
    }

    Tensor* out = node->output;

    if (!out->data && out->numel > 0) {
        size_t size = out->numel * cml_dtype_size(out->dtype);
        out->data   = cml_buffer_cache_alloc(size);
        if (!out->data) {
            LOG_ERROR("Failed to allocate output tensor data");
            return -1;
        }
        out->owns_data = true;
    }

    float* out_data = (float*)out->data;

    float* in1_data  = NULL;
    float* in2_data  = NULL;
    size_t in1_numel = 0;
    size_t in2_numel = 0;

    if (node->num_inputs >= 1 && node->inputs && node->inputs[0]) {
        in1_data  = (float*)node->inputs[0]->data;
        in1_numel = node->inputs[0]->numel;
    }
    if (node->num_inputs >= 2 && node->inputs && node->inputs[1]) {
        in2_data  = (float*)node->inputs[1]->data;
        in2_numel = node->inputs[1]->numel;
    }

// Helper macro for numpy-style broadcast indexing
// For output flat index i, compute the corresponding flat index in a tensor
// with potentially fewer elements (broadcast dimensions have size 1)
#define BROADCAST_IDX(tensor_ptr, out_ptr, flat_i) _broadcast_idx(tensor_ptr, out_ptr, flat_i)

    switch (node->type) {
    case UOP_ADD:
        if (!in1_data || !in2_data)
            return -1;
        if (in1_numel == in2_numel && in1_numel == out->numel) {
            simd_add_f32(in1_data, in2_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                size_t i1   = BROADCAST_IDX(node->inputs[0], out, i);
                size_t i2   = BROADCAST_IDX(node->inputs[1], out, i);
                out_data[i] = in1_data[i1] + in2_data[i2];
            }
        }
        break;

    case UOP_SUB:
        if (!in1_data || !in2_data)
            return -1;
        if (in1_numel == in2_numel && in1_numel == out->numel) {
            simd_sub_f32(in1_data, in2_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                size_t i1   = BROADCAST_IDX(node->inputs[0], out, i);
                size_t i2   = BROADCAST_IDX(node->inputs[1], out, i);
                out_data[i] = in1_data[i1] - in2_data[i2];
            }
        }
        break;

    case UOP_MUL:
        if (!in1_data || !in2_data)
            return -1;
        if (in1_numel == in2_numel && in1_numel == out->numel) {
            simd_mul_f32(in1_data, in2_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                size_t i1   = BROADCAST_IDX(node->inputs[0], out, i);
                size_t i2   = BROADCAST_IDX(node->inputs[1], out, i);
                out_data[i] = in1_data[i1] * in2_data[i2];
            }
        }
        break;

    case UOP_DIV:
        if (!in1_data || !in2_data)
            return -1;
        if (in1_numel == in2_numel && in1_numel == out->numel) {
            simd_div_f32(in1_data, in2_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                size_t i1   = BROADCAST_IDX(node->inputs[0], out, i);
                size_t i2   = BROADCAST_IDX(node->inputs[1], out, i);
                out_data[i] = in1_data[i1] / (in2_data[i2] + 1e-8f);
            }
        }
        break;

    case UOP_NEG:
        if (!in1_data)
            return -1;
        if (in1_numel == out->numel) {
            simd_neg_f32(in1_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                out_data[i] = -in1_data[i % in1_numel];
            }
        }
        break;

    case UOP_EXP:
        if (!in1_data)
            return -1;
        if (in1_numel == out->numel) {
            simd_exp_f32(in1_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                out_data[i] = expf(in1_data[i % in1_numel]);
            }
        }
        break;

    case UOP_LOG:
        if (!in1_data)
            return -1;
        if (in1_numel == out->numel) {
            simd_log_f32(in1_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                out_data[i] = logf(in1_data[i % in1_numel] + 1e-8f);
            }
        }
        break;

    case UOP_SQRT:
        if (!in1_data)
            return -1;
        if (in1_numel == out->numel) {
            simd_sqrt_f32(in1_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                out_data[i] = sqrtf(fabsf(in1_data[i % in1_numel]));
            }
        }
        break;

    case UOP_ABS:
        if (!in1_data)
            return -1;
        if (in1_numel == out->numel) {
            simd_abs_f32(in1_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                out_data[i] = fabsf(in1_data[i % in1_numel]);
            }
        }
        break;

    case UOP_SIGMOID:
        if (!in1_data)
            return -1;
        if (in1_numel == out->numel) {
            simd_sigmoid_f32(in1_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                float x     = in1_data[i % in1_numel];
                out_data[i] = 1.0f / (1.0f + expf(-x));
            }
        }
        break;

    case UOP_TANH:
        if (!in1_data)
            return -1;
        if (in1_numel == out->numel) {
            simd_tanh_f32(in1_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                out_data[i] = tanhf(in1_data[i % in1_numel]);
            }
        }
        break;

    case UOP_SIN:
        if (!in1_data)
            return -1;
        if (in1_numel == out->numel) {
            simd_sin_f32(in1_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                out_data[i] = sinf(in1_data[i % in1_numel]);
            }
        }
        break;

    case UOP_COS:
        if (!in1_data)
            return -1;
        if (in1_numel == out->numel) {
            simd_cos_f32(in1_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                out_data[i] = cosf(in1_data[i % in1_numel]);
            }
        }
        break;

    case UOP_TAN:
        if (!in1_data)
            return -1;
        if (in1_numel == out->numel) {
            simd_tan_f32(in1_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                out_data[i] = tanf(in1_data[i % in1_numel]);
            }
        }
        break;

    case UOP_RECIP:
        if (!in1_data)
            return -1;
        if (in1_numel == out->numel) {
            simd_recip_f32(in1_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                float x     = in1_data[i % in1_numel];
                out_data[i] = (x != 0.0f) ? (1.0f / x) : 0.0f;
            }
        }
        break;

    case UOP_POW:
        if (!in1_data || !in2_data)
            return -1;
        if (in1_numel == out->numel && in2_numel == out->numel) {
            simd_pow_f32(in1_data, in2_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                out_data[i] = powf(in1_data[i % in1_numel], in2_data[i % in2_numel]);
            }
        }
        break;

    case UOP_SUM:
    case UOP_MEAN: {
        if (!in1_data)
            return -1;
        ReduceParams* rp = (ReduceParams*)node->params;
        Tensor* inp      = node->inputs[0];

        /* Check if this is a per-dimension reduction (not global) */
        if (rp && rp->num_dims == 1 && inp->ndim >= 2) {
            int reduce_dim = rp->dims[0];
            if (reduce_dim < 0)
                reduce_dim += inp->ndim;

            /* For 2D: reduce along dim => iterate over other dims */
            if (inp->ndim == 2) {
                int rows = inp->shape[0];
                int cols = inp->shape[1];
                if (reduce_dim == 1) {
                    /* Reduce along cols: output [rows] or [rows,1] */
                    for (int r = 0; r < rows; r++) {
                        float acc = 0.0f;
                        for (int c = 0; c < cols; c++)
                            acc += in1_data[r * cols + c];
                        if (node->type == UOP_MEAN && cols > 0)
                            acc /= (float)cols;
                        out_data[r] = acc;
                    }
                } else { /* reduce_dim == 0 */
                    /* Reduce along rows: output [cols] or [1,cols] */
                    for (int c = 0; c < cols; c++) {
                        float acc = 0.0f;
                        for (int r = 0; r < rows; r++)
                            acc += in1_data[r * cols + c];
                        if (node->type == UOP_MEAN && rows > 0)
                            acc /= (float)rows;
                        out_data[c] = acc;
                    }
                }
            } else {
                /* Generic N-dim: compute strides and iterate */
                /* For now, fall back to global reduction for >2D */
                float sum = simd_sum_float(in1_data, in1_numel);
                if (node->type == UOP_MEAN && in1_numel > 0)
                    sum /= (float)in1_numel;
                out_data[0] = sum;
            }
        } else {
            /* Global reduction */
            float sum = simd_sum_float(in1_data, in1_numel);
            if (node->type == UOP_MEAN && in1_numel > 0) {
                sum /= (float)in1_numel;
            }
            out_data[0] = sum;
        }
        break;
    }

    case UOP_MAX_REDUCE: {
        if (!in1_data || in1_numel == 0)
            return -1;
        ReduceParams* rp = (ReduceParams*)node->params;
        Tensor* inp      = node->inputs[0];

        /* Check if this is a per-dimension reduction (not global) */
        if (rp && rp->num_dims == 1 && inp->ndim == 2) {
            int reduce_dim = rp->dims[0];
            if (reduce_dim < 0)
                reduce_dim += inp->ndim;
            int rows = inp->shape[0];
            int cols = inp->shape[1];

            if (reduce_dim == 1) {
                /* Max along cols: output [rows] or [rows,1] */
                for (int r = 0; r < rows; r++) {
                    float mx = in1_data[r * cols];
                    for (int c = 1; c < cols; c++) {
                        float v = in1_data[r * cols + c];
                        if (v > mx)
                            mx = v;
                    }
                    out_data[r] = mx;
                }
            } else { /* reduce_dim == 0 */
                /* Max along rows: output [cols] or [1,cols] */
                for (int c = 0; c < cols; c++) {
                    float mx = in1_data[c];
                    for (int r = 1; r < rows; r++) {
                        float v = in1_data[r * cols + c];
                        if (v > mx)
                            mx = v;
                    }
                    out_data[c] = mx;
                }
            }
        } else {
            /* Global max */
            out_data[0] = simd_max_float(in1_data, in1_numel);
        }
        break;
    }

    case UOP_MATMUL: {
        if (!in1_data || !in2_data)
            return -1;
        // Simple 2D matmul: [M,K] x [K,N] = [M,N]
        Tensor* a = node->inputs[0];
        Tensor* b = node->inputs[1];
        if (a->ndim < 2 || b->ndim < 2)
            return -1;

        int M = a->shape[a->ndim - 2];
        int K = a->shape[a->ndim - 1];
        int N = b->shape[b->ndim - 1];

        // Zero output first
        memset(out_data, 0, out->numel * sizeof(float));

        // Try BLAS acceleration
        CMLBlasContext* blas = get_blas_context();
        if (blas && blas->initialized) {
            // Use BLAS sgemm: C = alpha * A @ B + beta * C
            // alpha = 1.0, beta = 0.0
            int result = cml_blas_sgemm(blas, in1_data, in2_data, out_data, M, N, K, 1.0f, 0.0f);
            if (result == 0) {
                break; // BLAS succeeded
            }
            // Fall through to naive implementation if BLAS failed
            LOG_WARNING("BLAS sgemm failed, falling back to naive matmul");
        }

        // Naive matmul fallback with cache-friendly access pattern
        // Use blocking for better cache utilization
        const int BLOCK = 32;
        for (int m0 = 0; m0 < M; m0 += BLOCK) {
            for (int n0 = 0; n0 < N; n0 += BLOCK) {
                for (int k0 = 0; k0 < K; k0 += BLOCK) {
                    int m_end = (m0 + BLOCK < M) ? m0 + BLOCK : M;
                    int n_end = (n0 + BLOCK < N) ? n0 + BLOCK : N;
                    int k_end = (k0 + BLOCK < K) ? k0 + BLOCK : K;

                    for (int m = m0; m < m_end; m++) {
                        for (int k = k0; k < k_end; k++) {
                            float a_mk = in1_data[m * K + k];
                            for (int n = n0; n < n_end; n++) {
                                out_data[m * N + n] += a_mk * in2_data[k * N + n];
                            }
                        }
                    }
                }
            }
        }
        break;
    }

    case UOP_CMPLT:
        if (!in1_data || !in2_data)
            return -1;
        if (in1_numel == in2_numel && in1_numel == out->numel) {
            simd_cmplt_f32(in1_data, in2_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                size_t i1   = (in1_numel == 1) ? 0 : i % in1_numel;
                size_t i2   = (in2_numel == 1) ? 0 : i % in2_numel;
                out_data[i] = (in1_data[i1] < in2_data[i2]) ? 1.0f : 0.0f;
            }
        }
        break;

    case UOP_MAX:
        if (!in1_data || !in2_data)
            return -1;
        if (in1_numel == in2_numel && in1_numel == out->numel) {
            simd_max_f32(in1_data, in2_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                size_t i1   = (in1_numel == 1) ? 0 : i % in1_numel;
                size_t i2   = (in2_numel == 1) ? 0 : i % in2_numel;
                out_data[i] = (in1_data[i1] > in2_data[i2]) ? in1_data[i1] : in2_data[i2];
            }
        }
        break;

    case UOP_WHERE: {
        // where(cond, a, b) - select from a where cond is true, else b
        if (node->num_inputs < 3)
            return -1;
        float* cond_data = (float*)node->inputs[0]->data;
        float* a_data    = (float*)node->inputs[1]->data;
        float* b_data    = (float*)node->inputs[2]->data;
        if (!cond_data || !a_data || !b_data)
            return -1;

        size_t cond_numel = node->inputs[0]->numel;
        size_t a_numel    = node->inputs[1]->numel;
        size_t b_numel    = node->inputs[2]->numel;

        // Fast path: all same size, use SIMD
        if (cond_numel == a_numel && a_numel == b_numel && a_numel == out->numel) {
            simd_where_f32(cond_data, a_data, b_data, out_data, out->numel);
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                size_t ci   = (cond_numel == 1) ? 0 : i % cond_numel;
                size_t ai   = (a_numel == 1) ? 0 : i % a_numel;
                size_t bi   = (b_numel == 1) ? 0 : i % b_numel;
                out_data[i] = (cond_data[ci] != 0.0f) ? a_data[ai] : b_data[bi];
            }
        }
        break;
    }

    case UOP_PERMUTE: {
        // 2D transpose: swap rows and columns
        Tensor* in = node->inputs[0];
        if (!in || !in->data || in->ndim != 2) {
            LOG_WARNING("CPU fallback: UOP_PERMUTE requires 2D input");
            for (size_t i = 0; i < out->numel; i++) {
                out_data[i] = 0.0f;
            }
            break;
        }

        float* in_data_perm = (float*)in->data;
        int rows            = in->shape[0];
        int cols            = in->shape[1];

        // Use cache-blocked SIMD transpose
        simd_transpose_f32(in_data_perm, out_data, rows, cols);
        break;
    }

    case UOP_FILL: {
        // Fill tensor with constant value
        FillParams* params = (FillParams*)node->params;
        float value        = params ? params->value : 0.0f;
        for (size_t i = 0; i < out->numel; i++) {
            out_data[i] = value;
        }
        break;
    }

    case UOP_GATHER: {
        // Gather elements by index: out[i] = input[i, indices[i]]
        // For cross-entropy: input is [N, C], indices is [N], output is [N]
        if (node->num_inputs < 2)
            return -1;
        float* input_data = (float*)node->inputs[0]->data;
        float* index_data = (float*)node->inputs[1]->data;
        if (!input_data || !index_data)
            return -1;

        GatherParams* params = (GatherParams*)node->params;
        int dim              = params ? params->dim : -1;
        Tensor* input        = node->inputs[0];
        Tensor* indices      = node->inputs[1];

        // Handle negative dim
        if (dim < 0)
            dim = input->ndim + dim;

        // For 2D input [N, C] with dim=-1 (last dim) and 1D indices [N]:
        // out[i] = input[i * C + indices[i]]
        if (input->ndim == 2 && indices->ndim == 1 && dim == 1) {
            size_t n_rows = (size_t)input->shape[0];
            size_t n_cols = (size_t)input->shape[1];
            for (size_t i = 0; i < n_rows && i < out->numel; i++) {
                int idx = (int)index_data[i];
                if (idx < 0 || idx >= (int)n_cols) {
                    LOG_ERROR("UOP_GATHER: index %d out of bounds [0, %zu)", idx, n_cols);
                    return -1;
                }
                out_data[i] = input_data[i * n_cols + (size_t)idx];
            }
        } else {
            // Generic N-dimensional gather (fallback - slower)
            LOG_WARNING("UOP_GATHER: using generic N-dim implementation");
            // For now, just handle the most common case above
            // More complex cases can be added later
            for (size_t i = 0; i < out->numel; i++) {
                out_data[i] = 0.0f;
            }
        }
        break;
    }

    case UOP_SIGN:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            float x = in1_data[i % in1_numel];
            out_data[i] = (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
        }
        break;

    case UOP_FLOOR:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = floorf(in1_data[i % in1_numel]);
        break;

    case UOP_CEIL:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = ceilf(in1_data[i % in1_numel]);
        break;

    case UOP_ROUND:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = roundf(in1_data[i % in1_numel]);
        break;

    case UOP_LOG2:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = log2f(in1_data[i % in1_numel] + 1e-8f);
        break;

    case UOP_EXP2:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = exp2f(in1_data[i % in1_numel]);
        break;

    case UOP_ASIN:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = asinf(in1_data[i % in1_numel]);
        break;

    case UOP_ACOS:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = acosf(in1_data[i % in1_numel]);
        break;

    case UOP_ATAN:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = atanf(in1_data[i % in1_numel]);
        break;

    case UOP_SQUARE:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            float x = in1_data[i % in1_numel];
            out_data[i] = x * x;
        }
        break;

    case UOP_RSQRT:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            float x = in1_data[i % in1_numel];
            out_data[i] = 1.0f / sqrtf(fabsf(x) + 1e-8f);
        }
        break;

    case UOP_ERF:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = erff(in1_data[i % in1_numel]);
        break;

    case UOP_CLAMP: {
        if (!in1_data) return -1;
        ClampParams* cp = (ClampParams*)node->params;
        float mn = cp ? cp->min_val : -INFINITY;
        float mx = cp ? cp->max_val : INFINITY;
        for (size_t i = 0; i < out->numel; i++) {
            float x = in1_data[i % in1_numel];
            if (x < mn) x = mn;
            if (x > mx) x = mx;
            out_data[i] = x;
        }
        break;
    }

    case UOP_PROD: {
        if (!in1_data) return -1;
        ReduceParams* rp = (ReduceParams*)node->params;
        Tensor* inp = node->inputs[0];

        if (rp && rp->num_dims == 1 && inp->ndim == 2) {
            int reduce_dim = rp->dims[0];
            if (reduce_dim < 0) reduce_dim += inp->ndim;
            int rows = inp->shape[0];
            int cols = inp->shape[1];

            if (reduce_dim == 1) {
                for (int r = 0; r < rows; r++) {
                    float prod = 1.0f;
                    for (int c = 0; c < cols; c++)
                        prod *= in1_data[r * cols + c];
                    out_data[r] = prod;
                }
            } else {
                for (int c = 0; c < cols; c++) {
                    float prod = 1.0f;
                    for (int r = 0; r < rows; r++)
                        prod *= in1_data[r * cols + c];
                    out_data[c] = prod;
                }
            }
        } else {
            // Global product
            float prod = 1.0f;
            for (size_t i = 0; i < in1_numel; i++)
                prod *= in1_data[i];
            out_data[0] = prod;
        }
        break;
    }

    case UOP_ARGMAX: {
        if (!in1_data || in1_numel == 0) return -1;
        ReduceParams* rp = (ReduceParams*)node->params;
        Tensor* inp = node->inputs[0];

        if (rp && rp->num_dims == 1 && inp->ndim == 2) {
            int reduce_dim = rp->dims[0];
            if (reduce_dim < 0) reduce_dim += inp->ndim;
            int rows = inp->shape[0];
            int cols = inp->shape[1];

            if (reduce_dim == 1) {
                for (int r = 0; r < rows; r++) {
                    float mx = in1_data[r * cols];
                    int idx = 0;
                    for (int c = 1; c < cols; c++) {
                        if (in1_data[r * cols + c] > mx) {
                            mx = in1_data[r * cols + c];
                            idx = c;
                        }
                    }
                    out_data[r] = (float)idx;
                }
            } else {
                for (int c = 0; c < cols; c++) {
                    float mx = in1_data[c];
                    int idx = 0;
                    for (int r = 1; r < rows; r++) {
                        if (in1_data[r * cols + c] > mx) {
                            mx = in1_data[r * cols + c];
                            idx = r;
                        }
                    }
                    out_data[c] = (float)idx;
                }
            }
        } else {
            // Global argmax
            float mx = in1_data[0];
            int idx = 0;
            for (size_t i = 1; i < in1_numel; i++) {
                if (in1_data[i] > mx) { mx = in1_data[i]; idx = (int)i; }
            }
            out_data[0] = (float)idx;
        }
        break;
    }

    case UOP_ARGMIN: {
        if (!in1_data || in1_numel == 0) return -1;
        ReduceParams* rp = (ReduceParams*)node->params;
        Tensor* inp = node->inputs[0];

        if (rp && rp->num_dims == 1 && inp->ndim == 2) {
            int reduce_dim = rp->dims[0];
            if (reduce_dim < 0) reduce_dim += inp->ndim;
            int rows = inp->shape[0];
            int cols = inp->shape[1];

            if (reduce_dim == 1) {
                for (int r = 0; r < rows; r++) {
                    float mn = in1_data[r * cols];
                    int idx = 0;
                    for (int c = 1; c < cols; c++) {
                        if (in1_data[r * cols + c] < mn) {
                            mn = in1_data[r * cols + c];
                            idx = c;
                        }
                    }
                    out_data[r] = (float)idx;
                }
            } else {
                for (int c = 0; c < cols; c++) {
                    float mn = in1_data[c];
                    int idx = 0;
                    for (int r = 1; r < rows; r++) {
                        if (in1_data[r * cols + c] < mn) {
                            mn = in1_data[r * cols + c];
                            idx = r;
                        }
                    }
                    out_data[c] = (float)idx;
                }
            }
        } else {
            float mn = in1_data[0];
            int idx = 0;
            for (size_t i = 1; i < in1_numel; i++) {
                if (in1_data[i] < mn) { mn = in1_data[i]; idx = (int)i; }
            }
            out_data[0] = (float)idx;
        }
        break;
    }

    case UOP_CUMSUM: {
        if (!in1_data) return -1;
        CumsumParams* cp = (CumsumParams*)node->params;
        Tensor* inp = node->inputs[0];
        int dim = cp ? cp->dim : 0;
        if (dim < 0) dim += inp->ndim;

        if (inp->ndim == 1) {
            float acc = 0.0f;
            for (size_t i = 0; i < in1_numel; i++) {
                acc += in1_data[i];
                out_data[i] = acc;
            }
        } else if (inp->ndim == 2) {
            int rows = inp->shape[0];
            int cols = inp->shape[1];
            if (dim == 1) {
                for (int r = 0; r < rows; r++) {
                    float acc = 0.0f;
                    for (int c = 0; c < cols; c++) {
                        acc += in1_data[r * cols + c];
                        out_data[r * cols + c] = acc;
                    }
                }
            } else {
                for (int c = 0; c < cols; c++) {
                    float acc = 0.0f;
                    for (int r = 0; r < rows; r++) {
                        acc += in1_data[r * cols + c];
                        out_data[r * cols + c] = acc;
                    }
                }
            }
        } else {
            // Fallback: just copy for >2D
            memcpy(out_data, in1_data, in1_numel * sizeof(float));
        }
        break;
    }

    case UOP_TRIU: {
        if (!in1_data) return -1;
        TriParams* tp = (TriParams*)node->params;
        int diagonal = tp ? tp->diagonal : 0;
        Tensor* inp = node->inputs[0];
        int rows = inp->shape[inp->ndim - 2];
        int cols = inp->shape[inp->ndim - 1];
        size_t batch = in1_numel / (size_t)(rows * cols);

        for (size_t b = 0; b < batch; b++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    size_t idx = b * (size_t)(rows * cols) + (size_t)(r * cols + c);
                    out_data[idx] = (c >= r + diagonal) ? in1_data[idx] : 0.0f;
                }
            }
        }
        break;
    }

    case UOP_TRIL: {
        if (!in1_data) return -1;
        TriParams* tp = (TriParams*)node->params;
        int diagonal = tp ? tp->diagonal : 0;
        Tensor* inp = node->inputs[0];
        int rows = inp->shape[inp->ndim - 2];
        int cols = inp->shape[inp->ndim - 1];
        size_t batch = in1_numel / (size_t)(rows * cols);

        for (size_t b = 0; b < batch; b++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    size_t idx = b * (size_t)(rows * cols) + (size_t)(r * cols + c);
                    out_data[idx] = (c <= r + diagonal) ? in1_data[idx] : 0.0f;
                }
            }
        }
        break;
    }

    case UOP_PAD: {
        if (!in1_data) return -1;
        PadParams* pp = (PadParams*)node->params;
        Tensor* inp = node->inputs[0];
        PadMode pad_mode = pp->mode;

        if (pad_mode == PAD_REFLECT || pad_mode == PAD_REPLICATE) {
            // Reflect or replicate padding
            if (inp->ndim == 1) {
                int pad_before = pp->pad_widths[0];
                int in_len = inp->shape[0];
                int out_len = out->shape[0];
                for (int i = 0; i < out_len; i++) {
                    int src = i - pad_before;
                    if (src < 0)
                        src = (pad_mode == PAD_REFLECT) ? -src : 0;
                    else if (src >= in_len)
                        src = (pad_mode == PAD_REFLECT) ? 2 * in_len - 2 - src : in_len - 1;
                    if (src < 0) src = 0;
                    if (src >= in_len) src = in_len - 1;
                    out_data[i] = in1_data[src];
                }
            } else if (inp->ndim == 2) {
                int pad_r_before = pp->pad_widths[0];
                int pad_c_before = pp->pad_widths[2];
                int in_rows = inp->shape[0], in_cols = inp->shape[1];
                int out_rows = out->shape[0], out_cols = out->shape[1];
                for (int r = 0; r < out_rows; r++) {
                    int sr = r - pad_r_before;
                    if (sr < 0) sr = (pad_mode == PAD_REFLECT) ? -sr : 0;
                    else if (sr >= in_rows) sr = (pad_mode == PAD_REFLECT) ? 2 * in_rows - 2 - sr : in_rows - 1;
                    if (sr < 0) sr = 0;
                    if (sr >= in_rows) sr = in_rows - 1;
                    for (int c = 0; c < out_cols; c++) {
                        int sc = c - pad_c_before;
                        if (sc < 0) sc = (pad_mode == PAD_REFLECT) ? -sc : 0;
                        else if (sc >= in_cols) sc = (pad_mode == PAD_REFLECT) ? 2 * in_cols - 2 - sc : in_cols - 1;
                        if (sc < 0) sc = 0;
                        if (sc >= in_cols) sc = in_cols - 1;
                        out_data[r * out_cols + c] = in1_data[sr * in_cols + sc];
                    }
                }
            } else {
                LOG_WARNING("UOP_PAD: reflect/replicate only for 1D/2D, falling back to constant");
                for (size_t i = 0; i < out->numel; i++) out_data[i] = 0.0f;
                for (size_t i = 0; i < in1_numel && i < out->numel; i++) out_data[i] = in1_data[i];
            }
        } else {
            // PAD_CONSTANT (default)
            for (size_t i = 0; i < out->numel; i++)
                out_data[i] = pp->value;

            if (inp->ndim == 1) {
                int pad_before = pp->pad_widths[0];
                for (int i = 0; i < inp->shape[0]; i++)
                    out_data[pad_before + i] = in1_data[i];
            } else if (inp->ndim == 2) {
                int pad_r_before = pp->pad_widths[0];
                int pad_c_before = pp->pad_widths[2];
                int in_rows = inp->shape[0], in_cols = inp->shape[1];
                int out_cols = out->shape[1];
                for (int r = 0; r < in_rows; r++)
                    for (int c = 0; c < in_cols; c++)
                        out_data[(pad_r_before + r) * out_cols + (pad_c_before + c)] =
                            in1_data[r * in_cols + c];
            } else {
                // Generic N-dim: copy element by element
                LOG_WARNING("UOP_PAD: generic N-dim padding, may be slow");
                for (size_t i = 0; i < in1_numel && i < out->numel; i++)
                    out_data[i] = in1_data[i];
            }
        }
        break;
    }

    case UOP_SORT: {
        if (!in1_data) return -1;
        SortParams* sp = (SortParams*)node->params;
        Tensor* inp = node->inputs[0];
        int dim = sp ? sp->dim : -1;
        bool desc = sp ? sp->descending : false;
        if (dim < 0) dim += inp->ndim;

        memcpy(out_data, in1_data, in1_numel * sizeof(float));

        if (inp->ndim == 1) {
            int n = inp->shape[0];
            for (int i = 1; i < n; i++) {
                float key = out_data[i];
                int j = i - 1;
                while (j >= 0 && (desc ? out_data[j] < key : out_data[j] > key)) {
                    out_data[j + 1] = out_data[j]; j--;
                }
                out_data[j + 1] = key;
            }
        } else if (inp->ndim == 2) {
            int rows = inp->shape[0], cols = inp->shape[1];
            if (dim == 1) {
                for (int r = 0; r < rows; r++) {
                    float* row = out_data + r * cols;
                    for (int i = 1; i < cols; i++) {
                        float key = row[i];
                        int j = i - 1;
                        while (j >= 0 && (desc ? row[j] < key : row[j] > key)) {
                            row[j + 1] = row[j]; j--;
                        }
                        row[j + 1] = key;
                    }
                }
            } else {
                for (int c = 0; c < cols; c++) {
                    for (int i = 1; i < rows; i++) {
                        float key = out_data[i * cols + c];
                        int j = i - 1;
                        while (j >= 0 && (desc ? out_data[j * cols + c] < key : out_data[j * cols + c] > key)) {
                            out_data[(j + 1) * cols + c] = out_data[j * cols + c]; j--;
                        }
                        out_data[(j + 1) * cols + c] = key;
                    }
                }
            }
        }
        break;
    }

    case UOP_ARGSORT: {
        if (!in1_data) return -1;
        SortParams* sp = (SortParams*)node->params;
        Tensor* inp = node->inputs[0];
        int dim = sp ? sp->dim : -1;
        bool desc = sp ? sp->descending : false;
        if (dim < 0) dim += inp->ndim;

        if (inp->ndim == 1) {
            int n = inp->shape[0];
            for (int i = 0; i < n; i++) out_data[i] = (float)i;
            for (int i = 1; i < n; i++) {
                float ki = out_data[i];
                float kv = in1_data[(int)ki];
                int j = i - 1;
                while (j >= 0 && (desc ? in1_data[(int)out_data[j]] < kv : in1_data[(int)out_data[j]] > kv)) {
                    out_data[j + 1] = out_data[j]; j--;
                }
                out_data[j + 1] = ki;
            }
        } else if (inp->ndim == 2) {
            int rows = inp->shape[0], cols = inp->shape[1];
            if (dim == 1) {
                for (int r = 0; r < rows; r++) {
                    float* idx_row = out_data + r * cols;
                    const float* val_row = in1_data + r * cols;
                    for (int i = 0; i < cols; i++) idx_row[i] = (float)i;
                    for (int i = 1; i < cols; i++) {
                        float ki = idx_row[i];
                        float kv = val_row[(int)ki];
                        int j = i - 1;
                        while (j >= 0 && (desc ? val_row[(int)idx_row[j]] < kv : val_row[(int)idx_row[j]] > kv)) {
                            idx_row[j + 1] = idx_row[j]; j--;
                        }
                        idx_row[j + 1] = ki;
                    }
                }
            } else {
                for (int c = 0; c < cols; c++) {
                    for (int i = 0; i < rows; i++) out_data[i * cols + c] = (float)i;
                    for (int i = 1; i < rows; i++) {
                        float ki = out_data[i * cols + c];
                        float kv = in1_data[(int)ki * cols + c];
                        int j = i - 1;
                        while (j >= 0 && (desc ? in1_data[(int)out_data[j * cols + c] * cols + c] < kv
                                               : in1_data[(int)out_data[j * cols + c] * cols + c] > kv)) {
                            out_data[(j + 1) * cols + c] = out_data[j * cols + c]; j--;
                        }
                        out_data[(j + 1) * cols + c] = ki;
                    }
                }
            }
        }
        break;
    }

    case UOP_TOPK: {
        if (!in1_data) return -1;
        TopkParams* tp = (TopkParams*)node->params;
        Tensor* inp = node->inputs[0];
        int k = tp ? tp->k : 1;
        bool largest = tp ? tp->largest : true;
        (void)inp;

        if (node->inputs[0]->ndim == 1) {
            int n = node->inputs[0]->shape[0];
            float* tmp = malloc((size_t)n * sizeof(float));
            if (!tmp) return -1;
            memcpy(tmp, in1_data, (size_t)n * sizeof(float));
            for (int i = 0; i < k && i < n; i++) {
                int best = i;
                for (int j = i + 1; j < n; j++) {
                    if (largest ? tmp[j] > tmp[best] : tmp[j] < tmp[best])
                        best = j;
                }
                float t = tmp[i]; tmp[i] = tmp[best]; tmp[best] = t;
                out_data[i] = tmp[i];
            }
            free(tmp);
        } else {
            int copy_n = k < (int)in1_numel ? k : (int)in1_numel;
            memcpy(out_data, in1_data, (size_t)copy_n * sizeof(float));
        }
        break;
    }

    case UOP_CUMPROD: {
        if (!in1_data) return -1;
        CumsumParams* cp = (CumsumParams*)node->params;
        Tensor* inp = node->inputs[0];
        int dim = cp ? cp->dim : 0;
        if (dim < 0) dim += inp->ndim;

        if (inp->ndim == 1) {
            float acc = 1.0f;
            for (size_t i = 0; i < in1_numel; i++) {
                acc *= in1_data[i];
                out_data[i] = acc;
            }
        } else if (inp->ndim == 2) {
            int rows = inp->shape[0], cols = inp->shape[1];
            if (dim == 1) {
                for (int r = 0; r < rows; r++) {
                    float acc = 1.0f;
                    for (int c = 0; c < cols; c++) {
                        acc *= in1_data[r * cols + c];
                        out_data[r * cols + c] = acc;
                    }
                }
            } else {
                for (int c = 0; c < cols; c++) {
                    float acc = 1.0f;
                    for (int r = 0; r < rows; r++) {
                        acc *= in1_data[r * cols + c];
                        out_data[r * cols + c] = acc;
                    }
                }
            }
        } else {
            memcpy(out_data, in1_data, in1_numel * sizeof(float));
        }
        break;
    }

    case UOP_BITWISE_AND: {
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            int32_t a = (int32_t)in1_data[i % in1_numel];
            int32_t b = (int32_t)in2_data[i % in2_numel];
            out_data[i] = (float)(a & b);
        }
        break;
    }

    case UOP_BITWISE_OR: {
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            int32_t a = (int32_t)in1_data[i % in1_numel];
            int32_t b = (int32_t)in2_data[i % in2_numel];
            out_data[i] = (float)(a | b);
        }
        break;
    }

    case UOP_BITWISE_XOR: {
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            int32_t a = (int32_t)in1_data[i % in1_numel];
            int32_t b = (int32_t)in2_data[i % in2_numel];
            out_data[i] = (float)(a ^ b);
        }
        break;
    }

    case UOP_BITWISE_NOT: {
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            int32_t a = (int32_t)in1_data[i];
            out_data[i] = (float)(~a);
        }
        break;
    }

    case UOP_NONZERO: {
        if (!in1_data) return -1;
        Tensor* inp = node->inputs[0];
        int idx = 0;
        if (inp->ndim == 1) {
            for (size_t i = 0; i < in1_numel; i++) {
                if (in1_data[i] != 0.0f && idx < (int)out->numel)
                    out_data[idx++] = (float)i;
            }
        } else if (inp->ndim == 2) {
            int cols = inp->shape[1];
            for (size_t i = 0; i < in1_numel; i++) {
                if (in1_data[i] != 0.0f && idx + 1 < (int)out->numel) {
                    out_data[idx++] = (float)((int)i / cols);
                    out_data[idx++] = (float)((int)i % cols);
                }
            }
        }
        for (int i = idx; i < (int)out->numel; i++)
            out_data[i] = -1.0f;
        break;
    }

    case UOP_MASKED_FILL: {
        if (!in1_data || !in2_data) return -1;
        MaskedFillParams* mfp = (MaskedFillParams*)node->params;
        float fill_val = mfp ? mfp->value : 0.0f;
        for (size_t i = 0; i < out->numel; i++) {
            size_t mask_idx = i % in2_numel;
            out_data[i] = (in2_data[mask_idx] != 0.0f) ? fill_val : in1_data[i];
        }
        break;
    }

    case UOP_LOG10:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = log10f(in1_data[i % in1_numel]);
        break;

    case UOP_SINH:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = sinhf(in1_data[i % in1_numel]);
        break;

    case UOP_COSH:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = coshf(in1_data[i % in1_numel]);
        break;

    case UOP_ASINH:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = asinhf(in1_data[i % in1_numel]);
        break;

    case UOP_ACOSH:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = acoshf(in1_data[i % in1_numel]);
        break;

    case UOP_ATANH:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = atanhf(in1_data[i % in1_numel]);
        break;

    case UOP_TRUNC:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = truncf(in1_data[i % in1_numel]);
        break;

    case UOP_ISINF:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = isinf(in1_data[i % in1_numel]) ? 1.0f : 0.0f;
        break;

    case UOP_ISNAN:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = isnan(in1_data[i % in1_numel]) ? 1.0f : 0.0f;
        break;

    case UOP_ISFINITE:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = isfinite(in1_data[i % in1_numel]) ? 1.0f : 0.0f;
        break;

    case UOP_LOGICAL_NOT:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = (in1_data[i % in1_numel] == 0.0f) ? 1.0f : 0.0f;
        break;

    case UOP_IDIV:
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            size_t i1 = BROADCAST_IDX(node->inputs[0], out, i);
            size_t i2 = BROADCAST_IDX(node->inputs[1], out, i);
            out_data[i] = floorf(in1_data[i1] / (in2_data[i2] + 1e-8f));
        }
        break;

    case UOP_MOD:
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            size_t i1 = BROADCAST_IDX(node->inputs[0], out, i);
            size_t i2 = BROADCAST_IDX(node->inputs[1], out, i);
            out_data[i] = fmodf(in1_data[i1], in2_data[i2]);
        }
        break;

    case UOP_MINIMUM:
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            size_t i1 = BROADCAST_IDX(node->inputs[0], out, i);
            size_t i2 = BROADCAST_IDX(node->inputs[1], out, i);
            out_data[i] = fminf(in1_data[i1], in2_data[i2]);
        }
        break;

    case UOP_COPYSIGN:
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            size_t i1 = BROADCAST_IDX(node->inputs[0], out, i);
            size_t i2 = BROADCAST_IDX(node->inputs[1], out, i);
            out_data[i] = copysignf(in1_data[i1], in2_data[i2]);
        }
        break;

    case UOP_LOGADDEXP:
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            size_t i1 = BROADCAST_IDX(node->inputs[0], out, i);
            size_t i2 = BROADCAST_IDX(node->inputs[1], out, i);
            float a_val = in1_data[i1], b_val = in2_data[i2];
            float mx = fmaxf(a_val, b_val);
            out_data[i] = mx + logf(expf(a_val - mx) + expf(b_val - mx));
        }
        break;

    case UOP_LSHIFT:
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            size_t i1 = BROADCAST_IDX(node->inputs[0], out, i);
            size_t i2 = BROADCAST_IDX(node->inputs[1], out, i);
            out_data[i] = (float)((int)in1_data[i1] << (int)in2_data[i2]);
        }
        break;

    case UOP_RSHIFT:
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            size_t i1 = BROADCAST_IDX(node->inputs[0], out, i);
            size_t i2 = BROADCAST_IDX(node->inputs[1], out, i);
            out_data[i] = (float)((int)in1_data[i1] >> (int)in2_data[i2]);
        }
        break;

    case UOP_LOGICAL_AND:
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            size_t i1 = BROADCAST_IDX(node->inputs[0], out, i);
            size_t i2 = BROADCAST_IDX(node->inputs[1], out, i);
            out_data[i] = (in1_data[i1] != 0.0f && in2_data[i2] != 0.0f) ? 1.0f : 0.0f;
        }
        break;

    case UOP_LOGICAL_OR:
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            size_t i1 = BROADCAST_IDX(node->inputs[0], out, i);
            size_t i2 = BROADCAST_IDX(node->inputs[1], out, i);
            out_data[i] = (in1_data[i1] != 0.0f || in2_data[i2] != 0.0f) ? 1.0f : 0.0f;
        }
        break;

    case UOP_CMPEQ:
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            size_t i1 = BROADCAST_IDX(node->inputs[0], out, i);
            size_t i2 = BROADCAST_IDX(node->inputs[1], out, i);
            out_data[i] = (in1_data[i1] == in2_data[i2]) ? 1.0f : 0.0f;
        }
        break;

    case UOP_CMPNE:
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            size_t i1 = BROADCAST_IDX(node->inputs[0], out, i);
            size_t i2 = BROADCAST_IDX(node->inputs[1], out, i);
            out_data[i] = (in1_data[i1] != in2_data[i2]) ? 1.0f : 0.0f;
        }
        break;

    case UOP_CMPLE:
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            size_t i1 = BROADCAST_IDX(node->inputs[0], out, i);
            size_t i2 = BROADCAST_IDX(node->inputs[1], out, i);
            out_data[i] = (in1_data[i1] <= in2_data[i2]) ? 1.0f : 0.0f;
        }
        break;

    case UOP_CMPGT:
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            size_t i1 = BROADCAST_IDX(node->inputs[0], out, i);
            size_t i2 = BROADCAST_IDX(node->inputs[1], out, i);
            out_data[i] = (in1_data[i1] > in2_data[i2]) ? 1.0f : 0.0f;
        }
        break;

    case UOP_CMPGE:
        if (!in1_data || !in2_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            size_t i1 = BROADCAST_IDX(node->inputs[0], out, i);
            size_t i2 = BROADCAST_IDX(node->inputs[1], out, i);
            out_data[i] = (in1_data[i1] >= in2_data[i2]) ? 1.0f : 0.0f;
        }
        break;

    case UOP_MIN_REDUCE: {
        if (!in1_data) return -1;
        Tensor* inp = node->inputs[0];
        ReduceParams* rp = (ReduceParams*)node->params;
        int dim = rp && rp->dims && rp->num_dims > 0 ? rp->dims[0] : -1;

        if (dim < 0 || dim >= inp->ndim) {
            // Global min
            float mn = in1_data[0];
            for (size_t i = 1; i < in1_numel; i++)
                if (in1_data[i] < mn) mn = in1_data[i];
            out_data[0] = mn;
        } else if (inp->ndim == 2) {
            int rows = inp->shape[0], cols = inp->shape[1];
            if (dim == 0) {
                for (int c = 0; c < cols; c++) {
                    float mn = in1_data[c];
                    for (int r = 1; r < rows; r++)
                        if (in1_data[r * cols + c] < mn) mn = in1_data[r * cols + c];
                    out_data[c] = mn;
                }
            } else {
                for (int r = 0; r < rows; r++) {
                    float mn = in1_data[r * cols];
                    for (int c = 1; c < cols; c++)
                        if (in1_data[r * cols + c] < mn) mn = in1_data[r * cols + c];
                    out_data[r] = mn;
                }
            }
        } else if (inp->ndim == 1) {
            float mn = in1_data[0];
            for (size_t i = 1; i < in1_numel; i++)
                if (in1_data[i] < mn) mn = in1_data[i];
            out_data[0] = mn;
        }
        break;
    }

    case UOP_VAR: {
        if (!in1_data) return -1;
        Tensor* inp = node->inputs[0];
        ReduceParams* rp = (ReduceParams*)node->params;
        int dim = rp && rp->dims && rp->num_dims > 0 ? rp->dims[0] : -1;

        if (dim < 0 || dim >= inp->ndim) {
            float mean_val = 0;
            for (size_t i = 0; i < in1_numel; i++) mean_val += in1_data[i];
            mean_val /= (float)in1_numel;
            float var_val = 0;
            for (size_t i = 0; i < in1_numel; i++) {
                float diff = in1_data[i] - mean_val;
                var_val += diff * diff;
            }
            out_data[0] = var_val / (float)in1_numel;
        } else if (inp->ndim == 2) {
            int rows = inp->shape[0], cols = inp->shape[1];
            if (dim == 0) {
                for (int c = 0; c < cols; c++) {
                    float mean_val = 0;
                    for (int r = 0; r < rows; r++) mean_val += in1_data[r * cols + c];
                    mean_val /= (float)rows;
                    float var_val = 0;
                    for (int r = 0; r < rows; r++) {
                        float diff = in1_data[r * cols + c] - mean_val;
                        var_val += diff * diff;
                    }
                    out_data[c] = var_val / (float)rows;
                }
            } else {
                for (int r = 0; r < rows; r++) {
                    float mean_val = 0;
                    for (int c = 0; c < cols; c++) mean_val += in1_data[r * cols + c];
                    mean_val /= (float)cols;
                    float var_val = 0;
                    for (int c = 0; c < cols; c++) {
                        float diff = in1_data[r * cols + c] - mean_val;
                        var_val += diff * diff;
                    }
                    out_data[r] = var_val / (float)cols;
                }
            }
        } else if (inp->ndim == 1) {
            float mean_val = 0;
            for (size_t i = 0; i < in1_numel; i++) mean_val += in1_data[i];
            mean_val /= (float)in1_numel;
            float var_val = 0;
            for (size_t i = 0; i < in1_numel; i++) {
                float diff = in1_data[i] - mean_val;
                var_val += diff * diff;
            }
            out_data[0] = var_val / (float)in1_numel;
        }
        break;
    }

    case UOP_STD: {
        if (!in1_data) return -1;
        Tensor* inp = node->inputs[0];
        ReduceParams* rp = (ReduceParams*)node->params;
        int dim = rp && rp->dims && rp->num_dims > 0 ? rp->dims[0] : -1;

        if (dim < 0 || dim >= inp->ndim) {
            float mean_val = 0;
            for (size_t i = 0; i < in1_numel; i++) mean_val += in1_data[i];
            mean_val /= (float)in1_numel;
            float var_val = 0;
            for (size_t i = 0; i < in1_numel; i++) {
                float diff = in1_data[i] - mean_val;
                var_val += diff * diff;
            }
            out_data[0] = sqrtf(var_val / (float)in1_numel);
        } else if (inp->ndim == 2) {
            int rows = inp->shape[0], cols = inp->shape[1];
            if (dim == 0) {
                for (int c = 0; c < cols; c++) {
                    float mean_val = 0;
                    for (int r = 0; r < rows; r++) mean_val += in1_data[r * cols + c];
                    mean_val /= (float)rows;
                    float var_val = 0;
                    for (int r = 0; r < rows; r++) {
                        float diff = in1_data[r * cols + c] - mean_val;
                        var_val += diff * diff;
                    }
                    out_data[c] = sqrtf(var_val / (float)rows);
                }
            } else {
                for (int r = 0; r < rows; r++) {
                    float mean_val = 0;
                    for (int c = 0; c < cols; c++) mean_val += in1_data[r * cols + c];
                    mean_val /= (float)cols;
                    float var_val = 0;
                    for (int c = 0; c < cols; c++) {
                        float diff = in1_data[r * cols + c] - mean_val;
                        var_val += diff * diff;
                    }
                    out_data[r] = sqrtf(var_val / (float)cols);
                }
            }
        } else if (inp->ndim == 1) {
            float mean_val = 0;
            for (size_t i = 0; i < in1_numel; i++) mean_val += in1_data[i];
            mean_val /= (float)in1_numel;
            float var_val = 0;
            for (size_t i = 0; i < in1_numel; i++) {
                float diff = in1_data[i] - mean_val;
                var_val += diff * diff;
            }
            out_data[0] = sqrtf(var_val / (float)in1_numel);
        }
        break;
    }

    case UOP_ANY: {
        if (!in1_data) return -1;
        Tensor* inp = node->inputs[0];
        ReduceParams* rp = (ReduceParams*)node->params;
        int dim = rp && rp->dims && rp->num_dims > 0 ? rp->dims[0] : -1;

        if (dim < 0 || dim >= inp->ndim) {
            float result = 0.0f;
            for (size_t i = 0; i < in1_numel; i++)
                if (in1_data[i] != 0.0f) { result = 1.0f; break; }
            out_data[0] = result;
        } else if (inp->ndim == 2) {
            int rows = inp->shape[0], cols = inp->shape[1];
            if (dim == 0) {
                for (int c = 0; c < cols; c++) {
                    out_data[c] = 0.0f;
                    for (int r = 0; r < rows; r++)
                        if (in1_data[r * cols + c] != 0.0f) { out_data[c] = 1.0f; break; }
                }
            } else {
                for (int r = 0; r < rows; r++) {
                    out_data[r] = 0.0f;
                    for (int c = 0; c < cols; c++)
                        if (in1_data[r * cols + c] != 0.0f) { out_data[r] = 1.0f; break; }
                }
            }
        }
        break;
    }

    case UOP_ALL: {
        if (!in1_data) return -1;
        Tensor* inp = node->inputs[0];
        ReduceParams* rp = (ReduceParams*)node->params;
        int dim = rp && rp->dims && rp->num_dims > 0 ? rp->dims[0] : -1;

        if (dim < 0 || dim >= inp->ndim) {
            float result = 1.0f;
            for (size_t i = 0; i < in1_numel; i++)
                if (in1_data[i] == 0.0f) { result = 0.0f; break; }
            out_data[0] = result;
        } else if (inp->ndim == 2) {
            int rows = inp->shape[0], cols = inp->shape[1];
            if (dim == 0) {
                for (int c = 0; c < cols; c++) {
                    out_data[c] = 1.0f;
                    for (int r = 0; r < rows; r++)
                        if (in1_data[r * cols + c] == 0.0f) { out_data[c] = 0.0f; break; }
                }
            } else {
                for (int r = 0; r < rows; r++) {
                    out_data[r] = 1.0f;
                    for (int c = 0; c < cols; c++)
                        if (in1_data[r * cols + c] == 0.0f) { out_data[r] = 0.0f; break; }
                }
            }
        }
        break;
    }

    case UOP_LOGSUMEXP: {
        if (!in1_data) return -1;
        Tensor* inp = node->inputs[0];
        ReduceParams* rp = (ReduceParams*)node->params;
        int dim = rp && rp->dims && rp->num_dims > 0 ? rp->dims[0] : -1;

        if (dim < 0 || dim >= inp->ndim) {
            float mx = in1_data[0];
            for (size_t i = 1; i < in1_numel; i++)
                if (in1_data[i] > mx) mx = in1_data[i];
            float sum = 0;
            for (size_t i = 0; i < in1_numel; i++)
                sum += expf(in1_data[i] - mx);
            out_data[0] = mx + logf(sum);
        } else if (inp->ndim == 2) {
            int rows = inp->shape[0], cols = inp->shape[1];
            if (dim == 0) {
                for (int c = 0; c < cols; c++) {
                    float mx = in1_data[c];
                    for (int r = 1; r < rows; r++)
                        if (in1_data[r * cols + c] > mx) mx = in1_data[r * cols + c];
                    float sum = 0;
                    for (int r = 0; r < rows; r++)
                        sum += expf(in1_data[r * cols + c] - mx);
                    out_data[c] = mx + logf(sum);
                }
            } else {
                for (int r = 0; r < rows; r++) {
                    float mx = in1_data[r * cols];
                    for (int c = 1; c < cols; c++)
                        if (in1_data[r * cols + c] > mx) mx = in1_data[r * cols + c];
                    float sum = 0;
                    for (int c = 0; c < cols; c++)
                        sum += expf(in1_data[r * cols + c] - mx);
                    out_data[r] = mx + logf(sum);
                }
            }
        }
        break;
    }

    case UOP_CUMMAX: {
        if (!in1_data) return -1;
        CumsumParams* cp = (CumsumParams*)node->params;
        int cdim = cp ? cp->dim : 0;
        Tensor* inp = node->inputs[0];

        if (inp->ndim == 1) {
            out_data[0] = in1_data[0];
            for (size_t i = 1; i < in1_numel; i++)
                out_data[i] = fmaxf(out_data[i-1], in1_data[i]);
        } else if (inp->ndim == 2) {
            int rows = inp->shape[0], cols = inp->shape[1];
            if (cdim == 0) {
                for (int c = 0; c < cols; c++) {
                    out_data[c] = in1_data[c];
                    for (int r = 1; r < rows; r++)
                        out_data[r * cols + c] = fmaxf(out_data[(r-1) * cols + c], in1_data[r * cols + c]);
                }
            } else {
                for (int r = 0; r < rows; r++) {
                    out_data[r * cols] = in1_data[r * cols];
                    for (int c = 1; c < cols; c++)
                        out_data[r * cols + c] = fmaxf(out_data[r * cols + c - 1], in1_data[r * cols + c]);
                }
            }
        }
        break;
    }

    case UOP_CUMMIN: {
        if (!in1_data) return -1;
        CumsumParams* cp = (CumsumParams*)node->params;
        int cdim = cp ? cp->dim : 0;
        Tensor* inp = node->inputs[0];

        if (inp->ndim == 1) {
            out_data[0] = in1_data[0];
            for (size_t i = 1; i < in1_numel; i++)
                out_data[i] = fminf(out_data[i-1], in1_data[i]);
        } else if (inp->ndim == 2) {
            int rows = inp->shape[0], cols = inp->shape[1];
            if (cdim == 0) {
                for (int c = 0; c < cols; c++) {
                    out_data[c] = in1_data[c];
                    for (int r = 1; r < rows; r++)
                        out_data[r * cols + c] = fminf(out_data[(r-1) * cols + c], in1_data[r * cols + c]);
                }
            } else {
                for (int r = 0; r < rows; r++) {
                    out_data[r * cols] = in1_data[r * cols];
                    for (int c = 1; c < cols; c++)
                        out_data[r * cols + c] = fminf(out_data[r * cols + c - 1], in1_data[r * cols + c]);
                }
            }
        }
        break;
    }

    case UOP_CAT: {
        CatParams* cp = (CatParams*)node->params;
        int cat_dim = cp ? cp->dim : 0;
        int num_t = node->num_inputs;
        size_t offset = 0;

        if (out->ndim == 1) {
            for (int t = 0; t < num_t; t++) {
                float* tdata = (float*)node->inputs[t]->data;
                if (!tdata) return -1;
                size_t tlen = node->inputs[t]->numel;
                memcpy(out_data + offset, tdata, tlen * sizeof(float));
                offset += tlen;
            }
        } else if (out->ndim == 2) {
            int out_cols = out->shape[1];
            if (cat_dim == 0) {
                for (int t = 0; t < num_t; t++) {
                    float* tdata = (float*)node->inputs[t]->data;
                    if (!tdata) return -1;
                    size_t tlen = node->inputs[t]->numel;
                    memcpy(out_data + offset, tdata, tlen * sizeof(float));
                    offset += tlen;
                }
            } else {
                int rows = out->shape[0];
                for (int r = 0; r < rows; r++) {
                    int col_off = 0;
                    for (int t = 0; t < num_t; t++) {
                        float* tdata = (float*)node->inputs[t]->data;
                        if (!tdata) return -1;
                        int tcols = node->inputs[t]->shape[1];
                        memcpy(out_data + r * out_cols + col_off, tdata + r * tcols, (size_t)tcols * sizeof(float));
                        col_off += tcols;
                    }
                }
            }
        }
        break;
    }

    case UOP_STACK: {
        StackParams* sp = (StackParams*)node->params;
        int stack_dim = sp ? sp->dim : 0;
        int num_t = node->num_inputs;

        if (stack_dim == 0) {
            size_t offset = 0;
            for (int t = 0; t < num_t; t++) {
                float* tdata = (float*)node->inputs[t]->data;
                if (!tdata) return -1;
                size_t tlen = node->inputs[t]->numel;
                memcpy(out_data + offset, tdata, tlen * sizeof(float));
                offset += tlen;
            }
        } else {
            // General case: interleave along stack_dim
            size_t out_numel = out->numel;
            size_t per_tensor = out_numel / (size_t)num_t;
            // Simple: for each output position, determine which tensor and source index
            size_t outer_size = 1;
            for (int d = 0; d < stack_dim; d++) outer_size *= (size_t)out->shape[d];
            size_t inner_size = per_tensor / outer_size;

            for (size_t o = 0; o < outer_size; o++) {
                for (int t = 0; t < num_t; t++) {
                    float* tdata = (float*)node->inputs[t]->data;
                    if (!tdata) return -1;
                    memcpy(out_data + (o * (size_t)num_t + (size_t)t) * inner_size,
                           tdata + o * inner_size, inner_size * sizeof(float));
                }
            }
        }
        break;
    }

    case UOP_SCATTER: {
        if (!in1_data || !in2_data) return -1;
        ScatterParams* sp = (ScatterParams*)node->params;
        int sdim = sp ? sp->dim : 0;
        float* src_data = NULL;
        if (node->num_inputs >= 3 && node->inputs[2])
            src_data = (float*)node->inputs[2]->data;
        if (!src_data) return -1;

        // Copy input first
        memcpy(out_data, in1_data, out->numel * sizeof(float));

        // Scatter: out[index[i]] = src[i] along sdim
        Tensor* idx_t = node->inputs[1];
        for (size_t i = 0; i < idx_t->numel; i++) {
            if (out->ndim == 1) {
                int idx = (int)in2_data[i];
                if (idx >= 0 && idx < (int)out->numel)
                    out_data[idx] = src_data[i];
            } else if (out->ndim == 2) {
                int rows = out->shape[0], cols = out->shape[1];
                int r = (int)i / idx_t->shape[1], c = (int)i % idx_t->shape[1];
                int idx = (int)in2_data[i];
                if (sdim == 0 && idx >= 0 && idx < rows)
                    out_data[idx * cols + c] = src_data[i];
                else if (sdim == 1 && idx >= 0 && idx < cols)
                    out_data[r * cols + idx] = src_data[i];
            }
        }
        break;
    }

    case UOP_ROLL: {
        if (!in1_data) return -1;
        RollParams* rp = (RollParams*)node->params;
        int shift = rp ? rp->shift : 0;
        int rdim = rp ? rp->dim : 0;
        Tensor* inp = node->inputs[0];

        if (inp->ndim == 1) {
            int n = inp->shape[0];
            int s = ((shift % n) + n) % n;
            for (int i = 0; i < n; i++)
                out_data[(i + s) % n] = in1_data[i];
        } else if (inp->ndim == 2) {
            int rows = inp->shape[0], cols = inp->shape[1];
            if (rdim == 0) {
                int s = ((shift % rows) + rows) % rows;
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++)
                        out_data[((r + s) % rows) * cols + c] = in1_data[r * cols + c];
            } else {
                int s = ((shift % cols) + cols) % cols;
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++)
                        out_data[r * cols + ((c + s) % cols)] = in1_data[r * cols + c];
            }
        }
        break;
    }

    case UOP_FLATTEN: {
        if (!in1_data) return -1;
        memcpy(out_data, in1_data, out->numel * sizeof(float));
        break;
    }

    case UOP_UNFLATTEN: {
        if (!in1_data) return -1;
        memcpy(out_data, in1_data, out->numel * sizeof(float));
        break;
    }

    case UOP_DIAG: {
        if (!in1_data) return -1;
        Tensor* inp = node->inputs[0];
        DiagParams* dp = (DiagParams*)node->params;
        int offset = dp ? dp->offset : 0;

        if (inp->ndim == 1) {
            // Create diagonal matrix
            int n = out->shape[0];
            memset(out_data, 0, out->numel * sizeof(float));
            for (int i = 0; i < inp->shape[0]; i++) {
                int r = (offset >= 0) ? i : i - offset;
                int c = (offset >= 0) ? i + offset : i;
                if (r >= 0 && r < n && c >= 0 && c < n)
                    out_data[r * n + c] = in1_data[i];
            }
        } else if (inp->ndim == 2) {
            // Extract diagonal
            int rows = inp->shape[0], cols = inp->shape[1];
            int oi = 0;
            for (int i = 0; oi < (int)out->numel; i++) {
                int r = (offset >= 0) ? i : i - offset;
                int c = (offset >= 0) ? i + offset : i;
                if (r >= 0 && r < rows && c >= 0 && c < cols)
                    out_data[oi++] = in1_data[r * cols + c];
                else break;
            }
        }
        break;
    }

    case UOP_ONE_HOT: {
        if (!in1_data) return -1;
        OneHotParams* ohp = (OneHotParams*)node->params;
        int nc = ohp ? ohp->num_classes : 0;
        Tensor* inp = node->inputs[0];

        memset(out_data, 0, out->numel * sizeof(float));
        for (size_t i = 0; i < inp->numel; i++) {
            int cls = (int)in1_data[i];
            if (cls >= 0 && cls < nc)
                out_data[i * (size_t)nc + (size_t)cls] = 1.0f;
        }
        break;
    }

    case UOP_ERFC:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++)
            out_data[i] = erfcf(in1_data[i % in1_numel]);
        break;

    case UOP_TILE: {
        if (!in1_data) return -1;
        TileParams* tp = (TileParams*)node->params;
        Tensor* inp = node->inputs[0];

        if (inp->ndim == 1) {
            int n = inp->shape[0];
            int reps = tp->repeats[0];
            for (int r = 0; r < reps; r++)
                memcpy(out_data + r * n, in1_data, (size_t)n * sizeof(float));
        } else if (inp->ndim == 2) {
            int rows = inp->shape[0], cols = inp->shape[1];
            int rreps = tp->repeats[0], creps = tp->repeats[1];
            int out_cols = cols * creps;
            for (int rr = 0; rr < rreps; rr++) {
                for (int r = 0; r < rows; r++) {
                    for (int cr = 0; cr < creps; cr++) {
                        memcpy(out_data + (rr * rows + r) * out_cols + cr * cols,
                               in1_data + r * cols, (size_t)cols * sizeof(float));
                    }
                }
            }
        } else {
            // Generic fallback using modular indexing
            for (size_t i = 0; i < out->numel; i++) {
                size_t src_idx = 0;
                size_t remaining = i;
                size_t in_stride = 1;
                for (int d = out->ndim - 1; d >= 0; d--) {
                    int coord = (int)(remaining % (size_t)out->shape[d]);
                    remaining /= (size_t)out->shape[d];
                    int in_coord = coord % inp->shape[d];
                    src_idx += (size_t)in_coord * in_stride;
                    in_stride *= (size_t)inp->shape[d];
                }
                out_data[i] = in1_data[src_idx];
            }
        }
        break;
    }

    case UOP_REPEAT_INTERLEAVE: {
        if (!in1_data) return -1;
        RepeatInterleaveParams* rip = (RepeatInterleaveParams*)node->params;
        Tensor* inp = node->inputs[0];
        int dim = rip->dim;
        int reps = rip->repeats;

        if (inp->ndim == 1) {
            int n = inp->shape[0];
            for (int i = 0; i < n; i++)
                for (int r = 0; r < reps; r++)
                    out_data[i * reps + r] = in1_data[i];
        } else if (inp->ndim == 2) {
            int rows = inp->shape[0], cols = inp->shape[1];
            if (dim == 0) {
                for (int r = 0; r < rows; r++)
                    for (int rep = 0; rep < reps; rep++)
                        memcpy(out_data + (r * reps + rep) * cols, in1_data + r * cols, (size_t)cols * sizeof(float));
            } else {
                int out_cols = cols * reps;
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++)
                        for (int rep = 0; rep < reps; rep++)
                            out_data[r * out_cols + c * reps + rep] = in1_data[r * cols + c];
            }
        }
        break;
    }

    case UOP_TRACE: {
        if (!in1_data) return -1;
        Tensor* inp = node->inputs[0];
        int rows = inp->shape[0], cols = inp->shape[1];
        int n = rows < cols ? rows : cols;
        float sum = 0;
        for (int i = 0; i < n; i++)
            sum += in1_data[i * cols + i];
        out_data[0] = sum;
        break;
    }

    case UOP_SHRINK: {
        if (!in1_data) return -1;
        ShrinkParams* sp = (ShrinkParams*)node->params;
        Tensor* inp = node->inputs[0];

        if (inp->ndim == 1) {
            int start = sp->starts[0];
            int len = sp->ends[0] - start;
            memcpy(out_data, in1_data + start, (size_t)len * sizeof(float));
        } else if (inp->ndim == 2) {
            int in_cols = inp->shape[1];
            int r_start = sp->starts[0], c_start = sp->starts[1];
            int out_rows = sp->ends[0] - r_start, out_cols = sp->ends[1] - c_start;
            for (int r = 0; r < out_rows; r++)
                memcpy(out_data + r * out_cols, in1_data + (r_start + r) * in_cols + c_start,
                       (size_t)out_cols * sizeof(float));
        }
        break;
    }

    case UOP_LOGCUMSUMEXP: {
        if (!in1_data) return -1;
        CumsumParams* cp = (CumsumParams*)node->params;
        Tensor* inp = node->inputs[0];
        int dim = cp ? cp->dim : 0;

        if (inp->ndim == 1) {
            float mx = in1_data[0];
            out_data[0] = in1_data[0];
            for (size_t i = 1; i < in1_numel; i++) {
                mx = fmaxf(mx, in1_data[i]);
                float sum = 0;
                for (size_t j = 0; j <= i; j++)
                    sum += expf(in1_data[j] - mx);
                out_data[i] = mx + logf(sum);
            }
        } else if (inp->ndim == 2) {
            int rows = inp->shape[0], cols = inp->shape[1];
            if (dim == 1) {
                for (int r = 0; r < rows; r++) {
                    float mx = in1_data[r * cols];
                    out_data[r * cols] = in1_data[r * cols];
                    for (int c = 1; c < cols; c++) {
                        mx = fmaxf(mx, in1_data[r * cols + c]);
                        float sum = 0;
                        for (int j = 0; j <= c; j++)
                            sum += expf(in1_data[r * cols + j] - mx);
                        out_data[r * cols + c] = mx + logf(sum);
                    }
                }
            } else {
                for (int c = 0; c < cols; c++) {
                    float mx = in1_data[c];
                    out_data[c] = in1_data[c];
                    for (int r = 1; r < rows; r++) {
                        mx = fmaxf(mx, in1_data[r * cols + c]);
                        float sum = 0;
                        for (int j = 0; j <= r; j++)
                            sum += expf(in1_data[j * cols + c] - mx);
                        out_data[r * cols + c] = mx + logf(sum);
                    }
                }
            }
        }
        break;
    }

    case UOP_RELU6:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            float x = in1_data[i % in1_numel];
            out_data[i] = fminf(fmaxf(x, 0.0f), 6.0f);
        }
        break;

    case UOP_HARD_SIGMOID:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            float x = (in1_data[i % in1_numel] + 3.0f) / 6.0f;
            out_data[i] = fminf(fmaxf(x, 0.0f), 1.0f);
        }
        break;

    case UOP_HARD_TANH:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            float x = in1_data[i % in1_numel];
            out_data[i] = fminf(fmaxf(x, -1.0f), 1.0f);
        }
        break;

    case UOP_CELU: {
        if (!in1_data) return -1;
        ClampParams* cp = (ClampParams*)node->params;
        float alpha = cp ? cp->min_val : 1.0f;
        for (size_t i = 0; i < out->numel; i++) {
            float x = in1_data[i % in1_numel];
            out_data[i] = fmaxf(0.0f, x) + fminf(0.0f, alpha * (expf(x / alpha) - 1.0f));
        }
        break;
    }

    case UOP_QUICK_GELU:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            float x = in1_data[i % in1_numel];
            float sig = 1.0f / (1.0f + expf(-1.702f * x));
            out_data[i] = x * sig;
        }
        break;

    case UOP_SOFTPLUS:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            float x = in1_data[i % in1_numel];
            // Numerically stable: log(1 + exp(x)) = max(x,0) + log(1 + exp(-|x|))
            out_data[i] = fmaxf(x, 0.0f) + logf(1.0f + expf(-fabsf(x)));
        }
        break;

    case UOP_SOFTSIGN:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            float x = in1_data[i % in1_numel];
            out_data[i] = x / (1.0f + fabsf(x));
        }
        break;

    case UOP_LOGSIGMOID:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            float x = in1_data[i % in1_numel];
            // log(sigmoid(x)) = -softplus(-x) = -max(-x,0) - log(1+exp(-|-x|))
            // = min(x,0) - log(1+exp(-|x|))
            out_data[i] = fminf(x, 0.0f) - logf(1.0f + expf(-fabsf(x)));
        }
        break;

    case UOP_ELU: {
        if (!in1_data) return -1;
        ClampParams* cp = (ClampParams*)node->params;
        float alpha = cp ? cp->min_val : 1.0f;
        for (size_t i = 0; i < out->numel; i++) {
            float x = in1_data[i % in1_numel];
            out_data[i] = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
        }
        break;
    }

    case UOP_SELU: {
        if (!in1_data) return -1;
        const float selu_alpha = 1.6732632423543772f;
        const float selu_scale = 1.0507009873554805f;
        for (size_t i = 0; i < out->numel; i++) {
            float x = in1_data[i % in1_numel];
            out_data[i] = selu_scale * (x > 0.0f ? x : selu_alpha * (expf(x) - 1.0f));
        }
        break;
    }

    case UOP_MISH:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            float x = in1_data[i % in1_numel];
            // softplus(x) = log(1 + exp(x)), numerically stable
            float sp = fmaxf(x, 0.0f) + logf(1.0f + expf(-fabsf(x)));
            out_data[i] = x * tanhf(sp);
        }
        break;

    case UOP_SILU:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            float x = in1_data[i % in1_numel];
            out_data[i] = x / (1.0f + expf(-x));
        }
        break;

    case UOP_HARDSWISH:
        if (!in1_data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            float x = in1_data[i % in1_numel];
            if (x >= 3.0f) out_data[i] = x;
            else if (x <= -3.0f) out_data[i] = 0.0f;
            else out_data[i] = x * (x + 3.0f) / 6.0f;
        }
        break;

    case UOP_MASKED_SELECT: {
        if (!in1_data) return -1;
        float* mask_data = (node->num_inputs >= 2 && node->inputs[1]) ?
                           (float*)node->inputs[1]->data : NULL;
        if (!mask_data) return -1;
        size_t mask_numel = node->inputs[1]->numel;
        size_t count = 0;
        for (size_t i = 0; i < in1_numel && i < mask_numel; i++) {
            if (mask_data[i] != 0.0f) {
                out_data[count++] = in1_data[i];
            }
        }
        // Update actual output size
        out->numel = count;
        out->shape[0] = (int)count;
        break;
    }

    case UOP_DIAGONAL: {
        if (!in1_data) return -1;
        DiagParams* dp = (DiagParams*)node->params;
        int offset = dp ? dp->offset : 0;
        Tensor* inp = node->inputs[0];
        // For 2D input: extract diagonal
        if (inp->ndim == 2) {
            int rows = inp->shape[0];
            int cols = inp->shape[1];
            int diag_len = (int)out->numel;
            for (int i = 0; i < diag_len; i++) {
                int r = (offset >= 0) ? i : i - offset;
                int c = (offset >= 0) ? i + offset : i;
                if (r < rows && c < cols) {
                    out_data[i] = in1_data[r * cols + c];
                }
            }
        }
        break;
    }

    case UOP_UNFOLD: {
        if (!in1_data || !node->params) return -1;
        UnfoldParams* up = (UnfoldParams*)node->params;
        int ks = up->kernel_size;
        int stride = up->stride;
        // Input: [..., L], Output: [..., num_windows, ks]
        int ndim_in = node->inputs[0]->ndim;
        int last_dim = node->inputs[0]->shape[ndim_in - 1];
        int num_windows = (last_dim - ks) / stride + 1;

        if (ndim_in == 1) {
            // [L] -> [num_windows, ks]
            for (int w = 0; w < num_windows; w++) {
                for (int k = 0; k < ks; k++) {
                    out_data[w * ks + k] = in1_data[w * stride + k];
                }
            }
        } else if (ndim_in == 2) {
            // [N, L] -> [N, num_windows, ks]
            int N = node->inputs[0]->shape[0];
            for (int n = 0; n < N; n++) {
                for (int w = 0; w < num_windows; w++) {
                    for (int k = 0; k < ks; k++) {
                        out_data[(n * num_windows + w) * ks + k] =
                            in1_data[n * last_dim + w * stride + k];
                    }
                }
            }
        } else {
            // Generic: batch dims + [L] -> batch dims + [num_windows, ks]
            size_t batch_size = 1;
            for (int d = 0; d < ndim_in - 1; d++)
                batch_size *= (size_t)node->inputs[0]->shape[d];
            for (size_t b = 0; b < batch_size; b++) {
                for (int w = 0; w < num_windows; w++) {
                    for (int k = 0; k < ks; k++) {
                        out_data[(b * (size_t)num_windows + w) * ks + k] =
                            in1_data[b * (size_t)last_dim + w * stride + k];
                    }
                }
            }
        }
        break;
    }

    default:
        LOG_WARNING("CPU fallback: unsupported op type %d", node->type);
        for (size_t i = 0; i < out->numel; i++) {
            out_data[i] = 0.0f;
        }
        break;
    }

    node->is_executed = true;
    out->is_executed  = true;

    return 0;
}

// Debug counter for execution calls
static size_t g_cpu_exec_calls       = 0;
static size_t g_total_nodes_executed = 0;

// Execute IR graph using CPU interpreter
// Non-static to allow use from dispatch layer
int cpu_execute_ir(CMLGraph_t ir) {
    if (!ir)
        return -1;

    g_cpu_exec_calls++;

    LOG_DEBUG("Executing IR using CPU fallback interpreter (call #%zu)", g_cpu_exec_calls);

    // Decompose composite ops into primitives before execution
    if (!ir->is_decomposed) {
        cml_ir_decompose(ir);
    }

    // Execute nodes in order
    struct IRNode* node = ir->head;
    while (node) {
        // Ensure inputs are executed first
        for (int i = 0; i < node->num_inputs; i++) {
            if (node->inputs && node->inputs[i] && !node->inputs[i]->is_executed) {
                // Input is from another IR node - should already be executed
                // if traversal order is correct
            }
        }

        // Execute this node
        if (cpu_execute_node(node) != 0) {
            LOG_WARNING("CPU fallback: failed to execute node");
        }
        g_total_nodes_executed++;

        node = node->next;
    }

    ir->is_executed = true;
    return 0;
}

// Print execution statistics
void cml_print_exec_stats(void) {
    printf("Execution Stats:\n");
    printf("  cpu_execute_ir calls: %zu\n", g_cpu_exec_calls);
    printf("  Total nodes executed: %zu\n", g_total_nodes_executed);
    if (g_cpu_exec_calls > 0) {
        printf("  Avg nodes per call: %.1f\n", (double)g_total_nodes_executed / g_cpu_exec_calls);
    }
}

void cml_reset_exec_stats(void) {
    g_cpu_exec_calls       = 0;
    g_total_nodes_executed = 0;
}

int cml_ir_execute(CMLGraph_t ir) {
    if (!ir) {
        LOG_ERROR("NULL IR passed to cml_ir_execute");
        return -1;
    }

    /* Use V2 fusion scheduler when CML_SCHEDULE_V2=1 */
    const char* v2_env = getenv("CML_SCHEDULE_V2");
    if (v2_env && v2_env[0] == '1') {
        return cml_ir_execute_v2(ir);
    }

    return cpu_execute_ir(ir);
}

int cml_ir_execute_up_to(CMLGraph_t ir, struct IRNode* target_node) {
    if (!ir || !target_node) {
        LOG_ERROR("Invalid arguments to cml_ir_execute_up_to");
        return -1;
    }

    // PERFORMANCE: Always use direct CPU execution for now
    target_node->is_used = true;
    return cpu_execute_ir(ir);
}
