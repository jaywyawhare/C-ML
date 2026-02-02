#include "ops/ir/ir.h"
#include "ops/ir/execution.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/ir/graph_cache.h"
#include "ops/ir/mlir/mlir_dispatch.h"
#include "ops/ir/mlir/backends/cuda_backend.h"
#include "ops/ir/mlir/backends/rocm_backend.h"
#include "core/logging.h"
#include "ops/ir/mlir/mlir_backend.h"
#include "ops/ir/mlir/mlir_internal.h"
#include "ops/ir/mlir/mlir_uops_builder.h"
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

// Global BLAS context for fast matmul
static CMLBlasContext* g_exec_blas_ctx = NULL;

// ============================================================================
// Buffer Cache Implementation
// ============================================================================
// Simple buffer cache using power-of-2 size buckets for O(1) lookup
// This avoids malloc/free overhead during repeated forward passes

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

// Get bucket index for a size (rounds up to next power of 2)
static int get_bucket_index(size_t size) {
    if (size == 0)
        return -1;

    // Find the highest set bit position (log2)
    int bucket = 0;
    size_t s   = size - 1;
    while (s > 0) {
        s >>= 1;
        bucket++;
    }

    // Clamp to valid range
    if (bucket < BUFFER_CACHE_MIN_BUCKET)
        bucket = BUFFER_CACHE_MIN_BUCKET;
    if (bucket > BUFFER_CACHE_MAX_BUCKET)
        return -1; // Too large

    return bucket - BUFFER_CACHE_MIN_BUCKET;
}

// Initialize buffer cache (lazy)
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

// Allocate from cache or malloc
void* cml_buffer_cache_alloc(size_t size) {
    if (size == 0)
        return NULL;

    init_buffer_cache();

    int bucket_idx = get_bucket_index(size);
    if (bucket_idx < 0) {
        // Too large for cache, use direct malloc
        g_buffer_cache.cache_misses++;
        g_buffer_cache.bytes_allocated += size;
        return calloc(1, size);
    }

    BufferBucket* bucket = &g_buffer_cache.buckets[bucket_idx];

    if (bucket->free_list) {
        // Cache hit: reuse existing buffer
        CachedBuffer* cached = bucket->free_list;
        bucket->free_list    = cached->next;
        bucket->count--;

        void* data = cached->data;
        free(cached);

        g_buffer_cache.cache_hits++;
        g_buffer_cache.bytes_cached -= bucket->bucket_size;

        // Zero the buffer for safety
        memset(data, 0, size);
        return data;
    }

    // Cache miss: allocate new buffer with bucket size
    g_buffer_cache.cache_misses++;
    g_buffer_cache.bytes_allocated += bucket->bucket_size;
    return calloc(1, bucket->bucket_size);
}

// Return buffer to cache or free
void cml_buffer_cache_free(void* ptr, size_t size) {
    if (!ptr)
        return;

    init_buffer_cache();

    int bucket_idx = get_bucket_index(size);
    if (bucket_idx < 0) {
        // Too large for cache, just free
        free(ptr);
        return;
    }

    BufferBucket* bucket = &g_buffer_cache.buckets[bucket_idx];

    // Check if bucket is full
    if (bucket->count >= BUFFER_CACHE_MAX_PER_BUCKET) {
        free(ptr);
        return;
    }

    // Add to free list
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

// Initialize BLAS for execution (called lazily)
static CMLBlasContext* get_blas_context(void) {
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
 * @brief IR execution with MLIR JIT or CPU fallback interpreter
 *
 * This file implements IR execution. When MLIR is available, uses JIT.
 * When MLIR is not available, uses a simple CPU interpreter as fallback.
 */

// ============================================================================
// CPU Fallback Interpreter (always available as backup)
// ============================================================================

// Execute a single IR node on CPU
static int cpu_execute_node(struct IRNode* node) {
    if (!node || !node->output) {
        return -1;
    }

    // Get output tensor
    Tensor* out = node->output;

    // Allocate output data if needed (using buffer cache for reuse)
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

    // Get input data
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

    // Execute based on operation type
    switch (node->type) {
    case UOP_ADD:
        if (!in1_data || !in2_data)
            return -1;
        // Fast path: same size - direct SIMD
        if (in1_numel == in2_numel && in1_numel == out->numel) {
            simd_add_f32(in1_data, in2_data, out_data, out->numel);
        } else {
            // Broadcast path - let compiler auto-vectorize
            for (size_t i = 0; i < out->numel; i++) {
                size_t i1   = (in1_numel == 1) ? 0 : i % in1_numel;
                size_t i2   = (in2_numel == 1) ? 0 : i % in2_numel;
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
                size_t i1   = (in1_numel == 1) ? 0 : i % in1_numel;
                size_t i2   = (in2_numel == 1) ? 0 : i % in2_numel;
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
                size_t i1   = (in1_numel == 1) ? 0 : i % in1_numel;
                size_t i2   = (in2_numel == 1) ? 0 : i % in2_numel;
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
                size_t i1   = (in1_numel == 1) ? 0 : i % in1_numel;
                size_t i2   = (in2_numel == 1) ? 0 : i % in2_numel;
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
        // Use SIMD-optimized sum from simd_utils
        float sum = simd_sum_float(in1_data, in1_numel);
        if (node->type == UOP_MEAN && in1_numel > 0) {
            sum /= (float)in1_numel;
        }
        out_data[0] = sum;
        break;
    }

    case UOP_MAX_REDUCE: {
        if (!in1_data || in1_numel == 0)
            return -1;
        // Use SIMD-optimized max from simd_utils
        out_data[0] = simd_max_float(in1_data, in1_numel);
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

    default:
        LOG_WARNING("CPU fallback: unsupported op type %d", node->type);
        // Return zeros for unsupported ops
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
int cpu_execute_ir(CMLIR_t ir) {
    if (!ir)
        return -1;

    g_cpu_exec_calls++;

    LOG_DEBUG("Executing IR using CPU fallback interpreter (call #%zu)", g_cpu_exec_calls);

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

int cml_ir_execute(CMLIR_t ir) {
    if (!ir) {
        LOG_ERROR("NULL IR passed to cml_ir_execute");
        return -1;
    }

// NOTE: Graph cache disabled - the current architecture creates new IR graphs
// each forward pass, so caching at the execution level doesn't help.
// For now, use direct execution. Future: implement model-level graph reuse.
//
// The graph cache infrastructure remains available for:
// - Reusing graphs when the same model is called with same-shaped inputs
// - Future eager execution mode where graphs are reused

// PERFORMANCE: Skip dispatch layer for CPU-only execution
// The dispatch layer adds overhead without benefit for pure CPU workloads
// Use direct CPU execution path for better performance
#if 1 // Set to 0 to re-enable dispatch layer
    return cpu_execute_ir(ir);
#endif

    // SLOW PATH: Use the unified dispatch layer for execution
    // This handles backend selection, caching, and fallback
    CMLDispatchContext* dispatch_ctx = cml_dispatch_get_global();

    if (dispatch_ctx && dispatch_ctx->initialized) {
        // Collect inputs and outputs from IR for dispatch
        // The dispatch layer will handle backend-specific execution
        Tensor** inputs  = NULL;
        Tensor** outputs = NULL;
        int nin = 0, nout = 0;

        // Count inputs/outputs from IR nodes
        struct IRNode* node = ir->head;
        while (node) {
            // Count unique inputs (tensors that aren't outputs of other nodes)
            for (int i = 0; i < node->num_inputs; i++) {
                if (node->inputs && node->inputs[i]) {
                    // Check if this input is an output of another node
                    bool is_intermediate = false;
                    struct IRNode* check = ir->head;
                    while (check) {
                        if (check->output == node->inputs[i] && check != node) {
                            is_intermediate = true;
                            break;
                        }
                        check = check->next;
                    }
                    if (!is_intermediate) {
                        nin++;
                    }
                }
            }
            // The tail output is the final output
            if (!node->next && node->output) {
                nout = 1;
            }
            node = node->next;
        }

        // For simplicity, extract from the MLIR context if already initialized
        // Otherwise fall through to direct execution
#ifdef CML_HAS_MLIR
        if (ir->mlir_ctx) {
            CMLMLIRContext* mlir_ctx = (CMLMLIRContext*)ir->mlir_ctx;
            inputs                   = mlir_ctx->inputs;
            nin                      = mlir_ctx->num_inputs;
            outputs                  = mlir_ctx->outputs;
            nout                     = mlir_ctx->num_outputs;
        }
#endif

        // Select best backend for this IR
        CMLBackendType best = cml_dispatch_select_backend(dispatch_ctx, ir);
        if (best != dispatch_ctx->active) {
            LOG_DEBUG("Dispatch selecting backend %s for this IR", cml_dispatch_backend_name(best));
        }

        // Execute through dispatch layer
        int result = cml_dispatch_execute(dispatch_ctx, ir, inputs, nin, outputs, nout);

        if (result == 0) {
            ir->is_executed = true;
            LOG_DEBUG("IR execution via dispatch completed successfully");
        }

        return result;
    }

    // Fallback: direct execution if dispatch not available
    LOG_DEBUG("Dispatch not initialized, using direct execution");

#ifndef CML_HAS_MLIR
    // Use CPU fallback interpreter when MLIR is not available
    return cpu_execute_ir(ir);
#else
    // Initialize MLIR context if not already done
    if (!ir->mlir_ctx) {
        ir->mlir_ctx = cml_mlir_init();
        if (!ir->mlir_ctx) {
            LOG_ERROR("Failed to initialize MLIR context");
            return -1;
        }
    }

    CMLMLIRContext* ctx = (CMLMLIRContext*)ir->mlir_ctx;

    // Optimize IR graph before conversion
    if (!ir->is_optimized) {
        LOG_DEBUG("Optimizing IR graph before MLIR conversion");
        if (cml_ir_optimize(ir) != 0) {
            LOG_WARNING("IR optimization failed, continuing anyway");
        }
        ir->is_optimized = true;
    }

    // Build MLIR module from IR (uops→MLIR builder)
    LOG_DEBUG("Building MLIR module from IR (uops builder)");
    if (!cml_mlir_build_from_ir(ctx, ir)) {
        LOG_ERROR("Failed to build MLIR module from IR");
        return -1;
    }

    // Apply MLIR optimization passes (with simplified lowering)
    LOG_DEBUG("Applying MLIR optimization passes");
    if (cml_mlir_optimize(ctx->module.ptr, ctx->context.ptr) != 0) {
        LOG_WARNING("MLIR optimization failed, continuing anyway");
    }

    // Execute via MLIR JIT
    LOG_DEBUG("Executing MLIR module (inputs: %d, outputs: %d)", ctx->num_inputs, ctx->num_outputs);

    // Ensure output tensors have memory allocated (Destination Passing Style)
    for (int i = 0; i < ctx->num_outputs; i++) {
        Tensor* out = ctx->outputs[i];
        if (out && !out->data && !out->buffer_handle) {
            // Allocate memory for output
            size_t size = out->numel * cml_dtype_size(out->dtype);
            if (out->device == DEVICE_CPU || out->device == DEVICE_AUTO) {
                out->data = cml_buffer_cache_alloc(size); // Use buffer cache for reuse
                if (!out->data) {
                    LOG_ERROR("Failed to allocate memory for output tensor %d", i);
                    return -1;
                }
            } else if (out->device == DEVICE_CUDA) {
                // CUDA device allocation
                if (!device_cuda_available()) {
                    LOG_WARNING("CUDA not available, falling back to CPU for output %d", i);
                    out->device = DEVICE_CPU;
                    out->data   = cml_buffer_cache_alloc(size);
                    if (!out->data) {
                        LOG_ERROR("Failed to allocate CPU memory for output tensor %d", i);
                        return -1;
                    }
                } else {
                    // Allocate on GPU via dispatch context
                    CMLDispatchContext* dispatch = cml_dispatch_get_global();
                    if (dispatch && dispatch->backend_contexts[CML_BACKEND_CUDA]) {
                        CMLCUDABackend* cuda =
                            (CMLCUDABackend*)dispatch->backend_contexts[CML_BACKEND_CUDA];
                        CUdeviceptr ptr = cml_cuda_malloc(cuda, size);
                        if (ptr) {
                            out->buffer_handle = (void*)ptr;
                        } else {
                            LOG_WARNING("CUDA allocation failed, falling back to CPU for output %d",
                                        i);
                            out->device = DEVICE_CPU;
                            out->data   = cml_buffer_cache_alloc(size);
                            if (!out->data)
                                return -1;
                        }
                    } else {
                        // No CUDA context, fallback to CPU
                        out->device = DEVICE_CPU;
                        out->data   = cml_buffer_cache_alloc(size);
                        if (!out->data)
                            return -1;
                    }
                }
            } else if (out->device == DEVICE_ROCM) {
                // ROCm device allocation - similar pattern
                if (!device_rocm_available()) {
                    LOG_WARNING("ROCm not available, falling back to CPU for output %d", i);
                    out->device = DEVICE_CPU;
                    out->data   = cml_buffer_cache_alloc(size);
                    if (!out->data)
                        return -1;
                } else {
                    CMLDispatchContext* dispatch = cml_dispatch_get_global();
                    if (dispatch && dispatch->backend_contexts[CML_BACKEND_ROCM]) {
                        CMLROCmBackend* rocm =
                            (CMLROCmBackend*)dispatch->backend_contexts[CML_BACKEND_ROCM];
                        void* ptr = cml_rocm_malloc(rocm, size);
                        if (ptr) {
                            out->buffer_handle = ptr;
                        } else {
                            LOG_WARNING("ROCm allocation failed, falling back to CPU for output %d",
                                        i);
                            out->device = DEVICE_CPU;
                            out->data   = cml_buffer_cache_alloc(size);
                            if (!out->data)
                                return -1;
                        }
                    } else {
                        out->device = DEVICE_CPU;
                        out->data   = cml_buffer_cache_alloc(size);
                        if (!out->data)
                            return -1;
                    }
                }
            } else {
                // Unknown device, fallback to CPU
                LOG_WARNING("Unknown device %d, falling back to CPU for output %d", out->device, i);
                out->device = DEVICE_CPU;
                out->data   = cml_buffer_cache_alloc(size);
                if (!out->data)
                    return -1;
            }
        }
        // Don't zero existing buffers - they may contain valid data from previous ops
    }

    int result =
        cml_mlir_execute(ctx, ctx->inputs, ctx->num_inputs, ctx->outputs, ctx->num_outputs);

    if (result == 0) {
        ir->is_executed = true;
        LOG_DEBUG("IR execution completed successfully");
    } else {
        LOG_ERROR("MLIR execution failed");
    }

    return result;
#endif
}

int cml_ir_execute_up_to(CMLIR_t ir, struct IRNode* target_node) {
    if (!ir || !target_node) {
        LOG_ERROR("Invalid arguments to cml_ir_execute_up_to");
        return -1;
    }

    // PERFORMANCE: Always use direct CPU execution for now
    // This avoids the MLIR/dispatch overhead
    target_node->is_used = true;
    return cpu_execute_ir(ir);
}
