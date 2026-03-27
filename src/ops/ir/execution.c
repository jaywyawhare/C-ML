#include "ops/ir/ir.h"
#include "ops/ir/execution.h"
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
#include "ops/winograd.h"
#include "ops/ir/dispatch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
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
        return malloc(size);
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

        /* Don't zero — callers that need zeroing (e.g. reduce ops) do it themselves.
         * Most ops (matmul, add, relu, conv) write every output element. */
        return data;
    }

    g_buffer_cache.cache_misses++;
    g_buffer_cache.bytes_allocated += bucket->bucket_size;
    return malloc(bucket->bucket_size);
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

/* Winograd F(2,3) hardcoded transforms — adds/subs only, no multiply.
 * Input transform: V = B^T * d * B for a 4x4 tile
 * B^T = [1,0,-1,0; 0,1,1,0; 0,-1,1,0; 0,1,0,-1] */
#ifdef __SSE__
static inline void wino_f23_input_transform(const float d[16], float V[16]) {
    /* SSE: process all 4 columns simultaneously.
     * Step 1: B^T * d (column-wise, 4 columns in parallel) */
    __m128 d0 = _mm_loadu_ps(&d[0]);   /* row 0 */
    __m128 d1 = _mm_loadu_ps(&d[4]);   /* row 1 */
    __m128 d2 = _mm_loadu_ps(&d[8]);   /* row 2 */
    __m128 d3 = _mm_loadu_ps(&d[12]);  /* row 3 */

    __m128 t0 = _mm_sub_ps(d0, d2);        /* d0 - d2 */
    __m128 t1 = _mm_add_ps(d1, d2);        /* d1 + d2 */
    __m128 t2 = _mm_sub_ps(d2, d1);        /* -d1 + d2 */
    __m128 t3 = _mm_sub_ps(d1, d3);        /* d1 - d3 */

    /* Step 2: Transpose to get columns as rows, apply B again, transpose back.
     * _MM_TRANSPOSE4_PS does in-place 4x4 transpose. */
    _MM_TRANSPOSE4_PS(t0, t1, t2, t3);

    __m128 v0 = _mm_sub_ps(t0, t2);        /* t[col0] - t[col2] */
    __m128 v1 = _mm_add_ps(t1, t2);        /* t[col1] + t[col2] */
    __m128 v2 = _mm_sub_ps(t2, t1);        /* -t[col1] + t[col2] */
    __m128 v3 = _mm_sub_ps(t1, t3);        /* t[col1] - t[col3] */

    /* Transpose back to row-major layout */
    _MM_TRANSPOSE4_PS(v0, v1, v2, v3);

    _mm_storeu_ps(&V[0],  v0);
    _mm_storeu_ps(&V[4],  v1);
    _mm_storeu_ps(&V[8],  v2);
    _mm_storeu_ps(&V[12], v3);
}
#else
static inline void wino_f23_input_transform(const float d[16], float V[16]) {
    float t[16];
    for (int j = 0; j < 4; j++) {
        t[0*4+j] = d[0*4+j] - d[2*4+j];
        t[1*4+j] = d[1*4+j] + d[2*4+j];
        t[2*4+j] = -d[1*4+j] + d[2*4+j];
        t[3*4+j] = d[1*4+j] - d[3*4+j];
    }
    for (int i = 0; i < 4; i++) {
        V[i*4+0] = t[i*4+0] - t[i*4+2];
        V[i*4+1] = t[i*4+1] + t[i*4+2];
        V[i*4+2] = -t[i*4+1] + t[i*4+2];
        V[i*4+3] = t[i*4+1] - t[i*4+3];
    }
}
#endif

/* Weight transform: U = G * g * G^T for 3x3 filter
 * G = [1,0,0; 0.5,0.5,0.5; 0.5,-0.5,0.5; 0,0,1] */
static inline void wino_f23_weight_transform(const float g[9], float U[16]) {
    float t[12]; /* 4x3 */
    for (int j = 0; j < 3; j++) {
        t[0*3+j] = g[0*3+j];
        t[1*3+j] = 0.5f*(g[0*3+j] + g[1*3+j] + g[2*3+j]);
        t[2*3+j] = 0.5f*(g[0*3+j] - g[1*3+j] + g[2*3+j]);
        t[3*3+j] = g[2*3+j];
    }
    for (int i = 0; i < 4; i++) {
        U[i*4+0] = t[i*3+0];
        U[i*4+1] = 0.5f*(t[i*3+0] + t[i*3+1] + t[i*3+2]);
        U[i*4+2] = 0.5f*(t[i*3+0] - t[i*3+1] + t[i*3+2]);
        U[i*4+3] = t[i*3+2];
    }
}

/* Output transform: Y = A^T * M * A for 4x4 → 2x2
 * A^T = [1,1,1,0; 0,1,-1,-1] */
#ifdef __SSE__
static inline void wino_f23_output_transform(const float M[16], float Y[4]) {
    /* SSE: A^T * M (process 4 columns in parallel), then * A */
    __m128 m0 = _mm_loadu_ps(&M[0]);
    __m128 m1 = _mm_loadu_ps(&M[4]);
    __m128 m2 = _mm_loadu_ps(&M[8]);
    __m128 m3 = _mm_loadu_ps(&M[12]);

    /* A^T * M: t0 = m0+m1+m2, t1 = m1-m2-m3 */
    __m128 t0 = _mm_add_ps(_mm_add_ps(m0, m1), m2);
    __m128 t1 = _mm_sub_ps(_mm_sub_ps(m1, m2), m3);

    /* Now t0 and t1 are 1x4 rows. Apply * A column-wise:
     * Y[0,0] = t0[0]+t0[1]+t0[2], Y[0,1] = t0[1]-t0[2]-t0[3]
     * Y[1,0] = t1[0]+t1[1]+t1[2], Y[1,1] = t1[1]-t1[2]-t1[3]
     * Extract scalar since output is only 2x2 */
    float t0v[4], t1v[4];
    _mm_storeu_ps(t0v, t0);
    _mm_storeu_ps(t1v, t1);
    Y[0] = t0v[0] + t0v[1] + t0v[2];
    Y[1] = t0v[1] - t0v[2] - t0v[3];
    Y[2] = t1v[0] + t1v[1] + t1v[2];
    Y[3] = t1v[1] - t1v[2] - t1v[3];
}
#else
static inline void wino_f23_output_transform(const float M[16], float Y[4]) {
    float t[8];
    for (int j = 0; j < 4; j++) {
        t[0*4+j] = M[0*4+j] + M[1*4+j] + M[2*4+j];
        t[1*4+j] = M[1*4+j] - M[2*4+j] - M[3*4+j];
    }
    Y[0] = t[0] + t[1] + t[2];
    Y[1] = t[1] - t[2] - t[3];
    Y[2] = t[4] + t[5] + t[6];
    Y[3] = t[5] - t[6] - t[7];
}
#endif

/* BLAS-batched Winograd F(2,3) with hardcoded transforms.
 * Uses add/sub-only transforms (no generic matmul) + batched GEMM/inline
 * multiply at each of the 16 Winograd points. */
static int winograd_conv2d_blas(
    CMLBlasContext* blas,
    const float* input, const float* weight, const float* bias, float* output,
    int batch, int in_channels, int out_channels,
    int H, int W, int pad_h, int pad_w, int groups)
{
    const int ks = 3, ot = 2;
    int out_h = H + 2 * pad_h - ks + 1;
    int out_w = W + 2 * pad_w - ks + 1;
    if (out_h <= 0 || out_w <= 0) return -1;

    int tiles_h = (out_h + ot - 1) / ot;
    int tiles_w = (out_w + ot - 1) / ot;
    int total_tiles = batch * tiles_h * tiles_w;

    int ic_pg = in_channels / groups;
    int oc_pg = out_channels / groups;

    /* U[16][oc_pg * ic_pg], V[16][ic_pg * total_tiles], M[16][oc_pg * total_tiles] */
    size_t U_sz = (size_t)16 * oc_pg * ic_pg;
    size_t V_sz = (size_t)16 * ic_pg * total_tiles;
    size_t M_sz = (size_t)16 * oc_pg * total_tiles;
    float* U_buf = (float*)malloc(U_sz * sizeof(float));
    float* V_buf = (float*)malloc(V_sz * sizeof(float));
    float* M_buf = (float*)malloc(M_sz * sizeof(float));
    if (!U_buf || !V_buf || !M_buf) {
        free(U_buf); free(V_buf); free(M_buf);
        return -1;
    }

    memset(output, 0, (size_t)batch * out_channels * out_h * out_w * sizeof(float));

    for (int g = 0; g < groups; g++) {
        int ic_start = g * ic_pg;
        int oc_start = g * oc_pg;

        /* Step 1: Transform weights with hardcoded add/sub formula */
        for (int oc = 0; oc < oc_pg; oc++) {
            for (int ic = 0; ic < ic_pg; ic++) {
                const float* w = weight + ((size_t)(oc_start+oc) * in_channels + (ic_start+ic)) * 9;
                float U[16];
                wino_f23_weight_transform(w, U);
                for (int p = 0; p < 16; p++)
                    U_buf[p * oc_pg * ic_pg + oc * ic_pg + ic] = U[p];
            }
        }

        /* Step 2: Transform input tiles with hardcoded add/sub formula */
        for (int b = 0; b < batch; b++) {
            for (int th = 0; th < tiles_h; th++) {
                for (int tw = 0; tw < tiles_w; tw++) {
                    int tile_idx = b * tiles_h * tiles_w + th * tiles_w + tw;
                    int trow = th * ot, tcol = tw * ot;

                    for (int ic = 0; ic < ic_pg; ic++) {
                        const float* in_ch = input + ((size_t)b * in_channels + ic_start + ic) * H * W;
                        float d[16];
                        for (int i = 0; i < 4; i++) {
                            int r = trow + i - pad_h;
                            for (int j = 0; j < 4; j++) {
                                int c = tcol + j - pad_w;
                                d[i*4+j] = (r >= 0 && r < H && c >= 0 && c < W)
                                            ? in_ch[r * W + c] : 0.0f;
                            }
                        }
                        float V[16];
                        wino_f23_input_transform(d, V);
                        for (int p = 0; p < 16; p++)
                            V_buf[p * ic_pg * total_tiles + ic * total_tiles + tile_idx] = V[p];
                    }
                }
            }
        }

        /* Step 3: Pointwise multiply — BLAS GEMM for large K, inline for small */
        if (ic_pg >= 8) {
            for (int p = 0; p < 16; p++) {
                float* Up = U_buf + (size_t)p * oc_pg * ic_pg;
                float* Vp = V_buf + (size_t)p * ic_pg * total_tiles;
                float* Mp = M_buf + (size_t)p * oc_pg * total_tiles;
                cml_blas_sgemm(blas, Up, Vp, Mp, oc_pg, total_tiles, ic_pg, 1.0f, 0.0f);
            }
        } else {
            for (int p = 0; p < 16; p++) {
                const float* Up = U_buf + (size_t)p * oc_pg * ic_pg;
                const float* Vp = V_buf + (size_t)p * ic_pg * total_tiles;
                float* Mp = M_buf + (size_t)p * oc_pg * total_tiles;
                for (int oc = 0; oc < oc_pg; oc++) {
                    const float* u_row = Up + oc * ic_pg;
                    float* m_row = Mp + oc * total_tiles;
                    memset(m_row, 0, total_tiles * sizeof(float));
                    for (int ic = 0; ic < ic_pg; ic++) {
                        float u_val = u_row[ic];
                        const float* v_row = Vp + ic * total_tiles;
                        for (int t = 0; t < total_tiles; t++)
                            m_row[t] += u_val * v_row[t];
                    }
                }
            }
        }

        /* Step 4: Inverse transform with hardcoded formula */
        for (int b = 0; b < batch; b++) {
            for (int th = 0; th < tiles_h; th++) {
                for (int tw = 0; tw < tiles_w; tw++) {
                    int tile_idx = b * tiles_h * tiles_w + th * tiles_w + tw;
                    int out_row = th * ot, out_col = tw * ot;

                    for (int oc = 0; oc < oc_pg; oc++) {
                        float Mtile[16];
                        for (int p = 0; p < 16; p++)
                            Mtile[p] = M_buf[p * oc_pg * total_tiles + oc * total_tiles + tile_idx];

                        float Y[4];
                        wino_f23_output_transform(Mtile, Y);

                        float* out_plane = output + ((size_t)b * out_channels + oc_start + oc) * out_h * out_w;
                        for (int i = 0; i < 2 && out_row + i < out_h; i++)
                            for (int j = 0; j < 2 && out_col + j < out_w; j++)
                                out_plane[(out_row+i) * out_w + (out_col+j)] += Y[i*2+j];
                    }
                }
            }
        }

        if (bias) {
            for (int b = 0; b < batch; b++)
                for (int oc = 0; oc < oc_pg; oc++) {
                    float bv = bias[oc_start + oc];
                    float* out_plane = output + ((size_t)b * out_channels + oc_start + oc) * out_h * out_w;
                    for (int i = 0; i < out_h * out_w; i++)
                        out_plane[i] += bv;
                }
        }
    }

    free(U_buf); free(V_buf); free(M_buf);
    return 0;
}

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

/* Broadcast pattern detection for 2D binary ops.
 * Returns: 0=use generic, 1=[R,C]op[1,C] (row broadcast), 2=[R,C]op[R,1] (col broadcast) */
static inline int _detect_broadcast_2d(Tensor* a, Tensor* b, Tensor* out,
                                        size_t* rows, size_t* cols) {
    if (out->ndim != 2) return 0;
    *rows = (size_t)out->shape[0];
    *cols = (size_t)out->shape[1];

    /* a is the big tensor, b is the broadcast one */
    if (a->numel == out->numel) {
        /* [R,C] op [1,C] — b is a row vector broadcast across rows */
        if (b->ndim == 2 && b->shape[0] == 1 && (size_t)b->shape[1] == *cols) return 1;
        if (b->ndim == 1 && (size_t)b->shape[0] == *cols) return 1;
        /* [R,C] op [R,1] — b is a column vector broadcast across cols */
        if (b->ndim == 2 && (size_t)b->shape[0] == *rows && b->shape[1] == 1) return 2;
    }
    return 0;
}

/* Fast broadcast binary op for 2D: [R,C] op [1,C] (row broadcast) */
#define BROADCAST_ROW_OP(op_expr) do { \
    for (size_t r = 0; r < rows; r++) { \
        const float* a_row = in1_data + r * cols; \
        float* o_row = out_data + r * cols; \
        for (size_t c = 0; c < cols; c++) { \
            o_row[c] = a_row[c] op_expr in2_data[c]; \
        } \
    } \
} while(0)

/* Fast broadcast binary op for 2D: [R,C] op [R,1] (col broadcast) */
#define BROADCAST_COL_OP(op_expr) do { \
    for (size_t r = 0; r < rows; r++) { \
        const float* a_row = in1_data + r * cols; \
        float* o_row = out_data + r * cols; \
        float bval = in2_data[r]; \
        for (size_t c = 0; c < cols; c++) { \
            o_row[c] = a_row[c] op_expr bval; \
        } \
    } \
} while(0)

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
        out->owns_data        = true;
        out->from_buffer_cache = true;
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

#define BROADCAST_IDX(tensor_ptr, out_ptr, flat_i) _broadcast_idx(tensor_ptr, out_ptr, flat_i)

    switch (node->type) {
    case UOP_ADD:
        if (!in1_data || !in2_data)
            return -1;
        if (in1_numel == in2_numel && in1_numel == out->numel) {
            simd_add_f32(in1_data, in2_data, out_data, out->numel);
        } else {
            size_t rows, cols;
            int bcast = _detect_broadcast_2d(node->inputs[0], node->inputs[1], out, &rows, &cols);
            if (bcast == 1) { BROADCAST_ROW_OP(+); }
            else if (bcast == 2) { BROADCAST_COL_OP(+); }
            else {
                for (size_t i = 0; i < out->numel; i++) {
                    size_t i1   = BROADCAST_IDX(node->inputs[0], out, i);
                    size_t i2   = BROADCAST_IDX(node->inputs[1], out, i);
                    out_data[i] = in1_data[i1] + in2_data[i2];
                }
            }
        }
        break;

    case UOP_SUB:
        if (!in1_data || !in2_data)
            return -1;
        if (in1_numel == in2_numel && in1_numel == out->numel) {
            simd_sub_f32(in1_data, in2_data, out_data, out->numel);
        } else {
            size_t rows, cols;
            int bcast = _detect_broadcast_2d(node->inputs[0], node->inputs[1], out, &rows, &cols);
            if (bcast == 1) { BROADCAST_ROW_OP(-); }
            else if (bcast == 2) { BROADCAST_COL_OP(-); }
            else {
                for (size_t i = 0; i < out->numel; i++) {
                    size_t i1   = BROADCAST_IDX(node->inputs[0], out, i);
                    size_t i2   = BROADCAST_IDX(node->inputs[1], out, i);
                    out_data[i] = in1_data[i1] - in2_data[i2];
                }
            }
        }
        break;

    case UOP_MUL:
        if (!in1_data || !in2_data)
            return -1;
        if (in1_numel == in2_numel && in1_numel == out->numel) {
            simd_mul_f32(in1_data, in2_data, out_data, out->numel);
        } else {
            size_t rows, cols;
            int bcast = _detect_broadcast_2d(node->inputs[0], node->inputs[1], out, &rows, &cols);
            if (bcast == 1) { BROADCAST_ROW_OP(*); }
            else if (bcast == 2) { BROADCAST_COL_OP(*); }
            else {
                for (size_t i = 0; i < out->numel; i++) {
                    size_t i1   = BROADCAST_IDX(node->inputs[0], out, i);
                    size_t i2   = BROADCAST_IDX(node->inputs[1], out, i);
                    out_data[i] = in1_data[i1] * in2_data[i2];
                }
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
                /* Generic N-dim per-dimension reduction */
                int rdim = reduce_dim;
                int reduce_size = inp->shape[rdim];

                /* Compute the stride of the reduce dimension and outer/inner sizes */
                size_t inner = 1;
                for (int d = rdim + 1; d < inp->ndim; d++)
                    inner *= (size_t)inp->shape[d];
                size_t outer = 1;
                for (int d = 0; d < rdim; d++)
                    outer *= (size_t)inp->shape[d];

                /* For each (outer, inner) position, sum along reduce_dim */
                for (size_t o = 0; o < outer; o++) {
                    for (size_t i = 0; i < inner; i++) {
                        float acc = 0.0f;
                        for (int r = 0; r < reduce_size; r++) {
                            acc += in1_data[o * (size_t)reduce_size * inner + (size_t)r * inner + i];
                        }
                        if (node->type == UOP_MEAN && reduce_size > 0)
                            acc /= (float)reduce_size;
                        out_data[o * inner + i] = acc;
                    }
                }
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

        CMLBlasContext* blas = get_blas_context();
        if (blas && blas->initialized) {
            /* Check if input B is a transpose node — fuse transpose into sgemm */
            struct IRNode* b_node = b->ir_node;
            if (b_node && b_node->type == UOP_PERMUTE && b_node->num_inputs == 1 &&
                b_node->inputs[0] && b_node->inputs[0]->ndim == 2) {
                /* B = transpose(B_orig), so matmul A @ B = A @ B_orig^T
                 * Use sgemm_ex with transB=true on the original (non-transposed) data */
                Tensor* b_orig = b_node->inputs[0];
                float* b_orig_data = (float*)b_orig->data;
                if (!b_orig_data) b_orig_data = (float*)tensor_data_ptr(b_orig);
                if (b_orig_data) {
                    /* B_orig is [N, K], we want A[M,K] @ B_orig[N,K]^T = [M,N] */
                    int result = cml_blas_sgemm_ex(blas, in1_data, b_orig_data, out_data,
                                                    M, N, K, 1.0f, 0.0f, false, true);
                    if (result == 0) break;
                }
            }
            int result = cml_blas_sgemm(blas, in1_data, in2_data, out_data, M, N, K, 1.0f, 0.0f);
            if (result == 0) {
                break; // BLAS succeeded
            }
            // Fall through to naive implementation if BLAS failed
            LOG_WARNING("BLAS sgemm failed, falling back to naive matmul");
        }

        // Naive matmul fallback with cache-friendly access pattern
        memset(out_data, 0, out->numel * sizeof(float));
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
            // Generic N-dimensional gather along specified dim
            // out[i0,...,i_{dim-1}, j, i_{dim+1},...] = input[i0,...,i_{dim-1}, indices[j], i_{dim+1},...]
            if (dim < 0 || dim >= input->ndim) {
                LOG_ERROR("UOP_GATHER: dim %d out of range for %d-dim input", dim, input->ndim);
                return -1;
            }

            // Compute strides for input
            size_t outer_size = 1, inner_size = 1;
            for (int d = 0; d < dim; d++) outer_size *= (size_t)input->shape[d];
            for (int d = dim + 1; d < input->ndim; d++) inner_size *= (size_t)input->shape[d];
            size_t dim_size = (size_t)input->shape[dim];

            size_t out_idx = 0;
            for (size_t o = 0; o < outer_size; o++) {
                for (size_t j = 0; j < indices->numel; j++) {
                    int idx = (int)index_data[j];
                    if (idx < 0 || idx >= (int)dim_size) {
                        LOG_ERROR("UOP_GATHER: index %d out of bounds [0, %zu)", idx, dim_size);
                        return -1;
                    }
                    for (size_t k = 0; k < inner_size; k++) {
                        size_t src = o * dim_size * inner_size + (size_t)idx * inner_size + k;
                        if (out_idx < out->numel) {
                            out_data[out_idx++] = input_data[src];
                        }
                    }
                }
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

    case UOP_RELU:
        if (!in1_data) return -1;
        if (in1_numel == out->numel) {
            /* Same-size: vectorizable loop */
            for (size_t i = 0; i < out->numel; i++) {
                float x = in1_data[i];
                out_data[i] = x > 0.0f ? x : 0.0f;
            }
        } else {
            for (size_t i = 0; i < out->numel; i++) {
                float x = in1_data[i % in1_numel];
                out_data[i] = x > 0.0f ? x : 0.0f;
            }
        }
        break;

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

    case UOP_LINEAR: {
        /* Fused linear: output = input[M,K] @ weight[N,K]^T + bias[N]
         * inputs[0] = input, inputs[1] = weight, inputs[2] = bias (optional)
         */
        if (!in1_data || !in2_data)
            return -1;
        Tensor* input_t = node->inputs[0];
        Tensor* weight_t = node->inputs[1];
        if (input_t->ndim < 2 || weight_t->ndim != 2)
            return -1;

        int M = input_t->shape[input_t->ndim - 2]; /* batch (or rows) */
        int K = input_t->shape[input_t->ndim - 1]; /* in_features */
        int N = weight_t->shape[0];                 /* out_features */

        CMLBlasContext* blas = get_blas_context();
        if (blas && blas->initialized) {
            /* C[M,N] = input[M,K] @ weight[N,K]^T */
            cml_blas_sgemm_ex(blas, in1_data, in2_data, out_data,
                              M, N, K, 1.0f, 0.0f, false, true);
        } else {
            /* Naive fallback: C = A @ B^T */
            memset(out_data, 0, out->numel * sizeof(float));
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++)
                        sum += in1_data[m * K + k] * in2_data[n * K + k];
                    out_data[m * N + n] = sum;
                }
            }
        }

        /* Add bias if present */
        if (node->num_inputs >= 3 && node->inputs[2]) {
            float* bias_data = (float*)node->inputs[2]->data;
            if (bias_data) {
                for (int m = 0; m < M; m++) {
                    float* row = out_data + m * N;
                    for (int n = 0; n < N; n++)
                        row[n] += bias_data[n];
                }
            }
        }
        break;
    }

    case UOP_CONV2D: {
        if (!in1_data || !in2_data)
            return -1;
        Tensor* input_t  = node->inputs[0];
        Tensor* weight_t = node->inputs[1];
        if (input_t->ndim != 4 || weight_t->ndim != 4)
            return -1;

        Conv2DParams* p = (Conv2DParams*)node->params;
        int batch       = input_t->shape[0];
        int in_channels = input_t->shape[1];
        int in_h        = input_t->shape[2];
        int in_w        = input_t->shape[3];
        int out_channels = weight_t->shape[0];
        int kernel_h     = weight_t->shape[2];
        int kernel_w     = weight_t->shape[3];
        int stride_h   = p && p->stride   ? p->stride[0]   : 1;
        int stride_w   = p && p->stride   ? p->stride[1]   : 1;
        int pad_h      = p && p->padding  ? p->padding[0]  : 0;
        int pad_w      = p && p->padding  ? p->padding[1]  : 0;
        int dilation_h = p && p->dilation ? p->dilation[0] : 1;
        int dilation_w = p && p->dilation ? p->dilation[1] : 1;
        int groups     = p ? p->groups : 1;
        if (groups < 1) groups = 1;

        int out_h = out->shape[2];
        int out_w = out->shape[3];

        float* bias_data = NULL;
        if (node->num_inputs >= 3 && node->inputs[2])
            bias_data = (float*)node->inputs[2]->data;

        int ch_per_group_in  = in_channels  / groups;
        int ch_per_group_out = out_channels / groups;

        CMLBlasContext* conv_blas = get_blas_context();

        /* Winograd F(2,3) for 3x3, stride 1, dilation 1 convolutions.
         * Only beneficial when in_channels >= 16 (BLAS GEMM needs K >= 8).
         * For shallow channels (e.g., 3 RGB), im2col+GEMM is faster. */
        if (p && p->use_winograd && conv_blas && conv_blas->initialized &&
            ch_per_group_in >= 16 &&
            kernel_h == 3 && kernel_w == 3 &&
            stride_h == 1 && stride_w == 1 &&
            dilation_h == 1 && dilation_w == 1) {
            int ret = winograd_conv2d_blas(conv_blas,
                in1_data, in2_data, bias_data, out_data,
                batch, in_channels, out_channels,
                in_h, in_w, pad_h, pad_w, groups);
            if (ret == 0) break;
            /* Fall through to im2col on failure */
        }

        /* im2col + BLAS matmul path (fast) or naive fallback */
        size_t col_h = (size_t)ch_per_group_in * kernel_h * kernel_w;
        size_t col_w = (size_t)out_h * out_w;
        int spatial = out_h * out_w;
        int fast_im2col = (stride_h == 1 && stride_w == 1 &&
                           dilation_h == 1 && dilation_w == 1 &&
                           pad_h == 0 && pad_w == 0);

        /* Batched im2col only helps when per-batch GEMMs are very small
         * (BLAS dispatch overhead dominates). For typical conv sizes, per-batch
         * im2col+GEMM is faster since it avoids the output rearrangement. */
        int use_batched = (groups == 1 && fast_im2col && batch > 1 &&
                           col_h * col_w < 4096);
        size_t total_col_w = use_batched ? col_w * batch : col_w;
        size_t needed = col_h * total_col_w * sizeof(float);

        float* col_buf = NULL;
        static float* s_col_buf = NULL;
        static size_t s_col_buf_size = 0;

        if (conv_blas && conv_blas->initialized) {
            if (needed > s_col_buf_size) {
                free(s_col_buf);
                s_col_buf = (float*)malloc(needed);
                s_col_buf_size = s_col_buf ? needed : 0;
            }
            col_buf = s_col_buf;
        }

        if (col_buf && use_batched) {
            /* Batched im2col: all batch items into col_buf[col_h, batch*col_w] */
            for (int b = 0; b < batch; b++) {
                size_t b_offset = (size_t)b * col_w;
                for (int ic = 0; ic < ch_per_group_in; ic++) {
                    const float* in_ch = in1_data + ((size_t)b * in_channels + ic) * (in_h * in_w);
                    for (int kh_i = 0; kh_i < kernel_h; kh_i++) {
                        for (int kw_i = 0; kw_i < kernel_w; kw_i++) {
                            size_t col_row = ((size_t)ic * kernel_h + kh_i) * kernel_w + kw_i;
                            float* col_dst = col_buf + col_row * total_col_w + b_offset;
                            for (int oh_i = 0; oh_i < out_h; oh_i++) {
                                memcpy(col_dst + oh_i * out_w,
                                       in_ch + (oh_i + kh_i) * in_w + kw_i,
                                       (size_t)out_w * sizeof(float));
                            }
                        }
                    }
                }
            }

            /* Single GEMM: result[oc, batch*spatial] = W[oc, col_h] @ col[col_h, batch*spatial] */
            /* Use a static temp buffer for the rearrangement */
            static float* s_gemm_buf = NULL;
            static size_t s_gemm_buf_size = 0;
            size_t gemm_sz = (size_t)out_channels * total_col_w * sizeof(float);
            if (gemm_sz > s_gemm_buf_size) {
                free(s_gemm_buf);
                s_gemm_buf = (float*)malloc(gemm_sz);
                s_gemm_buf_size = s_gemm_buf ? gemm_sz : 0;
            }

            if (s_gemm_buf) {
                cml_blas_sgemm(conv_blas, in2_data, col_buf, s_gemm_buf,
                               out_channels, (int)total_col_w, (int)col_h, 1.0f, 0.0f);

                /* Rearrange: result[oc][b*spatial+s] → out[b][oc][s] */
                for (int oc = 0; oc < out_channels; oc++) {
                    const float* src_row = s_gemm_buf + (size_t)oc * total_col_w;
                    for (int b = 0; b < batch; b++) {
                        float* dst = out_data + ((size_t)b * out_channels + oc) * spatial;
                        memcpy(dst, src_row + (size_t)b * col_w, spatial * sizeof(float));
                    }
                }

                /* Add bias */
                if (bias_data) {
                    for (int b = 0; b < batch; b++) {
                        for (int oc = 0; oc < out_channels; oc++) {
                            float bv = bias_data[oc];
                            float* row = out_data + ((size_t)b * out_channels + oc) * spatial;
                            for (int j = 0; j < spatial; j++)
                                row[j] += bv;
                        }
                    }
                }
            }
        } else if (col_buf) {
            /* Per-batch im2col + GEMM (for groups > 1 or general stride/pad) */
            for (int b = 0; b < batch; b++) {
                for (int g = 0; g < groups; g++) {
                    if (fast_im2col) {
                        for (int ic = 0; ic < ch_per_group_in; ic++) {
                            int ic_abs = g * ch_per_group_in + ic;
                            const float* in_ch = in1_data + ((size_t)b * in_channels + ic_abs) * (in_h * in_w);
                            for (int kh_i = 0; kh_i < kernel_h; kh_i++) {
                                for (int kw_i = 0; kw_i < kernel_w; kw_i++) {
                                    size_t col_row = ((size_t)ic * kernel_h + kh_i) * kernel_w + kw_i;
                                    float* col_dst = col_buf + col_row * col_w;
                                    for (int oh_i = 0; oh_i < out_h; oh_i++) {
                                        memcpy(col_dst + oh_i * out_w,
                                               in_ch + (oh_i + kh_i) * in_w + kw_i,
                                               (size_t)out_w * sizeof(float));
                                    }
                                }
                            }
                        }
                    } else {
                        for (int ic = 0; ic < ch_per_group_in; ic++) {
                            int ic_abs = g * ch_per_group_in + ic;
                            for (int kh_i = 0; kh_i < kernel_h; kh_i++) {
                                for (int kw_i = 0; kw_i < kernel_w; kw_i++) {
                                    size_t col_row = ((size_t)ic * kernel_h + kh_i) * kernel_w + kw_i;
                                    for (int oh_i = 0; oh_i < out_h; oh_i++) {
                                        for (int ow_i = 0; ow_i < out_w; ow_i++) {
                                            int ih = oh_i * stride_h - pad_h + kh_i * dilation_h;
                                            int iw = ow_i * stride_w - pad_w + kw_i * dilation_w;
                                            size_t col_col = (size_t)oh_i * out_w + ow_i;
                                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                                size_t in_idx = ((size_t)b * in_channels + ic_abs) *
                                                                (in_h * in_w) + ih * in_w + iw;
                                                col_buf[col_row * col_w + col_col] = in1_data[in_idx];
                                            } else {
                                                col_buf[col_row * col_w + col_col] = 0.0f;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    int oc_offset = g * ch_per_group_out;
                    float* out_ptr = out_data + ((size_t)b * out_channels + oc_offset) * spatial;
                    const float* w_ptr = in2_data + (size_t)oc_offset * col_h;

                    cml_blas_sgemm(conv_blas, w_ptr, col_buf, out_ptr,
                                   ch_per_group_out, (int)col_w, (int)col_h, 1.0f, 0.0f);

                    if (bias_data) {
                        for (int oc = oc_offset; oc < oc_offset + ch_per_group_out; oc++) {
                            float bv = bias_data[oc];
                            float* row = out_data + ((size_t)b * out_channels + oc) * spatial;
                            for (int j = 0; j < spatial; j++)
                                row[j] += bv;
                        }
                    }
                }
            }
        } else {
            /* Naive direct convolution fallback */
            for (int b = 0; b < batch; b++) {
                for (int g = 0; g < groups; g++) {
                    for (int oc = 0; oc < ch_per_group_out; oc++) {
                        int oc_abs = g * ch_per_group_out + oc;
                        for (int oh_i = 0; oh_i < out_h; oh_i++) {
                            for (int ow_i = 0; ow_i < out_w; ow_i++) {
                                float sum = 0.0f;
                                for (int ic = 0; ic < ch_per_group_in; ic++) {
                                    int ic_abs = g * ch_per_group_in + ic;
                                    for (int kh_i = 0; kh_i < kernel_h; kh_i++) {
                                        for (int kw_i = 0; kw_i < kernel_w; kw_i++) {
                                            int ih = oh_i * stride_h - pad_h + kh_i * dilation_h;
                                            int iw = ow_i * stride_w - pad_w + kw_i * dilation_w;
                                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                                size_t in_idx = ((size_t)b * in_channels + ic_abs) *
                                                                (in_h * in_w) + ih * in_w + iw;
                                                size_t w_idx  = ((size_t)oc_abs * (in_channels / groups) + ic) *
                                                                (kernel_h * kernel_w) + kh_i * kernel_w + kw_i;
                                                sum += in1_data[in_idx] * in2_data[w_idx];
                                            }
                                        }
                                    }
                                }
                                if (bias_data)
                                    sum += bias_data[oc_abs];
                                size_t out_idx = ((size_t)b * out_channels + oc_abs) *
                                                 (out_h * out_w) + oh_i * out_w + ow_i;
                                out_data[out_idx] = sum;
                            }
                        }
                    }
                }
            }
        }
        break;
    }

    case UOP_RESHAPE:
        /* Reshape is a view — same data, different shape. Just copy. */
        if (!in1_data)
            return -1;
        if (out_data == in1_data) {
            /* View shares data — data is already in place, nothing to do */
        } else if (in1_numel == out->numel) {
            memcpy(out_data, in1_data, out->numel * sizeof(float));
        } else {
            for (size_t i = 0; i < out->numel; i++)
                out_data[i] = in1_data[i % in1_numel];
        }
        break;

    case UOP_EXPAND: {
        /* Expand broadcasts input to output shape.
         * The output may be a strided view sharing the input buffer,
         * but the expanded numel can exceed the input buffer size,
         * so we must allocate a fresh output buffer. */
        if (!in1_data)
            return -1;
        if (in1_numel == out->numel) {
            if (out_data != in1_data)
                memcpy(out_data, in1_data, out->numel * sizeof(float));
        } else {
            /* Need a separate output buffer — the view buffer is too small */
            if (out_data == in1_data || out->numel > in1_numel) {
                size_t size = out->numel * cml_dtype_size(out->dtype);
                float* new_buf = (float*)cml_buffer_cache_alloc(size);
                if (!new_buf)
                    return -1;
                out->data      = new_buf;
                out->owns_data = true;
                out_data       = new_buf;
            }
            for (size_t i = 0; i < out->numel; i++) {
                size_t src  = BROADCAST_IDX(node->inputs[0], out, i);
                out_data[i] = in1_data[src];
            }
        }
        break;
    }

    case UOP_LERP: {
        /* lerp(a, b, t) = a + t * (b - a)
         * inputs[0]=a, inputs[1]=b, inputs[2]=t (scalar or tensor) */
        if (!in1_data || !in2_data) return -1;
        float t_scalar = 0.5f;
        bool t_is_tensor = (node->num_inputs >= 3 && node->inputs[2] &&
                            node->inputs[2]->data);
        if (!t_is_tensor && node->num_inputs >= 3 && node->inputs[2] &&
            node->inputs[2]->ir_node) {
            /* t may be stored via a fill IR node; use default scalar if unavailable */
            t_scalar = 0.5f;
        }
        float* t_data = t_is_tensor ? (float*)node->inputs[2]->data : NULL;
        for (size_t i = 0; i < out->numel; i++) {
            float a = in1_data[i % in1_numel];
            float b = in2_data[i % (node->inputs[1]->numel)];
            float t = t_data ? t_data[i % node->inputs[2]->numel] : t_scalar;
            out_data[i] = a + t * (b - a);
        }
        break;
    }

    case UOP_STRIDE: {
        /* View with different strides — copy with stride access.
         * params->new_strides holds new strides; fall back to identity copy. */
        if (!in1_data) return -1;
        StrideParams* sp = node->params ? (StrideParams*)node->params : NULL;
        if (sp && sp->new_strides && sp->num_dims == out->ndim) {
            for (size_t i = 0; i < out->numel; i++) {
                /* Compute flat index using provided strides */
                size_t src = 0;
                size_t rem = i;
                for (int d = out->ndim - 1; d >= 0; d--) {
                    size_t coord = rem % (size_t)out->shape[d];
                    rem /= (size_t)out->shape[d];
                    src += coord * (size_t)sp->new_strides[d];
                }
                out_data[i] = (src < in1_numel) ? in1_data[src] : 0.0f;
            }
        } else {
            size_t n = out->numel < in1_numel ? out->numel : in1_numel;
            memcpy(out_data, in1_data, n * sizeof(float));
        }
        break;
    }

    case UOP_SLICE: {
        /* Slice tensor: extract sub-region defined by start/end/step per dim */
        if (!in1_data) return -1;
        SliceParams* sp = node->params ? (SliceParams*)node->params : NULL;
        if (!sp || !sp->start || !sp->end) {
            /* No params — identity copy */
            size_t n = out->numel < in1_numel ? out->numel : in1_numel;
            memcpy(out_data, in1_data, n * sizeof(float));
            break;
        }
        /* Multi-dim slice: iterate output indices, map to input */
        int ndim = node->inputs[0]->ndim;
        for (size_t i = 0; i < out->numel; i++) {
            size_t src = 0;
            size_t rem = i;
            bool valid = true;
            for (int d = ndim - 1; d >= 0; d--) {
                int out_dim_size = out->shape[d];
                size_t coord = rem % (size_t)out_dim_size;
                rem /= (size_t)out_dim_size;
                int step  = sp->step  ? sp->step[d]  : 1;
                int start = sp->start[d];
                int src_coord = start + (int)coord * step;
                if (src_coord < 0 || src_coord >= node->inputs[0]->shape[d]) {
                    valid = false;
                    break;
                }
                /* Accumulate using input strides */
                size_t in_stride = 1;
                for (int dd = d + 1; dd < ndim; dd++)
                    in_stride *= (size_t)node->inputs[0]->shape[dd];
                src += (size_t)src_coord * in_stride;
            }
            out_data[i] = (valid && src < in1_numel) ? in1_data[src] : 0.0f;
        }
        break;
    }

    case UOP_SPLIT:
    case UOP_CHUNK: {
        /* Split/Chunk: copy a contiguous slice of the input to output.
         * split_dim and split_index are not stored in IRNode; default to dim=0, idx=0. */
        if (!in1_data) return -1;
        int dim      = 0;
        int idx      = 0;
        int in_dim   = node->inputs[0]->shape[dim];
        int chunk_sz = out->shape[dim];
        int offset   = idx * chunk_sz;
        if (offset >= in_dim) { memset(out_data, 0, out->numel * sizeof(float)); break; }

        /* Stride before and after split dim */
        size_t outer = 1, inner = 1;
        for (int d = 0; d < dim; d++)        outer *= (size_t)node->inputs[0]->shape[d];
        for (int d = dim + 1; d < node->inputs[0]->ndim; d++) inner *= (size_t)node->inputs[0]->shape[d];

        for (size_t o = 0; o < outer; o++) {
            for (int c = 0; c < chunk_sz && (offset + c) < in_dim; c++) {
                size_t src_base = (o * (size_t)in_dim  + (size_t)(offset + c)) * inner;
                size_t dst_base = (o * (size_t)chunk_sz + (size_t)c)           * inner;
                memcpy(out_data + dst_base, in1_data + src_base, inner * sizeof(float));
            }
        }
        break;
    }

    case UOP_MESHGRID: {
        /* meshgrid of 1-D inputs — output is N-D grid for input[0] repeated along dim 0.
         * For CPU we only handle the first output (dim 0 grid). */
        if (!in1_data) return -1;
        size_t n = node->inputs[0]->numel;
        size_t repeat = out->numel / (n > 0 ? n : 1);
        for (size_t r = 0; r < repeat; r++)
            memcpy(out_data + r * n, in1_data, n * sizeof(float));
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

    /* Graph cache: reuse pre-allocated buffers for repeated graph structures.
     * Compute signature → lookup → if hit, assign cached buffers to node outputs
     * so cpu_execute_node() doesn't need to malloc per-node. */
    static uint64_t s_last_sig = 0;
    static CMLExecutionPlan* s_last_plan = NULL;
    uint64_t sig = cml_graph_compute_signature(ir);
    CMLGraphCache* cache = cml_get_graph_cache();
    CMLExecutionPlan* plan;
    if (sig == s_last_sig && s_last_plan && s_last_plan->valid) {
        plan = s_last_plan; /* fast path: same graph as last time */
    } else {
        plan = cache ? cml_graph_cache_lookup(cache, sig) : NULL;
        if (plan) { s_last_sig = sig; s_last_plan = plan; }
    }

    if (plan && plan->valid) {
        /* Cache hit: pre-assign buffers to matching nodes */
        struct IRNode* node = ir->head;
        size_t idx = 0;
        while (node && idx < plan->num_nodes) {
            if (node->output && node->output->numel > 0 && !node->output->data) {
                if (plan->buffers[idx] && plan->buffer_sizes[idx] == (size_t)node->output->numel) {
                    node->output->data = plan->buffers[idx];
                    node->output->owns_data = false; /* cache owns the buffer */
                }
            }
            idx++;
            node = node->next;
        }
    }

    struct IRNode* node = ir->head;
    while (node) {
        // Skip nodes that have already been executed and still have valid output
        if (node->is_executed && node->output && node->output->is_executed) {
            node = node->next;
            continue;
        }

        // Skip nodes whose output tensor has been freed (output == NULL)
        if (!node->output) {
            node = node->next;
            continue;
        }

        if (cpu_execute_node(node) != 0) {
            LOG_WARNING("CPU fallback: failed to execute node");
        }
        g_total_nodes_executed++;

        node = node->next;
    }

    /* Cache miss: create plan and cache it for future iterations */
    if (cache && !plan) {
        CMLExecutionPlan* new_plan = cml_create_execution_plan(ir);
        if (new_plan) {
            /* Copy current output data into plan buffers so they persist
             * across cml_reset_ir_context() calls */
            struct IRNode* n = ir->head;
            size_t idx = 0;
            while (n && idx < new_plan->num_nodes) {
                if (n->output && n->output->data && new_plan->buffers[idx] &&
                    new_plan->buffer_sizes[idx] == (size_t)n->output->numel) {
                    memcpy(new_plan->buffers[idx], n->output->data,
                           n->output->numel * sizeof(float));
                }
                idx++;
                n = n->next;
            }
            cml_graph_cache_insert(cache, sig, new_plan);
        }
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

    /* Check if graph should target a GPU backend via CML_BACKEND env */
    const char* backend_env = getenv("CML_BACKEND");
    if (backend_env && (strcasecmp(backend_env, "opencl") == 0 ||
                        strcasecmp(backend_env, "cl") == 0)) {
        CMLDispatchContext* ctx = cml_dispatch_get_global();
        if (ctx) {
            int r = cml_dispatch_execute_on(ctx, CML_BACKEND_OPENCL, ir, NULL, 0, NULL, 0);
            if (r == 0) return 0;
        }
        /* Fall through to CPU if OpenCL fails */
    }

    /* Use V2 fusion scheduler when CML_SCHEDULE_V2=1 (cached) */
    {
        static int s_v2_checked = 0, s_use_v2 = 0;
        if (!s_v2_checked) {
            const char* v2_env = getenv("CML_SCHEDULE_V2");
            s_use_v2 = (v2_env && v2_env[0] == '1');
            s_v2_checked = 1;
        }
        if (s_use_v2) return cml_ir_execute_v2(ir);
    }

    return cpu_execute_ir(ir);
}

int cml_ir_execute_up_to(CMLGraph_t ir, struct IRNode* target_node) {
    if (!ir || !target_node) {
        LOG_ERROR("Invalid arguments to cml_ir_execute_up_to");
        return -1;
    }

    target_node->is_used = true;

    /* Check if graph should target a GPU backend via CML_BACKEND env.
     * Cache env check + dispatch context to avoid repeated getenv()/lookup. */
    static int s_opencl_checked = 0;
    static int s_use_opencl = 0;
    static CMLDispatchContext* s_dispatch_ctx = NULL;
    if (!s_opencl_checked) {
        const char* backend_env = getenv("CML_BACKEND");
        s_use_opencl = (backend_env && (strcasecmp(backend_env, "opencl") == 0 ||
                                         strcasecmp(backend_env, "cl") == 0));
        if (s_use_opencl)
            s_dispatch_ctx = cml_dispatch_get_global();
        s_opencl_checked = 1;
    }
    if (s_use_opencl && s_dispatch_ctx) {
        int r = cml_dispatch_execute_on(s_dispatch_ctx, CML_BACKEND_OPENCL, ir, NULL, 0, NULL, 0);
        if (r == 0) return 0;
    }

    /* Use V2 fusion scheduler when CML_SCHEDULE_V2=1 (cached) */
    {
        static int s_v2_checked = 0, s_use_v2 = 0;
        if (!s_v2_checked) {
            const char* v2_env = getenv("CML_SCHEDULE_V2");
            s_use_v2 = (v2_env && v2_env[0] == '1');
            s_v2_checked = 1;
        }
        if (s_use_v2) return cml_ir_execute_v2(ir);
    }

    return cpu_execute_ir(ir);
}
