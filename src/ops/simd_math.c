/*
 * Portable element-wise / transcendental math kernels.
 *
 * These were formerly hand-written SSE/AVX/AVX-512/NEON intrinsic kernels with
 * runtime CPUID dispatch and an optional SLEEF dependency.  That hand-rolled
 * SIMD has been removed: shape-specialized SIMD is now emitted by the LLVM JIT
 * backend (src/ops/ir/llvm/llvm_backend.c), which bakes each tensor's shape and
 * broadcast pattern in as compile-time constants and lets LLVM select the host
 * vector width.  What remains here is a plain, portable scalar reference that
 * the C compiler auto-vectorizes at -O3; it backs the interpreter fallback
 * (cpu_execute_node) and eager paths that do not go through the JIT.
 *
 * The public API (including CMLSimdCaps) is preserved for source compatibility.
 */
#include "ops/simd_math.h"
#include "ops/simd_utils.h"
#include "core/cml_flags.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Capability reporting.  No hand-rolled SIMD remains, so nothing is reported
 * as available; the compiler and the JIT handle vectorization.
 * ---------------------------------------------------------------------- */
CMLSimdCaps cml_detect_simd_caps(void) {
    CMLSimdCaps caps = {0};
    return caps;
}

const CMLSimdCaps* cml_get_simd_caps(void) {
    static CMLSimdCaps caps = {0};
    return &caps;
}

void cml_print_simd_caps(void) {
    printf("SIMD: portable scalar kernels (vectorized by the compiler / LLVM JIT)\n");
}

/* -------------------------------------------------------------------------
 * Unary: out[i] = f(in[i])
 * ---------------------------------------------------------------------- */
#define UNARY(name, expr)                                                  \
    void name(const float* in, float* out, size_t n) {                     \
        if (!in || !out || n == 0) return;                                 \
        for (size_t i = 0; i < n; i++) { float x = in[i]; out[i] = (expr); } \
    }

/* Schraudolph-style polynomial exp: packs a linear function of x into the IEEE
 * exponent field (~1e-3 relative error, no libm call). Used only when
 * TRANSCENDENTAL forces approximation (>=2); default keeps the accurate expf. */
static inline float fast_expf(float x) {
    if (x > 88.0f)  x = 88.0f;
    if (x < -88.0f) x = -88.0f;
    union { uint32_t i; float f; } v;
    v.i = (uint32_t)(12102203.0f * x + 1064866805.0f);
    return v.f;
}

void simd_exp_f32(const float* in, float* out, size_t n) {
    if (!in || !out || n == 0) return;
    int approx = cml_flag(CML_FLAG_TRANSCENDENTAL) >= 2;
    for (size_t i = 0; i < n; i++) {
        float x = in[i];
        out[i]  = approx ? fast_expf(x) : expf(x);
    }
}

UNARY(simd_log_f32,     logf(x))
UNARY(simd_sqrt_f32,    sqrtf(x))
UNARY(simd_rsqrt_f32,   1.0f / sqrtf(x))
UNARY(simd_recip_f32,   1.0f / x)
UNARY(simd_abs_f32,     fabsf(x))
UNARY(simd_sin_f32,     sinf(x))
UNARY(simd_cos_f32,     cosf(x))
UNARY(simd_tan_f32,     tanf(x))
UNARY(simd_tanh_f32,    tanhf(x))
UNARY(simd_sigmoid_f32, 1.0f / (1.0f + expf(-x)))
UNARY(simd_neg_f32,     -x)

#undef UNARY

/* -------------------------------------------------------------------------
 * Binary: out[i] = f(a[i], b[i])
 * ---------------------------------------------------------------------- */
#define BINARY(name, expr)                                                 \
    void name(const float* a, const float* b, float* out, size_t n) {      \
        if (!a || !b || !out || n == 0) return;                            \
        for (size_t i = 0; i < n; i++) { float x = a[i], y = b[i]; out[i] = (expr); } \
    }

BINARY(simd_pow_f32,   powf(x, y))
BINARY(simd_cmplt_f32, (x <  y) ? 1.0f : 0.0f)
BINARY(simd_cmpgt_f32, (x >  y) ? 1.0f : 0.0f)
BINARY(simd_min_f32,   (x <  y) ? x : y)
BINARY(simd_max_f32,   (x >  y) ? x : y)
BINARY(simd_add_f32,   x + y)
BINARY(simd_sub_f32,   x - y)
BINARY(simd_mul_f32,   x * y)
BINARY(simd_div_f32,   x / y)

#undef BINARY

/* out[i] = cond[i] != 0 ? a[i] : b[i] */
void simd_where_f32(const float* cond, const float* a, const float* b, float* out, size_t n) {
    if (!cond || !a || !b || !out || n == 0) return;
    for (size_t i = 0; i < n; i++)
        out[i] = (cond[i] != 0.0f) ? a[i] : b[i];
}

/* -------------------------------------------------------------------------
 * 2D transpose (cache-blocked scalar; compiler vectorizes the inner copy).
 * ---------------------------------------------------------------------- */
void simd_transpose_f32(const float* in, float* out, int rows, int cols) {
    if (!in || !out || rows <= 0 || cols <= 0) return;

    const int BLOCK = 8;
    for (int i0 = 0; i0 < rows; i0 += BLOCK) {
        for (int j0 = 0; j0 < cols; j0 += BLOCK) {
            int i_end = (i0 + BLOCK < rows) ? i0 + BLOCK : rows;
            int j_end = (j0 + BLOCK < cols) ? j0 + BLOCK : cols;
            for (int i = i0; i < i_end; i++)
                for (int j = j0; j < j_end; j++)
                    out[j * rows + i] = in[i * cols + j];
        }
    }
}

/* -------------------------------------------------------------------------
 * Scalar-broadcast helpers.
 * ---------------------------------------------------------------------- */
void simd_add_scalar_f32(const float* a, float scalar, float* out, size_t n) {
    if (!a || !out || n == 0) return;
    for (size_t i = 0; i < n; i++) out[i] = a[i] + scalar;
}

void simd_mul_scalar_f32(const float* a, float scalar, float* out, size_t n) {
    if (!a || !out || n == 0) return;
    for (size_t i = 0; i < n; i++) out[i] = a[i] * scalar;
}

/* -------------------------------------------------------------------------
 * Broadcasting elementwise (out[i] = a[i%a_n] op b[i%b_n]).
 * ---------------------------------------------------------------------- */
void simd_add_broadcast_f32(const float* a, size_t a_n, const float* b, size_t b_n, float* out,
                            size_t out_n) {
    if (!a || !b || !out || out_n == 0) return;
    if (a_n == 1) { simd_add_scalar_f32(b, a[0], out, out_n); return; }
    if (b_n == 1) { simd_add_scalar_f32(a, b[0], out, out_n); return; }
    if (a_n == b_n && a_n == out_n) { simd_add_f32(a, b, out, out_n); return; }
    if (a_n == out_n && out_n % b_n == 0) {
        size_t repeats = out_n / b_n;
        for (size_t r = 0; r < repeats; r++)
            simd_add_f32(&a[r * b_n], b, &out[r * b_n], b_n);
        return;
    }
    for (size_t i = 0; i < out_n; i++) {
        size_t ai = (a_n == 1) ? 0 : i % a_n;
        size_t bi = (b_n == 1) ? 0 : i % b_n;
        out[i] = a[ai] + b[bi];
    }
}

void simd_mul_broadcast_f32(const float* a, size_t a_n, const float* b, size_t b_n, float* out,
                            size_t out_n) {
    if (!a || !b || !out || out_n == 0) return;
    if (a_n == 1) { simd_mul_scalar_f32(b, a[0], out, out_n); return; }
    if (b_n == 1) { simd_mul_scalar_f32(a, b[0], out, out_n); return; }
    if (a_n == b_n && a_n == out_n) { simd_mul_f32(a, b, out, out_n); return; }
    if (a_n == out_n && out_n % b_n == 0) {
        size_t repeats = out_n / b_n;
        for (size_t r = 0; r < repeats; r++)
            simd_mul_f32(&a[r * b_n], b, &out[r * b_n], b_n);
        return;
    }
    for (size_t i = 0; i < out_n; i++) {
        size_t ai = (a_n == 1) ? 0 : i % a_n;
        size_t bi = (b_n == 1) ? 0 : i % b_n;
        out[i] = a[ai] * b[bi];
    }
}

void simd_max_broadcast_f32(const float* a, size_t a_n, const float* b, size_t b_n, float* out,
                            size_t out_n) {
    if (!a || !b || !out || out_n == 0) return;
    if (a_n == b_n && a_n == out_n) { simd_max_f32(a, b, out, out_n); return; }
    if (b_n == 1) {
        float scalar = b[0];
        for (size_t i = 0; i < out_n; i++) {
            size_t ai = (a_n == 1) ? 0 : i % a_n;
            out[i] = (a[ai] > scalar) ? a[ai] : scalar;
        }
        return;
    }
    if (a_n == out_n && out_n % b_n == 0) {
        size_t repeats = out_n / b_n;
        for (size_t r = 0; r < repeats; r++)
            simd_max_f32(&a[r * b_n], b, &out[r * b_n], b_n);
        return;
    }
    for (size_t i = 0; i < out_n; i++) {
        size_t ai = (a_n == 1) ? 0 : i % a_n;
        size_t bi = (b_n == 1) ? 0 : i % b_n;
        out[i] = (a[ai] > b[bi]) ? a[ai] : b[bi];
    }
}

/* -------------------------------------------------------------------------
 * Thread-parallel variants (threading only — the per-chunk kernel is scalar).
 * ---------------------------------------------------------------------- */
#include "backend/threadpool.h"
#include "alloc/cml_allocator.h"

static size_t g_parallel_threshold = 10000;

void simd_set_parallel_threshold(size_t threshold) { g_parallel_threshold = threshold; }

typedef struct { const float* a; const float* b; float* out; } ParallelBinaryData;

static void parallel_add_task(void* data, size_t start, size_t end) {
    ParallelBinaryData* d = (ParallelBinaryData*)data;
    simd_add_f32(&d->a[start], &d->b[start], &d->out[start], end - start);
}

void simd_add_f32_parallel(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out || n == 0) return;
    if (n < g_parallel_threshold) { simd_add_f32(a, b, out, n); return; }
    ThreadPool* pool = threadpool_get_global();
    if (!pool) { simd_add_f32(a, b, out, n); return; }
    ParallelBinaryData data = {a, b, out};
    threadpool_parallel_for(pool, parallel_add_task, &data, n);
}

static void parallel_mul_task(void* data, size_t start, size_t end) {
    ParallelBinaryData* d = (ParallelBinaryData*)data;
    simd_mul_f32(&d->a[start], &d->b[start], &d->out[start], end - start);
}

void simd_mul_f32_parallel(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out || n == 0) return;
    if (n < g_parallel_threshold) { simd_mul_f32(a, b, out, n); return; }
    ThreadPool* pool = threadpool_get_global();
    if (!pool) { simd_mul_f32(a, b, out, n); return; }
    ParallelBinaryData data = {a, b, out};
    threadpool_parallel_for(pool, parallel_mul_task, &data, n);
}

static void parallel_sub_task(void* data, size_t start, size_t end) {
    ParallelBinaryData* d = (ParallelBinaryData*)data;
    simd_sub_f32(&d->a[start], &d->b[start], &d->out[start], end - start);
}
void simd_sub_f32_parallel(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out || n == 0) return;
    if (n < g_parallel_threshold) { simd_sub_f32(a, b, out, n); return; }
    ThreadPool* pool = threadpool_get_global();
    if (!pool) { simd_sub_f32(a, b, out, n); return; }
    ParallelBinaryData data = {a, b, out};
    threadpool_parallel_for(pool, parallel_sub_task, &data, n);
}

static void parallel_max_task(void* data, size_t start, size_t end) {
    ParallelBinaryData* d = (ParallelBinaryData*)data;
    simd_max_f32(&d->a[start], &d->b[start], &d->out[start], end - start);
}
void simd_max_f32_parallel(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out || n == 0) return;
    if (n < g_parallel_threshold) { simd_max_f32(a, b, out, n); return; }
    ThreadPool* pool = threadpool_get_global();
    if (!pool) { simd_max_f32(a, b, out, n); return; }
    ParallelBinaryData data = {a, b, out};
    threadpool_parallel_for(pool, parallel_max_task, &data, n);
}

typedef struct { const float* in; float* out; } ParallelUnaryData;

static void parallel_exp_task(void* data, size_t start, size_t end) {
    ParallelUnaryData* d = (ParallelUnaryData*)data;
    simd_exp_f32(&d->in[start], &d->out[start], end - start);
}

void simd_exp_f32_parallel(const float* in, float* out, size_t n) {
    if (!in || !out || n == 0) return;
    if (n < g_parallel_threshold) { simd_exp_f32(in, out, n); return; }
    ThreadPool* pool = threadpool_get_global();
    if (!pool) { simd_exp_f32(in, out, n); return; }
    ParallelUnaryData data = {in, out};
    threadpool_parallel_for(pool, parallel_exp_task, &data, n);
}

typedef struct { const float* data; float* partial_sums; size_t num_threads; } ParallelSumData;

static void parallel_sum_task(void* data, size_t start, size_t end) {
    ParallelSumData* d = (ParallelSumData*)data;
    float sum = simd_sum_float(&d->data[start], end - start);
    size_t chunk_size = (end - start);
    if (chunk_size > 0) {
        size_t thread_idx = start / chunk_size;
        if (thread_idx < d->num_threads) d->partial_sums[thread_idx] = sum;
    }
}

float simd_sum_f32_parallel(const float* data, size_t n) {
    if (!data || n == 0) return 0.0f;
    if (n < g_parallel_threshold) return simd_sum_float(data, n);
    ThreadPool* pool = threadpool_get_global();
    if (!pool) return simd_sum_float(data, n);
    size_t num_threads = threadpool_get_num_threads(pool);
    if (num_threads == 0) num_threads = 1;
    float* partial_sums = cml_calloc(num_threads, sizeof(float));
    if (!partial_sums) return simd_sum_float(data, n);
    ParallelSumData pdata = {data, partial_sums, num_threads};
    threadpool_parallel_for(pool, parallel_sum_task, &pdata, n);
    float total = 0.0f;
    for (size_t i = 0; i < num_threads; i++) total += partial_sums[i];
    cml_free(partial_sums);
    return total;
}
