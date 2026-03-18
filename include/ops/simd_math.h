/*
 * SIMD-optimized math operations for all transcendental and element-wise functions.
 * Supports SSE, AVX, AVX-512, and ARM NEON with runtime detection.
 * Uses SLEEF library if available for high-accuracy transcendentals,
 * with custom polynomial fallbacks otherwise.
 */

#ifndef CML_SIMD_MATH_H
#define CML_SIMD_MATH_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bool has_sse;    // SSE/SSE2 (128-bit, 4 floats)
    bool has_sse4;   // SSE4.1/4.2
    bool has_avx;    // AVX (256-bit, 8 floats)
    bool has_avx2;   // AVX2 + FMA
    bool has_avx512; // AVX-512F (512-bit, 16 floats)
    bool has_neon;   // ARM NEON (128-bit, 4 floats)
    bool has_sleef;  // SLEEF library available
} CMLSimdCaps;

CMLSimdCaps cml_detect_simd_caps(void);
const CMLSimdCaps* cml_get_simd_caps(void);
void cml_print_simd_caps(void);

/* out[i] = f(in[i]) */
void simd_exp_f32(const float* in, float* out, size_t n);
void simd_log_f32(const float* in, float* out, size_t n);
void simd_sqrt_f32(const float* in, float* out, size_t n);
void simd_rsqrt_f32(const float* in, float* out, size_t n);
void simd_recip_f32(const float* in, float* out, size_t n);
void simd_abs_f32(const float* in, float* out, size_t n);
void simd_sin_f32(const float* in, float* out, size_t n);
void simd_cos_f32(const float* in, float* out, size_t n);
void simd_tan_f32(const float* in, float* out, size_t n);
void simd_tanh_f32(const float* in, float* out, size_t n);
void simd_sigmoid_f32(const float* in, float* out, size_t n);
void simd_neg_f32(const float* in, float* out, size_t n);

/* out[i] = f(a[i], b[i]) */
void simd_pow_f32(const float* a, const float* b, float* out, size_t n);
void simd_cmplt_f32(const float* a, const float* b, float* out, size_t n);
void simd_cmpgt_f32(const float* a, const float* b, float* out, size_t n);
void simd_min_f32(const float* a, const float* b, float* out, size_t n);
void simd_max_f32(const float* a, const float* b, float* out, size_t n);
void simd_add_f32(const float* a, const float* b, float* out, size_t n);
void simd_sub_f32(const float* a, const float* b, float* out, size_t n);
void simd_mul_f32(const float* a, const float* b, float* out, size_t n);
void simd_div_f32(const float* a, const float* b, float* out, size_t n);

/* out[i] = cond[i] != 0 ? a[i] : b[i] */
void simd_where_f32(const float* cond, const float* a, const float* b, float* out, size_t n);

void simd_transpose_f32(const float* in, float* out, int rows, int cols);

/* out[i] = a[i] + scalar */
void simd_add_scalar_f32(const float* a, float scalar, float* out, size_t n);
/* out[i] = a[i] * scalar */
void simd_mul_scalar_f32(const float* a, float scalar, float* out, size_t n);

/* out[i] = a[i % a_n] + b[i % b_n] */
void simd_add_broadcast_f32(const float* a, size_t a_n, const float* b, size_t b_n, float* out,
                            size_t out_n);
void simd_mul_broadcast_f32(const float* a, size_t a_n, const float* b, size_t b_n, float* out,
                            size_t out_n);
void simd_max_broadcast_f32(const float* a, size_t a_n, const float* b, size_t b_n, float* out,
                            size_t out_n);

void simd_set_parallel_threshold(size_t threshold);
void simd_add_f32_parallel(const float* a, const float* b, float* out, size_t n);
void simd_mul_f32_parallel(const float* a, const float* b, float* out, size_t n);
void simd_exp_f32_parallel(const float* in, float* out, size_t n);
float simd_sum_f32_parallel(const float* data, size_t n);

#ifdef __cplusplus
}
#endif

#endif // CML_SIMD_MATH_H
