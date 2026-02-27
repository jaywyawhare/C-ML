/**
 * @file simd_math.h
 * @brief SIMD-optimized math operations for all transcendental and element-wise functions
 *
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

// Runtime SIMD Capability Detection

/**
 * @brief SIMD capability flags detected at runtime
 */
typedef struct {
    bool has_sse;    // SSE/SSE2 (128-bit, 4 floats)
    bool has_sse4;   // SSE4.1/4.2
    bool has_avx;    // AVX (256-bit, 8 floats)
    bool has_avx2;   // AVX2 + FMA
    bool has_avx512; // AVX-512F (512-bit, 16 floats)
    bool has_neon;   // ARM NEON (128-bit, 4 floats)
    bool has_sleef;  // SLEEF library available
} CMLSimdCaps;

/**
 * @brief Detect available SIMD capabilities at runtime
 * @return Structure with capability flags
 */
CMLSimdCaps cml_detect_simd_caps(void);

/**
 * @brief Get cached SIMD capabilities (lazy initialization)
 * @return Pointer to cached capabilities
 */
const CMLSimdCaps* cml_get_simd_caps(void);

/**
 * @brief Print SIMD capabilities to stdout
 */
void cml_print_simd_caps(void);

// Unary Math Operations (Vectorized)

/**
 * @brief Vectorized exponential: out[i] = exp(in[i])
 * @param in Input array
 * @param out Output array
 * @param n Number of elements
 */
void simd_exp_f32(const float* in, float* out, size_t n);

/**
 * @brief Vectorized natural logarithm: out[i] = log(in[i])
 * @param in Input array
 * @param out Output array
 * @param n Number of elements
 */
void simd_log_f32(const float* in, float* out, size_t n);

/**
 * @brief Vectorized square root: out[i] = sqrt(in[i])
 * @param in Input array
 * @param out Output array
 * @param n Number of elements
 */
void simd_sqrt_f32(const float* in, float* out, size_t n);

/**
 * @brief Vectorized reciprocal square root: out[i] = 1/sqrt(in[i])
 * @param in Input array
 * @param out Output array
 * @param n Number of elements
 */
void simd_rsqrt_f32(const float* in, float* out, size_t n);

/**
 * @brief Vectorized reciprocal: out[i] = 1/in[i]
 * @param in Input array
 * @param out Output array
 * @param n Number of elements
 */
void simd_recip_f32(const float* in, float* out, size_t n);

/**
 * @brief Vectorized absolute value: out[i] = |in[i]|
 * @param in Input array
 * @param out Output array
 * @param n Number of elements
 */
void simd_abs_f32(const float* in, float* out, size_t n);

/**
 * @brief Vectorized sine: out[i] = sin(in[i])
 * @param in Input array
 * @param out Output array
 * @param n Number of elements
 */
void simd_sin_f32(const float* in, float* out, size_t n);

/**
 * @brief Vectorized cosine: out[i] = cos(in[i])
 * @param in Input array
 * @param out Output array
 * @param n Number of elements
 */
void simd_cos_f32(const float* in, float* out, size_t n);

/**
 * @brief Vectorized tangent: out[i] = tan(in[i])
 * @param in Input array
 * @param out Output array
 * @param n Number of elements
 */
void simd_tan_f32(const float* in, float* out, size_t n);

/**
 * @brief Vectorized hyperbolic tangent: out[i] = tanh(in[i])
 * @param in Input array
 * @param out Output array
 * @param n Number of elements
 */
void simd_tanh_f32(const float* in, float* out, size_t n);

/**
 * @brief Vectorized sigmoid: out[i] = 1/(1+exp(-in[i]))
 * @param in Input array
 * @param out Output array
 * @param n Number of elements
 */
void simd_sigmoid_f32(const float* in, float* out, size_t n);

/**
 * @brief Vectorized negation: out[i] = -in[i]
 * @param in Input array
 * @param out Output array
 * @param n Number of elements
 */
void simd_neg_f32(const float* in, float* out, size_t n);

// Binary Math Operations (Vectorized)

/**
 * @brief Vectorized power: out[i] = a[i]^b[i]
 * @param a Base array
 * @param b Exponent array
 * @param out Output array
 * @param n Number of elements
 */
void simd_pow_f32(const float* a, const float* b, float* out, size_t n);

/**
 * @brief Vectorized less-than comparison: out[i] = (a[i] < b[i]) ? 1.0f : 0.0f
 * @param a First array
 * @param b Second array
 * @param out Output array
 * @param n Number of elements
 */
void simd_cmplt_f32(const float* a, const float* b, float* out, size_t n);

/**
 * @brief Vectorized greater-than comparison: out[i] = (a[i] > b[i]) ? 1.0f : 0.0f
 * @param a First array
 * @param b Second array
 * @param out Output array
 * @param n Number of elements
 */
void simd_cmpgt_f32(const float* a, const float* b, float* out, size_t n);

/**
 * @brief Vectorized element-wise minimum: out[i] = min(a[i], b[i])
 * @param a First array
 * @param b Second array
 * @param out Output array
 * @param n Number of elements
 */
void simd_min_f32(const float* a, const float* b, float* out, size_t n);

/**
 * @brief Vectorized element-wise maximum: out[i] = max(a[i], b[i])
 * @param a First array
 * @param b Second array
 * @param out Output array
 * @param n Number of elements
 */
void simd_max_f32(const float* a, const float* b, float* out, size_t n);

/**
 * @brief Vectorized addition: out[i] = a[i] + b[i]
 * @param a First array
 * @param b Second array
 * @param out Output array
 * @param n Number of elements
 */
void simd_add_f32(const float* a, const float* b, float* out, size_t n);

/**
 * @brief Vectorized subtraction: out[i] = a[i] - b[i]
 * @param a First array
 * @param b Second array
 * @param out Output array
 * @param n Number of elements
 */
void simd_sub_f32(const float* a, const float* b, float* out, size_t n);

/**
 * @brief Vectorized multiplication: out[i] = a[i] * b[i]
 * @param a First array
 * @param b Second array
 * @param out Output array
 * @param n Number of elements
 */
void simd_mul_f32(const float* a, const float* b, float* out, size_t n);

/**
 * @brief Vectorized division: out[i] = a[i] / b[i]
 * @param a First array
 * @param b Second array
 * @param out Output array
 * @param n Number of elements
 */
void simd_div_f32(const float* a, const float* b, float* out, size_t n);

// Ternary Operations (Vectorized)

/**
 * @brief Vectorized conditional select: out[i] = cond[i] != 0 ? a[i] : b[i]
 * @param cond Condition array
 * @param a Array for true case
 * @param b Array for false case
 * @param out Output array
 * @param n Number of elements
 */
void simd_where_f32(const float* cond, const float* a, const float* b, float* out, size_t n);

// Matrix/Tensor Operations

/**
 * @brief Cache-blocked matrix transpose
 * @param in Input matrix (rows x cols, row-major)
 * @param out Output matrix (cols x rows, row-major)
 * @param rows Number of rows in input
 * @param cols Number of columns in input
 */
void simd_transpose_f32(const float* in, float* out, int rows, int cols);

// Broadcast Operations (SIMD-accelerated)

/**
 * @brief Broadcast scalar addition: out[i] = a[i] + scalar
 */
void simd_add_scalar_f32(const float* a, float scalar, float* out, size_t n);

/**
 * @brief Broadcast scalar multiplication: out[i] = a[i] * scalar
 */
void simd_mul_scalar_f32(const float* a, float scalar, float* out, size_t n);

/**
 * @brief Broadcast add with different sizes: out[i] = a[i % a_n] + b[i % b_n]
 * @param a First array
 * @param a_n Size of first array
 * @param b Second array
 * @param b_n Size of second array
 * @param out Output array
 * @param out_n Output size
 */
void simd_add_broadcast_f32(const float* a, size_t a_n, const float* b, size_t b_n, float* out,
                            size_t out_n);

/**
 * @brief Broadcast multiply with different sizes
 */
void simd_mul_broadcast_f32(const float* a, size_t a_n, const float* b, size_t b_n, float* out,
                            size_t out_n);

/**
 * @brief Broadcast max with different sizes
 */
void simd_max_broadcast_f32(const float* a, size_t a_n, const float* b, size_t b_n, float* out,
                            size_t out_n);

// Parallel Operations (multi-threaded)

/**
 * @brief Set minimum array size for parallel execution (default: 10000)
 */
void simd_set_parallel_threshold(size_t threshold);

/**
 * @brief Parallel SIMD add: out[i] = a[i] + b[i]
 */
void simd_add_f32_parallel(const float* a, const float* b, float* out, size_t n);

/**
 * @brief Parallel SIMD multiply: out[i] = a[i] * b[i]
 */
void simd_mul_f32_parallel(const float* a, const float* b, float* out, size_t n);

/**
 * @brief Parallel SIMD exp: out[i] = exp(a[i])
 */
void simd_exp_f32_parallel(const float* in, float* out, size_t n);

/**
 * @brief Parallel sum reduction
 */
float simd_sum_f32_parallel(const float* data, size_t n);

#ifdef __cplusplus
}
#endif

#endif // CML_SIMD_MATH_H
