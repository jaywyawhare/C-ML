/**
 * @file simd_views.h
 * @brief SIMD-optimized view operations: transpose, gather, scatter, slice copy
 *
 * Follows the same pattern as simd_math.h: compile-time detection,
 * runtime dispatch, scalar fallback. Supports SSE, AVX, AVX-512, ARM NEON.
 */

#ifndef CML_SIMD_VIEWS_H
#define CML_SIMD_VIEWS_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief SIMD-optimized 2D matrix transpose (float32)
 *
 * Uses 4x4 (SSE/NEON) or 8x8 (AVX) register-level transpose blocks.
 * Falls back to cache-blocked scalar for non-aligned remainders.
 *
 * @param src Input matrix in row-major order (rows x cols)
 * @param dst Output matrix in row-major order (cols x rows)
 * @param rows Number of rows in source
 * @param cols Number of columns in source
 */
void simd_transpose_2d_f32(const float* src, float* dst, int rows, int cols);

/**
 * @brief In-place square matrix transpose (float32)
 * @param data Matrix data (n x n)
 * @param n Matrix dimension
 */
void simd_transpose_inplace_f32(float* data, int n);

/**
 * @brief Batched transpose: transpose each matrix in a batch
 * @param src Input [batch, rows, cols]
 * @param dst Output [batch, cols, rows]
 * @param batch Batch size
 * @param rows Rows per matrix
 * @param cols Cols per matrix
 */
void simd_transpose_batched_f32(const float* src, float* dst, int batch, int rows, int cols);

/**
 * @brief Gather elements by index: out[i] = src[indices[i]]
 *
 * SIMD-accelerated using gather intrinsics (AVX2: _mm256_i32gather_ps).
 * Falls back to scalar loop on SSE/NEON.
 *
 * @param src Source array
 * @param indices Index array (int32)
 * @param out Output array
 * @param n Number of elements to gather
 */
void simd_gather_f32(const float* src, const int32_t* indices, float* out, size_t n);

/**
 * @brief Scatter elements by index: dst[indices[i]] = src[i]
 *
 * Note: scatter has no SIMD intrinsic on most platforms, but we
 * optimize with prefetching and loop unrolling.
 *
 * @param src Source array
 * @param indices Index array (int32)
 * @param dst Destination array
 * @param n Number of elements to scatter
 */
void simd_scatter_f32(const float* src, const int32_t* indices, float* dst, size_t n);

/**
 * @brief Scatter-add: dst[indices[i]] += src[i]
 *
 * @param src Source values
 * @param indices Index array
 * @param dst Destination array (accumulated)
 * @param n Number of elements
 */
void simd_scatter_add_f32(const float* src, const int32_t* indices, float* dst, size_t n);

/**
 * @brief Strided copy: copy with different src/dst strides
 *
 * @param src Source pointer
 * @param dst Destination pointer
 * @param n Number of elements to copy
 * @param src_stride Source stride (in floats)
 * @param dst_stride Destination stride (in floats)
 */
void simd_strided_copy_f32(const float* src, float* dst, size_t n,
                           size_t src_stride, size_t dst_stride);

/**
 * @brief Contiguous slice copy (memcpy-like but SIMD-aligned)
 *
 * @param src Source pointer
 * @param dst Destination pointer
 * @param n Number of floats to copy
 */
void simd_copy_f32(const float* src, float* dst, size_t n);

/**
 * @brief Fill array with constant value
 * @param dst Destination
 * @param value Value to fill
 * @param n Number of elements
 */
void simd_fill_f32(float* dst, float value, size_t n);

/**
 * @brief Broadcast copy: replicate src[0..src_n-1] to fill dst[0..dst_n-1]
 *
 * @param src Source array
 * @param src_n Source length
 * @param dst Destination array
 * @param dst_n Destination length (must be multiple of src_n)
 */
void simd_broadcast_copy_f32(const float* src, size_t src_n, float* dst, size_t dst_n);

/**
 * @brief Permute dimensions of an N-D tensor
 *
 * General N-D permute using stride-based copy.
 *
 * @param src Source data
 * @param dst Destination data
 * @param shape Source shape
 * @param strides Source strides (in elements)
 * @param perm Permutation array
 * @param ndim Number of dimensions
 * @param numel Total number of elements
 */
void simd_permute_nd_f32(const float* src, float* dst, const int* shape,
                         const size_t* strides, const int* perm, int ndim, size_t numel);

#ifdef __cplusplus
}
#endif

#endif /* CML_SIMD_VIEWS_H */
