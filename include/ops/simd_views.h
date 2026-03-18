/*
 * SIMD-optimized view operations: transpose, gather, scatter, slice copy.
 * Compile-time detection, runtime dispatch, scalar fallback.
 * Supports SSE, AVX, AVX-512, ARM NEON.
 */

#ifndef CML_SIMD_VIEWS_H
#define CML_SIMD_VIEWS_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Uses 4x4 (SSE/NEON) or 8x8 (AVX) register-level transpose blocks.
   Falls back to cache-blocked scalar for non-aligned remainders. */
void simd_transpose_2d_f32(const float* src, float* dst, int rows, int cols);
void simd_transpose_inplace_f32(float* data, int n);
void simd_transpose_batched_f32(const float* src, float* dst, int batch, int rows, int cols);

/* SIMD-accelerated using gather intrinsics (AVX2: _mm256_i32gather_ps).
   Falls back to scalar loop on SSE/NEON. */
void simd_gather_f32(const float* src, const int32_t* indices, float* out, size_t n);

/* No SIMD scatter intrinsic on most platforms; optimized with prefetching and loop unrolling. */
void simd_scatter_f32(const float* src, const int32_t* indices, float* dst, size_t n);
void simd_scatter_add_f32(const float* src, const int32_t* indices, float* dst, size_t n);

void simd_strided_copy_f32(const float* src, float* dst, size_t n,
                           size_t src_stride, size_t dst_stride);
void simd_copy_f32(const float* src, float* dst, size_t n);
void simd_fill_f32(float* dst, float value, size_t n);

/* Replicate src[0..src_n-1] to fill dst[0..dst_n-1] */
void simd_broadcast_copy_f32(const float* src, size_t src_n, float* dst, size_t dst_n);

/* General N-D permute using stride-based copy. */
void simd_permute_nd_f32(const float* src, float* dst, const int* shape,
                         const size_t* strides, const int* perm, int ndim, size_t numel);

#ifdef __cplusplus
}
#endif

#endif /* CML_SIMD_VIEWS_H */
