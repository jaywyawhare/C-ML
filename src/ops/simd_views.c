/**
 * @file simd_views.c
 * @brief SIMD-optimized view operations: transpose, gather, scatter, slice copy
 */

#include "ops/simd_views.h"
#include <string.h>
#include <stdlib.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #define CML_X86 1
  #include <immintrin.h>
  #ifdef __SSE__
    #define CML_HAS_SSE 1
  #endif
  #ifdef __AVX__
    #define CML_HAS_AVX 1
  #endif
  #ifdef __AVX2__
    #define CML_HAS_AVX2 1
  #endif
  #ifdef __AVX512F__
    #define CML_HAS_AVX512 1
  #endif
#elif defined(__aarch64__) || defined(__arm__)
  #define CML_ARM 1
  #include <arm_neon.h>
  #define CML_HAS_NEON 1
#endif

/* Scalar cache-blocked transpose (used as fallback when no SIMD available) */
__attribute__((unused))
static void transpose_scalar(const float* src, float* dst, int rows, int cols) {
    const int BLOCK = 32;
    for (int i = 0; i < rows; i += BLOCK) {
        for (int j = 0; j < cols; j += BLOCK) {
            int i_end = (i + BLOCK < rows) ? i + BLOCK : rows;
            int j_end = (j + BLOCK < cols) ? j + BLOCK : cols;
            for (int ii = i; ii < i_end; ii++) {
                for (int jj = j; jj < j_end; jj++) {
                    dst[jj * rows + ii] = src[ii * cols + jj];
                }
            }
        }
    }
}

#ifdef CML_HAS_SSE
/* 4x4 SSE transpose micro-kernel */
static inline void transpose_4x4_sse(const float* src, int src_stride,
                                      float* dst, int dst_stride) {
    __m128 r0 = _mm_loadu_ps(src);
    __m128 r1 = _mm_loadu_ps(src + src_stride);
    __m128 r2 = _mm_loadu_ps(src + 2 * src_stride);
    __m128 r3 = _mm_loadu_ps(src + 3 * src_stride);

    __m128 t0 = _mm_unpacklo_ps(r0, r1);
    __m128 t1 = _mm_unpackhi_ps(r0, r1);
    __m128 t2 = _mm_unpacklo_ps(r2, r3);
    __m128 t3 = _mm_unpackhi_ps(r2, r3);

    r0 = _mm_movelh_ps(t0, t2);
    r1 = _mm_movehl_ps(t2, t0);
    r2 = _mm_movelh_ps(t1, t3);
    r3 = _mm_movehl_ps(t3, t1);

    _mm_storeu_ps(dst, r0);
    _mm_storeu_ps(dst + dst_stride, r1);
    _mm_storeu_ps(dst + 2 * dst_stride, r2);
    _mm_storeu_ps(dst + 3 * dst_stride, r3);
}
#endif

#ifdef CML_HAS_AVX
/* 8x8 AVX transpose micro-kernel */
static inline void transpose_8x8_avx(const float* src, int src_stride,
                                      float* dst, int dst_stride) {
    __m256 r0 = _mm256_loadu_ps(src);
    __m256 r1 = _mm256_loadu_ps(src + src_stride);
    __m256 r2 = _mm256_loadu_ps(src + 2 * src_stride);
    __m256 r3 = _mm256_loadu_ps(src + 3 * src_stride);
    __m256 r4 = _mm256_loadu_ps(src + 4 * src_stride);
    __m256 r5 = _mm256_loadu_ps(src + 5 * src_stride);
    __m256 r6 = _mm256_loadu_ps(src + 6 * src_stride);
    __m256 r7 = _mm256_loadu_ps(src + 7 * src_stride);

    __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    __m256 t7 = _mm256_unpackhi_ps(r6, r7);

    __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    r0 = _mm256_permute2f128_ps(s0, s4, 0x20);
    r1 = _mm256_permute2f128_ps(s1, s5, 0x20);
    r2 = _mm256_permute2f128_ps(s2, s6, 0x20);
    r3 = _mm256_permute2f128_ps(s3, s7, 0x20);
    r4 = _mm256_permute2f128_ps(s0, s4, 0x31);
    r5 = _mm256_permute2f128_ps(s1, s5, 0x31);
    r6 = _mm256_permute2f128_ps(s2, s6, 0x31);
    r7 = _mm256_permute2f128_ps(s3, s7, 0x31);

    _mm256_storeu_ps(dst, r0);
    _mm256_storeu_ps(dst + dst_stride, r1);
    _mm256_storeu_ps(dst + 2 * dst_stride, r2);
    _mm256_storeu_ps(dst + 3 * dst_stride, r3);
    _mm256_storeu_ps(dst + 4 * dst_stride, r4);
    _mm256_storeu_ps(dst + 5 * dst_stride, r5);
    _mm256_storeu_ps(dst + 6 * dst_stride, r6);
    _mm256_storeu_ps(dst + 7 * dst_stride, r7);
}
#endif

void simd_transpose_2d_f32(const float* src, float* dst, int rows, int cols) {
#ifdef CML_HAS_AVX
    int bi, bj;
    for (bi = 0; bi + 8 <= rows; bi += 8) {
        for (bj = 0; bj + 8 <= cols; bj += 8) {
            transpose_8x8_avx(src + bi * cols + bj, cols, dst + bj * rows + bi, rows);
        }
    }
    /* Handle remainder */
    for (int i = bi; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst[j * rows + i] = src[i * cols + j];
    for (int i = 0; i < bi; i++)
        for (int j = bj; j < cols; j++)
            dst[j * rows + i] = src[i * cols + j];
#elif defined(CML_HAS_SSE)
    int bi, bj;
    for (bi = 0; bi + 4 <= rows; bi += 4) {
        for (bj = 0; bj + 4 <= cols; bj += 4) {
            transpose_4x4_sse(src + bi * cols + bj, cols, dst + bj * rows + bi, rows);
        }
    }
    for (int i = bi; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst[j * rows + i] = src[i * cols + j];
    for (int i = 0; i < bi; i++)
        for (int j = bj; j < cols; j++)
            dst[j * rows + i] = src[i * cols + j];
#else
    transpose_scalar(src, dst, rows, cols);
#endif
}

void simd_transpose_inplace_f32(float* data, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float tmp = data[i * n + j];
            data[i * n + j] = data[j * n + i];
            data[j * n + i] = tmp;
        }
    }
}

void simd_transpose_batched_f32(const float* src, float* dst, int batch, int rows, int cols) {
    size_t src_stride = (size_t)rows * cols;
    size_t dst_stride = (size_t)cols * rows;
    for (int b = 0; b < batch; b++) {
        simd_transpose_2d_f32(src + b * src_stride, dst + b * dst_stride, rows, cols);
    }
}

void simd_gather_f32(const float* src, const int32_t* indices, float* out, size_t n) {
#ifdef CML_HAS_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i idx = _mm256_loadu_si256((const __m256i*)(indices + i));
        __m256 gathered = _mm256_i32gather_ps(src, idx, 4);
        _mm256_storeu_ps(out + i, gathered);
    }
    for (; i < n; i++) {
        out[i] = src[indices[i]];
    }
#else
    for (size_t i = 0; i < n; i++) {
        out[i] = src[indices[i]];
    }
#endif
}

void simd_scatter_f32(const float* src, const int32_t* indices, float* dst, size_t n) {
    /* No SIMD scatter intrinsic on most platforms - use scalar with prefetch */
    for (size_t i = 0; i < n; i++) {
#if defined(CML_X86) && defined(__SSE__)
        if (i + 4 < n)
            _mm_prefetch((const char*)(dst + indices[i + 4]), _MM_HINT_T0);
#endif
        dst[indices[i]] = src[i];
    }
}

void simd_scatter_add_f32(const float* src, const int32_t* indices, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
#if defined(CML_X86) && defined(__SSE__)
        if (i + 4 < n)
            _mm_prefetch((const char*)(dst + indices[i + 4]), _MM_HINT_T0);
#endif
        dst[indices[i]] += src[i];
    }
}

void simd_strided_copy_f32(const float* src, float* dst, size_t n,
                           size_t src_stride, size_t dst_stride) {
    if (src_stride == 1 && dst_stride == 1) {
        simd_copy_f32(src, dst, n);
        return;
    }
    for (size_t i = 0; i < n; i++) {
        dst[i * dst_stride] = src[i * src_stride];
    }
}

void simd_copy_f32(const float* src, float* dst, size_t n) {
#ifdef CML_HAS_AVX
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, v);
    }
    for (; i < n; i++)
        dst[i] = src[i];
#elif defined(CML_HAS_SSE)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 v = _mm_loadu_ps(src + i);
        _mm_storeu_ps(dst + i, v);
    }
    for (; i < n; i++)
        dst[i] = src[i];
#elif defined(CML_HAS_NEON)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        vst1q_f32(dst + i, v);
    }
    for (; i < n; i++)
        dst[i] = src[i];
#else
    memcpy(dst, src, n * sizeof(float));
#endif
}

void simd_fill_f32(float* dst, float value, size_t n) {
#ifdef CML_HAS_AVX
    __m256 v = _mm256_set1_ps(value);
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, v);
    for (; i < n; i++)
        dst[i] = value;
#elif defined(CML_HAS_SSE)
    __m128 v = _mm_set1_ps(value);
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        _mm_storeu_ps(dst + i, v);
    for (; i < n; i++)
        dst[i] = value;
#elif defined(CML_HAS_NEON)
    float32x4_t v = vdupq_n_f32(value);
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(dst + i, v);
    for (; i < n; i++)
        dst[i] = value;
#else
    for (size_t i = 0; i < n; i++)
        dst[i] = value;
#endif
}

void simd_broadcast_copy_f32(const float* src, size_t src_n, float* dst, size_t dst_n) {
    if (src_n == 0 || dst_n == 0)
        return;
    if (src_n == 1) {
        simd_fill_f32(dst, src[0], dst_n);
        return;
    }
    /* Copy the first block */
    simd_copy_f32(src, dst, src_n);
    /* Double the copied portion each time */
    size_t copied = src_n;
    while (copied < dst_n) {
        size_t to_copy = copied;
        if (copied + to_copy > dst_n)
            to_copy = dst_n - copied;
        memcpy(dst + copied, dst, to_copy * sizeof(float));
        copied += to_copy;
    }
}

void simd_permute_nd_f32(const float* src, float* dst, const int* shape,
                         const size_t* strides, const int* perm, int ndim, size_t numel) {
    /* Compute destination strides from permuted shape */
    int* dst_shape = (int*)malloc(ndim * sizeof(int));
    size_t* dst_strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (!dst_shape || !dst_strides) {
        free(dst_shape);
        free(dst_strides);
        /* Fallback: plain copy */
        memcpy(dst, src, numel * sizeof(float));
        return;
    }

    for (int i = 0; i < ndim; i++)
        dst_shape[i] = shape[perm[i]];

    dst_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
        dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1];

    /* Map each source element to destination via stride calculation */
    int* coords = (int*)calloc(ndim, sizeof(int));
    if (!coords) {
        free(dst_shape);
        free(dst_strides);
        memcpy(dst, src, numel * sizeof(float));
        return;
    }

    for (size_t idx = 0; idx < numel; idx++) {
        /* Compute source offset from coords */
        size_t src_offset = 0;
        for (int d = 0; d < ndim; d++)
            src_offset += coords[d] * strides[d];

        /* Compute destination offset using permuted coords */
        size_t dst_offset = 0;
        for (int d = 0; d < ndim; d++)
            dst_offset += coords[perm[d]] * dst_strides[d];

        dst[dst_offset] = src[src_offset];

        /* Increment coords */
        for (int d = ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < shape[d])
                break;
            coords[d] = 0;
        }
    }

    free(coords);
    free(dst_shape);
    free(dst_strides);
}
