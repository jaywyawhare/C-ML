/**
 * @file simd_utils.c
 * @brief SIMD optimization utilities for reductions and element-wise operations
 */

#include "ops/simd_utils.h"
#include <string.h>
#include <stddef.h>

#ifdef __SSE__
#include <xmmintrin.h>
#include <emmintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

float simd_sum_float(const float* data, size_t count) {
    if (!data || count == 0)
        return 0.0f;

#ifdef __SSE__
    if (count >= 4) {
        __m128 sum_vec = _mm_setzero_ps();
        size_t i       = 0;
        for (; i + 4 <= count; i += 4) {
            __m128 vec = _mm_loadu_ps(&data[i]);
            sum_vec    = _mm_add_ps(sum_vec, vec);
        }

        // Horizontal sum (SSE2 compatible)
        __m128 shuf = _mm_shuffle_ps(sum_vec, sum_vec, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(sum_vec, shuf);
        shuf        = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(1, 0, 3, 2));
        sums        = _mm_add_ps(sums, shuf);
        float sum   = _mm_cvtss_f32(sums);
        for (; i < count; i++) {
            sum += data[i];
        }

        return sum;
    }
#elif defined(__ARM_NEON)
    if (count >= 4) {
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        size_t i            = 0;
        for (; i + 4 <= count; i += 4) {
            float32x4_t vec = vld1q_f32(&data[i]);
            sum_vec         = vaddq_f32(sum_vec, vec);
        }
        float sum = vaddvq_f32(sum_vec);
        for (; i < count; i++) {
            sum += data[i];
        }

        return sum;
    }
#endif
    float sum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        sum += data[i];
    }
    return sum;
}

float simd_sum_float_strided(const float* data, size_t count, size_t stride) {
    if (!data || count == 0)
        return 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        sum += data[i * stride];
    }
    return sum;
}

float simd_max_float(const float* data, size_t count) {
    if (!data || count == 0)
        return 0.0f;

#ifdef __SSE__
    if (count >= 4) {
        __m128 max_vec = _mm_loadu_ps(data);
        size_t i       = 4;
        for (; i + 4 <= count; i += 4) {
            __m128 vec = _mm_loadu_ps(&data[i]);
            max_vec    = _mm_max_ps(max_vec, vec);
        }
        __m128 shuf   = _mm_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 maxs   = _mm_max_ps(max_vec, shuf);
        shuf          = _mm_shuffle_ps(maxs, maxs, _MM_SHUFFLE(1, 0, 3, 2));
        maxs          = _mm_max_ps(maxs, shuf);
        float max_val = _mm_cvtss_f32(maxs);
        for (; i < count; i++) {
            if (data[i] > max_val) {
                max_val = data[i];
            }
        }

        return max_val;
    }
#elif defined(__ARM_NEON)
    if (count >= 4) {
        float32x4_t max_vec = vld1q_f32(data);
        size_t i            = 4;
        for (; i + 4 <= count; i += 4) {
            float32x4_t vec = vld1q_f32(&data[i]);
            max_vec         = vmaxq_f32(max_vec, vec);
        }
        float max_val = vmaxvq_f32(max_vec);
        for (; i < count; i++) {
            if (data[i] > max_val) {
                max_val = data[i];
            }
        }

        return max_val;
    }
#endif
    float max_val = data[0];
    for (size_t i = 1; i < count; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
}

float simd_min_float(const float* data, size_t count) {
    if (!data || count == 0)
        return 0.0f;

#ifdef __SSE__
    if (count >= 4) {
        __m128 min_vec = _mm_loadu_ps(data);
        size_t i       = 4;
        for (; i + 4 <= count; i += 4) {
            __m128 vec = _mm_loadu_ps(&data[i]);
            min_vec    = _mm_min_ps(min_vec, vec);
        }
        __m128 shuf   = _mm_shuffle_ps(min_vec, min_vec, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 mins   = _mm_min_ps(min_vec, shuf);
        shuf          = _mm_shuffle_ps(mins, mins, _MM_SHUFFLE(1, 0, 3, 2));
        mins          = _mm_min_ps(mins, shuf);
        float min_val = _mm_cvtss_f32(mins);
        for (; i < count; i++) {
            if (data[i] < min_val) {
                min_val = data[i];
            }
        }

        return min_val;
    }
#elif defined(__ARM_NEON)
    if (count >= 4) {
        float32x4_t min_vec = vld1q_f32(data);
        size_t i            = 4;
        for (; i + 4 <= count; i += 4) {
            float32x4_t vec = vld1q_f32(&data[i]);
            min_vec         = vminq_f32(min_vec, vec);
        }
        float min_val = vminvq_f32(min_vec);
        for (; i < count; i++) {
            if (data[i] < min_val) {
                min_val = data[i];
            }
        }

        return min_val;
    }
#endif
    float min_val = data[0];
    for (size_t i = 1; i < count; i++) {
        if (data[i] < min_val) {
            min_val = data[i];
        }
    }
    return min_val;
}
