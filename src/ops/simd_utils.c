/*
 * Portable scalar reductions (sum / strided sum / max / min).
 *
 * The hand-rolled SSE/NEON horizontal-reduction intrinsics were removed; these
 * simple loops auto-vectorize at -O3, and the JIT backend emits shape-specific
 * reduction kernels for the compiled path.
 */
#include "ops/simd_utils.h"
#include <stddef.h>

/* Pairwise summation.
 *
 * A single running float accumulator loses O(count * eps) relative accuracy,
 * because every partial sum is rounded against a total that grows without
 * bound. Summing 1e6 copies of 0.1f that way gave 100958 instead of 100000 --
 * ~1%% error, which is material for a loss or a normalisation statistic.
 * Recursively halving keeps the error at O(eps * log2(count)) for the same
 * number of additions; the same reason numpy and PyTorch sum this way.
 *
 * The base case uses four independent accumulators so the leaf still
 * auto-vectorises and already shortens each dependency chain by 4x. */
#define CML_PAIRWISE_BLOCK 128

static float sum_pairwise(const float* data, size_t count, size_t stride) {
    if (count <= CML_PAIRWISE_BLOCK) {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        size_t i = 0;
        for (; i + 4 <= count; i += 4) {
            s0 += data[(i + 0) * stride];
            s1 += data[(i + 1) * stride];
            s2 += data[(i + 2) * stride];
            s3 += data[(i + 3) * stride];
        }
        for (; i < count; i++)
            s0 += data[i * stride];
        return (s0 + s1) + (s2 + s3);
    }
    size_t half = count / 2;
    return sum_pairwise(data, half, stride) +
           sum_pairwise(data + half * stride, count - half, stride);
}

float simd_sum_float(const float* data, size_t count) {
    if (!data || count == 0)
        return 0.0f;
    return sum_pairwise(data, count, 1);
}

float simd_sum_float_strided(const float* data, size_t count, size_t stride) {
    if (!data || count == 0)
        return 0.0f;
    return sum_pairwise(data, count, stride);
}

float simd_max_float(const float* data, size_t count) {
    if (!data || count == 0)
        return 0.0f;
    float max_val = data[0];
    for (size_t i = 1; i < count; i++)
        if (data[i] > max_val)
            max_val = data[i];
    return max_val;
}

float simd_min_float(const float* data, size_t count) {
    if (!data || count == 0)
        return 0.0f;
    float min_val = data[0];
    for (size_t i = 1; i < count; i++)
        if (data[i] < min_val)
            min_val = data[i];
    return min_val;
}
