/*
 * Portable scalar reductions (sum / strided sum / max / min).
 *
 * The hand-rolled SSE/NEON horizontal-reduction intrinsics were removed; these
 * simple loops auto-vectorize at -O3, and the JIT backend emits shape-specific
 * reduction kernels for the compiled path.
 */
#include "ops/simd_utils.h"
#include <stddef.h>

float simd_sum_float(const float* data, size_t count) {
    if (!data || count == 0)
        return 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < count; i++)
        sum += data[i];
    return sum;
}

float simd_sum_float_strided(const float* data, size_t count, size_t stride) {
    if (!data || count == 0)
        return 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < count; i++)
        sum += data[i * stride];
    return sum;
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
