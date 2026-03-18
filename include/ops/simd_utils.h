#ifndef CML_SIMD_UTILS_H
#define CML_SIMD_UTILS_H

#include <stddef.h>

float simd_sum_float(const float* data, size_t count);
float simd_sum_float_strided(const float* data, size_t count, size_t stride);
float simd_max_float(const float* data, size_t count);
float simd_min_float(const float* data, size_t count);

#endif // CML_SIMD_UTILS_H
