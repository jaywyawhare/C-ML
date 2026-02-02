/**
 * @file simd_utils.h
 * @brief SIMD optimization utilities for reductions and element-wise operations
 */

#ifndef CML_SIMD_UTILS_H
#define CML_SIMD_UTILS_H

#include <stddef.h>

/**
 * SIMD-optimized sum reduction for contiguous float arrays
 * @param data Pointer to float array
 * @param count Number of elements
 * @return Sum of all elements
 */
float simd_sum_float(const float* data, size_t count);

/**
 * SIMD-optimized sum reduction for strided float arrays
 * @param data Pointer to float array
 * @param count Number of elements
 * @param stride Stride between elements
 * @return Sum of all elements
 */
float simd_sum_float_strided(const float* data, size_t count, size_t stride);

/**
 * SIMD-optimized max reduction for contiguous float arrays
 * @param data Pointer to float array
 * @param count Number of elements
 * @return Maximum value
 */
float simd_max_float(const float* data, size_t count);

/**
 * SIMD-optimized min reduction for contiguous float arrays
 * @param data Pointer to float array
 * @param count Number of elements
 * @return Minimum value
 */
float simd_min_float(const float* data, size_t count);

#endif // CML_SIMD_UTILS_H
