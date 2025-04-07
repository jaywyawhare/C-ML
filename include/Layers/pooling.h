#ifndef POOLING_H
#define POOLING_H

#include "../../include/Core/memory_management.h"

/**
 * @brief Structure representing a Pooling Layer.
 *
 * @param kernel_size Size of the pooling kernel.
 * @param stride Stride of the pooling operation.
 */
typedef struct
{
    int kernel_size;
    int stride;
} PoolingLayer;

/**
 * @brief Initializes a Pooling Layer.
 *
 * @param layer Pointer to the PoolingLayer structure.
 * @param kernel_size Size of the kernel (must be > 0).
 * @param stride Stride of the kernel (must be > 0).
 * @return int Error code.
 */
int initialize_pooling(PoolingLayer *layer, int kernel_size, int stride);

/**
 * @brief Computes the output size for the Pooling Layer.
 *
 * @param input_size Size of the input data.
 * @param kernel_size Size of the kernel.
 * @param stride Stride of the kernel.
 * @return int Output size, or an error code on invalid input.
 */
int compute_pooling_output_size(int input_size, int kernel_size, int stride);

/**
 * @brief Performs the forward pass for the Pooling Layer.
 *
 * @param layer Pointer to the PoolingLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param input_size Size of the input data.
 * @return int Number of output elements, or an error code on failure.
 */
int forward_pooling(PoolingLayer *layer, const float *input, float *output, int input_size);

/**
 * @brief Frees the memory allocated for the Pooling Layer.
 *
 * @param layer Pointer to the PoolingLayer structure.
 * @return int Error code.
 */
int free_pooling(PoolingLayer *layer);

#endif