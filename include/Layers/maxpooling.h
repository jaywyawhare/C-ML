#ifndef MAXPOOLING_H
#define MAXPOOLING_H

#include "../../include/Core/memory_management.h"

/**
 * @brief Structure representing a MaxPooling Layer.
 *
 * @param kernel_size Size of the pooling kernel.
 * @param stride Stride of the pooling operation.
 */
typedef struct
{
    int kernel_size;
    int stride;
} MaxPoolingLayer;

/**
 * @brief Initializes a MaxPooling Layer.
 *
 * @param layer Pointer to the MaxPoolingLayer structure.
 * @param kernel_size Size of the kernel (must be > 0).
 * @param stride Stride of the kernel (must be > 0).
 * @return int Error code.
 */
int initialize_maxpooling(MaxPoolingLayer *layer, int kernel_size, int stride);

/**
 * @brief Computes the output size for the MaxPooling Layer.
 *
 * @param input_size Size of the input data.
 * @param kernel_size Size of the kernel.
 * @param stride Stride of the kernel.
 * @return int Output size, or an error code on invalid input.
 */
int compute_maxpooling_output_size(int input_size, int kernel_size, int stride);

/**
 * @brief Performs the forward pass for the MaxPooling Layer.
 *
 * @param layer Pointer to the MaxPoolingLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param input_size Size of the input data.
 * @return int Number of output elements, or an error code on failure.
 */
int forward_maxpooling(MaxPoolingLayer *layer, const float *input, float *output, int input_size);

/**
 * @brief Frees the memory allocated for the MaxPooling Layer.
 *
 * @param layer Pointer to the MaxPoolingLayer structure.
 * @return int Error code.
 */
int free_maxpooling(MaxPoolingLayer *layer);

#endif