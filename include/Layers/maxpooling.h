#ifndef MAXPOOLING_H
#define MAXPOOLING_H

#include "../../include/Core/memory_management.h"

/**
 * Structure representing a max pooling layer.
 * kernel_size: Size of the pooling kernel.
 * stride: Stride of the pooling operation.
 */
typedef struct
{
    int kernel_size;
    int stride;
} MaxPoolingLayer;

/**
 * Creates a new max pooling layer.
 * @param kernel_size Size of the pooling kernel (must be > 0).
 * @param stride Stride of the pooling operation (must be > 0).
 * @return Pointer to the created MaxPoolingLayer, or NULL if invalid parameters are provided.
 */
MaxPoolingLayer *maxpooling_layer_create(int kernel_size, int stride);

/**
 * Calculates the output size of the max pooling layer.
 * @param input_size Size of the input.
 * @param kernel_size Size of the pooling kernel.
 * @param stride Stride of the pooling operation.
 * @return The calculated output size.
 */
int maxpooling_layer_output_size(int input_size, int kernel_size, int stride);

/**
 * Performs the forward pass of the max pooling layer.
 * @param layer Pointer to the MaxPoolingLayer.
 * @param input Pointer to the input data.
 * @param output Pointer to the output data.
 * @param input_size Size of the input data.
 * @return The number of output elements written, or a CM_Error code:
 *         CM_NULL_POINTER_ERROR: Null input or output pointer.
 *         CM_INVALID_STRIDE_ERROR: Stride is less than or equal to 0.
 *         CM_INVALID_KERNEL_SIZE_ERROR: Kernel size is less than or equal to 0.
 *         CM_INPUT_SIZE_SMALLER_THAN_KERNEL_ERROR: Input size is smaller than kernel size.
 */
int maxpooling_layer_forward(MaxPoolingLayer *layer, const float *input, float *output, int input_size);

/**
 * Frees the memory allocated for the max pooling layer.
 * @param layer Pointer to the MaxPoolingLayer to be freed.
 */
void maxpooling_layer_free(MaxPoolingLayer *layer);

#endif // MAXPOOLING_H
