#ifndef POLLING_H
#define POLLING_H

#include "../../include/Core/memory_management.h"

/**
 * Structure representing a polling layer.
 * kernel_size: Size of the pooling kernel.
 * stride: Stride of the pooling operation.
 */
typedef struct
{
    int kernel_size;
    int stride;
} PollingLayer;

/**
 * Creates a new polling layer.
 * @param kernel_size Size of the pooling kernel (must be > 0).
 * @param stride Stride of the pooling operation (must be > 0).
 * @return Pointer to the created PollingLayer, or NULL if invalid parameters are provided.
 */
PollingLayer *polling_layer_create(int kernel_size, int stride);

/**
 * Calculates the output size of the polling layer.
 * @param input_size Size of the input.
 * @param kernel_size Size of the pooling kernel.
 * @param stride Stride of the pooling operation.
 * @return The calculated output size.
 */
int polling_layer_output_size(int input_size, int kernel_size, int stride);

/**
 * Performs the forward pass of the polling layer.
 * @param layer Pointer to the PollingLayer.
 * @param input Pointer to the input data.
 * @param output Pointer to the output data.
 * @param input_size Size of the input data.
 * @return The number of output elements written, or a CM_Error code:
 *         CM_NULL_POINTER_ERROR: Null input or output pointer.
 *         CM_INVALID_STRIDE_ERROR: Stride is less than or equal to 0.
 *         CM_INVALID_KERNEL_SIZE_ERROR: Kernel size is less than or equal to 0.
 *         CM_INPUT_SIZE_SMALLER_THAN_KERNEL_ERROR: Input size is smaller than kernel size.
 */
int polling_layer_forward(PollingLayer *layer, const float *input, float *output, int input_size);

/**
 * Frees the memory allocated for the polling layer.
 * @param layer Pointer to the PollingLayer to be freed.
 */
void polling_layer_free(PollingLayer *layer);

#endif // POLLING_H
