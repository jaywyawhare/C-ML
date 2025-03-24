#ifndef POLLING_H
#define POLLING_H

#include "../../include/Core/memory_management.h"

/**
 * @brief Structure representing a Polling Layer.
 *
 * @param kernel_size Size of the pooling kernel.
 * @param stride Stride of the pooling operation.
 */
typedef struct
{
    int kernel_size;
    int stride;
} PollingLayer;

/**
 * @brief Initializes a Polling Layer.
 *
 * @param layer Pointer to the PollingLayer structure.
 * @param kernel_size Size of the kernel (must be > 0).
 * @param stride Stride of the kernel (must be > 0).
 * @return int Error code.
 */
int initialize_polling(PollingLayer *layer, int kernel_size, int stride);

/**
 * @brief Computes the output size for the Polling Layer.
 *
 * @param input_size Size of the input data.
 * @param kernel_size Size of the kernel.
 * @param stride Stride of the kernel.
 * @return int Output size, or an error code on invalid input.
 */
int compute_polling_output_size(int input_size, int kernel_size, int stride);

/**
 * @brief Performs the forward pass for the Polling Layer.
 *
 * @param layer Pointer to the PollingLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param input_size Size of the input data.
 * @return int Number of output elements, or an error code on failure.
 */
int forward_polling(PollingLayer *layer, const float *input, float *output, int input_size);

/**
 * @brief Frees the memory allocated for the Polling Layer.
 *
 * @param layer Pointer to the PollingLayer structure.
 * @return int Error code.
 */
int free_polling(PollingLayer *layer);

#endif