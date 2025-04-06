#include <stdlib.h>
#include <stdio.h>
#include "../../include/Layers/pooling.h"
#include "../../include/Core/error_codes.h"

#ifndef DEBUG_LOGGING
#define DEBUG_LOGGING 0
#endif

/**
 * @brief Initializes a Polling Layer.
 *
 * @param layer Pointer to the PollingLayer structure.
 * @param kernel_size Size of the kernel.
 * @param stride Stride of the kernel.
 * @return int Error code.
 */
int initialize_polling(PollingLayer *layer, int kernel_size, int stride)
{
    if (layer == NULL)
    {
        fprintf(stderr, "[initialize_polling] Error: Layer is NULL.\n");
        return CM_NULL_POINTER_ERROR;
    }

    if (kernel_size <= 0)
    {
        fprintf(stderr, "[initialize_polling] Error: Invalid kernel size.\n");
        return CM_INVALID_KERNEL_SIZE_ERROR;
    }

    if (stride <= 0)
    {
        fprintf(stderr, "[initialize_polling] Error: Invalid stride.\n");
        return CM_INVALID_STRIDE_ERROR;
    }

    layer->kernel_size = kernel_size;
    layer->stride = stride;
    return CM_SUCCESS;
}

/**
 * @brief Computes the output size for the Polling Layer.
 *
 * @param input_size Size of the input data.
 * @param kernel_size Size of the kernel.
 * @param stride Stride of the kernel.
 * @return int Output size, or an error code on invalid input.
 */
int compute_polling_output_size(int input_size, int kernel_size, int stride)
{
    if (input_size <= 0)
    {
        fprintf(stderr, "[compute_polling_output_size] Error: Input size must be greater than 0.\n");
        return CM_INVALID_INPUT_ERROR;
    }

    if (kernel_size <= 0)
    {
        fprintf(stderr, "[compute_polling_output_size] Error: Invalid kernel size.\n");
        return CM_INVALID_KERNEL_SIZE_ERROR;
    }

    if (stride <= 0)
    {
        fprintf(stderr, "[compute_polling_output_size] Error: Invalid stride.\n");
        return CM_INVALID_STRIDE_ERROR;
    }

    if (input_size < kernel_size)
    {
        fprintf(stderr, "[compute_polling_output_size] Error: Input size is smaller than kernel size.\n");
        return CM_INPUT_SIZE_SMALLER_THAN_KERNEL_ERROR;
    }

    return (input_size - kernel_size) / stride + 1;
}

/**
 * @brief Performs the forward pass for the Polling Layer.
 *
 * @param layer Pointer to the PollingLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param input_size Size of the input data.
 * @return int Number of output elements, or an error code on failure.
 */
int forward_polling(PollingLayer *layer, const float *input, float *output, int input_size)
{
    if (layer == NULL)
    {
        fprintf(stderr, "[forward_polling] Error: Layer is NULL.\n");
        return CM_NULL_POINTER_ERROR;
    }

    if (input == NULL || output == NULL)
    {
        fprintf(stderr, "[forward_polling] Error: Null input or output pointer.\n");
        return CM_NULL_POINTER_ERROR;
    }

    if (layer->kernel_size <= 0)
    {
        fprintf(stderr, "[forward_polling] Error: Invalid kernel size.\n");
        return CM_INVALID_KERNEL_SIZE_ERROR;
    }

    if (layer->stride <= 0)
    {
        fprintf(stderr, "[forward_polling] Error: Invalid stride.\n");
        return CM_INVALID_STRIDE_ERROR;
    }

    if (input_size < layer->kernel_size)
    {
        fprintf(stderr, "[forward_polling] Error: Input size is smaller than kernel size.\n");
        return CM_INPUT_SIZE_SMALLER_THAN_KERNEL_ERROR;
    }

    float kernel_reciprocal = 1.0f / layer->kernel_size;
    int output_index = 0;
    for (int i = 0; i <= input_size - layer->kernel_size; i += layer->stride)
    {
        float sum = 0.0f;
        for (int j = 0; j < layer->kernel_size; ++j)
        {
            sum += input[i + j];
        }
        output[output_index++] = sum * kernel_reciprocal;

#if DEBUG_LOGGING
        printf("[forward_polling] Debug: Output[%d]: %f\n", output_index - 1, output[output_index - 1]);
#endif
    }

    return output_index;
}

/**
 * @brief Frees the memory allocated for the Polling Layer.
 *
 * @param layer Pointer to the PollingLayer structure.
 * @return int Error code.
 */
int free_polling(PollingLayer *layer)
{
    if (layer == NULL)
    {
        fprintf(stderr, "[free_polling] Error: Layer is NULL.\n");
        return CM_NULL_POINTER_ERROR;
    }
    return CM_SUCCESS;
}
