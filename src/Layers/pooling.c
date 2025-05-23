#include <stdlib.h>
#include <stdio.h>
#include "../../include/Layers/pooling.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Initializes a Pooling Layer.
 *
 * @param layer Pointer to the PoolingLayer structure.
 * @param kernel_size Size of the kernel.
 * @param stride Stride of the kernel.
 * @return int Error code.
 */
int initialize_pooling(PoolingLayer *layer, int kernel_size, int stride)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    if (kernel_size <= 0)
    {
        LOG_ERROR("Invalid kernel size.");
        return CM_INVALID_KERNEL_SIZE_ERROR;
    }

    if (stride <= 0)
    {
        LOG_ERROR("Invalid stride.");
        return CM_INVALID_STRIDE_ERROR;
    }

    layer->kernel_size = kernel_size;
    layer->stride = stride;
    return CM_SUCCESS;
}

/**
 * @brief Computes the output size for the Pooling Layer.
 *
 * @param input_size Size of the input data.
 * @param kernel_size Size of the kernel.
 * @param stride Stride of the kernel.
 * @return int Output size, or an error code on invalid input.
 */
int compute_pooling_output_size(int input_size, int kernel_size, int stride)
{
    if (input_size <= 0)
    {
        LOG_ERROR("Input size must be greater than 0.");
        return CM_INVALID_INPUT_ERROR;
    }

    if (kernel_size <= 0)
    {
        LOG_ERROR("Invalid kernel size.");
        return CM_INVALID_KERNEL_SIZE_ERROR;
    }

    if (stride <= 0)
    {
        LOG_ERROR("Invalid stride.");
        return CM_INVALID_STRIDE_ERROR;
    }

    if (input_size < kernel_size)
    {
        LOG_ERROR("Input size is smaller than kernel size.");
        return CM_INPUT_SIZE_SMALLER_THAN_KERNEL_ERROR;
    }

    return (input_size - kernel_size) / stride + 1;
}

/**
 * @brief Performs the forward pass for the Pooling Layer.
 *
 * @param layer Pointer to the PoolingLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param input_size Size of the input data.
 * @return int Number of output elements, or an error code on failure.
 */
int forward_pooling(PoolingLayer *layer, const float *input, float *output, int input_size)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    if (input == NULL || output == NULL)
    {
        LOG_ERROR("Null input or output pointer.");
        return CM_NULL_POINTER_ERROR;
    }

    if (layer->kernel_size <= 0)
    {
        LOG_ERROR("Invalid kernel size.");
        return CM_INVALID_KERNEL_SIZE_ERROR;
    }

    if (layer->stride <= 0)
    {
        LOG_ERROR("Invalid stride.");
        return CM_INVALID_STRIDE_ERROR;
    }

    if (input_size < layer->kernel_size)
    {
        LOG_ERROR("Input size is smaller than kernel size.");
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
        LOG_DEBUG("Output[%d]: %f", output_index - 1, output[output_index - 1]);
    }

    return output_index;
}

/**
 * @brief Frees the memory allocated for the Pooling Layer.
 *
 * @param layer Pointer to the PoolingLayer structure.
 * @return int Error code.
 */
int free_pooling(PoolingLayer *layer)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_POINTER_ERROR;
    }
    return CM_SUCCESS;
}
