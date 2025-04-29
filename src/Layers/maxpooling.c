#include <stdlib.h>
#include <stdio.h>
#include "../../include/Core/autograd.h"
#include "../../include/Layers/maxpooling.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Initializes a MaxPooling Layer.
 *
 * @param layer Pointer to the MaxPoolingLayer structure.
 * @param kernel_size Size of the kernel.
 * @param stride Stride of the kernel.
 * @return int Error code.
 */
int initialize_maxpooling(MaxPoolingLayer *layer, int kernel_size, int stride)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_LAYER_ERROR;
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
 * @brief Performs the forward pass for the MaxPooling Layer.
 *
 * @param layer Pointer to the MaxPoolingLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param input_size Size of the input data.
 * @return int Number of output elements, or an error code on failure.
 */
int forward_maxpooling(MaxPoolingLayer *layer, const float *input, float *output, int input_size)
{
    if (!layer || !input || !output)
        return CM_NULL_POINTER_ERROR;

    int output_index = 0;
    for (int i = 0; i <= input_size - layer->kernel_size; i += layer->stride)
    {
        Node *max = tensor(input[i], 1);
        for (int j = 1; j < layer->kernel_size; j++)
        {
            Node *curr = tensor(input[i + j], 1);
            Node *comp = max->value > curr->value ? max : curr;
            if (comp != max)
            {
                cm_safe_free((void **)&max);
            }
            max = comp;
            cm_safe_free((void **)&curr);
        }
        output[output_index++] = max->value;
        cm_safe_free((void **)&max);
    }
    return output_index;
}

// Remove compute_maxpooling_output_size - handled by autograd graph

// Add backward pass with autograd
int backward_maxpooling(MaxPoolingLayer *layer, const float *input, const float *output,
                        float *d_output, float *d_input, int input_size)
{
    if (!layer || !input || !output || !d_output || !d_input)
    {
        LOG_ERROR("Layer or data pointers are NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    // Initialize gradients to zero
    for (int i = 0; i < input_size; i++)
    {
        d_input[i] = 0.0f;
    }

    int output_index = 0;
    for (int i = 0; i <= input_size - layer->kernel_size; i += layer->stride)
    {
        // Find max index in window
        int max_idx = i;
        float max_val = input[i];
        for (int j = 1; j < layer->kernel_size; j++)
        {
            if (input[i + j] > max_val)
            {
                max_val = input[i + j];
                max_idx = i + j;
            }
        }
        // Propagate gradient only to max element
        d_input[max_idx] += d_output[output_index++];
    }

    return CM_SUCCESS;
}

/**
 * @brief Frees the memory allocated for the MaxPooling Layer.
 *
 * @param layer Pointer to the MaxPoolingLayer structure.
 * @return int CM_SUCCESS on success.
 */
int free_maxpooling(MaxPoolingLayer *layer)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_LAYER_ERROR;
    }

    if (layer->kernel_size <= 0 || layer->stride <= 0)
    {
        LOG_ERROR("[free_maxpooling] Warning: Layer has invalid dimensions.\n");
    }

    return CM_SUCCESS;
}
