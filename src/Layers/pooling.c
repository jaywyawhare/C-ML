#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../../include/Core/autograd.h"
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

int forward_pooling(PoolingLayer *layer, const float *input, float *output, int input_size)
{
    if (!layer || !input || !output)
        return CM_NULL_POINTER_ERROR;

    Node *scale_node = tensor(1.0f / layer->kernel_size, 0);
    int output_index = 0;

    for (int i = 0; i <= input_size - layer->kernel_size; i += layer->stride)
    {
        Node *sum_node = tensor(0.0f, 1);
        for (int j = 0; j < layer->kernel_size; j++)
        {
            Node *val_node = tensor(input[i + j], 1);
            Node *new_sum = tensor_add(sum_node, val_node);
            cm_safe_free((void **)&sum_node);
            sum_node = new_sum;
            cm_safe_free((void **)&val_node);
        }
        Node *result = tensor_mul(sum_node, scale_node);
        output[output_index++] = result->value;
        cm_safe_free((void **)&result);
        cm_safe_free((void **)&sum_node);
    }

    cm_safe_free((void **)&scale_node);
    return output_index;
}

// Remove compute_pooling_output_size - handled by autograd

// Add backward pass with autograd
int backward_pooling(PoolingLayer *layer, const float *input, float *output,
                     float *d_output, float *d_input, int input_size)
{
    if (!layer || !input || !output || !d_output || !d_input)
    {
        LOG_ERROR("Layer or data pointers are NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    float grad_scale = 1.0f / layer->kernel_size;
    Node *scale_node = tensor(grad_scale, 0);

    for (int i = 0; i < input_size; i++)
    {
        d_input[i] = 0.0f;
    }

    int output_index = 0;
    for (int i = 0; i <= input_size - layer->kernel_size; i += layer->stride)
    {
        Node *grad_node = tensor(d_output[output_index++], 1);
        Node *scaled_grad = tensor_mul(grad_node, scale_node);

        // Distribute gradient to all inputs in window
        for (int j = 0; j < layer->kernel_size; j++)
        {
            d_input[i + j] += scaled_grad->value;
        }

        cm_safe_free((void **)&grad_node);
        cm_safe_free((void **)&scaled_grad);
    }

    cm_safe_free((void **)&scale_node);
    return CM_SUCCESS;
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
