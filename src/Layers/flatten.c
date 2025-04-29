#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../../include/Core/autograd.h"
#include "../../include/Layers/flatten.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/logging.h"

int initialize_flatten(FlattenLayer *layer, int input_size)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_LAYER_ERROR;
    }

    if (input_size <= 0)
    {
        LOG_ERROR("Invalid input size (%d).", input_size);
        return CM_INVALID_LAYER_DIMENSIONS_ERROR;
    }

    layer->input_size = input_size;
    layer->output_size = input_size;
    return CM_SUCCESS;
}

int forward_flatten(FlattenLayer *layer, float *input, float *output)
{
    if (!layer || !input || !output)
    {
        LOG_ERROR("Layer, input, or output is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    // Create autograd nodes for input and output
    for (int i = 0; i < layer->input_size; i++)
    {
        Node *input_node = tensor(input[i], 1);
        output[i] = input_node->value;
        cm_safe_free((void **)&input_node);
    }

    return CM_SUCCESS;
}

int backward_flatten(FlattenLayer *layer, float *input, float *output, float *d_output, float *d_input)
{
    if (!layer || !input || !output || !d_output || !d_input)
    {
        LOG_ERROR("One or more arguments are NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    // Use autograd for gradient propagation
    for (int i = 0; i < layer->input_size; i++)
    {
        Node *grad_node = tensor(d_output[i], 1);
        d_input[i] = grad_node->value;
        cm_safe_free((void **)&grad_node);
    }

    return CM_SUCCESS;
}

int free_flatten(FlattenLayer *layer)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    return CM_SUCCESS;
}
