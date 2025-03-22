#include <stdlib.h>
#include <stdio.h>            // For error messages
#include "../../include/Layers/pooling.h"
#include "../../include/Core/error_codes.h" // Include the new error codes
#include "../../include/Core/memory_management.h"

#define DEBUG_LOGGING 0 // Set to 1 to enable debug logs

PollingLayer *polling_layer_create(int kernel_size, int stride)
{
    if (kernel_size <= 0 || stride <= 0)
    {
        fprintf(stderr, "[polling_layer_create] Error: Invalid kernel size (%d) or stride (%d).\n", kernel_size, stride);
        return NULL;
    }

    PollingLayer *layer = (PollingLayer *)cm_safe_malloc(sizeof(PollingLayer), __FILE__, __LINE__);
    if (!layer)
    {
        fprintf(stderr, "[polling_layer_create] Error: Memory allocation failed for PollingLayer.\n");
        return NULL;
    }
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    return layer;
}

int polling_layer_output_size(int input_size, int kernel_size, int stride)
{
    if (input_size < 0)
    {
        fprintf(stderr, "[polling_layer_output_size] Error: Input size (%d) cannot be negative.\n", input_size);
        return (int)CM_INVALID_PARAMETER_ERROR;
    }
    return (input_size - kernel_size) / stride + 1;
}

int polling_layer_forward(PollingLayer *layer, const float *input, float *output, int input_size)
{
    if (!layer)
    {
        fprintf(stderr, "[polling_layer_forward] Error: Null layer pointer.\n");
        return (int)CM_NULL_POINTER_ERROR;
    }

    if (!input || !output)
    {
        fprintf(stderr, "[polling_layer_forward] Error: Null input (%p) or output (%p) pointer.\n", input, output);
        return (int)CM_NULL_POINTER_ERROR;
    }

    if (layer->stride <= 0)
    {
        fprintf(stderr, "[polling_layer_forward] Error: Stride (%d) must be greater than 0.\n", layer->stride);
        return (int)CM_INVALID_STRIDE_ERROR;
    }

    if (layer->kernel_size <= 0)
    {
        fprintf(stderr, "[polling_layer_forward] Error: Kernel size (%d) must be greater than 0.\n", layer->kernel_size);
        return (int)CM_INVALID_KERNEL_SIZE_ERROR;
    }

    if (input_size < layer->kernel_size)
    {
        fprintf(stderr, "[polling_layer_forward] Error: Input size (%d) is smaller than kernel size (%d).\n", input_size, layer->kernel_size);
        return (int)CM_INPUT_SIZE_SMALLER_THAN_KERNEL_ERROR;
    }

    float kernel_reciprocal = 1.0f / layer->kernel_size; // Precompute reciprocal
    int output_index = 0;
    for (int i = 0; i <= input_size - layer->kernel_size; i += layer->stride)
    {
        float sum = 0.0f;
        for (int j = 0; j < layer->kernel_size; ++j)
        {
            sum += input[i + j];
        }
        output[output_index++] = sum * kernel_reciprocal; // Use multiplication instead of division
#if DEBUG_LOGGING
        printf("[polling_layer_forward] Output[%d]: %f\n", output_index - 1, output[output_index - 1]);
#endif
    }
    return output_index; // Return the number of output elements
}

void polling_layer_free(PollingLayer *layer)
{
    if (layer)
    {
        cm_safe_free(layer);
    }
}
