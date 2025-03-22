#include <stdlib.h>
#include <stdio.h>
#include "../../include/Layers/maxpooling.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"

#define DEBUG_LOGGING 0 

MaxPoolingLayer *maxpooling_layer_create(int kernel_size, int stride)
{
    if (kernel_size <= 0 || stride <= 0)
    {
        fprintf(stderr, "[maxpooling_layer_create] Error: Invalid kernel size (%d) or stride (%d).\n", kernel_size, stride);
        return NULL;
    }

    MaxPoolingLayer *layer = (MaxPoolingLayer *)cm_safe_malloc(sizeof(MaxPoolingLayer), __FILE__, __LINE__);
    if (!layer)
    {
        fprintf(stderr, "[maxpooling_layer_create] Error: Memory allocation failed for MaxPoolingLayer.\n");
        return NULL;
    }
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    return layer;
}

int maxpooling_layer_output_size(int input_size, int kernel_size, int stride)
{
    if (input_size < 0)
    {
        fprintf(stderr, "[maxpooling_layer_output_size] Error: Input size (%d) cannot be negative.\n", input_size);
        return (int)CM_INVALID_PARAMETER_ERROR;
    }
    return (input_size - kernel_size) / stride + 1;
}

int maxpooling_layer_forward(MaxPoolingLayer *layer, const float *input, float *output, int input_size)
{
    if (!layer)
    {
        fprintf(stderr, "[maxpooling_layer_forward] Error: Null layer pointer.\n");
        return (int)CM_NULL_POINTER_ERROR;
    }

    if (!input || !output)
    {
        fprintf(stderr, "[maxpooling_layer_forward] Error: Null input (%p) or output (%p) pointer.\n", input, output);
        return (int)CM_NULL_POINTER_ERROR;
    }

    if (layer->stride <= 0)
    {
        fprintf(stderr, "[maxpooling_layer_forward] Error: Stride (%d) must be greater than 0.\n", layer->stride);
        return (int)CM_INVALID_STRIDE_ERROR;
    }

    if (layer->kernel_size <= 0)
    {
        fprintf(stderr, "[maxpooling_layer_forward] Error: Kernel size (%d) must be greater than 0.\n", layer->kernel_size);
        return (int)CM_INVALID_KERNEL_SIZE_ERROR;
    }

    if (input_size < layer->kernel_size)
    {
        fprintf(stderr, "[maxpooling_layer_forward] Error: Input size (%d) is smaller than kernel size (%d).\n", input_size, layer->kernel_size);
        return (int)CM_INPUT_SIZE_SMALLER_THAN_KERNEL_ERROR;
    }

    int output_index = 0;
    for (int i = 0; i <= input_size - layer->kernel_size; i += layer->stride)
    {
        float max_value = input[i];
        for (int j = 1; j < layer->kernel_size; ++j)
        {
            if (input[i + j] > max_value)
            {
                max_value = input[i + j];
            }
        }
        output[output_index++] = max_value;
#if DEBUG_LOGGING
        printf("[maxpooling_layer_forward] Output[%d]: %f\n", output_index - 1, max_value);
#endif
    }
    return output_index; 
}


void maxpooling_layer_free(MaxPoolingLayer *layer)
{
    if (layer)
    {
        cm_safe_free(layer);
    }
}
