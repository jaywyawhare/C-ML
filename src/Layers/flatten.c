#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../../include/Layers/flatten.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"

#ifndef DEBUG_LOGGING
#define DEBUG_LOGGING 0
#endif 

int initialize_flatten(FlattenLayer *layer, int input_size)
{
    if (layer == NULL)
    {
        fprintf(stderr, "[initialize_flatten] Error: Layer is NULL.\n");
        return CM_NULL_LAYER_ERROR; 
    }

    if (input_size <= 0)
    {
        fprintf(stderr, "[initialize_flatten] Error: Invalid input size (%d).\n", input_size);
        return CM_INVALID_LAYER_DIMENSIONS_ERROR; 
    }

    layer->input_size = input_size;
    layer->output_size = input_size;
    return CM_SUCCESS;
}

int forward_flatten(FlattenLayer *layer, float *input, float *output)
{
    if (layer == NULL || input == NULL || output == NULL)
    {
        fprintf(stderr, "[forward_flatten] Error: Layer, input, or output is NULL.\n");
        return CM_NULL_POINTER_ERROR;
    }

    if (layer->input_size <= 0 || layer->output_size <= 0)
    {
        fprintf(stderr, "[forward_flatten] Error: Invalid layer dimensions.\n");
        return CM_INVALID_LAYER_DIMENSIONS_ERROR; 
    }

    for (int i = 0; i < layer->input_size; i++)
    {
        output[i] = input[i];
#if DEBUG_LOGGING
        printf("[forward_flatten] Output[%d]: %f\n", i, output[i]);
#endif
    }

    return CM_SUCCESS;
}

int backward_flatten(FlattenLayer *layer, float *input, float *output, float *d_output, float *d_input)
{
    if (layer == NULL || input == NULL || output == NULL || d_output == NULL || d_input == NULL)
    {
        fprintf(stderr, "[backward_flatten] Error: One or more arguments are NULL.\n");
        return CM_NULL_POINTER_ERROR;
    }

    if (layer->input_size <= 0 || layer->output_size <= 0)
    {
        fprintf(stderr, "[backward_flatten] Error: Invalid layer dimensions.\n");
        return CM_INVALID_LAYER_DIMENSIONS_ERROR; 
    }

    for (int i = 0; i < layer->input_size; i++)
    {
        d_input[i] = d_output[i];
    }

    return CM_SUCCESS;
}

int free_flatten(FlattenLayer *layer)
{
    if (layer == NULL)
    {
        fprintf(stderr, "[free_flatten] Error: Layer is NULL.\n");
        return CM_NULL_POINTER_ERROR;
    }

    return CM_SUCCESS;
}
