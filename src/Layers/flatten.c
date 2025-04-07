#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../../include/Layers/flatten.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/logging.h"

#ifndef DEBUG_LOGGING
#define DEBUG_LOGGING 0
#endif 

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
    if (layer == NULL || input == NULL || output == NULL)
    {
        LOG_ERROR("Layer, input, or output is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    if (layer->input_size <= 0 || layer->output_size <= 0)
    {
        LOG_ERROR("Invalid layer dimensions.");
        return CM_INVALID_LAYER_DIMENSIONS_ERROR; 
    }

    for (int i = 0; i < layer->input_size; i++)
    {
        output[i] = input[i];
        LOG_DEBUG("Output[%d]: %f", i, output[i]);
    }

    return CM_SUCCESS;
}

int backward_flatten(FlattenLayer *layer, float *input, float *output, float *d_output, float *d_input)
{
    if (layer == NULL || input == NULL || output == NULL || d_output == NULL || d_input == NULL)
    {
        LOG_ERROR("One or more arguments are NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    if (layer->input_size <= 0 || layer->output_size <= 0)
    {
        LOG_ERROR("Invalid layer dimensions.");
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
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    return CM_SUCCESS;
}
