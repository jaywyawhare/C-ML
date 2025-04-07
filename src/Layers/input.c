#include <stdlib.h>
#include <stdio.h>
#include "../../include/Layers/input.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/logging.h"

/**
 * @brief Initializes an Input Layer.
 *
 * @param layer Pointer to the InputLayer structure.
 * @param input_size Size of the input.
 * @return int Error code.
 */
int initialize_input(InputLayer *layer, int input_size)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    if (input_size <= 0)
    {
        LOG_ERROR("Invalid input size.");
        return CM_INVALID_INPUT_ERROR;
    }

    layer->input_size = input_size;
    return CM_SUCCESS;
}

/**
 * @brief Performs the forward pass for the Input Layer.
 *
 * @param layer Pointer to the InputLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @return int Error code.
 */
int forward_input(InputLayer *layer, float *input, float *output)
{
    if (layer == NULL || input == NULL || output == NULL)
    {
        LOG_ERROR("Layer, input, or output is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    for (int i = 0; i < layer->input_size; i++)
    {
        output[i] = input[i];
    }

    return CM_SUCCESS;
}

/**
 * @brief Performs the backward pass for the Input Layer.
 *
 * @param layer Pointer to the InputLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param d_output Gradient of the output.
 * @param d_input Gradient of the input.
 * @return int Error code.
 */
int backward_input(InputLayer *layer, float *input, float *output, float *d_output, float *d_input)
{
    if (layer == NULL || input == NULL || output == NULL || d_output == NULL || d_input == NULL)
    {
        LOG_ERROR("Layer, input, output, or gradients are NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    for (int i = 0; i < layer->input_size; i++)
    {
        d_input[i] = d_output[i];
    }

    return CM_SUCCESS;
}

/**
 * @brief Frees the memory allocated for the Input Layer.
 *
 * @param layer Pointer to the InputLayer structure.
 * @return int Error code.
 */
int free_input(InputLayer *layer)
{
    if (layer == NULL)
    {
        return CM_NULL_POINTER_ERROR;
    }
    return CM_SUCCESS;
}
