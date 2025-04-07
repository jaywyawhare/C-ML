#include <stdlib.h>
#include <stdio.h>
#include "../../include/Layers/reshape.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Initializes a Reshape Layer.
 *
 * @param layer Pointer to the ReshapeLayer structure.
 * @param input_size Size of the input data.
 * @param output_size Size of the output data.
 * @return int Error code.
 */
int initialize_reshape(ReshapeLayer *layer, int input_size, int output_size)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    if (input_size <= 0 || output_size <= 0 || input_size != output_size)
    {
        LOG_ERROR("Invalid parameters for Reshape layer.");
        return CM_INVALID_PARAMETER_ERROR;
    }

    layer->input_size = input_size;
    layer->output_size = output_size;
    LOG_DEBUG("Initialized ReshapeLayer with input size: %d and output size: %d", input_size, output_size);

    return CM_SUCCESS;
}

/**
 * @brief Performs the forward pass for the Reshape Layer.
 *
 * @param layer Pointer to the ReshapeLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @return int Error code.
 */
int forward_reshape(ReshapeLayer *layer, float *input, float *output)
{
    if (layer == NULL || input == NULL || output == NULL)
    {
        LOG_ERROR("Layer, input, or output is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    LOG_DEBUG("Performing forward pass for ReshapeLayer.");

    for (int i = 0; i < layer->input_size; i++)
    {
        output[i] = input[i];
        LOG_DEBUG("Output[%d]: %f", i, output[i]);
    }

    return CM_SUCCESS;
}

/**
 * @brief Performs the backward pass for the Reshape Layer.
 *
 * @param layer Pointer to the ReshapeLayer structure.
 * @param d_output Gradient of the output.
 * @param d_input Gradient of the input.
 * @return int Error code.
 */
int backward_reshape(ReshapeLayer *layer, float *d_output, float *d_input)
{
    if (layer == NULL || d_output == NULL || d_input == NULL)
    {
        LOG_ERROR("Layer or gradients are NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    LOG_DEBUG("Performing backward pass for ReshapeLayer.");

    for (int i = 0; i < layer->input_size; i++)
    {
        d_input[i] = d_output[i];
        LOG_DEBUG("d_input[%d]: %f", i, d_input[i]);
    }

    return CM_SUCCESS;
}

/**
 * @brief Frees the memory allocated for the Reshape Layer.
 *
 * @param layer Pointer to the ReshapeLayer structure.
 * @return int Error code.
 */
int free_reshape(ReshapeLayer *layer)
{
    if (layer == NULL)
    {
        return CM_NULL_POINTER_ERROR;
    }
    return CM_SUCCESS;
}