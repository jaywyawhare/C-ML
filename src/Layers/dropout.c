#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../../include/Layers/dropout.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/memory_management.h"



static int is_seed_initialized = 0;

/**
 * @brief Initializes a Dropout Layer with a given dropout rate.
 *
 * @param layer Pointer to the DropoutLayer structure.
 * @param dropout_rate Dropout rate (0.0 to 1.0).
 * @return int Error code (0 for success, non-zero for error).
 */
int initialize_dropout(DropoutLayer *layer, float dropout_rate)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    if (dropout_rate < 0.0f || dropout_rate > 1.0f)
    {
        LOG_ERROR("Invalid dropout rate. Must be between 0.0 and 1.0.");
        return CM_INVALID_PARAMETER_ERROR;
    }

    layer->dropout_rate = dropout_rate;
    if (!is_seed_initialized)
    {
        srand((unsigned int)time(NULL));
        is_seed_initialized = 1;
    }
    return CM_SUCCESS;
}

/**
 * @brief Performs the forward pass for the Dropout Layer.
 *
 * @param layer Pointer to the DropoutLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param size Size of the input/output arrays.
 * @return int Error code (0 for success, non-zero for error).
 */
int forward_dropout(DropoutLayer *layer, float *input, float *output, int size)
{
    if (layer == NULL || input == NULL || output == NULL)
    {
        LOG_ERROR("Layer, input, or output is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    for (int i = 0; i < size; i++)
    {
        if ((float)rand() / RAND_MAX < layer->dropout_rate)
        {
            output[i] = 0;
        }
        else
        {
            output[i] = input[i] / (1 - layer->dropout_rate);
        }
        LOG_DEBUG("Output[%d]: %f", i, output[i]);
    }
    return CM_SUCCESS;
}

/**
 * @brief Performs the backward pass for the Dropout Layer.
 *
 * @param layer Pointer to the DropoutLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param d_output Gradient of the output.
 * @param d_input Gradient of the input.
 * @param size Size of the input/output arrays.
 * @return int Error code (0 for success, non-zero for error).
 */
int backward_dropout(DropoutLayer *layer, float *input, float *output, float *d_output, float *d_input, int size)
{
    if (layer == NULL || input == NULL || output == NULL || d_output == NULL || d_input == NULL)
    {
        LOG_ERROR("One or more arguments are NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    for (int i = 0; i < size; i++)
    {
        if (output[i] == 0.0f)
        {
            d_input[i] = 0.0f;
        }
        else
        {
            d_input[i] = d_output[i] / (1 - layer->dropout_rate);
        }
    }
    return CM_SUCCESS;
}
