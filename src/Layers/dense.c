#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../../include/Layers/dense.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/logging.h"

#ifndef DEBUG_LOGGING
#define DEBUG_LOGGING 0
#endif

/**
 * @brief Initializes a Dense Layer with random weights and biases.
 *
 * @param layer Pointer to the DenseLayer structure.
 * @param input_size Number of input neurons.
 * @param output_size Number of output neurons.
 * @return int Error code.
 */
int initialize_dense(DenseLayer *layer, int input_size, int output_size)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL");
        return CM_NULL_POINTER_ERROR;
    }

    if (input_size <= 0 || output_size <= 0)
    {
        LOG_ERROR("Invalid input size (%d) or output size (%d)", input_size, output_size);
        return CM_INVALID_PARAMETER_ERROR;
    }

    cm_safe_free((void **)&layer->weights);
    cm_safe_free((void **)&layer->biases);
    cm_safe_free((void **)&layer->adam_v_w);
    cm_safe_free((void **)&layer->adam_v_b);
    cm_safe_free((void **)&layer->adam_s_w);
    cm_safe_free((void **)&layer->adam_s_b);

    // if we don't see this Log message, we had a prolem zero-ing out memory
    LOG_DEBUG("Initialized DenseLayer with input size (%d) and output size (%d)", input_size, output_size);

    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->weights = (float *)cm_safe_malloc(input_size * output_size * sizeof(float), __FILE__, __LINE__);
    layer->biases = (float *)cm_safe_malloc(output_size * sizeof(float), __FILE__, __LINE__);

    if (layer->weights == (void *)CM_MEMORY_ALLOCATION_ERROR || layer->biases == (void *)CM_MEMORY_ALLOCATION_ERROR)
    {
        LOG_ERROR("Memory allocation failed");
        cm_safe_free((void **)&layer->weights);
        cm_safe_free((void **)&layer->biases);
        return CM_MEMORY_ALLOCATION_ERROR;
    }

    for (int i = 0; i < input_size * output_size; i++)
    {
        layer->weights[i] = ((float)rand() / RAND_MAX) - 0.5;
    }

    for (int i = 0; i < output_size; i++)
    {
        layer->biases[i] = ((float)rand() / RAND_MAX) - 0.5;
    }

    // Allocate and initialize Adam optimizer's moment vectors
    layer->adam_v_w = (float *)cm_safe_malloc(input_size * output_size * sizeof(float), __FILE__, __LINE__);
    layer->adam_v_b = (float *)cm_safe_malloc(output_size * sizeof(float), __FILE__, __LINE__);
    layer->adam_s_w = (float *)cm_safe_malloc(input_size * output_size * sizeof(float), __FILE__, __LINE__);
    layer->adam_s_b = (float *)cm_safe_malloc(output_size * sizeof(float), __FILE__, __LINE__);

    if (layer->adam_v_w == (void *)CM_MEMORY_ALLOCATION_ERROR || layer->adam_v_b == (void *)CM_MEMORY_ALLOCATION_ERROR ||
        layer->adam_s_w == (void *)CM_MEMORY_ALLOCATION_ERROR || layer->adam_s_b == (void *)CM_MEMORY_ALLOCATION_ERROR)
    {
        LOG_ERROR("Error: Memory allocation failed for Adam optimizer's moment vectors.\n");
        cm_safe_free((void **)&layer->adam_v_w);
        cm_safe_free((void **)&layer->adam_v_b);
        cm_safe_free((void **)&layer->adam_s_w);
        cm_safe_free((void **)&layer->adam_s_b);
        return CM_MEMORY_ALLOCATION_ERROR;
    }

    memset(layer->adam_v_w, 0, input_size * output_size * sizeof(float));
    memset(layer->adam_v_b, 0, output_size * sizeof(float));
    memset(layer->adam_s_w, 0, input_size * output_size * sizeof(float));
    memset(layer->adam_s_b, 0, output_size * sizeof(float));

    return CM_SUCCESS;
}

/**
 * @brief Performs the forward pass for the Dense Layer.
 *
 * @param layer Pointer to the DenseLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @return int Error code.
 */
int forward_dense(DenseLayer *layer, float *input, float *output)
{
    if (layer == NULL || input == NULL || output == NULL)
    {
        LOG_ERROR("Layer, input, or output is NULL");
        return CM_NULL_POINTER_ERROR;
    }

    LOG_DEBUG("forward_dense(layer->input_size: %d, layer->output_size: %d)", layer->input_size, layer->output_size);

    for (int i = 0; i < layer->output_size; i++)
    {
        output[i] = 0;
        for (int j = 0; j < layer->input_size; j++)
        {
            output[i] += input[j] * layer->weights[j + i * layer->input_size];
        }
        output[i] += layer->biases[i];
        LOG_DEBUG("Output[%d]: %f", i, output[i]);
    }

    return CM_SUCCESS;
}

/**
 * @brief Performs the backward pass for the Dense Layer.
 *
 * @param layer Pointer to the DenseLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param d_output Gradient of the output.
 * @param d_input Gradient of the input.
 * @param d_weights Gradient of the weights.
 * @param d_biases Gradient of the biases.
 * @return int Error code.
 */
int backward_dense(DenseLayer *layer, float *input, float *output, float *d_output, float *d_input, float *d_weights, float *d_biases)
{
    if (layer == NULL || input == NULL || output == NULL || d_output == NULL || d_input == NULL || d_weights == NULL || d_biases == NULL)
    {
        LOG_ERROR("One or more arguments are NULL");
        return CM_NULL_POINTER_ERROR;
    }

    for (int i = 0; i < layer->input_size; i++)
    {
        d_input[i] = 0;
        for (int j = 0; j < layer->output_size; j++)
        {
            // Breakpoint condition: Check for potential out-of-bounds access
            // For GDB:
            // break dense.c:108 if (i + j * layer->input_size) >= (layer->input_size * layer->output_size)
            d_input[i] += d_output[j] * layer->weights[i + j * layer->input_size];
        }
    }

    for (int i = 0; i < layer->output_size; i++)
    {
        d_biases[i] = d_output[i];
        for (int j = 0; j < layer->input_size; j++)
        {
            d_weights[j + i * layer->input_size] = input[j] * d_output[i];
        }
    }

    return CM_SUCCESS;
}

/**
 * @brief Updates the weights and biases of the Dense Layer.
 *
 * @param layer Pointer to the DenseLayer structure.
 * @param d_weights Gradient of the weights.
 * @param d_biases Gradient of the biases.
 * @param learning_rate Learning rate for the update.
 * @return int Error code.
 */
int update_dense(DenseLayer *layer, float *d_weights, float *d_biases, float learning_rate)
{
    if (layer == NULL || d_weights == NULL || d_biases == NULL)
    {
        LOG_ERROR("Layer or gradients are NULL");
        return CM_NULL_POINTER_ERROR;
    }

    for (int i = 0; i < layer->input_size * layer->output_size; i++)
    {
        layer->weights[i] -= learning_rate * d_weights[i];
    }

    for (int i = 0; i < layer->output_size; i++)
    {
        layer->biases[i] -= learning_rate * d_biases[i];
    }

    return CM_SUCCESS;
}

/**
 * @brief Frees the memory allocated for the Dense Layer.
 *
 * @param layer Pointer to the DenseLayer structure.
 * @return int Error code.
 */
int free_dense(DenseLayer *layer)
{
    if (layer == NULL)
    {
        return CM_NULL_POINTER_ERROR;
    }

    if (layer->weights != NULL)
    {
        cm_safe_free((void **)&layer->weights);
    }
    if (layer->biases != NULL)
    {
        cm_safe_free((void **)&layer->biases);
    }
    if (layer->adam_v_w != NULL)
    {
        cm_safe_free((void **)&layer->adam_v_w);
    }
    if (layer->adam_v_b != NULL)
    {
        cm_safe_free((void **)&layer->adam_v_b);
    }
    if (layer->adam_s_w != NULL)
    {
        cm_safe_free((void **)&layer->adam_s_w);
    }
    if (layer->adam_s_b != NULL)
    {
        cm_safe_free((void **)&layer->adam_s_b);
    }
    return CM_SUCCESS;
}
