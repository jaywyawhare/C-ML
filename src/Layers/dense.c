#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../../include/Layers/dense.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"

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
        fprintf(stderr, "[initializeDense] Error: Layer is NULL.\n");
        return CM_NULL_POINTER_ERROR;
    }

    if (input_size <= 0 || output_size <= 0)
    {
        fprintf(stderr, "[initializeDense] Error: Invalid input size (%d) or output size (%d).\n", input_size, output_size);
        return CM_INVALID_PARAMETER_ERROR;
    }

    free_dense(layer);

    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->weights = (float *)cm_safe_malloc(input_size * output_size * sizeof(float), __FILE__, __LINE__);
    layer->biases = (float *)cm_safe_malloc(output_size * sizeof(float), __FILE__, __LINE__);

    if (layer->weights == (void *)CM_MEMORY_ALLOCATION_ERROR || layer->biases == (void *)CM_MEMORY_ALLOCATION_ERROR)
    {
        fprintf(stderr, "[initializeDense] Error: Memory allocation failed.\n");
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
        fprintf(stderr, "[forwardDense] Error: Layer, input, or output is NULL.\n");
        return CM_NULL_POINTER_ERROR;
    }

    for (int i = 0; i < layer->output_size; i++)
    {
        output[i] = 0;
        for (int j = 0; j < layer->input_size; j++)
        {
            // Breakpoint condition: Check for potential out-of-bounds access
            // For GDB:
            // break dense.c:74 if (j + i * layer->input_size) >= (layer->input_size * layer->output_size)
            output[i] += input[j] * layer->weights[j + i * layer->input_size];
        }
        output[i] += layer->biases[i];
#if DEBUG_LOGGING
        printf("[forwardDense] Output[%d]: %f\n", i, output[i]);
#endif
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
        fprintf(stderr, "[backwardDense] Error: One or more arguments are NULL.\n");
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
        fprintf(stderr, "[updateDense] Error: Layer or gradients are NULL.\n");
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
    return CM_SUCCESS;
}