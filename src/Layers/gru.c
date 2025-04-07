#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../../include/Layers/gru.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Activations/sigmoid.h"
#include "../../include/Activations/tanh.h"

/**
 * @brief Initializes a GRU Layer.
 *
 * @param layer Pointer to the GRULayer structure.
 * @param input_size Size of the input data.
 * @param hidden_size Size of the hidden state.
 * @return int Error code.
 */
int initialize_gru(GRULayer *layer, int input_size, int hidden_size)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    if (input_size <= 0 || hidden_size <= 0)
    {
        LOG_ERROR("Invalid parameters for GRU layer.");
        return CM_INVALID_PARAMETER_ERROR;
    }

    layer->input_size = input_size;
    layer->hidden_size = hidden_size;

    layer->weights_input = (float *)cm_safe_malloc(input_size * hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->weights_hidden = (float *)cm_safe_malloc(hidden_size * hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->biases = (float *)cm_safe_malloc(hidden_size * sizeof(float), __FILE__, __LINE__);

    layer->hidden_state = (float *)cm_safe_malloc(hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->reset_gate = (float *)cm_safe_malloc(hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->update_gate = (float *)cm_safe_malloc(hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->new_gate = (float *)cm_safe_malloc(hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->d_weights_input = (float *)cm_safe_malloc(input_size * hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->d_weights_hidden = (float *)cm_safe_malloc(hidden_size * hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->d_biases = (float *)cm_safe_malloc(hidden_size * sizeof(float), __FILE__, __LINE__);

    if (layer->weights_input == (void *)CM_MEMORY_ALLOCATION_ERROR ||
        layer->weights_hidden == (void *)CM_MEMORY_ALLOCATION_ERROR ||
        layer->biases == (void *)CM_MEMORY_ALLOCATION_ERROR ||
        layer->hidden_state == (void *)CM_MEMORY_ALLOCATION_ERROR ||
        layer->reset_gate == (void *)CM_MEMORY_ALLOCATION_ERROR ||
        layer->update_gate == (void *)CM_MEMORY_ALLOCATION_ERROR ||
        layer->new_gate == (void *)CM_MEMORY_ALLOCATION_ERROR ||
        layer->d_weights_input == (void *)CM_MEMORY_ALLOCATION_ERROR ||
        layer->d_weights_hidden == (void *)CM_MEMORY_ALLOCATION_ERROR ||
        layer->d_biases == (void *)CM_MEMORY_ALLOCATION_ERROR)
    {
        LOG_ERROR("Memory allocation failed for GRU layer.");
        cm_safe_free((void **)&layer->weights_input);
        cm_safe_free((void **)&layer->weights_hidden);
        cm_safe_free((void **)&layer->biases);
        cm_safe_free((void **)&layer->hidden_state);
        cm_safe_free((void **)&layer->reset_gate);
        cm_safe_free((void **)&layer->update_gate);
        cm_safe_free((void **)&layer->new_gate);
        cm_safe_free((void **)&layer->d_weights_input);
        cm_safe_free((void **)&layer->d_weights_hidden);
        cm_safe_free((void **)&layer->d_biases);
        return CM_MEMORY_ALLOCATION_ERROR;
    }

    for (int i = 0; i < input_size * hidden_size; i++)
    {
        layer->weights_input[i] = ((float)rand() / RAND_MAX) - 0.5;
    }

    for (int i = 0; i < hidden_size * hidden_size; i++)
    {
        layer->weights_hidden[i] = ((float)rand() / RAND_MAX) - 0.5;
    }

    for (int i = 0; i < hidden_size; i++)
    {
        layer->biases[i] = ((float)rand() / RAND_MAX) - 0.5;
    }

    memset(layer->hidden_state, 0, hidden_size * sizeof(float));
    memset(layer->reset_gate, 0, hidden_size * sizeof(float));
    memset(layer->update_gate, 0, hidden_size * sizeof(float));
    memset(layer->new_gate, 0, hidden_size * sizeof(float));
    memset(layer->d_weights_input, 0, input_size * hidden_size * sizeof(float));
    memset(layer->d_weights_hidden, 0, hidden_size * hidden_size * sizeof(float));
    memset(layer->d_biases, 0, hidden_size * sizeof(float));

    LOG_DEBUG("Initialized GRULayer with input size: %d and hidden size: %d", input_size, hidden_size);

    return CM_SUCCESS;
}

/**
 * @brief Resets the state of the GRU Layer.
 *
 * @param layer Pointer to the GRULayer structure.
 * @return int Error code.
 */
int reset_state_gru(GRULayer *layer)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    memset(layer->hidden_state, 0, layer->hidden_size * sizeof(float));
    memset(layer->reset_gate, 0, layer->hidden_size * sizeof(float));
    memset(layer->update_gate, 0, layer->hidden_size * sizeof(float));
    memset(layer->new_gate, 0, layer->hidden_size * sizeof(float));

    return CM_SUCCESS;
}

/**
 * @brief Performs the forward pass for the GRU Layer.
 *
 * @param layer Pointer to the GRULayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @return int Error code.
 */
int forward_gru(GRULayer *layer, float *input, float *output)
{
    if (layer == NULL || input == NULL || output == NULL)
    {
        LOG_ERROR("Layer, input, or output is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    LOG_DEBUG("Performing forward pass for GRULayer.");

    for (int i = 0; i < layer->hidden_size; i++)
    {
        layer->reset_gate[i] = sigmoid(input[i] * layer->weights_input[i] + layer->hidden_state[i] * layer->weights_hidden[i] + layer->biases[i]);
        layer->update_gate[i] = sigmoid(input[i] * layer->weights_input[i + layer->hidden_size] + layer->hidden_state[i] * layer->weights_hidden[i + layer->hidden_size] + layer->biases[i + layer->hidden_size]);
        layer->new_gate[i] = tanh(input[i] * layer->weights_input[i + 2 * layer->hidden_size] + (layer->reset_gate[i] * layer->hidden_state[i]) * layer->weights_hidden[i + 2 * layer->hidden_size] + layer->biases[i + 2 * layer->hidden_size]);
        output[i] = layer->hidden_state[i] = (1 - layer->update_gate[i]) * layer->hidden_state[i] + layer->update_gate[i] * layer->new_gate[i];
    }

    return CM_SUCCESS;
}

/**
 * @brief Performs the backward pass for the GRU Layer.
 *
 * @param layer Pointer to the GRULayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param d_output Gradient of the output.
 * @param d_input Gradient of the input.
 * @return int Error code.
 */
int backward_gru(GRULayer *layer, float *input, float *output, float *d_output, float *d_input)
{
    if (layer == NULL || input == NULL || output == NULL || d_output == NULL || d_input == NULL)
    {
        LOG_ERROR("Layer or gradients are NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    LOG_DEBUG("Performing backward pass for GRULayer.");

    memset(d_input, 0, layer->input_size * sizeof(float));

    for (int i = 0; i < layer->hidden_size; i++)
    {
        float d_new_gate = d_output[i] * layer->update_gate[i] * (1 - layer->new_gate[i] * layer->new_gate[i]);
        float d_update_gate = d_output[i] * (layer->new_gate[i] - layer->hidden_state[i]) * layer->update_gate[i] * (1 - layer->update_gate[i]);
        float d_reset_gate = d_new_gate * layer->hidden_state[i] * layer->reset_gate[i] * (1 - layer->reset_gate[i]);

        d_input[i] = d_new_gate * layer->weights_input[i + 2 * layer->hidden_size] + d_update_gate * layer->weights_input[i + layer->hidden_size] + d_reset_gate * layer->weights_input[i];
    }

    return CM_SUCCESS;
}

/**
 * @brief Updates the weights and biases of the GRU Layer.
 *
 * @param layer Pointer to the GRULayer structure.
 * @param learning_rate Learning rate for the update.
 * @return int Error code.
 */
int update_gru(GRULayer *layer, float learning_rate)
{
    if (layer == NULL)
    {
        return CM_NULL_POINTER_ERROR;
    }

    int input_weight_size = layer->input_size * layer->hidden_size;
    int hidden_weight_size = layer->hidden_size * layer->hidden_size;

    for (int i = 0; i < input_weight_size; i++)
    {
        layer->weights_input[i] -= learning_rate * layer->d_weights_input[i];
    }

    for (int i = 0; i < hidden_weight_size; i++)
    {
        layer->weights_hidden[i] -= learning_rate * layer->d_weights_hidden[i];
    }

    for (int i = 0; i < layer->hidden_size; i++)
    {
        layer->biases[i] -= learning_rate * layer->d_biases[i];
    }

    memset(layer->d_weights_input, 0, layer->input_size * layer->hidden_size * sizeof(float));
    memset(layer->d_weights_hidden, 0, layer->hidden_size * layer->hidden_size * sizeof(float));
    memset(layer->d_biases, 0, layer->hidden_size * sizeof(float));

    LOG_DEBUG("Updated GRULayer weights and biases with learning rate: %f", learning_rate);

    return CM_SUCCESS;
}

/**
 * @brief Frees the memory allocated for the GRU Layer.
 *
 * @param layer Pointer to the GRULayer structure.
 * @return int Error code.
 */
int free_gru(GRULayer *layer)
{
    if (layer == NULL)
    {
        return CM_NULL_POINTER_ERROR;
    }

    if (layer->weights_input != NULL)
    {
        cm_safe_free((void **)&layer->weights_input);
    }
    if (layer->weights_hidden != NULL)
    {
        cm_safe_free((void **)&layer->weights_hidden);
    }
    if (layer->biases != NULL)
    {
        cm_safe_free((void **)&layer->biases);
    }
    if (layer->hidden_state != NULL)
    {
        cm_safe_free((void **)&layer->hidden_state);
    }
    if (layer->reset_gate != NULL)
    {
        cm_safe_free((void **)&layer->reset_gate);
    }
    if (layer->update_gate != NULL)
    {
        cm_safe_free((void **)&layer->update_gate);
    }
    if (layer->new_gate != NULL)
    {
        cm_safe_free((void **)&layer->new_gate);
    }
    if (layer->d_weights_input != NULL)
    {
        cm_safe_free((void **)&layer->d_weights_input);
    }
    if (layer->d_weights_hidden != NULL)
    {
        cm_safe_free((void **)&layer->d_weights_hidden);
    }
    if (layer->d_biases != NULL)
    {
        cm_safe_free((void **)&layer->d_biases);
    }

    return CM_SUCCESS;
}