#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../../include/Layers/lstm.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Activations/sigmoid.h"
#include "../../include/Activations/tanh.h"

/**
 * @brief Initializes an LSTM Layer.
 *
 * @param layer Pointer to the LSTMLayer structure.
 * @param input_size Size of the input data.
 * @param hidden_size Size of the hidden state.
 * @return int Error code.
 */
int initialize_lstm(LSTMLayer *layer, int input_size, int hidden_size)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    if (input_size <= 0 || hidden_size <= 0)
    {
        LOG_ERROR("Invalid parameters for LSTM layer.");
        return CM_INVALID_PARAMETER_ERROR;
    }

    layer->input_size = input_size;
    layer->hidden_size = hidden_size;

    layer->weights_input = (float *)cm_safe_malloc(input_size * hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->weights_hidden = (float *)cm_safe_malloc(hidden_size * hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->biases = (float *)cm_safe_malloc(hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->cell_state = (float *)cm_safe_malloc(hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->hidden_state = (float *)cm_safe_malloc(hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->d_weights_input = (float *)cm_safe_malloc(input_size * hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->d_weights_hidden = (float *)cm_safe_malloc(hidden_size * hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->d_biases = (float *)cm_safe_malloc(hidden_size * sizeof(float), __FILE__, __LINE__);

    layer->forget_gate = (float *)cm_safe_malloc(hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->input_gate = (float *)cm_safe_malloc(hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->cell_gate = (float *)cm_safe_malloc(hidden_size * sizeof(float), __FILE__, __LINE__);
    layer->output_gate = (float *)cm_safe_malloc(hidden_size * sizeof(float), __FILE__, __LINE__);

    if (layer->weights_input == (void *)CM_MEMORY_ALLOCATION_ERROR || layer->weights_hidden == (void *)CM_MEMORY_ALLOCATION_ERROR || layer->biases == (void *)CM_MEMORY_ALLOCATION_ERROR || layer->cell_state == (void *)CM_MEMORY_ALLOCATION_ERROR || layer->hidden_state == (void *)CM_MEMORY_ALLOCATION_ERROR || layer->d_weights_input == (void *)CM_MEMORY_ALLOCATION_ERROR || layer->d_weights_hidden == (void *)CM_MEMORY_ALLOCATION_ERROR || layer->d_biases == (void *)CM_MEMORY_ALLOCATION_ERROR || layer->forget_gate == (void *)CM_MEMORY_ALLOCATION_ERROR || layer->input_gate == (void *)CM_MEMORY_ALLOCATION_ERROR || layer->cell_gate == (void *)CM_MEMORY_ALLOCATION_ERROR || layer->output_gate == (void *)CM_MEMORY_ALLOCATION_ERROR)
    {
        LOG_ERROR("Memory allocation failed for LSTM layer.");
        cm_safe_free((void **)&layer->weights_input);
        cm_safe_free((void **)&layer->weights_hidden);
        cm_safe_free((void **)&layer->biases);
        cm_safe_free((void **)&layer->cell_state);
        cm_safe_free((void **)&layer->hidden_state);
        cm_safe_free((void **)&layer->d_weights_input);
        cm_safe_free((void **)&layer->d_weights_hidden);
        cm_safe_free((void **)&layer->d_biases);
        cm_safe_free((void **)&layer->forget_gate);
        cm_safe_free((void **)&layer->input_gate);
        cm_safe_free((void **)&layer->cell_gate);
        cm_safe_free((void **)&layer->output_gate);
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

    memset(layer->cell_state, 0, hidden_size * sizeof(float));
    memset(layer->hidden_state, 0, hidden_size * sizeof(float));
    memset(layer->forget_gate, 0, hidden_size * sizeof(float));
    memset(layer->input_gate, 0, hidden_size * sizeof(float));
    memset(layer->cell_gate, 0, hidden_size * sizeof(float));
    memset(layer->output_gate, 0, hidden_size * sizeof(float));

    LOG_DEBUG("Initialized LSTMLayer with input size: %d and hidden size: %d", input_size, hidden_size);

    return CM_SUCCESS;
}

/**
 * @brief Performs the forward pass for the LSTM Layer.
 *
 * @param layer Pointer to the LSTMLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @return int Error code.
 */
int forward_lstm(LSTMLayer *layer, float *input, float *output)
{
    if (layer == NULL || input == NULL || output == NULL)
    {
        LOG_ERROR("Layer, input, or output is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    LOG_DEBUG("Performing forward pass for LSTMLayer.");

    for (int i = 0; i < layer->hidden_size; i++)
    {
        layer->forget_gate[i] = sigmoid(input[i] * layer->weights_input[i] + layer->hidden_state[i] * layer->weights_hidden[i] + layer->biases[i]);
        layer->input_gate[i] = sigmoid(input[i] * layer->weights_input[i + layer->hidden_size] + layer->hidden_state[i] * layer->weights_hidden[i + layer->hidden_size] + layer->biases[i + layer->hidden_size]);
        layer->cell_gate[i] = tanh(input[i] * layer->weights_input[i + 2 * layer->hidden_size] + layer->hidden_state[i] * layer->weights_hidden[i + 2 * layer->hidden_size] + layer->biases[i + 2 * layer->hidden_size]);
        layer->output_gate[i] = sigmoid(input[i] * layer->weights_input[i + 3 * layer->hidden_size] + layer->hidden_state[i] * layer->weights_hidden[i + 3 * layer->hidden_size] + layer->biases[i + 3 * layer->hidden_size]);

        layer->cell_state[i] = layer->forget_gate[i] * layer->cell_state[i] + layer->input_gate[i] * layer->cell_gate[i];
        output[i] = layer->hidden_state[i] = layer->output_gate[i] * tanh(layer->cell_state[i]);
    }

    return CM_SUCCESS;
}

/**
 * @brief Performs the backward pass for the LSTM Layer.
 *
 * @param layer Pointer to the LSTMLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param d_output Gradient of the output.
 * @param d_input Gradient of the input.
 * @return int Error code.
 */
int backward_lstm(LSTMLayer *layer, float *input, float *output, float *d_output, float *d_input)
{
    if (layer == NULL || input == NULL || output == NULL || d_output == NULL || d_input == NULL)
    {
        LOG_ERROR("One or more arguments are NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    LOG_DEBUG("Performing backward pass for LSTMLayer.");

    memset(d_input, 0, layer->input_size * sizeof(float));

    for (int i = 0; i < layer->hidden_size; i++)
    {
        float d_output_gate = d_output[i] * tanh(layer->cell_state[i]) * layer->output_gate[i] * (1 - layer->output_gate[i]);
        float d_cell_state = d_output[i] * layer->output_gate[i] * (1 - tanh(layer->cell_state[i]) * tanh(layer->cell_state[i]));
        float d_forget_gate = d_cell_state * layer->cell_state[i] * layer->forget_gate[i] * (1 - layer->forget_gate[i]);
        float d_input_gate = d_cell_state * layer->cell_gate[i] * layer->input_gate[i] * (1 - layer->input_gate[i]);
        float d_cell_gate = d_cell_state * layer->input_gate[i] * (1 - layer->cell_gate[i] * layer->cell_gate[i]);

        d_input[i] = d_output_gate * layer->weights_input[i + 3 * layer->hidden_size] + d_forget_gate * layer->weights_input[i] + d_input_gate * layer->weights_input[i + layer->hidden_size] + d_cell_gate * layer->weights_input[i + 2 * layer->hidden_size];
    }

    return CM_SUCCESS;
}

/**
 * @brief Resets the state of the LSTM Layer.
 *
 * @param layer Pointer to the LSTMLayer structure.
 * @return int Error code.
 */
int reset_state_lstm(LSTMLayer *layer)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    memset(layer->cell_state, 0, layer->hidden_size * sizeof(float));
    memset(layer->hidden_state, 0, layer->hidden_size * sizeof(float));

    return CM_SUCCESS;
}

/**
 * @brief Updates the weights and biases of the LSTM Layer.
 *
 * @param layer Pointer to the LSTMLayer structure.
 * @param learning_rate Learning rate for the update.
 * @return int Error code.
 */
int update_lstm(LSTMLayer *layer, float learning_rate)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
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

    return CM_SUCCESS;
}

/**
 * @brief Frees the memory allocated for the LSTM Layer.
 *
 * @param layer Pointer to the LSTMLayer structure.
 * @return int Error code.
 */
int free_lstm(LSTMLayer *layer)
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
    if (layer->cell_state != NULL)
    {
        cm_safe_free((void **)&layer->cell_state);
    }
    if (layer->hidden_state != NULL)
    {
        cm_safe_free((void **)&layer->hidden_state);
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
    if (layer->forget_gate != NULL)
    {
        cm_safe_free((void **)&layer->forget_gate);
    }
    if (layer->input_gate != NULL)
    {
        cm_safe_free((void **)&layer->input_gate);
    }
    if (layer->cell_gate != NULL)
    {
        cm_safe_free((void **)&layer->cell_gate);
    }
    if (layer->output_gate != NULL)
    {
        cm_safe_free((void **)&layer->output_gate);
    }

    return CM_SUCCESS;
}