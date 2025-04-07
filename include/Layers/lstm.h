#ifndef LSTM_H
#define LSTM_H

#include "../../include/Core/memory_management.h"
#include "../../include/Activations/sigmoid.h"
#include "../../include/Activations/tanh.h"

/**
 * @brief Structure representing an LSTM Layer.
 *
 * @param input_size Size of the input data.
 * @param hidden_size Size of the hidden state.
 * @param weights_input Pointer to the input weights array.
 * @param weights_hidden Pointer to the hidden weights array.
 * @param biases Pointer to the biases array.
 * @param cell_state Pointer to the cell state array.
 * @param hidden_state Pointer to the hidden state array.
 * @param forget_gate Pointer to the forget gate array.
 * @param input_gate Pointer to the input gate array.
 * @param cell_gate Pointer to the cell gate array.
 * @param output_gate Pointer to the output gate array.
 * @param d_weights_input Pointer to the input weights gradients array.
 * @param d_weights_hidden Pointer to the hidden weights gradients array.
 * @param d_biases Pointer to the biases gradients array.
 */
typedef struct
{
    int input_size;
    int hidden_size;
    float *weights_input;
    float *weights_hidden;
    float *biases;
    float *cell_state;
    float *hidden_state;
    float *forget_gate;
    float *input_gate;
    float *cell_gate;
    float *output_gate;
    float *d_weights_input;
    float *d_weights_hidden;
    float *d_biases;
} LSTMLayer;

int initialize_lstm(LSTMLayer *layer, int input_size, int hidden_size);
int forward_lstm(LSTMLayer *layer, float *input, float *output);
int backward_lstm(LSTMLayer *layer, float *input, float *output, float *d_output, float *d_input);
int update_lstm(LSTMLayer *layer, float learning_rate);
int reset_state_lstm(LSTMLayer *layer);
int free_lstm(LSTMLayer *layer);

#endif
