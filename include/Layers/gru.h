#ifndef GRU_H
#define GRU_H

#include "../../include/Core/memory_management.h"
#include "../../include/Activations/sigmoid.h"
#include "../../include/Activations/tanh.h"

/**
 * @brief Structure representing a GRU Layer.
 *
 * @param input_size Size of the input data.
 * @param hidden_size Size of the hidden state.
 * @param weights_input Pointer to the input weights array.
 * @param weights_hidden Pointer to the hidden weights array.
 * @param biases Pointer to the biases array.
 * @param hidden_state Pointer to the hidden state array.
 * @param reset_gate Pointer to the reset gate array.
 * @param update_gate Pointer to the update gate array.
 * @param new_gate Pointer to the new gate array.
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
    float *hidden_state;
    float *reset_gate;
    float *update_gate;
    float *new_gate;
    float *d_weights_input;
    float *d_weights_hidden;
    float *d_biases;
} GRULayer;

int initialize_gru(GRULayer *layer, int input_size, int hidden_size);
int forward_gru(GRULayer *layer, float *input, float *output);
int backward_gru(GRULayer *layer, float *input, float *output, float *d_output, float *d_input);
int update_gru(GRULayer *layer, float learning_rate);
int reset_state_gru(GRULayer *layer);
int free_gru(GRULayer *layer);

#endif
