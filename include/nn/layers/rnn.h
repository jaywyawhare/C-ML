/**
 * @file rnn.h
 * @brief Recurrent neural network layers: RNN, LSTM, GRU
 */

#ifndef CML_NN_LAYERS_RNN_H
#define CML_NN_LAYERS_RNN_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RNNCell {
    Module base;
    int input_size;
    int hidden_size;
    bool use_bias;
    Parameter* weight_ih; // [hidden_size, input_size]
    Parameter* weight_hh; // [hidden_size, hidden_size]
    Parameter* bias_ih;   // [hidden_size]
    Parameter* bias_hh;   // [hidden_size]
} RNNCell;

RNNCell* nn_rnn_cell(int input_size, int hidden_size, bool use_bias,
                     DType dtype, DeviceType device);
Tensor* rnn_cell_forward(RNNCell* cell, Tensor* input, Tensor* hidden);

typedef struct LSTMCell {
    Module base;
    int input_size;
    int hidden_size;
    bool use_bias;
    Parameter* weight_ih; // [4*hidden_size, input_size] (gates: i, f, g, o)
    Parameter* weight_hh; // [4*hidden_size, hidden_size]
    Parameter* bias_ih;   // [4*hidden_size]
    Parameter* bias_hh;   // [4*hidden_size]
} LSTMCell;

LSTMCell* nn_lstm_cell(int input_size, int hidden_size, bool use_bias,
                       DType dtype, DeviceType device);
void lstm_cell_forward(LSTMCell* cell, Tensor* input, Tensor* h_prev, Tensor* c_prev,
                       Tensor** h_out, Tensor** c_out);

typedef struct GRUCell {
    Module base;
    int input_size;
    int hidden_size;
    bool use_bias;
    Parameter* weight_ih; // [3*hidden_size, input_size] (gates: r, z, n)
    Parameter* weight_hh; // [3*hidden_size, hidden_size]
    Parameter* bias_ih;   // [3*hidden_size]
    Parameter* bias_hh;   // [3*hidden_size]
} GRUCell;

GRUCell* nn_gru_cell(int input_size, int hidden_size, bool use_bias,
                     DType dtype, DeviceType device);
Tensor* gru_cell_forward(GRUCell* cell, Tensor* input, Tensor* hidden);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_RNN_H
