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

/**
 * @brief Multi-layer RNN wrapper
 *
 * Stacks multiple RNNCell layers, with optional bidirectional support
 * and inter-layer dropout.
 *
 * Input:  [seq_len, batch, input_size]  (or [batch, seq_len, input_size] if batch_first)
 * Output: (output_sequence, h_n)
 *   output_sequence: [seq_len, batch, num_directions * hidden_size]
 *   h_n: [num_layers * num_directions, batch, hidden_size]
 */
typedef struct RNN {
    Module base;
    int input_size;
    int hidden_size;
    int num_layers;
    bool bidirectional;
    bool batch_first;
    float dropout_p;
    bool use_bias;
    DType dtype;
    DeviceType device;
    RNNCell** cells;    // [num_layers * num_directions]
    int num_directions;
} RNN;

RNN* nn_rnn(int input_size, int hidden_size, int num_layers, bool bidirectional,
            bool batch_first, float dropout, bool use_bias, DType dtype, DeviceType device);
void rnn_forward(RNN* rnn, Tensor* input, Tensor* h_0,
                 Tensor** output, Tensor** h_n);

/**
 * @brief Multi-layer LSTM wrapper
 *
 * Input:  [seq_len, batch, input_size]  (or [batch, seq_len, input_size] if batch_first)
 * Output: (output_sequence, h_n, c_n)
 */
typedef struct LSTM {
    Module base;
    int input_size;
    int hidden_size;
    int num_layers;
    bool bidirectional;
    bool batch_first;
    float dropout_p;
    bool use_bias;
    DType dtype;
    DeviceType device;
    LSTMCell** cells;   // [num_layers * num_directions]
    int num_directions;
} LSTM;

LSTM* nn_lstm(int input_size, int hidden_size, int num_layers, bool bidirectional,
              bool batch_first, float dropout, bool use_bias, DType dtype, DeviceType device);
void lstm_forward(LSTM* lstm, Tensor* input, Tensor* h_0, Tensor* c_0,
                  Tensor** output, Tensor** h_n, Tensor** c_n);

/**
 * @brief Multi-layer GRU wrapper
 *
 * Input:  [seq_len, batch, input_size]  (or [batch, seq_len, input_size] if batch_first)
 * Output: (output_sequence, h_n)
 */
typedef struct GRU {
    Module base;
    int input_size;
    int hidden_size;
    int num_layers;
    bool bidirectional;
    bool batch_first;
    float dropout_p;
    bool use_bias;
    DType dtype;
    DeviceType device;
    GRUCell** cells;    // [num_layers * num_directions]
    int num_directions;
} GRU;

GRU* nn_gru(int input_size, int hidden_size, int num_layers, bool bidirectional,
            bool batch_first, float dropout, bool use_bias, DType dtype, DeviceType device);
void gru_forward(GRU* gru, Tensor* input, Tensor* h_0,
                 Tensor** output, Tensor** h_n);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_RNN_H
