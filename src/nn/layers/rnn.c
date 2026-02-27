/**
 * @file rnn.c
 * @brief Recurrent neural network layers: RNN, LSTM, GRU
 *
 * Implements eager computation for recurrent cells using
 * matrix-vector multiplications and element-wise gate operations.
 */

#include "nn/layers/rnn.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/**
 * @brief Batched matrix-vector multiply with optional bias addition.
 *
 * Computes: out[b][r] = sum_c(weight[r][c] * input[b][c]) + bias[r]
 *
 * @param out    Output buffer [batch, rows]
 * @param weight Weight matrix  [rows, cols]  (row-major)
 * @param input  Input matrix   [batch, cols] (row-major)
 * @param bias   Bias vector    [rows] or NULL
 * @param rows   Number of output rows (hidden dimension)
 * @param cols   Number of input columns (input dimension)
 * @param batch  Batch size
 */
static void matmul_add(float* out, const float* weight, const float* input,
                       const float* bias, int rows, int cols, int batch) {
    for (int b = 0; b < batch; b++) {
        for (int r = 0; r < rows; r++) {
            float sum = bias ? bias[r] : 0.0f;
            for (int c = 0; c < cols; c++) {
                sum += weight[r * cols + c] * input[b * cols + c];
            }
            out[b * rows + r] = sum;
        }
    }
}

static float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
static float tanh_f(float x) { return tanhf(x); }

static Tensor* rnn_cell_module_forward(Module* module, Tensor* input) {
    /* The Module interface does not carry the hidden state, so users
       should call rnn_cell_forward() directly. */
    (void)module;
    (void)input;
    return NULL;
}

static void rnn_cell_free(Module* module) { free(module); }

Tensor* rnn_cell_forward(RNNCell* cell, Tensor* input, Tensor* hidden) {
    if (!cell || !input) return NULL;

    tensor_ensure_executed(input);
    tensor_ensure_executed(cell->weight_ih->tensor);
    tensor_ensure_executed(cell->weight_hh->tensor);

    int batch = input->shape[0];
    int hs    = cell->hidden_size;
    int is    = cell->input_size;

    /* Initialise hidden to zeros when NULL */
    bool free_hidden = false;
    if (!hidden) {
        int h_shape[] = {batch, hs};
        TensorConfig cfg = {.dtype = input->dtype, .device = input->device,
                            .has_dtype = true, .has_device = true};
        hidden = tensor_zeros(h_shape, 2, &cfg);
        free_hidden = true;
    }
    tensor_ensure_executed(hidden);

    float* x_data = (float*)tensor_data_ptr(input);
    float* h_data = (float*)tensor_data_ptr(hidden);
    float* wih    = (float*)tensor_data_ptr(cell->weight_ih->tensor);
    float* whh    = (float*)tensor_data_ptr(cell->weight_hh->tensor);

    float* bih = NULL;
    float* bhh = NULL;
    if (cell->bias_ih) {
        tensor_ensure_executed(cell->bias_ih->tensor);
        bih = (float*)tensor_data_ptr(cell->bias_ih->tensor);
    }
    if (cell->bias_hh) {
        tensor_ensure_executed(cell->bias_hh->tensor);
        bhh = (float*)tensor_data_ptr(cell->bias_hh->tensor);
    }

    /* Temporary buffers for the two matrix-vector products */
    float* ih_out = calloc((size_t)(batch * hs), sizeof(float));
    float* hh_out = calloc((size_t)(batch * hs), sizeof(float));
    if (!ih_out || !hh_out) {
        free(ih_out);
        free(hh_out);
        if (free_hidden) tensor_free(hidden);
        return NULL;
    }

    matmul_add(ih_out, wih, x_data, bih, hs, is, batch);
    matmul_add(hh_out, whh, h_data, bhh, hs, hs, batch);

    /* Allocate output tensor */
    int h_shape[] = {batch, hs};
    TensorConfig cfg = {.dtype = input->dtype, .device = input->device,
                        .has_dtype = true, .has_device = true};
    Tensor* h_new = tensor_empty(h_shape, 2, &cfg);
    tensor_ensure_executed(h_new);
    float* h_new_data = (float*)tensor_data_ptr(h_new);

    for (int i = 0; i < batch * hs; i++) {
        h_new_data[i] = tanh_f(ih_out[i] + hh_out[i]);
    }

    free(ih_out);
    free(hh_out);
    if (free_hidden) tensor_free(hidden);

    return h_new;
}

RNNCell* nn_rnn_cell(int input_size, int hidden_size, bool use_bias,
                     DType dtype, DeviceType device) {
    RNNCell* cell = malloc(sizeof(RNNCell));
    if (!cell) return NULL;

    if (module_init((Module*)cell, "RNNCell",
                    rnn_cell_module_forward, rnn_cell_free) != 0) {
        free(cell);
        return NULL;
    }

    cell->input_size  = input_size;
    cell->hidden_size = hidden_size;
    cell->use_bias    = use_bias;

    TensorConfig cfg = {.dtype = dtype, .device = device,
                        .has_dtype = true, .has_device = true};
    float scale = 1.0f / sqrtf((float)hidden_size);

    /* weight_ih [hidden_size, input_size] */
    int wih_shape[] = {hidden_size, input_size};
    Tensor* wih = tensor_empty(wih_shape, 2, &cfg);
    tensor_ensure_executed(wih);
    float* wih_data = (float*)tensor_data_ptr(wih);
    for (size_t i = 0; i < wih->numel; i++)
        wih_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    module_add_parameter((Module*)cell, wih, "weight_ih", true);
    cell->weight_ih = module_get_parameter((Module*)cell, "weight_ih");

    /* weight_hh [hidden_size, hidden_size] */
    int whh_shape[] = {hidden_size, hidden_size};
    Tensor* whh = tensor_empty(whh_shape, 2, &cfg);
    tensor_ensure_executed(whh);
    float* whh_data = (float*)tensor_data_ptr(whh);
    for (size_t i = 0; i < whh->numel; i++)
        whh_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    module_add_parameter((Module*)cell, whh, "weight_hh", true);
    cell->weight_hh = module_get_parameter((Module*)cell, "weight_hh");

    /* Biases */
    if (use_bias) {
        int b_shape[] = {hidden_size};

        Tensor* bih = tensor_zeros(b_shape, 1, &cfg);
        module_add_parameter((Module*)cell, bih, "bias_ih", true);
        cell->bias_ih = module_get_parameter((Module*)cell, "bias_ih");

        Tensor* bhh = tensor_zeros(b_shape, 1, &cfg);
        module_add_parameter((Module*)cell, bhh, "bias_hh", true);
        cell->bias_hh = module_get_parameter((Module*)cell, "bias_hh");
    } else {
        cell->bias_ih = NULL;
        cell->bias_hh = NULL;
    }

    LOG_DEBUG("Created RNNCell: input_size=%d, hidden_size=%d, bias=%d",
              input_size, hidden_size, use_bias);

    return cell;
}

static Tensor* lstm_cell_module_forward(Module* module, Tensor* input) {
    (void)module;
    (void)input;
    return NULL;
}

static void lstm_cell_free(Module* module) { free(module); }

void lstm_cell_forward(LSTMCell* cell, Tensor* input,
                       Tensor* h_prev, Tensor* c_prev,
                       Tensor** h_out, Tensor** c_out) {
    if (!cell || !input || !h_out || !c_out) return;

    tensor_ensure_executed(input);
    tensor_ensure_executed(cell->weight_ih->tensor);
    tensor_ensure_executed(cell->weight_hh->tensor);

    int batch = input->shape[0];
    int hs    = cell->hidden_size;
    int is    = cell->input_size;
    int gs    = 4 * hs; /* gate size */

    /* Initialise states to zeros when NULL */
    bool free_h = false, free_c = false;
    if (!h_prev) {
        int s[] = {batch, hs};
        TensorConfig cfg = {.dtype = input->dtype, .device = input->device,
                            .has_dtype = true, .has_device = true};
        h_prev = tensor_zeros(s, 2, &cfg);
        free_h = true;
    }
    if (!c_prev) {
        int s[] = {batch, hs};
        TensorConfig cfg = {.dtype = input->dtype, .device = input->device,
                            .has_dtype = true, .has_device = true};
        c_prev = tensor_zeros(s, 2, &cfg);
        free_c = true;
    }
    tensor_ensure_executed(h_prev);
    tensor_ensure_executed(c_prev);

    float* x_data = (float*)tensor_data_ptr(input);
    float* h_data = (float*)tensor_data_ptr(h_prev);
    float* c_data = (float*)tensor_data_ptr(c_prev);
    float* wih    = (float*)tensor_data_ptr(cell->weight_ih->tensor);
    float* whh    = (float*)tensor_data_ptr(cell->weight_hh->tensor);

    float* bih = NULL;
    float* bhh = NULL;
    if (cell->bias_ih) {
        tensor_ensure_executed(cell->bias_ih->tensor);
        bih = (float*)tensor_data_ptr(cell->bias_ih->tensor);
    }
    if (cell->bias_hh) {
        tensor_ensure_executed(cell->bias_hh->tensor);
        bhh = (float*)tensor_data_ptr(cell->bias_hh->tensor);
    }

    /* Compute gates = W_ih @ x + b_ih + W_hh @ h + b_hh */
    float* ih_out = calloc((size_t)(batch * gs), sizeof(float));
    float* hh_out = calloc((size_t)(batch * gs), sizeof(float));
    if (!ih_out || !hh_out) {
        free(ih_out);
        free(hh_out);
        if (free_h) tensor_free(h_prev);
        if (free_c) tensor_free(c_prev);
        return;
    }

    matmul_add(ih_out, wih, x_data, bih, gs, is, batch);
    matmul_add(hh_out, whh, h_data, bhh, gs, hs, batch);

    /* Allocate output tensors */
    int s[] = {batch, hs};
    TensorConfig cfg = {.dtype = input->dtype, .device = input->device,
                        .has_dtype = true, .has_device = true};
    Tensor* h_new = tensor_empty(s, 2, &cfg);
    Tensor* c_new = tensor_empty(s, 2, &cfg);
    tensor_ensure_executed(h_new);
    tensor_ensure_executed(c_new);
    float* h_new_data = (float*)tensor_data_ptr(h_new);
    float* c_new_data = (float*)tensor_data_ptr(c_new);

    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < hs; j++) {
            int base = b * gs;
            float gate_i = ih_out[base + 0 * hs + j] + hh_out[base + 0 * hs + j];
            float gate_f = ih_out[base + 1 * hs + j] + hh_out[base + 1 * hs + j];
            float gate_g = ih_out[base + 2 * hs + j] + hh_out[base + 2 * hs + j];
            float gate_o = ih_out[base + 3 * hs + j] + hh_out[base + 3 * hs + j];

            float i = sigmoid_f(gate_i);
            float f = sigmoid_f(gate_f);
            float g = tanh_f(gate_g);
            float o = sigmoid_f(gate_o);

            int idx = b * hs + j;
            c_new_data[idx] = f * c_data[idx] + i * g;
            h_new_data[idx] = o * tanh_f(c_new_data[idx]);
        }
    }

    free(ih_out);
    free(hh_out);
    if (free_h) tensor_free(h_prev);
    if (free_c) tensor_free(c_prev);

    *h_out = h_new;
    *c_out = c_new;
}

LSTMCell* nn_lstm_cell(int input_size, int hidden_size, bool use_bias,
                       DType dtype, DeviceType device) {
    LSTMCell* cell = malloc(sizeof(LSTMCell));
    if (!cell) return NULL;

    if (module_init((Module*)cell, "LSTMCell",
                    lstm_cell_module_forward, lstm_cell_free) != 0) {
        free(cell);
        return NULL;
    }

    cell->input_size  = input_size;
    cell->hidden_size = hidden_size;
    cell->use_bias    = use_bias;

    TensorConfig cfg = {.dtype = dtype, .device = device,
                        .has_dtype = true, .has_device = true};
    int gs    = 4 * hidden_size;
    float scale = 1.0f / sqrtf((float)hidden_size);

    /* weight_ih [4*hidden_size, input_size] */
    int wih_shape[] = {gs, input_size};
    Tensor* wih = tensor_empty(wih_shape, 2, &cfg);
    tensor_ensure_executed(wih);
    float* wih_data = (float*)tensor_data_ptr(wih);
    for (size_t i = 0; i < wih->numel; i++)
        wih_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    module_add_parameter((Module*)cell, wih, "weight_ih", true);
    cell->weight_ih = module_get_parameter((Module*)cell, "weight_ih");

    /* weight_hh [4*hidden_size, hidden_size] */
    int whh_shape[] = {gs, hidden_size};
    Tensor* whh = tensor_empty(whh_shape, 2, &cfg);
    tensor_ensure_executed(whh);
    float* whh_data = (float*)tensor_data_ptr(whh);
    for (size_t i = 0; i < whh->numel; i++)
        whh_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    module_add_parameter((Module*)cell, whh, "weight_hh", true);
    cell->weight_hh = module_get_parameter((Module*)cell, "weight_hh");

    /* Biases */
    if (use_bias) {
        int b_shape[] = {gs};

        Tensor* bih = tensor_zeros(b_shape, 1, &cfg);
        module_add_parameter((Module*)cell, bih, "bias_ih", true);
        cell->bias_ih = module_get_parameter((Module*)cell, "bias_ih");

        Tensor* bhh = tensor_zeros(b_shape, 1, &cfg);
        module_add_parameter((Module*)cell, bhh, "bias_hh", true);
        cell->bias_hh = module_get_parameter((Module*)cell, "bias_hh");
    } else {
        cell->bias_ih = NULL;
        cell->bias_hh = NULL;
    }

    LOG_DEBUG("Created LSTMCell: input_size=%d, hidden_size=%d, bias=%d",
              input_size, hidden_size, use_bias);

    return cell;
}

static Tensor* gru_cell_module_forward(Module* module, Tensor* input) {
    (void)module;
    (void)input;
    return NULL;
}

static void gru_cell_free(Module* module) { free(module); }

Tensor* gru_cell_forward(GRUCell* cell, Tensor* input, Tensor* hidden) {
    if (!cell || !input) return NULL;

    tensor_ensure_executed(input);
    tensor_ensure_executed(cell->weight_ih->tensor);
    tensor_ensure_executed(cell->weight_hh->tensor);

    int batch = input->shape[0];
    int hs    = cell->hidden_size;
    int is    = cell->input_size;
    int gs    = 3 * hs; /* gate size */

    /* Initialise hidden to zeros when NULL */
    bool free_hidden = false;
    if (!hidden) {
        int s[] = {batch, hs};
        TensorConfig cfg = {.dtype = input->dtype, .device = input->device,
                            .has_dtype = true, .has_device = true};
        hidden = tensor_zeros(s, 2, &cfg);
        free_hidden = true;
    }
    tensor_ensure_executed(hidden);

    float* x_data = (float*)tensor_data_ptr(input);
    float* h_data = (float*)tensor_data_ptr(hidden);
    float* wih    = (float*)tensor_data_ptr(cell->weight_ih->tensor);
    float* whh    = (float*)tensor_data_ptr(cell->weight_hh->tensor);

    float* bih = NULL;
    float* bhh = NULL;
    if (cell->bias_ih) {
        tensor_ensure_executed(cell->bias_ih->tensor);
        bih = (float*)tensor_data_ptr(cell->bias_ih->tensor);
    }
    if (cell->bias_hh) {
        tensor_ensure_executed(cell->bias_hh->tensor);
        bhh = (float*)tensor_data_ptr(cell->bias_hh->tensor);
    }

    /* Compute ih = W_ih @ x + b_ih  and  hh = W_hh @ h + b_hh */
    float* ih_out = calloc((size_t)(batch * gs), sizeof(float));
    float* hh_out = calloc((size_t)(batch * gs), sizeof(float));
    if (!ih_out || !hh_out) {
        free(ih_out);
        free(hh_out);
        if (free_hidden) tensor_free(hidden);
        return NULL;
    }

    matmul_add(ih_out, wih, x_data, bih, gs, is, batch);
    matmul_add(hh_out, whh, h_data, bhh, gs, hs, batch);

    /* Allocate output tensor */
    int s[] = {batch, hs};
    TensorConfig cfg = {.dtype = input->dtype, .device = input->device,
                        .has_dtype = true, .has_device = true};
    Tensor* h_new = tensor_empty(s, 2, &cfg);
    tensor_ensure_executed(h_new);
    float* h_new_data = (float*)tensor_data_ptr(h_new);

    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < hs; j++) {
            int base = b * gs;

            /* Reset and update gates combine ih and hh linearly */
            float r = sigmoid_f(ih_out[base + 0 * hs + j] + hh_out[base + 0 * hs + j]);
            float z = sigmoid_f(ih_out[base + 1 * hs + j] + hh_out[base + 1 * hs + j]);

            /* New gate: ih_n + r * hh_n */
            float n = tanh_f(ih_out[base + 2 * hs + j] + r * hh_out[base + 2 * hs + j]);

            int idx = b * hs + j;
            h_new_data[idx] = (1.0f - z) * n + z * h_data[idx];
        }
    }

    free(ih_out);
    free(hh_out);
    if (free_hidden) tensor_free(hidden);

    return h_new;
}

GRUCell* nn_gru_cell(int input_size, int hidden_size, bool use_bias,
                     DType dtype, DeviceType device) {
    GRUCell* cell = malloc(sizeof(GRUCell));
    if (!cell) return NULL;

    if (module_init((Module*)cell, "GRUCell",
                    gru_cell_module_forward, gru_cell_free) != 0) {
        free(cell);
        return NULL;
    }

    cell->input_size  = input_size;
    cell->hidden_size = hidden_size;
    cell->use_bias    = use_bias;

    TensorConfig cfg = {.dtype = dtype, .device = device,
                        .has_dtype = true, .has_device = true};
    int gs    = 3 * hidden_size;
    float scale = 1.0f / sqrtf((float)hidden_size);

    /* weight_ih [3*hidden_size, input_size] */
    int wih_shape[] = {gs, input_size};
    Tensor* wih = tensor_empty(wih_shape, 2, &cfg);
    tensor_ensure_executed(wih);
    float* wih_data = (float*)tensor_data_ptr(wih);
    for (size_t i = 0; i < wih->numel; i++)
        wih_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    module_add_parameter((Module*)cell, wih, "weight_ih", true);
    cell->weight_ih = module_get_parameter((Module*)cell, "weight_ih");

    /* weight_hh [3*hidden_size, hidden_size] */
    int whh_shape[] = {gs, hidden_size};
    Tensor* whh = tensor_empty(whh_shape, 2, &cfg);
    tensor_ensure_executed(whh);
    float* whh_data = (float*)tensor_data_ptr(whh);
    for (size_t i = 0; i < whh->numel; i++)
        whh_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    module_add_parameter((Module*)cell, whh, "weight_hh", true);
    cell->weight_hh = module_get_parameter((Module*)cell, "weight_hh");

    /* Biases */
    if (use_bias) {
        int b_shape[] = {gs};

        Tensor* bih = tensor_zeros(b_shape, 1, &cfg);
        module_add_parameter((Module*)cell, bih, "bias_ih", true);
        cell->bias_ih = module_get_parameter((Module*)cell, "bias_ih");

        Tensor* bhh = tensor_zeros(b_shape, 1, &cfg);
        module_add_parameter((Module*)cell, bhh, "bias_hh", true);
        cell->bias_hh = module_get_parameter((Module*)cell, "bias_hh");
    } else {
        cell->bias_ih = NULL;
        cell->bias_hh = NULL;
    }

    LOG_DEBUG("Created GRUCell: input_size=%d, hidden_size=%d, bias=%d",
              input_size, hidden_size, use_bias);

    return cell;
}
