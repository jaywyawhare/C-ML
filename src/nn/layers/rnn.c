#include "nn/layers/rnn.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

/* out[b][r] = sum_c(weight[r][c] * input[b][c]) + bias[r] */
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

    return cell;
}

static void register_cell_params(Module* parent, Module* cell,
                                 int layer, int dir) {
    const char* dir_str = (dir == 0) ? "fwd" : "rev";
    Parameter** params = NULL;
    int num_params = 0;
    if (module_collect_parameters(cell, &params, &num_params, false) == 0) {
        for (int i = 0; i < num_params; i++) {
            if (params[i]) {
                char pname[256];
                snprintf(pname, sizeof(pname), "layers.%d.%s.%s",
                         layer, dir_str,
                         params[i]->name ? params[i]->name : "unnamed");
                module_add_parameter(parent, params[i]->tensor, pname,
                                     params[i]->requires_grad);
            }
        }
        if (params) free(params);
    }
}

/**
 * @brief Helper: extract a [batch, features] slice from a 3-D tensor
 *        at a given time-step index along dimension 0.
 *
 * src shape: [seq_len, batch, features]
 * Returns a NEW tensor of shape [batch, features] with a copy of the data.
 */
static Tensor* slice_timestep(Tensor* src, int t, DType dtype, DeviceType device) {
    int batch = src->shape[1];
    int feat  = src->shape[2];
    int out_shape[] = {batch, feat};
    TensorConfig cfg = {.dtype = dtype, .device = device,
                        .has_dtype = true, .has_device = true};
    Tensor* out = tensor_empty(out_shape, 2, &cfg);
    tensor_ensure_executed(out);
    float* dst_data = (float*)tensor_data_ptr(out);
    float* src_data = (float*)tensor_data_ptr(src);
    size_t stride = (size_t)(batch * feat);
    memcpy(dst_data, src_data + (size_t)t * stride, stride * sizeof(float));
    return out;
}

/**
 * @brief Helper: write a [batch, features] slice into a 3-D tensor
 *        at a given time-step index along dimension 0.
 */
static void write_timestep(Tensor* dst, int t, Tensor* src) {
    int batch = dst->shape[1];
    int feat  = dst->shape[2];
    float* dst_data = (float*)tensor_data_ptr(dst);
    float* src_data = (float*)tensor_data_ptr(src);
    size_t stride = (size_t)(batch * feat);
    memcpy(dst_data + (size_t)t * stride, src_data, stride * sizeof(float));
}

/**
 * @brief Helper: transpose a 3-D tensor between
 *        [dim0, dim1, dim2] <-> [dim1, dim0, dim2].
 *
 * Used to convert between batch_first and seq_first layouts.
 * Returns a NEW tensor; caller is responsible for freeing it.
 */
static Tensor* transpose_01(Tensor* src) {
    int d0 = src->shape[0];
    int d1 = src->shape[1];
    int d2 = src->shape[2];
    int out_shape[] = {d1, d0, d2};
    TensorConfig cfg = {.dtype = src->dtype, .device = src->device,
                        .has_dtype = true, .has_device = true};
    Tensor* out = tensor_empty(out_shape, 3, &cfg);
    tensor_ensure_executed(out);
    float* s = (float*)tensor_data_ptr(src);
    float* d = (float*)tensor_data_ptr(out);
    for (int i = 0; i < d0; i++) {
        for (int j = 0; j < d1; j++) {
            memcpy(d + ((size_t)j * d0 + i) * d2,
                   s + ((size_t)i * d1 + j) * d2,
                   (size_t)d2 * sizeof(float));
        }
    }
    return out;
}

/**
 * @brief Helper: concatenate two [seq_len, batch, feat] tensors along the
 *        feature (last) dimension, producing [seq_len, batch, feat_a + feat_b].
 */
static Tensor* concat_features(Tensor* a, Tensor* b) {
    int seq   = a->shape[0];
    int batch = a->shape[1];
    int fa    = a->shape[2];
    int fb    = b->shape[2];
    int fc    = fa + fb;
    int out_shape[] = {seq, batch, fc};
    TensorConfig cfg = {.dtype = a->dtype, .device = a->device,
                        .has_dtype = true, .has_device = true};
    Tensor* out = tensor_empty(out_shape, 3, &cfg);
    tensor_ensure_executed(out);
    float* od = (float*)tensor_data_ptr(out);
    float* ad = (float*)tensor_data_ptr(a);
    float* bd = (float*)tensor_data_ptr(b);
    for (int t = 0; t < seq; t++) {
        for (int n = 0; n < batch; n++) {
            size_t o_off = ((size_t)t * batch + n) * fc;
            size_t a_off = ((size_t)t * batch + n) * fa;
            size_t b_off = ((size_t)t * batch + n) * fb;
            memcpy(od + o_off,      ad + a_off, (size_t)fa * sizeof(float));
            memcpy(od + o_off + fa, bd + b_off, (size_t)fb * sizeof(float));
        }
    }
    return out;
}

/**
 * @brief Helper: write a hidden-state slice into h_n at the correct position.
 *
 * h_n shape:  [num_layers * num_directions, batch, hidden_size]
 * h   shape:  [batch, hidden_size]
 * index:      layer * num_directions + dir
 */
static void write_hidden(Tensor* h_n, int index, Tensor* h) {
    int batch = h_n->shape[1];
    int hs    = h_n->shape[2];
    float* dst = (float*)tensor_data_ptr(h_n);
    float* src = (float*)tensor_data_ptr(h);
    size_t stride = (size_t)(batch * hs);
    memcpy(dst + (size_t)index * stride, src, stride * sizeof(float));
}

static Tensor* rnn_module_forward(Module* module, Tensor* input) {
    (void)module; (void)input;
    return NULL;
}

static void rnn_free(Module* module) {
    RNN* rnn = (RNN*)module;
    if (rnn->cells) {
        int total = rnn->num_layers * rnn->num_directions;
        for (int i = 0; i < total; i++) {
            if (rnn->cells[i]) {
                rnn_cell_free((Module*)rnn->cells[i]);
            }
        }
        free(rnn->cells);
    }
    free(rnn);
}

RNN* nn_rnn(int input_size, int hidden_size, int num_layers, bool bidirectional,
            bool batch_first, float dropout, bool use_bias,
            DType dtype, DeviceType device) {
    RNN* rnn = malloc(sizeof(RNN));
    if (!rnn) return NULL;

    if (module_init((Module*)rnn, "RNN", rnn_module_forward, rnn_free) != 0) {
        free(rnn);
        return NULL;
    }

    rnn->input_size    = input_size;
    rnn->hidden_size   = hidden_size;
    rnn->num_layers    = num_layers;
    rnn->bidirectional = bidirectional;
    rnn->batch_first   = batch_first;
    rnn->dropout_p     = dropout;
    rnn->use_bias      = use_bias;
    rnn->dtype         = dtype;
    rnn->device        = device;
    rnn->num_directions = bidirectional ? 2 : 1;

    int total = num_layers * rnn->num_directions;
    rnn->cells = calloc((size_t)total, sizeof(RNNCell*));
    if (!rnn->cells) { free(rnn); return NULL; }

    for (int l = 0; l < num_layers; l++) {
        int cell_input = (l == 0) ? input_size
                                  : hidden_size * rnn->num_directions;
        for (int d = 0; d < rnn->num_directions; d++) {
            int idx = l * rnn->num_directions + d;
            rnn->cells[idx] = nn_rnn_cell(cell_input, hidden_size,
                                           use_bias, dtype, device);
            if (!rnn->cells[idx]) {
                rnn_free((Module*)rnn);
                return NULL;
            }
            register_cell_params((Module*)rnn, (Module*)rnn->cells[idx], l, d);
        }
    }
    return rnn;
}

void rnn_forward(RNN* rnn, Tensor* input, Tensor* h_0,
                 Tensor** output, Tensor** h_n) {
    if (!rnn || !input || !output || !h_n) return;

    tensor_ensure_executed(input);

    int nd = rnn->num_directions;
    int hs = rnn->hidden_size;
    DType dt = rnn->dtype;
    DeviceType dv = rnn->device;

    /* batch_first: [batch, seq, feat] -> [seq, batch, feat] */
    Tensor* x = input;
    bool did_transpose = false;
    if (rnn->batch_first) {
        x = transpose_01(input);
        did_transpose = true;
    }

    int seq_len = x->shape[0];
    int batch   = x->shape[1];

    /* Allocate h_n: [num_layers * num_directions, batch, hidden_size] */
    int hn_shape[] = {rnn->num_layers * nd, batch, hs};
    TensorConfig cfg = {.dtype = dt, .device = dv,
                        .has_dtype = true, .has_device = true};
    Tensor* hn = tensor_zeros(hn_shape, 3, &cfg);
    tensor_ensure_executed(hn);

    /* Current layer input starts as x */
    Tensor* layer_input = x;
    bool free_layer_input = false;

    for (int l = 0; l < rnn->num_layers; l++) {
        (void)layer_input->shape[2]; // feature dim used implicitly by cells


        RNNCell* fwd_cell = rnn->cells[l * nd + 0];

        /* Get initial hidden for this layer/direction from h_0 */
        Tensor* h_fwd = NULL;
        bool free_h_fwd_init = false;
        if (h_0) {
            tensor_ensure_executed(h_0);
            h_fwd = slice_timestep(h_0, l * nd + 0, dt, dv);
            free_h_fwd_init = true;
        }

        /* Output buffer: [seq_len, batch, hidden_size] */
        int fwd_shape[] = {seq_len, batch, hs};
        Tensor* fwd_out = tensor_empty(fwd_shape, 3, &cfg);
        tensor_ensure_executed(fwd_out);

        for (int t = 0; t < seq_len; t++) {
            Tensor* xt = slice_timestep(layer_input, t, dt, dv);
            Tensor* h_new = rnn_cell_forward(fwd_cell, xt, h_fwd);
            tensor_ensure_executed(h_new);
            write_timestep(fwd_out, t, h_new);
            if (h_fwd) tensor_free(h_fwd);
            h_fwd = h_new;
            tensor_free(xt);
        }
        if (free_h_fwd_init) { /* already freed in loop */ }

        /* Save final hidden state */
        write_hidden(hn, l * nd + 0, h_fwd);
        tensor_free(h_fwd);

        Tensor* layer_output;

        if (nd == 2) {

            RNNCell* rev_cell = rnn->cells[l * nd + 1];

            Tensor* h_rev = NULL;
            if (h_0) {
                h_rev = slice_timestep(h_0, l * nd + 1, dt, dv);
            }

            int rev_shape[] = {seq_len, batch, hs};
            Tensor* rev_out = tensor_empty(rev_shape, 3, &cfg);
            tensor_ensure_executed(rev_out);

            for (int t = seq_len - 1; t >= 0; t--) {
                Tensor* xt = slice_timestep(layer_input, t, dt, dv);
                Tensor* h_new = rnn_cell_forward(rev_cell, xt, h_rev);
                tensor_ensure_executed(h_new);
                write_timestep(rev_out, t, h_new);
                if (h_rev) tensor_free(h_rev);
                h_rev = h_new;
                tensor_free(xt);
            }

            write_hidden(hn, l * nd + 1, h_rev);
            tensor_free(h_rev);

            /* Concatenate forward and reverse along feature dim */
            layer_output = concat_features(fwd_out, rev_out);
            tensor_free(fwd_out);
            tensor_free(rev_out);
        } else {
            layer_output = fwd_out;
        }

        /* Free previous layer input (unless it is the original x) */
        if (free_layer_input) tensor_free(layer_input);
        layer_input = layer_output;
        free_layer_input = true;
    }

    /* batch_first: [seq, batch, feat] -> [batch, seq, feat] */
    if (rnn->batch_first) {
        Tensor* tmp = transpose_01(layer_input);
        if (free_layer_input) tensor_free(layer_input);
        layer_input = tmp;
        free_layer_input = true;
    }

    if (did_transpose) tensor_free(x);

    *output = layer_input;
    *h_n = hn;
}

static Tensor* lstm_module_forward(Module* module, Tensor* input) {
    (void)module; (void)input;
    return NULL;
}

static void lstm_free(Module* module) {
    LSTM* lstm = (LSTM*)module;
    if (lstm->cells) {
        int total = lstm->num_layers * lstm->num_directions;
        for (int i = 0; i < total; i++) {
            if (lstm->cells[i]) {
                lstm_cell_free((Module*)lstm->cells[i]);
            }
        }
        free(lstm->cells);
    }
    free(lstm);
}

LSTM* nn_lstm(int input_size, int hidden_size, int num_layers, bool bidirectional,
              bool batch_first, float dropout, bool use_bias,
              DType dtype, DeviceType device) {
    LSTM* lstm = malloc(sizeof(LSTM));
    if (!lstm) return NULL;

    if (module_init((Module*)lstm, "LSTM", lstm_module_forward, lstm_free) != 0) {
        free(lstm);
        return NULL;
    }

    lstm->input_size    = input_size;
    lstm->hidden_size   = hidden_size;
    lstm->num_layers    = num_layers;
    lstm->bidirectional = bidirectional;
    lstm->batch_first   = batch_first;
    lstm->dropout_p     = dropout;
    lstm->use_bias      = use_bias;
    lstm->dtype         = dtype;
    lstm->device        = device;
    lstm->num_directions = bidirectional ? 2 : 1;

    int total = num_layers * lstm->num_directions;
    lstm->cells = calloc((size_t)total, sizeof(LSTMCell*));
    if (!lstm->cells) { free(lstm); return NULL; }

    for (int l = 0; l < num_layers; l++) {
        int cell_input = (l == 0) ? input_size
                                  : hidden_size * lstm->num_directions;
        for (int d = 0; d < lstm->num_directions; d++) {
            int idx = l * lstm->num_directions + d;
            lstm->cells[idx] = nn_lstm_cell(cell_input, hidden_size,
                                             use_bias, dtype, device);
            if (!lstm->cells[idx]) {
                lstm_free((Module*)lstm);
                return NULL;
            }
            register_cell_params((Module*)lstm, (Module*)lstm->cells[idx], l, d);
        }
    }
    return lstm;
}

void lstm_forward(LSTM* lstm, Tensor* input, Tensor* h_0, Tensor* c_0,
                  Tensor** output, Tensor** h_n, Tensor** c_n) {
    if (!lstm || !input || !output || !h_n || !c_n) return;

    tensor_ensure_executed(input);

    int nd = lstm->num_directions;
    int hs = lstm->hidden_size;
    DType dt = lstm->dtype;
    DeviceType dv = lstm->device;

    /* batch_first: [batch, seq, feat] -> [seq, batch, feat] */
    Tensor* x = input;
    bool did_transpose = false;
    if (lstm->batch_first) {
        x = transpose_01(input);
        did_transpose = true;
    }

    int seq_len = x->shape[0];
    int batch   = x->shape[1];

    TensorConfig cfg = {.dtype = dt, .device = dv,
                        .has_dtype = true, .has_device = true};

    /* Allocate h_n, c_n: [num_layers * num_directions, batch, hidden_size] */
    int hn_shape[] = {lstm->num_layers * nd, batch, hs};
    Tensor* hn = tensor_zeros(hn_shape, 3, &cfg);
    Tensor* cn = tensor_zeros(hn_shape, 3, &cfg);
    tensor_ensure_executed(hn);
    tensor_ensure_executed(cn);

    Tensor* layer_input = x;
    bool free_layer_input = false;

    for (int l = 0; l < lstm->num_layers; l++) {


        LSTMCell* fwd_cell = lstm->cells[l * nd + 0];

        Tensor* h_fwd = NULL;
        Tensor* c_fwd = NULL;
        if (h_0) {
            tensor_ensure_executed(h_0);
            h_fwd = slice_timestep(h_0, l * nd + 0, dt, dv);
        }
        if (c_0) {
            tensor_ensure_executed(c_0);
            c_fwd = slice_timestep(c_0, l * nd + 0, dt, dv);
        }

        int fwd_shape[] = {seq_len, batch, hs};
        Tensor* fwd_out = tensor_empty(fwd_shape, 3, &cfg);
        tensor_ensure_executed(fwd_out);

        for (int t = 0; t < seq_len; t++) {
            Tensor* xt = slice_timestep(layer_input, t, dt, dv);
            Tensor* h_new = NULL;
            Tensor* c_new = NULL;
            lstm_cell_forward(fwd_cell, xt, h_fwd, c_fwd, &h_new, &c_new);
            tensor_ensure_executed(h_new);
            tensor_ensure_executed(c_new);
            write_timestep(fwd_out, t, h_new);
            if (h_fwd) tensor_free(h_fwd);
            if (c_fwd) tensor_free(c_fwd);
            h_fwd = h_new;
            c_fwd = c_new;
            tensor_free(xt);
        }

        write_hidden(hn, l * nd + 0, h_fwd);
        write_hidden(cn, l * nd + 0, c_fwd);
        tensor_free(h_fwd);
        tensor_free(c_fwd);

        Tensor* layer_output;

        if (nd == 2) {

            LSTMCell* rev_cell = lstm->cells[l * nd + 1];

            Tensor* h_rev = NULL;
            Tensor* c_rev = NULL;
            if (h_0) {
                h_rev = slice_timestep(h_0, l * nd + 1, dt, dv);
            }
            if (c_0) {
                c_rev = slice_timestep(c_0, l * nd + 1, dt, dv);
            }

            int rev_shape[] = {seq_len, batch, hs};
            Tensor* rev_out = tensor_empty(rev_shape, 3, &cfg);
            tensor_ensure_executed(rev_out);

            for (int t = seq_len - 1; t >= 0; t--) {
                Tensor* xt = slice_timestep(layer_input, t, dt, dv);
                Tensor* h_new = NULL;
                Tensor* c_new = NULL;
                lstm_cell_forward(rev_cell, xt, h_rev, c_rev, &h_new, &c_new);
                tensor_ensure_executed(h_new);
                tensor_ensure_executed(c_new);
                write_timestep(rev_out, t, h_new);
                if (h_rev) tensor_free(h_rev);
                if (c_rev) tensor_free(c_rev);
                h_rev = h_new;
                c_rev = c_new;
                tensor_free(xt);
            }

            write_hidden(hn, l * nd + 1, h_rev);
            write_hidden(cn, l * nd + 1, c_rev);
            tensor_free(h_rev);
            tensor_free(c_rev);

            layer_output = concat_features(fwd_out, rev_out);
            tensor_free(fwd_out);
            tensor_free(rev_out);
        } else {
            layer_output = fwd_out;
        }

        if (free_layer_input) tensor_free(layer_input);
        layer_input = layer_output;
        free_layer_input = true;
    }

    /* batch_first: [seq, batch, feat] -> [batch, seq, feat] */
    if (lstm->batch_first) {
        Tensor* tmp = transpose_01(layer_input);
        if (free_layer_input) tensor_free(layer_input);
        layer_input = tmp;
        free_layer_input = true;
    }

    if (did_transpose) tensor_free(x);

    *output = layer_input;
    *h_n = hn;
    *c_n = cn;
}

static Tensor* gru_module_forward(Module* module, Tensor* input) {
    (void)module; (void)input;
    return NULL;
}

static void gru_free(Module* module) {
    GRU* gru = (GRU*)module;
    if (gru->cells) {
        int total = gru->num_layers * gru->num_directions;
        for (int i = 0; i < total; i++) {
            if (gru->cells[i]) {
                gru_cell_free((Module*)gru->cells[i]);
            }
        }
        free(gru->cells);
    }
    free(gru);
}

GRU* nn_gru(int input_size, int hidden_size, int num_layers, bool bidirectional,
            bool batch_first, float dropout, bool use_bias,
            DType dtype, DeviceType device) {
    GRU* gru = malloc(sizeof(GRU));
    if (!gru) return NULL;

    if (module_init((Module*)gru, "GRU", gru_module_forward, gru_free) != 0) {
        free(gru);
        return NULL;
    }

    gru->input_size    = input_size;
    gru->hidden_size   = hidden_size;
    gru->num_layers    = num_layers;
    gru->bidirectional = bidirectional;
    gru->batch_first   = batch_first;
    gru->dropout_p     = dropout;
    gru->use_bias      = use_bias;
    gru->dtype         = dtype;
    gru->device        = device;
    gru->num_directions = bidirectional ? 2 : 1;

    int total = num_layers * gru->num_directions;
    gru->cells = calloc((size_t)total, sizeof(GRUCell*));
    if (!gru->cells) { free(gru); return NULL; }

    for (int l = 0; l < num_layers; l++) {
        int cell_input = (l == 0) ? input_size
                                  : hidden_size * gru->num_directions;
        for (int d = 0; d < gru->num_directions; d++) {
            int idx = l * gru->num_directions + d;
            gru->cells[idx] = nn_gru_cell(cell_input, hidden_size,
                                           use_bias, dtype, device);
            if (!gru->cells[idx]) {
                gru_free((Module*)gru);
                return NULL;
            }
            register_cell_params((Module*)gru, (Module*)gru->cells[idx], l, d);
        }
    }
    return gru;
}

void gru_forward(GRU* gru, Tensor* input, Tensor* h_0,
                 Tensor** output, Tensor** h_n) {
    if (!gru || !input || !output || !h_n) return;

    tensor_ensure_executed(input);

    int nd = gru->num_directions;
    int hs = gru->hidden_size;
    DType dt = gru->dtype;
    DeviceType dv = gru->device;

    /* batch_first: [batch, seq, feat] -> [seq, batch, feat] */
    Tensor* x = input;
    bool did_transpose = false;
    if (gru->batch_first) {
        x = transpose_01(input);
        did_transpose = true;
    }

    int seq_len = x->shape[0];
    int batch   = x->shape[1];

    TensorConfig cfg = {.dtype = dt, .device = dv,
                        .has_dtype = true, .has_device = true};

    /* Allocate h_n: [num_layers * num_directions, batch, hidden_size] */
    int hn_shape[] = {gru->num_layers * nd, batch, hs};
    Tensor* hn = tensor_zeros(hn_shape, 3, &cfg);
    tensor_ensure_executed(hn);

    Tensor* layer_input = x;
    bool free_layer_input = false;

    for (int l = 0; l < gru->num_layers; l++) {


        GRUCell* fwd_cell = gru->cells[l * nd + 0];

        Tensor* h_fwd = NULL;
        if (h_0) {
            tensor_ensure_executed(h_0);
            h_fwd = slice_timestep(h_0, l * nd + 0, dt, dv);
        }

        int fwd_shape[] = {seq_len, batch, hs};
        Tensor* fwd_out = tensor_empty(fwd_shape, 3, &cfg);
        tensor_ensure_executed(fwd_out);

        for (int t = 0; t < seq_len; t++) {
            Tensor* xt = slice_timestep(layer_input, t, dt, dv);
            Tensor* h_new = gru_cell_forward(fwd_cell, xt, h_fwd);
            tensor_ensure_executed(h_new);
            write_timestep(fwd_out, t, h_new);
            if (h_fwd) tensor_free(h_fwd);
            h_fwd = h_new;
            tensor_free(xt);
        }

        write_hidden(hn, l * nd + 0, h_fwd);
        tensor_free(h_fwd);

        Tensor* layer_output;

        if (nd == 2) {

            GRUCell* rev_cell = gru->cells[l * nd + 1];

            Tensor* h_rev = NULL;
            if (h_0) {
                h_rev = slice_timestep(h_0, l * nd + 1, dt, dv);
            }

            int rev_shape[] = {seq_len, batch, hs};
            Tensor* rev_out = tensor_empty(rev_shape, 3, &cfg);
            tensor_ensure_executed(rev_out);

            for (int t = seq_len - 1; t >= 0; t--) {
                Tensor* xt = slice_timestep(layer_input, t, dt, dv);
                Tensor* h_new = gru_cell_forward(rev_cell, xt, h_rev);
                tensor_ensure_executed(h_new);
                write_timestep(rev_out, t, h_new);
                if (h_rev) tensor_free(h_rev);
                h_rev = h_new;
                tensor_free(xt);
            }

            write_hidden(hn, l * nd + 1, h_rev);
            tensor_free(h_rev);

            layer_output = concat_features(fwd_out, rev_out);
            tensor_free(fwd_out);
            tensor_free(rev_out);
        } else {
            layer_output = fwd_out;
        }

        if (free_layer_input) tensor_free(layer_input);
        layer_input = layer_output;
        free_layer_input = true;
    }

    /* batch_first: [seq, batch, feat] -> [batch, seq, feat] */
    if (gru->batch_first) {
        Tensor* tmp = transpose_01(layer_input);
        if (free_layer_input) tensor_free(layer_input);
        layer_input = tmp;
        free_layer_input = true;
    }

    if (did_transpose) tensor_free(x);

    *output = layer_input;
    *h_n = hn;
}
