#include "nn/layers/rnn.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>


static Tensor* rnn_cell_module_forward(Module* module, Tensor* input) {
    /* The Module interface does not carry the hidden state, so users
       should call rnn_cell_forward() directly. */
    (void)module;
    (void)input;
    return NULL;
}

static void rnn_cell_free(Module* module) {
    if (module->parameters) {
        for (int i = 0; i < module->num_parameters; i++) {
            Parameter* p = module->parameters[i];
            if (!p)
                continue;
            if (p->name)
                free(p->name);
            if (p->tensor)
                tensor_free(p->tensor);
            free(p);
        }
        free(module->parameters);
        module->parameters = NULL;
    }
    if (module->name) {
        free(module->name);
        module->name = NULL;
    }
    free(module);
}

Tensor* rnn_cell_forward(RNNCell* cell, Tensor* input, Tensor* hidden) {
    if (!cell || !input) return NULL;

    int batch = input->shape[0];
    int hs    = cell->hidden_size;

    /* Initialise hidden to zeros when NULL */
    if (!hidden) {
        int h_shape[] = {batch, hs};
        TensorConfig cfg = {.dtype = input->dtype, .device = input->device,
                            .has_dtype = true, .has_device = true};
        hidden = tensor_zeros(h_shape, 2, &cfg);
    }

    /* h_new = tanh(input @ W_ih^T + hidden @ W_hh^T + b_ih + b_hh)
     * All ops go through autograd so gradients reach the weight parameters. */
    Tensor* wih_t = tensor_transpose(cell->weight_ih->tensor, 0, 1);
    Tensor* whh_t = tensor_transpose(cell->weight_hh->tensor, 0, 1);

    Tensor* h_new = tensor_add(tensor_matmul(input, wih_t),
                               tensor_matmul(hidden, whh_t));

    if (cell->bias_ih)
        h_new = tensor_add(h_new, cell->bias_ih->tensor);
    if (cell->bias_hh)
        h_new = tensor_add(h_new, cell->bias_hh->tensor);

    h_new = uop_tanh(h_new);

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

static void lstm_cell_free(Module* module) {
    if (module->parameters) {
        for (int i = 0; i < module->num_parameters; i++) {
            Parameter* p = module->parameters[i];
            if (!p)
                continue;
            if (p->name)
                free(p->name);
            if (p->tensor)
                tensor_free(p->tensor);
            free(p);
        }
        free(module->parameters);
        module->parameters = NULL;
    }
    if (module->name) {
        free(module->name);
        module->name = NULL;
    }
    free(module);
}

void lstm_cell_forward(LSTMCell* cell, Tensor* input,
                       Tensor* h_prev, Tensor* c_prev,
                       Tensor** h_out, Tensor** c_out) {
    if (!cell || !input || !h_out || !c_out) return;

    int batch = input->shape[0];
    int hs    = cell->hidden_size;
    (void)(4 * hs); /* gs unused after autograd rewrite */

    /* Initialise states to zeros when NULL */
    if (!h_prev) {
        int s[] = {batch, hs};
        TensorConfig cfg = {.dtype = input->dtype, .device = input->device,
                            .has_dtype = true, .has_device = true};
        h_prev = tensor_zeros(s, 2, &cfg);
    }
    if (!c_prev) {
        int s[] = {batch, hs};
        TensorConfig cfg = {.dtype = input->dtype, .device = input->device,
                            .has_dtype = true, .has_device = true};
        c_prev = tensor_zeros(s, 2, &cfg);
    }

    /* gates = input @ W_ih^T + h_prev @ W_hh^T + b_ih + b_hh
     * All ops go through autograd so gradients reach the weight parameters. */
    Tensor* wih_t = tensor_transpose(cell->weight_ih->tensor, 0, 1); /* [is, 4*hs] */
    Tensor* whh_t = tensor_transpose(cell->weight_hh->tensor, 0, 1); /* [hs, 4*hs] */

    Tensor* gates = tensor_add(tensor_matmul(input, wih_t),
                               tensor_matmul(h_prev, whh_t));  /* [batch, 4*hs] */

    if (cell->bias_ih)
        gates = tensor_add(gates, cell->bias_ih->tensor);
    if (cell->bias_hh)
        gates = tensor_add(gates, cell->bias_hh->tensor);

    /* Split gates into i, f, g, o via shrink — each [batch, hs] */
    int starts_full[] = {0, 0};
    int ends_full[]   = {batch, hs};

    int starts_i[] = {0, 0 * hs}, ends_i[] = {batch, 1 * hs};
    int starts_f[] = {0, 1 * hs}, ends_f[] = {batch, 2 * hs};
    int starts_g[] = {0, 2 * hs}, ends_g[] = {batch, 3 * hs};
    int starts_o[] = {0, 3 * hs}, ends_o[] = {batch, 4 * hs};
    (void)starts_full; (void)ends_full;

    Tensor* gate_i = uop_shrink(gates, starts_i, ends_i, 2); /* input gate   */
    Tensor* gate_f = uop_shrink(gates, starts_f, ends_f, 2); /* forget gate  */
    Tensor* gate_g = uop_shrink(gates, starts_g, ends_g, 2); /* cell gate    */
    Tensor* gate_o = uop_shrink(gates, starts_o, ends_o, 2); /* output gate  */

    Tensor* i_act = uop_sigmoid(gate_i);
    Tensor* f_act = uop_sigmoid(gate_f);
    Tensor* g_act = uop_tanh(gate_g);
    Tensor* o_act = uop_sigmoid(gate_o);

    /* c_new = f ⊙ c_prev + i ⊙ g */
    Tensor* c_new = tensor_add(tensor_mul(f_act, c_prev),
                               tensor_mul(i_act, g_act));

    /* h_new = o ⊙ tanh(c_new) */
    Tensor* h_new = tensor_mul(o_act, uop_tanh(c_new));

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

static void gru_cell_free(Module* module) {
    if (module->parameters) {
        for (int i = 0; i < module->num_parameters; i++) {
            Parameter* p = module->parameters[i];
            if (!p)
                continue;
            if (p->name)
                free(p->name);
            if (p->tensor)
                tensor_free(p->tensor);
            free(p);
        }
        free(module->parameters);
        module->parameters = NULL;
    }
    if (module->name) {
        free(module->name);
        module->name = NULL;
    }
    free(module);
}

Tensor* gru_cell_forward(GRUCell* cell, Tensor* input, Tensor* hidden) {
    if (!cell || !input) return NULL;

    int batch = input->shape[0];
    int hs    = cell->hidden_size;

    /* Initialise hidden to zeros when NULL */
    if (!hidden) {
        int s[] = {batch, hs};
        TensorConfig cfg = {.dtype = input->dtype, .device = input->device,
                            .has_dtype = true, .has_device = true};
        hidden = tensor_zeros(s, 2, &cfg);
    }

    /* GRU forward using autograd ops so gradients flow to parameters.
     * ih = input @ W_ih^T + b_ih   [batch, 3*hs]
     * hh = hidden @ W_hh^T + b_hh  [batch, 3*hs] */
    Tensor* wih_t = tensor_transpose(cell->weight_ih->tensor, 0, 1);
    Tensor* whh_t = tensor_transpose(cell->weight_hh->tensor, 0, 1);

    Tensor* ih = tensor_matmul(input, wih_t);
    Tensor* hh = tensor_matmul(hidden, whh_t);

    if (cell->bias_ih)
        ih = tensor_add(ih, cell->bias_ih->tensor);
    if (cell->bias_hh)
        hh = tensor_add(hh, cell->bias_hh->tensor);

    /* Split ih and hh into 3 gates of size hs each */
    int s_r[] = {0, 0 * hs}, e_r[] = {batch, 1 * hs};
    int s_z[] = {0, 1 * hs}, e_z[] = {batch, 2 * hs};
    int s_n[] = {0, 2 * hs}, e_n[] = {batch, 3 * hs};

    Tensor* ih_r = uop_shrink(ih, s_r, e_r, 2);
    Tensor* ih_z = uop_shrink(ih, s_z, e_z, 2);
    Tensor* ih_n = uop_shrink(ih, s_n, e_n, 2);

    Tensor* hh_r = uop_shrink(hh, s_r, e_r, 2);
    Tensor* hh_z = uop_shrink(hh, s_z, e_z, 2);
    Tensor* hh_n = uop_shrink(hh, s_n, e_n, 2);

    /* r = sigmoid(ih_r + hh_r), z = sigmoid(ih_z + hh_z) */
    Tensor* r = uop_sigmoid(tensor_add(ih_r, hh_r));
    Tensor* z = uop_sigmoid(tensor_add(ih_z, hh_z));

    /* n = tanh(ih_n + r * hh_n) */
    Tensor* n = uop_tanh(tensor_add(ih_n, tensor_mul(r, hh_n)));

    /* h_new = (1-z)*n + z*hidden = n + z*(hidden - n) */
    Tensor* h_new = tensor_add(n, tensor_mul(z, tensor_sub(hidden, n)));

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
                Tensor* pt = params[i]->tensor;
                nn_tensor_param_alias(pt);
                if (module_add_parameter(parent, pt, pname, params[i]->requires_grad) != 0)
                    pt->ref_count--;
            }
        }
        if (params) free(params);
    }
}

/* Extract [batch, features] slice at time-step t from [seq_len, batch, features]. */
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

static void write_timestep(Tensor* dst, int t, Tensor* src) {
    int batch = dst->shape[1];
    int feat  = dst->shape[2];
    float* dst_data = (float*)tensor_data_ptr(dst);
    float* src_data = (float*)tensor_data_ptr(src);
    size_t stride = (size_t)(batch * feat);
    memcpy(dst_data + (size_t)t * stride, src_data, stride * sizeof(float));
}

/* Transpose dims 0 and 1 of a 3-D tensor (batch_first <-> seq_first). */
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
