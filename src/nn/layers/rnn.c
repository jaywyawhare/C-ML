#include "nn/layers/rnn.h"
#include "nn/init.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "alloc/cml_allocator.h"

/** Module-interface stub: the generic forward cannot carry a hidden state, so it always returns
 *  NULL; call rnn_cell_forward() directly instead. */
static Tensor* rnn_cell_module_forward(Module* module, Tensor* input) {
    /* The Module interface does not carry the hidden state, so users
       should call rnn_cell_forward() directly. */
    (void)module;
    (void)input;
    return NULL;
}

/** torch.nn.RNNCell step: h' = tanh(input @ W_ih^T + hidden @ W_hh^T + b_ih + b_hh). A NULL hidden
 *  is treated as zeros. Returns NULL on NULL cell/input. */
Tensor* rnn_cell_forward(RNNCell* cell, Tensor* input, Tensor* hidden) {
    if (!cell || !input)
        return NULL;

    int batch = input->shape[0];
    int hs    = cell->hidden_size;

    /* Initialise hidden to zeros when NULL */
    if (!hidden) {
        int h_shape[]    = {batch, hs};
        TensorConfig cfg = {
            .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
        hidden = tensor_zeros(h_shape, 2, &cfg);
    }

    /* h_new = tanh(input @ W_ih^T + hidden @ W_hh^T + b_ih + b_hh). Each
     * input/hidden projection is one fused LINEAR (x@W^T + b) rather than a
     * transpose + matmul + bias-add, cutting per-timestep op (and backward-VJP)
     * count -- the recurrent cost is dispatch, not arithmetic. */
    Tensor* bih = cell->bias_ih ? cell->bias_ih->tensor : NULL;
    Tensor* bhh = cell->bias_hh ? cell->bias_hh->tensor : NULL;
    Tensor* ih  = uop_linear(input, cell->weight_ih->tensor, bih);
    Tensor* hh  = uop_linear(hidden, cell->weight_hh->tensor, bhh);
    /* One fused kernel for tanh(ih+hh) and its backward, instead of add + tanh. */
    return uop_rnn_cell(ih, hh);
}

/** Register a recurrent cell's parameter set: weight_ih [gates, input_size],
 *  weight_hh [gates, hidden_size] and, when `use_bias`, the two [gates] biases.
 *  The RNN, LSTM and GRU cells differ only in how many gates they stack. Weights are
 *  initialized uniform in [-scale, scale], biases to zero. */
static void add_cell_params(Module* cell, int gates, int input_size, int hidden_size, float scale,
                            bool use_bias, TensorConfig* cfg, Parameter** weight_ih,
                            Parameter** weight_hh, Parameter** bias_ih, Parameter** bias_hh) {
    int wih_shape[] = {gates, input_size};
    Tensor* wih     = tensor_empty(wih_shape, 2, cfg);
    nn_init_uniform(wih, -scale, scale);
    module_add_parameter(cell, wih, "weight_ih", true);
    *weight_ih = module_get_parameter(cell, "weight_ih");

    int whh_shape[] = {gates, hidden_size};
    Tensor* whh     = tensor_empty(whh_shape, 2, cfg);
    nn_init_uniform(whh, -scale, scale);
    module_add_parameter(cell, whh, "weight_hh", true);
    *weight_hh = module_get_parameter(cell, "weight_hh");

    if (!use_bias) {
        *bias_ih = NULL;
        *bias_hh = NULL;
        return;
    }

    int b_shape[] = {gates};
    module_add_parameter(cell, tensor_zeros(b_shape, 1, cfg), "bias_ih", true);
    *bias_ih = module_get_parameter(cell, "bias_ih");
    module_add_parameter(cell, tensor_zeros(b_shape, 1, cfg), "bias_hh", true);
    *bias_hh = module_get_parameter(cell, "bias_hh");
}

/** Construct a torch.nn.RNNCell (single tanh gate). Returns NULL on failure. */
RNNCell* nn_rnn_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                     DeviceType device) {
    RNNCell* cell = cml_malloc(sizeof(RNNCell));
    if (!cell)
        return NULL;

    if (module_init((Module*)cell, "RNNCell", rnn_cell_module_forward, NULL) != 0) {
        cml_free(cell);
        return NULL;
    }

    cell->input_size  = input_size;
    cell->hidden_size = hidden_size;
    cell->use_bias    = use_bias;

    TensorConfig cfg = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    float scale      = 1.0f / sqrtf((float)hidden_size);

    add_cell_params((Module*)cell, hidden_size, input_size, hidden_size, scale, use_bias, &cfg,
                    &cell->weight_ih, &cell->weight_hh, &cell->bias_ih, &cell->bias_hh);

    return cell;
}

/** Module-interface stub: returns NULL because the generic forward cannot carry (h, c) state;
 *  call lstm_cell_forward() directly. */
static Tensor* lstm_cell_module_forward(Module* module, Tensor* input) {
    (void)module;
    (void)input;
    return NULL;
}

/** torch.nn.LSTMCell step: compute the four gates, then c' = f*c_prev + i*g and h' = o*tanh(c'),
 *  writing h'/c' to h_out/c_out. NULL h_prev/c_prev are treated as zeros. No-op on NULL args. */
void lstm_cell_forward(LSTMCell* cell, Tensor* input, Tensor* h_prev, Tensor* c_prev,
                       Tensor** h_out, Tensor** c_out) {
    if (!cell || !input || !h_out || !c_out)
        return;

    int batch = input->shape[0];
    int hs    = cell->hidden_size;
    (void)(4 * hs); /* gs unused after autograd rewrite */

    /* Initialise states to zeros when NULL */
    if (!h_prev) {
        int s[]          = {batch, hs};
        TensorConfig cfg = {
            .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
        h_prev = tensor_zeros(s, 2, &cfg);
    }
    if (!c_prev) {
        int s[]          = {batch, hs};
        TensorConfig cfg = {
            .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
        c_prev = tensor_zeros(s, 2, &cfg);
    }

    /* gates = input @ W_ih^T + h_prev @ W_hh^T + b_ih + b_hh. One fused LINEAR
     * per projection instead of transpose + matmul + bias-add. */
    Tensor* bih = cell->bias_ih ? cell->bias_ih->tensor : NULL;
    Tensor* bhh = cell->bias_hh ? cell->bias_hh->tensor : NULL;
    Tensor* gates =
        tensor_add(uop_linear(input, cell->weight_ih->tensor, bih),
                   uop_linear(h_prev, cell->weight_hh->tensor, bhh)); /* [batch, 4*hs] */

    /* One fused kernel for the whole gate computation (i/f/g/o -> c_new, h_new)
     * instead of 4 shrinks + sigmoid/tanh/mul/add; the backward is one fused op
     * too. The packed [B,2H] output is sliced back into h_new and c_new. */
    Tensor* packed = uop_lstm_cell(gates, c_prev);
    int sh_h[] = {0, 0}, eh_h[] = {batch, hs};
    int sh_c[] = {0, hs}, eh_c[] = {batch, 2 * hs};
    *h_out = uop_shrink(packed, sh_h, eh_h, 2);
    *c_out = uop_shrink(packed, sh_c, eh_c, 2);
}

/** Construct a torch.nn.LSTMCell (4 stacked gates). Returns NULL on failure. */
LSTMCell* nn_lstm_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                       DeviceType device) {
    LSTMCell* cell = cml_malloc(sizeof(LSTMCell));
    if (!cell)
        return NULL;

    if (module_init((Module*)cell, "LSTMCell", lstm_cell_module_forward, NULL) != 0) {
        cml_free(cell);
        return NULL;
    }

    cell->input_size  = input_size;
    cell->hidden_size = hidden_size;
    cell->use_bias    = use_bias;

    TensorConfig cfg = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    int gs           = 4 * hidden_size;
    float scale      = 1.0f / sqrtf((float)hidden_size);

    add_cell_params((Module*)cell, gs, input_size, hidden_size, scale, use_bias, &cfg,
                    &cell->weight_ih, &cell->weight_hh, &cell->bias_ih, &cell->bias_hh);

    return cell;
}

/** Module-interface stub: returns NULL because the generic forward cannot carry hidden state;
 *  call gru_cell_forward() directly. */
static Tensor* gru_cell_module_forward(Module* module, Tensor* input) {
    (void)module;
    (void)input;
    return NULL;
}

/** torch.nn.GRUCell step: reset/update/new gates giving h' = (1-z)*n + z*hidden. A NULL hidden is
 *  treated as zeros. Returns NULL on NULL cell/input. */
Tensor* gru_cell_forward(GRUCell* cell, Tensor* input, Tensor* hidden) {
    if (!cell || !input)
        return NULL;

    int batch = input->shape[0];
    int hs    = cell->hidden_size;

    /* Initialise hidden to zeros when NULL */
    if (!hidden) {
        int s[]          = {batch, hs};
        TensorConfig cfg = {
            .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
        hidden = tensor_zeros(s, 2, &cfg);
    }

    /* GRU forward using autograd ops so gradients flow to parameters.
     * ih = input @ W_ih^T + b_ih   [batch, 3*hs]
     * hh = hidden @ W_hh^T + b_hh  [batch, 3*hs] */
    /* One fused LINEAR per projection instead of transpose + matmul + bias-add. */
    Tensor* ih =
        uop_linear(input, cell->weight_ih->tensor, cell->bias_ih ? cell->bias_ih->tensor : NULL);
    Tensor* hh =
        uop_linear(hidden, cell->weight_hh->tensor, cell->bias_hh ? cell->bias_hh->tensor : NULL);

    /* One fused kernel for the whole gate computation (r/z/n + blend) instead
     * of 6 shrinks + sigmoid/tanh/mul/add; the backward is one fused op too. */
    return uop_gru_cell(ih, hh, hidden);
}

/** Construct a torch.nn.GRUCell (3 stacked gates). Returns NULL on failure. */
GRUCell* nn_gru_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                     DeviceType device) {
    GRUCell* cell = cml_malloc(sizeof(GRUCell));
    if (!cell)
        return NULL;

    if (module_init((Module*)cell, "GRUCell", gru_cell_module_forward, NULL) != 0) {
        cml_free(cell);
        return NULL;
    }

    cell->input_size  = input_size;
    cell->hidden_size = hidden_size;
    cell->use_bias    = use_bias;

    TensorConfig cfg = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    int gs           = 3 * hidden_size;
    float scale      = 1.0f / sqrtf((float)hidden_size);

    add_cell_params((Module*)cell, gs, input_size, hidden_size, scale, use_bias, &cfg,
                    &cell->weight_ih, &cell->weight_hh, &cell->bias_ih, &cell->bias_hh);

    return cell;
}

/** Re-export a cell's parameters on the parent RNN/LSTM/GRU under "layers.<layer>.<fwd|rev>.<name>"
 *  names, aliasing the tensors rather than copying. */
static void register_cell_params(Module* parent, Module* cell, int layer, int dir) {
    const char* dir_str = (dir == 0) ? "fwd" : "rev";
    Parameter** params  = NULL;
    int num_params      = 0;
    if (module_collect_parameters(cell, &params, &num_params, false) == 0) {
        for (int i = 0; i < num_params; i++) {
            if (params[i]) {
                char pname[256];
                snprintf(pname, sizeof(pname), "layers.%d.%s.%s", layer, dir_str,
                         params[i]->name ? params[i]->name : "unnamed");
                Tensor* pt = params[i]->tensor;
                nn_tensor_param_alias(pt);
                if (module_add_parameter(parent, pt, pname, params[i]->requires_grad) != 0)
                    pt->ref_count--;
            }
        }
        if (params)
            cml_free(params);
    }
}

/** Lazy slice: [seq, batch, feat] -> [batch, feat] at time step t. NULL on op failure. */
static Tensor* slice_timestep(Tensor* src, int t) {
    int batch      = src->shape[1];
    int feat       = src->shape[2];
    int starts[]   = {t, 0, 0};
    int ends[]     = {t + 1, batch, feat};
    int steps[]    = {1, 1, 1};
    SliceParams sp = {.start = starts, .end = ends, .step = steps, .num_dims = 3};
    Tensor* sliced = uop_slice(src, &sp); /* [1, batch, feat] */
    if (!sliced)
        return NULL;
    int shape2[]     = {batch, feat};
    ReshapeParams rp = {.new_shape = shape2, .new_ndim = 2};
    return uop_reshape(sliced, &rp); /* [batch, feat] */
}

/** Lazy transpose of dims 0 and 1 of a 3-D tensor (batch_first <-> seq_first). */
static Tensor* transpose_01(Tensor* src) {
    int perm[]       = {1, 0, 2};
    PermuteParams pp = {.perm = perm, .num_dims = 3};
    return uop_permute(src, &pp);
}

/** Lazy concatenation of forward and reverse outputs along the feature axis. */
static Tensor* concat_features(Tensor* a, Tensor* b) {
    Tensor* inputs[] = {a, b};
    return uop_cat(inputs, 2, 2); /* cat along dim 2 (feature) */
}

/** Module-interface forward: run rnn_forward with a zero initial state and return just the
 *  sequence output (h_n is discarded, like torch's second return value). */
static Tensor* rnn_module_forward(Module* module, Tensor* input) {
    /* Module interface returns the sequence output; h_n is an extra graph output
     * (left for the graph teardown, like torch's second return value). */
    Tensor* output = NULL;
    Tensor* h_n    = NULL;
    rnn_forward((RNN*)module, input, NULL, &output, &h_n);
    return output;
}

/** Free the RNN module and every per-layer/direction cell it owns. */
static void rnn_free(Module* module) {
    RNN* rnn = (RNN*)module;
    if (rnn->cells) {
        int total = rnn->num_layers * rnn->num_directions;
        for (int i = 0; i < total; i++) {
            if (rnn->cells[i]) {
                module_free((Module*)rnn->cells[i]);
            }
        }
        cml_free(rnn->cells);
    }
    cml_free(rnn);
}

/** Construct a multi-layer (optionally bidirectional) torch.nn.RNN, building one RNNCell per
 *  layer/direction and registering their parameters. Returns NULL on failure. */
RNN* nn_rnn(int input_size, int hidden_size, int num_layers, bool bidirectional, bool batch_first,
            float dropout, bool use_bias, DType dtype, DeviceType device) {
    RNN* rnn = cml_malloc(sizeof(RNN));
    if (!rnn)
        return NULL;

    if (module_init((Module*)rnn, "RNN", rnn_module_forward, rnn_free) != 0) {
        cml_free(rnn);
        return NULL;
    }

    rnn->input_size     = input_size;
    rnn->hidden_size    = hidden_size;
    rnn->num_layers     = num_layers;
    rnn->bidirectional  = bidirectional;
    rnn->batch_first    = batch_first;
    rnn->dropout_p      = dropout;
    rnn->use_bias       = use_bias;
    rnn->dtype          = dtype;
    rnn->device         = device;
    rnn->num_directions = bidirectional ? 2 : 1;

    int total  = num_layers * rnn->num_directions;
    rnn->cells = cml_calloc((size_t)total, sizeof(RNNCell*));
    if (!rnn->cells) {
        cml_free(rnn);
        return NULL;
    }

    for (int l = 0; l < num_layers; l++) {
        int cell_input = (l == 0) ? input_size : hidden_size * rnn->num_directions;
        for (int d = 0; d < rnn->num_directions; d++) {
            int idx         = l * rnn->num_directions + d;
            rnn->cells[idx] = nn_rnn_cell(cell_input, hidden_size, use_bias, dtype, device);
            if (!rnn->cells[idx]) {
                rnn_free((Module*)rnn);
                return NULL;
            }
            register_cell_params((Module*)rnn, (Module*)rnn->cells[idx], l, d);
        }
    }
    return rnn;
}

/** Run the full RNN over a sequence: stack per-timestep hidden states into `output` and the final
 *  per-layer/direction states into `h_n`. Honors batch_first and bidirectional. No-op on NULL
 *  args; a NULL h_0 starts from zeros. */
void rnn_forward(RNN* rnn, Tensor* input, Tensor* h_0, Tensor** output, Tensor** h_n) {
    if (!rnn || !input || !output || !h_n)
        return;

    int nd = rnn->num_directions;

    Tensor* x = rnn->batch_first ? transpose_01(input) : input;

    int seq_len    = x->shape[0];
    int total_dirs = rnn->num_layers * nd;

    Tensor** final_h = calloc((size_t)total_dirs, sizeof(Tensor*));
    if (!final_h)
        return;
    Tensor* layer_input = x;

    for (int l = 0; l < rnn->num_layers; l++) {
        RNNCell* fwd_cell = rnn->cells[l * nd + 0];

        Tensor* h_fwd      = h_0 ? slice_timestep(h_0, l * nd + 0) : NULL;
        Tensor** fwd_steps = malloc((size_t)seq_len * sizeof(Tensor*));
        if (!fwd_steps) {
            free(final_h);
            return;
        }

        for (int t = 0; t < seq_len; t++) {
            Tensor* xt   = slice_timestep(layer_input, t);
            h_fwd        = rnn_cell_forward(fwd_cell, xt, h_fwd);
            fwd_steps[t] = h_fwd;
        }
        final_h[l * nd + 0] = h_fwd;

        Tensor* fwd_out = uop_stack(fwd_steps, seq_len, 0);
        cml_free(fwd_steps);

        Tensor* layer_output;
        if (nd == 2) {
            RNNCell* rev_cell  = rnn->cells[l * nd + 1];
            Tensor* h_rev      = h_0 ? slice_timestep(h_0, l * nd + 1) : NULL;
            Tensor** rev_steps = malloc((size_t)seq_len * sizeof(Tensor*));
            if (!rev_steps) {
                free(fwd_steps);
                free(final_h);
                return;
            }

            for (int t = seq_len - 1; t >= 0; t--) {
                Tensor* xt   = slice_timestep(layer_input, t);
                h_rev        = rnn_cell_forward(rev_cell, xt, h_rev);
                rev_steps[t] = h_rev;
            }
            final_h[l * nd + 1] = h_rev;

            Tensor* rev_out = uop_stack(rev_steps, seq_len, 0);
            cml_free(rev_steps);

            layer_output = concat_features(fwd_out, rev_out);
        } else {
            layer_output = fwd_out;
        }

        layer_input = layer_output;
    }

    Tensor* hn = uop_stack(final_h, total_dirs, 0);
    cml_free(final_h);

    if (rnn->batch_first)
        layer_input = transpose_01(layer_input);

    *output = layer_input;
    *h_n    = hn;
}

/** Module-interface forward: run lstm_forward with zero initial states and return just the
 *  sequence output (h_n, c_n discarded). */
static Tensor* lstm_module_forward(Module* module, Tensor* input) {
    Tensor* output = NULL;
    Tensor* h_n    = NULL;
    Tensor* c_n    = NULL;
    lstm_forward((LSTM*)module, input, NULL, NULL, &output, &h_n, &c_n);
    return output;
}

/** Free the LSTM module and every per-layer/direction cell it owns. */
static void lstm_free(Module* module) {
    LSTM* lstm = (LSTM*)module;
    if (lstm->cells) {
        int total = lstm->num_layers * lstm->num_directions;
        for (int i = 0; i < total; i++) {
            if (lstm->cells[i]) {
                module_free((Module*)lstm->cells[i]);
            }
        }
        cml_free(lstm->cells);
    }
    cml_free(lstm);
}

/** Construct a multi-layer (optionally bidirectional) torch.nn.LSTM, building one LSTMCell per
 *  layer/direction and registering their parameters. Returns NULL on failure. */
LSTM* nn_lstm(int input_size, int hidden_size, int num_layers, bool bidirectional, bool batch_first,
              float dropout, bool use_bias, DType dtype, DeviceType device) {
    LSTM* lstm = cml_malloc(sizeof(LSTM));
    if (!lstm)
        return NULL;

    if (module_init((Module*)lstm, "LSTM", lstm_module_forward, lstm_free) != 0) {
        cml_free(lstm);
        return NULL;
    }

    lstm->input_size     = input_size;
    lstm->hidden_size    = hidden_size;
    lstm->num_layers     = num_layers;
    lstm->bidirectional  = bidirectional;
    lstm->batch_first    = batch_first;
    lstm->dropout_p      = dropout;
    lstm->use_bias       = use_bias;
    lstm->dtype          = dtype;
    lstm->device         = device;
    lstm->num_directions = bidirectional ? 2 : 1;

    int total   = num_layers * lstm->num_directions;
    lstm->cells = cml_calloc((size_t)total, sizeof(LSTMCell*));
    if (!lstm->cells) {
        cml_free(lstm);
        return NULL;
    }

    for (int l = 0; l < num_layers; l++) {
        int cell_input = (l == 0) ? input_size : hidden_size * lstm->num_directions;
        for (int d = 0; d < lstm->num_directions; d++) {
            int idx          = l * lstm->num_directions + d;
            lstm->cells[idx] = nn_lstm_cell(cell_input, hidden_size, use_bias, dtype, device);
            if (!lstm->cells[idx]) {
                lstm_free((Module*)lstm);
                return NULL;
            }
            register_cell_params((Module*)lstm, (Module*)lstm->cells[idx], l, d);
        }
    }
    return lstm;
}

/** Run the full LSTM over a sequence: stack per-timestep hidden states into `output` and the final
 *  per-layer/direction (h, c) into `h_n`/`c_n`. Honors batch_first and bidirectional. No-op on NULL
 *  args; NULL h_0/c_0 start from zeros. */
void lstm_forward(LSTM* lstm, Tensor* input, Tensor* h_0, Tensor* c_0, Tensor** output,
                  Tensor** h_n, Tensor** c_n) {
    if (!lstm || !input || !output || !h_n || !c_n)
        return;

    int nd         = lstm->num_directions;
    int total_dirs = lstm->num_layers * nd;

    Tensor* x = lstm->batch_first ? transpose_01(input) : input;

    int seq_len = x->shape[0];

    Tensor** final_h    = cml_calloc((size_t)total_dirs, sizeof(Tensor*));
    Tensor** final_c    = cml_calloc((size_t)total_dirs, sizeof(Tensor*));
    Tensor* layer_input = x;

    for (int l = 0; l < lstm->num_layers; l++) {
        LSTMCell* fwd_cell = lstm->cells[l * nd + 0];

        Tensor* h_fwd      = h_0 ? slice_timestep(h_0, l * nd + 0) : NULL;
        Tensor* c_fwd      = c_0 ? slice_timestep(c_0, l * nd + 0) : NULL;
        Tensor** fwd_steps = cml_malloc((size_t)seq_len * sizeof(Tensor*));

        for (int t = 0; t < seq_len; t++) {
            Tensor* xt    = slice_timestep(layer_input, t);
            Tensor* h_new = NULL;
            Tensor* c_new = NULL;
            lstm_cell_forward(fwd_cell, xt, h_fwd, c_fwd, &h_new, &c_new);
            /* On cell failure keep the last valid state rather than letting a
             * NULL propagate (which would restart the sequence from zeros and
             * also feed uop_stack a NULL entry). */
            if (h_new)
                h_fwd = h_new;
            if (c_new)
                c_fwd = c_new;
            fwd_steps[t] = h_fwd;
        }
        final_h[l * nd + 0] = h_fwd;
        final_c[l * nd + 0] = c_fwd;

        Tensor* fwd_out = uop_stack(fwd_steps, seq_len, 0);
        cml_free(fwd_steps);

        Tensor* layer_output;
        if (nd == 2) {
            LSTMCell* rev_cell = lstm->cells[l * nd + 1];
            Tensor* h_rev      = h_0 ? slice_timestep(h_0, l * nd + 1) : NULL;
            Tensor* c_rev      = c_0 ? slice_timestep(c_0, l * nd + 1) : NULL;
            Tensor** rev_steps = cml_malloc((size_t)seq_len * sizeof(Tensor*));

            for (int t = seq_len - 1; t >= 0; t--) {
                Tensor* xt    = slice_timestep(layer_input, t);
                Tensor* h_new = NULL;
                Tensor* c_new = NULL;
                lstm_cell_forward(rev_cell, xt, h_rev, c_rev, &h_new, &c_new);
                if (h_new)
                    h_rev = h_new;
                if (c_new)
                    c_rev = c_new;
                rev_steps[t] = h_rev;
            }
            final_h[l * nd + 1] = h_rev;
            final_c[l * nd + 1] = c_rev;

            Tensor* rev_out = uop_stack(rev_steps, seq_len, 0);
            cml_free(rev_steps);

            layer_output = concat_features(fwd_out, rev_out);
        } else {
            layer_output = fwd_out;
        }

        layer_input = layer_output;
    }

    Tensor* hn = uop_stack(final_h, total_dirs, 0);
    Tensor* cn = uop_stack(final_c, total_dirs, 0);
    cml_free(final_h);
    cml_free(final_c);

    if (lstm->batch_first)
        layer_input = transpose_01(layer_input);

    *output = layer_input;
    *h_n    = hn;
    *c_n    = cn;
}

/** Module-interface forward: run gru_forward with a zero initial state and return just the
 *  sequence output (h_n discarded). */
static Tensor* gru_module_forward(Module* module, Tensor* input) {
    Tensor* output = NULL;
    Tensor* h_n    = NULL;
    gru_forward((GRU*)module, input, NULL, &output, &h_n);
    return output;
}

/** Free the GRU module and every per-layer/direction cell it owns. */
static void gru_free(Module* module) {
    GRU* gru = (GRU*)module;
    if (gru->cells) {
        int total = gru->num_layers * gru->num_directions;
        for (int i = 0; i < total; i++) {
            if (gru->cells[i]) {
                module_free((Module*)gru->cells[i]);
            }
        }
        cml_free(gru->cells);
    }
    cml_free(gru);
}

/** Construct a multi-layer (optionally bidirectional) torch.nn.GRU, building one GRUCell per
 *  layer/direction and registering their parameters. Returns NULL on failure. */
GRU* nn_gru(int input_size, int hidden_size, int num_layers, bool bidirectional, bool batch_first,
            float dropout, bool use_bias, DType dtype, DeviceType device) {
    GRU* gru = cml_malloc(sizeof(GRU));
    if (!gru)
        return NULL;

    if (module_init((Module*)gru, "GRU", gru_module_forward, gru_free) != 0) {
        cml_free(gru);
        return NULL;
    }

    gru->input_size     = input_size;
    gru->hidden_size    = hidden_size;
    gru->num_layers     = num_layers;
    gru->bidirectional  = bidirectional;
    gru->batch_first    = batch_first;
    gru->dropout_p      = dropout;
    gru->use_bias       = use_bias;
    gru->dtype          = dtype;
    gru->device         = device;
    gru->num_directions = bidirectional ? 2 : 1;

    int total  = num_layers * gru->num_directions;
    gru->cells = cml_calloc((size_t)total, sizeof(GRUCell*));
    if (!gru->cells) {
        cml_free(gru);
        return NULL;
    }

    for (int l = 0; l < num_layers; l++) {
        int cell_input = (l == 0) ? input_size : hidden_size * gru->num_directions;
        for (int d = 0; d < gru->num_directions; d++) {
            int idx         = l * gru->num_directions + d;
            gru->cells[idx] = nn_gru_cell(cell_input, hidden_size, use_bias, dtype, device);
            if (!gru->cells[idx]) {
                gru_free((Module*)gru);
                return NULL;
            }
            register_cell_params((Module*)gru, (Module*)gru->cells[idx], l, d);
        }
    }
    return gru;
}

/** Run the full GRU over a sequence: stack per-timestep hidden states into `output` and the final
 *  per-layer/direction states into `h_n`. Honors batch_first and bidirectional. No-op on NULL
 *  args; a NULL h_0 starts from zeros. */
void gru_forward(GRU* gru, Tensor* input, Tensor* h_0, Tensor** output, Tensor** h_n) {
    if (!gru || !input || !output || !h_n)
        return;

    int nd         = gru->num_directions;
    int total_dirs = gru->num_layers * nd;

    Tensor* x = gru->batch_first ? transpose_01(input) : input;

    int seq_len = x->shape[0];

    Tensor** final_h = cml_calloc((size_t)total_dirs, sizeof(Tensor*));
    if (!final_h)
        return;
    Tensor* layer_input = x;

    for (int l = 0; l < gru->num_layers; l++) {
        GRUCell* fwd_cell = gru->cells[l * nd + 0];

        Tensor* h_fwd      = h_0 ? slice_timestep(h_0, l * nd + 0) : NULL;
        Tensor** fwd_steps = cml_malloc((size_t)seq_len * sizeof(Tensor*));
        if (!fwd_steps) {
            cml_free(final_h);
            return;
        }

        for (int t = 0; t < seq_len; t++) {
            Tensor* xt   = slice_timestep(layer_input, t);
            h_fwd        = gru_cell_forward(fwd_cell, xt, h_fwd);
            fwd_steps[t] = h_fwd;
        }
        final_h[l * nd + 0] = h_fwd;

        Tensor* fwd_out = uop_stack(fwd_steps, seq_len, 0);
        cml_free(fwd_steps);

        Tensor* layer_output;
        if (nd == 2) {
            GRUCell* rev_cell  = gru->cells[l * nd + 1];
            Tensor* h_rev      = h_0 ? slice_timestep(h_0, l * nd + 1) : NULL;
            Tensor** rev_steps = cml_malloc((size_t)seq_len * sizeof(Tensor*));
            if (!rev_steps) {
                cml_free(final_h);
                return;
            }

            for (int t = seq_len - 1; t >= 0; t--) {
                Tensor* xt   = slice_timestep(layer_input, t);
                h_rev        = gru_cell_forward(rev_cell, xt, h_rev);
                rev_steps[t] = h_rev;
            }
            final_h[l * nd + 1] = h_rev;

            Tensor* rev_out = uop_stack(rev_steps, seq_len, 0);
            cml_free(rev_steps);

            layer_output = concat_features(fwd_out, rev_out);
        } else {
            layer_output = fwd_out;
        }

        layer_input = layer_output;
    }

    Tensor* hn = uop_stack(final_h, total_dirs, 0);
    cml_free(final_h);

    if (gru->batch_first)
        layer_input = transpose_01(layer_input);

    *output = layer_input;
    *h_n    = hn;
}
