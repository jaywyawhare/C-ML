#include "nn/layers/transformer.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

static void xavier_init(float* data, size_t numel, int fan_in, int fan_out) {
    float scale = sqrtf(2.0f / (float)(fan_in + fan_out));
    for (size_t i = 0; i < numel; i++) {
        data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

/* Compose LayerNorm from primitives so it records into the IR (fully lazy). */
static Tensor* apply_layernorm(Tensor* x, Tensor* weight, Tensor* bias, float eps) {
    if (!x) return NULL;
    int last_dim = x->ndim - 1;
    ReduceParams rp = {.dims = &last_dim, .num_dims = 1, .keepdim = true};

    Tensor* mean     = uop_mean(x, &rp);
    if (!mean) return NULL;
    Tensor* centered = uop_sub(x, mean);
    if (!centered) return NULL;
    Tensor* sq       = uop_mul(centered, centered);
    if (!sq) return NULL;
    Tensor* var      = uop_mean(sq, &rp);
    if (!var) return NULL;

    TensorConfig cfg = {.dtype = x->dtype, .device = x->device,
                        .has_dtype = true, .has_device = true};
    Tensor* eps_t = tensor_full(var->shape, var->ndim, &cfg, eps);
    if (!eps_t) return NULL;

    Tensor* std = uop_sqrt(uop_add(var, eps_t));
    if (!std) return NULL;
    Tensor* norm = uop_div(centered, std);
    if (!norm) return NULL;

    if (weight && bias) {
        ExpandParams ep = {.new_shape = x->shape, .new_ndim = x->ndim};
        Tensor* w_b = uop_expand(weight, &ep);
        Tensor* b_b = uop_expand(bias, &ep);
        if (!w_b || !b_b) return NULL;
        Tensor* scaled = uop_mul(w_b, norm);
        if (!scaled) return NULL;
        return uop_add(scaled, b_b);
    }
    return norm;
}

static Tensor* mha_module_forward(Module* module, Tensor* input) {
    MultiHeadAttention* mha = (MultiHeadAttention*)module;
    return multihead_attention_forward(mha, input, input, input, NULL);
}

static void mha_free(Module* module) {
    MultiHeadAttention* mha = (MultiHeadAttention*)module;
    if (!mha) return;
    free(mha);
}

MultiHeadAttention* nn_multihead_attention(int embed_dim, int num_heads, float dropout,
                                            DType dtype, DeviceType device) {
    if (embed_dim % num_heads != 0) {
        LOG_ERROR("embed_dim (%d) must be divisible by num_heads (%d)", embed_dim, num_heads);
        return NULL;
    }

    MultiHeadAttention* mha = malloc(sizeof(MultiHeadAttention));
    if (!mha) {
        LOG_ERROR("Failed to allocate MultiHeadAttention");
        return NULL;
    }

    if (module_init((Module*)mha, "MultiHeadAttention", mha_module_forward, mha_free) != 0) {
        free(mha);
        return NULL;
    }

    mha->embed_dim = embed_dim;
    mha->num_heads = num_heads;
    mha->head_dim = embed_dim / num_heads;
    mha->dropout = dropout;

    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};

    int weight_shape[] = {embed_dim, embed_dim};
    int bias_shape[] = {embed_dim};

    struct { const char* w_name; const char* b_name; Parameter** w_ptr; Parameter** b_ptr; } params[] = {
        {"W_q", "b_q", &mha->W_q, &mha->b_q},
        {"W_k", "b_k", &mha->W_k, &mha->b_k},
        {"W_v", "b_v", &mha->W_v, &mha->b_v},
        {"W_o", "b_o", &mha->W_o, &mha->b_o},
    };

    for (int i = 0; i < 4; i++) {
        Tensor* w = tensor_empty(weight_shape, 2, &config);
        if (!w) { module_free((Module*)mha); return NULL; }
        float* w_data = (float*)tensor_data_ptr(w);
        if (w_data) xavier_init(w_data, (size_t)embed_dim * embed_dim, embed_dim, embed_dim);

        if (module_add_parameter((Module*)mha, w, params[i].w_name, true) != 0) {
            tensor_free(w); module_free((Module*)mha); return NULL;
        }
        *params[i].w_ptr = module_get_parameter((Module*)mha, params[i].w_name);

        Tensor* b = tensor_zeros(bias_shape, 1, &config);
        if (!b) { module_free((Module*)mha); return NULL; }

        if (module_add_parameter((Module*)mha, b, params[i].b_name, true) != 0) {
            tensor_free(b); module_free((Module*)mha); return NULL;
        }
        *params[i].b_ptr = module_get_parameter((Module*)mha, params[i].b_name);
    }

    return mha;
}

Tensor* multihead_attention_forward(MultiHeadAttention* mha, Tensor* query, Tensor* key,
                                     Tensor* value, Tensor* mask) {
    if (!mha || !query || !key || !value) {
        LOG_ERROR("MultiHeadAttention forward: NULL input");
        return NULL;
    }
    if (query->ndim != 3 || key->ndim != 3 || value->ndim != 3) {
        LOG_ERROR("MultiHeadAttention forward: expected 3D inputs [batch, seq, embed_dim]");
        return NULL;
    }

    int batch     = query->shape[0];
    int seq_q     = query->shape[1];
    int seq_k     = key->shape[1];
    int embed_dim = mha->embed_dim;
    int num_heads = mha->num_heads;
    int head_dim  = mha->head_dim;

    /* Linear projections: [B, S, E] */
    Tensor* Q = uop_linear(query, mha->W_q->tensor, mha->b_q->tensor);
    Tensor* K = uop_linear(key,   mha->W_k->tensor, mha->b_k->tensor);
    Tensor* V = uop_linear(value, mha->W_v->tensor, mha->b_v->tensor);
    if (!Q || !K || !V) return NULL;

    /* Reshape [B, S, E] -> [B, S, H, D] -> permute -> [B, H, S, D] */
    int q_shape4[] = {batch, seq_q, num_heads, head_dim};
    ReshapeParams qrp = {.new_shape = q_shape4, .new_ndim = 4};
    int k_shape4[] = {batch, seq_k, num_heads, head_dim};
    ReshapeParams krp = {.new_shape = k_shape4, .new_ndim = 4};
    int v_shape4[] = {batch, seq_k, num_heads, head_dim};
    ReshapeParams vrp = {.new_shape = v_shape4, .new_ndim = 4};

    Tensor* Q_r = uop_reshape(Q, &qrp);
    Tensor* K_r = uop_reshape(K, &krp);
    Tensor* V_r = uop_reshape(V, &vrp);
    if (!Q_r || !K_r || !V_r) return NULL;

    int perm4[] = {0, 2, 1, 3};
    PermuteParams pp4 = {.perm = perm4, .num_dims = 4};
    Tensor* Q_h = uop_permute(Q_r, &pp4);
    Tensor* K_h = uop_permute(K_r, &pp4);
    Tensor* V_h = uop_permute(V_r, &pp4);
    if (!Q_h || !K_h || !V_h) return NULL;

    /* Scaled dot-product attention: [B, H, S_q, D] */
    Tensor* attn_out = uop_scaled_dot_product_attention(Q_h, K_h, V_h, mask);
    if (!attn_out) return NULL;

    /* [B, H, S_q, D] -> [B, S_q, H, D] -> [B, S_q, E] */
    Tensor* attn_t = uop_permute(attn_out, &pp4); /* {0,2,1,3} is its own inverse */
    if (!attn_t) return NULL;

    int out_shape3[] = {batch, seq_q, embed_dim};
    ReshapeParams orp = {.new_shape = out_shape3, .new_ndim = 3};
    Tensor* concat = uop_reshape(attn_t, &orp);
    if (!concat) return NULL;

    return uop_linear(concat, mha->W_o->tensor, mha->b_o->tensor);
}

static Tensor* encoder_layer_forward(Module* module, Tensor* input) {
    TransformerEncoderLayer* layer = (TransformerEncoderLayer*)module;
    if (!layer || !input) return NULL;
    if (input->ndim != 3) {
        LOG_ERROR("TransformerEncoderLayer forward: expected 3D input [batch, seq, d_model]");
        return NULL;
    }

    /* Self-attention + residual + LN1 */
    Tensor* attn = multihead_attention_forward(layer->self_attn, input, input, input, NULL);
    if (!attn) return NULL;
    Tensor* x1 = uop_add(input, attn);
    if (!x1) return NULL;
    Tensor* x1_ln = apply_layernorm(x1,
                                    layer->norm1_weight->tensor,
                                    layer->norm1_bias->tensor,
                                    layer->norm_eps);
    if (!x1_ln) return NULL;

    /* FFN: Linear1 -> ReLU -> Linear2 */
    Tensor* ff1 = uop_linear(x1_ln,
                             layer->linear1_weight->tensor,
                             layer->linear1_bias->tensor);
    if (!ff1) return NULL;
    Tensor* ff1_act = uop_relu(ff1);
    if (!ff1_act) return NULL;
    Tensor* ff2 = uop_linear(ff1_act,
                             layer->linear2_weight->tensor,
                             layer->linear2_bias->tensor);
    if (!ff2) return NULL;

    /* Residual + LN2 */
    Tensor* x2 = uop_add(x1_ln, ff2);
    if (!x2) return NULL;
    return apply_layernorm(x2,
                           layer->norm2_weight->tensor,
                           layer->norm2_bias->tensor,
                           layer->norm_eps);
}

static void encoder_layer_free(Module* module) {
    TransformerEncoderLayer* layer = (TransformerEncoderLayer*)module;
    if (!layer) return;
    if (layer->self_attn) module_free((Module*)layer->self_attn);
    free(layer);
}

TransformerEncoderLayer* nn_transformer_encoder_layer(int d_model, int nhead, int dim_feedforward,
                                                       float dropout, DType dtype, DeviceType device) {
    TransformerEncoderLayer* layer = malloc(sizeof(TransformerEncoderLayer));
    if (!layer) {
        LOG_ERROR("Failed to allocate TransformerEncoderLayer");
        return NULL;
    }

    if (module_init((Module*)layer, "TransformerEncoderLayer", encoder_layer_forward, encoder_layer_free) != 0) {
        free(layer);
        return NULL;
    }

    layer->d_model = d_model;
    layer->nhead = nhead;
    layer->dim_feedforward = dim_feedforward;
    layer->dropout = dropout;
    layer->norm_eps = 1e-5f;

    layer->self_attn = nn_multihead_attention(d_model, nhead, dropout, dtype, device);
    if (!layer->self_attn) { module_free((Module*)layer); return NULL; }

    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};

    int l1_w_shape[] = {dim_feedforward, d_model};
    int l1_b_shape[] = {dim_feedforward};
    int l2_w_shape[] = {d_model, dim_feedforward};
    int l2_b_shape[] = {d_model};
    int norm_shape[] = {d_model};

    Tensor* l1w = tensor_empty(l1_w_shape, 2, &config);
    if (!l1w) { module_free((Module*)layer); return NULL; }
    float* l1w_data = (float*)tensor_data_ptr(l1w);
    if (l1w_data) xavier_init(l1w_data, (size_t)dim_feedforward * d_model, d_model, dim_feedforward);
    if (module_add_parameter((Module*)layer, l1w, "linear1_weight", true) != 0) {
        tensor_free(l1w); module_free((Module*)layer); return NULL;
    }
    layer->linear1_weight = module_get_parameter((Module*)layer, "linear1_weight");

    Tensor* l1b = tensor_zeros(l1_b_shape, 1, &config);
    if (!l1b) { module_free((Module*)layer); return NULL; }
    if (module_add_parameter((Module*)layer, l1b, "linear1_bias", true) != 0) {
        tensor_free(l1b); module_free((Module*)layer); return NULL;
    }
    layer->linear1_bias = module_get_parameter((Module*)layer, "linear1_bias");

    Tensor* l2w = tensor_empty(l2_w_shape, 2, &config);
    if (!l2w) { module_free((Module*)layer); return NULL; }
    float* l2w_data = (float*)tensor_data_ptr(l2w);
    if (l2w_data) xavier_init(l2w_data, (size_t)d_model * dim_feedforward, dim_feedforward, d_model);
    if (module_add_parameter((Module*)layer, l2w, "linear2_weight", true) != 0) {
        tensor_free(l2w); module_free((Module*)layer); return NULL;
    }
    layer->linear2_weight = module_get_parameter((Module*)layer, "linear2_weight");

    Tensor* l2b = tensor_zeros(l2_b_shape, 1, &config);
    if (!l2b) { module_free((Module*)layer); return NULL; }
    if (module_add_parameter((Module*)layer, l2b, "linear2_bias", true) != 0) {
        tensor_free(l2b); module_free((Module*)layer); return NULL;
    }
    layer->linear2_bias = module_get_parameter((Module*)layer, "linear2_bias");

    Tensor* n1w = tensor_ones(norm_shape, 1, &config);
    if (!n1w) { module_free((Module*)layer); return NULL; }
    if (module_add_parameter((Module*)layer, n1w, "norm1_weight", true) != 0) {
        tensor_free(n1w); module_free((Module*)layer); return NULL;
    }
    layer->norm1_weight = module_get_parameter((Module*)layer, "norm1_weight");

    Tensor* n1b = tensor_zeros(norm_shape, 1, &config);
    if (!n1b) { module_free((Module*)layer); return NULL; }
    if (module_add_parameter((Module*)layer, n1b, "norm1_bias", true) != 0) {
        tensor_free(n1b); module_free((Module*)layer); return NULL;
    }
    layer->norm1_bias = module_get_parameter((Module*)layer, "norm1_bias");

    Tensor* n2w = tensor_ones(norm_shape, 1, &config);
    if (!n2w) { module_free((Module*)layer); return NULL; }
    if (module_add_parameter((Module*)layer, n2w, "norm2_weight", true) != 0) {
        tensor_free(n2w); module_free((Module*)layer); return NULL;
    }
    layer->norm2_weight = module_get_parameter((Module*)layer, "norm2_weight");

    Tensor* n2b = tensor_zeros(norm_shape, 1, &config);
    if (!n2b) { module_free((Module*)layer); return NULL; }
    if (module_add_parameter((Module*)layer, n2b, "norm2_bias", true) != 0) {
        tensor_free(n2b); module_free((Module*)layer); return NULL;
    }
    layer->norm2_bias = module_get_parameter((Module*)layer, "norm2_bias");

    return layer;
}

static Tensor* transformer_encoder_forward(Module* module, Tensor* input) {
    TransformerEncoder* enc = (TransformerEncoder*)module;
    if (!enc || !input) return NULL;

    Tensor* x = input;
    bool owns_x = false;

    for (int i = 0; i < enc->num_layers; i++) {
        Tensor* out = module_forward((Module*)enc->layers[i], x);
        if (owns_x) tensor_free(x);
        if (!out) return NULL;
        x = out;
        owns_x = true;
    }

    Tensor* out = apply_layernorm(x,
                                  enc->norm_weight->tensor,
                                  enc->norm_bias->tensor,
                                  enc->norm_eps);
    if (owns_x) tensor_free(x);
    return out;
}

static void transformer_encoder_free(Module* module) {
    TransformerEncoder* enc = (TransformerEncoder*)module;
    if (!enc) return;
    if (enc->layers) {
        for (int i = 0; i < enc->num_layers; i++) {
            if (enc->layers[i]) module_free((Module*)enc->layers[i]);
        }
        free(enc->layers);
    }
    free(enc);
}

TransformerEncoder* nn_transformer_encoder(int d_model, int nhead, int dim_feedforward,
                                            float dropout, int num_layers,
                                            DType dtype, DeviceType device) {
    TransformerEncoder* enc = malloc(sizeof(TransformerEncoder));
    if (!enc) return NULL;

    if (module_init((Module*)enc, "TransformerEncoder", transformer_encoder_forward, transformer_encoder_free) != 0) {
        free(enc); return NULL;
    }

    enc->d_model = d_model;
    enc->num_layers = num_layers;
    enc->norm_eps = 1e-5f;

    enc->layers = malloc(num_layers * sizeof(TransformerEncoderLayer*));
    if (!enc->layers) { module_free((Module*)enc); return NULL; }

    for (int i = 0; i < num_layers; i++) {
        enc->layers[i] = nn_transformer_encoder_layer(d_model, nhead, dim_feedforward, dropout, dtype, device);
        if (!enc->layers[i]) {
            enc->num_layers = i;
            module_free((Module*)enc);
            return NULL;
        }
    }

    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    int norm_shape[] = {d_model};

    Tensor* nw = tensor_ones(norm_shape, 1, &config);
    if (!nw) { module_free((Module*)enc); return NULL; }
    if (module_add_parameter((Module*)enc, nw, "norm_weight", true) != 0) {
        tensor_free(nw); module_free((Module*)enc); return NULL;
    }
    enc->norm_weight = module_get_parameter((Module*)enc, "norm_weight");

    Tensor* nb = tensor_zeros(norm_shape, 1, &config);
    if (!nb) { module_free((Module*)enc); return NULL; }
    if (module_add_parameter((Module*)enc, nb, "norm_bias", true) != 0) {
        tensor_free(nb); module_free((Module*)enc); return NULL;
    }
    enc->norm_bias = module_get_parameter((Module*)enc, "norm_bias");

    return enc;
}

static Tensor* decoder_layer_forward_wrapper(Module* module, Tensor* input) {
    TransformerDecoderLayer* layer = (TransformerDecoderLayer*)module;
    return transformer_decoder_layer_forward(layer, input, NULL, NULL, NULL);
}

Tensor* transformer_decoder_layer_forward(TransformerDecoderLayer* layer, Tensor* tgt,
                                           Tensor* memory, Tensor* tgt_mask, Tensor* memory_mask) {
    if (!layer || !tgt) return NULL;
    if (tgt->ndim != 3) return NULL;

    /* Self-attention + residual + LN1 */
    Tensor* self_attn = multihead_attention_forward(layer->self_attn, tgt, tgt, tgt, tgt_mask);
    if (!self_attn) return NULL;
    Tensor* x1 = uop_add(tgt, self_attn);
    if (!x1) return NULL;
    Tensor* x = apply_layernorm(x1,
                                layer->norm1_weight->tensor,
                                layer->norm1_bias->tensor,
                                layer->norm_eps);
    if (!x) return NULL;

    /* Cross-attention + residual + LN2 (when memory provided) */
    if (memory) {
        Tensor* cross_attn = multihead_attention_forward(layer->cross_attn, x, memory, memory, memory_mask);
        if (!cross_attn) return NULL;
        Tensor* x2 = uop_add(x, cross_attn);
        if (!x2) return NULL;
        x = apply_layernorm(x2,
                            layer->norm2_weight->tensor,
                            layer->norm2_bias->tensor,
                            layer->norm_eps);
        if (!x) return NULL;
    }

    /* FFN: Linear1 -> ReLU -> Linear2 + residual + LN3 */
    Tensor* ff1     = uop_linear(x, layer->linear1_weight->tensor, layer->linear1_bias->tensor);
    if (!ff1) return NULL;
    Tensor* ff1_act = uop_relu(ff1);
    if (!ff1_act) return NULL;
    Tensor* ff2     = uop_linear(ff1_act, layer->linear2_weight->tensor, layer->linear2_bias->tensor);
    if (!ff2) return NULL;
    Tensor* x3      = uop_add(x, ff2);
    if (!x3) return NULL;
    return apply_layernorm(x3,
                           layer->norm3_weight->tensor,
                           layer->norm3_bias->tensor,
                           layer->norm_eps);
}

static void decoder_layer_free(Module* module) {
    TransformerDecoderLayer* layer = (TransformerDecoderLayer*)module;
    if (!layer) return;
    if (layer->self_attn) module_free((Module*)layer->self_attn);
    if (layer->cross_attn) module_free((Module*)layer->cross_attn);
    free(layer);
}

TransformerDecoderLayer* nn_transformer_decoder_layer(int d_model, int nhead, int dim_feedforward,
                                                       float dropout, DType dtype, DeviceType device) {
    TransformerDecoderLayer* layer = malloc(sizeof(TransformerDecoderLayer));
    if (!layer) return NULL;

    if (module_init((Module*)layer, "TransformerDecoderLayer", decoder_layer_forward_wrapper, decoder_layer_free) != 0) {
        free(layer); return NULL;
    }

    layer->d_model = d_model;
    layer->nhead = nhead;
    layer->dim_feedforward = dim_feedforward;
    layer->dropout = dropout;
    layer->norm_eps = 1e-5f;

    layer->self_attn = nn_multihead_attention(d_model, nhead, dropout, dtype, device);
    if (!layer->self_attn) { module_free((Module*)layer); return NULL; }

    layer->cross_attn = nn_multihead_attention(d_model, nhead, dropout, dtype, device);
    if (!layer->cross_attn) { module_free((Module*)layer); return NULL; }

    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};

    int l1_w_shape[] = {dim_feedforward, d_model};
    int l1_b_shape[] = {dim_feedforward};
    int l2_w_shape[] = {d_model, dim_feedforward};
    int l2_b_shape[] = {d_model};
    int norm_shape[] = {d_model};

    Tensor* l1w = tensor_empty(l1_w_shape, 2, &config);
    if (!l1w) { module_free((Module*)layer); return NULL; }
    float* l1w_data = (float*)tensor_data_ptr(l1w);
    if (l1w_data) xavier_init(l1w_data, (size_t)dim_feedforward * d_model, d_model, dim_feedforward);
    if (module_add_parameter((Module*)layer, l1w, "linear1_weight", true) != 0) {
        tensor_free(l1w); module_free((Module*)layer); return NULL;
    }
    layer->linear1_weight = module_get_parameter((Module*)layer, "linear1_weight");

    Tensor* l1b = tensor_zeros(l1_b_shape, 1, &config);
    if (!l1b) { module_free((Module*)layer); return NULL; }
    if (module_add_parameter((Module*)layer, l1b, "linear1_bias", true) != 0) {
        tensor_free(l1b); module_free((Module*)layer); return NULL;
    }
    layer->linear1_bias = module_get_parameter((Module*)layer, "linear1_bias");

    Tensor* l2w = tensor_empty(l2_w_shape, 2, &config);
    if (!l2w) { module_free((Module*)layer); return NULL; }
    float* l2w_data = (float*)tensor_data_ptr(l2w);
    if (l2w_data) xavier_init(l2w_data, (size_t)d_model * dim_feedforward, dim_feedforward, d_model);
    if (module_add_parameter((Module*)layer, l2w, "linear2_weight", true) != 0) {
        tensor_free(l2w); module_free((Module*)layer); return NULL;
    }
    layer->linear2_weight = module_get_parameter((Module*)layer, "linear2_weight");

    Tensor* l2b = tensor_zeros(l2_b_shape, 1, &config);
    if (!l2b) { module_free((Module*)layer); return NULL; }
    if (module_add_parameter((Module*)layer, l2b, "linear2_bias", true) != 0) {
        tensor_free(l2b); module_free((Module*)layer); return NULL;
    }
    layer->linear2_bias = module_get_parameter((Module*)layer, "linear2_bias");

    struct { const char* wn; const char* bn; Parameter** wp; Parameter** bp; } norms[] = {
        {"norm1_weight", "norm1_bias", &layer->norm1_weight, &layer->norm1_bias},
        {"norm2_weight", "norm2_bias", &layer->norm2_weight, &layer->norm2_bias},
        {"norm3_weight", "norm3_bias", &layer->norm3_weight, &layer->norm3_bias},
    };
    for (int i = 0; i < 3; i++) {
        Tensor* nw = tensor_ones(norm_shape, 1, &config);
        if (!nw) { module_free((Module*)layer); return NULL; }
        if (module_add_parameter((Module*)layer, nw, norms[i].wn, true) != 0) {
            tensor_free(nw); module_free((Module*)layer); return NULL;
        }
        *norms[i].wp = module_get_parameter((Module*)layer, norms[i].wn);

        Tensor* nb = tensor_zeros(norm_shape, 1, &config);
        if (!nb) { module_free((Module*)layer); return NULL; }
        if (module_add_parameter((Module*)layer, nb, norms[i].bn, true) != 0) {
            tensor_free(nb); module_free((Module*)layer); return NULL;
        }
        *norms[i].bp = module_get_parameter((Module*)layer, norms[i].bn);
    }

    return layer;
}

static Tensor* transformer_decoder_forward(Module* module, Tensor* input) {
    TransformerDecoder* dec = (TransformerDecoder*)module;
    if (!dec || !input) return NULL;

    Tensor* x = input;
    bool owns_x = false;

    for (int i = 0; i < dec->num_layers; i++) {
        Tensor* out = transformer_decoder_layer_forward(dec->layers[i], x, NULL, NULL, NULL);
        if (owns_x) tensor_free(x);
        if (!out) return NULL;
        x = out;
        owns_x = true;
    }

    Tensor* out = apply_layernorm(x,
                                  dec->norm_weight->tensor,
                                  dec->norm_bias->tensor,
                                  dec->norm_eps);
    if (owns_x) tensor_free(x);
    return out;
}

static void transformer_decoder_free(Module* module) {
    TransformerDecoder* dec = (TransformerDecoder*)module;
    if (!dec) return;
    if (dec->layers) {
        for (int i = 0; i < dec->num_layers; i++) {
            if (dec->layers[i]) module_free((Module*)dec->layers[i]);
        }
        free(dec->layers);
    }
    free(dec);
}

TransformerDecoder* nn_transformer_decoder(int d_model, int nhead, int dim_feedforward,
                                            float dropout, int num_layers,
                                            DType dtype, DeviceType device) {
    TransformerDecoder* dec = malloc(sizeof(TransformerDecoder));
    if (!dec) return NULL;

    if (module_init((Module*)dec, "TransformerDecoder", transformer_decoder_forward, transformer_decoder_free) != 0) {
        free(dec); return NULL;
    }

    dec->d_model = d_model;
    dec->num_layers = num_layers;
    dec->norm_eps = 1e-5f;

    dec->layers = malloc(num_layers * sizeof(TransformerDecoderLayer*));
    if (!dec->layers) { module_free((Module*)dec); return NULL; }

    for (int i = 0; i < num_layers; i++) {
        dec->layers[i] = nn_transformer_decoder_layer(d_model, nhead, dim_feedforward, dropout, dtype, device);
        if (!dec->layers[i]) {
            dec->num_layers = i;
            module_free((Module*)dec);
            return NULL;
        }
    }

    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    int norm_shape[] = {d_model};

    Tensor* nw = tensor_ones(norm_shape, 1, &config);
    if (!nw) { module_free((Module*)dec); return NULL; }
    if (module_add_parameter((Module*)dec, nw, "norm_weight", true) != 0) {
        tensor_free(nw); module_free((Module*)dec); return NULL;
    }
    dec->norm_weight = module_get_parameter((Module*)dec, "norm_weight");

    Tensor* nb = tensor_zeros(norm_shape, 1, &config);
    if (!nb) { module_free((Module*)dec); return NULL; }
    if (module_add_parameter((Module*)dec, nb, "norm_bias", true) != 0) {
        tensor_free(nb); module_free((Module*)dec); return NULL;
    }
    dec->norm_bias = module_get_parameter((Module*)dec, "norm_bias");

    return dec;
}

KVCache* kv_cache_create(int batch, int num_heads, int max_seq_len, int head_dim,
                          DType dtype, DeviceType device) {
    if (batch <= 0 || num_heads <= 0 || max_seq_len <= 0 || head_dim <= 0) {
        LOG_ERROR("kv_cache_create: invalid dimensions");
        return NULL;
    }

    KVCache* cache = calloc(1, sizeof(KVCache));
    if (!cache) return NULL;

    cache->max_seq_len = max_seq_len;
    cache->current_len = 0;

    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    int shape[] = {batch, num_heads, max_seq_len, head_dim};

    cache->key_cache = tensor_zeros(shape, 4, &config);
    if (!cache->key_cache) { free(cache); return NULL; }

    cache->value_cache = tensor_zeros(shape, 4, &config);
    if (!cache->value_cache) {
        tensor_free(cache->key_cache);
        free(cache);
        return NULL;
    }

    return cache;
}

void kv_cache_free(KVCache* cache) {
    if (!cache) return;
    if (cache->key_cache)   tensor_free(cache->key_cache);
    if (cache->value_cache) tensor_free(cache->value_cache);
    free(cache);
}

void kv_cache_reset(KVCache* cache) {
    if (!cache) return;
    cache->current_len = 0;
    /* Realize and zero the cache tensors (mutable state cannot be lazy). */
    if (cache->key_cache) {
        tensor_ensure_executed(cache->key_cache);
        float* k = (float*)tensor_data_ptr(cache->key_cache);
        if (k) memset(k, 0, cache->key_cache->numel * sizeof(float));
    }
    if (cache->value_cache) {
        tensor_ensure_executed(cache->value_cache);
        float* v = (float*)tensor_data_ptr(cache->value_cache);
        if (v) memset(v, 0, cache->value_cache->numel * sizeof(float));
    }
}

Tensor* flash_attention_forward(MultiHeadAttention* mha, Tensor* query, Tensor* key,
                                 Tensor* value, Tensor* mask, FlashAttentionConfig* config) {
    (void)config;
    /* Route to the lazy SDPA-based implementation; FlashAttention is a memory
     * optimization that is not needed when execution is deferred. */
    return multihead_attention_forward(mha, query, key, value, mask);
}

Tensor* multihead_attention_forward_cached(MultiHeadAttention* mha, Tensor* query,
                                            Tensor* key, Tensor* value,
                                            Tensor* mask, KVCache* cache) {
    if (!mha || !query || !key || !value) return NULL;
    if (!cache) return multihead_attention_forward(mha, query, key, value, mask);

    /* KV cache involves mutable state so we materialize inputs first. */
    tensor_ensure_executed(query);
    tensor_ensure_executed(key);
    tensor_ensure_executed(value);
    tensor_ensure_executed(cache->key_cache);
    tensor_ensure_executed(cache->value_cache);

    if (query->ndim != 3 || key->ndim != 3 || value->ndim != 3) return NULL;

    int batch     = query->shape[0];
    int seq_q     = query->shape[1];
    int seq_k     = key->shape[1];
    int embed_dim = mha->embed_dim;
    int num_heads = mha->num_heads;
    int head_dim  = mha->head_dim;
    int total_q   = batch * seq_q;
    int total_k   = batch * seq_k;

    if (cache->current_len + seq_k > cache->max_seq_len) {
        LOG_ERROR("multihead_attention_forward_cached: cache overflow");
        return NULL;
    }

    /* Project Q, K, V eagerly so we can write K/V into the cache. */
    Tensor* Q_t = uop_linear(query, mha->W_q->tensor, mha->b_q->tensor);
    Tensor* K_t = uop_linear(key,   mha->W_k->tensor, mha->b_k->tensor);
    Tensor* V_t = uop_linear(value, mha->W_v->tensor, mha->b_v->tensor);
    if (!Q_t || !K_t || !V_t) return NULL;

    tensor_ensure_executed(Q_t);
    tensor_ensure_executed(K_t);
    tensor_ensure_executed(V_t);

    float* Q_data  = (float*)tensor_data_ptr(Q_t);
    float* K_data  = (float*)tensor_data_ptr(K_t);
    float* V_data  = (float*)tensor_data_ptr(V_t);
    float* kc_data = (float*)tensor_data_ptr(cache->key_cache);
    float* vc_data = (float*)tensor_data_ptr(cache->value_cache);
    if (!Q_data || !K_data || !V_data || !kc_data || !vc_data) return NULL;

    int max_sl  = cache->max_seq_len;
    int cur_pos = cache->current_len;

    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_k; s++) {
            for (int h = 0; h < num_heads; h++) {
                for (int d = 0; d < head_dim; d++) {
                    size_t ci = (size_t)b * num_heads * max_sl * head_dim +
                                h * max_sl * head_dim + (cur_pos + s) * head_dim + d;
                    size_t pi = (size_t)b * seq_k * embed_dim + s * embed_dim + h * head_dim + d;
                    kc_data[ci] = K_data[pi];
                    vc_data[ci] = V_data[pi];
                }
            }
        }
    }
    cache->current_len = cur_pos + seq_k;
    int cached_len = cache->current_len;

    /* Build contiguous Q/K/V tensors for the lazy attention path. */
    int q4_shape[] = {batch, num_heads, seq_q, head_dim};
    float* Q_mh = malloc((size_t)batch * num_heads * seq_q * head_dim * sizeof(float));
    if (!Q_mh) return NULL;
    for (int b = 0; b < batch; b++)
        for (int s = 0; s < seq_q; s++)
            for (int h = 0; h < num_heads; h++)
                for (int d = 0; d < head_dim; d++)
                    Q_mh[b*num_heads*seq_q*head_dim + h*seq_q*head_dim + s*head_dim + d] =
                        Q_data[b*seq_q*embed_dim + s*embed_dim + h*head_dim + d];

    float* K_cached = malloc((size_t)batch * num_heads * cached_len * head_dim * sizeof(float));
    float* V_cached = malloc((size_t)batch * num_heads * cached_len * head_dim * sizeof(float));
    if (!K_cached || !V_cached) { free(Q_mh); free(K_cached); free(V_cached); return NULL; }

    for (int b = 0; b < batch; b++)
        for (int h = 0; h < num_heads; h++)
            for (int s = 0; s < cached_len; s++)
                for (int d = 0; d < head_dim; d++) {
                    size_t ci = (size_t)b * num_heads * max_sl * head_dim +
                                h * max_sl * head_dim + s * head_dim + d;
                    K_cached[b*num_heads*cached_len*head_dim + h*cached_len*head_dim + s*head_dim + d] = kc_data[ci];
                    V_cached[b*num_heads*cached_len*head_dim + h*cached_len*head_dim + s*head_dim + d] = vc_data[ci];
                }

    TensorConfig cfg = {.dtype = query->dtype, .device = query->device,
                        .has_dtype = true, .has_device = true};
    Tensor* Q_lazy = tensor_from_data(Q_mh, q4_shape, 4, &cfg); free(Q_mh);
    int k4_shape[] = {batch, num_heads, cached_len, head_dim};
    Tensor* K_lazy = tensor_from_data(K_cached, k4_shape, 4, &cfg); free(K_cached);
    Tensor* V_lazy = tensor_from_data(V_cached, k4_shape, 4, &cfg); free(V_cached);
    if (!Q_lazy || !K_lazy || !V_lazy) return NULL;

    Tensor* attn_out = uop_scaled_dot_product_attention(Q_lazy, K_lazy, V_lazy, mask);
    if (!attn_out) return NULL;

    int inv_perm[] = {0, 2, 1, 3};
    PermuteParams ipp = {.perm = inv_perm, .num_dims = 4};
    Tensor* attn_t = uop_permute(attn_out, &ipp);
    if (!attn_t) return NULL;

    int out3[] = {batch, seq_q, embed_dim};
    ReshapeParams orp = {.new_shape = out3, .new_ndim = 3};
    Tensor* concat = uop_reshape(attn_t, &orp);
    if (!concat) return NULL;

    (void)total_q; (void)total_k;
    return uop_linear(concat, mha->W_o->tensor, mha->b_o->tensor);
}

void multihead_attention_set_flash(MultiHeadAttention* mha, bool enabled, bool causal) {
    if (!mha) return;
    (void)enabled; (void)causal;
}
