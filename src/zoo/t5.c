#include "zoo/t5.h"
#include "nn/layers.h"
#include "autograd/forward_ops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>

T5Config cml_zoo_t5_config_small(void) {
    return (T5Config){
        .vocab_size    = 32128,
        .n_layer       = 6,
        .n_head        = 8,
        .d_model       = 512,
        .d_ff          = 2048,
        .max_position  = 512,
        .num_buckets   = 32
    };
}

T5Config cml_zoo_t5_config_base(void) {
    return (T5Config){
        .vocab_size    = 32128,
        .n_layer       = 12,
        .n_head        = 12,
        .d_model       = 768,
        .d_ff          = 3072,
        .max_position  = 512,
        .num_buckets   = 32
    };
}

T5Config cml_zoo_t5_config_large(void) {
    return (T5Config){
        .vocab_size    = 32128,
        .n_layer       = 24,
        .n_head        = 16,
        .d_model       = 1024,
        .d_ff          = 4096,
        .max_position  = 512,
        .num_buckets   = 32
    };
}

/* T5 encoder block (pre-norm, ReLU MLP) */

typedef struct {
    Module base;
    LayerNorm* norm1;
    MultiHeadAttention* self_attn;
    LayerNorm* norm2;
    Sequential* mlp;
    Parameter* rel_bias;
    int n_head;
    int num_buckets;
} T5EncoderBlock;

static Tensor* t5_enc_block_forward(Module* module, Tensor* input) {
    T5EncoderBlock* block = (T5EncoderBlock*)module;
    if (!block || !input) return NULL;

    Tensor* normed = module_forward((Module*)block->norm1, input);
    if (!normed) return NULL;

    Tensor* attn_out = multihead_attention_forward(
        block->self_attn, normed, normed, normed, NULL);
    if (!attn_out) return NULL;

    Tensor* x = tensor_add(input, attn_out);
    if (!x) return NULL;

    normed = module_forward((Module*)block->norm2, x);
    if (!normed) return NULL;

    Tensor* mlp_out = module_forward((Module*)block->mlp, normed);
    if (!mlp_out) return NULL;

    return tensor_add(x, mlp_out);
}

static void t5_enc_block_free(Module* module) {
    T5EncoderBlock* block = (T5EncoderBlock*)module;
    if (!block) return;
    if (block->norm1) module_free((Module*)block->norm1);
    if (block->self_attn) module_free((Module*)block->self_attn);
    if (block->norm2) module_free((Module*)block->norm2);
    if (block->mlp) module_free((Module*)block->mlp);
    free(block);
}

static Module* create_t5_enc_block(int d_model, int n_head, int d_ff, int num_buckets,
                                    int n_layer, DType dtype, DeviceType device) {
    T5EncoderBlock* block = malloc(sizeof(T5EncoderBlock));
    if (!block) return NULL;

    if (module_init((Module*)block, "T5EncoderBlock",
                    t5_enc_block_forward, t5_enc_block_free) != 0) {
        free(block);
        return NULL;
    }

    block->n_head = n_head;
    block->num_buckets = num_buckets;

    block->norm1 = nn_layernorm(d_model, 1e-6f, true, dtype, device);
    block->self_attn = nn_multihead_attention(d_model, n_head, 0.0f, dtype, device);
    block->norm2 = nn_layernorm(d_model, 1e-6f, true, dtype, device);

    block->mlp = nn_sequential();
    sequential_add(block->mlp, (Module*)nn_linear(d_model, d_ff, dtype, device, false));
    sequential_add(block->mlp, (Module*)nn_relu(false));
    Linear* mlp_proj = nn_linear(d_ff, d_model, dtype, device, false);
    sequential_add(block->mlp, (Module*)mlp_proj);

    TensorConfig tcfg = {.dtype = dtype, .device = device, };
    int bias_shape[] = {n_head, num_buckets};
    Tensor* bias_t = tensor_zeros(bias_shape, 2, &tcfg);
    module_add_parameter((Module*)block, bias_t, "rel_pos_bias", true);
    block->rel_bias = module_get_parameter((Module*)block, "rel_pos_bias");

    /* Scale residual path weights by 1/sqrt(2*n_layer) to prevent
       activation explosion through deep residual connections (GPT-2 style). */
    if (n_layer > 1) {
        float scale = 1.0f / sqrtf(2.0f * (float)n_layer);
        /* Scale MLP output projection */
        if (mlp_proj && mlp_proj->weight) {
            float* w = (float*)tensor_data_ptr(mlp_proj->weight->tensor);
            if (w) {
                for (size_t i = 0; i < mlp_proj->weight->tensor->numel; i++)
                    w[i] *= scale;
            }
        }
        /* Scale self-attention output projection (W_o) */
        if (block->self_attn && block->self_attn->W_o) {
            float* w = (float*)tensor_data_ptr(block->self_attn->W_o->tensor);
            if (w) {
                for (size_t i = 0; i < block->self_attn->W_o->tensor->numel; i++)
                    w[i] *= scale;
            }
        }
    }

    return (Module*)block;
}

/* T5 decoder block (pre-norm, self-attn + cross-attn + MLP) */

typedef struct {
    Module base;
    LayerNorm* norm1;
    MultiHeadAttention* self_attn;
    LayerNorm* norm2;
    MultiHeadAttention* cross_attn;
    LayerNorm* norm3;
    Sequential* mlp;
    Parameter* self_rel_bias;
    int n_head;
    int num_buckets;
} T5DecoderBlock;

static Tensor* t5_dec_block_forward(Module* module, Tensor* input) {
    T5DecoderBlock* block = (T5DecoderBlock*)module;
    if (!block || !input) return NULL;

    Tensor* normed = module_forward((Module*)block->norm1, input);
    if (!normed) return NULL;

    Tensor* self_out = multihead_attention_forward(
        block->self_attn, normed, normed, normed, NULL);
    if (!self_out) return NULL;

    Tensor* x = tensor_add(input, self_out);
    if (!x) return NULL;

    return x;
}

static Tensor* t5_dec_block_forward_with_memory(T5DecoderBlock* block, Tensor* input,
                                                  Tensor* memory) {
    if (!block || !input) return NULL;

    Tensor* normed = module_forward((Module*)block->norm1, input);
    if (!normed) return NULL;

    Tensor* self_out = multihead_attention_forward(
        block->self_attn, normed, normed, normed, NULL);
    if (!self_out) return NULL;

    Tensor* x = tensor_add(input, self_out);
    if (!x) return NULL;

    normed = module_forward((Module*)block->norm2, x);
    if (!normed) return NULL;

    Tensor* cross_out = multihead_attention_forward(
        block->cross_attn, normed, memory, memory, NULL);
    if (!cross_out) return NULL;

    x = tensor_add(x, cross_out);
    if (!x) return NULL;

    normed = module_forward((Module*)block->norm3, x);
    if (!normed) return NULL;

    Tensor* mlp_out = module_forward((Module*)block->mlp, normed);
    if (!mlp_out) return NULL;

    return tensor_add(x, mlp_out);
}

static void t5_dec_block_free(Module* module) {
    T5DecoderBlock* block = (T5DecoderBlock*)module;
    if (!block) return;
    if (block->norm1) module_free((Module*)block->norm1);
    if (block->self_attn) module_free((Module*)block->self_attn);
    if (block->norm2) module_free((Module*)block->norm2);
    if (block->cross_attn) module_free((Module*)block->cross_attn);
    if (block->norm3) module_free((Module*)block->norm3);
    if (block->mlp) module_free((Module*)block->mlp);
    free(block);
}

static Module* create_t5_dec_block(int d_model, int n_head, int d_ff, int num_buckets,
                                    int n_layer, DType dtype, DeviceType device) {
    T5DecoderBlock* block = malloc(sizeof(T5DecoderBlock));
    if (!block) return NULL;

    if (module_init((Module*)block, "T5DecoderBlock",
                    t5_dec_block_forward, t5_dec_block_free) != 0) {
        free(block);
        return NULL;
    }

    block->n_head = n_head;
    block->num_buckets = num_buckets;

    block->norm1 = nn_layernorm(d_model, 1e-6f, true, dtype, device);
    block->self_attn = nn_multihead_attention(d_model, n_head, 0.0f, dtype, device);
    multihead_attention_set_flash(block->self_attn, false, true);

    block->norm2 = nn_layernorm(d_model, 1e-6f, true, dtype, device);
    block->cross_attn = nn_multihead_attention(d_model, n_head, 0.0f, dtype, device);

    block->norm3 = nn_layernorm(d_model, 1e-6f, true, dtype, device);
    block->mlp = nn_sequential();
    sequential_add(block->mlp, (Module*)nn_linear(d_model, d_ff, dtype, device, false));
    sequential_add(block->mlp, (Module*)nn_relu(false));
    Linear* mlp_proj = nn_linear(d_ff, d_model, dtype, device, false);
    sequential_add(block->mlp, (Module*)mlp_proj);

    TensorConfig tcfg = {.dtype = dtype, .device = device, };
    int bias_shape[] = {n_head, num_buckets};
    Tensor* bias_t = tensor_zeros(bias_shape, 2, &tcfg);
    module_add_parameter((Module*)block, bias_t, "self_rel_pos_bias", true);
    block->self_rel_bias = module_get_parameter((Module*)block, "self_rel_pos_bias");

    /* Scale residual path weights by 1/sqrt(2*n_layer) to prevent
       activation explosion through deep residual connections (GPT-2 style). */
    if (n_layer > 1) {
        float scale = 1.0f / sqrtf(2.0f * (float)n_layer);
        /* Scale MLP output projection */
        if (mlp_proj && mlp_proj->weight) {
            float* w = (float*)tensor_data_ptr(mlp_proj->weight->tensor);
            if (w) {
                for (size_t i = 0; i < mlp_proj->weight->tensor->numel; i++)
                    w[i] *= scale;
            }
        }
        /* Scale self-attention output projection (W_o) */
        if (block->self_attn && block->self_attn->W_o) {
            float* w = (float*)tensor_data_ptr(block->self_attn->W_o->tensor);
            if (w) {
                for (size_t i = 0; i < block->self_attn->W_o->tensor->numel; i++)
                    w[i] *= scale;
            }
        }
        /* Scale cross-attention output projection (W_o) */
        if (block->cross_attn && block->cross_attn->W_o) {
            float* w = (float*)tensor_data_ptr(block->cross_attn->W_o->tensor);
            if (w) {
                for (size_t i = 0; i < block->cross_attn->W_o->tensor->numel; i++)
                    w[i] *= scale;
            }
        }
    }

    return (Module*)block;
}

/* T5 model */

typedef struct {
    Module base;
    Embedding* shared_embed;
    ModuleList* enc_blocks;
    LayerNorm* enc_norm;
    ModuleList* dec_blocks;
    LayerNorm* dec_norm;
    Linear* lm_head;
    int n_enc_layer;
    int n_dec_layer;
    int d_model;
    int vocab_size;
} T5Model;

static Tensor* t5_forward(Module* module, Tensor* input) {
    return t5_encode(module, input);
}

static void t5_free(Module* module) {
    T5Model* t5 = (T5Model*)module;
    if (!t5) return;
    if (t5->shared_embed) module_free((Module*)t5->shared_embed);
    if (t5->enc_blocks) module_free((Module*)t5->enc_blocks);
    if (t5->enc_norm) module_free((Module*)t5->enc_norm);
    if (t5->dec_blocks) module_free((Module*)t5->dec_blocks);
    if (t5->dec_norm) module_free((Module*)t5->dec_norm);
    if (t5->lm_head) module_free((Module*)t5->lm_head);
    free(t5);
}

Tensor* t5_encode(Module* module, Tensor* input) {
    T5Model* t5 = (T5Model*)module;
    if (!t5 || !input) return NULL;

    Tensor* x = module_forward((Module*)t5->shared_embed, input);
    if (!x) return NULL;

    for (int i = 0; i < t5->n_enc_layer; i++) {
        Module* block = module_list_get(t5->enc_blocks, i);
        if (!block) return NULL;
        x = module_forward(block, x);
        if (!x) return NULL;
    }

    return module_forward((Module*)t5->enc_norm, x);
}

Tensor* t5_decode(Module* module, Tensor* tgt, Tensor* memory) {
    T5Model* t5 = (T5Model*)module;
    if (!t5 || !tgt || !memory) return NULL;

    Tensor* x = module_forward((Module*)t5->shared_embed, tgt);
    if (!x) return NULL;

    for (int i = 0; i < t5->n_dec_layer; i++) {
        T5DecoderBlock* block = (T5DecoderBlock*)module_list_get(t5->dec_blocks, i);
        if (!block) return NULL;
        x = t5_dec_block_forward_with_memory(block, x, memory);
        if (!x) return NULL;
    }

    x = module_forward((Module*)t5->dec_norm, x);
    if (!x) return NULL;

    return module_forward((Module*)t5->lm_head, x);
}

Module* cml_zoo_t5_create(T5Config* config, DType dtype, DeviceType device) {
    if (!config)
        return NULL;

    T5Model* t5 = malloc(sizeof(T5Model));
    if (!t5)
        return NULL;

    if (module_init((Module*)t5, "T5", t5_forward, t5_free) != 0) {
        free(t5);
        return NULL;
    }

    t5->n_enc_layer = config->n_layer;
    t5->n_dec_layer = config->n_layer;
    t5->d_model = config->d_model;
    t5->vocab_size = config->vocab_size;

    t5->shared_embed = nn_embedding(config->vocab_size, config->d_model, -1, dtype, device);

    t5->enc_blocks = nn_module_list();
    for (int i = 0; i < config->n_layer; i++)
        module_list_append(t5->enc_blocks,
            create_t5_enc_block(config->d_model, config->n_head, config->d_ff,
                                config->num_buckets, config->n_layer, dtype, device));
    t5->enc_norm = nn_layernorm(config->d_model, 1e-6f, true, dtype, device);

    t5->dec_blocks = nn_module_list();
    for (int i = 0; i < config->n_layer; i++)
        module_list_append(t5->dec_blocks,
            create_t5_dec_block(config->d_model, config->n_head, config->d_ff,
                                config->num_buckets, config->n_layer, dtype, device));
    t5->dec_norm = nn_layernorm(config->d_model, 1e-6f, true, dtype, device);

    t5->lm_head = nn_linear(config->d_model, config->vocab_size, dtype, device, false);

    LOG_INFO("Created T5 (%d layers, %d hidden, %d heads, %d vocab)",
             config->n_layer, config->d_model, config->n_head, config->vocab_size);
    return (Module*)t5;
}
