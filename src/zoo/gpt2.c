#include "zoo/gpt2.h"
#include "nn/layers.h"
#include "autograd/forward_ops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

GPT2Config cml_zoo_gpt2_config_small(void) {
    return (GPT2Config){
        .vocab_size = 50257,
        .n_layer    = 12,
        .n_head     = 12,
        .n_embd     = 768,
        .block_size = 1024
    };
}

GPT2Config cml_zoo_gpt2_config_medium(void) {
    return (GPT2Config){
        .vocab_size = 50257,
        .n_layer    = 24,
        .n_head     = 16,
        .n_embd     = 1024,
        .block_size = 1024
    };
}

GPT2Config cml_zoo_gpt2_config_large(void) {
    return (GPT2Config){
        .vocab_size = 50257,
        .n_layer    = 36,
        .n_head     = 20,
        .n_embd     = 1280,
        .block_size = 1024
    };
}

GPT2Config cml_zoo_gpt2_config_xl(void) {
    return (GPT2Config){
        .vocab_size = 50257,
        .n_layer    = 48,
        .n_head     = 25,
        .n_embd     = 1600,
        .block_size = 1024
    };
}

typedef struct {
    Module base;
    Sequential* ln_attn;
    MultiHeadAttention* attn;
    Sequential* ln_mlp;
    Sequential* mlp;
} GPT2Block;

static Tensor* gpt2_block_forward(Module* module, Tensor* input) {
    GPT2Block* block = (GPT2Block*)module;
    if (!block || !input)
        return NULL;

    Tensor* normed = module_forward((Module*)block->ln_attn, input);
    if (!normed)
        return NULL;

    Tensor* attn_out = multihead_attention_forward(block->attn, normed, normed, normed, NULL);
    if (!attn_out)
        return NULL;

    Tensor* x = tensor_add(input, attn_out);
    if (!x)
        return NULL;

    normed = module_forward((Module*)block->ln_mlp, x);
    if (!normed)
        return NULL;

    Tensor* mlp_out = module_forward((Module*)block->mlp, normed);
    if (!mlp_out)
        return NULL;

    return tensor_add(x, mlp_out);
}

static void gpt2_block_free(Module* module) {
    GPT2Block* block = (GPT2Block*)module;
    if (!block)
        return;
    if (block->ln_attn)
        module_free((Module*)block->ln_attn);
    if (block->attn)
        module_free((Module*)block->attn);
    if (block->ln_mlp)
        module_free((Module*)block->ln_mlp);
    if (block->mlp)
        module_free((Module*)block->mlp);
    free(block);
}

static Module* create_gpt2_block(int n_embd, int n_head, int n_layer, DType dtype, DeviceType device) {
    GPT2Block* block = malloc(sizeof(GPT2Block));
    if (!block)
        return NULL;

    if (module_init((Module*)block, "GPT2Block", gpt2_block_forward, gpt2_block_free) != 0) {
        free(block);
        return NULL;
    }

    int ff_dim = 4 * n_embd;

    block->ln_attn = nn_sequential();
    sequential_add(block->ln_attn, (Module*)nn_layernorm(n_embd, 1e-5f, true, dtype, device));

    block->attn = nn_multihead_attention(n_embd, n_head, 0.0f, dtype, device);
    if (block->attn)
        multihead_attention_set_flash(block->attn, false, true);

    block->ln_mlp = nn_sequential();
    sequential_add(block->ln_mlp, (Module*)nn_layernorm(n_embd, 1e-5f, true, dtype, device));

    block->mlp = nn_sequential();
    sequential_add(block->mlp, (Module*)nn_linear(n_embd, ff_dim, dtype, device, true));
    sequential_add(block->mlp, (Module*)nn_gelu(false));
    Linear* mlp_proj = nn_linear(ff_dim, n_embd, dtype, device, true);
    sequential_add(block->mlp, (Module*)mlp_proj);

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
        /* Scale attention output projection (W_o) */
        if (block->attn && block->attn->W_o) {
            float* w = (float*)tensor_data_ptr(block->attn->W_o->tensor);
            if (w) {
                for (size_t i = 0; i < block->attn->W_o->tensor->numel; i++)
                    w[i] *= scale;
            }
        }
    }

    return (Module*)block;
}

typedef struct {
    Module base;
    Embedding* tok_emb;
    Embedding* pos_emb;
    ModuleList* blocks;
    LayerNorm* ln_f;
    Linear* lm_head;
    int n_layer;
    int n_embd;
    int vocab_size;
    int block_size;
} GPT2Model;

static Tensor* gpt2_forward(Module* module, Tensor* input) {
    GPT2Model* gpt2 = (GPT2Model*)module;
    if (!gpt2 || !input)
        return NULL;

    Tensor* tok = module_forward((Module*)gpt2->tok_emb, input);
    if (!tok)
        return NULL;

    /* Create position indices [0, 1, 2, ..., seq_len-1] */
    int seq_len = input->shape[input->ndim - 1];
    int pos_shape[] = {1, seq_len};
    TensorConfig tcfg = {.dtype = input->dtype, .device = input->device,
                         .has_dtype = true, .has_device = true};
    Tensor* positions = tensor_zeros(pos_shape, 2, &tcfg);
    if (positions) {
        float* pos_data = (float*)tensor_data_ptr(positions);
        if (pos_data) {
            for (int i = 0; i < seq_len; i++)
                pos_data[i] = (float)i;
        }
        Tensor* pos_emb = module_forward((Module*)gpt2->pos_emb, positions);
        if (pos_emb)
            tok = tensor_add(tok, pos_emb);
    }

    Tensor* x = tok;

    for (int i = 0; i < gpt2->n_layer; i++) {
        Module* block = module_list_get(gpt2->blocks, i);
        if (!block)
            return NULL;
        x = module_forward(block, x);
        if (!x)
            return NULL;
    }

    x = module_forward((Module*)gpt2->ln_f, x);
    if (!x)
        return NULL;

    return module_forward((Module*)gpt2->lm_head, x);
}

static void gpt2_free(Module* module) {
    GPT2Model* gpt2 = (GPT2Model*)module;
    if (!gpt2)
        return;
    if (gpt2->tok_emb)
        module_free((Module*)gpt2->tok_emb);
    if (gpt2->pos_emb)
        module_free((Module*)gpt2->pos_emb);
    if (gpt2->blocks)
        module_free((Module*)gpt2->blocks);
    if (gpt2->ln_f)
        module_free((Module*)gpt2->ln_f);
    if (gpt2->lm_head)
        module_free((Module*)gpt2->lm_head);
    free(gpt2);
}

Module* cml_zoo_gpt2_create(GPT2Config* config, DType dtype, DeviceType device) {
    if (!config)
        return NULL;

    GPT2Model* gpt2 = malloc(sizeof(GPT2Model));
    if (!gpt2)
        return NULL;

    if (module_init((Module*)gpt2, "GPT2", gpt2_forward, gpt2_free) != 0) {
        free(gpt2);
        return NULL;
    }

    gpt2->n_layer    = config->n_layer;
    gpt2->n_embd     = config->n_embd;
    gpt2->vocab_size = config->vocab_size;
    gpt2->block_size = config->block_size;

    gpt2->tok_emb = nn_embedding(config->vocab_size, config->n_embd, -1, dtype, device);
    gpt2->pos_emb = nn_embedding(config->block_size, config->n_embd, -1, dtype, device);

    gpt2->blocks = nn_module_list();
    for (int i = 0; i < config->n_layer; i++)
        module_list_append(gpt2->blocks, create_gpt2_block(config->n_embd, config->n_head, config->n_layer, dtype, device));

    gpt2->ln_f = nn_layernorm(config->n_embd, 1e-5f, true, dtype, device);
    gpt2->lm_head = nn_linear(config->n_embd, config->vocab_size, dtype, device, false);

    LOG_INFO("Created GPT-2 (%d layers, %d hidden, %d heads, %d vocab)",
             config->n_layer, config->n_embd, config->n_head, config->vocab_size);
    return (Module*)gpt2;
}
