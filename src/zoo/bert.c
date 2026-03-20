#include "zoo/bert.h"
#include "nn/layers.h"
#include "autograd/forward_ops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>

BERTConfig cml_zoo_bert_config_tiny(void) {
    return (BERTConfig){
        .vocab_size        = 30522,
        .n_layer           = 4,
        .n_head            = 2,
        .hidden_size       = 128,
        .intermediate_size = 512,
        .max_position      = 512
    };
}

BERTConfig cml_zoo_bert_config_mini(void) {
    return (BERTConfig){
        .vocab_size        = 30522,
        .n_layer           = 4,
        .n_head            = 4,
        .hidden_size       = 256,
        .intermediate_size = 1024,
        .max_position      = 512
    };
}

BERTConfig cml_zoo_bert_config_small(void) {
    return (BERTConfig){
        .vocab_size        = 30522,
        .n_layer           = 4,
        .n_head            = 8,
        .hidden_size       = 512,
        .intermediate_size = 2048,
        .max_position      = 512
    };
}

BERTConfig cml_zoo_bert_config_base(void) {
    return (BERTConfig){
        .vocab_size        = 30522,
        .n_layer           = 12,
        .n_head            = 12,
        .hidden_size       = 768,
        .intermediate_size = 3072,
        .max_position      = 512
    };
}

BERTConfig cml_zoo_bert_config_large(void) {
    return (BERTConfig){
        .vocab_size        = 30522,
        .n_layer           = 24,
        .n_head            = 16,
        .hidden_size       = 1024,
        .intermediate_size = 4096,
        .max_position      = 512
    };
}

typedef struct {
    Module base;
    MultiHeadAttention* attn;
    LayerNorm* attn_norm;
    Sequential* mlp;
    LayerNorm* mlp_norm;
} BERTEncoderBlock;

static Tensor* bert_block_forward(Module* module, Tensor* input) {
    BERTEncoderBlock* block = (BERTEncoderBlock*)module;
    if (!block || !input)
        return NULL;

    Tensor* attn_out = multihead_attention_forward(block->attn, input, input, input, NULL);
    if (!attn_out)
        return NULL;

    Tensor* x = tensor_add(input, attn_out);
    if (!x)
        return NULL;

    x = module_forward((Module*)block->attn_norm, x);
    if (!x)
        return NULL;

    Tensor* mlp_out = module_forward((Module*)block->mlp, x);
    if (!mlp_out)
        return NULL;

    Tensor* y = tensor_add(x, mlp_out);
    if (!y)
        return NULL;

    return module_forward((Module*)block->mlp_norm, y);
}

static void bert_block_free(Module* module) {
    BERTEncoderBlock* block = (BERTEncoderBlock*)module;
    if (!block)
        return;
    if (block->attn)
        module_free((Module*)block->attn);
    if (block->attn_norm)
        module_free((Module*)block->attn_norm);
    if (block->mlp)
        module_free((Module*)block->mlp);
    if (block->mlp_norm)
        module_free((Module*)block->mlp_norm);
    free(block);
}

static Module* create_bert_block(int hidden_size, int n_head, int intermediate_size,
                                  int n_layer, DType dtype, DeviceType device) {
    BERTEncoderBlock* block = malloc(sizeof(BERTEncoderBlock));
    if (!block)
        return NULL;

    if (module_init((Module*)block, "BERTEncoderBlock", bert_block_forward, bert_block_free) != 0) {
        free(block);
        return NULL;
    }

    block->attn = nn_multihead_attention(hidden_size, n_head, 0.0f, dtype, device);

    block->attn_norm = nn_layernorm(hidden_size, 1e-12f, true, dtype, device);

    block->mlp = nn_sequential();
    sequential_add(block->mlp, (Module*)nn_linear(hidden_size, intermediate_size, dtype, device, true));
    sequential_add(block->mlp, (Module*)nn_gelu(false));
    Linear* mlp_proj = nn_linear(intermediate_size, hidden_size, dtype, device, true);
    sequential_add(block->mlp, (Module*)mlp_proj);

    block->mlp_norm = nn_layernorm(hidden_size, 1e-12f, true, dtype, device);

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
    Embedding* seg_emb;
    LayerNorm* emb_norm;
    ModuleList* layers;
    Linear* pooler;
    int n_layer;
    int hidden_size;
} BERTModel;

static Tensor* bert_forward(Module* module, Tensor* input) {
    BERTModel* bert = (BERTModel*)module;
    if (!bert || !input)
        return NULL;

    Tensor* x = module_forward((Module*)bert->tok_emb, input);
    if (!x)
        return NULL;

    x = module_forward((Module*)bert->emb_norm, x);
    if (!x)
        return NULL;

    for (int i = 0; i < bert->n_layer; i++) {
        Module* layer = module_list_get(bert->layers, i);
        if (!layer)
            return NULL;
        x = module_forward(layer, x);
        if (!x)
            return NULL;
    }

    return module_forward((Module*)bert->pooler, x);
}

static void bert_free(Module* module) {
    BERTModel* bert = (BERTModel*)module;
    if (!bert)
        return;
    if (bert->tok_emb)
        module_free((Module*)bert->tok_emb);
    if (bert->pos_emb)
        module_free((Module*)bert->pos_emb);
    if (bert->seg_emb)
        module_free((Module*)bert->seg_emb);
    if (bert->emb_norm)
        module_free((Module*)bert->emb_norm);
    if (bert->layers)
        module_free((Module*)bert->layers);
    if (bert->pooler)
        module_free((Module*)bert->pooler);
    free(bert);
}

Module* cml_zoo_bert_create(BERTConfig* config, DType dtype, DeviceType device) {
    if (!config)
        return NULL;

    BERTModel* bert = malloc(sizeof(BERTModel));
    if (!bert)
        return NULL;

    if (module_init((Module*)bert, "BERT", bert_forward, bert_free) != 0) {
        free(bert);
        return NULL;
    }

    bert->n_layer     = config->n_layer;
    bert->hidden_size = config->hidden_size;

    bert->tok_emb  = nn_embedding(config->vocab_size, config->hidden_size, 0, dtype, device);
    bert->pos_emb  = nn_embedding(config->max_position, config->hidden_size, -1, dtype, device);
    bert->seg_emb  = nn_embedding(2, config->hidden_size, -1, dtype, device);
    bert->emb_norm = nn_layernorm(config->hidden_size, 1e-12f, true, dtype, device);

    bert->layers = nn_module_list();
    for (int i = 0; i < config->n_layer; i++)
        module_list_append(bert->layers, create_bert_block(
            config->hidden_size, config->n_head, config->intermediate_size,
            config->n_layer, dtype, device));

    bert->pooler = nn_linear(config->hidden_size, config->hidden_size, dtype, device, true);

    LOG_INFO("Created BERT (%d layers, %d hidden, %d heads, %d vocab)",
             config->n_layer, config->hidden_size, config->n_head, config->vocab_size);
    return (Module*)bert;
}
