#include "zoo/vit.h"
#include "nn/layers.h"
#include "autograd/forward_ops.h"
#include "tensor/tensor_manipulation.h"
#include "core/logging.h"
#include <stdlib.h>

ViTConfig cml_zoo_vit_config_tiny(void) {
    return (ViTConfig){
        .image_size  = 224,
        .patch_size  = 16,
        .num_classes = 1000,
        .n_layer     = 12,
        .n_head      = 3,
        .hidden_size = 192,
        .mlp_dim     = 768
    };
}

ViTConfig cml_zoo_vit_config_small(void) {
    return (ViTConfig){
        .image_size  = 224,
        .patch_size  = 16,
        .num_classes = 1000,
        .n_layer     = 12,
        .n_head      = 6,
        .hidden_size = 384,
        .mlp_dim     = 1536
    };
}

ViTConfig cml_zoo_vit_config_base(void) {
    return (ViTConfig){
        .image_size  = 224,
        .patch_size  = 16,
        .num_classes = 1000,
        .n_layer     = 12,
        .n_head      = 12,
        .hidden_size = 768,
        .mlp_dim     = 3072
    };
}

ViTConfig cml_zoo_vit_config_large(void) {
    return (ViTConfig){
        .image_size  = 224,
        .patch_size  = 16,
        .num_classes = 1000,
        .n_layer     = 24,
        .n_head      = 16,
        .hidden_size = 1024,
        .mlp_dim     = 4096
    };
}

typedef struct {
    Module base;
    MultiHeadAttention* attn;
    LayerNorm* norm1;
    Sequential* mlp;
    LayerNorm* norm2;
} ViTBlock;

static Tensor* vit_block_forward(Module* module, Tensor* input) {
    ViTBlock* block = (ViTBlock*)module;
    if (!block || !input)
        return NULL;

    Tensor* normed = module_forward((Module*)block->norm1, input);
    if (!normed)
        return NULL;

    Tensor* attn_out = multihead_attention_forward(block->attn, normed, normed, normed, NULL);
    if (!attn_out)
        return NULL;

    Tensor* x = tensor_add(input, attn_out);
    if (!x)
        return NULL;

    normed = module_forward((Module*)block->norm2, x);
    if (!normed)
        return NULL;

    Tensor* mlp_out = module_forward((Module*)block->mlp, normed);
    if (!mlp_out)
        return NULL;

    return tensor_add(x, mlp_out);
}

static void vit_block_free(Module* module) {
    ViTBlock* block = (ViTBlock*)module;
    if (!block)
        return;
    if (block->attn)
        module_free((Module*)block->attn);
    if (block->norm1)
        module_free((Module*)block->norm1);
    if (block->mlp)
        module_free((Module*)block->mlp);
    if (block->norm2)
        module_free((Module*)block->norm2);
    free(block);
}

static Module* create_vit_block(int hidden_size, int n_head, int mlp_dim,
                                 DType dtype, DeviceType device) {
    ViTBlock* block = malloc(sizeof(ViTBlock));
    if (!block)
        return NULL;

    if (module_init((Module*)block, "ViTBlock", vit_block_forward, vit_block_free) != 0) {
        free(block);
        return NULL;
    }

    block->norm1 = nn_layernorm(hidden_size, 1e-6f, true, dtype, device);
    block->attn = nn_multihead_attention(hidden_size, n_head, 0.0f, dtype, device);
    block->norm2 = nn_layernorm(hidden_size, 1e-6f, true, dtype, device);

    block->mlp = nn_sequential();
    sequential_add(block->mlp, (Module*)nn_linear(hidden_size, mlp_dim, dtype, device, true));
    sequential_add(block->mlp, (Module*)nn_gelu(false));
    sequential_add(block->mlp, (Module*)nn_linear(mlp_dim, hidden_size, dtype, device, true));

    return (Module*)block;
}

typedef struct {
    Module base;
    Conv2d* patch_embed;
    Parameter* cls_token;
    Parameter* pos_embed;
    ModuleList* blocks;
    LayerNorm* norm;
    Linear* head;
    int num_patches;
    int hidden_size;
    int n_layer;
} ViTModel;

static Tensor* vit_forward(Module* module, Tensor* input) {
    ViTModel* vit = (ViTModel*)module;
    if (!vit || !input)
        return NULL;

    Tensor* patches = module_forward((Module*)vit->patch_embed, input);
    if (!patches)
        return NULL;

    int batch = patches->shape[0];
    int seq_len = vit->num_patches;
    int dim = vit->hidden_size;

    int patch_shape[] = {batch, seq_len, dim};
    patches = tensor_reshape(patches, patch_shape, 3);
    if (!patches)
        return NULL;

    Tensor* cls = vit->cls_token->tensor;
    Tensor* tensors[] = {cls, patches};
    Tensor* x = tensor_concat(tensors, 2, 1);
    if (!x)
        return NULL;

    x = tensor_add(x, vit->pos_embed->tensor);
    if (!x)
        return NULL;

    for (int i = 0; i < vit->n_layer; i++) {
        Module* block = module_list_get(vit->blocks, i);
        if (!block)
            return NULL;
        x = module_forward(block, x);
        if (!x)
            return NULL;
    }

    x = module_forward((Module*)vit->norm, x);
    if (!x)
        return NULL;

    return module_forward((Module*)vit->head, x);
}

static void vit_free(Module* module) {
    ViTModel* vit = (ViTModel*)module;
    if (!vit)
        return;
    if (vit->patch_embed)
        module_free((Module*)vit->patch_embed);
    if (vit->blocks)
        module_free((Module*)vit->blocks);
    if (vit->norm)
        module_free((Module*)vit->norm);
    if (vit->head)
        module_free((Module*)vit->head);
    free(vit);
}

Module* cml_zoo_vit_create(ViTConfig* config, DType dtype, DeviceType device) {
    if (!config)
        return NULL;

    ViTModel* vit = malloc(sizeof(ViTModel));
    if (!vit)
        return NULL;

    if (module_init((Module*)vit, "ViT", vit_forward, vit_free) != 0) {
        free(vit);
        return NULL;
    }

    int num_patches = (config->image_size / config->patch_size) *
                      (config->image_size / config->patch_size);

    vit->hidden_size = config->hidden_size;
    vit->num_patches = num_patches;
    vit->n_layer = config->n_layer;

    vit->patch_embed = nn_conv2d(3, config->hidden_size, config->patch_size,
                                  config->patch_size, 0, 1, true, dtype, device);

    TensorConfig tcfg = {.dtype = dtype, .device = device};
    int cls_shape[] = {1, 1, config->hidden_size};
    Tensor* cls_tensor = tensor_zeros(cls_shape, 3, &tcfg);
    vit->cls_token = NULL;
    module_add_parameter((Module*)vit, cls_tensor, "cls_token", true);
    vit->cls_token = module_get_parameter((Module*)vit, "cls_token");

    int pos_shape[] = {1, num_patches + 1, config->hidden_size};
    Tensor* pos_tensor = tensor_zeros(pos_shape, 3, &tcfg);
    module_add_parameter((Module*)vit, pos_tensor, "pos_embed", true);
    vit->pos_embed = module_get_parameter((Module*)vit, "pos_embed");

    vit->blocks = nn_module_list();
    for (int i = 0; i < config->n_layer; i++)
        module_list_append(vit->blocks, create_vit_block(
            config->hidden_size, config->n_head, config->mlp_dim, dtype, device));

    vit->norm = nn_layernorm(config->hidden_size, 1e-6f, true, dtype, device);
    vit->head = nn_linear(config->hidden_size, config->num_classes, dtype, device, true);

    LOG_INFO("Created ViT (%d layers, %d hidden, %d heads, patch=%d)",
             config->n_layer, config->hidden_size, config->n_head, config->patch_size);
    return (Module*)vit;
}
