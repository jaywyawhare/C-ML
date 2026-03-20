#include "zoo/clip.h"
#include "nn/layers.h"
#include "autograd/forward_ops.h"
#include "tensor/tensor_manipulation.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>

CMLCLIPConfig cml_zoo_clip_config_vit_b32(void) {
    return (CMLCLIPConfig){
        .image_size    = 224,
        .patch_size    = 32,
        .vision_layers = 12,
        .vision_heads  = 12,
        .vision_dim    = 768,
        .text_layers   = 12,
        .text_heads    = 8,
        .text_dim      = 512,
        .vocab_size    = 49408,
        .max_text_len  = 77,
        .embed_dim     = 512
    };
}

CMLCLIPConfig cml_zoo_clip_config_vit_b16(void) {
    return (CMLCLIPConfig){
        .image_size    = 224,
        .patch_size    = 16,
        .vision_layers = 12,
        .vision_heads  = 12,
        .vision_dim    = 768,
        .text_layers   = 12,
        .text_heads    = 8,
        .text_dim      = 512,
        .vocab_size    = 49408,
        .max_text_len  = 77,
        .embed_dim     = 512
    };
}

CMLCLIPConfig cml_zoo_clip_config_vit_l14(void) {
    return (CMLCLIPConfig){
        .image_size    = 224,
        .patch_size    = 14,
        .vision_layers = 24,
        .vision_heads  = 16,
        .vision_dim    = 1024,
        .text_layers   = 12,
        .text_heads    = 12,
        .text_dim      = 768,
        .vocab_size    = 49408,
        .max_text_len  = 77,
        .embed_dim     = 768
    };
}

/* Vision encoder block */

typedef struct {
    Module base;
    LayerNorm* norm1;
    MultiHeadAttention* attn;
    LayerNorm* norm2;
    Sequential* mlp;
} CLIPVisionBlock;

static Tensor* clip_vision_block_forward(Module* module, Tensor* input) {
    CLIPVisionBlock* block = (CLIPVisionBlock*)module;
    if (!block || !input)
        return NULL;

    Tensor* normed = module_forward((Module*)block->norm1, input);
    if (!normed) return NULL;

    Tensor* attn_out = multihead_attention_forward(block->attn, normed, normed, normed, NULL);
    if (!attn_out) return NULL;

    Tensor* x = tensor_add(input, attn_out);
    if (!x) return NULL;

    normed = module_forward((Module*)block->norm2, x);
    if (!normed) return NULL;

    Tensor* mlp_out = module_forward((Module*)block->mlp, normed);
    if (!mlp_out) return NULL;

    return tensor_add(x, mlp_out);
}

static void clip_vision_block_free(Module* module) {
    CLIPVisionBlock* block = (CLIPVisionBlock*)module;
    if (!block) return;
    if (block->norm1) module_free((Module*)block->norm1);
    if (block->attn) module_free((Module*)block->attn);
    if (block->norm2) module_free((Module*)block->norm2);
    if (block->mlp) module_free((Module*)block->mlp);
    free(block);
}

static Module* create_vision_block(int dim, int n_head, DType dtype, DeviceType device) {
    CLIPVisionBlock* block = malloc(sizeof(CLIPVisionBlock));
    if (!block) return NULL;

    if (module_init((Module*)block, "CLIPVisionBlock",
                    clip_vision_block_forward, clip_vision_block_free) != 0) {
        free(block);
        return NULL;
    }

    int mlp_dim = dim * 4;
    block->norm1 = nn_layernorm(dim, 1e-5f, true, dtype, device);
    block->attn = nn_multihead_attention(dim, n_head, 0.0f, dtype, device);
    block->norm2 = nn_layernorm(dim, 1e-5f, true, dtype, device);

    block->mlp = nn_sequential();
    sequential_add(block->mlp, (Module*)nn_linear(dim, mlp_dim, dtype, device, true));
    sequential_add(block->mlp, (Module*)nn_gelu(false));
    sequential_add(block->mlp, (Module*)nn_linear(mlp_dim, dim, dtype, device, true));

    return (Module*)block;
}

/* Text encoder block (causal) */

typedef struct {
    Module base;
    LayerNorm* norm1;
    MultiHeadAttention* attn;
    LayerNorm* norm2;
    Sequential* mlp;
} CLIPTextBlock;

static Tensor* clip_text_block_forward(Module* module, Tensor* input) {
    CLIPTextBlock* block = (CLIPTextBlock*)module;
    if (!block || !input)
        return NULL;

    Tensor* normed = module_forward((Module*)block->norm1, input);
    if (!normed) return NULL;

    Tensor* attn_out = multihead_attention_forward(block->attn, normed, normed, normed, NULL);
    if (!attn_out) return NULL;

    Tensor* x = tensor_add(input, attn_out);
    if (!x) return NULL;

    normed = module_forward((Module*)block->norm2, x);
    if (!normed) return NULL;

    Tensor* mlp_out = module_forward((Module*)block->mlp, normed);
    if (!mlp_out) return NULL;

    return tensor_add(x, mlp_out);
}

static void clip_text_block_free(Module* module) {
    CLIPTextBlock* block = (CLIPTextBlock*)module;
    if (!block) return;
    if (block->norm1) module_free((Module*)block->norm1);
    if (block->attn) module_free((Module*)block->attn);
    if (block->norm2) module_free((Module*)block->norm2);
    if (block->mlp) module_free((Module*)block->mlp);
    free(block);
}

static Module* create_text_block(int dim, int n_head, DType dtype, DeviceType device) {
    CLIPTextBlock* block = malloc(sizeof(CLIPTextBlock));
    if (!block) return NULL;

    if (module_init((Module*)block, "CLIPTextBlock",
                    clip_text_block_forward, clip_text_block_free) != 0) {
        free(block);
        return NULL;
    }

    int mlp_dim = dim * 4;
    block->norm1 = nn_layernorm(dim, 1e-5f, true, dtype, device);
    block->attn = nn_multihead_attention(dim, n_head, 0.0f, dtype, device);
    multihead_attention_set_flash(block->attn, false, true);
    block->norm2 = nn_layernorm(dim, 1e-5f, true, dtype, device);

    block->mlp = nn_sequential();
    sequential_add(block->mlp, (Module*)nn_linear(dim, mlp_dim, dtype, device, true));
    sequential_add(block->mlp, (Module*)nn_gelu(false));
    sequential_add(block->mlp, (Module*)nn_linear(mlp_dim, dim, dtype, device, true));

    return (Module*)block;
}

/* CLIP model */

typedef struct {
    Module base;

    Conv2d* vision_patch_embed;
    Parameter* vision_cls_token;
    Parameter* vision_pos_embed;
    ModuleList* vision_blocks;
    LayerNorm* vision_norm;
    Linear* vision_proj;

    Embedding* text_tok_embed;
    Parameter* text_pos_embed;
    ModuleList* text_blocks;
    LayerNorm* text_norm;
    Linear* text_proj;

    Parameter* logit_scale;

    int vision_dim;
    int text_dim;
    int embed_dim;
    int num_vision_patches;
    int vision_layers;
    int text_layers;
} CLIPModel;

static Tensor* clip_forward(Module* module, Tensor* input) {
    return clip_encode_image(module, input);
}

static void clip_free(Module* module) {
    CLIPModel* clip = (CLIPModel*)module;
    if (!clip) return;

    if (clip->vision_patch_embed) module_free((Module*)clip->vision_patch_embed);
    if (clip->vision_blocks) module_free((Module*)clip->vision_blocks);
    if (clip->vision_norm) module_free((Module*)clip->vision_norm);
    if (clip->vision_proj) module_free((Module*)clip->vision_proj);

    if (clip->text_tok_embed) module_free((Module*)clip->text_tok_embed);
    if (clip->text_blocks) module_free((Module*)clip->text_blocks);
    if (clip->text_norm) module_free((Module*)clip->text_norm);
    if (clip->text_proj) module_free((Module*)clip->text_proj);

    free(clip);
}

Tensor* clip_encode_image(Module* module, Tensor* image) {
    CLIPModel* clip = (CLIPModel*)module;
    if (!clip || !image) return NULL;

    Tensor* patches = module_forward((Module*)clip->vision_patch_embed, image);
    if (!patches) return NULL;

    int batch = patches->shape[0];
    int seq_len = clip->num_vision_patches;
    int dim = clip->vision_dim;

    int patch_shape[] = {batch, seq_len, dim};
    patches = tensor_reshape(patches, patch_shape, 3);
    if (!patches) return NULL;

    Tensor* cls = clip->vision_cls_token->tensor;
    Tensor* tensors[] = {cls, patches};
    Tensor* x = tensor_concat(tensors, 2, 1);
    if (!x) return NULL;

    x = tensor_add(x, clip->vision_pos_embed->tensor);
    if (!x) return NULL;

    for (int i = 0; i < clip->vision_layers; i++) {
        Module* block = module_list_get(clip->vision_blocks, i);
        if (!block) return NULL;
        x = module_forward(block, x);
        if (!x) return NULL;
    }

    x = module_forward((Module*)clip->vision_norm, x);
    if (!x) return NULL;

    return module_forward((Module*)clip->vision_proj, x);
}

Tensor* clip_encode_text(Module* module, Tensor* text) {
    CLIPModel* clip = (CLIPModel*)module;
    if (!clip || !text) return NULL;

    Tensor* x = module_forward((Module*)clip->text_tok_embed, text);
    if (!x) return NULL;

    x = tensor_add(x, clip->text_pos_embed->tensor);
    if (!x) return NULL;

    for (int i = 0; i < clip->text_layers; i++) {
        Module* block = module_list_get(clip->text_blocks, i);
        if (!block) return NULL;
        x = module_forward(block, x);
        if (!x) return NULL;
    }

    x = module_forward((Module*)clip->text_norm, x);
    if (!x) return NULL;

    return module_forward((Module*)clip->text_proj, x);
}

Tensor* clip_contrastive_loss(Tensor* image_embeds, Tensor* text_embeds, float temperature) {
    if (!image_embeds || !text_embeds)
        return NULL;

    Tensor* text_t = tensor_transpose(text_embeds, 0, 1);
    if (!text_t) return NULL;

    Tensor* logits = tensor_matmul(image_embeds, text_t);
    if (!logits) return NULL;

    TensorConfig tcfg = {.dtype = image_embeds->dtype, .device = image_embeds->device};
    int scale_shape[] = {1};
    Tensor* scale = tensor_full(scale_shape, 1, &tcfg, 1.0f / temperature);
    if (!scale) return NULL;

    logits = tensor_mul(logits, scale);
    if (!logits) return NULL;

    int n = logits->shape[0];
    Tensor* log_softmax_rows = tensor_softmax(logits, 1);
    Tensor* log_softmax_cols = tensor_softmax(logits, 0);
    if (!log_softmax_rows || !log_softmax_cols)
        return NULL;

    Tensor* loss_img = tensor_log(log_softmax_rows);
    Tensor* loss_txt = tensor_log(log_softmax_cols);
    if (!loss_img || !loss_txt) return NULL;

    Tensor* loss = tensor_add(loss_img, loss_txt);
    if (!loss) return NULL;

    loss = tensor_neg(loss);
    if (!loss) return NULL;

    Tensor* mean_loss = tensor_mean(loss, -1, false);
    if (!mean_loss) return NULL;

    int result_shape[] = {1};
    Tensor* half = tensor_full(result_shape, 1, &tcfg, 0.5f / (float)n);
    return tensor_mul(mean_loss, half);
}

Module* cml_zoo_clip_create(CMLCLIPConfig* config, DType dtype, DeviceType device) {
    if (!config)
        return NULL;

    CLIPModel* clip = malloc(sizeof(CLIPModel));
    if (!clip)
        return NULL;

    if (module_init((Module*)clip, "CLIP", clip_forward, clip_free) != 0) {
        free(clip);
        return NULL;
    }

    int num_patches = (config->image_size / config->patch_size) *
                      (config->image_size / config->patch_size);

    clip->vision_dim = config->vision_dim;
    clip->text_dim = config->text_dim;
    clip->embed_dim = config->embed_dim;
    clip->num_vision_patches = num_patches;
    clip->vision_layers = config->vision_layers;
    clip->text_layers = config->text_layers;

    /* Vision encoder */
    clip->vision_patch_embed = nn_conv2d(3, config->vision_dim, config->patch_size,
                                          config->patch_size, 0, 1, true, dtype, device);

    TensorConfig tcfg = {.dtype = dtype, .device = device, };

    int cls_shape[] = {1, 1, config->vision_dim};
    Tensor* cls_t = tensor_zeros(cls_shape, 3, &tcfg);
    module_add_parameter((Module*)clip, cls_t, "vision_cls_token", true);
    clip->vision_cls_token = module_get_parameter((Module*)clip, "vision_cls_token");

    int vpos_shape[] = {1, num_patches + 1, config->vision_dim};
    Tensor* vpos_t = tensor_zeros(vpos_shape, 3, &tcfg);
    module_add_parameter((Module*)clip, vpos_t, "vision_pos_embed", true);
    clip->vision_pos_embed = module_get_parameter((Module*)clip, "vision_pos_embed");

    clip->vision_blocks = nn_module_list();
    for (int i = 0; i < config->vision_layers; i++)
        module_list_append(clip->vision_blocks,
            create_vision_block(config->vision_dim, config->vision_heads, dtype, device));

    clip->vision_norm = nn_layernorm(config->vision_dim, 1e-5f, true, dtype, device);
    clip->vision_proj = nn_linear(config->vision_dim, config->embed_dim, dtype, device, false);

    /* Text encoder */
    clip->text_tok_embed = nn_embedding(config->vocab_size, config->text_dim, -1, dtype, device);

    int tpos_shape[] = {1, config->max_text_len, config->text_dim};
    Tensor* tpos_t = tensor_zeros(tpos_shape, 3, &tcfg);
    module_add_parameter((Module*)clip, tpos_t, "text_pos_embed", true);
    clip->text_pos_embed = module_get_parameter((Module*)clip, "text_pos_embed");

    clip->text_blocks = nn_module_list();
    for (int i = 0; i < config->text_layers; i++)
        module_list_append(clip->text_blocks,
            create_text_block(config->text_dim, config->text_heads, dtype, device));

    clip->text_norm = nn_layernorm(config->text_dim, 1e-5f, true, dtype, device);
    clip->text_proj = nn_linear(config->text_dim, config->embed_dim, dtype, device, false);

    /* Learnable temperature */
    int scale_shape[] = {1};
    Tensor* scale_t = tensor_full(scale_shape, 1, &tcfg, logf(1.0f / 0.07f));
    module_add_parameter((Module*)clip, scale_t, "logit_scale", true);
    clip->logit_scale = module_get_parameter((Module*)clip, "logit_scale");

    LOG_INFO("Created CLIP (vision: %dL/%dd/patch%d, text: %dL/%dd, embed=%d)",
             config->vision_layers, config->vision_dim, config->patch_size,
             config->text_layers, config->text_dim, config->embed_dim);
    return (Module*)clip;
}
