#ifndef CML_ZOO_CLIP_H
#define CML_ZOO_CLIP_H

#include "nn.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int image_size;
    int patch_size;
    int vision_layers;
    int vision_heads;
    int vision_dim;
    int text_layers;
    int text_heads;
    int text_dim;
    int vocab_size;
    int max_text_len;
    int embed_dim;
} CMLCLIPConfig;

Module* cml_zoo_clip_create(CMLCLIPConfig* config, DType dtype, DeviceType device);

Tensor* clip_encode_image(Module* module, Tensor* image);
Tensor* clip_encode_text(Module* module, Tensor* text);
Tensor* clip_contrastive_loss(Tensor* image_embeds, Tensor* text_embeds, float temperature);

CMLCLIPConfig cml_zoo_clip_config_vit_b32(void);
CMLCLIPConfig cml_zoo_clip_config_vit_b16(void);
CMLCLIPConfig cml_zoo_clip_config_vit_l14(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_CLIP_H */
