#ifndef CML_ZOO_VIT_H
#define CML_ZOO_VIT_H

#include "nn.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int image_size;
    int patch_size;
    int num_classes;
    int n_layer;
    int n_head;
    int hidden_size;
    int mlp_dim;
} ViTConfig;

Module* cml_zoo_vit_create(ViTConfig* config, DType dtype, DeviceType device);

ViTConfig cml_zoo_vit_config_tiny(void);
ViTConfig cml_zoo_vit_config_small(void);
ViTConfig cml_zoo_vit_config_base(void);
ViTConfig cml_zoo_vit_config_large(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_VIT_H */
