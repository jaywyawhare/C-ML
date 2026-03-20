#ifndef CML_ZOO_UNET_H
#define CML_ZOO_UNET_H

#include "nn.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int in_channels;
    int num_classes;
    int depth;
    int base_filters;
} CMLUNetConfig;

Module* cml_zoo_unet_create(CMLUNetConfig* config, DType dtype, DeviceType device);

CMLUNetConfig cml_zoo_unet_config_default(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_UNET_H */
