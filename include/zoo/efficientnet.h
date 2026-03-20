#ifndef CML_ZOO_EFFICIENTNET_H
#define CML_ZOO_EFFICIENTNET_H

#include "nn.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int expand_ratio;
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int num_blocks;
    int se_ratio;
} EfficientNetBlockConfig;

typedef struct {
    int num_classes;
    float dropout_rate;
    float width_mult;
    float depth_mult;
    DType dtype;
    DeviceType device;
} EfficientNetConfig;

EfficientNetConfig efficientnet_b0_config(int num_classes);

Module* cml_zoo_efficientnet_b0(const EfficientNetConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_EFFICIENTNET_H */
