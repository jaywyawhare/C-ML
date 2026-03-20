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
EfficientNetConfig efficientnet_b1_config(int num_classes);
EfficientNetConfig efficientnet_b2_config(int num_classes);
EfficientNetConfig efficientnet_b3_config(int num_classes);
EfficientNetConfig efficientnet_b4_config(int num_classes);
EfficientNetConfig efficientnet_b5_config(int num_classes);
EfficientNetConfig efficientnet_b6_config(int num_classes);
EfficientNetConfig efficientnet_b7_config(int num_classes);

Module* cml_zoo_efficientnet_b0(const EfficientNetConfig* config);
Module* cml_zoo_efficientnet_b1(const EfficientNetConfig* config);
Module* cml_zoo_efficientnet_b2(const EfficientNetConfig* config);
Module* cml_zoo_efficientnet_b3(const EfficientNetConfig* config);
Module* cml_zoo_efficientnet_b4(const EfficientNetConfig* config);
Module* cml_zoo_efficientnet_b5(const EfficientNetConfig* config);
Module* cml_zoo_efficientnet_b6(const EfficientNetConfig* config);
Module* cml_zoo_efficientnet_b7(const EfficientNetConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_EFFICIENTNET_H */
