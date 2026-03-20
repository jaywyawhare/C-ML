#ifndef CML_ZOO_UNET3D_H
#define CML_ZOO_UNET3D_H

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
} CMLUNet3DConfig;

typedef struct {
    Module base;
    CMLUNet3DConfig config;
    DType dtype;
    DeviceType device;

    Module** enc_blocks;
    Module** enc_pools;
    Module*  bottleneck;
    Module** dec_upsamples;
    Module** dec_blocks;
    Module*  final_conv;

    int depth;
} CMLUNet3D;

Module* cml_zoo_unet3d_create(const CMLUNet3DConfig* config, DType dtype, DeviceType device);

CMLUNet3DConfig cml_zoo_unet3d_config_default(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_UNET3D_H */
