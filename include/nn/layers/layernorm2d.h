#ifndef CML_NN_LAYERS_LAYERNORM2D_H
#define CML_NN_LAYERS_LAYERNORM2D_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LayerNorm2d {
    Module base;

    int num_channels;
    float eps;
    bool affine;

    Parameter* weight; // [num_channels]
    Parameter* bias;   // [num_channels]
} LayerNorm2d;

LayerNorm2d* nn_layernorm2d(int num_channels, float eps, bool affine, DType dtype,
                             DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_LAYERNORM2D_H
