#ifndef CML_NN_LAYERS_LAYERNORM_H
#define CML_NN_LAYERS_LAYERNORM_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LayerNorm {
    Module base;

    int normalized_shape; // Number of features to normalize
    float eps;
    bool affine; // Whether to use learnable parameters

    Parameter* weight; // [normalized_shape] (gamma)
    Parameter* bias;   // [normalized_shape] (beta)
} LayerNorm;

LayerNorm* nn_layernorm(int normalized_shape, float eps, bool affine, DType dtype,
                        DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_LAYERNORM_H
