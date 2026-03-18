#ifndef CML_NN_LAYERS_BATCHNORM2D_H
#define CML_NN_LAYERS_BATCHNORM2D_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BatchNorm2d {
    Module base;

    int num_features;
    float eps;
    float momentum;
    bool affine; // Whether to use learnable parameters
    bool track_running_stats;

    Parameter* weight;
    Parameter* bias;

    Tensor* running_mean;
    Tensor* running_var;

    Tensor* current_mean;
    Tensor* current_var;
} BatchNorm2d;

BatchNorm2d* nn_batchnorm2d(int num_features, float eps, float momentum, bool affine,
                            bool track_running_stats, DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_BATCHNORM2D_H
