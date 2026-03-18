#ifndef CML_NN_LAYERS_BATCHNORM3D_H
#define CML_NN_LAYERS_BATCHNORM3D_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BatchNorm3d {
    Module base;

    int num_features;
    float eps;
    float momentum;
    bool affine;
    bool track_running_stats;

    Parameter* weight;
    Parameter* bias;

    Tensor* running_mean;
    Tensor* running_var;
} BatchNorm3d;

BatchNorm3d* nn_batchnorm3d(int num_features, float eps, float momentum, bool affine,
                             bool track_running_stats, DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_BATCHNORM3D_H
