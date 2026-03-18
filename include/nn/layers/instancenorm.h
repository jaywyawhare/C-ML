#ifndef CML_NN_LAYERS_INSTANCENORM_H
#define CML_NN_LAYERS_INSTANCENORM_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct InstanceNorm2d {
    Module base;

    int num_features;
    float eps;
    bool affine;

    Parameter* weight;
    Parameter* bias;
} InstanceNorm2d;

InstanceNorm2d* nn_instancenorm2d(int num_features, float eps, bool affine, DType dtype,
                                   DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_INSTANCENORM_H
