#ifndef CML_NN_LAYERS_RMSNORM_H
#define CML_NN_LAYERS_RMSNORM_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RMSNorm {
    Module base;

    int normalized_shape;
    float eps;

    Parameter* weight; // [normalized_shape] (gain)
} RMSNorm;

RMSNorm* nn_rmsnorm(int normalized_shape, float eps, DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_RMSNORM_H
