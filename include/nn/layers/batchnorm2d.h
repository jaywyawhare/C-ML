#ifndef CML_NN_LAYERS_BATCHNORM2D_H
#define CML_NN_LAYERS_BATCHNORM2D_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef BatchNormState BatchNorm2d;

BatchNorm2d* nn_batchnorm2d(int num_features, float eps, float momentum, bool affine,
                            bool track_running_stats, DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_BATCHNORM2D_H
