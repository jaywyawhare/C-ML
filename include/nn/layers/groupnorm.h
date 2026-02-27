/**
 * @file groupnorm.h
 * @brief Group Normalization layer
 */

#ifndef CML_NN_LAYERS_GROUPNORM_H
#define CML_NN_LAYERS_GROUPNORM_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GroupNorm {
    Module base;
    int num_groups;
    int num_channels;
    float eps;
    bool affine;
    Parameter* weight; // [num_channels]
    Parameter* bias;   // [num_channels]
} GroupNorm;

GroupNorm* nn_groupnorm(int num_groups, int num_channels, float eps, bool affine,
                        DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_GROUPNORM_H
