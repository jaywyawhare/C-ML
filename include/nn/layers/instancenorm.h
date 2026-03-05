/**
 * @file instancenorm.h
 * @brief Instance Normalization layer
 *
 * Implements nn.InstanceNorm2d: normalizes each channel of each sample independently.
 */

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

    // Learnable parameters (only if affine=true)
    Parameter* weight;
    Parameter* bias;
} InstanceNorm2d;

/**
 * @brief Create an InstanceNorm2d layer
 *
 * @param num_features Number of channels (C) in input
 * @param eps Epsilon for numerical stability (default: 1e-5)
 * @param affine Whether to include learnable affine parameters
 * @param dtype Data type
 * @param device Device
 * @return New InstanceNorm2d layer, or NULL on failure
 */
InstanceNorm2d* nn_instancenorm2d(int num_features, float eps, bool affine, DType dtype,
                                   DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_INSTANCENORM_H
