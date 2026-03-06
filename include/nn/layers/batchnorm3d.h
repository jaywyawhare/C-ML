/**
 * @file batchnorm3d.h
 * @brief Batch Normalization 3D layer
 *
 * Implements nn.BatchNorm3d for 5D inputs [N, C, D, H, W].
 */

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

/**
 * @brief Create a BatchNorm3d layer
 *
 * @param num_features Number of features (channels)
 * @param eps Epsilon for numerical stability
 * @param momentum Momentum for running stats
 * @param affine Whether to include learnable parameters
 * @param track_running_stats Whether to track running statistics
 * @param dtype Data type
 * @param device Device type
 * @return New BatchNorm3d layer, or NULL on failure
 */
BatchNorm3d* nn_batchnorm3d(int num_features, float eps, float momentum, bool affine,
                             bool track_running_stats, DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_BATCHNORM3D_H
