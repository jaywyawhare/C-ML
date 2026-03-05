/**
 * @file rmsnorm.h
 * @brief Root Mean Square Layer Normalization
 *
 * RMSNorm normalizes by the RMS of the input (no mean subtraction).
 * Formula: output = input / RMS(input) * weight
 * where RMS(x) = sqrt(mean(x^2) + eps)
 */

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

/**
 * @brief Create an RMSNorm layer
 *
 * @param normalized_shape Number of features to normalize
 * @param eps Small value for numerical stability (default: 1e-5)
 * @param dtype Data type
 * @param device Device type
 * @return New RMSNorm layer, or NULL on failure
 */
RMSNorm* nn_rmsnorm(int normalized_shape, float eps, DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_RMSNORM_H
