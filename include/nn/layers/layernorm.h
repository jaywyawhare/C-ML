/**
 * @file layernorm.h
 * @brief Layer Normalization layer
 *
 */

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

    // Learnable parameters
    Parameter* weight; // [normalized_shape] (gamma)
    Parameter* bias;   // [normalized_shape] (beta)
} LayerNorm;

/**
 * @brief Create a LayerNorm layer
 *
 * @param normalized_shape Number of features to normalize
 * @param eps Small value for numerical stability
 * @param affine Whether to use learnable scale and shift
 * @param dtype Data type
 * @param device Device type
 * @return New LayerNorm layer, or NULL on failure
 */
LayerNorm* nn_layernorm(int normalized_shape, float eps, bool affine, DType dtype,
                        DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_LAYERNORM_H
