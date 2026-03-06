/**
 * @file layernorm2d.h
 * @brief Layer Normalization 2D layer
 *
 * Implements nn.LayerNorm for 2D spatial inputs [N, C, H, W].
 * Normalizes over (C, H, W) dimensions per sample.
 */

#ifndef CML_NN_LAYERS_LAYERNORM2D_H
#define CML_NN_LAYERS_LAYERNORM2D_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LayerNorm2d {
    Module base;

    int num_channels;
    float eps;
    bool affine;

    Parameter* weight; // [num_channels]
    Parameter* bias;   // [num_channels]
} LayerNorm2d;

/**
 * @brief Create a LayerNorm2d layer
 *
 * @param num_channels Number of channels to normalize
 * @param eps Epsilon for numerical stability
 * @param affine Whether to include learnable parameters
 * @param dtype Data type
 * @param device Device type
 * @return New LayerNorm2d layer, or NULL on failure
 */
LayerNorm2d* nn_layernorm2d(int num_channels, float eps, bool affine, DType dtype,
                             DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_LAYERNORM2D_H
