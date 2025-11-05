/**
 * @file conv2d.h
 * @brief 2D Convolution layer
 *
 * Implements the nn.Conv2d layer
 */

#ifndef CML_NN_LAYERS_CONV2D_H
#define CML_NN_LAYERS_CONV2D_H

#include "nn/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Conv2d {
    Module base;

    int in_channels;
    int out_channels;
    int kernel_size[2]; // [height, width]
    int stride[2];
    int padding[2];
    int dilation[2];
    bool use_bias;

    Parameter* weight; // [out_channels, in_channels, kernel_h, kernel_w]
    Parameter* bias;   // [out_channels]
} Conv2d;

Conv2d* nn_conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                  int dilation, bool use_bias, DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_CONV2D_H
