/**
 * @file conv3d.h
 * @brief 3D Convolution layer
 */

#ifndef CML_NN_LAYERS_CONV3D_H
#define CML_NN_LAYERS_CONV3D_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Conv3d {
    Module base;
    int in_channels;
    int out_channels;
    int kernel_size[3]; // [depth, height, width]
    int stride[3];
    int padding[3];
    int dilation[3];
    bool use_bias;
    Parameter* weight; // [out_channels, in_channels, kd, kh, kw]
    Parameter* bias;   // [out_channels]
} Conv3d;

Conv3d* nn_conv3d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                  int dilation, bool use_bias, DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_CONV3D_H
