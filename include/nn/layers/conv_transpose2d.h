#ifndef CML_NN_LAYERS_CONV_TRANSPOSE2D_H
#define CML_NN_LAYERS_CONV_TRANSPOSE2D_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ConvTranspose2d {
    Module base;

    int in_channels;
    int out_channels;
    int kernel_size[2];
    int stride[2];
    int padding[2];
    int output_padding[2];
    int dilation[2];
    bool use_bias;

    Parameter* weight; // [in_channels, out_channels, kernel_h, kernel_w]
    Parameter* bias;   // [out_channels]
} ConvTranspose2d;

ConvTranspose2d* nn_conv_transpose2d(int in_channels, int out_channels, int kernel_size,
                                      int stride, int padding, int output_padding,
                                      bool use_bias, DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_CONV_TRANSPOSE2D_H
