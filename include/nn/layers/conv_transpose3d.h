#ifndef CML_NN_LAYERS_CONV_TRANSPOSE3D_H
#define CML_NN_LAYERS_CONV_TRANSPOSE3D_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ConvTranspose3d {
    Module base;

    int in_channels;
    int out_channels;
    int kernel_size[3];     // [depth, height, width]
    int stride[3];
    int padding[3];
    int output_padding[3];
    int dilation[3];
    bool use_bias;

    Parameter* weight; // [in_channels, out_channels, kd, kh, kw]
    Parameter* bias;   // [out_channels]
} ConvTranspose3d;

ConvTranspose3d* nn_conv_transpose3d(int in_channels, int out_channels, int kernel_size,
                                      int stride, int padding, int output_padding,
                                      bool use_bias, DType dtype, DeviceType device);

Tensor* conv_transpose3d_forward(Module* module, Tensor* input);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_CONV_TRANSPOSE3D_H
