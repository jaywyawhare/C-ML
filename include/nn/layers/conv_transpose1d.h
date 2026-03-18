#ifndef CML_NN_LAYERS_CONV_TRANSPOSE1D_H
#define CML_NN_LAYERS_CONV_TRANSPOSE1D_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ConvTranspose1d {
    Module base;

    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int output_padding;
    int dilation;
    bool use_bias;

    Parameter* weight; // [in_channels, out_channels, kernel_size]
    Parameter* bias;   // [out_channels]
} ConvTranspose1d;

ConvTranspose1d* nn_conv_transpose1d(int in_channels, int out_channels, int kernel_size,
                                      int stride, int padding, int output_padding,
                                      bool use_bias, DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_CONV_TRANSPOSE1D_H
