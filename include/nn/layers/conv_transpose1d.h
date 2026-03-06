/**
 * @file conv_transpose1d.h
 * @brief 1D Transposed Convolution (Deconvolution) layer
 *
 * Implements nn.ConvTranspose1d for 1D upsampling / decoder networks.
 */

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

/**
 * @brief Create a ConvTranspose1d layer
 *
 * @param in_channels Number of input channels
 * @param out_channels Number of output channels
 * @param kernel_size Kernel size
 * @param stride Stride
 * @param padding Input padding
 * @param output_padding Additional size added to output
 * @param use_bias Whether to use bias
 * @param dtype Data type
 * @param device Device type
 * @return New ConvTranspose1d layer, or NULL on failure
 */
ConvTranspose1d* nn_conv_transpose1d(int in_channels, int out_channels, int kernel_size,
                                      int stride, int padding, int output_padding,
                                      bool use_bias, DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_CONV_TRANSPOSE1D_H
