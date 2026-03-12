#ifndef CML_NN_LAYERS_CONV1D_H
#define CML_NN_LAYERS_CONV1D_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Conv1d {
    Module base;

    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    int groups;
    bool use_bias;

    Parameter* weight; // [out_channels, in_channels/groups, kernel_size]
    Parameter* bias;   // [out_channels]
} Conv1d;

Conv1d* nn_conv1d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                  int dilation, bool use_bias, DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_CONV1D_H
