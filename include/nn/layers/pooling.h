/**
 * @file pooling.h
 * @brief Pooling layers
 *
 */

#ifndef CML_NN_LAYERS_POOLING_H
#define CML_NN_LAYERS_POOLING_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MaxPool2d {
    Module base;
    int kernel_size[2];
    int stride[2];
    int padding[2];
    int dilation[2];
    bool ceil_mode;
} MaxPool2d;

MaxPool2d* nn_maxpool2d(int kernel_size, int stride, int padding, int dilation, bool ceil_mode);

typedef struct AvgPool2d {
    Module base;
    int kernel_size[2];
    int stride[2];
    int padding[2];
    bool ceil_mode;
    bool count_include_pad;
} AvgPool2d;

AvgPool2d* nn_avgpool2d(int kernel_size, int stride, int padding, bool ceil_mode,
                        bool count_include_pad);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_POOLING_H
