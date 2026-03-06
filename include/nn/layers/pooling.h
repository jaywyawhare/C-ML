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

typedef struct MaxPool1d {
    Module base;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    bool ceil_mode;
} MaxPool1d;

MaxPool1d* nn_maxpool1d(int kernel_size, int stride, int padding, int dilation, bool ceil_mode);

typedef struct AvgPool1d {
    Module base;
    int kernel_size;
    int stride;
    int padding;
    bool ceil_mode;
    bool count_include_pad;
} AvgPool1d;

AvgPool1d* nn_avgpool1d(int kernel_size, int stride, int padding, bool ceil_mode,
                        bool count_include_pad);

typedef struct MaxPool3d {
    Module base;
    int kernel_size[3];
    int stride[3];
    int padding[3];
    int dilation[3];
    bool ceil_mode;
} MaxPool3d;

MaxPool3d* nn_maxpool3d(int kernel_size, int stride, int padding, int dilation, bool ceil_mode);

typedef struct AvgPool3d {
    Module base;
    int kernel_size[3];
    int stride[3];
    int padding[3];
    bool ceil_mode;
    bool count_include_pad;
} AvgPool3d;

AvgPool3d* nn_avgpool3d(int kernel_size, int stride, int padding, bool ceil_mode,
                        bool count_include_pad);

typedef struct AdaptiveAvgPool2d {
    Module base;
    int output_size[2];
} AdaptiveAvgPool2d;

AdaptiveAvgPool2d* nn_adaptive_avgpool2d(int output_h, int output_w);

typedef struct AdaptiveAvgPool1d {
    Module base;
    int output_size;
} AdaptiveAvgPool1d;

AdaptiveAvgPool1d* nn_adaptive_avgpool1d(int output_size);

typedef struct AdaptiveMaxPool2d {
    Module base;
    int output_size[2];
} AdaptiveMaxPool2d;

AdaptiveMaxPool2d* nn_adaptive_maxpool2d(int output_h, int output_w);

typedef struct AdaptiveMaxPool1d {
    Module base;
    int output_size;
} AdaptiveMaxPool1d;

AdaptiveMaxPool1d* nn_adaptive_maxpool1d(int output_size);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_POOLING_H
