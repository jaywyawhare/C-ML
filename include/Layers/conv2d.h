#ifndef CONV2D_H
#define CONV2D_H

#include "../../include/Core/memory_management.h"
#include "../../include/Core/error_codes.h"

#define MAX_CONV2D_DIM 8192
#define MIN_CONV2D_DIM 1
#define MAX_CHANNELS 4096
#define MIN_CHANNELS 1
#define MAX_KERNEL_SIZE 32
#define MIN_KERNEL_SIZE 1

typedef struct
{
    int input_channels;
    int output_channels;
    int kernel_size;
    float *weights;
    float *biases;
    int input_height;
    int input_width;
    int padding;
    int stride;
    int dilation;
    float *d_weights;
    float *d_biases;
    int output_height;
    int output_width;
} Conv2DLayer;

int initialize_conv2d(Conv2DLayer *layer, int input_channels, int output_channels,
                      int kernel_size, int input_height, int input_width,
                      int padding, int stride, int dilation);

int forward_conv2d(Conv2DLayer *layer, const float *input, float *output);

int backward_conv2d(Conv2DLayer *layer, const float *input, const float *output, const float *d_output, float *d_input);

int update_conv2d(Conv2DLayer *layer, const float learning_rate);
int free_conv2d(Conv2DLayer *layer);
int compute_output_size_conv2d(int input_size, int kernel_size,
                               int padding, int stride, int dilation);
int validate_conv2d_params(const int input_channels, const int output_channels,
                           const int kernel_size, const int input_height,
                           const int input_width, const int padding,
                           const int stride, const int dilation);

#endif
