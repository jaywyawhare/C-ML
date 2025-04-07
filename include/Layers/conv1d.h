#ifndef CONV1D_H
#define CONV1D_H

#include "../../include/Core/memory_management.h"
#include "../../include/Core/error_codes.h"

#define MAX_CONV1D_LENGTH 65536
#define MIN_CONV1D_LENGTH 1
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
    int input_length;
    int padding;
    int stride;
    int dilation;
    float *d_weights;
    float *d_biases;
    int output_length;
} Conv1DLayer;

int validate_conv1d_params(const int input_channels, const int output_channels,
                           const int kernel_size, const int input_length,
                           const int padding, const int stride, const int dilation);

int initialize_conv1d(Conv1DLayer *layer, int input_channels, int output_channels,
                      int kernel_size, int input_length, int padding, int stride, int dilation);

int forward_conv1d(const Conv1DLayer *layer, const float *input, float *output);

int backward_conv1d(const Conv1DLayer *layer, const float *input, const float *output, const float *d_output, float *d_input);

int update_conv1d(Conv1DLayer *layer, float learning_rate);
int compute_output_length_conv1d(int input_length, int kernel_size,
                                 int padding, int stride, int dilation);
int free_conv1d(Conv1DLayer *layer);

#endif
