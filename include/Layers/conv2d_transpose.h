#ifndef CONV2D_TRANSPOSE_H
#define CONV2D_TRANSPOSE_H

#include "../../include/Core/memory_management.h"
#include "../../include/Core/error_codes.h"

#define MAX_CONV2D_DIM 8192
#define MIN_CONV2D_DIM 1
#define MAX_CHANNELS 4096
#define MIN_CHANNELS 1
#define MAX_KERNEL_SIZE 32
#define MIN_KERNEL_SIZE 1

/**
 * @brief Structure representing a 2D Convolution Transpose Layer.
 *
 * @param input_channels Number of input channels.
 * @param output_channels Number of output channels.
 * @param kernel_size Size of the convolution kernel (assumed square).
 * @param weights Pointer to the weights array.
 * @param biases Pointer to the biases array.
 * @param input_height Height of the input.
 * @param input_width Width of the input.
 * @param padding Padding applied to the input.
 * @param stride Stride of the convolution.
 * @param dilation Dilation factor for the convolution.
 * @param d_weights Pointer to the gradient of weights.
 * @param d_biases Pointer to the gradient of biases.
 * @param output_height Cached output height.
 * @param output_width Cached output width.
 */
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
} Conv2DTransposeLayer;

int initialize_conv2d_transpose(Conv2DTransposeLayer *layer, int input_channels, int output_channels,
                                int kernel_size, int input_height, int input_width,
                                int padding, int stride, int dilation);

int forward_conv2d_transpose(Conv2DTransposeLayer *layer, float *input, float *output);

int backward_conv2d_transpose(Conv2DTransposeLayer *layer, float *input, float *output, float *d_output, float *d_input);
int free_conv2d_transpose(Conv2DTransposeLayer *layer);
int update_conv2d_transpose(Conv2DTransposeLayer *layer, float learning_rate);
int compute_output_size_conv2d_transpose(int input_size, int kernel_size,
                                         int padding, int stride, int dilation);
int validate_conv2d_transpose_params(const int input_channels, const int output_channels,
                                     const int kernel_size, const int input_height,
                                     const int input_width, const int padding,
                                     const int stride, const int dilation);

#endif
