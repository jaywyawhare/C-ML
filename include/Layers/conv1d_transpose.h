#ifndef CONV1D_TRANSPOSE_H
#define CONV1D_TRANSPOSE_H

#include "../../include/Core/memory_management.h"
#include "../../include/Core/error_codes.h"

#define MAX_CONV1D_LENGTH 65536
#define MIN_CONV1D_LENGTH 1
#define MAX_CHANNELS 4096
#define MIN_CHANNELS 1
#define MAX_KERNEL_SIZE 32
#define MIN_KERNEL_SIZE 1

/**
 * @brief Structure representing a 1D Convolution Transpose Layer.
 *
 * @param input_channels Number of input channels.
 * @param output_channels Number of output channels.
 * @param kernel_size Size of the convolution kernel.
 * @param weights Pointer to the weights array.
 * @param biases Pointer to the biases array.
 * @param input_length Length of the input data.
 * @param padding Padding applied to the input data.
 * @param stride Stride of the convolution operation.
 * @param dilation Dilation factor for advanced transpose convolution.
 * @param groups Number of groups for grouped transpose convolution.
 * @param d_weights Pointer to the gradient of the weights.
 * @param d_biases Pointer to the gradient of the biases.
 * @param output_length Cached output length.
 */
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
    int groups;
    float *d_weights;
    float *d_biases;
    int output_length;
} Conv1DTransposeLayer;

/**
 * @brief Initialize a Conv1DTransposeLayer structure.
 *
 * @param layer Pointer to the Conv1DTransposeLayer structure.
 * @param input_channels Number of input channels.
 * @param output_channels Number of output channels.
 * @param kernel_size Size of the convolution kernel.
 * @param input_length Length of the input data.
 * @param padding Padding applied to the input data.
 * @param stride Stride of the convolution operation.
 * @param dilation Dilation factor for advanced transpose convolution.
 * @return Error code indicating success or failure.
 */
int initialize_conv1d_transpose(Conv1DTransposeLayer *layer, int input_channels,
                                int output_channels, int kernel_size, int input_length,
                                int padding, int stride, int dilation);

/**
 * @brief Perform the forward pass of the Conv1DTransposeLayer.
 *
 * @param layer Pointer to the Conv1DTransposeLayer structure.
 * @param input Pointer to the input data array.
 * @param output Pointer to the output data array.
 * @return Error code indicating success or failure.
 */
int forward_conv1d_transpose(Conv1DTransposeLayer *layer, float *input, float *output);

/**
 * @brief Perform the backward pass of the Conv1DTransposeLayer.
 *
 * @param layer Pointer to the Conv1DTransposeLayer structure.
 * @param input Pointer to the input data array.
 * @param output Pointer to the output data array.
 * @param d_output Pointer to the gradient of the output data array.
 * @param d_input Pointer to the gradient of the input data array.
 * @return Error code indicating success or failure.
 */
int backward_conv1d_transpose(Conv1DTransposeLayer *layer, float *input, float *output,
                              float *d_output, float *d_input);

/**
 * @brief Update the weights and biases of the Conv1DTransposeLayer.
 *
 * @param layer Pointer to the Conv1DTransposeLayer structure.
 * @param learning_rate Learning rate for the update.
 * @return Error code indicating success or failure.
 */
int update_conv1d_transpose(Conv1DTransposeLayer *layer,
                            const float learning_rate);

/**
 * @brief Validate the parameters for the Conv1DTransposeLayer.
 *
 * @param input_channels Number of input channels.
 * @param output_channels Number of output channels.
 * @param kernel_size Size of the convolution kernel.
 * @param input_length Length of the input data.
 * @param padding Padding applied to the input data.
 * @param stride Stride of the convolution operation.
 * @param dilation Dilation factor for advanced transpose convolution.
 * @return Error code indicating success or failure.
 */
int validate_conv1d_transpose_params(const int input_channels, const int output_channels,
                                     const int kernel_size, const int input_length,
                                     const int padding, const int stride, const int dilation);

/**
 * @brief Get the output shape of the Conv1DTransposeLayer.
 *
 * @param layer Pointer to the Conv1DTransposeLayer structure.
 * @param output_length Pointer to store the output length.
 * @return Error code indicating success or failure.
 */
int get_conv1d_transpose_output_shape(const Conv1DTransposeLayer *layer,
                                      int *output_length);

/**
 * @brief Compute the output length for the Conv1DTransposeLayer.
 *
 * @param input_length Length of the input data.
 * @param kernel_size Size of the convolution kernel.
 * @param padding Padding applied to the input data.
 * @param stride Stride of the convolution operation.
 * @param dilation Dilation factor for advanced transpose convolution.
 * @return Computed output length.
 */
int compute_output_length_conv1d_transpose(const int input_length, const int kernel_size,
                                           const int padding, const int stride, const int dilation);

/**
 * @brief Free the resources allocated for the Conv1DTransposeLayer.
 *
 * @param layer Pointer to the Conv1DTransposeLayer structure.
 */
int free_conv1d_transpose(Conv1DTransposeLayer *layer);

#endif
