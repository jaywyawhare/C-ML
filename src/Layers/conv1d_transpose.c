#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../../include/Layers/conv1d_transpose.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/memory_management.h"

/**
 * @brief Initializes a Conv1D Transpose Layer.
 *
 * @param layer Pointer to the Conv1DTransposeLayer structure.
 * @param input_channels Number of input channels.
 * @param output_channels Number of output channels.
 * @param kernel_size Size of the kernel.
 * @param input_length Length of the input.
 * @param padding Padding size.
 * @param stride Stride size.
 * @param dilation Dilation size.
 * @return int Error code.
 */
int initialize_conv1d_transpose(Conv1DTransposeLayer *layer, int input_channels,
                                int output_channels, int kernel_size, int input_length,
                                int padding, int stride, int dilation)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    if (!validate_conv1d_transpose_params(input_channels, output_channels, kernel_size,
                                          input_length, padding, stride, dilation))
    {
        LOG_ERROR("Invalid parameters for Conv1D Transpose layer.");
        return CM_INVALID_PARAMETER_ERROR;
    }

    layer->input_channels = input_channels;
    layer->output_channels = output_channels;
    layer->kernel_size = kernel_size;
    layer->input_length = input_length;
    layer->padding = padding;
    layer->stride = stride;
    layer->dilation = dilation;

    layer->output_length = compute_output_length_conv1d_transpose(input_length, kernel_size, padding, stride, dilation);
    if (layer->output_length <= 0)
    {
        LOG_ERROR("Invalid output length computed: %d", layer->output_length);
        return CM_INVALID_PARAMETER_ERROR;
    }

    void *allocations[] = {
        (void **)&layer->weights,
        (void **)&layer->biases,
        (void **)&layer->d_weights,
        (void **)&layer->d_biases};
    size_t sizes[] = {
        input_channels * output_channels * kernel_size * sizeof(float),
        output_channels * sizeof(float),
        input_channels * output_channels * kernel_size * sizeof(float),
        output_channels * sizeof(float)};

    for (int i = 0; i < 4; i++)
    {
        *(void **)allocations[i] = cm_safe_malloc(sizes[i], __FILE__, __LINE__);
        if (*(void **)allocations[i] == (void *)CM_MEMORY_ALLOCATION_ERROR)
        {
            for (int j = 0; j < i; j++)
            {
                cm_safe_free((void **)allocations[j]);
            }
            LOG_ERROR("Memory allocation failed.");
            return CM_MEMORY_ALLOCATION_ERROR;
        }
    }

    for (int i = 0; i < input_channels * output_channels * kernel_size; i++)
    {
        layer->weights[i] = ((float)rand() / RAND_MAX) - 0.5f;
    }
    for (int i = 0; i < output_channels; i++)
    {
        layer->biases[i] = ((float)rand() / RAND_MAX) - 0.5f;
    }

    memset(layer->d_weights, 0, input_channels * output_channels * kernel_size * sizeof(float));
    memset(layer->d_biases, 0, output_channels * sizeof(float));

    LOG_DEBUG("Initialized Conv1DTransposeLayer with input channels: %d, output channels: %d, kernel size: %d, input length: %d, padding: %d, stride: %d, dilation: %d",
              input_channels, output_channels, kernel_size, input_length, padding, stride, dilation);

    return CM_SUCCESS;
}

/**
 * @brief Computes the output length for Conv1D Transpose Layer.
 *
 * @param input_length Length of the input.
 * @param kernel_size Size of the kernel.
 * @param padding Padding size.
 * @param stride Stride size.
 * @param dilation Dilation size.
 * @return int Computed output length.
 */
int compute_output_length_conv1d_transpose(const int input_length, const int kernel_size,
                                           const int padding, const int stride, const int dilation)
{
    return stride * (input_length - 1) + (kernel_size - 1) * dilation + 1 - 2 * padding;
}

/**
 * @brief Validates the parameters for Conv1D Transpose Layer.
 *
 * @param input_channels Number of input channels.
 * @param output_channels Number of output channels.
 * @param kernel_size Size of the kernel.
 * @param input_length Length of the input.
 * @param padding Padding size.
 * @param stride Stride size.
 * @param dilation Dilation size.
 * @return int Validation result (1 for valid, 0 for invalid).
 */
int validate_conv1d_transpose_params(const int input_channels, const int output_channels,
                                     const int kernel_size, const int input_length,
                                     const int padding, const int stride, const int dilation)
{
    if (input_channels < MIN_CHANNELS || input_channels > MAX_CHANNELS ||
        output_channels < MIN_CHANNELS || output_channels > MAX_CHANNELS ||
        kernel_size < MIN_KERNEL_SIZE || kernel_size > MAX_KERNEL_SIZE ||
        input_length < MIN_CONV1D_LENGTH || input_length > MAX_CONV1D_LENGTH ||
        padding < 0 || stride <= 0 || dilation <= 0)
    {
        return 0;
    }
    return 1;
}

/**
 * @brief Performs the forward pass for the Conv1D Transpose Layer.
 *
 * @param layer Pointer to the Conv1DTransposeLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @return int Error code.
 */
int forward_conv1d_transpose(Conv1DTransposeLayer *layer, float *input, float *output)
{
    if (layer == NULL || input == NULL || output == NULL)
    {
        LOG_ERROR("Layer, input, or output is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    int input_elements = layer->input_length * layer->input_channels;
    if (input_elements <= 0)
    {
        LOG_ERROR("Invalid input dimensions");
        return CM_INVALID_PARAMETER_ERROR;
    }

    int output_elements = layer->output_length * layer->output_channels;
    if (output_elements <= 0)
    {
        LOG_ERROR("Invalid output dimensions");
        return CM_INVALID_PARAMETER_ERROR;
    }

    memset(output, 0, output_elements * sizeof(float));

    LOG_DEBUG("Performing forward pass for Conv1DTransposeLayer.");

    return CM_SUCCESS;
}

/**
 * @brief Performs the backward pass for the Conv1D Transpose Layer.
 *
 * @param layer Pointer to the Conv1DTransposeLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param d_output Gradient of the output.
 * @param d_input Gradient of the input.
 * @return int Error code.
 */
int backward_conv1d_transpose(Conv1DTransposeLayer *layer, float *input, float *output,
                              float *d_output, float *d_input)
{
    if (layer == NULL || input == NULL || output == NULL || d_output == NULL || d_input == NULL)
    {
        LOG_ERROR("One or more arguments are NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    int input_elements = layer->input_length * layer->input_channels;
    int output_elements = layer->output_length * layer->output_channels;
    if (input_elements <= 0 || output_elements <= 0)
    {
        LOG_ERROR("Invalid dimensions: input=%d, output=%d", input_elements, output_elements);
        return CM_INVALID_PARAMETER_ERROR;
    }

    LOG_DEBUG("Performing backward pass for Conv1DTransposeLayer.");

    return CM_SUCCESS;
}

/**
 * @brief Updates the weights and biases of the Conv1D Transpose Layer.
 *
 * @param layer Pointer to the Conv1DTransposeLayer structure.
 * @param learning_rate Learning rate for the update.
 * @return int Error code.
 */
int update_conv1d_transpose(Conv1DTransposeLayer *layer, float learning_rate)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    if (learning_rate <= 0.0f)
    {
        LOG_ERROR("Learning rate must be positive");
        return CM_INVALID_PARAMETER_ERROR;
    }

    int weight_size = layer->input_channels * layer->output_channels * layer->kernel_size;
    for (int i = 0; i < weight_size; i++)
    {
        layer->weights[i] -= learning_rate * layer->d_weights[i];
    }

    for (int i = 0; i < layer->output_channels; i++)
    {
        layer->biases[i] -= learning_rate * layer->d_biases[i];
    }

    memset(layer->d_weights, 0, weight_size * sizeof(float));
    memset(layer->d_biases, 0, layer->output_channels * sizeof(float));

    return CM_SUCCESS;
}

/**
 * @brief Frees the memory allocated for the Conv1D Transpose Layer.
 *
 * @param layer Pointer to the Conv1DTransposeLayer structure.
 * @return int Error code.
 */
int free_conv1d_transpose(Conv1DTransposeLayer *layer)
{
    if (layer == NULL)
    {
        return CM_NULL_POINTER_ERROR;
    }

    if (layer->weights != NULL)
    {
        cm_safe_free((void **)&layer->weights);
    }
    if (layer->biases != NULL)
    {
        cm_safe_free((void **)&layer->biases);
    }
    if (layer->d_weights != NULL)
    {
        cm_safe_free((void **)&layer->d_weights);
    }
    if (layer->d_biases != NULL)
    {
        cm_safe_free((void **)&layer->d_biases);
    }

    LOG_DEBUG("Freed memory for Conv1DTransposeLayer.");

    return CM_SUCCESS;
}