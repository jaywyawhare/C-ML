#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../../include/Layers/batchnorm.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/logging.h"

#define EPSILON 1e-7

/**
 * @brief Initializes a Batch Normalization Layer.
 *
 * @param layer Pointer to the BatchNormLayer structure.
 * @param num_features Number of features in the layer.
 * @return int Error code.
 */
int initialize_batchnorm(BatchNormLayer *layer, int num_features)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    if (num_features <= 0)
    {
        LOG_ERROR("Invalid number of features: %d", num_features);
        return CM_INVALID_PARAMETER_ERROR;
    }

    layer->num_features = num_features;
    layer->training_mode = 1;

    layer->gamma = (float *)cm_safe_malloc(num_features * sizeof(float), __FILE__, __LINE__);
    layer->beta = (float *)cm_safe_malloc(num_features * sizeof(float), __FILE__, __LINE__);

    if (layer->gamma == NULL || layer->beta == NULL)
    {
        LOG_ERROR("Memory allocation failed for gamma or beta.");
        cm_safe_free((void **)&layer->gamma);
        cm_safe_free((void **)&layer->beta);
        return CM_MEMORY_ALLOCATION_ERROR;
    }

    for (int i = 0; i < num_features; i++)
    {
        layer->gamma[i] = 1.0f;
        layer->beta[i] = 0.0f;
    }

    LOG_DEBUG("Initialized BatchNormLayer with %d features.", num_features);
    return CM_SUCCESS;
}

/**
 * @brief Frees the memory allocated for the BatchNorm Layer.
 *
 * @param layer Pointer to the BatchNormLayer structure.
 * @return int Error code.
 */
int free_batchnorm(BatchNormLayer *layer)
{
    if (layer == NULL)
    {
        return CM_NULL_POINTER_ERROR;
    }

    cm_safe_free((void **)&layer->gamma);
    cm_safe_free((void **)&layer->beta);

    LOG_DEBUG("Freed memory for BatchNormLayer.");
    return CM_SUCCESS;
}

/**
 * @brief Performs the forward pass for the BatchNorm Layer.
 *
 * @param layer Pointer to the BatchNormLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param mean Mean of the input data.
 * @param variance Variance of the input data.
 * @return int Error code.
 */
int forward_batchnorm(BatchNormLayer *layer, float *input, float *output, float *mean, float *variance)
{
    if (layer == NULL || input == NULL || output == NULL || mean == NULL || variance == NULL)
    {
        LOG_ERROR("One or more arguments are NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    LOG_DEBUG("Performing forward pass for BatchNorm layer with %d features.", layer->num_features);

    for (int i = 0; i < layer->num_features; i++)
    {
        if (variance[i] < EPSILON)
        {
            LOG_ERROR("Variance[%d] is too small, causing potential numerical instability.", i);
            return CM_INVALID_PARAMETER_ERROR;
        }

        output[i] = layer->gamma[i] * ((input[i] - mean[i]) / sqrtf(variance[i] + EPSILON)) + layer->beta[i];
        LOG_DEBUG("Output[%d]: %f", i, output[i]);
    }

    LOG_DEBUG("Performed forward pass for BatchNormLayer.");
    return CM_SUCCESS;
}

/**
 * @brief Performs the backward pass for the BatchNorm Layer.
 *
 * @param layer Pointer to the BatchNormLayer structure.
 * @param input Input data array.
 * @param d_output Gradient of the output.
 * @param d_input Gradient of the input.
 * @param d_gamma Gradient of gamma.
 * @param d_beta Gradient of beta.
 * @param mean Mean of the input data.
 * @param variance Variance of the input data.
 * @return int Error code.
 */
int backward_batchnorm(BatchNormLayer *layer, float *input, float *d_output, float *d_input, float *d_gamma, float *d_beta, float *mean, float *variance)
{
    if (layer == NULL || input == NULL || d_output == NULL || d_input == NULL || d_gamma == NULL || d_beta == NULL || mean == NULL || variance == NULL)
    {
        LOG_ERROR("One or more arguments are NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    LOG_DEBUG("Performing backward pass for BatchNorm layer.");

    for (int i = 0; i < layer->num_features; i++)
    {
        if (variance[i] < EPSILON)
        {
            LOG_ERROR("Variance[%d] is too small for backward pass.", i);
            return CM_INVALID_PARAMETER_ERROR;
        }

        d_gamma[i] += d_output[i] * ((input[i] - mean[i]) / sqrtf(variance[i] + EPSILON));
        d_beta[i] += d_output[i];
        d_input[i] = d_output[i] * layer->gamma[i] / sqrtf(variance[i] + EPSILON);
        LOG_DEBUG("d_input[%d]: %f, d_gamma[%d]: %f, d_beta[%d]: %f", i, d_input[i], i, d_gamma[i], i, d_beta[i]);
    }

    LOG_DEBUG("Performed backward pass for BatchNormLayer.");
    return CM_SUCCESS;
}

/**
 * @brief Updates the parameters of the BatchNorm Layer.
 *
 * @param layer Pointer to the BatchNormLayer structure.
 * @param d_gamma Gradient of gamma.
 * @param d_beta Gradient of beta.
 * @param learning_rate Learning rate for the update.
 * @return int Error code.
 */
int update_batchnorm(BatchNormLayer *layer, float *d_gamma, float *d_beta, float learning_rate)
{
    if (layer == NULL || d_gamma == NULL || d_beta == NULL)
    {
        LOG_ERROR("One or more arguments are NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    LOG_DEBUG("Updating BatchNorm layer parameters.");

    for (int i = 0; i < layer->num_features; i++)
    {
        layer->gamma[i] -= learning_rate * d_gamma[i];
        layer->beta[i] -= learning_rate * d_beta[i];
        LOG_DEBUG("Updated gamma[%d]: %f, beta[%d]: %f", i, layer->gamma[i], i, layer->beta[i]);
    }

    return CM_SUCCESS;
}