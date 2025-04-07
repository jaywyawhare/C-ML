#ifndef BATCHNORM_H
#define BATCHNORM_H

#include <math.h>
#include "../../include/Core/memory_management.h"

/**
 * @brief Structure representing a Batch Normalization Layer.
 *
 * @param num_features Number of features in the layer.
 * @param training_mode Training mode flag for the layer.
 * @param gamma Scale parameter.
 * @param beta Shift parameter.
 */
typedef struct
{
    int num_features;
    int training_mode;
    float *gamma;
    float *beta;
} BatchNormLayer;

/**
 * @brief Initializes a Batch Normalization Layer.
 *
 * @param layer Pointer to the BatchNormLayer structure.
 * @param num_features Number of features in the layer.
 * @return int Error code.
 */
int initialize_batchnorm(BatchNormLayer *layer, int num_features);

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
int forward_batchnorm(BatchNormLayer *layer, float *input, float *output, float *mean, float *variance);

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
int backward_batchnorm(BatchNormLayer *layer, float *input, float *d_output, float *d_input, float *d_gamma, float *d_beta, float *mean, float *variance);

/**
 * @brief Updates the parameters of the BatchNorm Layer.
 *
 * @param layer Pointer to the BatchNormLayer structure.
 * @param d_gamma Gradient of gamma.
 * @param d_beta Gradient of beta.
 * @param learning_rate Learning rate for the update.
 * @return int Error code.
 */
int update_batchnorm(BatchNormLayer *layer, float *d_gamma, float *d_beta, float learning_rate);

/**
 * @brief Frees the memory allocated for the BatchNorm Layer.
 *
 * @param layer Pointer to the BatchNormLayer structure.
 * @return int Error code.
 */
int free_batchnorm(BatchNormLayer *layer);

#endif
