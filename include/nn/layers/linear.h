/**
 * @file linear.h
 * @brief Linear (fully connected) layer
 *
 * This header defines the Linear layer, which implements a fully connected
 * neural network layer with optional bias. The layer computes:
 * output = input @ weight + bias
 */

#ifndef CML_NN_LAYERS_LINEAR_H
#define CML_NN_LAYERS_LINEAR_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Linear (fully connected) layer structure
 *
 * Implements a fully connected layer with the transformation:
 * output = input @ weight + bias
 */
typedef struct Linear {
    Module base; // Base module functionality

    int in_features;  // Number of input features
    int out_features; // Number of output features

    Parameter* weight; // Weight parameter matrix
    Parameter* bias;   // Bias parameter vector

    // Layer configuration
    bool use_bias;         // Whether to use bias
    bool transpose_weight; // Whether to transpose weight matrix
} Linear;

// Linear Layer Creation

/**
 * @brief Create a new Linear layer
 *
 * @param in_features Number of input features
 * @param out_features Number of output features
 * @param dtype Data type for parameters
 * @param device Device to place parameters on
 * @param use_bias Whether to include bias term
 * @return New Linear layer, or NULL on failure
 */
Linear* nn_linear(int in_features, int out_features, DType dtype, DeviceType device, bool use_bias);

/**
 * @brief Create a Linear layer with custom weight initialization
 *
 * @param in_features Number of input features
 * @param out_features Number of output features
 * @param dtype Data type for parameters
 * @param device Device to place parameters on
 * @param use_bias Whether to include bias term
 * @param weight_init Weight initialization function
 * @param bias_init Bias initialization function
 * @return New Linear layer, or NULL on failure
 */
Linear* nn_linear_with_init(int in_features, int out_features, DType dtype, DeviceType device,
                            bool use_bias, void (*weight_init)(Tensor*, int, int),
                            void (*bias_init)(Tensor*, int));

// Linear Layer Operations

/**
 * @brief Forward pass through Linear layer
 *
 * @param module Linear layer module
 * @param input Input tensor
 * @return Output tensor, or NULL on failure
 */
Tensor* linear_forward(Module* module, Tensor* input);

/**
 * @brief Get Linear layer input features
 *
 * @param linear Linear layer
 * @return Number of input features
 */
int linear_get_in_features(Linear* linear);

/**
 * @brief Get Linear layer output features
 *
 * @param linear Linear layer
 * @return Number of output features
 */
int linear_get_out_features(Linear* linear);

/**
 * @brief Get Linear layer weight parameter
 *
 * @param linear Linear layer
 * @return Weight parameter
 */
Parameter* linear_get_weight(Linear* linear);

/**
 * @brief Get Linear layer bias parameter
 *
 * @param linear Linear layer
 * @return Bias parameter, or NULL if no bias
 */
Parameter* linear_get_bias(Linear* linear);

/**
 * @brief Set Linear layer weight
 *
 * @param linear Linear layer
 * @param weight New weight tensor
 * @return 0 on success, negative value on failure
 */
int linear_set_weight(Linear* linear, Tensor* weight);

/**
 * @brief Set Linear layer bias
 *
 * @param linear Linear layer
 * @param bias New bias tensor
 * @return 0 on success, negative value on failure
 */
int linear_set_bias(Linear* linear, Tensor* bias);

// Linear Layer Configuration

/**
 * @brief Enable or disable bias in Linear layer
 *
 * @param linear Linear layer
 * @param use_bias Whether to use bias
 */
void linear_set_use_bias(Linear* linear, bool use_bias);

/**
 * @brief Check if Linear layer uses bias
 *
 * @param linear Linear layer
 * @return true if bias is used, false otherwise
 */
bool linear_get_use_bias(Linear* linear);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_LINEAR_H
