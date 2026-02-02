/**
 * @file activations.h
 * @brief Activation function layers
 *
 * This header defines activation function layers:
 * - ReLU, LeakyReLU, GELU
 * - Sigmoid, Tanh
 * - Softmax, LogSoftmax
 */

#ifndef CML_NN_LAYERS_ACTIVATIONS_H
#define CML_NN_LAYERS_ACTIVATIONS_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

// ReLU Layer
typedef struct ReLU {
    Module base;
    bool inplace; // Whether to do in-place operation
} ReLU;

ReLU* nn_relu(bool inplace);

// LeakyReLU Layer
typedef struct LeakyReLU {
    Module base;
    float negative_slope;
    bool inplace;
} LeakyReLU;

LeakyReLU* nn_leaky_relu(float negative_slope, bool inplace);

// Sigmoid Layer
typedef struct Sigmoid {
    Module base;
} Sigmoid;

Sigmoid* nn_sigmoid(void);

// Tanh Layer
typedef struct Tanh {
    Module base;
} Tanh;

Tanh* nn_tanh(void);

// GELU Layer
typedef struct GELU {
    Module base;
    bool approximate; // Use approximate GELU formula
} GELU;

GELU* nn_gelu(bool approximate);

// Softmax Layer
typedef struct Softmax {
    Module base;
    int dim; // Dimension along which to apply softmax
} Softmax;

Softmax* nn_softmax(int dim);

// LogSoftmax Layer
typedef struct LogSoftmax {
    Module base;
    int dim; // Dimension along which to apply log-softmax
} LogSoftmax;

LogSoftmax* nn_log_softmax(int dim);

/**
 * @brief Functional ReLU (no module allocation)
 *
 * @param input Input tensor
 * @return Output tensor, or NULL on failure
 */
Tensor* f_relu(Tensor* input);

/**
 * @brief Functional Sigmoid (no module allocation)
 *
 * @param input Input tensor
 * @return Output tensor, or NULL on failure
 */
Tensor* f_sigmoid(Tensor* input);

/**
 * @brief Functional Tanh (no module allocation)
 *
 * @param input Input tensor
 * @return Output tensor, or NULL on failure
 */
Tensor* f_tanh(Tensor* input);

/**
 * @brief Functional GELU (no module allocation)
 *
 * @param input Input tensor
 * @return Output tensor, or NULL on failure
 */
Tensor* f_gelu(Tensor* input);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_ACTIVATIONS_H
