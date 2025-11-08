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

#include "nn/module.h"

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

// ELU Layer
typedef struct ELU {
    Module base;
    float alpha;
    bool inplace;
} ELU;

ELU* nn_elu(float alpha, bool inplace);

// SELU Layer
typedef struct SELU {
    Module base;
    bool inplace;
} SELU;

SELU* nn_selu(bool inplace);

// Swish Layer
typedef struct Swish {
    Module base;
} Swish;

Swish* nn_swish(void);

// Mish Layer
typedef struct Mish {
    Module base;
} Mish;

Mish* nn_mish(void);

// Hard Swish Layer
typedef struct HardSwish {
    Module base;
} HardSwish;

HardSwish* nn_hard_swish(void);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_ACTIVATIONS_H
