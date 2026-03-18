#ifndef CML_NN_LAYERS_ACTIVATIONS_H
#define CML_NN_LAYERS_ACTIVATIONS_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ReLU {
    Module base;
    bool inplace; // Whether to do in-place operation
} ReLU;

ReLU* nn_relu(bool inplace);

typedef struct LeakyReLU {
    Module base;
    float negative_slope;
    bool inplace;
} LeakyReLU;

LeakyReLU* nn_leaky_relu(float negative_slope, bool inplace);

typedef struct Sigmoid {
    Module base;
} Sigmoid;

Sigmoid* nn_sigmoid(void);

typedef struct Tanh {
    Module base;
} Tanh;

Tanh* nn_tanh(void);

typedef struct GELU {
    Module base;
    bool approximate; // Use approximate GELU formula
} GELU;

GELU* nn_gelu(bool approximate);

typedef struct Softmax {
    Module base;
    int dim; // Dimension along which to apply softmax
} Softmax;

Softmax* nn_softmax(int dim);

typedef struct LogSoftmax {
    Module base;
    int dim; // Dimension along which to apply log-softmax
} LogSoftmax;

LogSoftmax* nn_log_softmax(int dim);

Tensor* f_relu(Tensor* input);

Tensor* f_sigmoid(Tensor* input);

Tensor* f_tanh(Tensor* input);

Tensor* f_gelu(Tensor* input);

typedef struct ELU {
    Module base;
    float alpha;
    bool inplace;
} ELU;

ELU* nn_elu(float alpha, bool inplace);

typedef struct SELU {
    Module base;
} SELU;

SELU* nn_selu(void);

typedef struct SiLU {
    Module base;
} SiLU;

SiLU* nn_silu(void);

typedef struct Mish {
    Module base;
} Mish;

Mish* nn_mish(void);

typedef struct HardSwish {
    Module base;
} HardSwish;

HardSwish* nn_hardswish(void);

Tensor* f_elu(Tensor* input, float alpha);
Tensor* f_selu(Tensor* input);
Tensor* f_silu(Tensor* input);
Tensor* f_mish(Tensor* input);
Tensor* f_hardswish(Tensor* input);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_ACTIVATIONS_H
