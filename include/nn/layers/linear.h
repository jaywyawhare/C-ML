#ifndef CML_NN_LAYERS_LINEAR_H
#define CML_NN_LAYERS_LINEAR_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Linear {
    Module base; // Base module functionality

    int in_features;  // Number of input features
    int out_features; // Number of output features

    Parameter* weight; // Weight parameter matrix
    Parameter* bias;   // Bias parameter vector

    bool use_bias;         // Whether to use bias
    bool transpose_weight; // Whether to transpose weight matrix
} Linear;

Linear* nn_linear(int in_features, int out_features, DType dtype, DeviceType device, bool use_bias);

Linear* nn_linear_with_init(int in_features, int out_features, DType dtype, DeviceType device,
                            bool use_bias, void (*weight_init)(Tensor*, int, int),
                            void (*bias_init)(Tensor*, int));

Tensor* linear_forward(Module* module, Tensor* input);

int linear_get_in_features(Linear* linear);

int linear_get_out_features(Linear* linear);

Parameter* linear_get_weight(Linear* linear);

Parameter* linear_get_bias(Linear* linear);

int linear_set_weight(Linear* linear, Tensor* weight);

int linear_set_bias(Linear* linear, Tensor* bias);

void linear_set_use_bias(Linear* linear, bool use_bias);

bool linear_get_use_bias(Linear* linear);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_LINEAR_H
