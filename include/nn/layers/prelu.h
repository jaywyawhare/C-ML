#ifndef CML_NN_LAYERS_PRELU_H
#define CML_NN_LAYERS_PRELU_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PReLU {
    Module base;
    int num_parameters_;
    Parameter* alpha;
} PReLU;

PReLU* nn_prelu(int num_parameters, float init, DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_PRELU_H
