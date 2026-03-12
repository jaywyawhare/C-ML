#ifndef CML_NN_LAYERS_FLATTEN_H
#define CML_NN_LAYERS_FLATTEN_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Flatten {
    Module base;
    int start_dim;
    int end_dim;
} Flatten;

Flatten* nn_flatten(int start_dim, int end_dim);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_FLATTEN_H
