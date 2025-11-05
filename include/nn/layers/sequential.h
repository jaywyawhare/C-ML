/**
 * @file sequential.h
 * @brief Sequential container
 *
 * Implements the nn.Sequential container
 */

#ifndef CML_NN_LAYERS_SEQUENTIAL_H
#define CML_NN_LAYERS_SEQUENTIAL_H

#include "nn/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Sequential {
    Module base;
    Module** modules;
    int num_modules;
    int capacity;
} Sequential;

Sequential* nn_sequential(void);

int sequential_add(Sequential* seq, Module* module);

Module* sequential_get(Sequential* seq, int index);

int sequential_get_length(Sequential* seq);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_SEQUENTIAL_H
