#ifndef CML_NN_LAYERS_DROPOUT_H
#define CML_NN_LAYERS_DROPOUT_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Dropout {
    Module base;  // Base module
    float p;      // Drop probability (0.0 - 1.0)
    bool inplace; // Whether to apply in-place (kept for API parity)
} Dropout;

Dropout* nn_dropout(float p, bool inplace);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_DROPOUT_H
