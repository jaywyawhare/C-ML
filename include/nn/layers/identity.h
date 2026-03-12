#ifndef CML_NN_LAYERS_IDENTITY_H
#define CML_NN_LAYERS_IDENTITY_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Identity {
    Module base;
} Identity;

Identity* nn_identity(void);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_IDENTITY_H
