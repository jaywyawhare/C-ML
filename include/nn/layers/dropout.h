/**
 * @file dropout.h
 * @brief Dropout layer declarations
 */

#ifndef CML_NN_LAYERS_DROPOUT_H
#define CML_NN_LAYERS_DROPOUT_H

#include "nn/module.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Dropout layer
 */
typedef struct Dropout {
    Module base;  // Base module
    float p;      // Drop probability (0.0 - 1.0)
    bool inplace; // Whether to apply in-place (kept for API parity)
} Dropout;

/**
 * @brief Create a Dropout layer
 *
 * @param p Drop probability in [0, 1)
 * @param inplace Whether to perform operation in-place (may be ignored)
 * @return Pointer to new Dropout module or NULL on failure
 */
Dropout* nn_dropout(float p, bool inplace);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_DROPOUT_H
