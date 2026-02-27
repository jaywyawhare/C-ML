/**
 * @file embedding.h
 * @brief Embedding lookup table layer
 */

#ifndef CML_NN_LAYERS_EMBEDDING_H
#define CML_NN_LAYERS_EMBEDDING_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Embedding {
    Module base;
    int num_embeddings;
    int embedding_dim;
    int padding_idx; // -1 if no padding
    Parameter* weight; // [num_embeddings, embedding_dim]
} Embedding;

Embedding* nn_embedding(int num_embeddings, int embedding_dim, int padding_idx,
                        DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_EMBEDDING_H
