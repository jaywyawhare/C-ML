#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "../../include/Core/memory_management.h"

/**
 * @brief Structure representing an Embedding Layer.
 *
 * @param weights Pointer to the embedding matrix.
 * @param vocab_size Size of the vocabulary.
 * @param embedding_dim Dimension of the embedding vectors.
 */
typedef struct
{
    float *weights;
    int vocab_size;
    int embedding_dim;
} EmbeddingLayer;

/**
 * @brief Initializes an Embedding Layer.
 *
 * @param layer Pointer to the EmbeddingLayer structure.
 * @param vocab_size Size of the vocabulary.
 * @param embedding_dim Dimension of the embedding vectors.
 * @return int Error code.
 */
int initialize_embedding(EmbeddingLayer *layer, int vocab_size, int embedding_dim);

/**
 * @brief Performs the forward pass for the Embedding Layer.
 *
 * @param layer Pointer to the EmbeddingLayer structure.
 * @param input Input indices array.
 * @param output Output embedding vectors array.
 * @param input_size Size of the input indices array.
 * @return int Error code.
 */
int forward_embedding(EmbeddingLayer *layer, const int *input, float *output, int input_size);

/**
 * @brief Performs the backward pass for the Embedding Layer.
 *
 * @param layer Pointer to the EmbeddingLayer structure.
 * @param input Input indices array.
 * @param d_output Gradient of the output.
 * @param d_weights Gradient of the weights.
 * @param input_size Size of the input indices array.
 * @return int Error code.
 */
int backward_embedding(EmbeddingLayer *layer, const int *input, float *d_output, float *d_weights, int input_size);

/**
 * @brief Updates the weights of the Embedding Layer.
 *
 * @param layer Pointer to the EmbeddingLayer structure.
 * @param d_weights Gradient of the weights.
 * @param learning_rate Learning rate for the update.
 * @return int Error code.
 */
int update_embedding(EmbeddingLayer *layer, float *d_weights, float learning_rate);

/**
 * @brief Frees the memory allocated for the Embedding Layer.
 *
 * @param layer Pointer to the EmbeddingLayer structure.
 * @return int Error code.
 */
int free_embedding(EmbeddingLayer *layer);

#endif
