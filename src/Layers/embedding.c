#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../../include/Layers/embedding.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/logging.h"

/**
 * @brief Initializes an Embedding Layer.
 *
 * @param layer Pointer to the EmbeddingLayer structure.
 * @param vocab_size Size of the vocabulary.
 * @param embedding_dim Dimension of the embedding vectors.
 * @return int Error code.
 */
int initialize_embedding(EmbeddingLayer *layer, int vocab_size, int embedding_dim)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    if (vocab_size <= 0 || embedding_dim <= 0)
    {
        LOG_ERROR("Invalid parameters for Embedding layer.");
        return CM_INVALID_PARAMETER_ERROR;
    }

    layer->vocab_size = vocab_size;
    layer->embedding_dim = embedding_dim;

    
    layer->weights = (float *)cm_safe_malloc(vocab_size * embedding_dim * sizeof(float), __FILE__, __LINE__);

    if (layer->weights == (void *)CM_MEMORY_ALLOCATION_ERROR)
    {
        LOG_ERROR("Memory allocation failed for Embedding layer.");
        return CM_MEMORY_ALLOCATION_ERROR;
    }

    
    for (int i = 0; i < vocab_size * embedding_dim; i++)
    {
        layer->weights[i] = ((float)rand() / RAND_MAX) - 0.5f;
    }

    LOG_DEBUG("Embedding layer initialized successfully.");

    return CM_SUCCESS;
}

/**
 * @brief Performs the forward pass for the Embedding Layer.
 *
 * @param layer Pointer to the EmbeddingLayer structure.
 * @param input Input indices array.
 * @param output Output embedding vectors array.
 * @param input_size Size of the input indices array.
 * @return int Error code.
 */
int forward_embedding(EmbeddingLayer *layer, const int *input, float *output, int input_size)
{
    if (layer == NULL || input == NULL || output == NULL)
    {
        LOG_ERROR("Layer, input, or output is NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    if (input_size <= 0)
    {
        LOG_ERROR("Invalid input size.");
        return CM_INVALID_PARAMETER_ERROR;
    }

    for (int i = 0; i < input_size; i++)
    {
        if (input[i] < 0 || input[i] >= layer->vocab_size)
        {
            LOG_ERROR("Input index out of bounds: %d", input[i]);
            return CM_INVALID_PARAMETER_ERROR;
        }

        for (int j = 0; j < layer->embedding_dim; j++)
        {
            output[i * layer->embedding_dim + j] = layer->weights[input[i] * layer->embedding_dim + j];
        }
    }

    return CM_SUCCESS;
}

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
int backward_embedding(EmbeddingLayer *layer, const int *input, float *d_output, float *d_weights, int input_size)
{
    if (layer == NULL || input == NULL || d_output == NULL || d_weights == NULL)
    {
        LOG_ERROR("Layer, input, gradients, or weights are NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    memset(d_weights, 0, layer->vocab_size * layer->embedding_dim * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        for (int j = 0; j < layer->embedding_dim; j++)
        {
            d_weights[input[i] * layer->embedding_dim + j] += d_output[i * layer->embedding_dim + j];
        }
    }

    return CM_SUCCESS;
}

/**
 * @brief Updates the weights of the Embedding Layer.
 *
 * @param layer Pointer to the EmbeddingLayer structure.
 * @param d_weights Gradient of the weights.
 * @param learning_rate Learning rate for the update.
 * @return int Error code.
 */
int update_embedding(EmbeddingLayer *layer, float *d_weights, float learning_rate)
{
    if (layer == NULL || d_weights == NULL)
    {
        LOG_ERROR("Layer or gradients are NULL.");
        return CM_NULL_POINTER_ERROR;
    }

    for (int i = 0; i < layer->vocab_size * layer->embedding_dim; i++)
    {
        layer->weights[i] -= learning_rate * d_weights[i];
    }

    return CM_SUCCESS;
}

/**
 * @brief Frees the memory allocated for the Embedding Layer.
 *
 * @param layer Pointer to the EmbeddingLayer structure.
 * @return int Error code.
 */
int free_embedding(EmbeddingLayer *layer)
{
    if (layer == NULL)
    {
        return CM_NULL_POINTER_ERROR;
    }

    cm_safe_free((void **)&layer->weights);

    return CM_SUCCESS;
}