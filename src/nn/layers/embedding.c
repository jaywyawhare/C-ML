#include "nn/layers/embedding.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

static Tensor* embedding_forward(Module* module, Tensor* input) {
    Embedding* emb = (Embedding*)module;
    if (!emb || !input) return NULL;

    tensor_ensure_executed(input);
    tensor_ensure_executed(emb->weight->tensor);

    // Input contains integer indices
    // Output shape: input_shape + [embedding_dim]
    int out_ndim = input->ndim + 1;
    int* out_shape = malloc((size_t)out_ndim * sizeof(int));
    if (!out_shape) return NULL;
    for (int i = 0; i < input->ndim; i++) out_shape[i] = input->shape[i];
    out_shape[input->ndim] = emb->embedding_dim;

    TensorConfig config = {.dtype = emb->weight->tensor->dtype, .device = input->device,
                           .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(out_shape, out_ndim, &config);
    free(out_shape);
    if (!output) return NULL;
    tensor_ensure_executed(output);

    float* out_data = (float*)tensor_data_ptr(output);
    float* weight_data = (float*)tensor_data_ptr(emb->weight->tensor);

    // Get indices - input could be float (indices stored as floats) or int
    // In this library tensors are float by default, so indices are stored as floats
    float* in_data = (float*)tensor_data_ptr(input);

    int dim = emb->embedding_dim;
    for (size_t i = 0; i < input->numel; i++) {
        int idx = (int)in_data[i];
        if (idx < 0 || idx >= emb->num_embeddings) {
            LOG_ERROR("Embedding index %d out of range [0, %d)", idx, emb->num_embeddings);
            tensor_free(output);
            return NULL;
        }
        memcpy(&out_data[i * dim], &weight_data[idx * dim], (size_t)dim * sizeof(float));
    }

    // Zero out padding_idx if set
    if (emb->padding_idx >= 0 && emb->padding_idx < emb->num_embeddings) {
        for (size_t i = 0; i < input->numel; i++) {
            int idx = (int)in_data[i];
            if (idx == emb->padding_idx) {
                memset(&out_data[i * dim], 0, (size_t)dim * sizeof(float));
            }
        }
    }

    return output;
}

static void embedding_free(Module* module) { free(module); }

Embedding* nn_embedding(int num_embeddings, int embedding_dim, int padding_idx,
                        DType dtype, DeviceType device) {
    Embedding* emb = malloc(sizeof(Embedding));
    if (!emb) return NULL;

    if (module_init((Module*)emb, "Embedding", embedding_forward, embedding_free) != 0) {
        free(emb);
        return NULL;
    }

    emb->num_embeddings = num_embeddings;
    emb->embedding_dim = embedding_dim;
    emb->padding_idx = padding_idx;

    // Create weight tensor [num_embeddings, embedding_dim]
    int weight_shape[] = {num_embeddings, embedding_dim};
    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* weight = tensor_empty(weight_shape, 2, &config);
    if (!weight) {
        module_free((Module*)emb);
        return NULL;
    }

    // Initialize with normal distribution (std = 1.0)
    tensor_ensure_executed(weight);
    float* data = (float*)tensor_data_ptr(weight);
    if (data) {
        for (size_t i = 0; i < weight->numel; i++) {
            data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
        }
        // Zero out padding row if specified
        if (padding_idx >= 0 && padding_idx < num_embeddings) {
            memset(&data[padding_idx * embedding_dim], 0, (size_t)embedding_dim * sizeof(float));
        }
    }

    if (module_add_parameter((Module*)emb, weight, "weight", true) != 0) {
        tensor_free(weight);
        module_free((Module*)emb);
        return NULL;
    }
    emb->weight = module_get_parameter((Module*)emb, "weight");

    LOG_DEBUG("Created Embedding layer: %d x %d", num_embeddings, embedding_dim);
    return emb;
}
