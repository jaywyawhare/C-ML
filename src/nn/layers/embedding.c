#include "nn/layers/embedding.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

static Tensor* embedding_forward(Module* module, Tensor* input) {
    Embedding* emb = (Embedding*)module;
    if (!emb || !input)
        return NULL;

    int nidx = (int)input->numel;
    if (nidx <= 0)
        return NULL;

    int flat_shape[] = {nidx};
    ReshapeParams rpf  = {.new_shape = flat_shape, .new_ndim = 1};
    Tensor* idxf       = uop_reshape(input, &rpf);
    if (!idxf)
        return NULL;

    Tensor* gathered = uop_gather(emb->weight->tensor, idxf, 0);
    if (!gathered)
        return NULL;

    Tensor* out_body = gathered;
    if (emb->padding_idx >= 0 && emb->padding_idx < emb->num_embeddings) {
        TensorConfig cfg = {.dtype      = input->dtype,
                            .device     = input->device,
                            .has_dtype  = true,
                            .has_device = true};
        Tensor* padv = tensor_full(flat_shape, 1, &cfg, (float)emb->padding_idx);
        if (!padv)
            return NULL;
        Tensor* mask1 = uop_cmpeq(idxf, padv);
        tensor_free(padv);
        if (!mask1)
            return NULL;

        int m21[] = {nidx, 1};
        ReshapeParams rm = {.new_shape = m21, .new_ndim = 2};
        Tensor* mask2    = uop_reshape(mask1, &rm);
        if (!mask2)
            return NULL;

        int exp_s[] = {nidx, emb->embedding_dim};
        ExpandParams ep = {.new_shape = exp_s, .new_ndim = 2};
        Tensor* mask_e  = uop_expand(mask2, &ep);
        if (!mask_e)
            return NULL;

        Tensor* z = tensor_zeros(exp_s, 2, &cfg);
        if (!z)
            return NULL;

        WhereParams wp = {.cond = mask_e, .a = z, .b = gathered};
        out_body         = uop_where(&wp);
        if (!out_body)
            return NULL;
    }

    int out_ndim = input->ndim + 1;
    int* out_shape = malloc((size_t)out_ndim * sizeof(int));
    if (!out_shape)
        return NULL;
    for (int i = 0; i < input->ndim; i++)
        out_shape[i] = input->shape[i];
    out_shape[input->ndim] = emb->embedding_dim;

    ReshapeParams rout = {.new_shape = out_shape, .new_ndim = out_ndim};
    Tensor* out        = uop_reshape(out_body, &rout);
    free(out_shape);
    return out;
}

static void embedding_free(Module* module) { free(module); }

Embedding* nn_embedding(int num_embeddings, int embedding_dim, int padding_idx,
                        DType dtype, DeviceType device) {
    Embedding* emb = malloc(sizeof(Embedding));
    if (!emb)
        return NULL;

    if (module_init((Module*)emb, "Embedding", embedding_forward, embedding_free) != 0) {
        free(emb);
        return NULL;
    }

    emb->num_embeddings = num_embeddings;
    emb->embedding_dim  = embedding_dim;
    emb->padding_idx    = padding_idx;

    int weight_shape[] = {num_embeddings, embedding_dim};
    TensorConfig config =
        (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* weight = tensor_empty(weight_shape, 2, &config);
    if (!weight) {
        module_free((Module*)emb);
        return NULL;
    }

    tensor_ensure_executed(weight);
    float* data = (float*)tensor_data_ptr(weight);
    if (data) {
        for (size_t i = 0; i < weight->numel; i++) {
            data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
        }
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

    return emb;
}
