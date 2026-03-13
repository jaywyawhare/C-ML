/**
 * @file lora.c
 * @brief LoRA (Low-Rank Adaptation) implementation
 *
 * Implements LoRA layers and adapters for efficient fine-tuning.
 * Uses explicit data loops for matmul operations to avoid dependency
 * on higher-level tensor ops that may not support all needed transposes.
 */

#include "nn/lora.h"
#include "core/logging.h"
#include "tensor/tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== LoRA Linear Layer ===== */

CMLLoRALinear* cml_lora_linear_create(Tensor* base_weight, int rank, float alpha) {
    if (!base_weight) {
        LOG_ERROR("base_weight is NULL");
        return NULL;
    }
    if (base_weight->ndim != 2) {
        LOG_ERROR("base_weight must be 2D [out_features, in_features], got ndim=%d",
                  base_weight->ndim);
        return NULL;
    }
    if (rank <= 0) {
        LOG_ERROR("rank must be positive, got %d", rank);
        return NULL;
    }

    int out_features = base_weight->shape[0];
    int in_features = base_weight->shape[1];

    CMLLoRALinear* lora = (CMLLoRALinear*)calloc(1, sizeof(CMLLoRALinear));
    if (!lora) {
        LOG_ERROR("Failed to allocate CMLLoRALinear");
        return NULL;
    }

    lora->in_features = in_features;
    lora->out_features = out_features;
    lora->rank = rank;
    lora->alpha = alpha;
    lora->scaling = alpha / (float)rank;
    lora->base_weight = base_weight;  /* not owned */
    lora->frozen_base = NULL;
    lora->merged = false;

    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .has_dtype = true,
        .has_device = true
    };

    /* Initialize lora_A with Xavier/small random values */
    int shape_A[2] = {rank, in_features};
    float xavier_scale = sqrtf(2.0f / (float)(rank + in_features));

    lora->lora_A = tensor_empty(shape_A, 2, &cfg);
    if (!lora->lora_A) {
        LOG_ERROR("Failed to allocate lora_A");
        free(lora);
        return NULL;
    }
    tensor_ensure_executed(lora->lora_A);
    float* data_A = (float*)tensor_data_ptr(lora->lora_A);
    if (data_A) {
        for (int i = 0; i < rank * in_features; i++) {
            /* Simple pseudo-random: use a deterministic-ish pattern scaled by Xavier */
            float u1 = (float)rand() / (float)RAND_MAX;
            float u2 = (float)rand() / (float)RAND_MAX;
            /* Box-Muller for approximate normal distribution */
            float z = sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * 3.14159265f * u2);
            data_A[i] = z * xavier_scale;
        }
    }

    /* Initialize lora_B with zeros */
    int shape_B[2] = {out_features, rank};
    lora->lora_B = tensor_zeros(shape_B, 2, &cfg);
    if (!lora->lora_B) {
        LOG_ERROR("Failed to allocate lora_B");
        tensor_free(lora->lora_A);
        free(lora);
        return NULL;
    }

    LOG_DEBUG("Created LoRA linear: in=%d, out=%d, rank=%d, alpha=%.1f, scaling=%.4f",
              in_features, out_features, rank, alpha, lora->scaling);

    return lora;
}

void cml_lora_linear_free(CMLLoRALinear* lora) {
    if (!lora) return;

    if (lora->lora_A) {
        tensor_free(lora->lora_A);
        lora->lora_A = NULL;
    }
    if (lora->lora_B) {
        tensor_free(lora->lora_B);
        lora->lora_B = NULL;
    }
    if (lora->frozen_base) {
        tensor_free(lora->frozen_base);
        lora->frozen_base = NULL;
    }
    /* Do NOT free base_weight - we don't own it */
    free(lora);
}

Tensor* cml_lora_linear_forward(CMLLoRALinear* lora, Tensor* input) {
    if (!lora || !input) {
        LOG_ERROR("NULL argument to cml_lora_linear_forward");
        return NULL;
    }
    if (input->ndim != 2) {
        LOG_ERROR("Input must be 2D [batch, in_features], got ndim=%d", input->ndim);
        return NULL;
    }

    int batch = input->shape[0];
    int in_f = input->shape[1];

    if (in_f != lora->in_features) {
        LOG_ERROR("Input in_features mismatch: got %d, expected %d", in_f, lora->in_features);
        return NULL;
    }

    int out_f = lora->out_features;
    int r = lora->rank;

    /* Ensure all tensors are executed and get data pointers */
    tensor_ensure_executed(input);
    tensor_ensure_executed(lora->base_weight);
    tensor_ensure_executed(lora->lora_A);
    tensor_ensure_executed(lora->lora_B);

    float* x_data = (float*)tensor_data_ptr(input);
    float* W_data = (float*)tensor_data_ptr(lora->base_weight);
    float* A_data = (float*)tensor_data_ptr(lora->lora_A);
    float* B_data = (float*)tensor_data_ptr(lora->lora_B);

    if (!x_data || !W_data || !A_data || !B_data) {
        LOG_ERROR("Failed to get data pointers for forward pass");
        return NULL;
    }

    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .has_dtype = true,
        .has_device = true
    };

    /* Allocate output tensor [batch, out_features] */
    int out_shape[2] = {batch, out_f};
    Tensor* output = tensor_zeros(out_shape, 2, &cfg);
    if (!output) {
        LOG_ERROR("Failed to allocate output tensor");
        return NULL;
    }
    tensor_ensure_executed(output);
    float* out_data = (float*)tensor_data_ptr(output);

    /*
     * Step 1: base_out = input @ base_weight^T
     * input: [batch, in_f], W: [out_f, in_f] => out: [batch, out_f]
     * out[b][o] = sum_i( input[b][i] * W[o][i] )
     */
    for (int b = 0; b < batch; b++) {
        for (int o = 0; o < out_f; o++) {
            float sum = 0.0f;
            for (int i = 0; i < in_f; i++) {
                sum += x_data[b * in_f + i] * W_data[o * in_f + i];
            }
            out_data[b * out_f + o] = sum;
        }
    }

    /* If already merged, the LoRA contribution is baked into W, so we're done */
    if (lora->merged) {
        return output;
    }

    /*
     * Step 2: lora_out = scaling * input @ A^T @ B^T
     *
     * First: tmp = input @ A^T
     * input: [batch, in_f], A: [rank, in_f] => tmp: [batch, rank]
     * tmp[b][r] = sum_i( input[b][i] * A[r][i] )
     */
    float* tmp = (float*)calloc((size_t)batch * (size_t)r, sizeof(float));
    if (!tmp) {
        LOG_ERROR("Failed to allocate temporary buffer for LoRA forward");
        tensor_free(output);
        return NULL;
    }

    for (int b = 0; b < batch; b++) {
        for (int ri = 0; ri < r; ri++) {
            float sum = 0.0f;
            for (int i = 0; i < in_f; i++) {
                sum += x_data[b * in_f + i] * A_data[ri * in_f + i];
            }
            tmp[b * r + ri] = sum;
        }
    }

    /*
     * Then: lora_result = tmp @ B^T
     * tmp: [batch, rank], B: [out_f, rank] => lora_result: [batch, out_f]
     * lora_result[b][o] = sum_r( tmp[b][r] * B[o][r] )
     *
     * Add scaling * lora_result to output
     */
    for (int b = 0; b < batch; b++) {
        for (int o = 0; o < out_f; o++) {
            float sum = 0.0f;
            for (int ri = 0; ri < r; ri++) {
                sum += tmp[b * r + ri] * B_data[o * r + ri];
            }
            out_data[b * out_f + o] += lora->scaling * sum;
        }
    }

    free(tmp);
    return output;
}

int cml_lora_linear_merge(CMLLoRALinear* lora) {
    if (!lora) {
        LOG_ERROR("NULL LoRA layer");
        return -1;
    }
    if (lora->merged) {
        LOG_ERROR("LoRA layer is already merged");
        return -1;
    }

    int out_f = lora->out_features;
    int in_f = lora->in_features;
    int r = lora->rank;

    /* Save a frozen copy of the base weight before merging */
    lora->frozen_base = tensor_clone(lora->base_weight);
    if (!lora->frozen_base) {
        LOG_ERROR("Failed to clone base_weight for frozen copy");
        return -1;
    }

    tensor_ensure_executed(lora->base_weight);
    tensor_ensure_executed(lora->lora_A);
    tensor_ensure_executed(lora->lora_B);

    float* W_data = (float*)tensor_data_ptr(lora->base_weight);
    float* A_data = (float*)tensor_data_ptr(lora->lora_A);
    float* B_data = (float*)tensor_data_ptr(lora->lora_B);

    if (!W_data || !A_data || !B_data) {
        LOG_ERROR("Failed to get data pointers for merge");
        tensor_free(lora->frozen_base);
        lora->frozen_base = NULL;
        return -1;
    }

    /*
     * Compute delta = B @ A
     * B: [out_f, rank], A: [rank, in_f] => delta: [out_f, in_f]
     * delta[o][i] = sum_r( B[o][r] * A[r][i] )
     *
     * Then: W[o][i] += scaling * delta[o][i]
     */
    for (int o = 0; o < out_f; o++) {
        for (int i = 0; i < in_f; i++) {
            float sum = 0.0f;
            for (int ri = 0; ri < r; ri++) {
                sum += B_data[o * r + ri] * A_data[ri * in_f + i];
            }
            W_data[o * in_f + i] += lora->scaling * sum;
        }
    }

    lora->merged = true;
    LOG_DEBUG("Merged LoRA into base weight (rank=%d, scaling=%.4f)", r, lora->scaling);
    return 0;
}

int cml_lora_linear_unmerge(CMLLoRALinear* lora) {
    if (!lora) {
        LOG_ERROR("NULL LoRA layer");
        return -1;
    }
    if (!lora->merged) {
        LOG_ERROR("LoRA layer is not merged, cannot unmerge");
        return -1;
    }
    if (!lora->frozen_base) {
        LOG_ERROR("No frozen base weight available for unmerge");
        return -1;
    }

    int out_f = lora->out_features;
    int in_f = lora->in_features;

    tensor_ensure_executed(lora->base_weight);
    tensor_ensure_executed(lora->frozen_base);

    float* W_data = (float*)tensor_data_ptr(lora->base_weight);
    float* frozen_data = (float*)tensor_data_ptr(lora->frozen_base);

    if (!W_data || !frozen_data) {
        LOG_ERROR("Failed to get data pointers for unmerge");
        return -1;
    }

    /* Restore base weight from frozen copy */
    memcpy(W_data, frozen_data, (size_t)out_f * (size_t)in_f * sizeof(float));

    /* Free the frozen copy */
    tensor_free(lora->frozen_base);
    lora->frozen_base = NULL;

    lora->merged = false;
    LOG_DEBUG("Unmerged LoRA from base weight");
    return 0;
}

/* ===== LoRA Adapter ===== */

CMLLoRAAdapter* cml_lora_adapter_create(const char* name, int rank, float alpha) {
    if (!name) {
        LOG_ERROR("Adapter name is NULL");
        return NULL;
    }
    if (rank <= 0) {
        LOG_ERROR("rank must be positive, got %d", rank);
        return NULL;
    }

    CMLLoRAAdapter* adapter = (CMLLoRAAdapter*)calloc(1, sizeof(CMLLoRAAdapter));
    if (!adapter) {
        LOG_ERROR("Failed to allocate CMLLoRAAdapter");
        return NULL;
    }

    strncpy(adapter->name, name, sizeof(adapter->name) - 1);
    adapter->name[sizeof(adapter->name) - 1] = '\0';
    adapter->rank = rank;
    adapter->alpha = alpha;
    adapter->scaling = alpha / (float)rank;
    adapter->num_layers = 0;
    adapter->layers = NULL;
    adapter->merged = false;

    LOG_DEBUG("Created LoRA adapter '%s' (rank=%d, alpha=%.1f)", name, rank, alpha);
    return adapter;
}

void cml_lora_adapter_free(CMLLoRAAdapter* adapter) {
    if (!adapter) return;

    for (int i = 0; i < adapter->num_layers; i++) {
        if (adapter->layers[i]) {
            cml_lora_linear_free(adapter->layers[i]);
        }
    }
    free(adapter->layers);
    free(adapter);
}

int cml_lora_adapter_add_layer(CMLLoRAAdapter* adapter, CMLLoRALinear* layer) {
    if (!adapter || !layer) {
        LOG_ERROR("NULL argument to cml_lora_adapter_add_layer");
        return -1;
    }

    int new_count = adapter->num_layers + 1;
    CMLLoRALinear** new_layers = (CMLLoRALinear**)realloc(
        adapter->layers, (size_t)new_count * sizeof(CMLLoRALinear*));
    if (!new_layers) {
        LOG_ERROR("Failed to reallocate adapter layers array");
        return -1;
    }

    new_layers[adapter->num_layers] = layer;
    adapter->layers = new_layers;
    adapter->num_layers = new_count;

    LOG_DEBUG("Added layer to adapter '%s' (total layers: %d)", adapter->name, new_count);
    return 0;
}

int cml_lora_adapter_merge_all(CMLLoRAAdapter* adapter) {
    if (!adapter) {
        LOG_ERROR("NULL adapter");
        return -1;
    }
    if (adapter->merged) {
        LOG_ERROR("Adapter '%s' is already merged", adapter->name);
        return -1;
    }

    for (int i = 0; i < adapter->num_layers; i++) {
        if (cml_lora_linear_merge(adapter->layers[i]) != 0) {
            LOG_ERROR("Failed to merge layer %d in adapter '%s'", i, adapter->name);
            /* Attempt to unmerge already-merged layers to maintain consistency */
            for (int j = 0; j < i; j++) {
                cml_lora_linear_unmerge(adapter->layers[j]);
            }
            return -1;
        }
    }

    adapter->merged = true;
    LOG_DEBUG("Merged all %d layers in adapter '%s'", adapter->num_layers, adapter->name);
    return 0;
}

int cml_lora_adapter_unmerge_all(CMLLoRAAdapter* adapter) {
    if (!adapter) {
        LOG_ERROR("NULL adapter");
        return -1;
    }
    if (!adapter->merged) {
        LOG_ERROR("Adapter '%s' is not merged, cannot unmerge", adapter->name);
        return -1;
    }

    for (int i = 0; i < adapter->num_layers; i++) {
        if (cml_lora_linear_unmerge(adapter->layers[i]) != 0) {
            LOG_ERROR("Failed to unmerge layer %d in adapter '%s'", i, adapter->name);
            return -1;
        }
    }

    adapter->merged = false;
    LOG_DEBUG("Unmerged all %d layers in adapter '%s'", adapter->num_layers, adapter->name);
    return 0;
}
