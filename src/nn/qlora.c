/**
 * @file qlora.c
 * @brief QLoRA (Quantized Low-Rank Adaptation) implementation
 *
 * Implements QLoRA layers combining NF4-quantized base weights with
 * trainable LoRA adapters. Uses explicit data loops for matmul operations
 * to avoid dependency on higher-level tensor ops.
 */

#include "nn/qlora.h"
#include "core/logging.h"
#include "tensor/tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== NF4 Tensor ===== */

CMLNF4Tensor* cml_nf4_tensor_create(Tensor* float_tensor, int block_size) {
    if (!float_tensor) {
        LOG_ERROR("cml_nf4_tensor_create: NULL tensor");
        return NULL;
    }
    if (block_size <= 0) {
        LOG_ERROR("cml_nf4_tensor_create: block_size must be positive, got %d", block_size);
        return NULL;
    }

    tensor_ensure_executed(float_tensor);
    if (!float_tensor->data) {
        LOG_ERROR("cml_nf4_tensor_create: tensor has no data");
        return NULL;
    }

    CMLNF4Tensor* nf4 = (CMLNF4Tensor*)calloc(1, sizeof(CMLNF4Tensor));
    if (!nf4) {
        LOG_ERROR("cml_nf4_tensor_create: failed to allocate CMLNF4Tensor");
        return NULL;
    }

    nf4->block_size = block_size;
    nf4->original_numel = float_tensor->numel;
    nf4->original_ndim = float_tensor->ndim;

    /* Copy original shape */
    nf4->original_shape = (int*)malloc((size_t)float_tensor->ndim * sizeof(int));
    if (!nf4->original_shape) {
        LOG_ERROR("cml_nf4_tensor_create: failed to allocate shape copy");
        free(nf4);
        return NULL;
    }
    memcpy(nf4->original_shape, float_tensor->shape,
           (size_t)float_tensor->ndim * sizeof(int));

    /* Quantize to NF4 */
    float* scales = NULL;
    int num_scales = 0;
    Tensor* packed = cml_quantize_nf4(float_tensor, block_size, &scales, &num_scales);
    if (!packed) {
        LOG_ERROR("cml_nf4_tensor_create: NF4 quantization failed");
        free(nf4->original_shape);
        free(nf4);
        return NULL;
    }

    nf4->packed_data = packed;
    nf4->scales = scales;
    nf4->num_scales = num_scales;

    LOG_DEBUG("Created NF4 tensor: numel=%zu, packed_size=%zu, blocks=%d",
              nf4->original_numel, packed->numel, num_scales);

    return nf4;
}

void cml_nf4_tensor_free(CMLNF4Tensor* nf4) {
    if (!nf4) return;

    if (nf4->packed_data) {
        tensor_free(nf4->packed_data);
        nf4->packed_data = NULL;
    }
    if (nf4->scales) {
        free(nf4->scales);
        nf4->scales = NULL;
    }
    if (nf4->original_shape) {
        free(nf4->original_shape);
        nf4->original_shape = NULL;
    }
    free(nf4);
}

Tensor* cml_nf4_tensor_dequantize(const CMLNF4Tensor* nf4) {
    if (!nf4) {
        LOG_ERROR("cml_nf4_tensor_dequantize: NULL NF4 tensor");
        return NULL;
    }
    if (!nf4->packed_data || !nf4->scales) {
        LOG_ERROR("cml_nf4_tensor_dequantize: incomplete NF4 tensor");
        return NULL;
    }

    /* Dequantize to flat float32 */
    Tensor* flat = cml_dequantize_nf4(nf4->packed_data, nf4->scales,
                                       nf4->num_scales, nf4->block_size,
                                       nf4->original_numel);
    if (!flat) {
        LOG_ERROR("cml_nf4_tensor_dequantize: dequantization failed");
        return NULL;
    }

    /* If original shape was not 1D, reshape by creating a new tensor with original shape */
    if (nf4->original_ndim != 1 || nf4->original_shape[0] != (int)nf4->original_numel) {
        tensor_ensure_executed(flat);
        float* fdata = (float*)tensor_data_ptr(flat);
        if (!fdata) {
            tensor_free(flat);
            return NULL;
        }

        int* shape_copy = (int*)malloc((size_t)nf4->original_ndim * sizeof(int));
        if (!shape_copy) {
            tensor_free(flat);
            return NULL;
        }
        memcpy(shape_copy, nf4->original_shape,
               (size_t)nf4->original_ndim * sizeof(int));

        TensorConfig config = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                               .has_dtype = true, .has_device = true};
        Tensor* reshaped = tensor_from_data(fdata, shape_copy, nf4->original_ndim, &config);
        free(shape_copy);
        tensor_free(flat);

        if (!reshaped) {
            LOG_ERROR("cml_nf4_tensor_dequantize: reshape failed");
            return NULL;
        }
        return reshaped;
    }

    return flat;
}

/* ===== QLoRA Linear Layer ===== */

CMLQLoRALinear* cml_qlora_linear_create(Tensor* base_weight, int rank,
                                         float alpha, int block_size) {
    if (!base_weight) {
        LOG_ERROR("cml_qlora_linear_create: base_weight is NULL");
        return NULL;
    }
    if (base_weight->ndim != 2) {
        LOG_ERROR("cml_qlora_linear_create: base_weight must be 2D [out_features, in_features], got ndim=%d",
                  base_weight->ndim);
        return NULL;
    }
    if (rank <= 0) {
        LOG_ERROR("cml_qlora_linear_create: rank must be positive, got %d", rank);
        return NULL;
    }
    if (block_size <= 0) {
        LOG_ERROR("cml_qlora_linear_create: block_size must be positive, got %d", block_size);
        return NULL;
    }

    int out_features = base_weight->shape[0];
    int in_features = base_weight->shape[1];

    CMLQLoRALinear* qlora = (CMLQLoRALinear*)calloc(1, sizeof(CMLQLoRALinear));
    if (!qlora) {
        LOG_ERROR("cml_qlora_linear_create: failed to allocate CMLQLoRALinear");
        return NULL;
    }

    qlora->in_features = in_features;
    qlora->out_features = out_features;
    qlora->rank = rank;
    qlora->alpha = alpha;
    qlora->scaling = alpha / (float)rank;
    qlora->enable_double_quant = false;

    /* Quantize base weight to NF4 */
    qlora->base_weight_nf4 = cml_nf4_tensor_create(base_weight, block_size);
    if (!qlora->base_weight_nf4) {
        LOG_ERROR("cml_qlora_linear_create: failed to quantize base weight to NF4");
        free(qlora);
        return NULL;
    }

    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .has_dtype = true,
        .has_device = true
    };

    /* Initialize lora_A with Xavier/small random values */
    int shape_A[2] = {rank, in_features};
    float xavier_scale = sqrtf(2.0f / (float)(rank + in_features));

    qlora->lora_A = tensor_empty(shape_A, 2, &cfg);
    if (!qlora->lora_A) {
        LOG_ERROR("cml_qlora_linear_create: failed to allocate lora_A");
        cml_nf4_tensor_free(qlora->base_weight_nf4);
        free(qlora);
        return NULL;
    }
    tensor_ensure_executed(qlora->lora_A);
    float* data_A = (float*)tensor_data_ptr(qlora->lora_A);
    if (data_A) {
        for (int i = 0; i < rank * in_features; i++) {
            float u1 = (float)rand() / (float)RAND_MAX;
            float u2 = (float)rand() / (float)RAND_MAX;
            /* Box-Muller for approximate normal distribution */
            float z = sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * 3.14159265f * u2);
            data_A[i] = z * xavier_scale;
        }
    }

    /* Initialize lora_B with zeros */
    int shape_B[2] = {out_features, rank};
    qlora->lora_B = tensor_zeros(shape_B, 2, &cfg);
    if (!qlora->lora_B) {
        LOG_ERROR("cml_qlora_linear_create: failed to allocate lora_B");
        tensor_free(qlora->lora_A);
        cml_nf4_tensor_free(qlora->base_weight_nf4);
        free(qlora);
        return NULL;
    }

    LOG_DEBUG("Created QLoRA linear: in=%d, out=%d, rank=%d, alpha=%.1f, scaling=%.4f, block_size=%d",
              in_features, out_features, rank, alpha, qlora->scaling, block_size);

    return qlora;
}

void cml_qlora_linear_free(CMLQLoRALinear* qlora) {
    if (!qlora) return;

    if (qlora->base_weight_nf4) {
        cml_nf4_tensor_free(qlora->base_weight_nf4);
        qlora->base_weight_nf4 = NULL;
    }
    if (qlora->lora_A) {
        tensor_free(qlora->lora_A);
        qlora->lora_A = NULL;
    }
    if (qlora->lora_B) {
        tensor_free(qlora->lora_B);
        qlora->lora_B = NULL;
    }
    free(qlora);
}

Tensor* cml_qlora_linear_forward(CMLQLoRALinear* qlora, Tensor* input) {
    if (!qlora || !input) {
        LOG_ERROR("cml_qlora_linear_forward: NULL argument");
        return NULL;
    }
    if (input->ndim != 2) {
        LOG_ERROR("cml_qlora_linear_forward: input must be 2D [batch, in_features], got ndim=%d",
                  input->ndim);
        return NULL;
    }

    int batch = input->shape[0];
    int in_f = input->shape[1];

    if (in_f != qlora->in_features) {
        LOG_ERROR("cml_qlora_linear_forward: input in_features mismatch: got %d, expected %d",
                  in_f, qlora->in_features);
        return NULL;
    }

    int out_f = qlora->out_features;
    int r = qlora->rank;

    /* Step 1: Dequantize base weight to float32 temporary */
    Tensor* dequant_weight = cml_nf4_tensor_dequantize(qlora->base_weight_nf4);
    if (!dequant_weight) {
        LOG_ERROR("cml_qlora_linear_forward: failed to dequantize base weight");
        return NULL;
    }

    tensor_ensure_executed(input);
    tensor_ensure_executed(dequant_weight);
    tensor_ensure_executed(qlora->lora_A);
    tensor_ensure_executed(qlora->lora_B);

    float* x_data = (float*)tensor_data_ptr(input);
    float* W_data = (float*)tensor_data_ptr(dequant_weight);
    float* A_data = (float*)tensor_data_ptr(qlora->lora_A);
    float* B_data = (float*)tensor_data_ptr(qlora->lora_B);

    if (!x_data || !W_data || !A_data || !B_data) {
        LOG_ERROR("cml_qlora_linear_forward: failed to get data pointers");
        tensor_free(dequant_weight);
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
        LOG_ERROR("cml_qlora_linear_forward: failed to allocate output tensor");
        tensor_free(dequant_weight);
        return NULL;
    }
    tensor_ensure_executed(output);
    float* out_data = (float*)tensor_data_ptr(output);

    /*
     * Step 2: base_out = input @ dequant_weight^T
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

    /* Free the dequantized weight temporary */
    tensor_free(dequant_weight);

    /*
     * Step 3: lora_out = scaling * input @ A^T @ B^T
     *
     * First: tmp = input @ A^T
     * input: [batch, in_f], A: [rank, in_f] => tmp: [batch, rank]
     * tmp[b][r] = sum_i( input[b][i] * A[r][i] )
     */
    float* tmp = (float*)calloc((size_t)batch * (size_t)r, sizeof(float));
    if (!tmp) {
        LOG_ERROR("cml_qlora_linear_forward: failed to allocate temporary buffer");
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
            out_data[b * out_f + o] += qlora->scaling * sum;
        }
    }

    free(tmp);
    return output;
}

size_t cml_qlora_memory_usage(const CMLQLoRALinear* qlora) {
    if (!qlora) return 0;

    size_t mem = 0;

    /* NF4 packed data: original_numel / 2 bytes (uint8) */
    if (qlora->base_weight_nf4) {
        mem += (qlora->base_weight_nf4->original_numel + 1) / 2;  /* packed uint8 */
        mem += (size_t)qlora->base_weight_nf4->num_scales * sizeof(float);  /* scales */
    }

    /* LoRA A: [rank, in_features] * sizeof(float) */
    mem += (size_t)qlora->rank * (size_t)qlora->in_features * sizeof(float);

    /* LoRA B: [out_features, rank] * sizeof(float) */
    mem += (size_t)qlora->out_features * (size_t)qlora->rank * sizeof(float);

    return mem;
}

size_t cml_qlora_full_memory_usage(int in_features, int out_features) {
    return (size_t)in_features * (size_t)out_features * sizeof(float);
}
