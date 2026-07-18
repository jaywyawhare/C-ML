#include "nn/qlora.h"
#include "core/threefry.h"
#include "core/logging.h"
#include "core/quantization.h"
#include "tensor/tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "alloc/cml_allocator.h"

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

    CMLNF4Tensor* nf4 = (CMLNF4Tensor*)cml_calloc(1, sizeof(CMLNF4Tensor));
    if (!nf4) {
        LOG_ERROR("cml_nf4_tensor_create: failed to allocate CMLNF4Tensor");
        return NULL;
    }

    nf4->block_size = block_size;
    nf4->original_numel = float_tensor->numel;
    nf4->original_ndim = float_tensor->ndim;

    /* Copy original shape */
    nf4->original_shape = (int*)cml_malloc((size_t)float_tensor->ndim * sizeof(int));
    if (!nf4->original_shape) {
        LOG_ERROR("cml_nf4_tensor_create: failed to allocate shape copy");
        cml_free(nf4);
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
        cml_free(nf4->original_shape);
        cml_free(nf4);
        return NULL;
    }

    nf4->packed_data = packed;
    nf4->scales = scales;
    nf4->num_scales = num_scales;

    return nf4;
}

void cml_nf4_tensor_free(CMLNF4Tensor* nf4) {
    if (!nf4) return;

    if (nf4->packed_data) {
        tensor_free(nf4->packed_data);
        nf4->packed_data = NULL;
    }
    if (nf4->scales) {
        cml_free(nf4->scales);
        nf4->scales = NULL;
    }
    if (nf4->original_shape) {
        cml_free(nf4->original_shape);
        nf4->original_shape = NULL;
    }
    cml_free(nf4);
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

        int* shape_copy = (int*)cml_malloc((size_t)nf4->original_ndim * sizeof(int));
        if (!shape_copy) {
            tensor_free(flat);
            return NULL;
        }
        memcpy(shape_copy, nf4->original_shape,
               (size_t)nf4->original_ndim * sizeof(int));

        TensorConfig config = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                               .has_dtype = true, .has_device = true};
        Tensor* reshaped = tensor_from_data(fdata, shape_copy, nf4->original_ndim, &config);
        cml_free(shape_copy);
        tensor_free(flat);

        if (!reshaped) {
            LOG_ERROR("cml_nf4_tensor_dequantize: reshape failed");
            return NULL;
        }
        return reshaped;
    }

    return flat;
}

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

    CMLQLoRALinear* qlora = (CMLQLoRALinear*)cml_calloc(1, sizeof(CMLQLoRALinear));
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
        cml_free(qlora);
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
        cml_free(qlora);
        return NULL;
    }
    tensor_ensure_executed(qlora->lora_A);
    float* data_A = (float*)tensor_data_ptr(qlora->lora_A);
    if (data_A) {
        CMLRNGState* rng = cml_rng_get_global();
        cml_rng_normal(rng, data_A, (size_t)(rank * in_features));
        for (int i = 0; i < rank * in_features; i++)
            data_A[i] *= xavier_scale;
    }

    /* Initialize lora_B with zeros */
    int shape_B[2] = {out_features, rank};
    qlora->lora_B = tensor_zeros(shape_B, 2, &cfg);
    if (!qlora->lora_B) {
        LOG_ERROR("cml_qlora_linear_create: failed to allocate lora_B");
        tensor_free(qlora->lora_A);
        cml_nf4_tensor_free(qlora->base_weight_nf4);
        cml_free(qlora);
        return NULL;
    }

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
    cml_free(qlora);
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

    const CMLNF4Tensor* nf4 = qlora->base_weight_nf4;
    if (!nf4 || !nf4->packed_data || !nf4->scales) {
        LOG_ERROR("cml_qlora_linear_forward: incomplete NF4 base weight");
        return NULL;
    }

    tensor_ensure_executed(input);
    tensor_ensure_executed(nf4->packed_data);
    tensor_ensure_executed(qlora->lora_A);
    tensor_ensure_executed(qlora->lora_B);

    float* x_data = (float*)tensor_data_ptr(input);
    const uint8_t* pdata = (const uint8_t*)tensor_data_ptr(nf4->packed_data);
    float* A_data = (float*)tensor_data_ptr(qlora->lora_A);
    float* B_data = (float*)tensor_data_ptr(qlora->lora_B);

    if (!x_data || !pdata || !A_data || !B_data) {
        LOG_ERROR("cml_qlora_linear_forward: failed to get data pointers");
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
        return NULL;
    }
    tensor_ensure_executed(output);
    float* out_data = (float*)tensor_data_ptr(output);

    /*
     * base_out = input @ W^T with W dequantized from NF4 one row at a time —
     * the full float32 weight is never materialized, so the NF4 memory saving
     * holds during forward, not just at rest. Peak extra memory: in_f floats.
     * out[b][o] = sum_i( input[b][i] * W[o][i] )
     */
    float* wrow = (float*)cml_malloc((size_t)in_f * sizeof(float));
    if (!wrow) {
        LOG_ERROR("cml_qlora_linear_forward: failed to allocate row buffer");
        tensor_free(output);
        return NULL;
    }
    for (int o = 0; o < out_f; o++) {
        size_t row_base = (size_t)o * (size_t)in_f;
        for (int i = 0; i < in_f; i++) {
            size_t e = row_base + (size_t)i;
            uint8_t byte = pdata[e >> 1];
            /* even flat index = high nibble (matches cml_dequantize_nf4) */
            int idx = (e & 1) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
            int blk = (int)(e / (size_t)nf4->block_size);
            if (blk >= nf4->num_scales) blk = nf4->num_scales - 1;
            wrow[i] = CML_NF4_TABLE[idx] * nf4->scales[blk];
        }
        for (int b = 0; b < batch; b++) {
            const float* xb = x_data + (size_t)b * (size_t)in_f;
            float sum = 0.0f;
            for (int i = 0; i < in_f; i++) {
                sum += xb[i] * wrow[i];
            }
            out_data[b * out_f + o] = sum;
        }
    }
    cml_free(wrow);

    /*
     * Step 3: lora_out = scaling * input @ A^T @ B^T
     *
     * First: tmp = input @ A^T
     * input: [batch, in_f], A: [rank, in_f] => tmp: [batch, rank]
     * tmp[b][r] = sum_i( input[b][i] * A[r][i] )
     */
    float* tmp = (float*)cml_calloc((size_t)batch * (size_t)r, sizeof(float));
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

    cml_free(tmp);
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
