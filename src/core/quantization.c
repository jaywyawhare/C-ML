/**
 * @file quantization.c
 * @brief Int8/uint8 quantization and dequantization implementation
 */

#include "core/quantization.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

QuantParams cml_quantize_compute_params(Tensor* tensor, bool symmetric) {
    QuantParams params = {.scale = 1.0f, .zero_point = 0};

    if (!tensor) return params;

    tensor_ensure_executed(tensor);
    if (!tensor->data) return params;

    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;

    for (size_t i = 0; i < tensor->numel; i++) {
        float v = tensor_get_float(tensor, i);
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    if (symmetric) {
        float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));
        params.scale = abs_max / 127.0f;
        params.zero_point = 0;
    } else {
        params.scale = (max_val - min_val) / 255.0f;
        if (params.scale < 1e-10f) params.scale = 1e-10f;
        params.zero_point = (int32_t)roundf(-min_val / params.scale) - 128;
    }

    if (params.scale < 1e-10f) params.scale = 1e-10f;

    return params;
}

Tensor* cml_quantize_int8(Tensor* tensor, const QuantParams* params, QuantParams* out_params) {
    if (!tensor) {
        LOG_ERROR("cml_quantize_int8: NULL tensor");
        return NULL;
    }

    tensor_ensure_executed(tensor);
    if (!tensor->data) {
        LOG_ERROR("cml_quantize_int8: tensor has no data");
        return NULL;
    }

    QuantParams qp;
    if (params) {
        qp = *params;
    } else {
        qp = cml_quantize_compute_params(tensor, true);
    }

    if (out_params) *out_params = qp;

    int* shape = tensor_shape_copy(tensor->shape, tensor->ndim);
    if (!shape) return NULL;

    TensorConfig config = {.dtype = DTYPE_INT8, .device = tensor->device,
                           .has_dtype = true, .has_device = true};
    Tensor* quantized = tensor_empty(shape, tensor->ndim, &config);
    free(shape);
    if (!quantized) return NULL;

    int8_t* qdata = (int8_t*)quantized->data;
    for (size_t i = 0; i < tensor->numel; i++) {
        float v = tensor_get_float(tensor, i);
        int32_t q = (int32_t)roundf(v / qp.scale) + qp.zero_point;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        qdata[i] = (int8_t)q;
    }

    return quantized;
}

Tensor* cml_dequantize_int8(Tensor* tensor, const QuantParams* params) {
    if (!tensor || !params) {
        LOG_ERROR("cml_dequantize_int8: NULL tensor or params");
        return NULL;
    }

    tensor_ensure_executed(tensor);
    if (!tensor->data) {
        LOG_ERROR("cml_dequantize_int8: tensor has no data");
        return NULL;
    }

    int* shape = tensor_shape_copy(tensor->shape, tensor->ndim);
    if (!shape) return NULL;

    TensorConfig config = {.dtype = DTYPE_FLOAT32, .device = tensor->device,
                           .has_dtype = true, .has_device = true};
    Tensor* dequantized = tensor_empty(shape, tensor->ndim, &config);
    free(shape);
    if (!dequantized) return NULL;

    int8_t* qdata = (int8_t*)tensor->data;
    float* fdata = (float*)dequantized->data;
    for (size_t i = 0; i < tensor->numel; i++) {
        fdata[i] = ((float)qdata[i] - (float)params->zero_point) * params->scale;
    }

    return dequantized;
}

Tensor* cml_quantize_uint8(Tensor* tensor, const QuantParams* params, QuantParams* out_params) {
    if (!tensor) {
        LOG_ERROR("cml_quantize_uint8: NULL tensor");
        return NULL;
    }

    tensor_ensure_executed(tensor);
    if (!tensor->data) {
        LOG_ERROR("cml_quantize_uint8: tensor has no data");
        return NULL;
    }

    QuantParams qp;
    if (params) {
        qp = *params;
    } else {
        qp = cml_quantize_compute_params(tensor, false);
        // Adjust for uint8 range [0, 255]
        float min_val = FLT_MAX;
        float max_val = -FLT_MAX;
        for (size_t i = 0; i < tensor->numel; i++) {
            float v = tensor_get_float(tensor, i);
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        qp.scale = (max_val - min_val) / 255.0f;
        if (qp.scale < 1e-10f) qp.scale = 1e-10f;
        qp.zero_point = (int32_t)roundf(-min_val / qp.scale);
    }

    if (out_params) *out_params = qp;

    int* shape = tensor_shape_copy(tensor->shape, tensor->ndim);
    if (!shape) return NULL;

    TensorConfig config = {.dtype = DTYPE_UINT8, .device = tensor->device,
                           .has_dtype = true, .has_device = true};
    Tensor* quantized = tensor_empty(shape, tensor->ndim, &config);
    free(shape);
    if (!quantized) return NULL;

    uint8_t* qdata = (uint8_t*)quantized->data;
    for (size_t i = 0; i < tensor->numel; i++) {
        float v = tensor_get_float(tensor, i);
        int32_t q = (int32_t)roundf(v / qp.scale) + qp.zero_point;
        if (q < 0) q = 0;
        if (q > 255) q = 255;
        qdata[i] = (uint8_t)q;
    }

    return quantized;
}

Tensor* cml_dequantize_uint8(Tensor* tensor, const QuantParams* params) {
    if (!tensor || !params) {
        LOG_ERROR("cml_dequantize_uint8: NULL tensor or params");
        return NULL;
    }

    tensor_ensure_executed(tensor);
    if (!tensor->data) {
        LOG_ERROR("cml_dequantize_uint8: tensor has no data");
        return NULL;
    }

    int* shape = tensor_shape_copy(tensor->shape, tensor->ndim);
    if (!shape) return NULL;

    TensorConfig config = {.dtype = DTYPE_FLOAT32, .device = tensor->device,
                           .has_dtype = true, .has_device = true};
    Tensor* dequantized = tensor_empty(shape, tensor->ndim, &config);
    free(shape);
    if (!dequantized) return NULL;

    uint8_t* qdata = (uint8_t*)tensor->data;
    float* fdata = (float*)dequantized->data;
    for (size_t i = 0; i < tensor->numel; i++) {
        fdata[i] = ((float)qdata[i] - (float)params->zero_point) * params->scale;
    }

    return dequantized;
}

/* ===== NF4 Quantization ===== */

const float CML_NF4_TABLE[16] = {
    -1.0f, -0.6962f, -0.5251f, -0.3949f, -0.2844f, -0.1848f, -0.0911f, 0.0f,
     0.0796f, 0.1609f, 0.2461f, 0.3379f, 0.4407f, 0.5626f, 0.7230f, 1.0f
};

/**
 * Find the nearest NF4 table index for a normalized value in [-1, 1].
 * Uses linear scan over the 16-entry table.
 */
static int nf4_find_nearest(float normalized) {
    int best_idx = 0;
    float best_dist = fabsf(normalized - CML_NF4_TABLE[0]);
    for (int i = 1; i < 16; i++) {
        float dist = fabsf(normalized - CML_NF4_TABLE[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

Tensor* cml_quantize_nf4(Tensor* tensor, int block_size,
                          float** out_scales, int* out_num_scales) {
    if (!tensor) {
        LOG_ERROR("cml_quantize_nf4: NULL tensor");
        return NULL;
    }
    if (block_size <= 0) {
        LOG_ERROR("cml_quantize_nf4: block_size must be positive, got %d", block_size);
        return NULL;
    }
    if (!out_scales || !out_num_scales) {
        LOG_ERROR("cml_quantize_nf4: NULL output pointers");
        return NULL;
    }

    tensor_ensure_executed(tensor);
    if (!tensor->data) {
        LOG_ERROR("cml_quantize_nf4: tensor has no data");
        return NULL;
    }

    size_t numel = tensor->numel;
    if (numel == 0) {
        LOG_ERROR("cml_quantize_nf4: tensor has 0 elements");
        return NULL;
    }

    /* Pad numel up to even for packing */
    size_t padded_numel = (numel + 1) & ~(size_t)1;
    size_t packed_size = padded_numel / 2;

    /* Compute number of blocks */
    int num_blocks = (int)((numel + (size_t)block_size - 1) / (size_t)block_size);

    float* scales = (float*)calloc((size_t)num_blocks, sizeof(float));
    if (!scales) {
        LOG_ERROR("cml_quantize_nf4: failed to allocate scales");
        return NULL;
    }

    float* fdata = (float*)tensor->data;

    /* Compute per-block absmax scales */
    for (int b = 0; b < num_blocks; b++) {
        size_t start = (size_t)b * (size_t)block_size;
        size_t end = start + (size_t)block_size;
        if (end > numel) end = numel;

        float absmax = 0.0f;
        for (size_t i = start; i < end; i++) {
            float av = fabsf(fdata[i]);
            if (av > absmax) absmax = av;
        }
        scales[b] = (absmax < 1e-10f) ? 1e-10f : absmax;
    }

    /* Create packed uint8 output tensor */
    int packed_shape[1] = {(int)packed_size};
    TensorConfig config = {.dtype = DTYPE_UINT8, .device = tensor->device,
                           .has_dtype = true, .has_device = true};
    Tensor* packed = tensor_empty(packed_shape, 1, &config);
    if (!packed) {
        free(scales);
        LOG_ERROR("cml_quantize_nf4: failed to allocate packed tensor");
        return NULL;
    }
    tensor_ensure_executed(packed);
    uint8_t* pdata = (uint8_t*)packed->data;
    memset(pdata, 0, packed_size);

    /* Quantize: for each element, normalize by block scale, find nearest NF4 index */
    uint8_t* indices = (uint8_t*)calloc(padded_numel, sizeof(uint8_t));
    if (!indices) {
        free(scales);
        tensor_free(packed);
        LOG_ERROR("cml_quantize_nf4: failed to allocate index buffer");
        return NULL;
    }

    for (size_t i = 0; i < numel; i++) {
        int block_idx = (int)(i / (size_t)block_size);
        float normalized = fdata[i] / scales[block_idx];
        /* Clamp to [-1, 1] */
        if (normalized > 1.0f) normalized = 1.0f;
        if (normalized < -1.0f) normalized = -1.0f;
        indices[i] = (uint8_t)nf4_find_nearest(normalized);
    }
    /* Pad element (if numel is odd) defaults to index 0 from calloc */

    /* Pack two 4-bit indices into one uint8: high nibble = even index, low nibble = odd index */
    for (size_t i = 0; i < packed_size; i++) {
        pdata[i] = (uint8_t)((indices[2 * i] << 4) | (indices[2 * i + 1] & 0x0F));
    }

    free(indices);

    *out_scales = scales;
    *out_num_scales = num_blocks;
    return packed;
}

Tensor* cml_dequantize_nf4(Tensor* nf4_tensor, const float* scales,
                            int num_scales, int block_size, size_t original_numel) {
    if (!nf4_tensor || !scales) {
        LOG_ERROR("cml_dequantize_nf4: NULL argument");
        return NULL;
    }
    if (num_scales <= 0 || block_size <= 0 || original_numel == 0) {
        LOG_ERROR("cml_dequantize_nf4: invalid parameters");
        return NULL;
    }

    tensor_ensure_executed(nf4_tensor);
    if (!nf4_tensor->data) {
        LOG_ERROR("cml_dequantize_nf4: tensor has no data");
        return NULL;
    }

    uint8_t* pdata = (uint8_t*)nf4_tensor->data;

    /* Create float32 output tensor with original shape (flat 1D) */
    int out_shape[1] = {(int)original_numel};
    TensorConfig config = {.dtype = DTYPE_FLOAT32, .device = nf4_tensor->device,
                           .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(out_shape, 1, &config);
    if (!output) {
        LOG_ERROR("cml_dequantize_nf4: failed to allocate output tensor");
        return NULL;
    }
    tensor_ensure_executed(output);
    float* fdata = (float*)output->data;

    /* Unpack and dequantize */
    size_t padded_numel = (original_numel + 1) & ~(size_t)1;
    size_t packed_size = padded_numel / 2;

    for (size_t i = 0; i < packed_size; i++) {
        uint8_t byte = pdata[i];
        int idx_hi = (byte >> 4) & 0x0F;
        int idx_lo = byte & 0x0F;

        size_t elem0 = 2 * i;
        size_t elem1 = 2 * i + 1;

        if (elem0 < original_numel) {
            int block_idx = (int)(elem0 / (size_t)block_size);
            if (block_idx >= num_scales) block_idx = num_scales - 1;
            fdata[elem0] = CML_NF4_TABLE[idx_hi] * scales[block_idx];
        }
        if (elem1 < original_numel) {
            int block_idx = (int)(elem1 / (size_t)block_size);
            if (block_idx >= num_scales) block_idx = num_scales - 1;
            fdata[elem1] = CML_NF4_TABLE[idx_lo] * scales[block_idx];
        }
    }

    return output;
}
