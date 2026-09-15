#include "core/quantization.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "alloc/cml_allocator.h"
#include "ops/uops.h"
#include "autograd/autograd.h"

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
    cml_free(shape);
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
    cml_free(shape);
    if (!dequantized) return NULL;

    int8_t* qdata = (int8_t*)tensor->data;
    float* fdata = (float*)dequantized->data;
    for (size_t i = 0; i < tensor->numel; i++) {
        fdata[i] = ((float)qdata[i] - (float)params->zero_point) * params->scale;
    }

    return dequantized;
}

Tensor* cml_quantize_weight_int8(Tensor* weight, bool symmetric) {
    if (!weight) {
        LOG_ERROR("cml_quantize_weight_int8: NULL weight");
        return NULL;
    }
    tensor_ensure_executed(weight);
    if (!weight->data) {
        LOG_ERROR("cml_quantize_weight_int8: weight has no data");
        return NULL;
    }

    QuantParams qp = cml_quantize_compute_params(weight, symmetric);

    int* shape = tensor_shape_copy(weight->shape, weight->ndim);
    if (!shape) return NULL;

    TensorConfig config = {.dtype = DTYPE_INT8, .device = weight->device,
                           .has_dtype = true, .has_device = true};
    Tensor* q = tensor_empty(shape, weight->ndim, &config);
    cml_free(shape);
    if (!q) return NULL;
    tensor_ensure_executed(q);
    if (!q->data) { tensor_free(q); return NULL; }

    int8_t* qdata = (int8_t*)q->data;
    for (size_t i = 0; i < weight->numel; i++) {
        float v = tensor_get_float(weight, i);
        int32_t qi = (int32_t)roundf(v / qp.scale) + qp.zero_point;
        if (qi < -128) qi = -128;
        if (qi > 127) qi = 127;
        qdata[i] = (int8_t)qi;
    }

    q->quant_type       = CML_QUANT_AFFINE_INT8;
    q->quant_scale      = qp.scale;
    q->quant_zero_point = qp.zero_point;
    return q;
}

int cml_qmatmul_affine_int8(const float* x, const int8_t* w, float scale,
                            int32_t zero_point, float* y, int M, int K, int N) {
    if (!x || !w || !y || M <= 0 || K <= 0 || N <= 0)
        return -1;

    for (int m = 0; m < M; m++) {
        float* yr       = y + (size_t)m * N;
        const float* xr = x + (size_t)m * K;
        memset(yr, 0, (size_t)N * sizeof(float));

        float xsum = 0.0f;
        for (int k = 0; k < K; k++) {
            float xmk = xr[k];
            xsum += xmk;
            const int8_t* wr = w + (size_t)k * N;
            /* contiguous in n — the compiler auto-vectorizes this */
            for (int n = 0; n < N; n++)
                yr[n] += xmk * (float)wr[n];
        }

        if (zero_point != 0) {
            float zc = (float)zero_point * xsum;
            for (int n = 0; n < N; n++)
                yr[n] = scale * (yr[n] - zc);
        } else {
            for (int n = 0; n < N; n++)
                yr[n] *= scale;
        }
    }
    return 0;
}

/* ---- packed 4-bit weight-only quantization (AUDIT #19 follow-up) ---- */

static int nf4_find_nearest(float normalized);

/* Nibble order: high nibble = even flat index, low nibble = odd flat index. */
static inline int int4_unpack_signed(const uint8_t* p, size_t idx) {
    uint8_t b   = p[idx >> 1];
    uint8_t nib = (idx & 1) ? (uint8_t)(b & 0x0F) : (uint8_t)(b >> 4);
    return (nib & 0x08) ? (int)nib - 16 : (int)nib;
}

static inline int nf4_unpack_index(const uint8_t* p, size_t idx) {
    uint8_t b = p[idx >> 1];
    return (idx & 1) ? (int)(b & 0x0F) : (int)(b >> 4);
}

Tensor* cml_quantize_weight_int4(Tensor* weight) {
    if (!weight) {
        LOG_ERROR("cml_quantize_weight_int4: NULL weight");
        return NULL;
    }
    tensor_ensure_executed(weight);
    if (!weight->data || weight->numel == 0) {
        LOG_ERROR("cml_quantize_weight_int4: weight has no data");
        return NULL;
    }

    float absmax = 0.0f;
    for (size_t i = 0; i < weight->numel; i++) {
        float av = fabsf(tensor_get_float(weight, i));
        if (av > absmax) absmax = av;
    }
    float scale = absmax / 8.0f;
    if (scale < 1e-10f) scale = 1e-10f;

    size_t packed_size = (weight->numel + 1) / 2;
    uint8_t* packed = (uint8_t*)cml_calloc(packed_size, 1);
    if (!packed) {
        LOG_ERROR("cml_quantize_weight_int4: failed to allocate payload");
        return NULL;
    }
    for (size_t i = 0; i < weight->numel; i++) {
        int32_t q = (int32_t)lrintf(tensor_get_float(weight, i) / scale);
        if (q < -8) q = -8;
        if (q > 7) q = 7;
        uint8_t nib = (uint8_t)(q & 0x0F);
        if ((i & 1) == 0)
            packed[i >> 1] |= (uint8_t)(nib << 4);
        else
            packed[i >> 1] |= nib;
    }

    /* GGUF convention: keep the logical f32 shape/dtype; payload in quant_data
     * so the matmul executor dispatches on quant_type without a dequant pass. */
    int* shape = tensor_shape_copy(weight->shape, weight->ndim);
    if (!shape) {
        cml_free(packed);
        return NULL;
    }
    Tensor* q = tensor_empty(shape, weight->ndim,
                             &(TensorConfig){.dtype = DTYPE_FLOAT32,
                                             .device = weight->device,
                                             .has_dtype = true, .has_device = true});
    cml_free(shape);
    if (!q) {
        cml_free(packed);
        LOG_ERROR("cml_quantize_weight_int4: failed to allocate tensor");
        return NULL;
    }
    q->quant_type       = CML_QUANT_AFFINE_INT4;
    q->quant_data       = packed;
    q->quant_data_bytes = packed_size;
    q->quant_scale      = scale;
    q->quant_zero_point = 0;
    return q;
}

Tensor* cml_quantize_weight_nf4(Tensor* weight, int block_size) {
    if (!weight) {
        LOG_ERROR("cml_quantize_weight_nf4: NULL weight");
        return NULL;
    }
    if (block_size <= 0) {
        LOG_ERROR("cml_quantize_weight_nf4: block_size must be positive, got %d", block_size);
        return NULL;
    }
    tensor_ensure_executed(weight);
    if (!weight->data || weight->numel == 0) {
        LOG_ERROR("cml_quantize_weight_nf4: weight has no data");
        return NULL;
    }

    size_t numel      = weight->numel;
    int num_blocks    = (int)((numel + (size_t)block_size - 1) / (size_t)block_size);
    size_t packed_size = (numel + 1) / 2;

    /* Payload layout: [num_scales floats][packed nibbles] */
    size_t blob_size = (size_t)num_blocks * sizeof(float) + packed_size;
    uint8_t* blob = (uint8_t*)cml_calloc(blob_size, 1);
    if (!blob) {
        LOG_ERROR("cml_quantize_weight_nf4: failed to allocate payload");
        return NULL;
    }
    float* scales = (float*)blob;
    uint8_t* packed = blob + (size_t)num_blocks * sizeof(float);

    const float* fdata = (const float*)weight->data;

    for (int b = 0; b < num_blocks; b++) {
        size_t start = (size_t)b * (size_t)block_size;
        size_t end   = start + (size_t)block_size;
        if (end > numel) end = numel;

        float absmax = 0.0f;
        for (size_t i = start; i < end; i++) {
            float av = fabsf(fdata[i]);
            if (av > absmax) absmax = av;
        }
        scales[b] = (absmax < 1e-10f) ? 1e-10f : absmax;
    }

    for (size_t i = 0; i < numel; i++) {
        float normalized = fdata[i] / scales[i / (size_t)block_size];
        if (normalized > 1.0f) normalized = 1.0f;
        if (normalized < -1.0f) normalized = -1.0f;
        uint8_t nib = (uint8_t)nf4_find_nearest(normalized);
        if ((i & 1) == 0)
            packed[i >> 1] |= (uint8_t)(nib << 4);
        else
            packed[i >> 1] |= nib;
    }

    int* shape = tensor_shape_copy(weight->shape, weight->ndim);
    if (!shape) {
        cml_free(blob);
        return NULL;
    }
    Tensor* q = tensor_empty(shape, weight->ndim,
                             &(TensorConfig){.dtype = DTYPE_FLOAT32,
                                             .device = weight->device,
                                             .has_dtype = true, .has_device = true});
    cml_free(shape);
    if (!q) {
        cml_free(blob);
        LOG_ERROR("cml_quantize_weight_nf4: failed to allocate tensor");
        return NULL;
    }
    q->quant_type       = CML_QUANT_NF4;
    q->quant_data       = blob;
    q->quant_data_bytes = blob_size;
    q->quant_block_size = block_size;
    return q;
}

int cml_qmatmul_affine_int4(const float* x, const uint8_t* w_packed, float scale,
                            int32_t zero_point, float* y, int M, int K, int N) {
    if (!x || !w_packed || !y || M <= 0 || K <= 0 || N <= 0)
        return -1;

    for (int m = 0; m < M; m++) {
        float* yr       = y + (size_t)m * N;
        const float* xr = x + (size_t)m * K;
        memset(yr, 0, (size_t)N * sizeof(float));

        float xsum = 0.0f;
        for (int k = 0; k < K; k++) {
            float xmk = xr[k];
            xsum += xmk;
            size_t base = (size_t)k * (size_t)N;
            for (int n = 0; n < N; n++)
                yr[n] += xmk * (float)int4_unpack_signed(w_packed, base + (size_t)n);
        }

        if (zero_point != 0) {
            float zc = (float)zero_point * xsum;
            for (int n = 0; n < N; n++)
                yr[n] = scale * (yr[n] - zc);
        } else {
            for (int n = 0; n < N; n++)
                yr[n] *= scale;
        }
    }
    return 0;
}

int cml_qmatmul_nf4(const float* x, const uint8_t* w_packed, const float* scales,
                    int num_scales, int block_size, float* y, int M, int K, int N) {
    if (!x || !w_packed || !scales || !y ||
        M <= 0 || K <= 0 || N <= 0 || num_scales <= 0 || block_size <= 0)
        return -1;

    for (int m = 0; m < M; m++) {
        float* yr       = y + (size_t)m * N;
        const float* xr = x + (size_t)m * K;
        memset(yr, 0, (size_t)N * sizeof(float));

        for (int k = 0; k < K; k++) {
            float xmk   = xr[k];
            size_t base = (size_t)k * (size_t)N;
            for (int n = 0; n < N; n++) {
                size_t idx   = base + (size_t)n;
                int blk      = (int)(idx / (size_t)block_size);
                if (blk >= num_scales) blk = num_scales - 1;
                yr[n] += xmk * CML_NF4_TABLE[nf4_unpack_index(w_packed, idx)] * scales[blk];
            }
        }
    }
    return 0;
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
    cml_free(shape);
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
    cml_free(shape);
    if (!dequantized) return NULL;

    uint8_t* qdata = (uint8_t*)tensor->data;
    float* fdata = (float*)dequantized->data;
    for (size_t i = 0; i < tensor->numel; i++) {
        fdata[i] = ((float)qdata[i] - (float)params->zero_point) * params->scale;
    }

    return dequantized;
}


const float CML_NF4_TABLE[16] = {
    -1.0f, -0.6962f, -0.5251f, -0.3949f, -0.2844f, -0.1848f, -0.0911f, 0.0f,
     0.0796f, 0.1609f, 0.2461f, 0.3379f, 0.4407f, 0.5626f, 0.7230f, 1.0f
};

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

    float* scales = (float*)cml_calloc((size_t)num_blocks, sizeof(float));
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
        cml_free(scales);
        LOG_ERROR("cml_quantize_nf4: failed to allocate packed tensor");
        return NULL;
    }
    tensor_ensure_executed(packed);
    uint8_t* pdata = (uint8_t*)packed->data;
    memset(pdata, 0, packed_size);

    /* Quantize: for each element, normalize by block scale, find nearest NF4 index */
    uint8_t* indices = (uint8_t*)cml_calloc(padded_numel, sizeof(uint8_t));
    if (!indices) {
        cml_free(scales);
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

    cml_free(indices);

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

/* ---- quantization-aware training (QAT) ---- */

QatObserver* cml_qat_observer_create(CmlQatObserverMode mode, float momentum) {
    if (mode != CML_QAT_OBS_MINMAX && mode != CML_QAT_OBS_MOVING_AVG_MINMAX) {
        LOG_ERROR("cml_qat_observer_create: unknown mode %d", (int)mode);
        return NULL;
    }
    if (mode == CML_QAT_OBS_MOVING_AVG_MINMAX &&
        !(momentum > 0.0f && momentum <= 1.0f)) {
        LOG_ERROR("cml_qat_observer_create: momentum must be in (0, 1], got %f",
                  (double)momentum);
        return NULL;
    }

    QatObserver* obs = (QatObserver*)cml_calloc(1, sizeof(QatObserver));
    if (!obs) {
        LOG_ERROR("cml_qat_observer_create: failed to allocate observer");
        return NULL;
    }
    obs->mode        = mode;
    obs->momentum    = (mode == CML_QAT_OBS_MOVING_AVG_MINMAX) ? momentum : 1.0f;
    obs->running_min = FLT_MAX;
    obs->running_max = -FLT_MAX;
    return obs;
}

void cml_qat_observer_free(QatObserver* obs) {
    if (!obs) return;
    cml_free(obs);
}

void cml_qat_observer_reset(QatObserver* obs) {
    if (!obs) return;
    obs->running_min  = FLT_MAX;
    obs->running_max  = -FLT_MAX;
    obs->initialized  = false;
    obs->num_updates  = 0;
}

int cml_qat_observer_update(QatObserver* obs, Tensor* tensor) {
    if (!obs || !tensor) {
        LOG_ERROR("cml_qat_observer_update: NULL observer or tensor");
        return -1;
    }
    tensor_ensure_executed(tensor);
    if (!tensor->data || tensor->numel == 0) {
        LOG_ERROR("cml_qat_observer_update: tensor has no data");
        return -1;
    }

    float batch_min = FLT_MAX;
    float batch_max = -FLT_MAX;
    for (size_t i = 0; i < tensor->numel; i++) {
        float v = tensor_get_float(tensor, i);
        if (v < batch_min) batch_min = v;
        if (v > batch_max) batch_max = v;
    }

    if (!obs->initialized) {
        obs->running_min = batch_min;
        obs->running_max = batch_max;
        obs->initialized = true;
    } else if (obs->mode == CML_QAT_OBS_MINMAX) {
        if (batch_min < obs->running_min) obs->running_min = batch_min;
        if (batch_max > obs->running_max) obs->running_max = batch_max;
    } else { /* CML_QAT_OBS_MOVING_AVG_MINMAX: torch MovingAverageMinMax */
        float m = obs->momentum;
        obs->running_min = (1.0f - m) * obs->running_min + m * batch_min;
        obs->running_max = (1.0f - m) * obs->running_max + m * batch_max;
    }

    obs->num_updates++;
    return 0;
}

QuantParams cml_qat_observer_params(const QatObserver* obs) {
    QuantParams params = {.scale = 1e-10f, .zero_point = 0};
    if (!obs || !obs->initialized) {
        LOG_ERROR("cml_qat_observer_params: no calibrated range "
                  "(missing update()?)");
        return params;
    }

    float abs_max = fmaxf(fabsf(obs->running_min), fabsf(obs->running_max));
    params.scale       = abs_max / 127.0f;
    params.zero_point  = 0;
    if (params.scale < 1e-10f) params.scale = 1e-10f;
    return params;
}

/* dequant(quant(t)) as a lazy uop composite; intermediates stay owned by the
 * IR context until teardown (same lifetime convention as the VJP builders). */
static Tensor* qat_fake_quant_composite(Tensor* t, float scale, int32_t zero_point) {
    /* q = clamp(round(t/scale) + zp, -128, 127); out = (q - zp) * scale */
    Tensor* s        = uop_fill(t->shape, t->ndim, scale);
    Tensor* z        = uop_fill(t->shape, t->ndim, (float)zero_point);
    Tensor* scaled   = s ? uop_div(t, s) : NULL;
    Tensor* shifted  = scaled ? uop_add(scaled, z) : NULL;
    Tensor* rounded  = shifted ? uop_round(shifted) : NULL;
    Tensor* clamped  = rounded ? uop_clamp(rounded, -128.0f, 127.0f) : NULL;
    Tensor* centered = clamped ? uop_sub(clamped, z) : NULL;
    Tensor* fq       = centered ? uop_mul(centered, s) : NULL;
    return fq;
}

Tensor* cml_qat_fake_quant(Tensor* tensor, const QuantParams* params) {
    if (!tensor || !params) {
        LOG_ERROR("cml_qat_fake_quant: NULL tensor or params");
        return NULL;
    }
    if (!(params->scale > 0.0f)) {
        LOG_ERROR("cml_qat_fake_quant: scale must be positive, got %f",
                  (double)params->scale);
        return NULL;
    }
    tensor_ensure_executed(tensor);
    if (!tensor->data) {
        LOG_ERROR("cml_qat_fake_quant: tensor has no data");
        return NULL;
    }

    Tensor* fq = qat_fake_quant_composite(tensor, params->scale, params->zero_point);
    if (!fq) return NULL;

    /* STE: out = t + stop_gradient(fq - t). The detach materializes the
     * rounding composite as a plain data leaf, so the backward walk sees a
     * single pass-through edge to `tensor` (identity gradient) and never
     * reaches the zero-derivative ROUND inside the correction term. */
    Tensor* corr = uop_sub(fq, tensor);
    if (!corr) return NULL;
    Tensor* corr_leaf = tensor_detach(corr);
    if (!corr_leaf) {
        LOG_ERROR("cml_qat_fake_quant: failed to detach correction term");
        return NULL;
    }
    return uop_add(tensor, corr_leaf);
}

Tensor* cml_qat_fake_quant_observed(Tensor* tensor, QatObserver* obs) {
    if (!tensor || !obs) {
        LOG_ERROR("cml_qat_fake_quant_observed: NULL tensor or observer");
        return NULL;
    }
    if (cml_qat_observer_update(obs, tensor) != 0) return NULL;
    QuantParams qp = cml_qat_observer_params(obs);
    return cml_qat_fake_quant(tensor, &qp);
}
