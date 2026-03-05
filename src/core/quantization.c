/**
 * @file quantization.c
 * @brief Int8/uint8 quantization and dequantization implementation
 */

#include "core/quantization.h"
#include "core/logging.h"
#include <stdlib.h>
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
