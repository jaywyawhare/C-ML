#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <unistd.h>
#include "tensor/tensor.h"
#include "core/serialization.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "backend/backend_buffer.h"
#include "core/logging.h"
#include "backend/device.h"
#include "core/error_stack.h"
#include "core/config.h"
#include "core/threefry.h"

static inline uint16_t float_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static inline float fp16_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t result;
    if (exp == 0) {
        result = sign; /* zero / subnormals → 0 */
    } else if (exp == 31) {
        result = sign | 0x7F800000 | (mant << 13);
    } else {
        result = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &result, sizeof(f));
    return f;
}

static inline uint16_t float_to_bf16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    return (uint16_t)(x >> 16);
}

static inline float bf16_to_float(uint16_t h) {
    uint32_t x = (uint32_t)h << 16;
    float f;
    memcpy(&f, &x, sizeof(f));
    return f;
}

// FP8 E4M3: 1 sign, 4 exponent, 3 mantissa, bias=7, no inf, NaN=0x7F
static inline uint8_t float_to_fp8_e4m3(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint8_t sign = (x >> 24) & 0x80;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 7;
    uint32_t mant = (x >> 20) & 0x07;
    if (exp <= 0) return sign;
    if (exp >= 15) return sign | 0x7E; // max finite: S.1111.110
    return sign | ((uint8_t)exp << 3) | (uint8_t)mant;
}

static inline float fp8_e4m3_to_float(uint8_t h) {
    uint32_t sign = ((uint32_t)(h & 0x80)) << 24;
    uint32_t exp = (h >> 3) & 0x0F;
    uint32_t mant = h & 0x07;
    if (exp == 0) { float f; uint32_t r = sign; memcpy(&f, &r, sizeof(f)); return f; }
    uint32_t result = sign | ((exp - 7 + 127) << 23) | (mant << 20);
    float f;
    memcpy(&f, &result, sizeof(f));
    return f;
}

// FP8 E5M2: 1 sign, 5 exponent, 2 mantissa, bias=15 (like IEEE fp8)
static inline uint8_t float_to_fp8_e5m2(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint8_t sign = (x >> 24) & 0x80;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 21) & 0x03;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C; // inf: S.11111.00
    return sign | ((uint8_t)exp << 2) | (uint8_t)mant;
}

static inline float fp8_e5m2_to_float(uint8_t h) {
    uint32_t sign = ((uint32_t)(h & 0x80)) << 24;
    uint32_t exp = (h >> 2) & 0x1F;
    uint32_t mant = h & 0x03;
    if (exp == 0) { float f; uint32_t r = sign; memcpy(&f, &r, sizeof(f)); return f; }
    if (exp == 31) { float f; uint32_t r = sign | 0x7F800000 | (mant << 21); memcpy(&f, &r, sizeof(f)); return f; }
    uint32_t result = sign | ((exp - 15 + 127) << 23) | (mant << 21);
    float f;
    memcpy(&f, &result, sizeof(f));
    return f;
}

static inline uint8_t float_to_fp8e4m3fnuz(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign_bit = (x >> 31) & 1;
    int32_t fp32_exp = ((x >> 23) & 0xFF);
    uint32_t fp32_mant = x & 0x7FFFFF;

    if (fp32_exp == 0xFF || (fp32_exp == 0 && fp32_mant == 0)) return 0x00;
    if (f == 0.0f || f == -0.0f) return 0x00;

    int32_t exp = fp32_exp - 127 + 8;
    uint32_t mant = (fp32_mant >> 20) & 0x07;

    if (exp <= 0) return 0x00;
    if (exp >= 16) return (uint8_t)((sign_bit << 7) | 0x7F);

    return (uint8_t)((sign_bit << 7) | ((uint8_t)exp << 3) | (uint8_t)mant);
}

static inline float fp8e4m3fnuz_to_float(uint8_t h) {
    if (h == 0x80) return NAN;
    if (h == 0x00) return 0.0f;
    uint32_t sign = ((uint32_t)(h >> 7)) << 31;
    uint32_t exp = (h >> 3) & 0x0F;
    uint32_t mant = h & 0x07;
    if (exp == 0) {
        float f;
        uint32_t r = sign;
        memcpy(&f, &r, sizeof(f));
        return f;
    }
    uint32_t result = sign | ((exp - 8 + 127) << 23) | (mant << 20);
    float f;
    memcpy(&f, &result, sizeof(f));
    return f;
}

static inline uint8_t float_to_fp8e5m2fnuz(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign_bit = (x >> 31) & 1;
    int32_t fp32_exp = ((x >> 23) & 0xFF);
    uint32_t fp32_mant = x & 0x7FFFFF;

    if (fp32_exp == 0xFF || (fp32_exp == 0 && fp32_mant == 0)) return 0x00;
    if (f == 0.0f || f == -0.0f) return 0x00;

    int32_t exp = fp32_exp - 127 + 16;
    uint32_t mant = (fp32_mant >> 21) & 0x03;

    if (exp <= 0) return 0x00;
    if (exp >= 32) return (uint8_t)((sign_bit << 7) | 0x7F);

    return (uint8_t)((sign_bit << 7) | ((uint8_t)exp << 2) | (uint8_t)mant);
}

static inline float fp8e5m2fnuz_to_float(uint8_t h) {
    if (h == 0x80) return NAN;
    if (h == 0x00) return 0.0f;
    uint32_t sign = ((uint32_t)(h >> 7)) << 31;
    uint32_t exp = (h >> 2) & 0x1F;
    uint32_t mant = h & 0x03;
    if (exp == 0) {
        float f;
        uint32_t r = sign;
        memcpy(&f, &r, sizeof(f));
        return f;
    }
    uint32_t result = sign | ((exp - 16 + 127) << 23) | (mant << 21);
    float f;
    memcpy(&f, &result, sizeof(f));
    return f;
}

static void resolve_config(const TensorConfig* config, DType* dtype, DeviceType* device) {
    if (!config) {
        *dtype  = DTYPE_FLOAT32;
        *device = device_get_default();
        return;
    }

    *dtype = config->dtype;
    if (config->device == DEVICE_AUTO) {
        *device = device_get_best_available();
    } else {
        *device = config->device;
    }
}

Tensor* tensor_create(DType dtype, DeviceType device, int ndim, const int* shape,
                      bool requires_grad) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t)
        return NULL;

    t->dtype  = dtype;
    t->device = device;
    t->ndim   = ndim;
    t->shape  = (int*)malloc(ndim * sizeof(int));
    if (!t->shape) {
        free(t);
        return NULL;
    }
    memcpy(t->shape, shape, ndim * sizeof(int));

    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }

    total_size *= cml_dtype_size(dtype);

    t->data = malloc(total_size);
    if (!t->data) {
        free(t->shape);
        free(t);
        return NULL;
    }

    t->requires_grad  = requires_grad;
    t->is_executed    = true;
    t->ir_context     = NULL;
    t->ir_node        = NULL;
    t->grad           = NULL;
    t->ref_count      = 1;
    t->base           = NULL;
    t->strides        = NULL;
    t->storage_offset = 0;
    t->is_contiguous  = true;
    t->buffer_handle  = NULL;
    t->user_data      = NULL;
    t->owns_data      = true;

    t->numel = 1;
    for (int i = 0; i < ndim; i++) {
        t->numel *= shape[i];
    }

    return t;
}

Tensor* tensor_zeros(int* shape, int ndim, const TensorConfig* config) {
    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    Tensor* t = tensor_create(dtype, device, ndim, shape, false);
    if (!t)
        return NULL;

    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }

    memset(t->data, 0, total_size * cml_dtype_size(dtype));
    return t;
}

Tensor* tensor_ones(int* shape, int ndim, const TensorConfig* config) {
    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    Tensor* t = tensor_create(dtype, device, ndim, shape, false);
    if (!t)
        return NULL;

    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }

    switch (dtype) {
    case DTYPE_BOOL:
        memset(t->data, 1, total_size * sizeof(uint8_t));
        break;
    case DTYPE_INT32:
        for (size_t i = 0; i < total_size; i++) {
            ((int32_t*)t->data)[i] = 1;
        }
        break;
    case DTYPE_INT64:
        for (size_t i = 0; i < total_size; i++) {
            ((int64_t*)t->data)[i] = 1;
        }
        break;
    case DTYPE_FLOAT32:
        for (size_t i = 0; i < total_size; i++) {
            ((float*)t->data)[i] = 1.0f;
        }
        break;
    case DTYPE_FLOAT64:
        for (size_t i = 0; i < total_size; i++) {
            ((double*)t->data)[i] = 1.0;
        }
        break;
    case DTYPE_FLOAT16: {
        uint16_t one_fp16 = float_to_fp16(1.0f);
        for (size_t i = 0; i < total_size; i++)
            ((uint16_t*)t->data)[i] = one_fp16;
        break;
    }
    case DTYPE_BFLOAT16: {
        uint16_t one_bf16 = float_to_bf16(1.0f);
        for (size_t i = 0; i < total_size; i++)
            ((uint16_t*)t->data)[i] = one_bf16;
        break;
    }
    case DTYPE_INT8:
        memset(t->data, 1, total_size);
        break;
    case DTYPE_UINT8:
        memset(t->data, 1, total_size);
        break;
    case DTYPE_INT16:
        for (size_t i = 0; i < total_size; i++)
            ((int16_t*)t->data)[i] = 1;
        break;
    case DTYPE_UINT16:
        for (size_t i = 0; i < total_size; i++)
            ((uint16_t*)t->data)[i] = 1;
        break;
    case DTYPE_UINT32:
        for (size_t i = 0; i < total_size; i++)
            ((uint32_t*)t->data)[i] = 1;
        break;
    case DTYPE_UINT64:
        for (size_t i = 0; i < total_size; i++)
            ((uint64_t*)t->data)[i] = 1;
        break;
    case DTYPE_FLOAT8_E4M3: {
        uint8_t one_e4m3 = float_to_fp8_e4m3(1.0f);
        for (size_t i = 0; i < total_size; i++)
            ((uint8_t*)t->data)[i] = one_e4m3;
        break;
    }
    case DTYPE_FLOAT8_E5M2: {
        uint8_t one_e5m2 = float_to_fp8_e5m2(1.0f);
        for (size_t i = 0; i < total_size; i++)
            ((uint8_t*)t->data)[i] = one_e5m2;
        break;
    }
    case DTYPE_FLOAT8_E4M3_FNUZ: {
        uint8_t one_fnuz = float_to_fp8e4m3fnuz(1.0f);
        for (size_t i = 0; i < total_size; i++)
            ((uint8_t*)t->data)[i] = one_fnuz;
        break;
    }
    case DTYPE_FLOAT8_E5M2_FNUZ: {
        uint8_t one_fnuz = float_to_fp8e5m2fnuz(1.0f);
        for (size_t i = 0; i < total_size; i++)
            ((uint8_t*)t->data)[i] = one_fnuz;
        break;
    }
    }

    return t;
}

Tensor* tensor_full(int* shape, int ndim, const TensorConfig* config, float value) {
    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    Tensor* t = tensor_create(dtype, device, ndim, shape, false);
    if (!t)
        return NULL;

    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }

    switch (dtype) {
    case DTYPE_FLOAT32:
        for (size_t i = 0; i < total_size; i++) {
            ((float*)t->data)[i] = (float)value;
        }
        break;
    case DTYPE_FLOAT64:
        for (size_t i = 0; i < total_size; i++) {
            ((double*)t->data)[i] = (double)value;
        }
        break;
    case DTYPE_INT32:
        for (size_t i = 0; i < total_size; i++) {
            ((int32_t*)t->data)[i] = (int32_t)value;
        }
        break;
    case DTYPE_INT64:
        for (size_t i = 0; i < total_size; i++) {
            ((int64_t*)t->data)[i] = (int64_t)value;
        }
        break;
    case DTYPE_BOOL:
        for (size_t i = 0; i < total_size; i++) {
            ((uint8_t*)t->data)[i] = (uint8_t)(fabsf(value) > 1e-9f);
        }
        break;
    case DTYPE_FLOAT16: {
        uint16_t v16 = float_to_fp16(value);
        for (size_t i = 0; i < total_size; i++)
            ((uint16_t*)t->data)[i] = v16;
        break;
    }
    case DTYPE_BFLOAT16: {
        uint16_t vbf = float_to_bf16(value);
        for (size_t i = 0; i < total_size; i++)
            ((uint16_t*)t->data)[i] = vbf;
        break;
    }
    case DTYPE_INT8:
        for (size_t i = 0; i < total_size; i++)
            ((int8_t*)t->data)[i] = (int8_t)value;
        break;
    case DTYPE_UINT8:
        for (size_t i = 0; i < total_size; i++)
            ((uint8_t*)t->data)[i] = (uint8_t)value;
        break;
    case DTYPE_INT16:
        for (size_t i = 0; i < total_size; i++)
            ((int16_t*)t->data)[i] = (int16_t)value;
        break;
    case DTYPE_UINT16:
        for (size_t i = 0; i < total_size; i++)
            ((uint16_t*)t->data)[i] = (uint16_t)value;
        break;
    case DTYPE_UINT32:
        for (size_t i = 0; i < total_size; i++)
            ((uint32_t*)t->data)[i] = (uint32_t)value;
        break;
    case DTYPE_UINT64:
        for (size_t i = 0; i < total_size; i++)
            ((uint64_t*)t->data)[i] = (uint64_t)value;
        break;
    case DTYPE_FLOAT8_E4M3: {
        uint8_t v8 = float_to_fp8_e4m3(value);
        for (size_t i = 0; i < total_size; i++)
            ((uint8_t*)t->data)[i] = v8;
        break;
    }
    case DTYPE_FLOAT8_E5M2: {
        uint8_t v8 = float_to_fp8_e5m2(value);
        for (size_t i = 0; i < total_size; i++)
            ((uint8_t*)t->data)[i] = v8;
        break;
    }
    case DTYPE_FLOAT8_E4M3_FNUZ: {
        uint8_t v8 = float_to_fp8e4m3fnuz(value);
        for (size_t i = 0; i < total_size; i++)
            ((uint8_t*)t->data)[i] = v8;
        break;
    }
    case DTYPE_FLOAT8_E5M2_FNUZ: {
        uint8_t v8 = float_to_fp8e5m2fnuz(value);
        for (size_t i = 0; i < total_size; i++)
            ((uint8_t*)t->data)[i] = v8;
        break;
    }
    }

    return t;
}

void* tensor_data_ptr(Tensor* t) {
    if (!t)
        return NULL;

    // Leaf tensors (no IR node) already have data, just return it
    if (!t->ir_node || !t->ir_context) {
        return t->data;
    }

    // If not executed, execute now! (lazy evaluation)
    if (!t->is_executed) {
        int ret = cml_ir_execute_up_to(t->ir_context, t->ir_node);
        if (ret == 0) {
            t->is_executed = true;
            // JIT execution uses destination-passing style, so data is written
            // directly to t->data by the JIT function. If data is still NULL,
            // allocate it (shouldn't happen, but be safe).
            if (!t->data && t->numel > 0) {
                size_t size = t->numel * cml_dtype_size(t->dtype);
                t->data     = calloc(1, size);
                if (!t->data) {
                    LOG_ERROR("Failed to allocate data after execution");
                    return NULL;
                }
            }
        } else {
            // Execution failed - try to allocate zero-initialized data to prevent crashes
            LOG_WARNING("IR execution failed, returning zero-initialized data");
            if (!t->data && t->numel > 0) {
                size_t size = t->numel * cml_dtype_size(t->dtype);
                t->data     = calloc(1, size);
            }
            t->is_executed = true;
            return t->data;
        }
    }

    return t->data;
}

size_t cml_dtype_size(DType dtype) {
    switch (dtype) {
    case DTYPE_FLOAT32:
        return sizeof(float);
    case DTYPE_FLOAT64:
        return sizeof(double);
    case DTYPE_INT32:
        return sizeof(int32_t);
    case DTYPE_INT64:
        return sizeof(int64_t);
    case DTYPE_BOOL:
        return sizeof(uint8_t);
    case DTYPE_FLOAT16:
    case DTYPE_BFLOAT16:
        return 2;
    case DTYPE_INT8:
    case DTYPE_UINT8:
        return 1;
    case DTYPE_INT16:
    case DTYPE_UINT16:
        return sizeof(int16_t);
    case DTYPE_UINT32:
        return sizeof(uint32_t);
    case DTYPE_UINT64:
        return sizeof(uint64_t);
    case DTYPE_FLOAT8_E4M3:
    case DTYPE_FLOAT8_E5M2:
    case DTYPE_FLOAT8_E4M3_FNUZ:
    case DTYPE_FLOAT8_E5M2_FNUZ:
        return 1;
    default:
        return sizeof(float);
    }
}

DType cml_promote_dtype(DType dtype1, DType dtype2) {
    if (dtype1 == dtype2) {
        return dtype1;
    }

    int rank1 = 0, rank2 = 0;

    // Promotion hierarchy: BOOL < UINT8 < INT8 < INT32 < INT64 < FLOAT16 < BFLOAT16 < FLOAT32 < FLOAT64
    static const int dtype_rank[] = {
        [DTYPE_FLOAT32]          = 9,
        [DTYPE_FLOAT64]          = 10,
        [DTYPE_INT32]            = 5,
        [DTYPE_INT64]            = 7,
        [DTYPE_BOOL]             = 0,
        [DTYPE_FLOAT16]          = 8,
        [DTYPE_BFLOAT16]         = 8,
        [DTYPE_INT8]             = 2,
        [DTYPE_UINT8]            = 1,
        [DTYPE_INT16]            = 3,
        [DTYPE_UINT16]           = 3,
        [DTYPE_UINT32]           = 5,
        [DTYPE_UINT64]           = 6,
        [DTYPE_FLOAT8_E4M3]      = 8,
        [DTYPE_FLOAT8_E5M2]      = 8,
        [DTYPE_FLOAT8_E4M3_FNUZ] = 8,
        [DTYPE_FLOAT8_E5M2_FNUZ] = 8,
    };
    rank1 = (dtype1 <= DTYPE_FLOAT8_E5M2_FNUZ) ? dtype_rank[dtype1] : 0;
    rank2 = (dtype2 <= DTYPE_FLOAT8_E5M2_FNUZ) ? dtype_rank[dtype2] : 0;

    return (rank1 > rank2) ? dtype1 : dtype2;
}

size_t tensor_numel(int* shape, int ndim) {
    size_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        numel *= (size_t)shape[i];
    }
    return numel;
}

size_t* compute_contiguous_strides(int* shape, int ndim) {
    if (!shape || ndim < 0)
        return NULL;

    if (ndim == 0) {
        // Scalar tensor has no strides, but some functions might expect a non-NULL pointer
        return (size_t*)malloc(sizeof(size_t));
    }

    size_t* strides = (size_t*)malloc((size_t)ndim * sizeof(size_t));
    if (!strides) {
        LOG_ERROR("Failed to allocate memory for strides");
        return NULL;
    }

    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * (size_t)shape[i + 1];
    }

    return strides;
}

bool tensor_check_is_contiguous(int* shape, size_t* strides, int ndim) {
    if (!shape || !strides || ndim <= 0)
        return false;

    size_t expected_stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        if (strides[i] != expected_stride)
            return false;
        expected_stride *= (size_t)shape[i];
    }

    return true;
}

size_t tensor_compute_storage_size(int* shape, size_t* strides, int ndim) {
    if (!shape || !strides || ndim <= 0)
        return 0;

    size_t max_offset = 0;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] > 1) {
            max_offset += (size_t)(shape[i] - 1) * strides[i];
        }
    }

    return max_offset + 1;
}

int* tensor_shape_copy(int* shape, int ndim) {
    int* new_shape = (int*)malloc((size_t)ndim * sizeof(int));
    if (!new_shape) {
        LOG_ERROR("Failed to allocate memory for tensor shape copy");
        return NULL;
    }
    memcpy(new_shape, shape, (size_t)ndim * sizeof(int));
    return new_shape;
}

float tensor_get_float(Tensor* t, size_t idx) {
    if (t && !t->is_executed) {
        void* data = tensor_data_ptr(t);
        if (!data)
            return 0.0f;
    }
    if (!t || !t->data || idx >= t->numel)
        return 0.0f;

    size_t offset = t->storage_offset;
    if (!t->is_contiguous) {
        size_t temp = idx;
        for (int d = t->ndim - 1; d >= 0; d--) {
            size_t coord = temp % (size_t)t->shape[d];
            temp /= (size_t)t->shape[d];
            offset += coord * t->strides[d];
        }
    } else {
        offset += idx;
    }

    switch (t->dtype) {
    case DTYPE_FLOAT32:
        return ((float*)t->data)[offset];
    case DTYPE_FLOAT64:
        return (float)((double*)t->data)[offset];
    case DTYPE_INT32:
        return (float)((int32_t*)t->data)[offset];
    case DTYPE_INT64:
        return (float)((int64_t*)t->data)[offset];
    case DTYPE_BOOL:
        return (float)((uint8_t*)t->data)[offset];
    case DTYPE_FLOAT16:
        return fp16_to_float(((uint16_t*)t->data)[offset]);
    case DTYPE_BFLOAT16:
        return bf16_to_float(((uint16_t*)t->data)[offset]);
    case DTYPE_INT8:
        return (float)((int8_t*)t->data)[offset];
    case DTYPE_UINT8:
        return (float)((uint8_t*)t->data)[offset];
    case DTYPE_INT16:
        return (float)((int16_t*)t->data)[offset];
    case DTYPE_UINT16:
        return (float)((uint16_t*)t->data)[offset];
    case DTYPE_UINT32:
        return (float)((uint32_t*)t->data)[offset];
    case DTYPE_UINT64:
        return (float)((uint64_t*)t->data)[offset];
    case DTYPE_FLOAT8_E4M3:
        return fp8_e4m3_to_float(((uint8_t*)t->data)[offset]);
    case DTYPE_FLOAT8_E5M2:
        return fp8_e5m2_to_float(((uint8_t*)t->data)[offset]);
    case DTYPE_FLOAT8_E4M3_FNUZ:
        return fp8e4m3fnuz_to_float(((uint8_t*)t->data)[offset]);
    case DTYPE_FLOAT8_E5M2_FNUZ:
        return fp8e5m2fnuz_to_float(((uint8_t*)t->data)[offset]);
    default:
        return 0.0f;
    }
}

void tensor_set_float(Tensor* t, size_t idx, float value) {
    if (!t || idx >= t->numel)
        return;
    if (!t->is_executed) {
        void* data = tensor_data_ptr(t);
        if (!data)
            return;
    }
    if (!t->data)
        return;

    size_t offset = t->storage_offset;
    if (!t->is_contiguous) {
        size_t temp = idx;
        for (int d = t->ndim - 1; d >= 0; d--) {
            size_t coord = temp % (size_t)t->shape[d];
            temp /= (size_t)t->shape[d];
            offset += coord * t->strides[d];
        }
    } else {
        offset += idx;
    }

    switch (t->dtype) {
    case DTYPE_FLOAT32:
        ((float*)t->data)[offset] = value;
        break;
    case DTYPE_FLOAT64:
        ((double*)t->data)[offset] = (double)value;
        break;
    case DTYPE_INT32:
        ((int32_t*)t->data)[offset] = (int32_t)value;
        break;
    case DTYPE_INT64:
        ((int64_t*)t->data)[offset] = (int64_t)value;
        break;
    case DTYPE_BOOL:
        ((uint8_t*)t->data)[offset] = (uint8_t)(fabsf(value) > 1e-9f);
        break;
    case DTYPE_FLOAT16:
        ((uint16_t*)t->data)[offset] = float_to_fp16(value);
        break;
    case DTYPE_BFLOAT16:
        ((uint16_t*)t->data)[offset] = float_to_bf16(value);
        break;
    case DTYPE_INT8:
        ((int8_t*)t->data)[offset] = (int8_t)value;
        break;
    case DTYPE_UINT8:
        ((uint8_t*)t->data)[offset] = (uint8_t)value;
        break;
    case DTYPE_INT16:
        ((int16_t*)t->data)[offset] = (int16_t)value;
        break;
    case DTYPE_UINT16:
        ((uint16_t*)t->data)[offset] = (uint16_t)value;
        break;
    case DTYPE_UINT32:
        ((uint32_t*)t->data)[offset] = (uint32_t)value;
        break;
    case DTYPE_UINT64:
        ((uint64_t*)t->data)[offset] = (uint64_t)value;
        break;
    case DTYPE_FLOAT8_E4M3:
        ((uint8_t*)t->data)[offset] = float_to_fp8_e4m3(value);
        break;
    case DTYPE_FLOAT8_E5M2:
        ((uint8_t*)t->data)[offset] = float_to_fp8_e5m2(value);
        break;
    case DTYPE_FLOAT8_E4M3_FNUZ:
        ((uint8_t*)t->data)[offset] = float_to_fp8e4m3fnuz(value);
        break;
    case DTYPE_FLOAT8_E5M2_FNUZ:
        ((uint8_t*)t->data)[offset] = float_to_fp8e5m2fnuz(value);
        break;
    }
}

Tensor* tensor_from_ir_node(struct IRNode* node, CMLGraph_t ir_context) {
    if (!node || !ir_context)
        return NULL;

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t)
        return NULL;

    t->ir_node    = node;
    t->ir_context = ir_context;

    node->output = t;

    // Shape from broadcasting
    if (node->output_shape) {
        t->shape = tensor_shape_copy(node->output_shape, node->output_ndim);
        if (!t->shape) {
            free(t);
            return NULL;
        }
        t->ndim  = node->output_ndim;
        t->numel = tensor_numel(node->output_shape, node->output_ndim);
    } else {
        // Fallback: use first input's shape if available
        if (node->inputs && node->inputs[0]) {
            t->shape = tensor_shape_copy(node->inputs[0]->shape, node->inputs[0]->ndim);
            if (!t->shape) {
                free(t);
                return NULL;
            }
            t->ndim  = node->inputs[0]->ndim;
            t->numel = node->inputs[0]->numel;
        } else {
            free(t);
            return NULL;
        }
    }

    if (node->inputs && node->inputs[0]) {
        t->dtype  = node->inputs[0]->dtype;
        t->device = node->inputs[0]->device;
    } else {
        t->dtype  = DTYPE_FLOAT32;
        t->device = DEVICE_CPU;
    }

    // Execution state (lazy)
    t->is_executed = false;
    t->data        = NULL;
    t->owns_data   = true; // Will own data when executed

    // Autograd
    t->requires_grad = node->requires_grad;
    t->grad          = NULL;

    // Memory management
    t->ref_count = 1;
    t->base      = NULL;

    t->strides = compute_contiguous_strides(t->shape, t->ndim);
    if (!t->strides && t->ndim > 0) {
        free(t->shape);
        free(t);
        return NULL;
    }
    t->storage_offset = 0;
    t->is_contiguous  = true;
    t->buffer_handle  = NULL;
    t->user_data      = NULL;

    node->output = t;

    return t;
}

int tensor_ensure_executed(Tensor* t) {
    if (!t)
        return -1;
    if (t->is_executed)
        return 0;

    void* data = tensor_data_ptr(t);
    return data ? 0 : -1;
}

CMLGraph_t tensor_get_ir_context(Tensor* t) { return t ? t->ir_context : NULL; }

bool tensor_is_scalar(Tensor* t) { return t && t->ndim == 0; }

bool tensor_is_contiguous(Tensor* t) {
    if (!t)
        return false;
    return t->is_contiguous;
}

Tensor* tensor_empty(int* shape, int ndim, const TensorConfig* config) {
    if (!shape || ndim < 0) {
        error_stack_push(CM_INVALID_ARGUMENT,
                         "Invalid arguments to tensor_empty: shape is NULL or ndim < 0", __FILE__,
                         __LINE__, __func__);
        return NULL;
    }

    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR,
                         "Failed to allocate memory for Tensor structure", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    t->shape = tensor_shape_copy(shape, ndim);
    if (!t->shape) {
        LOG_ERROR("Failed to allocate memory for tensor shape copy in tensor_empty");
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for tensor shape",
                         __FILE__, __LINE__, __func__);
        free(t);
        return NULL;
    }

    t->strides = compute_contiguous_strides(shape, ndim);
    if (!t->strides) {
        LOG_ERROR("Failed to allocate memory for tensor strides in tensor_empty");
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for tensor strides",
                         __FILE__, __LINE__, __func__);
        free(t->shape);
        free(t);
        return NULL;
    }

    t->ndim           = ndim;
    t->numel          = tensor_numel(shape, ndim);
    t->dtype          = dtype;
    t->device         = device;
    t->storage_offset = 0;

    // IR fields (leaf tensors are not IR nodes)
    t->ir_node     = NULL;
    t->ir_context  = NULL;
    t->is_executed = true; // Leaf tensors have data immediately
    t->data        = NULL;
    t->owns_data   = true;

    t->is_contiguous = true;
    t->buffer_handle = NULL;

    CMLBackendBufferType_t buft = cml_backend_buffer_type_for_device(device);
    if (buft) {
        size_t alloc_size = cml_backend_buffer_type_get_alloc_size(buft, t);
        if (alloc_size == 0) {
            alloc_size = t->numel * cml_dtype_size(dtype);
        }

        CMLBackendBuffer_t buffer = cml_backend_buffer_type_alloc_buffer(buft, alloc_size);
        fflush(stderr);
        if (!buffer) {
            LOG_ERROR("Failed to allocate backend buffer of size %zu for device %s", alloc_size,
                      device_get_name(device));
            char error_msg[256];
            snprintf(error_msg, sizeof(error_msg),
                     "Failed to allocate %zu bytes for tensor data on device %s", alloc_size,
                     device_get_name(device));
            error_stack_push(CM_MEMORY_ALLOCATION_ERROR, error_msg, __FILE__, __LINE__, __func__);
            free(t->strides);
            free(t->shape);
            free(t);
            return NULL;
        }

        if (cml_backend_buffer_init_tensor(buffer, t) != 0) {
            LOG_ERROR("Backend buffer initialization failed for device %s",
                      device_get_name(device));
            cml_backend_buffer_free(buffer);
            free(t->strides);
            free(t->shape);
            free(t);
            return NULL;
        }

        t->buffer_handle = buffer;
    } else {
        size_t data_size = t->numel * cml_dtype_size(dtype);
        t->data          = malloc(data_size);
        if (!t->data) {
            LOG_ERROR("Failed to allocate fallback memory for tensor data");
            char error_msg[256];
            snprintf(error_msg, sizeof(error_msg),
                     "Failed to allocate %zu bytes for tensor data (fallback)", data_size);
            error_stack_push(CM_MEMORY_ALLOCATION_ERROR, error_msg, __FILE__, __LINE__, __func__);
            free(t->strides);
            free(t->shape);
            free(t);
            return NULL;
        }
    }

    t->grad          = NULL;
    t->requires_grad = false;
    t->ref_count     = 1;
    t->base          = NULL;

    return t;
}

Tensor* tensor_from_data(const void* data, int* shape, int ndim, const TensorConfig* config) {
    if (!data) {
        error_stack_push(CM_INVALID_ARGUMENT, "Invalid argument to tensor_from_data: data is NULL",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    Tensor* t = tensor_empty(shape, ndim, config);
    if (!t)
        return NULL;

    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    // Use device-specific copy (assumes source data is on CPU)
    size_t data_size = t->numel * cml_dtype_size(dtype);
    int result       = device_copy_to_device(t->data, data, data_size, device);
    if (result != 0) {
        LOG_ERROR("Failed to copy data to device %s", device_get_name(device));
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "Failed to copy data to device %s",
                 device_get_name(device));
        error_stack_push(CM_OPERATION_FAILED, error_msg, __FILE__, __LINE__, __func__);
        tensor_free(t);
        return NULL;
    }
    return t;
}

void tensor_free(Tensor* t) {
    if (!t)
        return;

    t->ref_count--;
    if (t->ref_count > 0)
        return;

    // CRITICAL: Clear IR node's output pointer BEFORE freeing the tensor
    // This prevents dangling pointers when cml_ir_free is called later
    if (t->ir_node) {
        struct IRNode* node = (struct IRNode*)t->ir_node;
        if (node->output == t) {
            node->output = NULL;
        }
        t->ir_node    = NULL;
        t->ir_context = NULL;
    }

    if (t->owns_data && t->data) {
        if (t->buffer_handle) {
            cml_backend_buffer_free(t->buffer_handle);
            t->buffer_handle = NULL;
        } else {
            if (t->device == DEVICE_CPU || t->device == DEVICE_AUTO) {
                free(t->data);
            } else {
                device_free(t->data, t->device);
            }
        }
    }

    if (t->shape)
        free(t->shape);
    if (t->strides)
        free(t->strides);

    if (t->grad) {
        tensor_free(t->grad);
        t->grad = NULL;
    }

    free(t);
}

Tensor* tensor_clone(Tensor* t) {
    if (!t)
        return NULL;

    // Ensure source tensor data is materialized before copying
    tensor_ensure_executed(t);
    if (!t->data)
        return NULL;

    TensorConfig config = (TensorConfig){t->dtype, t->device, true, true};
    Tensor* clone       = tensor_empty(t->shape, t->ndim, &config);
    if (!clone)
        return NULL;

    if (t->is_contiguous) {
        memcpy(clone->data, (char*)t->data + t->storage_offset * cml_dtype_size(t->dtype),
               t->numel * cml_dtype_size(t->dtype));
    } else {
        float* clone_data = (float*)clone->data;
        for (size_t i = 0; i < t->numel; i++) {
            clone_data[i] = tensor_get_float(t, i);
        }
    }

    clone->requires_grad = t->requires_grad;
    return clone;
}

Tensor* tensor_from_flat(const float* data, int rows, int cols) {
    if (!data || rows <= 0 || cols <= 0) {
        return NULL;
    }

    int shape[2] = {rows, cols};

    DType dtype         = cml_get_default_dtype();
    DeviceType device   = cml_get_default_device();
    TensorConfig config = (TensorConfig){dtype, device, true, true};

    Tensor* tensor = tensor_empty(shape, 2, &config);
    if (!tensor) {
        return NULL;
    }

    float* tensor_data = (float*)tensor_data_ptr(tensor);
    if (!tensor_data) {
        tensor_free(tensor);
        return NULL;
    }

    memcpy(tensor_data, data, (size_t)(rows * cols) * sizeof(float));
    return tensor;
}

Tensor* tensor_from_array_2d(const float* data, int rows, int cols) {
    return tensor_from_flat(data, rows, cols);
}

Tensor* tensor_zeros_2d(int rows, int cols) {
    int shape[2] = {rows, cols};

    DType dtype         = cml_get_default_dtype();
    DeviceType device   = cml_get_default_device();
    TensorConfig config = (TensorConfig){dtype, device, true, true};
    return tensor_zeros(shape, 2, &config);
}

Tensor* tensor_ones_2d(int rows, int cols) {
    int shape[2] = {rows, cols};

    DType dtype         = cml_get_default_dtype();
    DeviceType device   = cml_get_default_device();
    TensorConfig config = (TensorConfig){dtype, device, true, true};
    return tensor_ones(shape, 2, &config);
}

Tensor* tensor_empty_2d(int rows, int cols) {
    int shape[2] = {rows, cols};

    DType dtype         = cml_get_default_dtype();
    DeviceType device   = cml_get_default_device();
    TensorConfig config = (TensorConfig){dtype, device, true, true};
    return tensor_empty(shape, 2, &config);
}

int* tensor_shape(int ndim, ...) {
    if (ndim <= 0) {
        return NULL;
    }

    int* shape = (int*)malloc((size_t)ndim * sizeof(int));
    if (!shape) {
        return NULL;
    }

    va_list args;
    va_start(args, ndim);

    for (int i = 0; i < ndim; i++) {
        shape[i] = va_arg(args, int);
    }

    va_end(args);
    return shape;
}

int tensor_to_device(Tensor* tensor, DeviceType device) {
    if (!tensor) {
        return -1;
    }
    return device_move_tensor(tensor, device);
}

Tensor* tensor_arange(float start, float end, float step, const TensorConfig* config) {
    if (step == 0.0f) return NULL;
    int count = (int)ceilf((end - start) / step);
    if (count <= 0) count = 0;

    int shape[] = {count};
    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    Tensor* t = tensor_create(dtype, device, 1, shape, false);
    if (!t) return NULL;
    float* data = (float*)t->data;
    for (int i = 0; i < count; i++) {
        data[i] = start + (float)i * step;
    }
    return t;
}

Tensor* tensor_linspace(float start, float end, int steps, const TensorConfig* config) {
    if (steps <= 0) return NULL;
    int shape[] = {steps};
    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    Tensor* t = tensor_create(dtype, device, 1, shape, false);
    if (!t) return NULL;
    float* data = (float*)t->data;
    if (steps == 1) {
        data[0] = start;
    } else {
        float step = (end - start) / (float)(steps - 1);
        for (int i = 0; i < steps; i++) {
            data[i] = start + (float)i * step;
        }
    }
    return t;
}

Tensor* tensor_eye(int n, const TensorConfig* config) {
    if (n <= 0) return NULL;
    int shape[] = {n, n};
    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    Tensor* t = tensor_create(dtype, device, 2, shape, false);
    if (!t) return NULL;
    float* data = (float*)t->data;
    memset(data, 0, (size_t)(n * n) * sizeof(float));
    for (int i = 0; i < n; i++) {
        data[i * n + i] = 1.0f;
    }
    return t;
}

void tensor_manual_seed(uint64_t seed) {
    cml_rng_set_global_seed(seed);
}

static float _rand_uniform(void) {
    float val;
    cml_rng_uniform(cml_rng_get_global(), &val, 1);
    return val;
}

Tensor* tensor_rand(int* shape, int ndim, const TensorConfig* config) {
    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    Tensor* t = tensor_create(dtype, device, ndim, shape, false);
    if (!t) return NULL;
    cml_rng_uniform(cml_rng_get_global(), (float*)t->data, t->numel);
    return t;
}

Tensor* tensor_randn(int* shape, int ndim, const TensorConfig* config) {
    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    Tensor* t = tensor_create(dtype, device, ndim, shape, false);
    if (!t) return NULL;
    cml_rng_normal(cml_rng_get_global(), (float*)t->data, t->numel);
    return t;
}

Tensor* tensor_randint(int low, int high, int* shape, int ndim, const TensorConfig* config) {
    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    Tensor* t = tensor_create(dtype, device, ndim, shape, false);
    if (!t) return NULL;
    float* data = (float*)t->data;
    int range = high - low;
    if (range <= 0) range = 1;
    for (size_t i = 0; i < t->numel; i++) {
        data[i] = (float)(low + (int)(_rand_uniform() * (float)range));
    }
    return t;
}

Tensor* tensor_zeros_like(Tensor* a) {
    if (!a) return NULL;
    TensorConfig config = {.dtype = a->dtype, .device = a->device, .has_dtype = true, .has_device = true};
    return tensor_zeros(a->shape, a->ndim, &config);
}

Tensor* tensor_ones_like(Tensor* a) {
    if (!a) return NULL;
    TensorConfig config = {.dtype = a->dtype, .device = a->device, .has_dtype = true, .has_device = true};
    return tensor_ones(a->shape, a->ndim, &config);
}

Tensor* tensor_rand_like(Tensor* a) {
    if (!a) return NULL;
    TensorConfig config = {.dtype = a->dtype, .device = a->device, .has_dtype = true, .has_device = true};
    return tensor_rand(a->shape, a->ndim, &config);
}

Tensor* tensor_randn_like(Tensor* a) {
    if (!a) return NULL;
    TensorConfig config = {.dtype = a->dtype, .device = a->device, .has_dtype = true, .has_device = true};
    return tensor_randn(a->shape, a->ndim, &config);
}

Tensor* tensor_full_like(Tensor* a, float value) {
    if (!a) return NULL;
    TensorConfig config = {.dtype = a->dtype, .device = a->device, .has_dtype = true, .has_device = true};
    return tensor_full(a->shape, a->ndim, &config, value);
}

Tensor* tensor_squeeze(Tensor* a, int dim) {
    if (!a) return NULL;

    // Count dimensions that are not 1 (or specific dim)
    int new_ndim = 0;
    for (int i = 0; i < a->ndim; i++) {
        if (dim >= 0) {
            if (i == dim && a->shape[i] == 1) continue;
        } else {
            if (a->shape[i] == 1) continue;
        }
        new_ndim++;
    }
    if (new_ndim == 0) new_ndim = 1;

    int* new_shape = malloc((size_t)new_ndim * sizeof(int));
    if (!new_shape) return NULL;

    int j = 0;
    for (int i = 0; i < a->ndim; i++) {
        if (dim >= 0) {
            if (i == dim && a->shape[i] == 1) continue;
        } else {
            if (a->shape[i] == 1) continue;
        }
        if (j < new_ndim) new_shape[j++] = a->shape[i];
    }
    if (j == 0) new_shape[0] = 1;

    Tensor* result = tensor_reshape(a, new_shape, new_ndim);
    free(new_shape);
    return result;
}

Tensor* tensor_unsqueeze(Tensor* a, int dim) {
    if (!a) return NULL;
    if (dim < 0) dim = a->ndim + 1 + dim;
    if (dim < 0 || dim > a->ndim) return NULL;

    int new_ndim = a->ndim + 1;
    int* new_shape = malloc((size_t)new_ndim * sizeof(int));
    if (!new_shape) return NULL;

    int j = 0;
    for (int i = 0; i < new_ndim; i++) {
        if (i == dim) {
            new_shape[i] = 1;
        } else {
            new_shape[i] = a->shape[j++];
        }
    }

    Tensor* result = tensor_reshape(a, new_shape, new_ndim);
    free(new_shape);
    return result;
}

Tensor* tensor_flip(Tensor* a, int dim) {
    if (!a) return NULL;
    tensor_ensure_executed(a);
    float* data = (float*)tensor_data_ptr(a);
    if (!data) return NULL;

    if (dim < 0) dim = a->ndim + dim;
    if (dim < 0 || dim >= a->ndim) return NULL;

    TensorConfig config = {.dtype = a->dtype, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* result = tensor_empty(a->shape, a->ndim, &config);
    if (!result) return NULL;
    tensor_ensure_executed(result);
    float* out_data = (float*)tensor_data_ptr(result);

    if (a->ndim == 1) {
        int n = a->shape[0];
        for (int i = 0; i < n; i++)
            out_data[i] = data[n - 1 - i];
    } else if (a->ndim == 2) {
        int rows = a->shape[0], cols = a->shape[1];
        if (dim == 0) {
            for (int r = 0; r < rows; r++)
                memcpy(out_data + r * cols, data + (rows - 1 - r) * cols, (size_t)cols * sizeof(float));
        } else {
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    out_data[r * cols + c] = data[r * cols + (cols - 1 - c)];
        }
    } else {
        memcpy(out_data, data, a->numel * sizeof(float));
    }
    return result;
}

Tensor* tensor_repeat(Tensor* a, int* repeats, int num_repeats) {
    if (!a || !repeats || num_repeats != a->ndim) return NULL;
    tensor_ensure_executed(a);
    float* data = (float*)tensor_data_ptr(a);
    if (!data) return NULL;

    int* new_shape = malloc((size_t)a->ndim * sizeof(int));
    if (!new_shape) return NULL;
    for (int i = 0; i < a->ndim; i++)
        new_shape[i] = a->shape[i] * repeats[i];

    TensorConfig config = {.dtype = a->dtype, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* result = tensor_empty(new_shape, a->ndim, &config);
    if (!result) { free(new_shape); return NULL; }
    tensor_ensure_executed(result);
    float* out_data = (float*)tensor_data_ptr(result);

    if (a->ndim == 1) {
        int n = a->shape[0];
        for (int r = 0; r < repeats[0]; r++)
            memcpy(out_data + r * n, data, (size_t)n * sizeof(float));
    } else if (a->ndim == 2) {
        int rows = a->shape[0], cols = a->shape[1];
        int out_cols = new_shape[1];
        for (int rr = 0; rr < repeats[0]; rr++) {
            for (int r = 0; r < rows; r++) {
                for (int cr = 0; cr < repeats[1]; cr++) {
                    memcpy(out_data + (rr * rows + r) * out_cols + cr * cols,
                           data + r * cols, (size_t)cols * sizeof(float));
                }
            }
        }
    } else {
        memcpy(out_data, data, a->numel * sizeof(float));
    }

    free(new_shape);
    return result;
}

// tensor_split is defined in tensor_manipulation.c

Tensor** tensor_chunk(Tensor* a, int chunks, int dim, int* out_count) {
    Tensor** result = tensor_split(a, chunks, dim, NULL);
    if (result && out_count) *out_count = chunks;
    return result;
}

Tensor* tensor_kaiming_uniform(int* shape, int ndim, int fan_in, const TensorConfig* config) {
    Tensor* t = tensor_rand(shape, ndim, config);
    if (!t) return NULL;
    float* data = (float*)t->data;
    float bound = sqrtf(6.0f / (float)fan_in); // gain=sqrt(2) for ReLU, a=sqrt(5)
    for (size_t i = 0; i < t->numel; i++)
        data[i] = data[i] * 2.0f * bound - bound; // Scale [0,1) to [-bound, bound)
    return t;
}

Tensor* tensor_kaiming_normal(int* shape, int ndim, int fan_in, const TensorConfig* config) {
    Tensor* t = tensor_randn(shape, ndim, config);
    if (!t) return NULL;
    float* data = (float*)t->data;
    float std_val = sqrtf(2.0f / (float)fan_in);
    for (size_t i = 0; i < t->numel; i++)
        data[i] *= std_val;
    return t;
}

Tensor* tensor_glorot_uniform(int* shape, int ndim, int fan_in, int fan_out, const TensorConfig* config) {
    Tensor* t = tensor_rand(shape, ndim, config);
    if (!t) return NULL;
    float* data = (float*)t->data;
    float bound = sqrtf(6.0f / (float)(fan_in + fan_out));
    for (size_t i = 0; i < t->numel; i++)
        data[i] = data[i] * 2.0f * bound - bound;
    return t;
}

Tensor* tensor_xavier_normal(int* shape, int ndim, int fan_in, int fan_out, const TensorConfig* config) {
    Tensor* t = tensor_randn(shape, ndim, config);
    if (!t) return NULL;
    float* data = (float*)t->data;
    float std_val = sqrtf(2.0f / (float)(fan_in + fan_out));
    for (size_t i = 0; i < t->numel; i++)
        data[i] *= std_val;
    return t;
}

Tensor* tensor_cast(Tensor* a, DType dtype) {
    if (!a) return NULL;
    if (a->dtype == dtype) return tensor_clone(a);
    tensor_ensure_executed(a);
    if (!a->data) return NULL;

    TensorConfig config = {.dtype = dtype, .device = a->device, .has_dtype = true, .has_device = true};
    Tensor* out = tensor_empty(a->shape, a->ndim, &config);
    if (!out) return NULL;
    tensor_ensure_executed(out);
    if (!out->data) { tensor_free(out); return NULL; }

    for (size_t i = 0; i < a->numel; i++) {
        float val = tensor_get_float(a, i);
        tensor_set_float(out, i, val);
    }
    return out;
}

Tensor* tensor_from_blob(void* data, int* shape, int ndim, const TensorConfig* config) {
    if (!data || !shape || ndim <= 0) return NULL;

    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    Tensor* t = (Tensor*)calloc(1, sizeof(Tensor));
    if (!t) return NULL;

    t->ndim = ndim;
    t->shape = (int*)malloc((size_t)ndim * sizeof(int));
    if (!t->shape) { free(t); return NULL; }
    memcpy(t->shape, shape, (size_t)ndim * sizeof(int));

    t->numel = tensor_numel(shape, ndim);
    t->dtype = dtype;
    t->device = device;
    t->data = data;
    t->owns_data = false;  // caller retains ownership
    t->is_executed = true;
    t->is_contiguous = true;
    t->strides = compute_contiguous_strides(shape, ndim);
    t->storage_offset = 0;
    t->ref_count = 1;

    return t;
}

Tensor* tensor_randperm(int n, const TensorConfig* config) {
    if (n <= 0) return NULL;

    int shape[] = {n};
    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    TensorConfig int_config = {.dtype = DTYPE_FLOAT32, .device = device, .has_dtype = true, .has_device = true};
    Tensor* t = tensor_empty(shape, 1, &int_config);
    if (!t) return NULL;
    tensor_ensure_executed(t);
    if (!t->data) { tensor_free(t); return NULL; }

    float* data = (float*)t->data;
    // Fisher-Yates shuffle
    for (int i = 0; i < n; i++) data[i] = (float)i;
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        float tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
    return t;
}

Tensor* tensor_half(Tensor* a) { return tensor_cast(a, DTYPE_FLOAT16); }
Tensor* tensor_float(Tensor* a) { return tensor_cast(a, DTYPE_FLOAT32); }
Tensor* tensor_double(Tensor* a) { return tensor_cast(a, DTYPE_FLOAT64); }
Tensor* tensor_int(Tensor* a) { return tensor_cast(a, DTYPE_INT32); }
Tensor* tensor_long(Tensor* a) { return tensor_cast(a, DTYPE_INT64); }
Tensor* tensor_short(Tensor* a) { return tensor_cast(a, DTYPE_INT16); }
Tensor* tensor_bool(Tensor* a) { return tensor_cast(a, DTYPE_BOOL); }
Tensor* tensor_bfloat16(Tensor* a) { return tensor_cast(a, DTYPE_BFLOAT16); }
Tensor* tensor_fp8e4m3fnuz(Tensor* a) { return tensor_cast(a, DTYPE_FLOAT8_E4M3_FNUZ); }
Tensor* tensor_fp8e5m2fnuz(Tensor* a) { return tensor_cast(a, DTYPE_FLOAT8_E5M2_FNUZ); }

Tensor* tensor_interpolate(Tensor* a, int* output_size, int num_dims, InterpMode mode) {
    if (!a || !output_size || num_dims < 1) return NULL;
    tensor_ensure_executed(a);
    if (!a->data) return NULL;

    // Support 4D [N, C, H, W]
    if (a->ndim != 4 || num_dims != 2) {
        LOG_ERROR("tensor_interpolate: only 4D [N,C,H,W] with 2D output_size supported");
        return NULL;
    }

    int N = a->shape[0], C = a->shape[1];
    int in_h = a->shape[2], in_w = a->shape[3];
    int out_h = output_size[0], out_w = output_size[1];

    int out_shape[] = {N, C, out_h, out_w};
    TensorConfig config = {.dtype = a->dtype, .device = a->device, .has_dtype = true, .has_device = true};
    Tensor* out = tensor_empty(out_shape, 4, &config);
    if (!out) return NULL;
    tensor_ensure_executed(out);
    float* in_data = (float*)a->data;
    float* out_data = (float*)out->data;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    size_t out_idx = (size_t)n * C * out_h * out_w + (size_t)c * out_h * out_w +
                                     (size_t)oh * out_w + ow;

                    if (mode == INTERP_NEAREST) {
                        int ih = (int)((float)oh * in_h / out_h);
                        int iw = (int)((float)ow * in_w / out_w);
                        if (ih >= in_h) ih = in_h - 1;
                        if (iw >= in_w) iw = in_w - 1;
                        size_t in_idx = (size_t)n * C * in_h * in_w + (size_t)c * in_h * in_w +
                                        (size_t)ih * in_w + iw;
                        out_data[out_idx] = in_data[in_idx];
                    } else { // INTERP_BILINEAR
                        float fy = (float)oh * (float)(in_h - 1) / (float)(out_h - 1 > 0 ? out_h - 1 : 1);
                        float fx = (float)ow * (float)(in_w - 1) / (float)(out_w - 1 > 0 ? out_w - 1 : 1);
                        int y0 = (int)floorf(fy), x0 = (int)floorf(fx);
                        int y1 = y0 + 1, x1 = x0 + 1;
                        if (y0 < 0) y0 = 0;
                        if (y1 >= in_h) y1 = in_h - 1;
                        if (x0 < 0) x0 = 0;
                        if (x1 >= in_w) x1 = in_w - 1;
                        float wy = fy - (float)y0, wx = fx - (float)x0;
                        size_t base = (size_t)n * C * in_h * in_w + (size_t)c * in_h * in_w;
                        float v00 = in_data[base + (size_t)y0 * in_w + x0];
                        float v01 = in_data[base + (size_t)y0 * in_w + x1];
                        float v10 = in_data[base + (size_t)y1 * in_w + x0];
                        float v11 = in_data[base + (size_t)y1 * in_w + x1];
                        out_data[out_idx] = (1 - wy) * (1 - wx) * v00 + (1 - wy) * wx * v01 +
                                            wy * (1 - wx) * v10 + wy * wx * v11;
                    }
                }
            }
        }
    }
    return out;
}

Tensor* tensor_dot(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;
    if (a->ndim != 1 || b->ndim != 1 || a->numel != b->numel) {
        LOG_ERROR("tensor_dot: both tensors must be 1D with same size");
        return NULL;
    }
    tensor_ensure_executed(a);
    tensor_ensure_executed(b);
    if (!a->data || !b->data) return NULL;

    float sum = 0.0f;
    for (size_t i = 0; i < a->numel; i++) {
        sum += tensor_get_float(a, i) * tensor_get_float(b, i);
    }

    int shape[] = {1};
    TensorConfig config = {.dtype = a->dtype, .device = a->device, .has_dtype = true, .has_device = true};
    Tensor* out = tensor_full(shape, 1, &config, sum);
    return out;
}

Tensor* tensor_scatter_reduce(Tensor* self, int dim, Tensor* index, Tensor* src, ScatterReduceMode mode) {
    if (!self || !index || !src) return NULL;
    if (dim < 0) dim += self->ndim;
    if (dim < 0 || dim >= self->ndim) {
        LOG_ERROR("tensor_scatter_reduce: invalid dim %d for %dD tensor", dim, self->ndim);
        return NULL;
    }

    tensor_ensure_executed(self);
    tensor_ensure_executed(index);
    tensor_ensure_executed(src);
    if (!self->data || !index->data || !src->data) return NULL;

    Tensor* output = tensor_clone(self);
    if (!output) return NULL;
    tensor_ensure_executed(output);

    // For simplicity, handle 1D and 2D cases
    if (self->ndim == 1) {
        for (size_t i = 0; i < index->numel; i++) {
            int idx = (int)tensor_get_float(index, i);
            if (idx < 0 || idx >= self->shape[0]) continue;
            float src_val = tensor_get_float(src, i);
            float cur_val = tensor_get_float(output, idx);
            float new_val;
            switch (mode) {
                case SCATTER_REDUCE_SUM:  new_val = cur_val + src_val; break;
                case SCATTER_REDUCE_PROD: new_val = cur_val * src_val; break;
                case SCATTER_REDUCE_AMAX: new_val = src_val > cur_val ? src_val : cur_val; break;
                case SCATTER_REDUCE_AMIN: new_val = src_val < cur_val ? src_val : cur_val; break;
                case SCATTER_REDUCE_MEAN: new_val = cur_val + src_val; break; // accumulate, divide later
                default: new_val = cur_val; break;
            }
            tensor_set_float(output, idx, new_val);
        }
        if (mode == SCATTER_REDUCE_MEAN) {
            // Count contributions per index
            int* counts = calloc(self->shape[0], sizeof(int));
            if (counts) {
                for (int i = 0; i < self->shape[0]; i++) counts[i] = 1; // self contributes 1
                for (size_t i = 0; i < index->numel; i++) {
                    int idx = (int)tensor_get_float(index, i);
                    if (idx >= 0 && idx < self->shape[0]) counts[idx]++;
                }
                for (int i = 0; i < self->shape[0]; i++) {
                    if (counts[i] > 1) {
                        tensor_set_float(output, i, tensor_get_float(output, i) / counts[i]);
                    }
                }
                free(counts);
            }
        }
    } else if (self->ndim == 2) {
        int rows = self->shape[0], cols = self->shape[1];
        for (size_t i = 0; i < index->numel; i++) {
            int r = (int)(i / index->shape[1]);
            int c = (int)(i % index->shape[1]);
            int idx = (int)tensor_get_float(index, i);
            size_t src_off = r * src->shape[1] + c;
            float src_val = tensor_get_float(src, src_off);

            size_t out_off;
            if (dim == 0) {
                if (idx < 0 || idx >= rows) continue;
                out_off = idx * cols + c;
            } else {
                if (idx < 0 || idx >= cols) continue;
                out_off = r * cols + idx;
            }
            float cur_val = tensor_get_float(output, out_off);
            float new_val;
            switch (mode) {
                case SCATTER_REDUCE_SUM:  new_val = cur_val + src_val; break;
                case SCATTER_REDUCE_PROD: new_val = cur_val * src_val; break;
                case SCATTER_REDUCE_AMAX: new_val = src_val > cur_val ? src_val : cur_val; break;
                case SCATTER_REDUCE_AMIN: new_val = src_val < cur_val ? src_val : cur_val; break;
                case SCATTER_REDUCE_MEAN: new_val = cur_val + src_val; break;
                default: new_val = cur_val; break;
            }
            tensor_set_float(output, out_off, new_val);
        }
    }
    return output;
}

Tensor* tensor_bitcast(Tensor* a, DType target_dtype) {
    if (!a) return NULL;
    tensor_ensure_executed(a);
    if (!a->data) return NULL;

    size_t src_size = cml_dtype_size(a->dtype);
    size_t dst_size = cml_dtype_size(target_dtype);

    if (src_size == 0 || dst_size == 0) {
        LOG_ERROR("tensor_bitcast: unsupported dtype size");
        return NULL;
    }

    size_t total_bytes = a->numel * src_size;
    if (total_bytes % dst_size != 0) {
        LOG_ERROR("tensor_bitcast: total byte size %zu not divisible by target element size %zu",
                  total_bytes, dst_size);
        return NULL;
    }

    size_t new_numel = total_bytes / dst_size;

    int new_ndim = a->ndim;
    int* new_shape = malloc(new_ndim * sizeof(int));
    if (!new_shape) return NULL;

    for (int i = 0; i < new_ndim - 1; i++) new_shape[i] = a->shape[i];
    size_t leading = 1;
    for (int i = 0; i < new_ndim - 1; i++) leading *= a->shape[i];
    if (leading == 0) { free(new_shape); return NULL; }
    new_shape[new_ndim - 1] = (int)(new_numel / leading);

    TensorConfig config = {.dtype = target_dtype, .device = a->device, .has_dtype = true, .has_device = true};
    Tensor* out = tensor_empty(new_shape, new_ndim, &config);
    free(new_shape);
    if (!out) return NULL;
    tensor_ensure_executed(out);

    // Raw memcpy — reinterpret bits
    memcpy(out->data, a->data, total_bytes);
    return out;
}

QRResult tensor_qr(Tensor* a) {
    QRResult result = {NULL, NULL};
    if (!a || a->ndim != 2) {
        LOG_ERROR("tensor_qr: input must be a 2D matrix");
        return result;
    }

    tensor_ensure_executed(a);
    if (!a->data) return result;

    int m = a->shape[0], n = a->shape[1];
    int k = m < n ? m : n;

    // Work on a copy of A (will become R)
    float* R = malloc((size_t)m * n * sizeof(float));
    if (!R) return result;
    for (int i = 0; i < m * n; i++)
        R[i] = tensor_get_float(a, i);

    // Q starts as identity [m, m]
    float* Q = calloc((size_t)m * m, sizeof(float));
    if (!Q) { free(R); return result; }
    for (int i = 0; i < m; i++) Q[i * m + i] = 1.0f;

    float* v = malloc((size_t)m * sizeof(float));
    if (!v) { free(R); free(Q); return result; }

    for (int j = 0; j < k; j++) {
        // Extract column j from row j..m-1
        float norm = 0.0f;
        for (int i = j; i < m; i++) {
            v[i] = R[i * n + j];
            norm += v[i] * v[i];
        }
        norm = sqrtf(norm);
        if (norm < 1e-12f) continue;

        float sign = (R[j * n + j] >= 0.0f) ? 1.0f : -1.0f;
        v[j] += sign * norm;

        float vnorm = 0.0f;
        for (int i = j; i < m; i++) vnorm += v[i] * v[i];
        if (vnorm < 1e-24f) continue;
        float inv_vnorm = 1.0f / vnorm;

        // Apply Householder to R: R = R - 2*v*(v^T * R) / (v^T * v)
        for (int c = j; c < n; c++) {
            float dot = 0.0f;
            for (int i = j; i < m; i++) dot += v[i] * R[i * n + c];
            dot *= 2.0f * inv_vnorm;
            for (int i = j; i < m; i++) R[i * n + c] -= dot * v[i];
        }

        // Apply Householder to Q: Q = Q - 2*Q*v*v^T / (v^T * v)
        for (int r = 0; r < m; r++) {
            float dot = 0.0f;
            for (int i = j; i < m; i++) dot += Q[r * m + i] * v[i];
            dot *= 2.0f * inv_vnorm;
            for (int i = j; i < m; i++) Q[r * m + i] -= dot * v[i];
        }
    }
    free(v);

    // Create reduced Q [m, k] and R [k, n]
    TensorConfig config = {.dtype = a->dtype, .device = a->device, .has_dtype = true, .has_device = true};

    int q_shape[] = {m, k};
    result.Q = tensor_empty(q_shape, 2, &config);
    if (!result.Q) { free(R); free(Q); return result; }
    tensor_ensure_executed(result.Q);
    float* q_out = (float*)result.Q->data;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            q_out[i * k + j] = Q[i * m + j];

    int r_shape[] = {k, n};
    result.R = tensor_empty(r_shape, 2, &config);
    if (!result.R) { free(R); free(Q); tensor_free(result.Q); result.Q = NULL; return result; }
    tensor_ensure_executed(result.R);
    float* r_out = (float*)result.R->data;
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            r_out[i * n + j] = R[i * n + j];

    free(R);
    free(Q);
    return result;
}

SVDResult tensor_svd(Tensor* a) {
    SVDResult result = {NULL, NULL, NULL};
    if (!a || a->ndim != 2) {
        LOG_ERROR("tensor_svd: input must be a 2D matrix");
        return result;
    }

    tensor_ensure_executed(a);
    if (!a->data) return result;

    int m = a->shape[0], n = a->shape[1];
    int k = m < n ? m : n;

    // Work on A^T * A for right singular vectors, or use Jacobi on A directly
    // Use one-sided Jacobi: iterate on columns of A copy
    float* W = malloc((size_t)m * n * sizeof(float));
    if (!W) return result;
    for (int i = 0; i < m * n; i++)
        W[i] = tensor_get_float(a, i);

    // V starts as identity [n, n]
    float* V = calloc((size_t)n * n, sizeof(float));
    if (!V) { free(W); return result; }
    for (int i = 0; i < n; i++) V[i * n + i] = 1.0f;

    // One-sided Jacobi rotations
    int max_iter = 100;
    for (int iter = 0; iter < max_iter; iter++) {
        float off = 0.0f;
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                // Compute col_p^T * col_q and norms
                float alpha = 0.0f, beta = 0.0f, gamma = 0.0f;
                for (int i = 0; i < m; i++) {
                    alpha += W[i * n + p] * W[i * n + p];
                    beta  += W[i * n + q] * W[i * n + q];
                    gamma += W[i * n + p] * W[i * n + q];
                }
                off += gamma * gamma;
                if (fabsf(gamma) < 1e-12f) continue;

                // Compute Jacobi rotation
                float tau = (beta - alpha) / (2.0f * gamma);
                float t;
                if (tau >= 0.0f)
                    t = 1.0f / (tau + sqrtf(1.0f + tau * tau));
                else
                    t = -1.0f / (-tau + sqrtf(1.0f + tau * tau));
                float c = 1.0f / sqrtf(1.0f + t * t);
                float s = t * c;

                for (int i = 0; i < m; i++) {
                    float wp = W[i * n + p], wq = W[i * n + q];
                    W[i * n + p] = c * wp - s * wq;
                    W[i * n + q] = s * wp + c * wq;
                }
                for (int i = 0; i < n; i++) {
                    float vp = V[i * n + p], vq = V[i * n + q];
                    V[i * n + p] = c * vp - s * vq;
                    V[i * n + q] = s * vp + c * vq;
                }
            }
        }
        if (off < 1e-20f) break;
    }

    // Compute singular values (column norms of W) and U = W / sigma
    float* sigma = malloc((size_t)k * sizeof(float));
    float* U = malloc((size_t)m * k * sizeof(float));
    if (!sigma || !U) { free(W); free(V); free(sigma); free(U); return result; }

    int* order = malloc((size_t)n * sizeof(int));
    if (!order) { free(W); free(V); free(sigma); free(U); return result; }
    for (int i = 0; i < n; i++) order[i] = i;

    float* col_norms = malloc((size_t)n * sizeof(float));
    if (!col_norms) { free(W); free(V); free(sigma); free(U); free(order); return result; }
    for (int j = 0; j < n; j++) {
        float norm = 0.0f;
        for (int i = 0; i < m; i++) norm += W[i * n + j] * W[i * n + j];
        col_norms[j] = sqrtf(norm);
    }

    // Simple selection sort by descending col_norms
    for (int i = 0; i < k; i++) {
        int max_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (col_norms[order[j]] > col_norms[order[max_idx]]) max_idx = j;
        }
        if (max_idx != i) { int tmp = order[i]; order[i] = order[max_idx]; order[max_idx] = tmp; }
    }

    for (int j = 0; j < k; j++) {
        int oj = order[j];
        sigma[j] = col_norms[oj];
        if (sigma[j] > 1e-12f) {
            for (int i = 0; i < m; i++)
                U[i * k + j] = W[i * n + oj] / sigma[j];
        } else {
            for (int i = 0; i < m; i++)
                U[i * k + j] = 0.0f;
        }
    }

    TensorConfig config = {.dtype = a->dtype, .device = a->device, .has_dtype = true, .has_device = true};

    int u_shape[] = {m, k};
    result.U = tensor_empty(u_shape, 2, &config);
    if (result.U) { tensor_ensure_executed(result.U); memcpy(result.U->data, U, (size_t)m * k * sizeof(float)); }

    int s_shape[] = {k};
    result.S = tensor_empty(s_shape, 1, &config);
    if (result.S) { tensor_ensure_executed(result.S); memcpy(result.S->data, sigma, (size_t)k * sizeof(float)); }

    // Vt = V^T but only rows corresponding to sorted order
    int vt_shape[] = {k, n};
    result.Vt = tensor_empty(vt_shape, 2, &config);
    if (result.Vt) {
        tensor_ensure_executed(result.Vt);
        float* vt_data = (float*)result.Vt->data;
        for (int j = 0; j < k; j++) {
            int oj = order[j];
            for (int i = 0; i < n; i++)
                vt_data[j * n + i] = V[i * n + oj];
        }
    }

    free(W); free(V); free(sigma); free(U); free(order); free(col_norms);
    return result;
}

Tensor* tensor_from_url(const char* url) {
    if (!url) return NULL;

    char tmppath[] = "/tmp/cml_tensor_XXXXXX";
    int fd = mkstemp(tmppath);
    if (fd < 0) {
        LOG_ERROR("tensor_from_url: failed to create temp file");
        return NULL;
    }
    close(fd);

    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "curl -fsSL -o '%s' '%s' 2>/dev/null || wget -q -O '%s' '%s' 2>/dev/null",
             tmppath, url, tmppath, url);

    int ret = system(cmd);
    if (ret != 0) {
        LOG_ERROR("tensor_from_url: download failed for %s", url);
        remove(tmppath);
        return NULL;
    }

    Tensor* t = tensor_read_file(tmppath);
    remove(tmppath);
    return t;
}

int tensor_assign(Tensor* t, Tensor* src) {
    if (!t || !src) return -1;
    if (t->ndim != src->ndim) return -1;
    for (int i = 0; i < t->ndim; i++) {
        if (t->shape[i] != src->shape[i]) return -1;
    }

    tensor_ensure_executed(src);
    if (!src->data) return -1;

    size_t nbytes = t->numel * cml_dtype_size(t->dtype);

    if (t->owns_data && t->data) {
        if (t->buffer_handle) {
            cml_backend_buffer_free(t->buffer_handle);
            t->buffer_handle = NULL;
        } else if (t->device == DEVICE_CPU || t->device == DEVICE_AUTO) {
            free(t->data);
        } else {
            device_free(t->data, t->device);
        }
    }

    if (t->device == src->device) {
        t->data = src->data;
        t->owns_data = false;
        t->buffer_handle = NULL;
        t->from_buffer_cache = false;
    } else {
        t->data = malloc(nbytes);
        if (!t->data) return -1;
        memcpy(t->data, src->data, nbytes);
        t->owns_data = true;
    }

    if (src->ir_node)
        t->ir_node = src->ir_node;

    t->is_executed = true;
    return 0;
}

int tensor_assign_data(Tensor* t, const void* data, size_t nbytes) {
    if (!t || !data || nbytes == 0) return -1;

    size_t expected = t->numel * cml_dtype_size(t->dtype);
    if (nbytes > expected) return -1;

    if (!t->data) {
        t->data = malloc(expected);
        if (!t->data) return -1;
        t->owns_data = true;
    }

    memcpy(t->data, data, nbytes);
    t->is_executed = true;
    return 0;
}
