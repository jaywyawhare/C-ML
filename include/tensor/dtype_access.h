#ifndef CML_TENSOR_DTYPE_ACCESS_H
#define CML_TENSOR_DTYPE_ACCESS_H

/* Element-level dtype access.
 *
 * The float32 executor paths read buffers as `float*` directly. Every other
 * dtype goes through the load/store accessors at the bottom of this file, which
 * let a kernel be written once and run for any dtype: the i64 path is exact for
 * every integer width, and the f64 path is a superset of f32/f16/bf16/fp8.
 * That trades per-element dispatch for coverage, so it is deliberately kept off
 * the f32 hot path.
 *
 * These were file-static in tensor.c; they live here so the executor's typed
 * kernels and tensor.c share one copy. */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "tensor/tensor.h"

static inline uint16_t float_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp   = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;
    if (exp <= 0)
        return (uint16_t)sign;
    if (exp >= 31)
        return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static inline float fp16_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
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
    uint8_t sign  = (x >> 24) & 0x80;
    int32_t exp   = ((x >> 23) & 0xFF) - 127 + 7;
    uint32_t mant = (x >> 20) & 0x07;
    if (exp <= 0)
        return sign;
    if (exp >= 15)
        return sign | 0x7E; // max finite: S.1111.110
    return sign | ((uint8_t)exp << 3) | (uint8_t)mant;
}

static inline float fp8_e4m3_to_float(uint8_t h) {
    uint32_t sign = ((uint32_t)(h & 0x80)) << 24;
    uint32_t exp  = (h >> 3) & 0x0F;
    uint32_t mant = h & 0x07;
    if (exp == 0) {
        float f;
        uint32_t r = sign;
        memcpy(&f, &r, sizeof(f));
        return f;
    }
    uint32_t result = sign | ((exp - 7 + 127) << 23) | (mant << 20);
    float f;
    memcpy(&f, &result, sizeof(f));
    return f;
}

// FP8 E5M2: 1 sign, 5 exponent, 2 mantissa, bias=15 (like IEEE fp8)
static inline uint8_t float_to_fp8_e5m2(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint8_t sign  = (x >> 24) & 0x80;
    int32_t exp   = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 21) & 0x03;
    if (exp <= 0)
        return sign;
    if (exp >= 31)
        return sign | 0x7C; // inf: S.11111.00
    return sign | ((uint8_t)exp << 2) | (uint8_t)mant;
}

static inline float fp8_e5m2_to_float(uint8_t h) {
    uint32_t sign = ((uint32_t)(h & 0x80)) << 24;
    uint32_t exp  = (h >> 2) & 0x1F;
    uint32_t mant = h & 0x03;
    if (exp == 0) {
        float f;
        uint32_t r = sign;
        memcpy(&f, &r, sizeof(f));
        return f;
    }
    if (exp == 31) {
        float f;
        uint32_t r = sign | 0x7F800000 | (mant << 21);
        memcpy(&f, &r, sizeof(f));
        return f;
    }
    uint32_t result = sign | ((exp - 15 + 127) << 23) | (mant << 21);
    float f;
    memcpy(&f, &result, sizeof(f));
    return f;
}

static inline uint8_t float_to_fp8e4m3fnuz(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign_bit  = (x >> 31) & 1;
    int32_t fp32_exp   = ((x >> 23) & 0xFF);
    uint32_t fp32_mant = x & 0x7FFFFF;

    if (fp32_exp == 0xFF || (fp32_exp == 0 && fp32_mant == 0))
        return 0x00;
    if (f == 0.0f || f == -0.0f)
        return 0x00;

    int32_t exp   = fp32_exp - 127 + 8;
    uint32_t mant = (fp32_mant >> 20) & 0x07;

    if (exp <= 0)
        return 0x00;
    if (exp >= 16)
        return (uint8_t)((sign_bit << 7) | 0x7F);

    return (uint8_t)((sign_bit << 7) | ((uint8_t)exp << 3) | (uint8_t)mant);
}

static inline float fp8e4m3fnuz_to_float(uint8_t h) {
    if (h == 0x80)
        return NAN;
    if (h == 0x00)
        return 0.0f;
    uint32_t sign = ((uint32_t)(h >> 7)) << 31;
    uint32_t exp  = (h >> 3) & 0x0F;
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
    uint32_t sign_bit  = (x >> 31) & 1;
    int32_t fp32_exp   = ((x >> 23) & 0xFF);
    uint32_t fp32_mant = x & 0x7FFFFF;

    if (fp32_exp == 0xFF || (fp32_exp == 0 && fp32_mant == 0))
        return 0x00;
    if (f == 0.0f || f == -0.0f)
        return 0x00;

    int32_t exp   = fp32_exp - 127 + 16;
    uint32_t mant = (fp32_mant >> 21) & 0x03;

    if (exp <= 0)
        return 0x00;
    if (exp >= 32)
        return (uint8_t)((sign_bit << 7) | 0x7F);

    return (uint8_t)((sign_bit << 7) | ((uint8_t)exp << 2) | (uint8_t)mant);
}

static inline float fp8e5m2fnuz_to_float(uint8_t h) {
    if (h == 0x80)
        return NAN;
    if (h == 0x00)
        return 0.0f;
    uint32_t sign = ((uint32_t)(h >> 7)) << 31;
    uint32_t exp  = (h >> 2) & 0x1F;
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

static inline bool cml_dtype_is_int(DType d) {
    return d == DTYPE_INT8 || d == DTYPE_INT16 || d == DTYPE_INT32 || d == DTYPE_INT64 ||
           d == DTYPE_UINT8 || d == DTYPE_UINT16 || d == DTYPE_UINT32 || d == DTYPE_UINT64 ||
           d == DTYPE_BOOL;
}

static inline bool cml_dtype_is_float(DType d) {
    return d == DTYPE_FLOAT32 || d == DTYPE_FLOAT64 || d == DTYPE_FLOAT16 || d == DTYPE_BFLOAT16 ||
           d == DTYPE_FLOAT8_E4M3 || d == DTYPE_FLOAT8_E5M2 || d == DTYPE_FLOAT8_E4M3_FNUZ ||
           d == DTYPE_FLOAT8_E5M2_FNUZ;
}

static inline bool cml_dtype_direct(DType d) {
    return d == DTYPE_FLOAT32 || d == DTYPE_FLOAT64 || d == DTYPE_FLOAT16 || d == DTYPE_BFLOAT16 ||
           cml_dtype_is_int(d);
}

static inline int64_t cml_load_i64(const void* p, size_t i, DType d) {
    switch (d) {
    case DTYPE_INT8:
        return ((const int8_t*)p)[i];
    case DTYPE_INT16:
        return ((const int16_t*)p)[i];
    case DTYPE_INT32:
        return ((const int32_t*)p)[i];
    case DTYPE_INT64:
        return ((const int64_t*)p)[i];
    case DTYPE_UINT8:
    case DTYPE_BOOL:
        return ((const uint8_t*)p)[i];
    case DTYPE_UINT16:
        return ((const uint16_t*)p)[i];
    case DTYPE_UINT32:
        return ((const uint32_t*)p)[i];
    case DTYPE_UINT64:
        return (int64_t)((const uint64_t*)p)[i];
    default:
        return 0;
    }
}

static inline void cml_store_i64(void* p, size_t i, DType d, int64_t v) {
    switch (d) {
    case DTYPE_INT8:
        ((int8_t*)p)[i] = (int8_t)v;
        break;
    case DTYPE_INT16:
        ((int16_t*)p)[i] = (int16_t)v;
        break;
    case DTYPE_INT32:
        ((int32_t*)p)[i] = (int32_t)v;
        break;
    case DTYPE_INT64:
        ((int64_t*)p)[i] = v;
        break;
    case DTYPE_UINT8:
        ((uint8_t*)p)[i] = (uint8_t)v;
        break;
    case DTYPE_BOOL:
        ((uint8_t*)p)[i] = v ? 1 : 0;
        break;
    case DTYPE_UINT16:
        ((uint16_t*)p)[i] = (uint16_t)v;
        break;
    case DTYPE_UINT32:
        ((uint32_t*)p)[i] = (uint32_t)v;
        break;
    case DTYPE_UINT64:
        ((uint64_t*)p)[i] = (uint64_t)v;
        break;
    default:
        break;
    }
}

static inline double cml_load_f64(const void* p, size_t i, DType d) {
    switch (d) {
    case DTYPE_FLOAT32:
        return (double)((const float*)p)[i];
    case DTYPE_FLOAT64:
        return ((const double*)p)[i];
    case DTYPE_FLOAT16:
        return (double)fp16_to_float(((const uint16_t*)p)[i]);
    case DTYPE_BFLOAT16:
        return (double)bf16_to_float(((const uint16_t*)p)[i]);
    case DTYPE_FLOAT8_E4M3:
        return (double)fp8_e4m3_to_float(((const uint8_t*)p)[i]);
    case DTYPE_FLOAT8_E5M2:
        return (double)fp8_e5m2_to_float(((const uint8_t*)p)[i]);
    case DTYPE_FLOAT8_E4M3_FNUZ:
        return (double)fp8e4m3fnuz_to_float(((const uint8_t*)p)[i]);
    case DTYPE_FLOAT8_E5M2_FNUZ:
        return (double)fp8e5m2fnuz_to_float(((const uint8_t*)p)[i]);
    default:
        return (double)cml_load_i64(p, i, d);
    }
}

static inline void cml_store_f64(void* p, size_t i, DType d, double v) {
    switch (d) {
    case DTYPE_FLOAT32:
        ((float*)p)[i] = (float)v;
        break;
    case DTYPE_FLOAT64:
        ((double*)p)[i] = v;
        break;
    case DTYPE_FLOAT16:
        ((uint16_t*)p)[i] = float_to_fp16((float)v);
        break;
    case DTYPE_BFLOAT16:
        ((uint16_t*)p)[i] = float_to_bf16((float)v);
        break;
    case DTYPE_FLOAT8_E4M3:
        ((uint8_t*)p)[i] = float_to_fp8_e4m3((float)v);
        break;
    case DTYPE_FLOAT8_E5M2:
        ((uint8_t*)p)[i] = float_to_fp8_e5m2((float)v);
        break;
    case DTYPE_FLOAT8_E4M3_FNUZ:
        ((uint8_t*)p)[i] = float_to_fp8e4m3fnuz((float)v);
        break;
    case DTYPE_FLOAT8_E5M2_FNUZ:
        ((uint8_t*)p)[i] = float_to_fp8e5m2fnuz((float)v);
        break;
    default:
        cml_store_i64(p, i, d, (int64_t)v);
        break;
    }
}

#endif /* CML_TENSOR_DTYPE_ACCESS_H */
