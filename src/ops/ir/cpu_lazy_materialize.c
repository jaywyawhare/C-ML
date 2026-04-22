#include "ops/ir/cpu_lazy_materialize.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* FP16/BF16/FP8 encode helpers (aligned with tensor.c). */
static uint16_t lz_float_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static uint16_t lz_float_to_bf16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    return (uint16_t)(x >> 16);
}

static uint8_t lz_float_to_fp8_e4m3(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint8_t sign = (x >> 24) & 0x80;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 7;
    uint32_t mant = (x >> 20) & 0x07;
    if (exp <= 0) return sign;
    if (exp >= 15) return (uint8_t)(sign | 0x7E);
    return (uint8_t)(sign | ((uint8_t)exp << 3) | (uint8_t)mant);
}

static uint8_t lz_float_to_fp8_e5m2(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint8_t sign = (x >> 24) & 0x80;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 21) & 0x03;
    if (exp <= 0) return sign;
    if (exp >= 31) return (uint8_t)(sign | 0x7C);
    return (uint8_t)(sign | ((uint8_t)exp << 2) | (uint8_t)mant);
}

static uint8_t lz_float_to_fp8e4m3fnuz(float f) {
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

static uint8_t lz_float_to_fp8e5m2fnuz(float f) {
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

void cml_cpu_lazy_store_float_elem(void* base, size_t i, DType dt, float v) {
    switch (dt) {
    case DTYPE_FLOAT32:
        ((float*)base)[i] = v;
        break;
    case DTYPE_FLOAT64:
        ((double*)base)[i] = (double)v;
        break;
    case DTYPE_INT32:
        ((int32_t*)base)[i] = (int32_t)v;
        break;
    case DTYPE_INT64:
        ((int64_t*)base)[i] = (int64_t)v;
        break;
    case DTYPE_BOOL:
        ((uint8_t*)base)[i] = (uint8_t)(fabsf(v) > 1e-9f);
        break;
    case DTYPE_FLOAT16:
        ((uint16_t*)base)[i] = lz_float_to_fp16(v);
        break;
    case DTYPE_BFLOAT16:
        ((uint16_t*)base)[i] = lz_float_to_bf16(v);
        break;
    case DTYPE_INT8:
        ((int8_t*)base)[i] = (int8_t)v;
        break;
    case DTYPE_UINT8:
        ((uint8_t*)base)[i] = (uint8_t)v;
        break;
    case DTYPE_INT16:
        ((int16_t*)base)[i] = (int16_t)v;
        break;
    case DTYPE_UINT16:
        ((uint16_t*)base)[i] = (uint16_t)v;
        break;
    case DTYPE_UINT32:
        ((uint32_t*)base)[i] = (uint32_t)v;
        break;
    case DTYPE_UINT64:
        ((uint64_t*)base)[i] = (uint64_t)v;
        break;
    case DTYPE_FLOAT8_E4M3:
        ((uint8_t*)base)[i] = lz_float_to_fp8_e4m3(v);
        break;
    case DTYPE_FLOAT8_E5M2:
        ((uint8_t*)base)[i] = lz_float_to_fp8_e5m2(v);
        break;
    case DTYPE_FLOAT8_E4M3_FNUZ:
        ((uint8_t*)base)[i] = lz_float_to_fp8e4m3fnuz(v);
        break;
    case DTYPE_FLOAT8_E5M2_FNUZ:
        ((uint8_t*)base)[i] = lz_float_to_fp8e5m2fnuz(v);
        break;
    default:
        ((float*)base)[i] = v;
        break;
    }
}

int cml_cpu_lazy_fill(Tensor* out, float value) {
    if (!out || !out->data || out->numel == 0)
        return -1;
    void* p = out->data;
    for (size_t i = 0; i < out->numel; i++)
        cml_cpu_lazy_store_float_elem(p, i, out->dtype, value);
    return 0;
}

int cml_cpu_lazy_const(Tensor* out, const void* data, size_t data_size) {
    if (!out || !out->data || !data)
        return -1;
    size_t esz = cml_dtype_size(out->dtype);
    if (esz == 0)
        return -1;
    size_t max_b = out->numel * esz;
    size_t n = data_size < max_b ? data_size : max_b;
    memcpy(out->data, data, n);
    if (n < max_b)
        memset((uint8_t*)out->data + n, 0, max_b - n);
    return 0;
}

int cml_cpu_lazy_rand_uniform(Tensor* out) {
    if (!out || !out->data)
        return -1;
    for (size_t i = 0; i < out->numel; i++) {
        float u = (float)rand() / ((float)RAND_MAX + 1.0f);
        cml_cpu_lazy_store_float_elem(out->data, i, out->dtype, u);
    }
    return 0;
}

int cml_cpu_lazy_rand_normal(Tensor* out) {
    if (!out || !out->data)
        return -1;
    size_t i = 0;
    for (; i + 1 < out->numel; i += 2) {
        float u1 = (float)(rand() + 1) / ((float)RAND_MAX + 2.0f);
        float u2 = (float)rand() / ((float)RAND_MAX + 1.0f);
        float r  = sqrtf(-2.0f * logf(u1));
        float th = 2.0f * 3.14159265358979323846f * u2;
        float z0 = r * cosf(th);
        float z1 = r * sinf(th);
        cml_cpu_lazy_store_float_elem(out->data, i, out->dtype, z0);
        cml_cpu_lazy_store_float_elem(out->data, i + 1, out->dtype, z1);
    }
    if (i < out->numel) {
        float u1 = (float)(rand() + 1) / ((float)RAND_MAX + 2.0f);
        float u2 = (float)rand() / ((float)RAND_MAX + 1.0f);
        float z  = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
        cml_cpu_lazy_store_float_elem(out->data, i, out->dtype, z);
    }
    return 0;
}

int cml_cpu_lazy_arange(Tensor* out, float start, float step) {
    if (!out || !out->data)
        return -1;
    float val = start;
    for (size_t i = 0; i < out->numel; i++, val += step)
        cml_cpu_lazy_store_float_elem(out->data, i, out->dtype, val);
    return 0;
}

int cml_cpu_lazy_eye(Tensor* out, int n) {
    if (!out || !out->data || n <= 0)
        return -1;
    size_t esz = cml_dtype_size(out->dtype);
    memset(out->data, 0, out->numel * esz);
    for (int i = 0; i < n; i++)
        cml_cpu_lazy_store_float_elem(out->data, (size_t)(i * n + i), out->dtype, 1.0f);
    return 0;
}

int cml_cpu_lazy_rand_int(Tensor* out, int low, int high) {
    if (!out || !out->data || high <= low)
        return -1;
    int range = high - low;
    for (size_t i = 0; i < out->numel; i++) {
        int k = low + (int)(rand() % range);
        cml_cpu_lazy_store_float_elem(out->data, i, out->dtype, (float)k);
    }
    return 0;
}
