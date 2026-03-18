#ifndef CML_CORE_GGUF_QUANT_H
#define CML_CORE_GGUF_QUANT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "core/gguf.h"

#define QK4_0 32
typedef struct {
    uint16_t d;           /* delta (fp16) */
    uint8_t  qs[QK4_0/2]; /* nibbles / quants */
} BlockQ4_0;

#define QK4_1 32
typedef struct {
    uint16_t d;           /* delta (fp16) */
    uint16_t m;           /* min (fp16) */
    uint8_t  qs[QK4_1/2]; /* nibbles / quants */
} BlockQ4_1;

#define QK8_0 32
typedef struct {
    uint16_t d;           /* delta (fp16) */
    int8_t   qs[QK8_0];  /* quants */
} BlockQ8_0;

#define QK_K 256

typedef struct {
    uint16_t d;           /* super-block scale (fp16) */
    uint16_t dmin;        /* super-block min   (fp16) */
    uint8_t  scales[12];  /* 4-bit scales & mins for sub-blocks */
    uint8_t  qs[QK_K/2]; /* 4-bit quants */
} BlockQ4_K;

typedef struct {
    uint16_t d;           /* super-block scale (fp16) */
    uint16_t dmin;        /* super-block min   (fp16) */
    uint8_t  scales[12];  /* 4-bit scales & mins */
    uint8_t  qh[QK_K/8]; /* high bits */
    uint8_t  qs[QK_K/2]; /* 4-bit quants (low bits) */
} BlockQ5_K;

typedef struct {
    uint8_t  ql[QK_K/2]; /* quants – lower 4 bits */
    uint8_t  qh[QK_K/4]; /* quants – upper 2 bits */
    int8_t   scales[QK_K/16]; /* scales, quantized with 8 bits */
    uint16_t d;           /* super-block scale (fp16) */
} BlockQ6_K;

bool gguf_type_is_quantized(GGUFTensorType type);
int gguf_quant_block_size(GGUFTensorType type);
size_t gguf_quant_type_size(GGUFTensorType type);

/* numel must be multiple of block size */
int gguf_dequantize(GGUFTensorType type, const void* src, float* dst, size_t numel);

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_GGUF_QUANT_H */
