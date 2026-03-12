/**
 * @file gguf_quant.c
 * @brief Dequantization kernels for GGUF quantized formats
 *
 * Implements dequantization for Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K block
 * formats, matching the llama.cpp layout.  All routines convert to float32.
 */

#include "core/gguf_quant.h"
#include <string.h>
#include <stdlib.h>

/* ── GGUF tensor type numeric IDs for quantized formats ──────────────────
 * These follow the GGUF spec and may not yet be present in the project's
 * GGUFTensorType enum, so we use the raw integer values directly.         */
#define GGUF_TENSOR_TYPE_Q4_0  2
#define GGUF_TENSOR_TYPE_Q4_1  3
#define GGUF_TENSOR_TYPE_Q8_0  7
#define GGUF_TENSOR_TYPE_Q4_K  12
#define GGUF_TENSOR_TYPE_Q5_K  13
#define GGUF_TENSOR_TYPE_Q6_K  14

/* ── fp16 -> fp32 conversion (pure bit manipulation, no HW intrinsics) ── */

/**
 * Convert an IEEE-754 half-precision (binary16) value stored in a uint16_t
 * to a single-precision float using only integer bit manipulation.
 *
 * Layout of binary16:  1 sign | 5 exponent | 10 mantissa
 * Layout of binary32:  1 sign | 8 exponent | 23 mantissa
 */
static float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;   /* bit 31          */
    const uint32_t exp  = (h >> 10) & 0x1Fu;                /* 5-bit exponent  */
    const uint32_t mant = h & 0x03FFu;                      /* 10-bit mantissa */

    if (exp == 0) {
        if (mant == 0) {
            /* Signed zero */
            float r;
            uint32_t v = sign;
            memcpy(&r, &v, sizeof(r));
            return r;
        }
        /* Subnormal: shift mantissa until the implicit leading 1 sits at
           bit 10, counting shifts to adjust the exponent.                   */
        uint32_t m = mant;
        int shift = 0;
        while ((m & 0x0400u) == 0) {
            m <<= 1;
            shift++;
        }
        m &= ~0x0400u;            /* remove the now-explicit leading 1      */
        shift++;                   /* account for the leading-1 position     */
        /* fp32 exponent: stored = 127 + (1 - 15 - shift) = 113 - shift     */
        uint32_t v = sign | ((uint32_t)(113 - shift) << 23) | (m << 13);
        float r;
        memcpy(&r, &v, sizeof(r));
        return r;
    }

    if (exp == 31) {
        /* Inf or NaN -- map directly to fp32 inf / NaN */
        uint32_t v = sign | 0x7F800000u | ((uint32_t)mant << 13);
        float r;
        memcpy(&r, &v, sizeof(r));
        return r;
    }

    /* Normal number.  Re-bias: fp16 bias = 15, fp32 bias = 127.
       Stored fp32 exponent = exp - 15 + 127 = exp + 112.                   */
    uint32_t v = sign | ((exp + 112u) << 23) | ((uint32_t)mant << 13);
    float r;
    memcpy(&r, &v, sizeof(r));
    return r;
}

/* ── Per-type dequantization kernels ─────────────────────────────────────
 *
 * Every kernel takes a source buffer of packed blocks, a destination float
 * buffer, and the total number of blocks.  The caller (gguf_dequantize)
 * has already validated alignment of numel to the block size.
 * ----------------------------------------------------------------------- */

/**
 * Q4_0: 32 elements per block.
 *   Block: fp16 delta `d`, then 16 bytes of nibble-packed quants.
 *   Byte j holds quant[j] (low nibble) and quant[j+16] (high nibble).
 *   value = (nibble - 8) * d
 */
static void dequantize_q4_0(const void *src, float *dst, size_t num_blocks) {
    const BlockQ4_0 *blocks = (const BlockQ4_0 *)src;

    for (size_t b = 0; b < num_blocks; b++) {
        const float d = fp16_to_fp32(blocks[b].d);
        const uint8_t *qs = blocks[b].qs;

        for (int j = 0; j < QK4_0 / 2; j++) {
            const uint8_t byte = qs[j];
            const int lo = (int)(byte & 0x0Fu);
            const int hi = (int)(byte >> 4);

            dst[b * QK4_0 + j]              = (float)(lo - 8) * d;
            dst[b * QK4_0 + j + QK4_0 / 2]  = (float)(hi - 8) * d;
        }
    }
}

/**
 * Q4_1: 32 elements per block.
 *   Block: fp16 delta `d`, fp16 min `m`, then 16 bytes of nibble-packed quants.
 *   Same nibble packing as Q4_0.
 *   value = nibble * d + m
 */
static void dequantize_q4_1(const void *src, float *dst, size_t num_blocks) {
    const BlockQ4_1 *blocks = (const BlockQ4_1 *)src;

    for (size_t b = 0; b < num_blocks; b++) {
        const float d = fp16_to_fp32(blocks[b].d);
        const float m = fp16_to_fp32(blocks[b].m);
        const uint8_t *qs = blocks[b].qs;

        for (int j = 0; j < QK4_1 / 2; j++) {
            const uint8_t byte = qs[j];
            const int lo = (int)(byte & 0x0Fu);
            const int hi = (int)(byte >> 4);

            dst[b * QK4_1 + j]              = (float)lo * d + m;
            dst[b * QK4_1 + j + QK4_1 / 2]  = (float)hi * d + m;
        }
    }
}

/**
 * Q8_0: 32 elements per block.
 *   Block: fp16 delta `d`, then 32 signed int8 quants.
 *   value = qs[i] * d
 */
static void dequantize_q8_0(const void *src, float *dst, size_t num_blocks) {
    const BlockQ8_0 *blocks = (const BlockQ8_0 *)src;

    for (size_t b = 0; b < num_blocks; b++) {
        const float d = fp16_to_fp32(blocks[b].d);
        const int8_t *qs = blocks[b].qs;

        for (int j = 0; j < QK8_0; j++) {
            dst[b * QK8_0 + j] = (float)qs[j] * d;
        }
    }
}

/**
 * Q4_K: 256 elements per super-block (8 sub-blocks of 32 elements).
 *
 *   Block layout:
 *     fp16    d          -- super-block scale
 *     fp16    dmin       -- super-block min
 *     uint8   scales[12] -- packed 6-bit sub-block scales and mins
 *     uint8   qs[128]    -- 4-bit quants (nibble-packed, 2 per byte)
 *
 *   Scale/min packing in scales[12] for 8 sub-blocks (i = 0..7):
 *     For i in 0..3:
 *       sc[i] = scales[i]     & 0x3F    (low 6 bits)
 *       mn[i] = scales[i + 4] & 0x3F    (low 6 bits)
 *     For i in 4..7  (let j = i - 4):
 *       sc[i] = (scales[j + 8] & 0x0F) | ((scales[j]     >> 6) << 4)
 *       mn[i] = (scales[j + 8] >> 4)   | ((scales[j + 4] >> 6) << 4)
 *
 *   Each sub-block of 32 elements is nibble-packed into 16 bytes.
 *   For sub-block i, the 16 bytes start at qs[i * 16].
 *     quant[k]      = qs[i*16 + k] & 0x0F      (k = 0..15)
 *     quant[k + 16] = qs[i*16 + k] >> 4
 *
 *   value = quant * (d * sc) - dmin * mn
 */
static void dequantize_q4_k(const void *src, float *dst, size_t num_blocks) {
    const BlockQ4_K *blocks = (const BlockQ4_K *)src;

    for (size_t b = 0; b < num_blocks; b++) {
        const BlockQ4_K *blk = &blocks[b];
        const float d    = fp16_to_fp32(blk->d);
        const float dmin = fp16_to_fp32(blk->dmin);
        const uint8_t *sc_raw = blk->scales;

        /* Decode the 6-bit scales and mins for 8 sub-blocks */
        uint8_t sc[8];
        uint8_t mn[8];

        for (int i = 0; i < 4; i++) {
            sc[i] = sc_raw[i]     & 0x3Fu;
            mn[i] = sc_raw[i + 4] & 0x3Fu;
        }
        for (int i = 4; i < 8; i++) {
            const int j = i - 4;
            sc[i] = (uint8_t)((sc_raw[j + 8] & 0x0Fu) | ((sc_raw[j]     >> 6) << 4));
            mn[i] = (uint8_t)((sc_raw[j + 8] >> 4)     | ((sc_raw[j + 4] >> 6) << 4));
        }

        /* Dequantize each of the 8 sub-blocks */
        for (int i = 0; i < 8; i++) {
            const float scale = d    * (float)sc[i];
            const float min   = dmin * (float)mn[i];
            const uint8_t *q  = blk->qs + i * 16;

            for (int k = 0; k < 16; k++) {
                const uint8_t byte = q[k];
                const int lo = (int)(byte & 0x0Fu);
                const int hi = (int)(byte >> 4);

                dst[b * QK_K + i * 32 + k]      = (float)lo * scale - min;
                dst[b * QK_K + i * 32 + k + 16]  = (float)hi * scale - min;
            }
        }
    }
}

/**
 * Q5_K: 256 elements per super-block (8 sub-blocks of 32 elements).
 *
 *   Like Q4_K but with 5-bit quants.  The extra high bit per element is
 *   stored in qh[32] (one bit per element, 256 bits = 32 bytes).
 *
 *   Block layout:
 *     fp16    d          -- super-block scale
 *     fp16    dmin       -- super-block min
 *     uint8   scales[12] -- packed 6-bit sub-block scales and mins (same as Q4_K)
 *     uint8   qh[32]     -- high bit for each element
 *     uint8   qs[128]    -- 4-bit low quants (nibble-packed, same as Q4_K)
 *
 *   For element at flat index e within the super-block:
 *     lo4      = 4-bit nibble from qs  (same packing as Q4_K)
 *     high_bit = (qh[e / 8] >> (e % 8)) & 1
 *     q5       = lo4 | (high_bit << 4)        (range 0..31)
 *     value    = q5 * (d * sc) - dmin * mn
 */
static void dequantize_q5_k(const void *src, float *dst, size_t num_blocks) {
    const BlockQ5_K *blocks = (const BlockQ5_K *)src;

    for (size_t b = 0; b < num_blocks; b++) {
        const BlockQ5_K *blk = &blocks[b];
        const float d    = fp16_to_fp32(blk->d);
        const float dmin = fp16_to_fp32(blk->dmin);
        const uint8_t *sc_raw = blk->scales;

        /* Unpack sub-block scales and mins -- identical to Q4_K */
        uint8_t sc[8];
        uint8_t mn[8];

        for (int i = 0; i < 4; i++) {
            sc[i] = sc_raw[i]     & 0x3Fu;
            mn[i] = sc_raw[i + 4] & 0x3Fu;
        }
        for (int i = 4; i < 8; i++) {
            const int j = i - 4;
            sc[i] = (uint8_t)((sc_raw[j + 8] & 0x0Fu) | ((sc_raw[j]     >> 6) << 4));
            mn[i] = (uint8_t)((sc_raw[j + 8] >> 4)     | ((sc_raw[j + 4] >> 6) << 4));
        }

        /* Dequantize each of the 8 sub-blocks */
        for (int i = 0; i < 8; i++) {
            const float scale = d    * (float)sc[i];
            const float min   = dmin * (float)mn[i];
            const uint8_t *q  = blk->qs + i * 16;
            const int base    = i * 32;

            for (int k = 0; k < 16; k++) {
                const uint8_t byte = q[k];
                const int lo4 = (int)(byte & 0x0Fu);
                const int hi4 = (int)(byte >> 4);

                /* Element indices within the super-block */
                const int idx_lo = base + k;
                const int idx_hi = base + k + 16;

                /* Extract the 5th bit from qh */
                const int hbit_lo = (blk->qh[idx_lo / 8] >> (idx_lo % 8)) & 1;
                const int hbit_hi = (blk->qh[idx_hi / 8] >> (idx_hi % 8)) & 1;

                const int q5_lo = lo4 | (hbit_lo << 4);
                const int q5_hi = hi4 | (hbit_hi << 4);

                dst[b * QK_K + idx_lo] = (float)q5_lo * scale - min;
                dst[b * QK_K + idx_hi] = (float)q5_hi * scale - min;
            }
        }
    }
}

/**
 * Q6_K: 256 elements per super-block (16 sub-blocks of 16 elements).
 *
 *   6-bit quants formed from 4-bit lower (ql) and 2-bit upper (qh) parts,
 *   with scales quantized to 8 bits.
 *
 *   Block layout:
 *     uint8   ql[128]    -- lower 4 bits of each quant (nibble-packed, 2 per byte)
 *     uint8   qh[64]     -- upper 2 bits of each quant (packed 4 per byte)
 *     int8    scales[16] -- 8-bit signed scale for each of 16 sub-blocks
 *     fp16    d          -- super-block scale
 *
 *   For element at flat index e (0..255):
 *     low4     = (e even) ? ql[e/2] & 0x0F : ql[e/2] >> 4
 *     high2    = (qh[e/4] >> (2 * (e % 4))) & 0x03
 *     q6       = low4 | (high2 << 4)          (range 0..63)
 *     signed_q = q6 - 32                      (range -32..31)
 *     value    = d * scales[e / 16] * signed_q
 */
static void dequantize_q6_k(const void *src, float *dst, size_t num_blocks) {
    const BlockQ6_K *blocks = (const BlockQ6_K *)src;

    for (size_t b = 0; b < num_blocks; b++) {
        const BlockQ6_K *blk = &blocks[b];
        const float d = fp16_to_fp32(blk->d);

        for (int e = 0; e < QK_K; e++) {
            /* Lower 4 bits: nibble-packed in ql */
            int low4;
            if ((e % 2) == 0) {
                low4 = (int)(blk->ql[e / 2] & 0x0Fu);
            } else {
                low4 = (int)(blk->ql[e / 2] >> 4);
            }

            /* Upper 2 bits: packed 4 per byte in qh */
            const int qh_shift = 2 * (e % 4);
            const int high2 = (int)((blk->qh[e / 4] >> qh_shift) & 0x03u);

            const int q6 = low4 | (high2 << 4);     /* 6-bit: 0..63       */
            const int signed_q = q6 - 32;            /* centered: -32..31  */

            const float sc = (float)blk->scales[e / 16];

            dst[b * QK_K + e] = d * sc * (float)signed_q;
        }
    }
}

/* ── Public API ──────────────────────────────────────────────────────────── */

bool gguf_type_is_quantized(GGUFTensorType type) {
    switch ((int)type) {
        case GGUF_TENSOR_TYPE_Q4_0:
        case GGUF_TENSOR_TYPE_Q4_1:
        case GGUF_TENSOR_TYPE_Q8_0:
        case GGUF_TENSOR_TYPE_Q4_K:
        case GGUF_TENSOR_TYPE_Q5_K:
        case GGUF_TENSOR_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

int gguf_quant_block_size(GGUFTensorType type) {
    switch ((int)type) {
        case GGUF_TENSOR_TYPE_Q4_0: return QK4_0;   /* 32  */
        case GGUF_TENSOR_TYPE_Q4_1: return QK4_1;   /* 32  */
        case GGUF_TENSOR_TYPE_Q8_0: return QK8_0;   /* 32  */
        case GGUF_TENSOR_TYPE_Q4_K: return QK_K;    /* 256 */
        case GGUF_TENSOR_TYPE_Q5_K: return QK_K;    /* 256 */
        case GGUF_TENSOR_TYPE_Q6_K: return QK_K;    /* 256 */
        default:                    return 0;
    }
}

size_t gguf_quant_type_size(GGUFTensorType type) {
    switch ((int)type) {
        case GGUF_TENSOR_TYPE_Q4_0: return sizeof(BlockQ4_0);
        case GGUF_TENSOR_TYPE_Q4_1: return sizeof(BlockQ4_1);
        case GGUF_TENSOR_TYPE_Q8_0: return sizeof(BlockQ8_0);
        case GGUF_TENSOR_TYPE_Q4_K: return sizeof(BlockQ4_K);
        case GGUF_TENSOR_TYPE_Q5_K: return sizeof(BlockQ5_K);
        case GGUF_TENSOR_TYPE_Q6_K: return sizeof(BlockQ6_K);
        default:                    return 0;
    }
}

int gguf_dequantize(GGUFTensorType type, const void *src, float *dst, size_t numel) {
    if (!src || !dst || numel == 0) {
        return -1;
    }

    const int block_size = gguf_quant_block_size(type);
    if (block_size == 0) {
        return -1;   /* not a supported quantized type */
    }

    if (numel % (size_t)block_size != 0) {
        return -1;   /* numel must be a multiple of block size */
    }

    const size_t num_blocks = numel / (size_t)block_size;

    switch ((int)type) {
        case GGUF_TENSOR_TYPE_Q4_0:
            dequantize_q4_0(src, dst, num_blocks);
            break;
        case GGUF_TENSOR_TYPE_Q4_1:
            dequantize_q4_1(src, dst, num_blocks);
            break;
        case GGUF_TENSOR_TYPE_Q8_0:
            dequantize_q8_0(src, dst, num_blocks);
            break;
        case GGUF_TENSOR_TYPE_Q4_K:
            dequantize_q4_k(src, dst, num_blocks);
            break;
        case GGUF_TENSOR_TYPE_Q5_K:
            dequantize_q5_k(src, dst, num_blocks);
            break;
        case GGUF_TENSOR_TYPE_Q6_K:
            dequantize_q6_k(src, dst, num_blocks);
            break;
        default:
            return -1;
    }

    return 0;
}
