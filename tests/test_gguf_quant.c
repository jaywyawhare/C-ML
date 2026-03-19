#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "core/gguf_quant.h"
#include "core/gguf.h"

/* Helper: convert float to fp16 (IEEE 754 half-precision) stored as uint16_t */
static uint16_t float_to_fp16(float value) {
    /* Simple conversion for small positive values */
    union { float f; uint32_t u; } bits;
    bits.f = value;
    uint32_t f32 = bits.u;
    uint32_t sign = (f32 >> 16) & 0x8000;
    int exponent = ((f32 >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (f32 >> 13) & 0x3FF;

    if (exponent <= 0) {
        return (uint16_t)sign; /* zero or subnormal */
    } else if (exponent >= 31) {
        return (uint16_t)(sign | 0x7C00); /* infinity */
    }
    return (uint16_t)(sign | (exponent << 10) | mantissa);
}

static void test_type_is_quantized(void) {
    printf("  test_type_is_quantized...");
    assert(gguf_type_is_quantized(GGUF_TENSOR_Q4_0) == true);
    assert(gguf_type_is_quantized(GGUF_TENSOR_Q4_1) == true);
    assert(gguf_type_is_quantized(GGUF_TENSOR_Q8_0) == true);
    assert(gguf_type_is_quantized(GGUF_TENSOR_Q4_K) == true);
    assert(gguf_type_is_quantized(GGUF_TENSOR_Q5_K) == true);
    assert(gguf_type_is_quantized(GGUF_TENSOR_Q6_K) == true);

    /* Non-quantized types should return false */
    assert(gguf_type_is_quantized(GGUF_TENSOR_F32) == false);
    assert(gguf_type_is_quantized(GGUF_TENSOR_F16) == false);
    printf(" PASS\n");
}

static void test_block_size(void) {
    printf("  test_block_size...");
    assert(gguf_quant_block_size(GGUF_TENSOR_Q4_0) == 32);
    assert(gguf_quant_block_size(GGUF_TENSOR_Q4_1) == 32);
    assert(gguf_quant_block_size(GGUF_TENSOR_Q8_0) == 32);
    assert(gguf_quant_block_size(GGUF_TENSOR_Q4_K) == 256);
    assert(gguf_quant_block_size(GGUF_TENSOR_Q5_K) == 256);
    assert(gguf_quant_block_size(GGUF_TENSOR_Q6_K) == 256);
    printf(" PASS\n");
}

static void test_type_size(void) {
    printf("  test_type_size...");
    /* Q4_0: 2 bytes d + 16 bytes qs = 18 */
    assert(gguf_quant_type_size(GGUF_TENSOR_Q4_0) == sizeof(BlockQ4_0));
    /* Q4_1: 2 bytes d + 2 bytes m + 16 bytes qs = 20 */
    assert(gguf_quant_type_size(GGUF_TENSOR_Q4_1) == sizeof(BlockQ4_1));
    /* Q8_0: 2 bytes d + 32 bytes qs = 34 */
    assert(gguf_quant_type_size(GGUF_TENSOR_Q8_0) == sizeof(BlockQ8_0));
    /* Q4_K: 2+2+12+128 = 144 */
    assert(gguf_quant_type_size(GGUF_TENSOR_Q4_K) == sizeof(BlockQ4_K));
    /* Q5_K: 2+2+12+32+128 = 176 */
    assert(gguf_quant_type_size(GGUF_TENSOR_Q5_K) == sizeof(BlockQ5_K));
    /* Q6_K: 128+64+16+2 = 210 */
    assert(gguf_quant_type_size(GGUF_TENSOR_Q6_K) == sizeof(BlockQ6_K));
    printf(" PASS\n");
}

static void test_q8_0_dequantize(void) {
    printf("  test_q8_0_dequantize...");
    BlockQ8_0 block;
    memset(&block, 0, sizeof(block));

    /* Set delta (d) to 1.0 in fp16 */
    block.d = float_to_fp16(1.0f);

    /* Set all quants to 1 */
    for (int i = 0; i < QK8_0; i++) {
        block.qs[i] = 1;
    }

    float output[QK8_0];
    memset(output, 0, sizeof(output));

    int ret = gguf_dequantize(GGUF_TENSOR_Q8_0, &block, output, QK8_0);
    assert(ret == 0);

    /* Each output should be d * qs[i] = 1.0 * 1 = 1.0 */
    for (int i = 0; i < QK8_0; i++) {
        assert(fabsf(output[i] - 1.0f) < 1e-2f);
    }
    printf(" PASS\n");
}

static void test_q4_0_dequantize(void) {
    printf("  test_q4_0_dequantize...");
    BlockQ4_0 block;
    memset(&block, 0, sizeof(block));

    /* Set delta to 2.0 in fp16 */
    block.d = float_to_fp16(2.0f);

    /*
     * Q4_0 stores nibbles. Each byte holds two 4-bit values.
     * Values are stored as unsigned 0..15 and shifted by -8 to get signed range [-8..7].
     * Set all nibbles to 0x88 => low nibble = 8, high nibble = 8.
     * Dequantized: (8 - 8) * 2.0 = 0.0
     */
    for (int i = 0; i < QK4_0 / 2; i++) {
        block.qs[i] = 0x88;
    }

    float output[QK4_0];
    memset(output, 0, sizeof(output));

    int ret = gguf_dequantize(GGUF_TENSOR_Q4_0, &block, output, QK4_0);
    assert(ret == 0);

    /* All values should be approximately 0.0 since (8-8)*delta = 0 */
    for (int i = 0; i < QK4_0; i++) {
        assert(fabsf(output[i]) < 1e-1f);
    }

    /* Now set nibbles to 0x99 => low nibble = 9, high nibble = 9. */
    /* Dequantized: (9 - 8) * 2.0 = 2.0 */
    for (int i = 0; i < QK4_0 / 2; i++) {
        block.qs[i] = 0x99;
    }

    ret = gguf_dequantize(GGUF_TENSOR_Q4_0, &block, output, QK4_0);
    assert(ret == 0);

    /* All values should be approximately 2.0 since (9-8)*delta = 2.0 */
    for (int i = 0; i < QK4_0; i++) {
        assert(fabsf(output[i] - 2.0f) < 1e-1f);
    }
    printf(" PASS\n");
}

int main(void) {
    printf("GGUF Quantization Tests\n");

    test_type_is_quantized();
    test_block_size();
    test_type_size();
    test_q8_0_dequantize();
    test_q4_0_dequantize();

    printf("All GGUF quantization tests passed.\n");
    return 0;
}
