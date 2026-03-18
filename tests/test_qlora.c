#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "cml.h"
#include "nn/qlora.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-55s ", #name); \
    fflush(stdout); \
    if (test_##name()) { \
        tests_passed++; \
        printf("[PASS]\n"); \
    } else { \
        printf("[FAIL]\n"); \
    } \
} while(0)


static int test_nf4_roundtrip(void) {
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .has_dtype = true,
        .has_device = true
    };

    /* Create a float tensor with values roughly normally distributed in [-1, 1] */
    int numel = 128;
    float data[128];
    for (int i = 0; i < numel; i++) {
        /* Simple pattern: values spread across [-1, 1] */
        data[i] = -1.0f + 2.0f * (float)i / (float)(numel - 1);
    }
    int shape[1] = {numel};
    Tensor* original = tensor_from_data(data, shape, 1, &cfg);
    if (!original) return 0;

    /* Quantize to NF4 */
    float* scales = NULL;
    int num_scales = 0;
    int block_size = 64;
    Tensor* packed = cml_quantize_nf4(original, block_size, &scales, &num_scales);
    if (!packed) {
        tensor_free(original);
        return 0;
    }

    /* Verify we got the expected number of blocks */
    int expected_blocks = (numel + block_size - 1) / block_size;
    if (num_scales != expected_blocks) {
        printf("(expected %d blocks, got %d) ", expected_blocks, num_scales);
        free(scales);
        tensor_free(packed);
        tensor_free(original);
        return 0;
    }

    /* Dequantize back */
    Tensor* reconstructed = cml_dequantize_nf4(packed, scales, num_scales,
                                                block_size, (size_t)numel);
    if (!reconstructed) {
        free(scales);
        tensor_free(packed);
        tensor_free(original);
        return 0;
    }

    /* Check that reconstruction error is small (NF4 has limited precision) */
    tensor_ensure_executed(original);
    tensor_ensure_executed(reconstructed);
    float* orig_data = (float*)tensor_data_ptr(original);
    float* recon_data = (float*)tensor_data_ptr(reconstructed);

    float max_error = 0.0f;
    float sum_sq_error = 0.0f;
    for (int i = 0; i < numel; i++) {
        float err = fabsf(orig_data[i] - recon_data[i]);
        if (err > max_error) max_error = err;
        sum_sq_error += err * err;
    }
    float rmse = sqrtf(sum_sq_error / (float)numel);

    int ok = 1;
    /* NF4 with 16 levels should have max error < 0.15 for normalized values */
    if (max_error > 0.15f) {
        printf("(max_error=%.4f too large) ", max_error);
        ok = 0;
    }
    if (rmse > 0.08f) {
        printf("(rmse=%.4f too large) ", rmse);
        ok = 0;
    }

    free(scales);
    tensor_free(packed);
    tensor_free(reconstructed);
    tensor_free(original);
    return ok;
}


static int test_nf4_table_values(void) {
    int ok = 1;

    /* Table should have 16 entries */
    /* Check sorted (monotonically non-decreasing) */
    for (int i = 1; i < 16; i++) {
        if (CML_NF4_TABLE[i] < CML_NF4_TABLE[i - 1]) {
            printf("(table not sorted at index %d: %.4f < %.4f) ",
                   i, CML_NF4_TABLE[i], CML_NF4_TABLE[i - 1]);
            ok = 0;
        }
    }

    /* Check boundary values */
    if (fabsf(CML_NF4_TABLE[0] - (-1.0f)) > 1e-6f) {
        printf("(table[0]=%.4f expected -1.0) ", CML_NF4_TABLE[0]);
        ok = 0;
    }
    if (fabsf(CML_NF4_TABLE[15] - 1.0f) > 1e-6f) {
        printf("(table[15]=%.4f expected 1.0) ", CML_NF4_TABLE[15]);
        ok = 0;
    }

    /* Check approximate symmetry: table[0] = -table[15], table[7] = 0 */
    if (fabsf(CML_NF4_TABLE[0] + CML_NF4_TABLE[15]) > 1e-6f) {
        printf("(table not symmetric: [0]=%.4f, [15]=%.4f) ",
               CML_NF4_TABLE[0], CML_NF4_TABLE[15]);
        ok = 0;
    }
    if (fabsf(CML_NF4_TABLE[7]) > 1e-6f) {
        printf("(table[7]=%.4f expected 0.0) ", CML_NF4_TABLE[7]);
        ok = 0;
    }

    /* Check that the zero point (index 7) is exactly zero */
    if (CML_NF4_TABLE[7] != 0.0f) {
        printf("(table[7] not exactly 0) ");
        ok = 0;
    }

    return ok;
}


static int test_qlora_create_and_forward_shape(void) {
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .has_dtype = true,
        .has_device = true
    };

    /* Create a base weight [out=4, in=8] */
    int W_shape[2] = {4, 8};
    Tensor* base_weight = tensor_zeros(W_shape, 2, &cfg);
    if (!base_weight) return 0;

    /* Fill with some values */
    tensor_ensure_executed(base_weight);
    float* W_data = (float*)tensor_data_ptr(base_weight);
    for (int i = 0; i < 32; i++) {
        W_data[i] = 0.1f * (float)(i - 16);
    }

    /* Create QLoRA with rank=2, alpha=2.0, block_size=16 */
    CMLQLoRALinear* qlora = cml_qlora_linear_create(base_weight, 2, 2.0f, 16);
    if (!qlora) {
        tensor_free(base_weight);
        return 0;
    }

    int ok = 1;

    /* Verify dimensions */
    if (qlora->in_features != 8) {
        printf("(in_features=%d expected 8) ", qlora->in_features);
        ok = 0;
    }
    if (qlora->out_features != 4) {
        printf("(out_features=%d expected 4) ", qlora->out_features);
        ok = 0;
    }
    if (qlora->rank != 2) {
        printf("(rank=%d expected 2) ", qlora->rank);
        ok = 0;
    }

    /* Verify scaling = alpha / rank = 2.0 / 2 = 1.0 */
    if (fabsf(qlora->scaling - 1.0f) > 1e-6f) {
        printf("(scaling=%.4f expected 1.0) ", qlora->scaling);
        ok = 0;
    }

    /* Verify NF4 base weight was created */
    if (!qlora->base_weight_nf4) {
        printf("(base_weight_nf4 is NULL) ");
        ok = 0;
    }

    /* Create input [batch=3, in=8] */
    int x_shape[2] = {3, 8};
    Tensor* input = tensor_zeros(x_shape, 2, &cfg);
    if (!input) {
        cml_qlora_linear_free(qlora);
        tensor_free(base_weight);
        return 0;
    }
    tensor_ensure_executed(input);
    float* x_data = (float*)tensor_data_ptr(input);
    for (int i = 0; i < 24; i++) {
        x_data[i] = 0.1f * (float)i;
    }

    /* Forward pass */
    Tensor* output = cml_qlora_linear_forward(qlora, input);
    if (!output) {
        printf("(forward returned NULL) ");
        tensor_free(input);
        cml_qlora_linear_free(qlora);
        tensor_free(base_weight);
        return 0;
    }

    /* Verify output shape is [3, 4] */
    if (output->ndim != 2) {
        printf("(output ndim=%d expected 2) ", output->ndim);
        ok = 0;
    } else {
        if (output->shape[0] != 3) {
            printf("(output shape[0]=%d expected 3) ", output->shape[0]);
            ok = 0;
        }
        if (output->shape[1] != 4) {
            printf("(output shape[1]=%d expected 4) ", output->shape[1]);
            ok = 0;
        }
    }

    tensor_free(output);
    tensor_free(input);
    cml_qlora_linear_free(qlora);
    tensor_free(base_weight);
    return ok;
}


static int test_qlora_memory_savings(void) {
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .has_dtype = true,
        .has_device = true
    };

    int in_features = 256;
    int out_features = 128;
    int rank = 8;
    int block_size = 64;

    /* Create a base weight [128, 256] */
    int W_shape[2] = {out_features, in_features};
    Tensor* base_weight = tensor_zeros(W_shape, 2, &cfg);
    if (!base_weight) return 0;

    /* Fill with random-ish values */
    tensor_ensure_executed(base_weight);
    float* W_data = (float*)tensor_data_ptr(base_weight);
    for (int i = 0; i < in_features * out_features; i++) {
        W_data[i] = sinf((float)i * 0.01f) * 0.5f;
    }

    CMLQLoRALinear* qlora = cml_qlora_linear_create(base_weight, rank, (float)rank, block_size);
    if (!qlora) {
        tensor_free(base_weight);
        return 0;
    }

    size_t qlora_mem = cml_qlora_memory_usage(qlora);
    size_t full_mem = cml_qlora_full_memory_usage(in_features, out_features);

    int ok = 1;
    if (qlora_mem >= full_mem) {
        printf("(qlora_mem=%zu >= full_mem=%zu, no savings) ", qlora_mem, full_mem);
        ok = 0;
    }

    /* For 256x128 weight with rank=8:
     * Full: 256*128*4 = 131072 bytes
     * NF4:  256*128/2 = 16384 bytes (packed) + scales
     * LoRA: (8*256 + 128*8)*4 = 12288 bytes
     * Total QLoRA should be roughly 28k-30k, much less than 131k
     */
    float ratio = (float)qlora_mem / (float)full_mem;
    if (ratio > 0.5f) {
        printf("(ratio=%.2f, expected significant savings) ", ratio);
        ok = 0;
    }

    cml_qlora_linear_free(qlora);
    tensor_free(base_weight);
    return ok;
}


static int test_qlora_lora_b_zero_init(void) {
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .has_dtype = true,
        .has_device = true
    };

    /* Create a base weight [out=3, in=4] with known values */
    float W_data[] = {
        1.0f, 0.5f, 0.0f, -0.5f,
        0.0f, 1.0f, 0.5f,  0.0f,
       -0.5f, 0.0f, 1.0f,  0.5f
    };
    int W_shape[2] = {3, 4};
    Tensor* base_weight = tensor_from_data(W_data, W_shape, 2, &cfg);
    if (!base_weight) return 0;

    /* Create QLoRA with rank=2, alpha=2.0, block_size=4 */
    CMLQLoRALinear* qlora = cml_qlora_linear_create(base_weight, 2, 2.0f, 4);
    if (!qlora) {
        tensor_free(base_weight);
        return 0;
    }

    /* Verify B is all zeros */
    tensor_ensure_executed(qlora->lora_B);
    float* B_data = (float*)tensor_data_ptr(qlora->lora_B);
    if (!B_data) {
        cml_qlora_linear_free(qlora);
        tensor_free(base_weight);
        return 0;
    }

    int ok = 1;
    int B_size = qlora->out_features * qlora->rank;
    for (int i = 0; i < B_size; i++) {
        if (fabsf(B_data[i]) > 1e-10f) {
            printf("(B[%d]=%.6f expected 0) ", i, B_data[i]);
            ok = 0;
            break;
        }
    }

    /* Verify A is NOT all zeros */
    tensor_ensure_executed(qlora->lora_A);
    float* A_data = (float*)tensor_data_ptr(qlora->lora_A);
    float sum_abs = 0.0f;
    int A_size = qlora->rank * qlora->in_features;
    for (int i = 0; i < A_size; i++) {
        sum_abs += fabsf(A_data[i]);
    }
    if (sum_abs < 1e-10f) {
        printf("(A is all zeros, expected Xavier init) ");
        ok = 0;
    }

    /* Forward pass with B=0: LoRA contribution should be zero.
     * Output should equal base_out = input @ dequant(W_nf4)^T
     * First compute what the base-only output should be by dequantizing
     * and doing the matmul manually.
     */
    float x_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
    int x_shape[2] = {1, 4};
    Tensor* input = tensor_from_data(x_data, x_shape, 2, &cfg);
    if (!input) {
        cml_qlora_linear_free(qlora);
        tensor_free(base_weight);
        return 0;
    }

    /* Get the dequantized weight to compute expected output */
    Tensor* dequant_w = cml_nf4_tensor_dequantize(qlora->base_weight_nf4);
    if (!dequant_w) {
        tensor_free(input);
        cml_qlora_linear_free(qlora);
        tensor_free(base_weight);
        return 0;
    }

    tensor_ensure_executed(dequant_w);
    float* dw_data = (float*)tensor_data_ptr(dequant_w);

    /* Compute expected: x @ dequant_W^T */
    float expected[3];
    for (int o = 0; o < 3; o++) {
        expected[o] = 0.0f;
        for (int i = 0; i < 4; i++) {
            expected[o] += x_data[i] * dw_data[o * 4 + i];
        }
    }

    /* QLoRA forward */
    Tensor* output = cml_qlora_linear_forward(qlora, input);
    if (!output) {
        tensor_free(dequant_w);
        tensor_free(input);
        cml_qlora_linear_free(qlora);
        tensor_free(base_weight);
        return 0;
    }

    tensor_ensure_executed(output);
    float* out = (float*)tensor_data_ptr(output);

    /* Since B=0, output should exactly match base-only (dequantized) output */
    for (int o = 0; o < 3; o++) {
        if (fabsf(out[o] - expected[o]) > 1e-5f) {
            printf("(out[%d]=%.4f expected %.4f) ", o, out[o], expected[o]);
            ok = 0;
        }
    }

    tensor_free(output);
    tensor_free(dequant_w);
    tensor_free(input);
    cml_qlora_linear_free(qlora);
    tensor_free(base_weight);
    return ok;
}


static int test_qlora_forward_nonzero_lora(void) {
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .has_dtype = true,
        .has_device = true
    };

    /* Create a base weight [out=2, in=4] */
    float W_data[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f
    };
    int W_shape[2] = {2, 4};
    Tensor* base_weight = tensor_from_data(W_data, W_shape, 2, &cfg);
    if (!base_weight) return 0;

    /* Create QLoRA with rank=2, alpha=2.0, block_size=4 */
    CMLQLoRALinear* qlora = cml_qlora_linear_create(base_weight, 2, 2.0f, 4);
    if (!qlora) {
        tensor_free(base_weight);
        return 0;
    }

    /* First do a forward with B=0 to get baseline output */
    float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int x_shape[2] = {1, 4};
    Tensor* input = tensor_from_data(x_data, x_shape, 2, &cfg);
    if (!input) {
        cml_qlora_linear_free(qlora);
        tensor_free(base_weight);
        return 0;
    }

    Tensor* base_output = cml_qlora_linear_forward(qlora, input);
    if (!base_output) {
        tensor_free(input);
        cml_qlora_linear_free(qlora);
        tensor_free(base_weight);
        return 0;
    }

    tensor_ensure_executed(base_output);
    float* base_out = (float*)tensor_data_ptr(base_output);
    float base_vals[2] = {base_out[0], base_out[1]};

    /* Now set B to non-zero values */
    tensor_ensure_executed(qlora->lora_B);
    float* B_data = (float*)tensor_data_ptr(qlora->lora_B);
    /* B: [out=2, rank=2] */
    B_data[0] = 1.0f; B_data[1] = 0.0f;
    B_data[2] = 0.0f; B_data[3] = 1.0f;

    /* Forward again with non-zero LoRA */
    Tensor* lora_output = cml_qlora_linear_forward(qlora, input);
    if (!lora_output) {
        tensor_free(base_output);
        tensor_free(input);
        cml_qlora_linear_free(qlora);
        tensor_free(base_weight);
        return 0;
    }

    tensor_ensure_executed(lora_output);
    float* lora_out = (float*)tensor_data_ptr(lora_output);

    /* With non-zero B, the output should differ from base-only output */
    int ok = 1;
    float diff0 = fabsf(lora_out[0] - base_vals[0]);
    float diff1 = fabsf(lora_out[1] - base_vals[1]);

    if (diff0 < 1e-6f && diff1 < 1e-6f) {
        printf("(output unchanged with non-zero LoRA: diff0=%.6f, diff1=%.6f) ",
               diff0, diff1);
        ok = 0;
    }

    tensor_free(lora_output);
    tensor_free(base_output);
    tensor_free(input);
    cml_qlora_linear_free(qlora);
    tensor_free(base_weight);
    return ok;
}


int main(void) {
    printf("\n");
    printf("  QLoRA (Quantized LoRA) Tests\n");
    printf("\n");

    printf("NF4 Quantization:\n");
    TEST(nf4_roundtrip);
    TEST(nf4_table_values);

    printf("\nQLoRA Linear Layer:\n");
    TEST(qlora_create_and_forward_shape);
    TEST(qlora_lora_b_zero_init);
    TEST(qlora_forward_nonzero_lora);

    printf("\nMemory:\n");
    TEST(qlora_memory_savings);

    printf("\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
