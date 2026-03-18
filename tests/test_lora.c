#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "cml.h"
#include "nn/lora.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-45s ", #name); \
    fflush(stdout); \
    if (test_##name()) { \
        tests_passed++; \
        printf("[PASS]\n"); \
    } else { \
        printf("[FAIL]\n"); \
    } \
} while(0)


static int test_lora_linear_forward(void) {
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .has_dtype = true,
        .has_device = true
    };

    /* Create a base weight [out=3, in=4] */
    float W_data[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    int W_shape[2] = {3, 4};
    Tensor* base_weight = tensor_from_data(W_data, W_shape, 2, &cfg);
    if (!base_weight) return 0;

    /* Create LoRA with rank=2, alpha=2.0 */
    CMLLoRALinear* lora = cml_lora_linear_create(base_weight, 2, 2.0f);
    if (!lora) { tensor_free(base_weight); return 0; }

    /* Verify dimensions */
    if (lora->in_features != 4) { cml_lora_linear_free(lora); tensor_free(base_weight); return 0; }
    if (lora->out_features != 3) { cml_lora_linear_free(lora); tensor_free(base_weight); return 0; }
    if (lora->rank != 2) { cml_lora_linear_free(lora); tensor_free(base_weight); return 0; }

    /* Verify scaling = alpha / rank = 2.0 / 2 = 1.0 */
    if (fabsf(lora->scaling - 1.0f) > 1e-6f) {
        cml_lora_linear_free(lora);
        tensor_free(base_weight);
        return 0;
    }

    /* Create input [batch=2, in=4] */
    float x_data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        0.5f, 1.5f, 2.5f, 3.5f
    };
    int x_shape[2] = {2, 4};
    Tensor* input = tensor_from_data(x_data, x_shape, 2, &cfg);
    if (!input) { cml_lora_linear_free(lora); tensor_free(base_weight); return 0; }

    /* Forward pass */
    Tensor* output = cml_lora_linear_forward(lora, input);
    if (!output) {
        tensor_free(input);
        cml_lora_linear_free(lora);
        tensor_free(base_weight);
        return 0;
    }

    /* Verify output shape is [2, 3] */
    if (output->ndim != 2 || output->shape[0] != 2 || output->shape[1] != 3) {
        tensor_free(output);
        tensor_free(input);
        cml_lora_linear_free(lora);
        tensor_free(base_weight);
        return 0;
    }

    /* With B initialized to zero, LoRA contribution is zero.
     * So output should match base_out = input @ W^T.
     * W is identity-like (first 3 cols of I_4):
     *   row 0: [1,2,3,4] @ [1,0,0,0]^T = 1.0
     *   row 0: [1,2,3,4] @ [0,1,0,0]^T = 2.0
     *   row 0: [1,2,3,4] @ [0,0,1,0]^T = 3.0
     *   row 1: [0.5,1.5,2.5,3.5] @ [1,0,0,0]^T = 0.5
     *   etc.
     */
    tensor_ensure_executed(output);
    float* out = (float*)tensor_data_ptr(output);
    int ok = 1;
    if (fabsf(out[0] - 1.0f) > 1e-5f) ok = 0;
    if (fabsf(out[1] - 2.0f) > 1e-5f) ok = 0;
    if (fabsf(out[2] - 3.0f) > 1e-5f) ok = 0;
    if (fabsf(out[3] - 0.5f) > 1e-5f) ok = 0;
    if (fabsf(out[4] - 1.5f) > 1e-5f) ok = 0;
    if (fabsf(out[5] - 2.5f) > 1e-5f) ok = 0;

    tensor_free(output);
    tensor_free(input);
    cml_lora_linear_free(lora);
    tensor_free(base_weight);
    return ok;
}


static int test_lora_merge_unmerge(void) {
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .has_dtype = true,
        .has_device = true
    };

    /* Create a simple base weight [out=2, in=3] */
    float W_data[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    int W_shape[2] = {2, 3};
    Tensor* base_weight = tensor_from_data(W_data, W_shape, 2, &cfg);
    if (!base_weight) return 0;

    /* Save original weight values for later comparison */
    tensor_ensure_executed(base_weight);
    float* W_ptr = (float*)tensor_data_ptr(base_weight);
    float original_W[6];
    memcpy(original_W, W_ptr, 6 * sizeof(float));

    /* Create LoRA with rank=1, alpha=1.0 (scaling = 1.0) */
    CMLLoRALinear* lora = cml_lora_linear_create(base_weight, 1, 1.0f);
    if (!lora) { tensor_free(base_weight); return 0; }

    /* Manually set A and B to known values for predictable merge result */
    tensor_ensure_executed(lora->lora_A);
    tensor_ensure_executed(lora->lora_B);
    float* A_ptr = (float*)tensor_data_ptr(lora->lora_A);
    float* B_ptr = (float*)tensor_data_ptr(lora->lora_B);

    /* A: [1, 3] = [1.0, 0.0, 0.0] */
    A_ptr[0] = 1.0f; A_ptr[1] = 0.0f; A_ptr[2] = 0.0f;
    /* B: [2, 1] = [0.5, 0.5] */
    B_ptr[0] = 0.5f; B_ptr[1] = 0.5f;

    /*
     * After merge: W += scaling * B @ A
     * B @ A = [2,1] @ [1,3] = [2,3]
     *   [0.5] @ [1.0, 0.0, 0.0] = [0.5, 0.0, 0.0]
     *   [0.5]                     = [0.5, 0.0, 0.0]
     * scaling = 1.0, so W += [[0.5,0,0],[0.5,0,0]]
     * Expected: [[1.5,2,3],[4.5,5,6]]
     */
    if (cml_lora_linear_merge(lora) != 0) {
        cml_lora_linear_free(lora);
        tensor_free(base_weight);
        return 0;
    }

    /* Verify merge happened */
    if (!lora->merged) {
        cml_lora_linear_free(lora);
        tensor_free(base_weight);
        return 0;
    }

    W_ptr = (float*)tensor_data_ptr(base_weight);
    int ok = 1;
    if (fabsf(W_ptr[0] - 1.5f) > 1e-5f) ok = 0;   /* 1.0 + 0.5 */
    if (fabsf(W_ptr[1] - 2.0f) > 1e-5f) ok = 0;   /* 2.0 + 0.0 */
    if (fabsf(W_ptr[2] - 3.0f) > 1e-5f) ok = 0;   /* 3.0 + 0.0 */
    if (fabsf(W_ptr[3] - 4.5f) > 1e-5f) ok = 0;   /* 4.0 + 0.5 */
    if (fabsf(W_ptr[4] - 5.0f) > 1e-5f) ok = 0;   /* 5.0 + 0.0 */
    if (fabsf(W_ptr[5] - 6.0f) > 1e-5f) ok = 0;   /* 6.0 + 0.0 */

    if (!ok) {
        printf("(merge values wrong) ");
        cml_lora_linear_free(lora);
        tensor_free(base_weight);
        return 0;
    }

    /* Unmerge: should restore original weights */
    if (cml_lora_linear_unmerge(lora) != 0) {
        cml_lora_linear_free(lora);
        tensor_free(base_weight);
        return 0;
    }

    if (lora->merged) {
        cml_lora_linear_free(lora);
        tensor_free(base_weight);
        return 0;
    }

    W_ptr = (float*)tensor_data_ptr(base_weight);
    for (int i = 0; i < 6; i++) {
        if (fabsf(W_ptr[i] - original_W[i]) > 1e-5f) {
            printf("(unmerge failed at index %d: got %.4f expected %.4f) ",
                   i, W_ptr[i], original_W[i]);
            ok = 0;
        }
    }

    cml_lora_linear_free(lora);
    tensor_free(base_weight);
    return ok;
}


static int test_lora_adapter(void) {
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .has_dtype = true,
        .has_device = true
    };

    /* Create two base weight tensors */
    float W1_data[] = {1.0f, 0.0f, 0.0f, 1.0f};
    int W1_shape[2] = {2, 2};
    Tensor* W1 = tensor_from_data(W1_data, W1_shape, 2, &cfg);
    if (!W1) return 0;

    float W2_data[] = {2.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f};
    int W2_shape[2] = {2, 3};
    Tensor* W2 = tensor_from_data(W2_data, W2_shape, 2, &cfg);
    if (!W2) { tensor_free(W1); return 0; }

    /* Save original weight values */
    tensor_ensure_executed(W1);
    tensor_ensure_executed(W2);
    float orig_W1[4], orig_W2[6];
    memcpy(orig_W1, (float*)tensor_data_ptr(W1), 4 * sizeof(float));
    memcpy(orig_W2, (float*)tensor_data_ptr(W2), 6 * sizeof(float));

    /* Create adapter */
    CMLLoRAAdapter* adapter = cml_lora_adapter_create("test_adapter", 2, 2.0f);
    if (!adapter) { tensor_free(W1); tensor_free(W2); return 0; }

    /* Verify adapter properties */
    if (adapter->rank != 2) { cml_lora_adapter_free(adapter); tensor_free(W1); tensor_free(W2); return 0; }
    if (adapter->num_layers != 0) { cml_lora_adapter_free(adapter); tensor_free(W1); tensor_free(W2); return 0; }

    /* Create and add LoRA layers */
    CMLLoRALinear* lora1 = cml_lora_linear_create(W1, 2, 2.0f);
    CMLLoRALinear* lora2 = cml_lora_linear_create(W2, 2, 2.0f);
    if (!lora1 || !lora2) {
        if (lora1) cml_lora_linear_free(lora1);
        if (lora2) cml_lora_linear_free(lora2);
        cml_lora_adapter_free(adapter);
        tensor_free(W1);
        tensor_free(W2);
        return 0;
    }

    if (cml_lora_adapter_add_layer(adapter, lora1) != 0 ||
        cml_lora_adapter_add_layer(adapter, lora2) != 0) {
        cml_lora_adapter_free(adapter);
        tensor_free(W1);
        tensor_free(W2);
        return 0;
    }

    if (adapter->num_layers != 2) {
        cml_lora_adapter_free(adapter);
        tensor_free(W1);
        tensor_free(W2);
        return 0;
    }

    /* Merge all */
    if (cml_lora_adapter_merge_all(adapter) != 0) {
        cml_lora_adapter_free(adapter);
        tensor_free(W1);
        tensor_free(W2);
        return 0;
    }

    if (!adapter->merged) {
        cml_lora_adapter_free(adapter);
        tensor_free(W1);
        tensor_free(W2);
        return 0;
    }

    /* Unmerge all */
    if (cml_lora_adapter_unmerge_all(adapter) != 0) {
        cml_lora_adapter_free(adapter);
        tensor_free(W1);
        tensor_free(W2);
        return 0;
    }

    if (adapter->merged) {
        cml_lora_adapter_free(adapter);
        tensor_free(W1);
        tensor_free(W2);
        return 0;
    }

    /* Verify original weights are restored */
    int ok = 1;
    float* w1_ptr = (float*)tensor_data_ptr(W1);
    float* w2_ptr = (float*)tensor_data_ptr(W2);
    for (int i = 0; i < 4; i++) {
        if (fabsf(w1_ptr[i] - orig_W1[i]) > 1e-5f) {
            printf("(W1 restore failed at %d) ", i);
            ok = 0;
        }
    }
    for (int i = 0; i < 6; i++) {
        if (fabsf(w2_ptr[i] - orig_W2[i]) > 1e-5f) {
            printf("(W2 restore failed at %d) ", i);
            ok = 0;
        }
    }

    cml_lora_adapter_free(adapter);
    tensor_free(W1);
    tensor_free(W2);
    return ok;
}


static int test_lora_zero_init_B(void) {
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .has_dtype = true,
        .has_device = true
    };

    /* Create a base weight [4, 3] */
    float W_data[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    };
    int W_shape[2] = {4, 3};
    Tensor* base_weight = tensor_from_data(W_data, W_shape, 2, &cfg);
    if (!base_weight) return 0;

    /* Create LoRA with rank=4, alpha=4.0 */
    CMLLoRALinear* lora = cml_lora_linear_create(base_weight, 4, 4.0f);
    if (!lora) { tensor_free(base_weight); return 0; }

    /* Check that B is all zeros */
    tensor_ensure_executed(lora->lora_B);
    float* B_data = (float*)tensor_data_ptr(lora->lora_B);
    if (!B_data) {
        cml_lora_linear_free(lora);
        tensor_free(base_weight);
        return 0;
    }

    int ok = 1;
    /* B shape is [out_features=4, rank=4] = 16 elements */
    int B_size = lora->out_features * lora->rank;
    for (int i = 0; i < B_size; i++) {
        if (fabsf(B_data[i]) > 1e-10f) {
            printf("(B[%d] = %.6f, expected 0) ", i, B_data[i]);
            ok = 0;
            break;
        }
    }

    /* Also verify that A is NOT all zeros (it should have Xavier init values) */
    tensor_ensure_executed(lora->lora_A);
    float* A_data = (float*)tensor_data_ptr(lora->lora_A);
    if (!A_data) {
        cml_lora_linear_free(lora);
        tensor_free(base_weight);
        return 0;
    }

    int A_size = lora->rank * lora->in_features;
    float sum_abs = 0.0f;
    for (int i = 0; i < A_size; i++) {
        sum_abs += fabsf(A_data[i]);
    }
    if (sum_abs < 1e-10f) {
        printf("(A is all zeros, expected Xavier init) ");
        ok = 0;
    }

    /*
     * Since B is zero, a forward pass should produce output identical to
     * the base linear (no LoRA contribution). Verify this property.
     */
    float x_data[] = {1.0f, 1.0f, 1.0f};
    int x_shape[2] = {1, 3};
    Tensor* input = tensor_from_data(x_data, x_shape, 2, &cfg);
    if (!input) {
        cml_lora_linear_free(lora);
        tensor_free(base_weight);
        return 0;
    }

    Tensor* output = cml_lora_linear_forward(lora, input);
    if (!output) {
        tensor_free(input);
        cml_lora_linear_free(lora);
        tensor_free(base_weight);
        return 0;
    }

    tensor_ensure_executed(output);
    float* out = (float*)tensor_data_ptr(output);

    /* Expected: input @ W^T
     * [1,1,1] @ W^T where W = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
     * out[0] = 1+2+3 = 6
     * out[1] = 4+5+6 = 15
     * out[2] = 7+8+9 = 24
     * out[3] = 10+11+12 = 33
     */
    if (fabsf(out[0] - 6.0f) > 1e-4f) { printf("(out[0]=%.4f expected 6) ", out[0]); ok = 0; }
    if (fabsf(out[1] - 15.0f) > 1e-4f) { printf("(out[1]=%.4f expected 15) ", out[1]); ok = 0; }
    if (fabsf(out[2] - 24.0f) > 1e-4f) { printf("(out[2]=%.4f expected 24) ", out[2]); ok = 0; }
    if (fabsf(out[3] - 33.0f) > 1e-4f) { printf("(out[3]=%.4f expected 33) ", out[3]); ok = 0; }

    tensor_free(output);
    tensor_free(input);
    cml_lora_linear_free(lora);
    tensor_free(base_weight);
    return ok;
}


int main(void) {
    printf("\n");
    printf("  LoRA (Low-Rank Adaptation) Tests\n");
    printf("\n");

    printf("LoRA Linear:\n");
    TEST(lora_linear_forward);
    TEST(lora_zero_init_B);

    printf("\nMerge / Unmerge:\n");
    TEST(lora_merge_unmerge);

    printf("\nAdapter:\n");
    TEST(lora_adapter);

    printf("\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
