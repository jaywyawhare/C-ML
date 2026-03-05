/**
 * @file test_new_features.c
 * @brief Tests for newly added features:
 *   - float16/bfloat16 dtype support
 *   - ConvTranspose2d layer
 *   - RMSNorm layer
 *   - Sparse cross-entropy loss
 *   - Int8 quantization
 *   - OpenCL backend availability check
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "tensor/tensor.h"
#include "nn/layers/rmsnorm.h"
#include "nn/layers/conv_transpose2d.h"
#include "autograd/loss_functions.h"
#include "core/quantization.h"
#include "backend/opencl_backend.h"
#include "ops/uops.h"
#include "nn.h"

static int tests_passed = 0;
static int tests_total  = 0;

static const TensorConfig cpu_f32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

#define TEST(name)                                     \
    do {                                               \
        tests_total++;                                 \
        printf("  TEST: %s ... ", #name);              \
        if (test_##name()) {                           \
            tests_passed++;                            \
            printf("PASSED\n");                        \
        } else {                                       \
            printf("FAILED\n");                        \
        }                                              \
    } while (0)

static int test_dtype_sizes(void) {
    if (cml_dtype_size(DTYPE_FLOAT16) != 2) return 0;
    if (cml_dtype_size(DTYPE_BFLOAT16) != 2) return 0;
    if (cml_dtype_size(DTYPE_INT8) != 1) return 0;
    if (cml_dtype_size(DTYPE_UINT8) != 1) return 0;
    return 1;
}

static int test_float16_tensor(void) {
    int shape[] = {2, 3};
    TensorConfig config = {.dtype = DTYPE_FLOAT16, .device = DEVICE_CPU,
                           .has_dtype = true, .has_device = true};
    Tensor* t = tensor_ones(shape, 2, &config);
    if (!t) return 0;
    // float16 ones should read back as ~1.0
    float v = tensor_get_float(t, 0);
    tensor_free(t);
    return fabsf(v - 1.0f) < 0.01f;
}

static int test_bfloat16_tensor(void) {
    int shape[] = {2, 3};
    TensorConfig config = {.dtype = DTYPE_BFLOAT16, .device = DEVICE_CPU,
                           .has_dtype = true, .has_device = true};
    Tensor* t = tensor_full(shape, 2, &config, 3.14f);
    if (!t) return 0;
    float v = tensor_get_float(t, 0);
    tensor_free(t);
    return fabsf(v - 3.14f) < 0.1f; // bfloat16 has limited precision
}

static int test_int8_tensor(void) {
    int shape[] = {4};
    TensorConfig config = {.dtype = DTYPE_INT8, .device = DEVICE_CPU,
                           .has_dtype = true, .has_device = true};
    Tensor* t = tensor_ones(shape, 1, &config);
    if (!t) return 0;
    float v = tensor_get_float(t, 0);
    tensor_free(t);
    return fabsf(v - 1.0f) < 0.01f;
}

static int test_rmsnorm_create(void) {
    RMSNorm* rn = nn_rmsnorm(64, 1e-5f, DTYPE_FLOAT32, DEVICE_CPU);
    if (!rn) return 0;
    if (rn->normalized_shape != 64) { module_free((Module*)rn); return 0; }
    module_free((Module*)rn);
    return 1;
}

static int test_rmsnorm_forward(void) {
    RMSNorm* rn = nn_rmsnorm(4, 1e-5f, DTYPE_FLOAT32, DEVICE_CPU);
    if (!rn) return 0;

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    int shape[] = {2, 4};
    Tensor* input = tensor_from_data(data, shape, 2, &cpu_f32);
    if (!input) { module_free((Module*)rn); return 0; }

    Tensor* output = module_forward((Module*)rn, input);
    if (!output) { tensor_free(input); module_free((Module*)rn); return 0; }

    tensor_ensure_executed(output);
    int ok = (output->data != NULL && output->shape[0] == 2 && output->shape[1] == 4);

    tensor_free(output);
    tensor_free(input);
    module_free((Module*)rn);
    return ok;
}

static int test_conv_transpose2d_create(void) {
    ConvTranspose2d* ct = nn_conv_transpose2d(16, 8, 3, 2, 1, 0, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!ct) return 0;
    if (ct->in_channels != 16 || ct->out_channels != 8) {
        free(ct);
        return 0;
    }
    free(ct);
    return 1;
}

static int test_conv_transpose2d_forward(void) {
    ConvTranspose2d* ct = nn_conv_transpose2d(1, 1, 3, 2, 1, 1, false, DTYPE_FLOAT32, DEVICE_CPU);
    if (!ct) return 0;

    // Input: [1, 1, 2, 2]
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int shape[] = {1, 1, 2, 2};
    Tensor* input = tensor_from_data(data, shape, 4, &cpu_f32);
    if (!input) { free(ct); return 0; }

    Tensor* output = module_forward((Module*)ct, input);
    if (!output) { tensor_free(input); free(ct); return 0; }

    // Output should be [1, 1, H_out, W_out]
    // H_out = (2-1)*2 - 2*1 + 1*(3-1) + 1 + 1 = 2 - 2 + 2 + 1 + 1 = 4
    int ok = (output->ndim == 4 && output->shape[0] == 1 && output->shape[1] == 1 &&
              output->shape[2] == 4 && output->shape[3] == 4);

    tensor_free(output);
    tensor_free(input);
    free(ct);
    return ok;
}

static int test_sparse_cross_entropy(void) {
    // Logits: [2, 3] (2 samples, 3 classes)
    float logit_data[] = {2.0f, 1.0f, 0.1f,  0.5f, 2.5f, 0.3f};
    int logit_shape[] = {2, 3};
    Tensor* logits = tensor_from_data(logit_data, logit_shape, 2, &cpu_f32);
    if (!logits) return 0;

    // Targets: [2] class indices
    float target_data[] = {0.0f, 1.0f};
    int target_shape[] = {2};
    Tensor* targets = tensor_from_data(target_data, target_shape, 1, &cpu_f32);
    if (!targets) { tensor_free(logits); return 0; }

    Tensor* loss = tensor_sparse_cross_entropy_loss(logits, targets);
    if (!loss) { tensor_free(logits); tensor_free(targets); return 0; }

    tensor_ensure_executed(loss);
    if (!loss->data) { tensor_free(logits); tensor_free(targets); return 0; }

    float loss_val = tensor_get_float(loss, 0);

    // Loss should be positive and finite
    int ok = (loss_val > 0.0f && loss_val < 100.0f && !isnan(loss_val));

    tensor_free(loss);
    tensor_free(logits);
    tensor_free(targets);
    return ok;
}

static int test_quantize_int8(void) {
    float data[] = {-1.0f, 0.0f, 0.5f, 1.0f, -0.5f, 2.0f};
    int shape[] = {2, 3};
    Tensor* t = tensor_from_data(data, shape, 2, &cpu_f32);
    if (!t) return 0;

    QuantParams qp;
    Tensor* q = cml_quantize_int8(t, NULL, &qp);
    if (!q) { tensor_free(t); return 0; }

    if (q->dtype != DTYPE_INT8) { tensor_free(q); tensor_free(t); return 0; }

    // Dequantize and check round-trip error
    Tensor* dq = cml_dequantize_int8(q, &qp);
    if (!dq) { tensor_free(q); tensor_free(t); return 0; }

    int ok = 1;
    for (size_t i = 0; i < t->numel; i++) {
        float orig = tensor_get_float(t, i);
        float recovered = tensor_get_float(dq, i);
        if (fabsf(orig - recovered) > 0.1f) { // Allow quantization error
            ok = 0;
            break;
        }
    }

    tensor_free(dq);
    tensor_free(q);
    tensor_free(t);
    return ok;
}

static int test_quantize_uint8(void) {
    float data[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    int shape[] = {5};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    if (!t) return 0;

    QuantParams qp;
    Tensor* q = cml_quantize_uint8(t, NULL, &qp);
    if (!q) { tensor_free(t); return 0; }

    if (q->dtype != DTYPE_UINT8) { tensor_free(q); tensor_free(t); return 0; }

    Tensor* dq = cml_dequantize_uint8(q, &qp);
    if (!dq) { tensor_free(q); tensor_free(t); return 0; }

    int ok = 1;
    for (size_t i = 0; i < t->numel; i++) {
        float orig = tensor_get_float(t, i);
        float recovered = tensor_get_float(dq, i);
        if (fabsf(orig - recovered) > 0.05f) {
            ok = 0;
            break;
        }
    }

    tensor_free(dq);
    tensor_free(q);
    tensor_free(t);
    return ok;
}

static int test_opencl_availability_check(void) {
    // Just verify the function doesn't crash
    bool avail = opencl_backend_is_available();
    (void)avail; // May or may not be available
    return 1;
}

int main(void) {
    printf("=== New Features Tests ===\n\n");

    printf("[float16/bfloat16 dtype]\n");
    TEST(dtype_sizes);
    TEST(float16_tensor);
    TEST(bfloat16_tensor);
    TEST(int8_tensor);

    printf("\n[RMSNorm]\n");
    TEST(rmsnorm_create);
    TEST(rmsnorm_forward);

    printf("\n[ConvTranspose2d]\n");
    TEST(conv_transpose2d_create);
    TEST(conv_transpose2d_forward);

    printf("\n[Sparse Cross Entropy]\n");
    TEST(sparse_cross_entropy);

    printf("\n[Int8 Quantization]\n");
    TEST(quantize_int8);
    TEST(quantize_uint8);

    printf("\n[OpenCL Backend]\n");
    TEST(opencl_availability_check);

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_total);

    return (tests_passed == tests_total) ? 0 : 1;
}
