#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "cml.h"
#include "ops/ir/dispatch.h"
#include "ops/ir/gpu/cuda_backend.h"
#include "ops/ir/gpu/rocm_backend.h"
#include "core/logging.h"

#ifdef CML_HAS_LLVM_BACKEND
#include "ops/ir/gpu/gpu_codegen.h"
#endif

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  Testing: %s... ", #name); \
    tests_run++; \
    if (test_##name()) { \
        printf("PASS\n"); \
        tests_passed++; \
    } else { \
        printf("FAIL\n"); \
    } \
} while(0)

#define EXPECT_NEAR(a, b, eps) (fabsf((a) - (b)) < (eps))

static bool has_gpu(void) {
    return cml_cuda_available() || cml_rocm_available();
}

static int test_gpu_codegen_lifecycle(void) {
#ifndef CML_HAS_LLVM_BACKEND
    printf("(skipped - no LLVM backend) ");
    return 1;
#else
    if (!has_gpu()) {
        printf("(skipped - no GPU) ");
        return 1;
    }

    // Use global dispatch context to get the already-initialized backend
    CMLDispatchContext* ctx = cml_dispatch_get_global();
    if (!ctx) return 0;

    CMLGPUCodegen* cg = NULL;
    if (cml_dispatch_backend_available(ctx, CML_BACKEND_CUDA)) {
        CMLCUDABackend* cuda = (CMLCUDABackend*)ctx->backend_contexts[CML_BACKEND_CUDA];
        if (!cuda) { printf("(no CUDA backend ctx) "); return 0; }
        cg = cml_gpu_codegen_create(GPU_TARGET_CUDA, cuda);
        if (!cg) return 0;
        if (!cg->initialized || cg->target != GPU_TARGET_CUDA) {
            cml_gpu_codegen_destroy(cg);
            return 0;
        }
        printf("(CUDA sm=%s) ", cg->target_cpu);
        cml_gpu_codegen_destroy(cg);
    } else if (cml_dispatch_backend_available(ctx, CML_BACKEND_ROCM)) {
        CMLROCmBackend* rocm = (CMLROCmBackend*)ctx->backend_contexts[CML_BACKEND_ROCM];
        if (!rocm) { printf("(no ROCm backend ctx) "); return 0; }
        cg = cml_gpu_codegen_create(GPU_TARGET_ROCM, rocm);
        if (!cg) return 0;
        printf("(ROCm %s) ", cg->target_cpu);
        cml_gpu_codegen_destroy(cg);
    } else {
        printf("(no GPU backend initialized) ");
        return 1;
    }
    return 1;
#endif
}

static int test_dispatch_gpu_detection(void) {
    // Use the global dispatch context to avoid double CUDA init/free
    CMLDispatchContext* ctx = cml_dispatch_get_global();
    if (!ctx) return 0;

    bool cuda_available = cml_dispatch_backend_available(ctx, CML_BACKEND_CUDA);
    bool rocm_available = cml_dispatch_backend_available(ctx, CML_BACKEND_ROCM);

    printf("(CUDA=%s ROCm=%s) ",
           cuda_available ? "yes" : "no",
           rocm_available ? "yes" : "no");

    // If GPU detected, best backend should be GPU
    if (cuda_available || rocm_available) {
        CMLBackendType best = cml_dispatch_get_best_backend(ctx);
        if (best != CML_BACKEND_CUDA && best != CML_BACKEND_ROCM) {
            return 0;
        }
    }

    return 1;
}

static int test_gpu_binary_add(void) {
#ifndef CML_HAS_LLVM_BACKEND
    printf("(skipped - no LLVM backend) ");
    return 1;
#else
    if (!has_gpu()) {
        printf("(skipped - no GPU) ");
        return 1;
    }

    cml_init();

    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    Tensor* a = cml_tensor_1d(a_data, 4);
    Tensor* b = cml_tensor_1d(b_data, 4);

    // GPU add
    Tensor* c = tensor_add(a, b);
    tensor_ensure_executed(c);

    // Verify
    float expected[] = {6.0f, 8.0f, 10.0f, 12.0f};
    bool ok = true;
    for (int i = 0; i < 4; i++) {
        if (!EXPECT_NEAR(((float*)c->data)[i], expected[i], 1e-5f)) {
            printf("(mismatch at %d: got %f, want %f) ", i, ((float*)c->data)[i], expected[i]);
            ok = false;
            break;
        }
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    cml_cleanup();
    return ok;
#endif
}

static int test_gpu_binary_mul(void) {
#ifndef CML_HAS_LLVM_BACKEND
    printf("(skipped - no LLVM backend) ");
    return 1;
#else
    if (!has_gpu()) {
        printf("(skipped - no GPU) ");
        return 1;
    }

    cml_init();

    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {2.0f, 3.0f, 4.0f, 5.0f};
    Tensor* a = cml_tensor_1d(a_data, 4);
    Tensor* b = cml_tensor_1d(b_data, 4);

    Tensor* c = tensor_mul(a, b);
    tensor_ensure_executed(c);

    float expected[] = {2.0f, 6.0f, 12.0f, 20.0f};
    bool ok = true;
    for (int i = 0; i < 4; i++) {
        if (!EXPECT_NEAR(((float*)c->data)[i], expected[i], 1e-5f)) {
            ok = false;
            break;
        }
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    cml_cleanup();
    return ok;
#endif
}

static int test_gpu_unary_exp(void) {
#ifndef CML_HAS_LLVM_BACKEND
    printf("(skipped - no LLVM backend) ");
    return 1;
#else
    if (!has_gpu()) {
        printf("(skipped - no GPU) ");
        return 1;
    }

    cml_init();

    float a_data[] = {0.0f, 1.0f, 2.0f, -1.0f};
    Tensor* a = cml_tensor_1d(a_data, 4);

    Tensor* b = tensor_exp(a);
    tensor_ensure_executed(b);

    bool ok = true;
    for (int i = 0; i < 4; i++) {
        float expected = expf(a_data[i]);
        if (!EXPECT_NEAR(((float*)b->data)[i], expected, 1e-4f)) {
            printf("(mismatch at %d: got %f, want %f) ", i, ((float*)b->data)[i], expected);
            ok = false;
            break;
        }
    }

    tensor_free(a);
    tensor_free(b);
    cml_cleanup();
    return ok;
#endif
}

static int test_gpu_broadcast(void) {
#ifndef CML_HAS_LLVM_BACKEND
    printf("(skipped - no LLVM backend) ");
    return 1;
#else
    if (!has_gpu()) {
        printf("(skipped - no GPU) ");
        return 1;
    }

    cml_init();

    // [2,3] * [1,3] should broadcast
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {10.0f, 20.0f, 30.0f};
    Tensor* a = cml_tensor_2d(a_data, 2, 3);
    Tensor* b = cml_tensor_2d(b_data, 1, 3);

    Tensor* c = tensor_mul(a, b);
    tensor_ensure_executed(c);

    float expected[] = {10.0f, 40.0f, 90.0f, 40.0f, 100.0f, 180.0f};
    bool ok = true;
    for (int i = 0; i < 6; i++) {
        if (!EXPECT_NEAR(((float*)c->data)[i], expected[i], 1e-4f)) {
            printf("(mismatch at %d: got %f, want %f) ", i, ((float*)c->data)[i], expected[i]);
            ok = false;
            break;
        }
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    cml_cleanup();
    return ok;
#endif
}

static int test_gpu_reduction_sum(void) {
#ifndef CML_HAS_LLVM_BACKEND
    printf("(skipped - no LLVM backend) ");
    return 1;
#else
    if (!has_gpu()) {
        printf("(skipped - no GPU) ");
        return 1;
    }

    cml_init();

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor* a = cml_tensor_1d(data, 4);

    Tensor* b = tensor_sum(a, 0, false);
    tensor_ensure_executed(b);

    float expected = 10.0f;
    bool ok = EXPECT_NEAR(((float*)b->data)[0], expected, 1e-3f);
    if (!ok) {
        printf("(got %f, want %f) ", ((float*)b->data)[0], expected);
    }

    tensor_free(a);
    tensor_free(b);
    cml_cleanup();
    return ok;
#endif
}

static int test_gpu_matmul(void) {
#ifndef CML_HAS_LLVM_BACKEND
    printf("(skipped - no LLVM backend) ");
    return 1;
#else
    if (!has_gpu()) {
        printf("(skipped - no GPU) ");
        return 1;
    }

    cml_init();

    // [2,3] @ [3,2]
    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[] = {7, 8, 9, 10, 11, 12};
    Tensor* a = cml_tensor_2d(a_data, 2, 3);
    Tensor* b = cml_tensor_2d(b_data, 3, 2);

    Tensor* c = tensor_matmul(a, b);
    tensor_ensure_executed(c);

    // Expected: [[58, 64], [139, 154]]
    float expected[] = {58.0f, 64.0f, 139.0f, 154.0f};
    bool ok = true;
    for (int i = 0; i < 4; i++) {
        if (!EXPECT_NEAR(((float*)c->data)[i], expected[i], 1e-3f)) {
            printf("(mismatch at %d: got %f, want %f) ", i, ((float*)c->data)[i], expected[i]);
            ok = false;
            break;
        }
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    cml_cleanup();
    return ok;
#endif
}

int main(void) {
    printf("\nGPU Codegen Tests\n");
    printf("GPU available: %s\n\n", has_gpu() ? "YES" : "NO");

    TEST(gpu_codegen_lifecycle);
    TEST(dispatch_gpu_detection);
    TEST(gpu_binary_add);
    TEST(gpu_binary_mul);
    TEST(gpu_unary_exp);
    TEST(gpu_broadcast);
    TEST(gpu_reduction_sum);
    TEST(gpu_matmul);

    printf("\nResults: %d/%d passed\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
