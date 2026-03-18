#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "cml.h"
#include "ops/ir/dispatch.h"
#include "ops/ir/gpu/cuda_backend.h"
#include "ops/ir/gpu/rocm_backend.h"
#include "backend/blas.h"
#include "backend/device.h"
#include "core/logging.h"

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


static int test_cuda_detection(void) {
    bool available = cml_cuda_available();
    printf("(available=%s) ", available ? "yes" : "no");

    // Test passes regardless of availability - we just verify the function works
    return 1;
}


static int test_cuda_lifecycle(void) {
    if (!cml_cuda_available()) {
        printf("(skipped - CUDA not available) ");
        return 1;
    }

    CMLCUDABackend* backend = cml_cuda_backend_create();
    if (!backend) return 0;

    int result = cml_cuda_backend_init(backend, 0);  // device 0
    printf("(init=%s) ", result == 0 ? "ok" : "failed");

    cml_cuda_backend_free(backend);
    return 1;
}


static int test_rocm_detection(void) {
    bool available = cml_rocm_available();
    printf("(available=%s) ", available ? "yes" : "no");
    return 1;
}


static int test_rocm_lifecycle(void) {
    if (!cml_rocm_available()) {
        printf("(skipped - ROCm not available) ");
        return 1;
    }

    CMLROCmBackend* backend = cml_rocm_backend_create();
    if (!backend) return 0;

    int result = cml_rocm_backend_init(backend, 0);  // device 0
    printf("(init=%s) ", result == 0 ? "ok" : "failed");

    cml_rocm_backend_free(backend);
    return 1;
}


static int test_blas_detection(void) {
    bool available = cml_blas_available();
    printf("(available=%s) ", available ? "yes" : "no");
    return 1;
}


static int test_blas_lifecycle(void) {
    CMLBlasContext* ctx = cml_blas_init();

    if (ctx) {
        printf("(library=%s) ", cml_blas_get_library_name(ctx));
        cml_blas_free(ctx);
    } else {
        printf("(no BLAS library found) ");
    }

    return 1;
}


static int test_blas_sgemm(void) {
    CMLBlasContext* ctx = cml_blas_init();
    if (!ctx) {
        printf("(skipped - BLAS not available) ");
        return 1;
    }

    // Test 2x2 matrix multiplication
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f};  // 2x2
    float B[] = {5.0f, 6.0f, 7.0f, 8.0f};  // 2x2
    float C[] = {0.0f, 0.0f, 0.0f, 0.0f};  // 2x2

    // C = 1.0 * A @ B + 0.0 * C
    int result = cml_blas_sgemm(ctx, A, B, C, 2, 2, 2, 1.0f, 0.0f);

    if (result != 0) {
        cml_blas_free(ctx);
        return 0;
    }

    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //         = [[19, 22], [43, 50]]
    float expected[] = {19.0f, 22.0f, 43.0f, 50.0f};
    int success = 1;
    for (int i = 0; i < 4; i++) {
        if (C[i] != expected[i]) {
            printf("(C[%d]=%.1f expected %.1f) ", i, C[i], expected[i]);
            success = 0;
        }
    }

    if (success) printf("(correct) ");

    cml_blas_free(ctx);
    return success;
}


static int test_blas_vector_ops(void) {
    CMLBlasContext* ctx = cml_blas_init();
    if (!ctx) {
        printf("(skipped - BLAS not available) ");
        return 1;
    }

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[] = {5.0f, 6.0f, 7.0f, 8.0f};

    // Test dot product
    float dot = cml_blas_sdot(ctx, x, y, 4);
    // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    if (dot != 70.0f) {
        printf("(dot=%.1f expected 70.0) ", dot);
        cml_blas_free(ctx);
        return 0;
    }

    // Test norm
    float norm = cml_blas_snrm2(ctx, x, 4);
    // Expected: sqrt(1 + 4 + 9 + 16) = sqrt(30) ~ 5.477
    if (norm < 5.4f || norm > 5.5f) {
        printf("(norm=%.3f expected ~5.477) ", norm);
        cml_blas_free(ctx);
        return 0;
    }

    printf("(dot=70.0, norm=%.3f) ", norm);
    cml_blas_free(ctx);
    return 1;
}


static int test_device_detection(void) {
    printf("(cuda=%s, rocm=%s) ",
           device_cuda_available() ? "yes" : "no",
           device_rocm_available() ? "yes" : "no");
    return 1;
}


static int test_all_backends_summary(void) {
    CMLDispatchContext* ctx = cml_dispatch_create();
    if (!ctx) return 0;

    int num = cml_dispatch_detect_backends(ctx);
    printf("(%d backends detected) ", num);

    // Print which ones
    printf("\n    Available: ");
    for (int i = 0; i < CML_BACKEND_COUNT; i++) {
        CMLBackendType bt = (CMLBackendType)i;
        if (cml_dispatch_backend_available(ctx, bt)) {
            printf("%s ", cml_dispatch_backend_name(bt));
        }
    }
    printf("\n    ");

    cml_dispatch_free(ctx);
    return (num >= 1);  // At least CPU fallback
}


int main(void) {
    printf("\n=== Backend Detection Unit Tests ===\n\n");

    printf("GPU Backends:\n");
    TEST(cuda_detection);
    TEST(cuda_lifecycle);
    TEST(rocm_detection);
    TEST(rocm_lifecycle);

    printf("\nBLAS:\n");
    TEST(blas_detection);
    TEST(blas_lifecycle);
    TEST(blas_sgemm);
    TEST(blas_vector_ops);

    printf("\nDevice Layer:\n");
    TEST(device_detection);

    printf("\nIntegration:\n");
    TEST(all_backends_summary);

    printf("\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
