/* GPU (Vulkan) execution path end-to-end test.
 *
 * Sets USE_VULKAN=1 *before* cml_init so the execution path routes supported
 * float32 nodes (elementwise + 2D matmul) to the Vulkan compute backend, then
 * verifies the results match a direct CPU computation.  When no Vulkan device is
 * present the path falls back to CPU, so this test also passes on CPU-only hosts. */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cml.h"
#include "ops/ir/gpu/vulkan_backend.h"
#include "test_harness.h"

static Tensor* mk(int r, int c, const float* vals) {
    Tensor* t = cml_zeros_2d(r, c);
    float* d  = (float*)tensor_data_ptr(t);
    for (int i = 0; i < r * c; i++)
        d[i] = vals[i];
    return t;
}

static void check(const char* name, Tensor* out, const float* expect, int n) {
    tensor_ensure_executed(out);
    float* d = (float*)tensor_data_ptr(out);
    int ok   = 1;
    for (int i = 0; i < n; i++)
        if (fabsf(d[i] - expect[i]) > 1e-4f) {
            if (ok)
                printf("  FAIL [%s] @%d: got %.5f want %.5f\n", name, i, d[i], expect[i]);
            ok = 0;
        }
    tests_run++;
    if (ok) {
        tests_passed++;
        printf("  %-20s PASS\n", name);
    } else {
        printf("  %-20s FAIL\n", name);
    }
}

int main(void) {
    setenv("USE_VULKAN", "1", 1); /* must precede any execution */
    cml_init();

    int vk = cml_vulkan_available();
    printf("GPU exec path test (Vulkan device: %s)\n\n", vk ? "yes" : "no (CPU fallback)");

    float a4[4] = {1, 2, 3, 4};
    float b4[4] = {10, 20, 30, 40};

    {
        float e[4] = {11, 22, 33, 44};
        check("add", cml_add(mk(2, 2, a4), mk(2, 2, b4)), e, 4);
    }
    {
        float e[4] = {10, 40, 90, 160};
        check("mul", cml_mul(mk(2, 2, a4), mk(2, 2, b4)), e, 4);
    }
    {
        float e[4] = {-9, -18, -27, -36};
        check("sub", cml_sub(mk(2, 2, a4), mk(2, 2, b4)), e, 4);
    }
    {
        float rin[4] = {-1, 2, -3, 4};
        float e[4]   = {0, 2, 0, 4};
        check("relu", cml_relu(mk(2, 2, rin)), e, 4);
    }

    /* matmul: [2x3] @ [3x2] = [2x2] */
    float A[6] = {1, 2, 3, 4, 5, 6};
    float B[6] = {7, 8, 9, 10, 11, 12};
    {
        float e[4] = {58, 64, 139, 154};
        check("matmul", cml_matmul(mk(2, 3, A), mk(3, 2, B)), e, 4);
    }

    /* chained ops -> multiple GPU dispatches in one graph */
    {
        float x[4]  = {58, 64, 139, 154};
        float y[4]  = {2, -100, 1, -200};
        float e2[4] = {60, 0, 140, 0};
        check("matmul_add_relu", cml_relu(cml_add(mk(2, 2, x), mk(2, 2, y))), e2, 4);
    }

    return TEST_SUMMARY();
}
