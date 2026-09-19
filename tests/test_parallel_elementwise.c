/* Large elementwise ops must use the threadpool-parallel SIMD path (n above the
 * parallel threshold) and stay bit-correct vs a serial reference. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cml.h"
#include "test_harness.h"

int main(void) {
    cml_init();
    printf("=== Parallel elementwise (large tensors) ===\n");
    int N    = 50000; /* > the 10000 parallel threshold */
    float* a = (float*)malloc((size_t)N * sizeof(float));
    float* b = (float*)malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) {
        a[i] = (float)(i % 13) - 6.0f;
        b[i] = (float)(i % 7) - 3.0f;
    }

    float* r;
    r = (float*)tensor_data_ptr(cml_add(cml_tensor_1d(a, N), cml_tensor_1d(b, N)));
    {
        int ok = 1;
        for (int i = 0; i < N; i++)
            if (fabsf(r[i] - (a[i] + b[i])) > 1e-4f) {
                ok = 0;
                break;
            }
        CHECK("parallel add correct", ok);
    }
    r = (float*)tensor_data_ptr(cml_sub(cml_tensor_1d(a, N), cml_tensor_1d(b, N)));
    {
        int ok = 1;
        for (int i = 0; i < N; i++)
            if (fabsf(r[i] - (a[i] - b[i])) > 1e-4f) {
                ok = 0;
                break;
            }
        CHECK("parallel sub correct", ok);
    }
    r = (float*)tensor_data_ptr(cml_mul(cml_tensor_1d(a, N), cml_tensor_1d(b, N)));
    {
        int ok = 1;
        for (int i = 0; i < N; i++)
            if (fabsf(r[i] - (a[i] * b[i])) > 1e-4f) {
                ok = 0;
                break;
            }
        CHECK("parallel mul correct", ok);
    }

    free(a);
    free(b);
    return TEST_SUMMARY();
}
