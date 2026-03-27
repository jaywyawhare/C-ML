/*
 * Test OpenCL IR backend — verifies GPU execution produces correct results.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "ops/ir/ir.h"
#include "ops/ir/execution.h"

#ifdef CML_HAS_OPENCL
#include "ops/ir/gpu/opencl_ir_backend.h"
#endif

static int tests_passed = 0;
static int tests_total  = 0;

#define CHECK(name, cond) do { \
    tests_total++; \
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); } \
    else { printf("  FAIL: %s\n", name); } \
} while(0)

static float max_abs_diff(float* a, float* b, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

/* Run an op on CPU, then on GPU (via CML_BACKEND=opencl), compare results */
static void test_matmul(void) {
    printf("Testing MATMUL on GPU...\n");
    int M = 64, K = 128, N = 64;

    /* CPU reference */
    TensorConfig cfg = {0};
    float* a_data = malloc(M * K * sizeof(float));
    float* b_data = malloc(K * N * sizeof(float));
    for (int i = 0; i < M * K; i++) a_data[i] = (float)(i % 7) * 0.1f;
    for (int i = 0; i < K * N; i++) b_data[i] = (float)(i % 5) * 0.1f;

    int shape_a[] = {M, K};
    int shape_b[] = {K, N};
    Tensor* ta_cpu = tensor_from_data(a_data, shape_a, 2, &cfg);
    Tensor* tb_cpu = tensor_from_data(b_data, shape_b, 2, &cfg);
    Tensor* tc_cpu = uop_matmul(ta_cpu, tb_cpu);
    float* cpu_result = (float*)tensor_data_ptr(tc_cpu);

    float* cpu_copy = malloc(M * N * sizeof(float));
    memcpy(cpu_copy, cpu_result, M * N * sizeof(float));

    tensor_free(ta_cpu);
    tensor_free(tb_cpu);
    tensor_free(tc_cpu);
    cml_reset_ir_context();

    /* GPU via OpenCL */
    setenv("CML_BACKEND", "opencl", 1);
    Tensor* ta_gpu = tensor_from_data(a_data, shape_a, 2, &cfg);
    Tensor* tb_gpu = tensor_from_data(b_data, shape_b, 2, &cfg);
    Tensor* tc_gpu = uop_matmul(ta_gpu, tb_gpu);
    float* gpu_result = (float*)tensor_data_ptr(tc_gpu);

    float diff = max_abs_diff(cpu_copy, gpu_result, M * N);
    CHECK("MATMUL 64x128 * 128x64 correctness", diff < 1e-2f);
    printf("    max abs diff: %e\n", diff);

    tensor_free(ta_gpu);
    tensor_free(tb_gpu);
    tensor_free(tc_gpu);
    cml_reset_ir_context();
    unsetenv("CML_BACKEND");

    free(a_data);
    free(b_data);
    free(cpu_copy);
}

static void test_elementwise(void) {
    printf("Testing elementwise ops on GPU...\n");
    int n = 1024;
    float* data = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) data[i] = (float)(i - 512) * 0.01f;

    int shape[] = {n};
    TensorConfig cfg = {0};

    /* Test RELU */
    {
        /* CPU */
        Tensor* t = tensor_from_data(data, shape, 1, &cfg);
        Tensor* r = uop_relu(t);
        float* cpu_res = (float*)tensor_data_ptr(r);
        float* cpu_copy = malloc(n * sizeof(float));
        memcpy(cpu_copy, cpu_res, n * sizeof(float));
        tensor_free(t); tensor_free(r);
        cml_reset_ir_context();

        /* GPU */
        setenv("CML_BACKEND", "opencl", 1);
        t = tensor_from_data(data, shape, 1, &cfg);
        r = uop_relu(t);
        float* gpu_res = (float*)tensor_data_ptr(r);
        float diff = max_abs_diff(cpu_copy, gpu_res, n);
        CHECK("RELU correctness", diff < 1e-5f);
        tensor_free(t); tensor_free(r);
        cml_reset_ir_context();
        unsetenv("CML_BACKEND");
        free(cpu_copy);
    }

    /* Test ADD with broadcast */
    {
        float* data2 = malloc(sizeof(float));
        data2[0] = 3.14f;
        int shape2[] = {1};

        /* CPU */
        Tensor* ta = tensor_from_data(data, shape, 1, &cfg);
        Tensor* tb = tensor_from_data(data2, shape2, 1, &cfg);
        Tensor* r = uop_add(ta, tb);
        float* cpu_res = (float*)tensor_data_ptr(r);
        float* cpu_copy = malloc(n * sizeof(float));
        memcpy(cpu_copy, cpu_res, n * sizeof(float));
        tensor_free(ta); tensor_free(tb); tensor_free(r);
        cml_reset_ir_context();

        /* GPU */
        setenv("CML_BACKEND", "opencl", 1);
        ta = tensor_from_data(data, shape, 1, &cfg);
        tb = tensor_from_data(data2, shape2, 1, &cfg);
        r = uop_add(ta, tb);
        float* gpu_res = (float*)tensor_data_ptr(r);
        float diff = max_abs_diff(cpu_copy, gpu_res, n);
        CHECK("ADD broadcast correctness", diff < 1e-5f);
        tensor_free(ta); tensor_free(tb); tensor_free(r);
        cml_reset_ir_context();
        unsetenv("CML_BACKEND");
        free(cpu_copy);
        free(data2);
    }

    free(data);
}

static void test_large_matmul_perf(void) {
    printf("Testing large MATMUL performance...\n");
    int M = 512, K = 512, N = 512;
    float* a_data = malloc(M * K * sizeof(float));
    float* b_data = malloc(K * N * sizeof(float));
    for (int i = 0; i < M * K; i++) a_data[i] = (float)(i % 11) * 0.01f;
    for (int i = 0; i < K * N; i++) b_data[i] = (float)(i % 13) * 0.01f;

    int shape_a[] = {M, K};
    int shape_b[] = {K, N};
    TensorConfig cfg = {0};

    /* Warmup + time GPU */
    setenv("CML_BACKEND", "opencl", 1);
    for (int warmup = 0; warmup < 2; warmup++) {
        Tensor* ta = tensor_from_data(a_data, shape_a, 2, &cfg);
        Tensor* tb = tensor_from_data(b_data, shape_b, 2, &cfg);
        Tensor* tc = uop_matmul(ta, tb);
        tensor_data_ptr(tc);
        tensor_free(ta); tensor_free(tb); tensor_free(tc);
        cml_reset_ir_context();
    }

    struct timespec t0, t1;
    int iters = 5;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) {
        Tensor* ta = tensor_from_data(a_data, shape_a, 2, &cfg);
        Tensor* tb = tensor_from_data(b_data, shape_b, 2, &cfg);
        Tensor* tc = uop_matmul(ta, tb);
        tensor_data_ptr(tc);
        tensor_free(ta); tensor_free(tb); tensor_free(tc);
        cml_reset_ir_context();
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gpu_ms = ((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6) / iters;
    unsetenv("CML_BACKEND");

    /* Time CPU */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) {
        Tensor* ta = tensor_from_data(a_data, shape_a, 2, &cfg);
        Tensor* tb = tensor_from_data(b_data, shape_b, 2, &cfg);
        Tensor* tc = uop_matmul(ta, tb);
        tensor_data_ptr(tc);
        tensor_free(ta); tensor_free(tb); tensor_free(tc);
        cml_reset_ir_context();
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_ms = ((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6) / iters;

    printf("    GEMM 512x512: GPU %.2fms, CPU %.2fms (%.2fx)\n", gpu_ms, cpu_ms, cpu_ms / gpu_ms);
    CHECK("MATMUL 512x512 GPU runs", gpu_ms > 0);

    free(a_data);
    free(b_data);
}

int main(void) {
#ifdef CML_HAS_OPENCL
    if (!cml_opencl_ir_available()) {
        printf("No OpenCL GPU found — skipping tests\n");
        return 0;
    }
    printf("OpenCL GPU detected\n\n");

    test_matmul();
    test_elementwise();
    test_large_matmul_perf();

    printf("\n%d/%d tests passed\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
#else
    printf("OpenCL not compiled — skipping\n");
    return 0;
#endif
}
