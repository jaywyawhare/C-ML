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
#include "alloc/cml_allocator.h"
#include "test_harness.h"

/* The helpers and tests below use CHECK/TEST_SUMMARY from test_harness.h and the
 * OpenCL backend, so they only exist when OpenCL is compiled in; otherwise main()
 * skips cleanly. Keeping them unguarded tripped -Werror (unused funcs, implicit
 * CHECK) on no-OpenCL CI configs. */

static float max_abs_diff(float* a, float* b, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

/* Run an op on CPU, then on GPU (via BACKEND=opencl), compare results */
static void test_matmul(void) {
    printf("Testing MATMUL on GPU...\n");
    int M = 64, K = 128, N = 64;

    /* CPU reference */
    TensorConfig cfg = {0};
    float* a_data = cml_malloc(M * K * sizeof(float));
    float* b_data = cml_malloc(K * N * sizeof(float));
    for (int i = 0; i < M * K; i++) a_data[i] = (float)(i % 7) * 0.1f;
    for (int i = 0; i < K * N; i++) b_data[i] = (float)(i % 5) * 0.1f;

    int shape_a[] = {M, K};
    int shape_b[] = {K, N};
    Tensor* ta_cpu = tensor_from_data(a_data, shape_a, 2, &cfg);
    Tensor* tb_cpu = tensor_from_data(b_data, shape_b, 2, &cfg);
    Tensor* tc_cpu = uop_matmul(ta_cpu, tb_cpu);
    float* cpu_result = (float*)tensor_data_ptr(tc_cpu);

    float* cpu_copy = cml_malloc(M * N * sizeof(float));
    memcpy(cpu_copy, cpu_result, M * N * sizeof(float));

    tensor_free(ta_cpu);
    tensor_free(tb_cpu);
    tensor_free(tc_cpu);
    cml_reset_ir_context();

    /* GPU via OpenCL */
    setenv("BACKEND", "opencl", 1);
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
    unsetenv("BACKEND");

    cml_free(a_data);
    cml_free(b_data);
    cml_free(cpu_copy);
}

static void test_batched_matmul(void) {
    printf("Testing batched MATMUL on GPU...\n");
    int B = 3, M = 64, K = 128, N = 64;

    TensorConfig cfg = {0};
    float* a_data = cml_malloc((size_t)B * M * K * sizeof(float));
    float* b_data = cml_malloc((size_t)B * K * N * sizeof(float));
    for (int i = 0; i < B * M * K; i++) a_data[i] = (float)(i % 7) * 0.1f;
    for (int i = 0; i < B * K * N; i++) b_data[i] = (float)(i % 5) * 0.1f;

    /* Hand-computed reference: independent GEMM per batch. */
    float* ref = cml_malloc((size_t)B * M * N * sizeof(float));
    for (int bi = 0; bi < B; bi++)
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++)
                    acc += a_data[(bi * M + m) * K + k] * b_data[(bi * K + k) * N + n];
                ref[(bi * M + m) * N + n] = acc;
            }

    int shape_a[] = {B, M, K};
    int shape_b[] = {B, K, N};
    setenv("BACKEND", "opencl", 1);
    Tensor* ta = tensor_from_data(a_data, shape_a, 3, &cfg);
    Tensor* tb = tensor_from_data(b_data, shape_b, 3, &cfg);
    Tensor* tc = uop_matmul(ta, tb);
    float* gpu_result = (float*)tensor_data_ptr(tc);

    float diff = max_abs_diff(ref, gpu_result, B * M * N);
    CHECK("batched MATMUL [3,64,128]x[3,128,64] correctness", diff < 1e-2f);
    printf("    max abs diff: %e\n", diff);

    tensor_free(ta);
    tensor_free(tb);
    tensor_free(tc);
    cml_reset_ir_context();
    unsetenv("BACKEND");

    cml_free(a_data);
    cml_free(b_data);
    cml_free(ref);
}

/* Validate each GPU-supported unary op numerically against the CPU path on the
 * real device (extends coverage beyond RELU). sqrt/log get positive inputs. */
static void test_gpu_unary_validation(void) {
    printf("Testing GPU unary ops vs CPU...\n");
    int n = 1024, shape[] = {n};
    TensorConfig cfg = {0};
    float* pos = cml_malloc(n * sizeof(float));
    float* mix = cml_malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) { pos[i] = (float)(i + 1) * 0.01f; mix[i] = (float)(i - 512) * 0.02f; }

    struct { const char* name; Tensor* (*fn)(Tensor*); int pos; } ops[] = {
        {"exp",     uop_exp,     0}, {"log",  uop_log,  1}, {"sqrt", uop_sqrt, 1},
        {"sigmoid", uop_sigmoid, 0}, {"tanh", uop_tanh, 0}, {"neg",  uop_neg,  0},
    };
    for (size_t k = 0; k < sizeof(ops) / sizeof(ops[0]); k++) {
        float* in = ops[k].pos ? pos : mix;
        Tensor* t = tensor_from_data(in, shape, 1, &cfg);
        Tensor* r = ops[k].fn(t);
        float* cpu = cml_malloc(n * sizeof(float));
        memcpy(cpu, tensor_data_ptr(r), n * sizeof(float));
        tensor_free(t); tensor_free(r); cml_reset_ir_context();

        setenv("BACKEND", "opencl", 1);
        t = tensor_from_data(in, shape, 1, &cfg);
        r = ops[k].fn(t);
        float diff = max_abs_diff(cpu, (float*)tensor_data_ptr(r), n);
        char lbl[64]; snprintf(lbl, sizeof(lbl), "GPU %s vs CPU", ops[k].name);
        CHECK(lbl, diff < 1e-3f);
        tensor_free(t); tensor_free(r); cml_reset_ir_context();
        unsetenv("BACKEND");
        cml_free(cpu);
    }
    cml_free(pos); cml_free(mix);
}

static void test_gpu_reduction_validation(void) {
    printf("Testing GPU reductions vs CPU...\n");
    int n = 4096, shape[] = {n};
    TensorConfig cfg = {0};
    float* data = cml_malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) data[i] = (float)((i % 97) - 48) * 0.1f;

    struct { const char* name; Tensor* (*fn)(Tensor*, ReduceParams*); } ops[] = {
        {"sum", uop_sum}, {"mean", uop_mean}, {"max_reduce", uop_max_reduce},
    };
    for (size_t k = 0; k < sizeof(ops) / sizeof(ops[0]); k++) {
        Tensor* t = tensor_from_data(data, shape, 1, &cfg);
        Tensor* r = ops[k].fn(t, NULL);
        float cpu = ((float*)tensor_data_ptr(r))[0];
        tensor_free(t); tensor_free(r); cml_reset_ir_context();

        setenv("BACKEND", "opencl", 1);
        t = tensor_from_data(data, shape, 1, &cfg);
        r = ops[k].fn(t, NULL);
        float gpu = ((float*)tensor_data_ptr(r))[0];
        char lbl[64]; snprintf(lbl, sizeof(lbl), "GPU %s vs CPU", ops[k].name);
        CHECK(lbl, fabsf(cpu - gpu) < 1e-2f * (1.0f + fabsf(cpu)));
        tensor_free(t); tensor_free(r); cml_reset_ir_context();
        unsetenv("BACKEND");
    }
    cml_free(data);
}

static void test_elementwise(void) {
    printf("Testing elementwise ops on GPU...\n");
    int n = 1024;
    float* data = cml_malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) data[i] = (float)(i - 512) * 0.01f;

    int shape[] = {n};
    TensorConfig cfg = {0};

    /* Test RELU */
    {
        /* CPU */
        Tensor* t = tensor_from_data(data, shape, 1, &cfg);
        Tensor* r = uop_relu(t);
        float* cpu_res = (float*)tensor_data_ptr(r);
        float* cpu_copy = cml_malloc(n * sizeof(float));
        memcpy(cpu_copy, cpu_res, n * sizeof(float));
        tensor_free(t); tensor_free(r);
        cml_reset_ir_context();

        /* GPU */
        setenv("BACKEND", "opencl", 1);
        t = tensor_from_data(data, shape, 1, &cfg);
        r = uop_relu(t);
        float* gpu_res = (float*)tensor_data_ptr(r);
        float diff = max_abs_diff(cpu_copy, gpu_res, n);
        CHECK("RELU correctness", diff < 1e-5f);
        tensor_free(t); tensor_free(r);
        cml_reset_ir_context();
        unsetenv("BACKEND");
        cml_free(cpu_copy);
    }

    /* Test ADD with broadcast */
    {
        float* data2 = cml_malloc(sizeof(float));
        data2[0] = 3.14f;
        int shape2[] = {1};

        /* CPU */
        Tensor* ta = tensor_from_data(data, shape, 1, &cfg);
        Tensor* tb = tensor_from_data(data2, shape2, 1, &cfg);
        Tensor* r = uop_add(ta, tb);
        float* cpu_res = (float*)tensor_data_ptr(r);
        float* cpu_copy = cml_malloc(n * sizeof(float));
        memcpy(cpu_copy, cpu_res, n * sizeof(float));
        tensor_free(ta); tensor_free(tb); tensor_free(r);
        cml_reset_ir_context();

        /* GPU */
        setenv("BACKEND", "opencl", 1);
        ta = tensor_from_data(data, shape, 1, &cfg);
        tb = tensor_from_data(data2, shape2, 1, &cfg);
        r = uop_add(ta, tb);
        float* gpu_res = (float*)tensor_data_ptr(r);
        float diff = max_abs_diff(cpu_copy, gpu_res, n);
        CHECK("ADD broadcast correctness", diff < 1e-5f);
        tensor_free(ta); tensor_free(tb); tensor_free(r);
        cml_reset_ir_context();
        unsetenv("BACKEND");
        cml_free(cpu_copy);
        cml_free(data2);
    }

    cml_free(data);
}

static void test_large_matmul_perf(void) {
    printf("Testing large MATMUL performance...\n");
    int M = 512, K = 512, N = 512;
    float* a_data = cml_malloc(M * K * sizeof(float));
    float* b_data = cml_malloc(K * N * sizeof(float));
    for (int i = 0; i < M * K; i++) a_data[i] = (float)(i % 11) * 0.01f;
    for (int i = 0; i < K * N; i++) b_data[i] = (float)(i % 13) * 0.01f;

    int shape_a[] = {M, K};
    int shape_b[] = {K, N};
    TensorConfig cfg = {0};

    /* Warmup + time GPU */
    setenv("BACKEND", "opencl", 1);
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
    unsetenv("BACKEND");

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

    cml_free(a_data);
    cml_free(b_data);
}

#endif /* CML_HAS_OPENCL */

int main(void) {
#ifdef CML_HAS_OPENCL
    if (!cml_opencl_ir_available()) {
        printf("No OpenCL GPU found — skipping tests\n");
        return 0;
    }
    printf("OpenCL GPU detected\n\n");

    test_matmul();
    test_batched_matmul();
    test_gpu_unary_validation();
    test_gpu_reduction_validation();
    test_elementwise();
    test_large_matmul_perf();

    return TEST_SUMMARY();
#else
    printf("OpenCL not compiled — skipping\n");
    return 0;
#endif
}
