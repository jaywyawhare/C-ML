/**
 * @file bench_gemm.c
 * @brief GEMM (matrix multiply) benchmark reporting FLOPS/TFLOPS.
 *
 * Usage:
 *   ./build/examples/bench_gemm [m] [n] [k] [reps]
 * Defaults:
 *   m=1024, n=1024, k=1024, reps=10
 */

#include "cml.h"
#include "Core/profiling.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    int m = 1024, n = 1024, k = 1024, reps = 10;
    if (argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }
    if (argc >= 5) {
        reps = atoi(argv[4]);
    }

    if (m <= 0 || n <= 0 || k <= 0 || reps <= 0) {
        fprintf(stderr, "Invalid dimensions or reps.\n");
        return 1;
    }

    if (cml_init() != 0) {
        fprintf(stderr, "Failed to initialize C-ML.\n");
        return 1;
    }

    int a_shape[] = {m, k};
    int b_shape[] = {k, n};
    Tensor* A     = tensor_empty(a_shape, 2, DTYPE_FLOAT32, DEVICE_CPU);
    Tensor* B     = tensor_empty(b_shape, 2, DTYPE_FLOAT32, DEVICE_CPU);
    if (!A || !B) {
        fprintf(stderr, "Alloc failure.\n");
        if (A)
            tensor_free(A);
        if (B)
            tensor_free(B);
        cml_cleanup();
        return 1;
    }

    float* a = (float*)tensor_data_ptr(A);
    float* b = (float*)tensor_data_ptr(B);
    for (long i = 0; i < (long)m * (long)k; i++)
        a[i] = (float)((i % 13) - 6) * 0.01f;
    for (long i = 0; i < (long)k * (long)n; i++)
        b[i] = (float)((i % 7) - 3) * 0.02f;

    Tensor* C = tensor_matmul(A, B);
    if (!C) {
        fprintf(stderr, "matmul failed\n");
        tensor_free(A);
        tensor_free(B);
        cml_cleanup();
        return 1;
    }
    tensor_free(C);

    Profiler* prof = profiler_create();
    int tid        = profiler_start(prof, "gemm");
    for (int r = 0; r < reps; r++) {
        Tensor* R = tensor_matmul(A, B);
        tensor_free(R);
    }
    profiler_stop(prof, tid);
    double elapsed_ms = profiler_get_total_time(prof, "gemm");
    double elapsed    = elapsed_ms / 1000.0;

    double flops_total = 2.0 * (double)m * (double)n * (double)k * (double)reps;
    double gflops      = (flops_total / 1e9) / elapsed;
    double tflops      = (flops_total / 1e12) / elapsed;

    printf("GEMM m=%d n=%d k=%d reps=%d\n", m, n, k, reps);
    printf("Elapsed: %.6f s\n", elapsed);
    printf("Throughput: %.3f GFLOPS (%.3f TFLOPS)\n", gflops, tflops);

    tensor_free(B);
    tensor_free(A);
    profiler_free(prof);
    cml_cleanup();
    return 0;
}
