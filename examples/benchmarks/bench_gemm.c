/**
 * @file bench_gemm.c
 * @brief GEMM throughput benchmark: Naive vs raw BLAS vs CML Tensor vs fused chain
 *
 * Compares GFLOPS for square matrix multiplication at various sizes.
 * Shows where raw BLAS wins (single op) and where CML's fusion
 * pipeline pays off (chained ops: matmul + add + relu).
 *
 * Usage:  ./build/bin/bench_gemm
 */

#include "cml.h"
#include "backend/blas.h"
#include "ops/simd_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double get_time_sec(void) {
#if defined(__linux__) || defined(__APPLE__)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

static void naive_sgemm(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

static float* alloc_random(int n) {
    float* p = malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++)
        p[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    return p;
}

static double gemm_flops(int N) { return 2.0 * N * N * N; }

// matmul + bias_add + relu: 2N^3 + 2N^2
static double fused_flops(int N) { return 2.0 * N * N * N + 2.0 * N * N; }

typedef struct {
    double gflops;
    double time_ms;
} BenchResult;

static void print_row(const char* label, BenchResult r) {
    printf("  %-30s %9.3f ms  %8.2f GFLOPS\n", label, r.time_ms, r.gflops);
}

static BenchResult bench_naive(int N, int iters) {
    int n2   = N * N;
    float* A = alloc_random(n2);
    float* B = alloc_random(n2);
    float* C = calloc((size_t)n2, sizeof(float));

    naive_sgemm(A, B, C, N);

    double start = get_time_sec();
    for (int i = 0; i < iters; i++)
        naive_sgemm(A, B, C, N);
    double elapsed = get_time_sec() - start;

    free(A);
    free(B);
    free(C);
    double flop = gemm_flops(N) * iters;
    return (BenchResult){.gflops = flop / elapsed / 1e9, .time_ms = elapsed / iters * 1e3};
}

static BenchResult bench_blas(CMLBlasContext* blas, int N, int iters) {
    int n2   = N * N;
    float* A = alloc_random(n2);
    float* B = alloc_random(n2);
    float* C = calloc((size_t)n2, sizeof(float));

    cml_blas_sgemm(blas, A, B, C, N, N, N, 1.0f, 0.0f);

    double start = get_time_sec();
    for (int i = 0; i < iters; i++)
        cml_blas_sgemm(blas, A, B, C, N, N, N, 1.0f, 0.0f);
    double elapsed = get_time_sec() - start;

    free(A);
    free(B);
    free(C);
    double flop = gemm_flops(N) * iters;
    return (BenchResult){.gflops = flop / elapsed / 1e9, .time_ms = elapsed / iters * 1e3};
}

static BenchResult bench_tensor_matmul(int N, int iters) {
    int shape[]      = {N, N};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* A = tensor_ones(shape, 2, &cfg);
    Tensor* B = tensor_ones(shape, 2, &cfg);

    float* ad = (float*)tensor_data_ptr(A);
    float* bd = (float*)tensor_data_ptr(B);
    for (int i = 0; i < N * N; i++) {
        ad[i] = (float)rand() / (float)RAND_MAX - 0.5f;
        bd[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    }

    Tensor* w = tensor_matmul(A, B);
    (void)tensor_data_ptr(w);

    double start = get_time_sec();
    for (int i = 0; i < iters; i++) {
        Tensor* C = tensor_matmul(A, B);
        (void)tensor_data_ptr(C);
    }
    double elapsed = get_time_sec() - start;

    tensor_free(A);
    tensor_free(B);
    double flop = gemm_flops(N) * iters;
    return (BenchResult){.gflops = flop / elapsed / 1e9, .time_ms = elapsed / iters * 1e3};
}

// BLAS + separate bias+relu pass (no fusion, extra memory traffic)
static BenchResult bench_blas_unfused(CMLBlasContext* blas, int N, int iters) {
    int n2      = N * N;
    float* A    = alloc_random(n2);
    float* B    = alloc_random(n2);
    float* C    = calloc((size_t)n2, sizeof(float));
    float* bias = alloc_random(N);

    cml_blas_sgemm(blas, A, B, C, N, N, N, 1.0f, 0.0f);

    double start = get_time_sec();
    for (int it = 0; it < iters; it++) {
        cml_blas_sgemm(blas, A, B, C, N, N, N, 1.0f, 0.0f);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                float v      = C[i * N + j] + bias[j];
                C[i * N + j] = v > 0.0f ? v : 0.0f;
            }
    }
    double elapsed = get_time_sec() - start;

    free(A);
    free(B);
    free(C);
    free(bias);
    double flop = fused_flops(N) * iters;
    return (BenchResult){.gflops = flop / elapsed / 1e9, .time_ms = elapsed / iters * 1e3};
}

// CML fused: tensor_relu(tensor_add(tensor_matmul(A,B), bias))
// The IR optimizer can fuse the add+relu into the matmul epilogue
static BenchResult bench_tensor_fused(int N, int iters) {
    int mat_shape[]  = {N, N};
    int bias_shape[] = {1, N};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

    Tensor* A    = tensor_ones(mat_shape, 2, &cfg);
    Tensor* B    = tensor_ones(mat_shape, 2, &cfg);
    Tensor* bias = tensor_ones(bias_shape, 2, &cfg);

    float* ad = (float*)tensor_data_ptr(A);
    float* bd = (float*)tensor_data_ptr(B);
    float* bi = (float*)tensor_data_ptr(bias);
    for (int i = 0; i < N * N; i++) {
        ad[i] = (float)rand() / (float)RAND_MAX - 0.5f;
        bd[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    }
    for (int i = 0; i < N; i++)
        bi[i] = (float)rand() / (float)RAND_MAX - 0.5f;

    Tensor* w = tensor_relu(tensor_add(tensor_matmul(A, B), bias));
    (void)tensor_data_ptr(w);

    double start = get_time_sec();
    for (int i = 0; i < iters; i++) {
        Tensor* C = tensor_relu(tensor_add(tensor_matmul(A, B), bias));
        (void)tensor_data_ptr(C);
    }
    double elapsed = get_time_sec() - start;

    tensor_free(A);
    tensor_free(B);
    tensor_free(bias);
    double flop = fused_flops(N) * iters;
    return (BenchResult){.gflops = flop / elapsed / 1e9, .time_ms = elapsed / iters * 1e3};
}

int main(void) {
    cml_init();
    srand(42);

    cml_print_simd_caps();

    CMLBlasContext* blas = cml_blas_get_context();
    int have_blas        = blas && blas->initialized;
    if (have_blas)
        printf("BLAS: %s\n", cml_blas_get_library_name(blas));
    else
        printf("BLAS: NOT AVAILABLE (using fallback)\n");

    printf("\nGEMM Throughput Benchmark (float32, C = A @ B, square NxN)\n\n");

    int sizes[]   = {2048, 4096};
    int num_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    for (int si = 0; si < num_sizes; si++) {
        int N     = sizes[si];
        int iters = 2;

        printf("N = %d  (%d iterations)\n", N, iters);

        if (N <= 256)
            print_row("Naive (triple loop)", bench_naive(N, iters));

        if (have_blas)
            print_row("Raw BLAS (cblas_sgemm)", bench_blas(blas, N, iters));

        // CML tensor paths have JIT compilation overhead — skip above 1024
        if (N <= 1024)
            print_row("CML tensor_matmul", bench_tensor_matmul(N, iters));

        printf("  -- fused: matmul + bias + relu --\n");

        if (have_blas)
            print_row("BLAS + manual bias+relu", bench_blas_unfused(blas, N, iters));

        if (N <= 1024)
            print_row("CML fused (matmul+add+relu)", bench_tensor_fused(N, iters));

        printf("\n");
    }

    printf("For a single GEMM, raw BLAS is fastest (zero overhead).\n");
    printf("CML adds IR/graph overhead per op. The win comes from fusion\n");
    printf("on chained ops -- fewer memory passes, better cache utilization.\n");

    cml_cleanup();
    return 0;
}
