#include "nn/quantization.h"
#include "alloc/cml_allocator.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test_assert.h"

static float frand(void) { return ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f; }

static void test_int8_roundtrip(void) {
    printf("test_int8_roundtrip...\n");
    int rows = 6, cols = 20;
    float* x  = (float*)cml_malloc((size_t)rows * cols * sizeof(float));
    int8_t* q = (int8_t*)cml_malloc((size_t)rows * cols);
    float* s  = (float*)cml_malloc((size_t)rows * sizeof(float));
    float* dq = (float*)cml_malloc((size_t)rows * cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++)
        x[i] = frand();
    cml_quantize_int8_row(x, rows, cols, q, s);
    cml_dequantize_int8_row(q, s, rows, cols, dq);
    /* Symmetric rounding: per-element error is at most half a quantization step. */
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            float err = fabsf(x[r * cols + c] - dq[r * cols + c]);
            ASSERT(err <= 0.5f * s[r] + 1e-6f, "int8 round-trip within half a step");
        }
    cml_free(x);
    cml_free(q);
    cml_free(s);
    cml_free(dq);
    tests_passed++;
}

static void test_int4_roundtrip(void) {
    printf("test_int4_roundtrip...\n");
    int n = 64, group = 16;
    int ngroups = (n + group - 1) / group;
    float* x    = (float*)cml_malloc((size_t)n * sizeof(float));
    int8_t* q   = (int8_t*)cml_malloc((size_t)n);
    float* s    = (float*)cml_malloc((size_t)ngroups * sizeof(float));
    float* dq   = (float*)cml_malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++)
        x[i] = frand();
    cml_quantize_int4_group(x, n, group, q, s);
    cml_dequantize_int4_group(q, s, n, group, dq);
    for (int i = 0; i < n; i++) {
        ASSERT(q[i] >= -8 && q[i] <= 7, "int4 codes in range");
        ASSERT(fabsf(x[i] - dq[i]) <= 0.5f * s[i / group] + 1e-6f,
               "int4 round-trip within half a step");
    }
    cml_free(x);
    cml_free(q);
    cml_free(s);
    cml_free(dq);
    tests_passed++;
}

static void test_int8_matmul(void) {
    printf("test_int8_matmul...\n");
    int M = 5, K = 32, N = 7;
    float* A  = (float*)cml_malloc((size_t)M * K * sizeof(float));
    float* Bt = (float*)cml_malloc((size_t)N * K * sizeof(float));
    for (int i = 0; i < M * K; i++)
        A[i] = frand();
    for (int i = 0; i < N * K; i++)
        Bt[i] = frand();

    int8_t* qA = (int8_t*)cml_malloc((size_t)M * K);
    int8_t* qB = (int8_t*)cml_malloc((size_t)N * K);
    float* sA  = (float*)cml_malloc((size_t)M * sizeof(float));
    float* sB  = (float*)cml_malloc((size_t)N * sizeof(float));
    cml_quantize_int8_row(A, M, K, qA, sA);
    cml_quantize_int8_row(Bt, N, K, qB, sB);

    float* C = (float*)cml_malloc((size_t)M * N * sizeof(float));
    ASSERT(cml_matmul_int8(qA, sA, M, K, qB, sB, N, C) == 0, "matmul returns 0");

    float amax = 0.0f, bmax = 0.0f;
    for (int i = 0; i < M * K; i++)
        if (fabsf(A[i]) > amax)
            amax = fabsf(A[i]);
    for (int i = 0; i < N * K; i++)
        if (fabsf(Bt[i]) > bmax)
            bmax = fabsf(Bt[i]);
    /* Quantization error per operand ~ amax/127; bound the accumulated matmul error. */
    float tol = 0.02f * (float)K * amax * bmax + 1e-4f;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float ref = 0.0f;
            for (int k = 0; k < K; k++)
                ref += A[m * K + k] * Bt[n * K + k];
            ASSERT(fabsf(ref - C[m * N + n]) < tol, "int8 matmul close to fp32");
        }
    cml_free(A);
    cml_free(Bt);
    cml_free(qA);
    cml_free(qB);
    cml_free(sA);
    cml_free(sB);
    cml_free(C);
    tests_passed++;
}

static void test_int8_linear(void) {
    printf("test_int8_linear...\n");
    int M = 4, K = 24, N = 6;
    float* x = (float*)cml_malloc((size_t)M * K * sizeof(float));
    float* W = (float*)cml_malloc((size_t)N * K * sizeof(float)); /* [N,K] nn.Linear layout */
    for (int i = 0; i < M * K; i++)
        x[i] = frand();
    for (int i = 0; i < N * K; i++)
        W[i] = frand();

    int8_t* Wq = (int8_t*)cml_malloc((size_t)N * K);
    float* Ws  = (float*)cml_malloc((size_t)N * sizeof(float));
    cml_quantize_int8_row(W, N, K, Wq, Ws);

    float* y = (float*)cml_malloc((size_t)M * N * sizeof(float));
    ASSERT(cml_linear_int8_forward(x, M, K, Wq, Ws, N, y) == 0, "int8 linear returns 0");

    float amax = 0.0f, wmax = 0.0f;
    for (int i = 0; i < M * K; i++)
        if (fabsf(x[i]) > amax)
            amax = fabsf(x[i]);
    for (int i = 0; i < N * K; i++)
        if (fabsf(W[i]) > wmax)
            wmax = fabsf(W[i]);
    float tol = 0.02f * (float)K * amax * wmax + 1e-4f;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float ref = 0.0f;
            for (int k = 0; k < K; k++)
                ref += x[m * K + k] * W[n * K + k];
            ASSERT(fabsf(ref - y[m * N + n]) < tol, "int8 linear close to fp32");
        }
    cml_free(x);
    cml_free(W);
    cml_free(Wq);
    cml_free(Ws);
    cml_free(y);
    tests_passed++;
}

int main(void) {
    printf("Quantization tests\n\n");
    srand(7);
    test_int8_roundtrip();
    test_int4_roundtrip();
    test_int8_matmul();
    test_int8_linear();
    printf("\nResults: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
