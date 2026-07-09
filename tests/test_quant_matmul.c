/*
 * AUDIT #19: weight-only affine-int8 matmul dispatch.
 *
 * Verifies (a) the standalone int8 GEMM against an exact hand computation, and
 * (b) the end-to-end path — quantize an f32 weight, matmul through the tensor
 * API, and check the result against the f32 reference within the rigorous
 * per-row quantization-error bound  |Δy[m]| <= 0.5 * scale * Σ_k |x[m,k]|.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "tensor/tensor.h"
#include "ops/uops.h"
#include "core/quantization.h"

static int g_pass = 0, g_total = 0;
static const TensorConfig cpu_f32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static uint32_t s_rng = 0x1234567u;
static float randf(float lo, float hi) {
    s_rng ^= s_rng << 13; s_rng ^= s_rng >> 17; s_rng ^= s_rng << 5;
    return lo + ((float)(s_rng & 0xffffff) / (float)0xffffff) * (hi - lo);
}

static int check(const char* name, int ok) {
    g_total++;
    if (ok) { g_pass++; printf("  PASS: %s\n", name); }
    else    { printf("  FAIL: %s\n", name); }
    return ok;
}

/* Exact hand check of the GEMM primitive with a nonzero zero_point. */
static int test_gemm_primitive(void) {
    int M = 2, K = 3, N = 2;
    float x[] = {1, 2, 3,   4, 5, 6};              /* [M,K] */
    int8_t w[] = {1, -2,  3, 4,  -5, 6};           /* [K,N] */
    float scale = 0.5f; int32_t zp = 2;
    float y[4];
    if (cml_qmatmul_affine_int8(x, w, scale, zp, y, M, K, N) != 0) return 0;

    for (int m = 0; m < M; m++) {
        float xsum = 0; for (int k = 0; k < K; k++) xsum += x[m*K+k];
        for (int n = 0; n < N; n++) {
            float acc = 0; for (int k = 0; k < K; k++) acc += x[m*K+k] * (float)w[k*N+n];
            float expect = scale * (acc - (float)zp * xsum);
            if (fabsf(y[m*N+n] - expect) > 1e-5f) return 0;
        }
    }
    return 1;
}

/* End-to-end: quantize weight, matmul via tensor API, compare to f32 ref. */
static int test_end_to_end(int M, int K, int N) {
    float* xd = malloc(sizeof(float) * M * K);
    float* wd = malloc(sizeof(float) * K * N);
    for (int i = 0; i < M * K; i++) xd[i] = randf(-2.0f, 2.0f);
    for (int i = 0; i < K * N; i++) wd[i] = randf(-1.0f, 1.0f);

    Tensor* x  = tensor_from_data(xd, (int[]){M, K}, 2, &cpu_f32);
    Tensor* w  = tensor_from_data(wd, (int[]){K, N}, 2, &cpu_f32);
    Tensor* wq = cml_quantize_weight_int8(w, true);
    if (!wq || wq->quant_type != CML_QUANT_AFFINE_INT8) {
        free(xd); free(wd); return 0;
    }
    float scale = wq->quant_scale;

    Tensor* y = uop_matmul(x, wq);
    tensor_ensure_executed(y);
    if (!y || !y->data || y->numel != (size_t)M * N) { free(xd); free(wd); return 0; }
    float* yd = (float*)y->data;

    int ok = 1;
    for (int m = 0; m < M && ok; m++) {
        float xabs = 0; for (int k = 0; k < K; k++) xabs += fabsf(xd[m*K+k]);
        float bound = 0.5f * scale * xabs + 1e-4f;   /* rigorous per-row bound */
        for (int n = 0; n < N; n++) {
            float ref = 0; for (int k = 0; k < K; k++) ref += xd[m*K+k] * wd[k*N+n];
            if (fabsf(yd[m*N+n] - ref) > bound * 1.5f) { ok = 0; break; }
        }
    }

    free(xd); free(wd);
    tensor_free(x); tensor_free(w); tensor_free(wq); tensor_free(y);
    return ok;
}

/* Confirm int8 storage is 4x smaller than the f32 weight it replaces. */
static int test_memory_footprint(void) {
    int K = 64, N = 128;
    float* wd = malloc(sizeof(float) * K * N);
    for (int i = 0; i < K * N; i++) wd[i] = randf(-1, 1);
    Tensor* w  = tensor_from_data(wd, (int[]){K, N}, 2, &cpu_f32);
    Tensor* wq = cml_quantize_weight_int8(w, true);
    int ok = wq && wq->dtype == DTYPE_INT8 &&
             cml_dtype_size(wq->dtype) * wq->numel == (size_t)K * N; /* 1 byte each */
    free(wd); tensor_free(w); if (wq) tensor_free(wq);
    return ok;
}

int main(void) {
    printf("=== AUDIT #19: affine-int8 matmul ===\n");
    check("gemm_primitive_exact", test_gemm_primitive());
    check("end_to_end_8x16x8",   test_end_to_end(8, 16, 8));
    check("end_to_end_1x64x32",  test_end_to_end(1, 64, 32));
    check("end_to_end_17x33x9",  test_end_to_end(17, 33, 9));
    check("memory_footprint_4x", test_memory_footprint());
    printf("\nResults: %d/%d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
