/*
 * AUDIT #19: weight-only affine-int8 matmul dispatch, plus the packed 4-bit
 * follow-up (affine int4 and block-wise NF4).
 *
 * Verifies (a) each GEMM primitive against an exact hand computation, and
 * (b) the end-to-end paths — quantize an f32 weight, matmul through the tensor
 * API, and check the result against the f32 reference within the rigorous
 * quantization-error bound.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "tensor/tensor.h"
#include "test_harness.h"
#include "ops/uops.h"
#include "core/quantization.h"

static const TensorConfig cpu_f32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static uint32_t s_rng = 0x1234567u;
static float randf(float lo, float hi) {
    s_rng ^= s_rng << 13; s_rng ^= s_rng >> 17; s_rng ^= s_rng << 5;
    return lo + ((float)(s_rng & 0xffffff) / (float)0xffffff) * (hi - lo);
}

static int check(const char* name, int ok) {
    tests_run++;
    if (ok) { tests_passed++; printf("  PASS: %s\n", name); }
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

/* ---- packed 4-bit ---- */

/* Exact hand check of the packed-int4 GEMM primitive. */
static int test_int4_gemm_primitive(void) {
    int M = 2, K = 3, N = 2;
    float x[] = {1, 2, 3,   4, 5, 6};              /* [M,K] */
    uint8_t packed[3];
    size_t wn = 6;
    int vals[6] = {1, -2, 3, 4, -5, 6};
    memset(packed, 0, sizeof(packed));
    for (size_t i = 0; i < wn; i++) {
        uint8_t nib = (uint8_t)((int8_t)vals[i] & 0x0F);
        if ((i & 1) == 0) packed[i >> 1] |= (uint8_t)(nib << 4);
        else              packed[i >> 1] |= nib;
    }
    float scale = 0.5f; /* zero_point 0 (symmetric) */
    float y[4];
    if (cml_qmatmul_affine_int4(x, packed, scale, 0, y, M, K, N) != 0) return 0;

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0;
            for (int k = 0; k < K; k++) acc += x[m*K+k] * (float)vals[k*N+n];
            if (fabsf(y[m*N+n] - scale * acc) > 1e-5f) return 0;
        }
    return 1;
}

/* End-to-end int4: quantize weight, matmul via tensor API vs f32 ref within
 * the rigorous bound |Δy| <= 0.5 * scale * Σ_k |x[m,k]| (|Δw| <= scale/2). */
static int test_int4_end_to_end(int M, int K, int N) {
    float* xd = malloc(sizeof(float) * M * K);
    float* wd = malloc(sizeof(float) * K * N);
    for (int i = 0; i < M * K; i++) xd[i] = randf(-2.0f, 2.0f);
    for (int i = 0; i < K * N; i++) wd[i] = randf(-1.0f, 1.0f);

    Tensor* x  = tensor_from_data(xd, (int[]){M, K}, 2, &cpu_f32);
    Tensor* w  = tensor_from_data(wd, (int[]){K, N}, 2, &cpu_f32);
    Tensor* wq = cml_quantize_weight_int4(w);
    if (!wq || wq->quant_type != CML_QUANT_AFFINE_INT4 || !wq->quant_data ||
        wq->quant_data_bytes != (size_t)(K * N + 1) / 2) {
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
        float bound = 0.5f * scale * xabs + 1e-4f;
        for (int n = 0; n < N; n++) {
            float ref = 0; for (int k = 0; k < K; k++) ref += xd[m*K+k] * wd[k*N+n];
            if (fabsf(yd[m*N+n] - ref) > bound * 1.5f) { ok = 0; break; }
        }
    }

    free(xd); free(wd);
    tensor_free(x); tensor_free(w); tensor_free(wq); tensor_free(y);
    return ok;
}

/* NF4 GEMM must match a plain matmul over the dequantized weights exactly
 * (same table, same scales), and stay near the f32 reference overall. */
static int test_nf4_end_to_end(int M, int K, int N, int block_size) {
    float* xd = malloc(sizeof(float) * M * K);
    float* wd = malloc(sizeof(float) * K * N);
    for (int i = 0; i < M * K; i++) xd[i] = randf(-2.0f, 2.0f);
    for (int i = 0; i < K * N; i++) wd[i] = randf(-1.0f, 1.0f);

    Tensor* x  = tensor_from_data(xd, (int[]){M, K}, 2, &cpu_f32);
    Tensor* w  = tensor_from_data(wd, (int[]){K, N}, 2, &cpu_f32);
    Tensor* wq = cml_quantize_weight_nf4(w, block_size);
    if (!wq || wq->quant_type != CML_QUANT_NF4 || !wq->quant_data ||
        wq->quant_block_size != block_size) {
        free(xd); free(wd); return 0;
    }

    size_t wn = (size_t)K * N;
    int num_scales = (int)((wn + (size_t)block_size - 1) / (size_t)block_size);
    const uint8_t* blob = (const uint8_t*)wq->quant_data;
    const float* scales = (const float*)blob;
    const uint8_t* packed = blob + (size_t)num_scales * sizeof(float);

    /* Dequantized-weight reference (unpack with the documented convention). */
    float* wdq = malloc(sizeof(float) * wn);
    for (size_t i = 0; i < wn; i++) {
        uint8_t b = packed[i >> 1];
        int idx = (i & 1) ? (b & 0x0F) : (b >> 4);
        int blk = (int)(i / (size_t)block_size);
        if (blk >= num_scales) blk = num_scales - 1;
        wdq[i] = CML_NF4_TABLE[idx] * scales[blk];
    }

    Tensor* y = uop_matmul(x, wq);
    tensor_ensure_executed(y);
    if (!y || !y->data || y->numel != (size_t)M * N) { free(xd); free(wd); free(wdq); return 0; }
    float* yd = (float*)y->data;

    int ok = 1;
    float worst_dequant_gap = 0.0f;
    float max_scale = 0.0f;
    for (int b = 0; b < num_scales; b++)
        if (scales[b] > max_scale) max_scale = scales[b];
    for (int m = 0; m < M && ok; m++) {
        float xabs = 0; for (int k = 0; k < K; k++) xabs += fabsf(xd[m*K+k]);
        /* max half-gap between adjacent NF4 table entries is ~0.152 */
        float bound = 0.152f * max_scale * xabs + 1e-3f;
        for (int n = 0; n < N; n++) {
            float ref_q = 0, ref_f = 0;
            for (int k = 0; k < K; k++) {
                ref_q += xd[m*K+k] * wdq[k*N+n];
                ref_f += xd[m*K+k] * wd[k*N+n];
            }
            /* exactness against the dequantized weights */
            float d1 = fabsf(yd[m*N+n] - ref_q);
            if (d1 > 1e-3f) { ok = 0; break; }
            float d2 = fabsf(yd[m*N+n] - ref_f);
            if (d2 > worst_dequant_gap) worst_dequant_gap = d2;
            if (d2 > bound * 1.5f) { ok = 0; break; }
        }
    }

    free(xd); free(wd); free(wdq);
    tensor_free(x); tensor_free(w); tensor_free(wq); tensor_free(y);
    return ok;
}

/* Footprint: int4 payload is 8x smaller than the f32 weight it replaces. */
static int test_int4_memory_footprint(void) {
    int K = 64, N = 128;
    float* wd = malloc(sizeof(float) * K * N);
    for (int i = 0; i < K * N; i++) wd[i] = randf(-1, 1);
    Tensor* w  = tensor_from_data(wd, (int[]){K, N}, 2, &cpu_f32);
    Tensor* wq = cml_quantize_weight_int4(w);
    int ok = wq && wq->quant_data_bytes == (size_t)K * N / 2; /* half byte each */
    free(wd); tensor_free(w); if (wq) tensor_free(wq);
    return ok;
}

int main(void) {
    printf("=== AUDIT #19: affine-int8 + packed int4/NF4 matmul ===\n");
    check("gemm_primitive_exact", test_gemm_primitive());
    check("end_to_end_8x16x8",   test_end_to_end(8, 16, 8));
    check("end_to_end_1x64x32",  test_end_to_end(1, 64, 32));
    check("end_to_end_17x33x9",  test_end_to_end(17, 33, 9));
    check("memory_footprint_4x", test_memory_footprint());
    check("int4_gemm_primitive_exact", test_int4_gemm_primitive());
    check("int4_end_to_end_8x16x8",   test_int4_end_to_end(8, 16, 8));
    check("int4_end_to_end_1x65x33",  test_int4_end_to_end(1, 65, 33)); /* odd K*N packing */
    check("int4_memory_footprint_8x", test_int4_memory_footprint());
    check("nf4_end_to_end_8x16x8_b16",  test_nf4_end_to_end(8, 16, 8, 16));
    check("nf4_end_to_end_4x32x16_b64", test_nf4_end_to_end(4, 32, 16, 64));
    check("nf4_end_to_end_2x17x9_b8",   test_nf4_end_to_end(2, 17, 9, 8)); /* odd numel */
    return TEST_SUMMARY();
}
