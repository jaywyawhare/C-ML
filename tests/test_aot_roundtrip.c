/*
 * AOT round-trip test: build an IR graph, AOT-compile it to a shared library,
 * load it back with dlopen/dlsym, execute, and compare against values computed
 * directly in this test. Exercises the multi-input ABI, intermediate buffers,
 * matmul, elementwise ops, activations, and scalar broadcasting.
 */
#include "cml.h"
#include "ops/ir/aot.h"
#include "ops/ir/context.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static int g_failures = 0;

static void check_close(const char* what, float got, float want) {
    float tol = 1e-4f * (1.0f + fabsf(want));
    if (fabsf(got - want) > tol) {
        printf("  FAIL %s: got %.6f want %.6f\n", what, got, want);
        g_failures++;
    }
}

/* out = relu(x @ W), x:[M,K], W:[K,N] -> out:[M,N]  (two graph inputs) */
static void test_matmul_relu(void) {
    printf("test_matmul_relu\n");
    cml_ir_reset_global_context();

    const int M = 2, K = 3, N = 2;
    float xdata[6] = {1, 2, 3, -1, 0, 1};      /* [2,3] */
    float wdata[6] = {1, -1, 0, 2, -1, 1};     /* [3,2] */
    int xs[2] = {M, K}, ws[2] = {K, N};

    Tensor* x = tensor_from_data(xdata, xs, 2, NULL);
    Tensor* w = tensor_from_data(wdata, ws, 2, NULL);
    Tensor* mm = uop_matmul(x, w);
    Tensor* out = uop_relu(mm);
    (void)out;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    AOTCompileOptions opts = cml_aot_default_options();
    opts.format = AOT_FORMAT_SHARED_LIB;
    const char* so = "/tmp/cml_aot_test_mm.so";
    if (cml_aot_compile(ir, so, &opts) != 0) {
        printf("  FAIL compile\n"); g_failures++; return;
    }

    CMLAOTModel* model = cml_aot_load(so);
    if (!model) { printf("  FAIL load\n"); g_failures++; return; }

    float outbuf[4] = {0};
    int os[2] = {M, N};
    Tensor* ot = tensor_from_data(outbuf, os, 2, NULL);
    Tensor* ins[2] = {x, w};
    Tensor* outs[1] = {ot};
    if (cml_aot_execute(model, ins, 2, outs, 1) != 0) {
        printf("  FAIL execute\n"); g_failures++; cml_aot_free(model); return;
    }

    /* Expected: relu(x @ w) */
    float expected[4];
    for (int m = 0; m < M; m++)
        for (int nn = 0; nn < N; nn++) {
            float acc = 0;
            for (int k = 0; k < K; k++) acc += xdata[m * K + k] * wdata[k * N + nn];
            expected[m * N + nn] = acc > 0 ? acc : 0;
        }
    const float* got = (const float*)ot->data;
    for (int i = 0; i < M * N; i++) {
        char lbl[32]; snprintf(lbl, sizeof(lbl), "mm_relu[%d]", i);
        check_close(lbl, got[i], expected[i]);
    }
    cml_aot_free(model);
}

/* out = sigmoid(x * 2 - 1), tests scalar broadcast + chained intermediates */
static void test_elementwise_chain(void) {
    printf("test_elementwise_chain\n");
    cml_ir_reset_global_context();

    const int N = 5;
    float xdata[5] = {-2, -0.5f, 0, 0.5f, 2};
    int xs[1] = {N};
    float two = 2.0f, one = 1.0f;
    int ss[1] = {1};

    Tensor* x = tensor_from_data(xdata, xs, 1, NULL);
    Tensor* c2 = tensor_from_data(&two, ss, 1, NULL);
    Tensor* c1 = tensor_from_data(&one, ss, 1, NULL);
    Tensor* scaled = uop_mul(x, c2);
    Tensor* shifted = uop_sub(scaled, c1);
    Tensor* out = uop_sigmoid(shifted);
    (void)out;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    AOTCompileOptions opts = cml_aot_default_options();
    const char* so = "/tmp/cml_aot_test_ew.so";
    if (cml_aot_compile(ir, so, &opts) != 0) {
        printf("  FAIL compile\n"); g_failures++; return;
    }
    CMLAOTModel* model = cml_aot_load(so);
    if (!model) { printf("  FAIL load\n"); g_failures++; return; }

    float outbuf[5] = {0};
    Tensor* ot = tensor_from_data(outbuf, xs, 1, NULL);
    /* Inputs in first-use order: x, c2, c1 */
    Tensor* ins[3] = {x, c2, c1};
    Tensor* outs[1] = {ot};
    if (cml_aot_execute(model, ins, 3, outs, 1) != 0) {
        printf("  FAIL execute\n"); g_failures++; cml_aot_free(model); return;
    }
    const float* got = (const float*)ot->data;
    for (int i = 0; i < N; i++) {
        float e = 1.0f / (1.0f + expf(-(xdata[i] * 2.0f - 1.0f)));
        char lbl[32]; snprintf(lbl, sizeof(lbl), "ew[%d]", i);
        check_close(lbl, got[i], e);
    }
    cml_aot_free(model);
}

int main(void) {
    if (cml_init() != 0) { printf("cml_init failed\n"); return 1; }

    test_matmul_relu();
    test_elementwise_chain();

    if (g_failures == 0) {
        printf("AOT round-trip: ALL PASSED\n");
        return 0;
    }
    printf("AOT round-trip: %d failure(s)\n", g_failures);
    return 1;
}
