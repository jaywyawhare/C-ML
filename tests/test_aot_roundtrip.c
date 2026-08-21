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

#include "test_harness.h"

static void check_close(const char* what, float got, float want) {
    float tol = 1e-4f * (1.0f + fabsf(want));
    tests_run++;
    if (fabsf(got - want) <= tol) {
        tests_passed++;
        return;
    }
    printf("  FAIL %s: got %.6f want %.6f\n", what, got, want);
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
        printf("  FAIL compile\n"); tests_run++; return;
    }

    CMLAOTModel* model = cml_aot_load(so);
    if (!model) { printf("  FAIL load\n"); tests_run++; return; }

    float outbuf[4] = {0};
    int os[2] = {M, N};
    Tensor* ot = tensor_from_data(outbuf, os, 2, NULL);
    Tensor* ins[2] = {x, w};
    Tensor* outs[1] = {ot};
    if (cml_aot_execute(model, ins, 2, outs, 1) != 0) {
        printf("  FAIL execute\n"); tests_run++; cml_aot_free(model); return;
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
        printf("  FAIL compile\n"); tests_run++; return;
    }
    CMLAOTModel* model = cml_aot_load(so);
    if (!model) { printf("  FAIL load\n"); tests_run++; return; }

    float outbuf[5] = {0};
    Tensor* ot = tensor_from_data(outbuf, xs, 1, NULL);
    /* Inputs in first-use order: x, c2, c1 */
    Tensor* ins[3] = {x, c2, c1};
    Tensor* outs[1] = {ot};
    if (cml_aot_execute(model, ins, 3, outs, 1) != 0) {
        printf("  FAIL execute\n"); tests_run++; cml_aot_free(model); return;
    }
    const float* got = (const float*)ot->data;
    for (int i = 0; i < N; i++) {
        float e = 1.0f / (1.0f + expf(-(xdata[i] * 2.0f - 1.0f)));
        char lbl[32]; snprintf(lbl, sizeof(lbl), "ew[%d]", i);
        check_close(lbl, got[i], e);
    }
    cml_aot_free(model);
}


/* Reductions: the emitter picks init/accumulate per op, and MIN_REDUCE used to
 * be produced by the switch's `default` arm rather than its own case -- correct
 * only for as long as it stayed the last unhandled member of the label group.
 * Nothing exercised the reduction path at all, so verify each one end to end. */
static void test_reductions(void) {
    printf("test_reductions\n");

    const int N = 6;
    float xdata[6] = {3.0f, -1.0f, 4.0f, -1.5f, 5.0f, 2.0f};
    int xs[1] = {N};
    int os[1] = {1};

    struct { const char* name; int which; float expected; } cases[] = {
        {"sum",  0, 3.0f - 1.0f + 4.0f - 1.5f + 5.0f + 2.0f},
        {"mean", 1, (3.0f - 1.0f + 4.0f - 1.5f + 5.0f + 2.0f) / 6.0f},
        {"max",  2, 5.0f},
        {"min",  3, -1.5f},
        {"prod", 4, 3.0f * -1.0f * 4.0f * -1.5f * 5.0f * 2.0f},
    };

    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        cml_ir_reset_global_context();
        Tensor* x = tensor_from_data(xdata, xs, 1, NULL);
        Tensor* out = NULL;
        switch (cases[c].which) {
        case 0: out = uop_sum(x, NULL); break;
        case 1: out = uop_mean(x, NULL); break;
        case 2: out = uop_max_reduce(x, NULL); break;
        case 3: out = uop_min_reduce(x, NULL); break;
        default: out = uop_prod(x, NULL); break;
        }
        (void)out;

        CMLGraph_t ir = cml_ir_get_or_create_context();
        AOTCompileOptions opts = cml_aot_default_options();
        char so[128];
        snprintf(so, sizeof(so), "/tmp/cml_aot_test_red_%zu.so", c);
        if (cml_aot_compile(ir, so, &opts) != 0) {
            printf("  FAIL compile %s\n", cases[c].name); tests_run++; continue;
        }
        CMLAOTModel* model = cml_aot_load(so);
        if (!model) { printf("  FAIL load %s\n", cases[c].name); tests_run++; continue; }

        float outbuf[1] = {0};
        Tensor* ot = tensor_from_data(outbuf, os, 1, NULL);
        Tensor* ins[1] = {x};
        Tensor* outs[1] = {ot};
        if (cml_aot_execute(model, ins, 1, outs, 1) != 0) {
            printf("  FAIL execute %s\n", cases[c].name); tests_run++;
            cml_aot_free(model); continue;
        }
        check_close(cases[c].name, ((const float*)ot->data)[0], cases[c].expected);
        cml_aot_free(model);
    }
}


/* A FILL of an integral value emitted `x[i] = 0f;` -- "%.9gf" renders 0.0f as
 * "0", and "0f" is not a C constant, so the generated file did not compile at
 * all. Any graph containing a whole-number constant hit this. */
static void test_integral_constant(void) {
    printf("test_integral_constant\n");
    cml_ir_reset_global_context();

    int shape[1] = {4};
    float xdata[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor* x = tensor_from_data(xdata, shape, 1, NULL);
    Tensor* ones = uop_fill(shape, 1, 1.0f);      /* renders as "1" -> "1f" */
    Tensor* zero = uop_fill(shape, 1, 0.0f);      /* renders as "0" -> "0f" */
    Tensor* sum  = uop_add(uop_add(x, ones), zero);
    (void)sum;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    AOTCompileOptions opts = cml_aot_default_options();
    opts.format = AOT_FORMAT_SHARED_LIB;
    const char* so = "/tmp/cml_aot_test_const.so";
    if (cml_aot_compile(ir, so, &opts) != 0) {
        printf("  FAIL compile (integral float literal)\n"); tests_run++; return;
    }
    printf("  integral constants compile: PASS\n");
}

/* The AOT backend is a third code generator, alongside the interpreter and the
 * JIT, and it had its own copies of the NaN-swallowing selection patterns:
 * relu emitted `x > 0.0f ? x : 0.0f` and the max/min reductions accumulated
 * with ordered compares, so a NaN input came back as a plausible finite value. */
static void test_nan_through_aot(void) {
    printf("test_nan_through_aot\n");
    cml_ir_reset_global_context();

    int shape[1] = {4};
    float xdata[4] = {NAN, -2.0f, 3.0f, 1.0f};
    Tensor* x = tensor_from_data(xdata, shape, 1, NULL);
    Tensor* out = uop_relu(x);
    (void)out;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    AOTCompileOptions opts = cml_aot_default_options();
    opts.format = AOT_FORMAT_SHARED_LIB;
    const char* so = "/tmp/cml_aot_test_nan.so";
    if (cml_aot_compile(ir, so, &opts) != 0) {
        printf("  FAIL compile\n"); tests_run++; return;
    }
    CMLAOTModel* model = cml_aot_load(so);
    if (!model) { printf("  FAIL load\n"); tests_run++; return; }

    float outbuf[4] = {0};
    Tensor* ot = tensor_from_data(outbuf, shape, 1, NULL);
    Tensor* ins[1] = {x};
    Tensor* outs[1] = {ot};
    if (cml_aot_execute(model, ins, 1, outs, 1) != 0) {
        printf("  FAIL execute\n"); tests_run++; cml_aot_free(model); return;
    }
    const float* got = (const float*)ot->data;
    if (!isnan(got[0])) {
        printf("  FAIL relu(nan) = %g, expected nan (AOT swallowed the NaN)\n", got[0]);
        tests_run++;
    } else if (got[1] != 0.0f || got[2] != 3.0f || got[3] != 1.0f) {
        printf("  FAIL relu finite values = [%g %g %g], expected [0 3 1]\n",
               got[1], got[2], got[3]);
        tests_run++;
    } else {
        printf("  relu propagates NaN: PASS\n");
    }
    cml_aot_free(model);
}

int main(void) {
    if (cml_init() != 0) { printf("cml_init failed\n"); return 1; }

    test_matmul_relu();
    test_elementwise_chain();
    test_reductions();
    test_integral_constant();
    test_nan_through_aot();

    return TEST_SUMMARY();
}
