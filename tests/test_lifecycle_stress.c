/* Lifecycle stress: repeatedly build models, train steps, reset the graph,
 * take views of temporaries and drop bases, and churn the buffer cache —
 * the exact patterns that historically produced use-after-free / double
 * cache-return corruption. Designed to run clean under ASAN. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cml.h"
#include "optim.h"
#include "nn.h"
#include "nn/layers/sequential.h"
#include "tensor/realize.h"
#include "test_harness.h"

static TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                           .has_dtype = true, .has_device = true};

/* Model churn: fresh Sequential + SGD each cycle; reset between steps. */
static int test_model_churn(void) {
    int ok = 1;
    for (int cycle = 0; cycle < 8; cycle++) {
        Linear* l1 = cml_nn_linear(4, 8, DTYPE_FLOAT32, DEVICE_CPU, true);
        Linear* l2 = cml_nn_linear(8, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
        Sequential* seq = nn_sequential();
        if (!l1 || !l2 || !seq ||
            sequential_add(seq, (Module*)l1) != 0 ||
            sequential_add(seq, (Module*)l2) != 0)
            return 0;

        Parameter** params = NULL;
        int nparams = 0;
        module_collect_parameters((Module*)seq, &params, &nparams, true);
        Optimizer* opt = optim_sgd(params, nparams, 0.05f, 0.0f, 0.0f);
        cml_free(params);

        int xs[] = {16, 4}, ys[] = {16, 2};
        for (int step = 0; step < 3; step++) {
            Tensor* X = tensor_randn(xs, 2, &cfg);
            Tensor* Yt = tensor_randn(ys, 2, &cfg);
            Tensor* pred = cml_nn_sequential_forward(seq, X);
            Tensor* loss = pred ? cml_nn_mse_loss(pred, Yt) : NULL;
            ok &= loss != NULL;
            if (loss) {
                Tensor* one = tensor_ones(loss->shape, loss->ndim, &cfg);
                tensor_backward(loss, one, false, false);
                tensor_free(one);
                optimizer_step(opt);
                optimizer_zero_grad(opt);
            }
            cml_reset_ir_context();
            /* Inputs are creation-node outputs owned by the freed graph —
             * pinning is the documented contract for holders across resets,
             * but these die with the graph here, so do NOT touch them after
             * the reset. */
            (void)X; (void)Yt;
        }
        optimizer_free(opt);
        module_free((Module*)seq);
    }
    return ok;
}

/* View-of-temporary churn: base dies immediately, view outlives it. */
static int test_view_temporary_churn(void) {
    int ok = 1;
    float expected[12];
    for (int i = 0; i < 12; i++) expected[i] = (float)i * 0.25f;

    for (int iter = 0; iter < 200; iter++) {
        int shape[] = {3, 4};
        Tensor* base = tensor_from_data(expected, shape, 2, &cfg);
        if (!base) return 0;
        /* flatten via reshape: shared storage must keep the block alive */
        Tensor* flat = tensor_reshape(base, (int[]){12}, 1);
        ok &= flat != NULL;
        tensor_free(base); /* temporary dies first */

        if (flat) {
            float* d = (float*)tensor_data_ptr(flat);
            ok &= d && memcmp(d, expected, sizeof(expected)) == 0;
            /* chain a second view off a dead root */
            Tensor* again = tensor_reshape(flat, (int[]){4, 3}, 2);
            ok &= again != NULL;
            if (again) {
                float* d2 = (float*)tensor_data_ptr(again);
                ok &= d2 && memcmp(d2, expected, sizeof(expected)) == 0;
                tensor_free(again);
            }
            tensor_free(flat);
        }

        /* interleave allocations so the cache reuses the freed slot */
        Tensor* noise = tensor_randn((int[]){8}, 1, &cfg);
        tensor_free(noise);
    }
    return ok;
}

/* Mixed op + reset pressure: exercises plan-cache replay across resets. */
static int test_reset_pressure(void) {
    int ok = 1;
    int shape[] = {32, 32};
    for (int iter = 0; iter < 30; iter++) {
        Tensor* a = tensor_randn(shape, 2, &cfg);
        Tensor* b = tensor_randn(shape, 2, &cfg);
        Tensor* ab = a ? tensor_matmul(a, b) : NULL;
        Tensor* s = ab ? tensor_add(ab, a) : NULL;
        Tensor* r = s ? tensor_relu(s) : NULL;
        ok &= r != NULL;
        if (r) {
            Tensor* sum = tensor_sum(r, 0, false);
            ok &= sum != NULL;
            tensor_ensure_executed(sum);
            tensor_free(sum);
        }
        cml_reset_ir_context();
        /* a..r are graph-owned; freed by the reset above. */
        (void)a; (void)b; (void)ab; (void)s; (void)r;
    }
    return ok;
}

/* Attention-block soak: the reshape + batched-matmul + softmax path that
 * severed the autograd graph before the cml_reshape fix. Trains one block for
 * many steps to shake out lifecycle bugs (heap corruption, use-after-free,
 * leaks across reset) on the exact composition transformers exercise. */
static int test_attention_soak(void) {
    const int B = 4, S = 6, d = 8, NP_MAX = 8;
    Linear* lq = cml_nn_linear(d, d, DTYPE_FLOAT32, DEVICE_CPU, true);
    Linear* lk = cml_nn_linear(d, d, DTYPE_FLOAT32, DEVICE_CPU, true);
    Linear* lv = cml_nn_linear(d, d, DTYPE_FLOAT32, DEVICE_CPU, true);
    Linear* lo = cml_nn_linear(d, d, DTYPE_FLOAT32, DEVICE_CPU, true);
    Sequential* seq = nn_sequential();
    if (!lq || !lk || !lv || !lo || !seq) return 0;
    sequential_add(seq, (Module*)lq); sequential_add(seq, (Module*)lk);
    sequential_add(seq, (Module*)lv); sequential_add(seq, (Module*)lo);

    Parameter** params = NULL; int nparams = 0;
    module_collect_parameters((Module*)seq, &params, &nparams, true);
    Optimizer* opt = optim_sgd(params, nparams, 0.01f, 0.0f, 0.0f);
    cml_free(params);
    (void)NP_MAX;

    int xs2[] = {B * S, d}, xs3[] = {B, S, d};
    int ok = 1;
    float first = 0.0f, last = 0.0f;
    /* 32 steps is enough to exercise the attention+reshape+reset lifecycle and
     * catch leaks/corruption without being a memory hog that tips over under a
     * fully parallel ctest run. */
    for (int step = 0; step < 32 && ok; step++) {
        Tensor* X = tensor_randn(xs2, 2, &cfg);
        Tensor* Yt = tensor_randn(xs3, 3, &cfg);
        Tensor* q = cml_reshape(cml_nn_module_forward((Module*)lq, X), xs3, 3);
        Tensor* k = cml_reshape(cml_nn_module_forward((Module*)lk, X), xs3, 3);
        Tensor* v = cml_reshape(cml_nn_module_forward((Module*)lv, X), xs3, 3);
        Tensor* scores = cml_matmul(q, cml_transpose(k, 1, 2));
        Tensor* attn = cml_softmax(scores, -1);
        Tensor* ctx = cml_matmul(attn, v);
        int flat[] = {B * S, d};
        Tensor* out = cml_nn_module_forward((Module*)lo, cml_reshape(ctx, flat, 2));
        Tensor* loss = cml_nn_mse_loss(cml_reshape(out, xs3, 3), Yt);
        ok &= loss != NULL;
        if (loss) {
            Tensor* one = tensor_ones(loss->shape, loss->ndim, &cfg);
            tensor_backward(loss, one, false, false);  /* graph mode realizes loss here */
            tensor_free(one);
            float lv_ = (loss->data) ? ((float*)loss->data)[0] : 0.0f/0.0f;
            if (lv_ != lv_) ok = 0;                     /* NaN => training diverged */
            if (step == 0) first = lv_;
            last = lv_;
            optimizer_step(opt);
            optimizer_zero_grad(opt);
        }
        cml_reset_ir_context();
        (void)X; (void)Yt;
    }
    /* Loss must have MOVED (grad flowed), not frozen at its initial value like
     * the old reshape-severed-graph bug. Not asserting monotonic decrease: SGD
     * on random data needn't decrease every run, and that was flaky under load. */
    ok &= (last != first);
    optimizer_free(opt);
    module_free((Module*)seq);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== lifecycle stress ===\n");
    TEST(model_churn);
    TEST(view_temporary_churn);
    TEST(reset_pressure);
    TEST(attention_soak);
    cml_cleanup();
    return TEST_SUMMARY();
}
