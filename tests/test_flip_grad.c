/* Gradient of flip on the eager backward engine.
 *
 * flip has a graph-mode VJP but had no rule in the CPU eager backward switch,
 * so flip(x).backward() under GRAD_MODE=eager silently dropped the gradient.
 * flip is its own inverse, so dL/dx = flip(dL/dy) along the same axis.
 */
#include <stdlib.h>
#include <math.h>
#include "cml.h"
#include "test_harness.h"

static TensorConfig cfg = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

/* 1-D: y = flip(x); loss = sum(y * w) => dL/dx = flip(w). */
static int test_flip_grad_1d(void) {
    float xd[4]  = {1, 2, 3, 4};
    float wd[4]  = {10, 20, 30, 40};
    int shape[1] = {4};
    Tensor* x    = tensor_from_data(xd, shape, 1, &cfg);
    Tensor* w    = tensor_from_data(wd, shape, 1, &cfg);
    if (!x || !w)
        return 0;
    cml_set_requires_grad(x, true);

    Tensor* y    = cml_flip(x, 0); /* [4,3,2,1] */
    Tensor* prod = cml_mul(y, w);
    Tensor* loss = cml_sum(prod, 0, false);
    tensor_backward(loss, NULL, false, false);

    Tensor* g = tensor_get_grad(x);
    int ok    = g != NULL;
    if (ok) {
        tensor_ensure_executed(g);
        /* dL/dx = flip(w) = [40,30,20,10] */
        float exp[4] = {40, 30, 20, 10};
        for (int i = 0; i < 4 && ok; i++)
            ok = fabsf(tensor_get_float(g, i) - exp[i]) < 1e-5f;
    }
    return ok;
}

/* 2-D flip along dim 1 to exercise the strided scatter path. */
static int test_flip_grad_2d(void) {
    float xd[6]  = {1, 2, 3, 4, 5, 6}; /* [[1,2,3],[4,5,6]] */
    float wd[6]  = {1, 2, 3, 4, 5, 6};
    int shape[2] = {2, 3};
    Tensor* x    = tensor_from_data(xd, shape, 2, &cfg);
    Tensor* w    = tensor_from_data(wd, shape, 2, &cfg);
    if (!x || !w)
        return 0;
    cml_set_requires_grad(x, true);

    Tensor* y    = cml_flip(x, 1); /* row-reverse */
    Tensor* prod = cml_mul(y, w);
    Tensor* loss = cml_sum(prod, 0, false); /* reduce all */
    tensor_backward(loss, NULL, false, false);

    Tensor* g = tensor_get_grad(x);
    int ok    = g != NULL;
    if (ok) {
        tensor_ensure_executed(g);
        /* grad[r][c] = w[r][ncol-1-c] => [[3,2,1],[6,5,4]] */
        float exp[6] = {3, 2, 1, 6, 5, 4};
        for (int i = 0; i < 6 && ok; i++)
            ok = fabsf(tensor_get_float(g, i) - exp[i]) < 1e-5f;
    }
    return ok;
}

int main(void) {
    setenv("GRAD_MODE", "eager", 1); /* force the CPU eager backward path */
    cml_init();
    printf("=== flip gradient (eager engine) ===\n");
    TEST(flip_grad_1d);
    TEST(flip_grad_2d);
    cml_cleanup();
    return TEST_SUMMARY();
}
