/*
 * Autograd gradcheck gate: central finite-difference verification of analytic
 * gradients for the core differentiable ops. For loss L = sum(op(x)),
 *   dL/dx[i] ~= (L(x+eps@i) - L(x-eps@i)) / (2 eps)
 * must match x->grad[i] within tolerance.
 *
 * Deliberately covers only ops/shapes that don't hit the pre-existing
 * batchnorm2d graph-cache crash (see grad_check.c) — this is a stable gate.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cml.h"
#include "test_harness.h"

#define EPS 1e-3f
#define TOL 2e-2f

static const TensorConfig F32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static Tensor* leaf(const float* v, int* shape, int ndim, bool rg) {
    Tensor* t = tensor_from_data((void*)v, shape, ndim, &F32);
    if (t)
        t->requires_grad = rg;
    return t;
}
static float sum_all(Tensor* t) {
    tensor_ensure_executed(t);
    const float* d = (const float*)t->data;
    float s        = 0;
    for (size_t i = 0; i < t->numel; i++)
        s += d[i];
    return s;
}

/* op selector for a unary loss L = sum(op(x)) */
typedef Tensor* (*unary_fn)(Tensor*);
static Tensor* op_relu(Tensor* x) { return uop_relu(x); }
static Tensor* op_sigmoid(Tensor* x) { return uop_sigmoid(x); }
static Tensor* op_tanh(Tensor* x) { return uop_tanh(x); }
static Tensor* op_exp(Tensor* x) { return uop_exp(x); }
static Tensor* op_square(Tensor* x) { return uop_mul(x, x); }
static Tensor* op_id(Tensor* x) { return uop_add(x, x); } /* 2x */

static void check_unary(const char* name, unary_fn op, float lo, float hi) {
    tests_run++;
    srand(1234);
    int shape[] = {6};
    float xd[6];
    for (int i = 0; i < 6; i++)
        xd[i] = lo + (float)rand() / (float)RAND_MAX * (hi - lo);

    Tensor* x = leaf(xd, shape, 1, true);
    Tensor* y = op(x);
    Tensor* L = uop_sum(y, &(ReduceParams){NULL, 0, false});
    tensor_ensure_executed(L);
    cml_backward(L, NULL, false, false);
    if (!x->grad) {
        printf("  FAIL: %s (null grad)\n", name);
        tensor_free(x);
        return;
    }
    tensor_ensure_executed(x->grad);

    const float* g = (const float*)x->grad->data;
    float* xdata   = (float*)x->data;
    int ok         = 1;
    for (int i = 0; i < 6; i++) {
        float orig = xdata[i];
        xdata[i]   = orig + EPS;
        Tensor* lp = uop_sum(op(x), &(ReduceParams){NULL, 0, false});
        float fp   = sum_all(lp);
        tensor_free(lp);
        xdata[i]   = orig - EPS;
        Tensor* lm = uop_sum(op(x), &(ReduceParams){NULL, 0, false});
        float fm   = sum_all(lm);
        tensor_free(lm);
        xdata[i]  = orig;
        float num = (fp - fm) / (2 * EPS);
        if (fabsf(num - g[i]) > TOL) {
            ok = 0;
            printf("    [%s i=%d] analytic=%.4f num=%.4f\n", name, i, g[i], num);
            break;
        }
    }
    if (ok) {
        tests_passed++;
        printf("  PASS: %s\n", name);
    } else {
        printf("  FAIL: %s\n", name);
    }
    tensor_free(x);
    tensor_free(y);
    tensor_free(L);
}

/* matmul grad: L = sum(x @ W), check dL/dx */
static void check_matmul(void) {
    tests_run++;
    int xs[] = {2, 3}, ws[] = {3, 2};
    float xd[6] = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f};
    float wd[6] = {1, 2, 3, 4, 5, 6};
    Tensor* x   = leaf(xd, xs, 2, true);
    Tensor* w   = leaf(wd, ws, 2, false);
    Tensor* y   = uop_matmul(x, w);
    Tensor* L   = uop_sum(y, &(ReduceParams){NULL, 0, false});
    tensor_ensure_executed(L);
    cml_backward(L, NULL, false, false);
    int ok = x->grad != NULL;
    if (ok) {
        tensor_ensure_executed(x->grad);
        const float* g = (const float*)x->grad->data;
        float* xdata   = (float*)x->data;
        for (int i = 0; i < 6 && ok; i++) {
            float orig = xdata[i];
            xdata[i]   = orig + EPS;
            Tensor* lp = uop_sum(uop_matmul(x, w), &(ReduceParams){NULL, 0, false});
            float fp   = sum_all(lp);
            tensor_free(lp);
            xdata[i]   = orig - EPS;
            Tensor* lm = uop_sum(uop_matmul(x, w), &(ReduceParams){NULL, 0, false});
            float fm   = sum_all(lm);
            tensor_free(lm);
            xdata[i]  = orig;
            float num = (fp - fm) / (2 * EPS);
            if (fabsf(num - g[i]) > TOL)
                ok = 0;
        }
    }
    if (ok) {
        tests_passed++;
        printf("  PASS: matmul\n");
    } else
        printf("  FAIL: matmul\n");
    tensor_free(x);
    tensor_free(w);
    tensor_free(y);
    tensor_free(L);
}

int main(void) {
    cml_init();
    printf("=== autograd gradcheck (core ops) ===\n");
    check_unary("relu", op_relu, -1.0f, 1.0f);
    check_unary("sigmoid", op_sigmoid, -2.0f, 2.0f);
    check_unary("tanh", op_tanh, -2.0f, 2.0f);
    check_unary("exp", op_exp, -1.0f, 1.0f);
    check_unary("square", op_square, -1.5f, 1.5f);
    check_unary("add_self", op_id, -1.0f, 1.0f);
    check_matmul();
    cml_cleanup();
    return TEST_SUMMARY();
}
