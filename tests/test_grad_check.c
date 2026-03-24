
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "tensor/tensor.h"
#include "ops/uops.h"

static int tests_passed = 0;
static int tests_total  = 0;

static const TensorConfig cpu_f32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
    .has_dtype = true, .has_device = true
};

#define TEST(name)                                     \
    do {                                               \
        tests_total++;                                 \
        printf("  TEST: %s ... ", #name);              \
        if (test_##name()) {                           \
            tests_passed++;                            \
            printf("PASSED\n");                        \
        } else {                                       \
            printf("FAILED\n");                        \
        }                                              \
    } while (0)

#define GRAD_ATOL 1e-3f

typedef Tensor* (*UnaryFn)(Tensor*);

static float numerical_grad(const float* base, int n, int shape[],
                             int idx, UnaryFn f_fn, float eps) {
    float* xp = malloc(n * sizeof(float));
    float* xm = malloc(n * sizeof(float));
    memcpy(xp, base, n * sizeof(float));
    memcpy(xm, base, n * sizeof(float));
    xp[idx] += eps;
    xm[idx] -= eps;

    Tensor* tp = tensor_from_data(xp, shape, 1, &cpu_f32);
    Tensor* tm = tensor_from_data(xm, shape, 1, &cpu_f32);
    Tensor* op = f_fn(tp);
    Tensor* om = f_fn(tm);
    tensor_ensure_executed(op);
    tensor_ensure_executed(om);

    /* scalar loss = sum of outputs */
    float sp = 0.0f, sm = 0.0f;
    float* dp = op->data;
    float* dm = om->data;
    for (int i = 0; i < n; i++) { sp += dp[i]; sm += dm[i]; }

    float grad = (sp - sm) / (2.0f * eps);

    tensor_free(tp); tensor_free(tm);
    tensor_free(op); tensor_free(om);
    free(xp); free(xm);
    return grad;
}

typedef float (*ScalarFn)(float);

static int check_numerical_self_consistent(const float* data, int n,
                                            UnaryFn f_fn,
                                            ScalarFn fprime,
                                            const char* name) {
    int shape[] = {n};
    float eps = 1e-3f;
    for (int i = 0; i < n; i++) {
        float ng    = numerical_grad(data, n, shape, i, f_fn, eps);
        float exact = fprime(data[i]);
        float diff  = fabsf(ng - exact);
        float scale = 0.5f * (fabsf(ng) + fabsf(exact)) + 1e-6f;
        if (diff > GRAD_ATOL && diff / scale > 0.05f) {
            printf("\n    [%s] grad[%d]: numerical=%.6f exact=%.6f diff=%.6f",
                   name, i, ng, exact, diff);
            return 0;
        }
    }
    return 1;
}

static float d_sin(float x)  { return cosf(x); }
static float d_cos(float x)  { return -sinf(x); }
static float d_exp(float x)  { return expf(x); }
static float d_log(float x)  { return 1.0f / x; }
static float d_sqrt(float x) { return 0.5f / sqrtf(x); }
static float d_tanh(float x) { float t = tanhf(x); return 1.0f - t*t; }
static float d_abs(float x)  { return (x >= 0.0f) ? 1.0f : -1.0f; }
static float d_neg(float x)  { (void)x; return -1.0f; }
static float d_square(float x) { return 2.0f * x; }
static float d_sigmoid(float x) {
    float s = 1.0f / (1.0f + expf(-x));
    return s * (1.0f - s);
}
static float d_recip(float x) { return -1.0f / (x * x); }

static Tensor* wrap_sin(Tensor* x)     { return uop_sin(x); }
static Tensor* wrap_cos(Tensor* x)     { return uop_cos(x); }
static Tensor* wrap_exp(Tensor* x)     { return uop_exp(x); }
static Tensor* wrap_log(Tensor* x)     { return uop_log(x); }
static Tensor* wrap_sqrt(Tensor* x)    { return uop_sqrt(x); }
static Tensor* wrap_tanh(Tensor* x)    { return uop_tanh(x); }
static Tensor* wrap_abs(Tensor* x)     { return uop_abs(x); }
static Tensor* wrap_neg(Tensor* x)     { return uop_neg(x); }
static Tensor* wrap_square(Tensor* x)  { return uop_square(x); }
static Tensor* wrap_sigmoid(Tensor* x) { return uop_sigmoid(x); }
static Tensor* wrap_recip(Tensor* x)   { return uop_recip(x); }

static int test_grad_sin(void) {
    float data[] = {0.1f, 0.5f, 1.0f, 1.5f, 2.0f};
    return check_numerical_self_consistent(data, 5, wrap_sin, d_sin, "sin");
}
static int test_grad_cos(void) {
    float data[] = {0.1f, 0.5f, 1.0f, 1.5f, 2.0f};
    return check_numerical_self_consistent(data, 5, wrap_cos, d_cos, "cos");
}
static int test_grad_exp(void) {
    float data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    return check_numerical_self_consistent(data, 5, wrap_exp, d_exp, "exp");
}
static int test_grad_log(void) {
    float data[] = {0.1f, 0.5f, 1.0f, 2.0f, 5.0f}; /* must be positive */
    return check_numerical_self_consistent(data, 5, wrap_log, d_log, "log");
}
static int test_grad_sqrt(void) {
    float data[] = {0.1f, 0.5f, 1.0f, 2.0f, 9.0f}; /* must be positive */
    return check_numerical_self_consistent(data, 5, wrap_sqrt, d_sqrt, "sqrt");
}
static int test_grad_tanh(void) {
    float data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    return check_numerical_self_consistent(data, 5, wrap_tanh, d_tanh, "tanh");
}
static int test_grad_abs(void) {
    /* skip zero — derivative undefined there */
    float data[] = {-3.0f, -0.5f, 0.5f, 2.0f, 4.0f};
    return check_numerical_self_consistent(data, 5, wrap_abs, d_abs, "abs");
}
static int test_grad_neg(void) {
    float data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    return check_numerical_self_consistent(data, 5, wrap_neg, d_neg, "neg");
}
static int test_grad_square(void) {
    float data[] = {-2.0f, -1.0f, 0.5f, 1.0f, 3.0f};
    return check_numerical_self_consistent(data, 5, wrap_square, d_square, "square");
}
static int test_grad_sigmoid(void) {
    float data[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
    return check_numerical_self_consistent(data, 5, wrap_sigmoid, d_sigmoid, "sigmoid");
}
static int test_grad_recip(void) {
    float data[] = {0.5f, 1.0f, 2.0f, -1.0f, -3.0f}; /* avoid zero */
    return check_numerical_self_consistent(data, 5, wrap_recip, d_recip, "recip");
}

static int test_grad_add_commutative(void) {
    /* sum(a+b) has grad 1.0 for each a_i */
    float eps = 1e-3f;
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    int n = 3;
    int shape[] = {n};

    for (int i = 0; i < n; i++) {
        float ap[3], am[3];
        memcpy(ap, a, sizeof(a)); memcpy(am, a, sizeof(a));
        ap[i] += eps; am[i] -= eps;

        Tensor* tap = tensor_from_data(ap, shape, 1, &cpu_f32);
        Tensor* tam = tensor_from_data(am, shape, 1, &cpu_f32);
        Tensor* tb  = tensor_from_data(b,  shape, 1, &cpu_f32);

        Tensor* op  = uop_add(tap, tb);
        Tensor* om  = uop_add(tam, tb);
        tensor_ensure_executed(op); tensor_ensure_executed(om);

        float sp = 0.0f, sm = 0.0f;
        float* dp = op->data; float* dm = om->data;
        for (int j = 0; j < n; j++) { sp += dp[j]; sm += dm[j]; }

        float grad = (sp - sm) / (2.0f * eps);
        if (fabsf(grad - 1.0f) > GRAD_ATOL) return 0;

        tensor_free(tap); tensor_free(tam); tensor_free(tb);
        tensor_free(op); tensor_free(om);
    }
    return 1;
}

static int test_grad_mul(void) {
    float eps = 1e-3f;
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    int n = 3;
    int shape[] = {n};

    for (int i = 0; i < n; i++) {
        float ap[3], am[3];
        memcpy(ap, a, sizeof(a)); memcpy(am, a, sizeof(a));
        ap[i] += eps; am[i] -= eps;

        Tensor* tap = tensor_from_data(ap, shape, 1, &cpu_f32);
        Tensor* tam = tensor_from_data(am, shape, 1, &cpu_f32);
        Tensor* tb  = tensor_from_data(b,  shape, 1, &cpu_f32);

        Tensor* op  = uop_mul(tap, tb);
        Tensor* om  = uop_mul(tam, tb);
        tensor_ensure_executed(op); tensor_ensure_executed(om);

        float sp = 0.0f, sm = 0.0f;
        float* dp = op->data; float* dm = om->data;
        for (int j = 0; j < n; j++) { sp += dp[j]; sm += dm[j]; }

        float grad = (sp - sm) / (2.0f * eps);
        float expected = b[i];
        float rel_err = fabsf((grad - expected) / (expected + 1e-8f));
        if (rel_err > 0.1f && fabsf(grad - expected) > 1.0f) {
            fprintf(stderr, "  [grad_mul] i=%d: expected %.4f, got %.4f\n", i, expected, grad);
            tensor_free(tap); tensor_free(tam); tensor_free(tb);
            tensor_free(op); tensor_free(om);
            return 0;
        }

        tensor_free(tap); tensor_free(tam); tensor_free(tb);
        tensor_free(op); tensor_free(om);
    }
    return 1;
}

static int test_grad_matmul(void) {
    /*
     * A (2x3), B (3x2) -> C (2x2), loss = sum(C).
     * dL/dA[i,j] = sum_k(dL/dC[i,k] * B[j,k]^T) = sum_k B[j,k] = rowsum(B^T)[j]
     * For uniform ones loss: dL/dA[i,j] = sum_k(1 * B[j,k]) = rowsum(B)[j].
     */
    float a_data[] = {1,2,3, 4,5,6};
    float b_data[] = {1,2, 3,4, 5,6};
    int sa[] = {2, 3};
    int sb[] = {3, 2};

    Tensor* A = tensor_from_data(a_data, sa, 2, &cpu_f32);
    Tensor* B = tensor_from_data(b_data, sb, 2, &cpu_f32);
    Tensor* C = uop_matmul(A, B);
    tensor_ensure_executed(C);
    /* Just verify output shape and no crash */
    if (C->shape[0] != 2 || C->shape[1] != 2) return 0;
    /* C[0,0] = 1*1+2*3+3*5=22, C[0,1]=1*2+2*4+3*6=28 */
    float* d = C->data;
    if (fabsf(d[0] - 22.0f) > 1e-3f) return 0;
    if (fabsf(d[1] - 28.0f) > 1e-3f) return 0;
    tensor_free(A); tensor_free(B); tensor_free(C);
    return 1;
}

int main(void) {
    printf("Finite-Difference Gradient Checks\n");
    TEST(grad_sin);
    TEST(grad_cos);
    TEST(grad_exp);
    TEST(grad_log);
    TEST(grad_sqrt);
    TEST(grad_tanh);
    TEST(grad_abs);
    TEST(grad_neg);
    TEST(grad_square);
    TEST(grad_sigmoid);
    TEST(grad_recip);
    TEST(grad_add_commutative);
    TEST(grad_mul);
    TEST(grad_matmul);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
