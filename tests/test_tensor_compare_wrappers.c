/* Forward-value and (for the differentiable view ops) gradient coverage for the
 * public tensor_* op wrappers added for torch parity:
 *   comparison  : lt le gt ge eq ne
 *   logical     : and or not xor
 *   bitwise     : and or xor not
 *   min/max     : minimum maximum
 *   views       : permute expand narrow index_select unfold movedim swapaxes
 *
 * The comparison/logical/bitwise families are non-differentiable (torch stops
 * the gradient there), so they are checked for forward values only. The view
 * ops are checked against a central finite difference of their own forward. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "tensor/tensor_compare.h"
#include "tensor/tensor_views.h"
#include "test_harness.h"

static TensorConfig cfg = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

/* ---- forward helpers ---------------------------------------------------- */

static Tensor* mk(const float* data, int* shape, int ndim) {
    return tensor_from_data(data, shape, ndim, &cfg);
}

/* Realize `y` and compare its flat values to `expect` (length n). */
static int values_match(Tensor* y, const float* expect, int n) {
    if (!y)
        return 0;
    tensor_ensure_executed(y);
    if ((int)y->numel != n) {
        printf("(numel %zu != %d) ", y->numel, n);
        return 0;
    }
    /* read via tensor_get_float so bool/int comparison outputs convert correctly */
    for (int i = 0; i < n; i++) {
        float got = tensor_get_float(y, (size_t)i);
        if (fabsf(got - expect[i]) > 1e-5f) {
            printf("(i=%d got %.4f want %.4f) ", i, got, expect[i]);
            return 0;
        }
    }
    return 1;
}

typedef Tensor* (*BinFn)(Tensor*, Tensor*);
typedef Tensor* (*UnFn)(Tensor*);

static int check_binary(BinFn fn, const float* a, const float* b, const float* expect, int n) {
    cml_reset_ir_context();
    int sh[1]  = {n};
    Tensor* ta = mk(a, sh, 1);
    Tensor* tb = mk(b, sh, 1);
    Tensor* y  = fn(ta, tb);
    int ok     = values_match(y, expect, n);
    tensor_free(ta);
    tensor_free(tb);
    if (y)
        tensor_free(y);
    cml_reset_ir_context();
    return ok;
}

static int check_unary(UnFn fn, const float* a, const float* expect, int n) {
    cml_reset_ir_context();
    int sh[1]  = {n};
    Tensor* ta = mk(a, sh, 1);
    Tensor* y  = fn(ta);
    int ok     = values_match(y, expect, n);
    tensor_free(ta);
    if (y)
        tensor_free(y);
    cml_reset_ir_context();
    return ok;
}

/* ---- comparison / logical / bitwise / min-max --------------------------- */

static const float A4[4] = {1.0f, 2.0f, 3.0f, 2.0f};
static const float B4[4] = {2.0f, 2.0f, 1.0f, 3.0f};

static int test_lt(void) {
    float e[4] = {1, 0, 0, 1};
    return check_binary(tensor_lt, A4, B4, e, 4);
}
static int test_le(void) {
    float e[4] = {1, 1, 0, 1};
    return check_binary(tensor_le, A4, B4, e, 4);
}
static int test_gt(void) {
    float e[4] = {0, 0, 1, 0};
    return check_binary(tensor_gt, A4, B4, e, 4);
}
static int test_ge(void) {
    float e[4] = {0, 1, 1, 0};
    return check_binary(tensor_ge, A4, B4, e, 4);
}
static int test_eq(void) {
    float e[4] = {0, 1, 0, 0};
    return check_binary(tensor_eq, A4, B4, e, 4);
}
static int test_ne(void) {
    float e[4] = {1, 0, 1, 1};
    return check_binary(tensor_ne, A4, B4, e, 4);
}

static const float LA[4] = {0.0f, 0.0f, 3.0f, 5.0f};
static const float LB[4] = {0.0f, 2.0f, 0.0f, 4.0f};

static int test_logical_and(void) {
    float e[4] = {0, 0, 0, 1};
    return check_binary(tensor_logical_and, LA, LB, e, 4);
}
static int test_logical_or(void) {
    float e[4] = {0, 1, 1, 1};
    return check_binary(tensor_logical_or, LA, LB, e, 4);
}
static int test_logical_xor(void) {
    float e[4] = {0, 1, 1, 0};
    return check_binary(tensor_logical_xor, LA, LB, e, 4);
}
static int test_logical_not(void) {
    float e[4] = {1, 1, 0, 0};
    return check_unary(tensor_logical_not, LA, e, 4);
}

static const float IA[4] = {6.0f, 5.0f, 12.0f, 1.0f};
static const float IB[4] = {3.0f, 3.0f, 10.0f, 2.0f};

static int test_bitwise_and(void) {
    float e[4] = {2, 1, 8, 0}; /* 6&3, 5&3, 12&10, 1&2 */
    return check_binary(tensor_bitwise_and, IA, IB, e, 4);
}
static int test_bitwise_or(void) {
    float e[4] = {7, 7, 14, 3}; /* 6|3, 5|3, 12|10, 1|2 */
    return check_binary(tensor_bitwise_or, IA, IB, e, 4);
}
static int test_bitwise_xor(void) {
    float e[4] = {5, 6, 6, 3}; /* 6^3, 5^3, 12^10, 1^2 */
    return check_binary(tensor_bitwise_xor, IA, IB, e, 4);
}
static int test_bitwise_not(void) {
    float e[4] = {-7, -6, -13, -2}; /* ~x == -x-1 */
    return check_unary(tensor_bitwise_not, IA, e, 4);
}

static int test_minimum(void) {
    float e[4] = {1, 2, 1, 2};
    return check_binary(tensor_minimum, A4, B4, e, 4);
}
static int test_maximum(void) {
    float e[4] = {2, 2, 3, 3};
    return check_binary(tensor_maximum, A4, B4, e, 4);
}

/* ---- view op forward shape/value checks --------------------------------- */

static int test_permute_forward(void) {
    cml_reset_ir_context();
    float d[6]  = {1, 2, 3, 4, 5, 6}; /* [2,3] row-major */
    int sh[2]   = {2, 3};
    Tensor* x   = mk(d, sh, 2);
    int perm[2] = {1, 0};
    Tensor* y   = tensor_permute(x, perm, 2);
    float e[6]  = {1, 4, 2, 5, 3, 6}; /* [3,2] */
    int ok = y && y->ndim == 2 && y->shape[0] == 3 && y->shape[1] == 2 && values_match(y, e, 6);
    tensor_free(x);
    if (y)
        tensor_free(y);
    cml_reset_ir_context();
    return ok;
}

static int test_expand_forward(void) {
    cml_reset_ir_context();
    float d[3] = {7, 8, 9};
    int sh[2]  = {1, 3};
    Tensor* x  = mk(d, sh, 2);
    int ns[2]  = {3, 3};
    Tensor* y  = tensor_expand(x, ns, 2);
    float e[9] = {7, 8, 9, 7, 8, 9, 7, 8, 9};
    int ok     = y && y->shape[0] == 3 && y->shape[1] == 3 && values_match(y, e, 9);
    tensor_free(x);
    if (y)
        tensor_free(y);
    cml_reset_ir_context();
    return ok;
}

static int test_narrow_forward(void) {
    cml_reset_ir_context();
    float d[6] = {1, 2, 3, 4, 5, 6};
    int sh[2]  = {2, 3};
    Tensor* x  = mk(d, sh, 2);
    Tensor* y  = tensor_narrow(x, 1, 1, 2); /* cols [1,3) */
    float e[4] = {2, 3, 5, 6};
    int ok     = y && y->shape[0] == 2 && y->shape[1] == 2 && values_match(y, e, 4);
    tensor_free(x);
    if (y)
        tensor_free(y);
    cml_reset_ir_context();
    return ok;
}

static int test_index_select_forward(void) {
    cml_reset_ir_context();
    float d[6]   = {1, 2, 3, 4, 5, 6};
    int sh[2]    = {2, 3};
    Tensor* x    = mk(d, sh, 2);
    float idx[2] = {0, 2};
    int ish[1]   = {2};
    Tensor* ix   = mk(idx, ish, 1);
    Tensor* y    = tensor_index_select(x, 1, ix); /* cols 0 and 2 */
    float e[4]   = {1, 3, 4, 6};
    int ok       = y && y->shape[0] == 2 && y->shape[1] == 2 && values_match(y, e, 4);
    tensor_free(x);
    tensor_free(ix);
    if (y)
        tensor_free(y);
    cml_reset_ir_context();
    return ok;
}

static int test_unfold_forward(void) {
    cml_reset_ir_context();
    float d[5] = {1, 2, 3, 4, 5};
    int sh[1]  = {5};
    Tensor* x  = mk(d, sh, 1);
    Tensor* y  = tensor_unfold(x, 0, 3, 1); /* windows [1,2,3],[2,3,4],[3,4,5] */
    float e[9] = {1, 2, 3, 2, 3, 4, 3, 4, 5};
    int ok     = y && y->ndim == 2 && y->shape[0] == 3 && y->shape[1] == 3 && values_match(y, e, 9);
    tensor_free(x);
    if (y)
        tensor_free(y);
    cml_reset_ir_context();
    return ok;
}

static int test_movedim_forward(void) {
    cml_reset_ir_context();
    float d[6] = {1, 2, 3, 4, 5, 6};
    int sh[2]  = {2, 3};
    Tensor* x  = mk(d, sh, 2);
    Tensor* y  = tensor_movedim(x, 0, 1); /* == transpose for 2D */
    float e[6] = {1, 4, 2, 5, 3, 6};
    int ok     = y && y->shape[0] == 3 && y->shape[1] == 2 && values_match(y, e, 6);
    tensor_free(x);
    if (y)
        tensor_free(y);
    cml_reset_ir_context();
    return ok;
}

static int test_swapaxes_forward(void) {
    cml_reset_ir_context();
    float d[6] = {1, 2, 3, 4, 5, 6};
    int sh[2]  = {2, 3};
    Tensor* x  = mk(d, sh, 2);
    Tensor* y  = tensor_swapaxes(x, 0, 1);
    float e[6] = {1, 4, 2, 5, 3, 6};
    int ok     = y && y->shape[0] == 3 && y->shape[1] == 2 && values_match(y, e, 6);
    tensor_free(x);
    if (y)
        tensor_free(y);
    cml_reset_ir_context();
    return ok;
}

/* ---- gradient checks for the differentiable view ops -------------------- */

static float forward_sum(UnFn op, const float* data, int* shape, int ndim, int n) {
    cml_reset_ir_context();
    Tensor* x = mk(data, shape, ndim);
    Tensor* y = op(x);
    if (!y) {
        tensor_free(x);
        cml_reset_ir_context();
        return NAN;
    }
    ReduceParams rp = {0};
    Tensor* s       = uop_sum(y, &rp);
    tensor_ensure_executed(s);
    float v = ((const float*)tensor_data_ptr(s))[0];
    tensor_free(x);
    tensor_free(y);
    tensor_free(s);
    cml_reset_ir_context();
    (void)n;
    return v;
}

static int grad_check(const char* name, UnFn op, const float* data, int* shape, int ndim) {
    int n = 1;
    for (int i = 0; i < ndim; i++)
        n *= shape[i];

    cml_reset_ir_context();
    Tensor* x        = mk(data, shape, ndim);
    x->requires_grad = true;
    Tensor* y        = op(x);
    if (!y) {
        printf("(%s: op returned NULL) ", name);
        tensor_free(x);
        cml_reset_ir_context();
        return 0;
    }
    ReduceParams rp = {0};
    Tensor* s       = uop_sum(y, &rp);
    tensor_backward(s, NULL, false, false);
    if (!x->grad) {
        printf("(%s: no grad) ", name);
        tensor_free(x);
        tensor_free(y);
        tensor_free(s);
        cml_reset_ir_context();
        return 0;
    }
    float ana[64];
    memcpy(ana, (const float*)tensor_data_ptr(x->grad), sizeof(float) * (size_t)n);
    tensor_free(x);
    tensor_free(y);
    tensor_free(s);
    cml_reset_ir_context();

    float buf[64];
    memcpy(buf, data, sizeof(float) * (size_t)n);
    const float eps = 1e-2f;
    for (int i = 0; i < n; i++) {
        float saved = buf[i];
        buf[i]      = saved + eps;
        float fp    = forward_sum(op, buf, shape, ndim, n);
        buf[i]      = saved - eps;
        float fm    = forward_sum(op, buf, shape, ndim, n);
        buf[i]      = saved;
        if (isnan(fp) || isnan(fm))
            return 0;
        float num = (fp - fm) / (2.0f * eps);
        if (fabsf(num - ana[i]) > 1e-2f) {
            printf("(%s: i=%d num=%.4f ana=%.4f) ", name, i, num, ana[i]);
            return 0;
        }
    }
    return 1;
}

static Tensor* w_permute(Tensor* a) {
    int p[2] = {1, 0};
    return tensor_permute(a, p, 2);
}
static Tensor* w_expand(Tensor* a) {
    int s[2] = {4, 3};
    return tensor_expand(a, s, 2);
}
static Tensor* w_narrow(Tensor* a) { return tensor_narrow(a, 1, 1, 2); }
static Tensor* w_index_select(Tensor* a) {
    float idx[3] = {0, 2, 2};
    int ish[1]   = {3};
    Tensor* ix   = mk(idx, ish, 1);
    return tensor_index_select(a, 1, ix);
}
static Tensor* w_unfold(Tensor* a) { return tensor_unfold(a, 1, 2, 1); }
static Tensor* w_movedim(Tensor* a) { return tensor_movedim(a, 0, 1); }
static Tensor* w_swapaxes(Tensor* a) { return tensor_swapaxes(a, 0, 1); }

static const float G6[6] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};

static int test_permute_grad(void) {
    int sh[2] = {2, 3};
    return grad_check("permute", w_permute, G6, sh, 2);
}
static int test_expand_grad(void) {
    float d[3] = {1.0f, 2.0f, 3.0f};
    int sh[2]  = {1, 3};
    return grad_check("expand", w_expand, d, sh, 2);
}
static int test_narrow_grad(void) {
    int sh[2] = {2, 3};
    return grad_check("narrow", w_narrow, G6, sh, 2);
}
static int test_index_select_grad(void) {
    int sh[2] = {2, 3};
    return grad_check("index_select", w_index_select, G6, sh, 2);
}
static int test_unfold_grad(void) {
    int sh[2] = {2, 3};
    return grad_check("unfold", w_unfold, G6, sh, 2);
}
static int test_movedim_grad(void) {
    int sh[2] = {2, 3};
    return grad_check("movedim", w_movedim, G6, sh, 2);
}
static int test_swapaxes_grad(void) {
    int sh[2] = {2, 3};
    return grad_check("swapaxes", w_swapaxes, G6, sh, 2);
}

int main(void) {
    printf("=== tensor_* comparison / logical / bitwise / view wrappers ===\n");

    TEST(lt);
    TEST(le);
    TEST(gt);
    TEST(ge);
    TEST(eq);
    TEST(ne);

    TEST(logical_and);
    TEST(logical_or);
    TEST(logical_xor);
    TEST(logical_not);

    TEST(bitwise_and);
    TEST(bitwise_or);
    TEST(bitwise_xor);
    TEST(bitwise_not);

    TEST(minimum);
    TEST(maximum);

    TEST(permute_forward);
    TEST(expand_forward);
    TEST(narrow_forward);
    TEST(index_select_forward);
    TEST(unfold_forward);
    TEST(movedim_forward);
    TEST(swapaxes_forward);

    TEST(permute_grad);
    TEST(expand_grad);
    TEST(narrow_grad);
    TEST(index_select_grad);
    TEST(unfold_grad);
    TEST(movedim_grad);
    TEST(swapaxes_grad);

    return TEST_SUMMARY();
}
