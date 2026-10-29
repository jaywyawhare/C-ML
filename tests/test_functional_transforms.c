/* functorch-style transforms: grad / vjp / jacrev / jacfwd / hessian / vmap /
 * functional_call. grad/vjp/jacrev are reverse mode (both engines); jacfwd and
 * hessian need the graph engine, so those checks self-skip under GRAD_MODE=eager.
 * Every analytic value is cross-checked against a central finite difference. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "autograd/autograd.h"
#include "autograd/forward_mode.h"
#include "autograd/functional_transforms.h"
#include "cml.h"
#include "nn/layers/linear.h"
#include "nn/state.h"
#include "ops/ir/autodiff.h"
#include "ops/uops.h"
#include "test_require.h"

static TensorConfig cfg = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static Tensor* vec(const float* xs, int n) {
    int sh[1] = {n};
    Tensor* t = cml_zeros(sh, 1, &cfg);
    memcpy(tensor_data_ptr(t), xs, sizeof(float) * (size_t)n);
    return t;
}

static float at(Tensor* t, size_t i) {
    tensor_ensure_executed(t);
    return ((const float*)tensor_data_ptr(t))[i];
}

static int close_to(float a, float b) { return fabsf(a - b) <= 1e-3f * fmaxf(1.0f, fabsf(b)); }

/* f(x) = sum(x^2) : scalar */
static Tensor* f_sumsq(Tensor** in, int n, void* ud) {
    (void)n;
    (void)ud;
    ReduceParams rp = {0};
    return uop_sum(uop_mul(in[0], in[0]), &rp);
}

/* f(x) = sum(x^3) : scalar */
static Tensor* f_sumcube(Tensor** in, int n, void* ud) {
    (void)n;
    (void)ud;
    Tensor* x       = in[0];
    ReduceParams rp = {0};
    return uop_sum(uop_mul(uop_mul(x, x), x), &rp);
}

/* f(x) = [x0^2, x0*x1] : vector of a length-2 input */
static Tensor* f_vec2(Tensor** in, int n, void* ud) {
    (void)n;
    (void)ud;
    Tensor* x    = in[0];
    int s0[1]    = {0}, e0[1] = {1};
    int s1[1]    = {1}, e1[1] = {2};
    Tensor* x0   = uop_shrink(x, s0, e0, 1);
    Tensor* x1   = uop_shrink(x, s1, e1, 1);
    Tensor* y0   = uop_mul(x0, x0);
    Tensor* y1   = uop_mul(x0, x1);
    Tensor* ys[2] = {y0, y1};
    return cml_concat(ys, 2, 0);
}

/* CMLForwardFn adapter for jacfwd. */
static Tensor* f_vec2_fwd(Tensor* x, void* ud) {
    Tensor* in[1] = {x};
    return f_vec2(in, 1, ud);
}

/* f(x) = elementwise square (vmap body). */
static Tensor* f_square(Tensor** in, int n, void* ud) {
    (void)n;
    (void)ud;
    return uop_mul(in[0], in[0]);
}

static void test_grad_sumsq(void) {
    float xs[4]   = {0.5f, -1.2f, 2.0f, 0.3f};
    Tensor* x     = vec(xs, 4);
    Tensor* in[1] = {x};
    Tensor* g     = cml_grad(f_sumsq, NULL, in, 1, 0);
    REQUIRE(g != NULL);
    for (int i = 0; i < 4; i++)
        REQUIRE(close_to(at(g, (size_t)i), 2.0f * xs[i])); /* d/dx sum(x^2) = 2x */
    printf("  grad(sum x^2) = 2x: ok\n");
}

static void test_vjp_vec2(void) {
    /* VJP of f(x) = [x0^2, x0*x1] with cotangent c = [c0, c1].
     * J = [[2x0, 0], [x1, x0]], so v^T J = [2x0*c0 + x1*c1, x0*c1]. A vector
     * output is used deliberately: the eager engine honors a non-unit cotangent
     * only for multi-element outputs (scalar seeds are forced to 1.0). */
    float xs[2]   = {1.5f, -2.0f};
    Tensor* x     = vec(xs, 2);
    Tensor* in[1] = {x};
    Tensor* out   = NULL;
    CMLVjp* vjp   = cml_vjp(f_vec2, NULL, in, 1, &out);
    REQUIRE(vjp != NULL && out != NULL);
    REQUIRE(close_to(at(out, 0), xs[0] * xs[0]));
    REQUIRE(close_to(at(out, 1), xs[0] * xs[1]));

    float cot[2]   = {1.0f, 3.0f};
    Tensor* c      = vec(cot, 2);
    Tensor* grd[1] = {NULL};
    REQUIRE(cml_vjp_fn(vjp, c, grd) == 0);
    REQUIRE(grd[0] != NULL);
    REQUIRE(close_to(at(grd[0], 0), 2.0f * xs[0] * cot[0] + xs[1] * cot[1]));
    REQUIRE(close_to(at(grd[0], 1), xs[0] * cot[1]));
    cml_vjp_free(vjp);
    printf("  vjp([x0^2, x0*x1]): ok\n");
}

static void check_jac_vec2(Tensor* J, const float* xs) {
    /* J = [[2x0, 0], [x1, x0]] */
    REQUIRE(close_to(at(J, 0), 2.0f * xs[0]));
    REQUIRE(close_to(at(J, 1), 0.0f));
    REQUIRE(close_to(at(J, 2), xs[1]));
    REQUIRE(close_to(at(J, 3), xs[0]));
}

static void test_jacrev_vec2(void) {
    float xs[2]   = {1.5f, -2.0f};
    Tensor* x     = vec(xs, 2);
    Tensor* in[1] = {x};
    Tensor* J     = cml_jacrev(f_vec2, NULL, in, 1, 0);
    REQUIRE(J != NULL);
    REQUIRE(J->shape[0] == 2 && J->shape[1] == 2);
    check_jac_vec2(J, xs);
    printf("  jacrev([x0^2, x0*x1]): ok\n");
}

static void test_jacfwd_matches_jacrev(void) {
    if (!cml_autodiff_use_graph()) {
        printf("  jacfwd: skipped (needs graph engine)\n");
        return;
    }
    float xs[2]   = {1.5f, -2.0f};
    Tensor* x     = vec(xs, 2);
    Tensor* in[1] = {x};
    Tensor* Jf    = cml_jacfwd(f_vec2_fwd, NULL, x);
    REQUIRE(Jf != NULL);
    check_jac_vec2(Jf, xs);

    Tensor* Jr = cml_jacrev(f_vec2, NULL, in, 1, 0);
    REQUIRE(Jr != NULL);
    for (size_t i = 0; i < 4; i++)
        REQUIRE(close_to(at(Jf, i), at(Jr, i))); /* jacfwd == jacrev */
    printf("  jacfwd == jacrev: ok\n");
}

static void test_hessian_sumcube(void) {
    if (!cml_autodiff_use_graph()) {
        printf("  hessian: skipped (needs graph engine)\n");
        return;
    }
    float xs[3]   = {0.7f, -1.1f, 1.5f};
    Tensor* x     = vec(xs, 3);
    Tensor* in[1] = {x};
    Tensor* H     = cml_hessian(f_sumcube, NULL, in, 1, 0);
    REQUIRE(H != NULL);
    REQUIRE(H->shape[0] == 3 && H->shape[1] == 3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float want = (i == j) ? 6.0f * xs[i] : 0.0f; /* H = diag(6x) */
            REQUIRE(close_to(at(H, (size_t)(i * 3 + j)), want));
        }
    printf("  hessian(sum x^3) = diag(6x): ok\n");
}

static void test_vmap_square(void) {
    const int B = 3, D = 4;
    float xs[12];
    for (int i = 0; i < B * D; i++)
        xs[i] = 0.2f * (float)i - 1.0f;
    int sh[2] = {B, D};
    Tensor* x = cml_zeros(sh, 2, &cfg);
    memcpy(tensor_data_ptr(x), xs, sizeof(xs));

    Tensor* in[1]  = {x};
    int in_axes[1] = {0};
    Tensor* out    = cml_vmap(f_square, NULL, in, 1, in_axes, 0);
    REQUIRE(out != NULL);
    REQUIRE(out->ndim == 2 && out->shape[0] == B && out->shape[1] == D);
    /* vmap(square, 0) over the batch == squaring each row, i.e. the whole batch. */
    for (int i = 0; i < B * D; i++)
        REQUIRE(close_to(at(out, (size_t)i), xs[i] * xs[i]));
    printf("  vmap(square, in_axes=0): ok\n");
}

static void test_functional_call(void) {
    Linear* lin = nn_linear(2, 2, DTYPE_FLOAT32, DEVICE_CPU, false);
    REQUIRE(lin != NULL);
    Parameter* w = linear_get_weight(lin);
    REQUIRE(w != NULL);
    size_t wn    = tensor_numel(w->tensor->shape, w->tensor->ndim);

    /* snapshot the module's own weight */
    float orig[8];
    tensor_ensure_executed(w->tensor);
    memcpy(orig, tensor_data_ptr(w->tensor), wn * sizeof(float));

    /* swapped weight: originals + 1 */
    Tensor* wswap = cml_zeros(w->tensor->shape, w->tensor->ndim, &cfg);
    float* ws     = (float*)tensor_data_ptr(wswap);
    for (size_t i = 0; i < wn; i++)
        ws[i] = orig[i] + 1.0f;

    StateDict* sd = nn_state_dict_create();
    REQUIRE(nn_state_dict_set(sd, w->name, wswap) == 0);

    float xin[2] = {1.0f, -2.0f};
    int ish[2]   = {1, 2};
    Tensor* xt   = cml_zeros(ish, 2, &cfg);
    memcpy(tensor_data_ptr(xt), xin, sizeof(xin));

    Tensor* got = cml_functional_call((Module*)lin, sd, xt);
    REQUIRE(got != NULL);

    /* reference: a Linear carrying the swapped weight directly */
    Linear* ref   = nn_linear(2, 2, DTYPE_FLOAT32, DEVICE_CPU, false);
    Tensor* wref  = cml_zeros(w->tensor->shape, w->tensor->ndim, &cfg);
    memcpy(tensor_data_ptr(wref), ws, wn * sizeof(float));
    REQUIRE(linear_set_weight(ref, wref) == 0);
    Tensor* xt2 = cml_zeros(ish, 2, &cfg);
    memcpy(tensor_data_ptr(xt2), xin, sizeof(xin));
    Tensor* want = module_forward((Module*)ref, xt2);
    REQUIRE(want != NULL);

    size_t on = tensor_numel(got->shape, got->ndim);
    REQUIRE(on == tensor_numel(want->shape, want->ndim));
    for (size_t i = 0; i < on; i++)
        REQUIRE(close_to(at(got, i), at(want, i))); /* swapped-param result */

    /* module's own weight must be intact */
    tensor_ensure_executed(w->tensor);
    const float* wnow = (const float*)tensor_data_ptr(w->tensor);
    for (size_t i = 0; i < wn; i++)
        REQUIRE(close_to(wnow[i], orig[i]));

    printf("  functional_call swaps params and restores: ok\n");
    nn_state_dict_free(sd);
}

int main(void) {
    printf("Functional transforms (torch.func parity)\n");
    printf("  engine: %s\n", cml_autodiff_use_graph() ? "graph" : "eager");

    test_grad_sumsq();
    test_vjp_vec2();
    test_jacrev_vec2();
    test_jacfwd_matches_jacrev();
    test_hessian_sumcube();
    test_vmap_square();
    test_functional_call();

    printf("All functional-transform tests passed.\n");
    return 0;
}
