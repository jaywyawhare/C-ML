#include "autograd/forward_mode.h"

#include <string.h>

#include "autograd/autograd.h"
#include "cml.h"
#include "ops/ir/autodiff.h"
#include "ops/uops.h"

static TensorConfig cfg_of(Tensor* t) {
    TensorConfig c = {
        .dtype = DTYPE_FLOAT32, .device = t->device, .has_dtype = true, .has_device = true};
    return c;
}

/**
 * Copy a (possibly lazy) tensor's realized data into a fresh grad-free tensor.
 * @param src (Tensor*)
 * @return (Tensor*)
 */
static Tensor* detach_copy(Tensor* src) {
    if (tensor_ensure_executed(src) != 0)
        return NULL;
    TensorConfig cfg = cfg_of(src);
    Tensor* out      = cml_zeros(src->shape, src->ndim, &cfg);
    if (!out)
        return NULL;
    memcpy(tensor_data_ptr(out), tensor_data_ptr(src),
           tensor_numel(src->shape, src->ndim) * sizeof(float));
    return out;
}

Tensor* cml_jvp(CMLForwardFn f, void* ctx, Tensor* x, Tensor* v) {
    if (!f || !x || !v || !cml_autodiff_use_graph())
        return NULL;

    tensor_zero_grad(x);
    x->requires_grad = true;
    Tensor* y        = f(x, ctx);
    if (!y)
        return NULL;

    /* Cotangent variable u: differentiating <J^T u, v> = <u, J v> w.r.t. u gives J v. */
    TensorConfig cfg = cfg_of(y);
    Tensor* u        = tensor_ones(y->shape, y->ndim, &cfg);
    if (!u)
        return NULL;
    u->requires_grad = true;

    ReduceParams rp = {0};
    Tensor* s1      = uop_sum(uop_mul(y, u), &rp); /* <y, u> */
    tensor_backward(s1, NULL, true, true);
    if (!x->grad || !x->grad->requires_grad)
        return NULL;

    Tensor* s2 = uop_sum(uop_mul(x->grad, v), &rp); /* <J^T u, v> */
    tensor_zero_grad(u);
    tensor_backward(s2, NULL, false, false);
    if (!u->grad)
        return NULL;
    return detach_copy(u->grad);
}

Tensor* cml_jacfwd(CMLForwardFn f, void* ctx, Tensor* x) {
    if (!f || !x || !cml_autodiff_use_graph())
        return NULL;

    size_t nx        = tensor_numel(x->shape, x->ndim);
    TensorConfig cfg = cfg_of(x);
    Tensor* e        = cml_zeros(x->shape, x->ndim, &cfg);
    if (!e)
        return NULL;
    float* ed = (float*)tensor_data_ptr(e);

    Tensor* jac = NULL;
    size_t ny   = 0;
    for (size_t j = 0; j < nx; j++) {
        memset(ed, 0, nx * sizeof(float));
        ed[j]       = 1.0f;
        Tensor* col = cml_jvp(f, ctx, x, e);
        if (!col)
            return jac;
        if (!jac) {
            ny         = tensor_numel(col->shape, col->ndim);
            int jsh[2] = {(int)ny, (int)nx};
            jac        = cml_zeros(jsh, 2, &cfg);
            if (!jac)
                return NULL;
        }
        const float* cd = (const float*)tensor_data_ptr(col);
        float* jd       = (float*)tensor_data_ptr(jac);
        for (size_t i = 0; i < ny; i++)
            jd[i * nx + j] = cd[i]; /* column j = J e_j */
    }
    return jac;
}
