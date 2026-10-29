#include "autograd/functional_transforms.h"

#include <stdlib.h>
#include <string.h>

#include "autograd/autograd.h"
#include "autograd/forward_mode.h"
#include "cml.h"
#include "ops/ir/autodiff.h"

/**
 * Grad-free float32 config matching a tensor's device.
 * @param t (Tensor*)
 * @return (TensorConfig)
 */
static TensorConfig cfg_of(Tensor* t) {
    TensorConfig c = {
        .dtype = DTYPE_FLOAT32, .device = t->device, .has_dtype = true, .has_device = true};
    return c;
}

/**
 * Realize a (possibly lazy) tensor into a fresh grad-free float32 copy.
 * @param src (Tensor*)
 * @return (Tensor*)
 */
static Tensor* detach_copy(Tensor* src) {
    if (!src || tensor_ensure_executed(src) != 0)
        return NULL;
    TensorConfig cfg = cfg_of(src);
    Tensor* out      = cml_zeros(src->shape, src->ndim, &cfg);
    if (!out)
        return NULL;
    memcpy(tensor_data_ptr(out), tensor_data_ptr(src),
           tensor_numel(src->shape, src->ndim) * sizeof(float));
    return out;
}

/**
 * Clear any stale gradient and mark a tensor to accumulate on the next backward.
 * @param x (Tensor*)
 */
static void prep_input(Tensor* x) {
    tensor_zero_grad(x);
    x->requires_grad = true;
}

/**
 * Gradient of a scalar-output f w.r.t. inputs[argnum] (reverse mode).
 * @param f (CMLTransformFn)
 * @param ud (void*)
 * @param inputs (Tensor**)
 * @param n (int)
 * @param argnum (int)
 * @return (Tensor*)
 */
Tensor* cml_grad(CMLTransformFn f, void* ud, Tensor** inputs, int n, int argnum) {
    if (!f || !inputs || argnum < 0 || argnum >= n)
        return NULL;
    Tensor* x = inputs[argnum];
    prep_input(x);
    Tensor* y = f(inputs, n, ud);
    if (!y)
        return NULL;
    tensor_backward(y, NULL, false, false); /* unit seed: f must be scalar-valued */
    if (!x->grad)
        return NULL;
    return detach_copy(x->grad);
}

/**
 * Gradient of a scalar-output f w.r.t. several inputs at once.
 * @param f (CMLTransformFn)
 * @param ud (void*)
 * @param inputs (Tensor**)
 * @param n (int)
 * @param argnums (const int*)
 * @param num_argnums (int)
 * @param grads_out (Tensor**)
 * @return (int)
 */
int cml_grad_multi(CMLTransformFn f, void* ud, Tensor** inputs, int n, const int* argnums,
                   int num_argnums, Tensor** grads_out) {
    if (!f || !inputs || !argnums || !grads_out || num_argnums <= 0)
        return -1;
    for (int k = 0; k < num_argnums; k++) {
        int a = argnums[k];
        if (a < 0 || a >= n)
            return -1;
        prep_input(inputs[a]);
    }
    Tensor* y = f(inputs, n, ud);
    if (!y)
        return -1;
    tensor_backward(y, NULL, false, false);
    for (int k = 0; k < num_argnums; k++)
        grads_out[k] = detach_copy(inputs[argnums[k]]->grad);
    return 0;
}

struct CMLVjp {
    CMLTransformFn f;
    void* ud;
    Tensor** inputs; /* borrowed */
    int n;
};

/**
 * Evaluate y = f(inputs) and build a VJP closure over borrowed inputs.
 * @param f (CMLTransformFn)
 * @param ud (void*)
 * @param inputs (Tensor**)
 * @param n (int)
 * @param output (Tensor**)
 * @return (CMLVjp*)
 */
CMLVjp* cml_vjp(CMLTransformFn f, void* ud, Tensor** inputs, int n, Tensor** output) {
    if (!f || !inputs || n <= 0)
        return NULL;
    for (int i = 0; i < n; i++)
        inputs[i]->requires_grad = true;
    Tensor* y = f(inputs, n, ud);
    if (!y)
        return NULL;
    if (output) {
        *output = detach_copy(y);
        if (!*output)
            return NULL;
    }
    CMLVjp* v = (CMLVjp*)malloc(sizeof(CMLVjp));
    if (!v)
        return NULL;
    v->f      = f;
    v->ud     = ud;
    v->inputs = inputs;
    v->n      = n;
    return v;
}

/**
 * Apply a VJP: propagate cotangent back to the n input gradients (v^T J).
 * @param vjp (CMLVjp*)
 * @param cotangent (Tensor*)
 * @param grads_out (Tensor**)
 * @return (int)
 */
int cml_vjp_fn(CMLVjp* vjp, Tensor* cotangent, Tensor** grads_out) {
    if (!vjp || !cotangent || !grads_out)
        return -1;
    for (int i = 0; i < vjp->n; i++)
        prep_input(vjp->inputs[i]);
    Tensor* y = vjp->f(vjp->inputs, vjp->n, vjp->ud); /* re-run so each cotangent is fresh */
    if (!y)
        return -1;
    tensor_backward(y, cotangent, false, false);
    for (int i = 0; i < vjp->n; i++)
        grads_out[i] = detach_copy(vjp->inputs[i]->grad);
    return 0;
}

/**
 * Free a VJP closure (leaves the borrowed inputs alone).
 * @param vjp (CMLVjp*)
 */
void cml_vjp_free(CMLVjp* vjp) { free(vjp); }

/**
 * Reverse-mode Jacobian of f w.r.t. inputs[argnum]: one VJP per output basis
 * cotangent, stacked as rows. Shape [numel(output), numel(inputs[argnum])].
 * @param f (CMLTransformFn)
 * @param ud (void*)
 * @param inputs (Tensor**)
 * @param n (int)
 * @param argnum (int)
 * @return (Tensor*)
 */
Tensor* cml_jacrev(CMLTransformFn f, void* ud, Tensor** inputs, int n, int argnum) {
    if (!f || !inputs || argnum < 0 || argnum >= n)
        return NULL;
    Tensor* x = inputs[argnum];
    size_t nx = tensor_numel(x->shape, x->ndim);

    prep_input(x); /* probe once for the output size */
    Tensor* y0 = f(inputs, n, ud);
    if (!y0)
        return NULL;
    size_t ny = tensor_numel(y0->shape, y0->ndim);

    TensorConfig cfg = cfg_of(x);
    int jsh[2]       = {(int)ny, (int)nx};
    Tensor* jac      = cml_zeros(jsh, 2, &cfg);
    if (!jac)
        return NULL;
    float* jd = (float*)tensor_data_ptr(jac);

    for (size_t i = 0; i < ny; i++) {
        prep_input(x);
        Tensor* yi = f(inputs, n, ud);
        if (!yi)
            return jac;
        TensorConfig ycfg = cfg_of(yi);
        Tensor* basis     = cml_zeros(yi->shape, yi->ndim, &ycfg);
        if (!basis)
            return jac;
        ((float*)tensor_data_ptr(basis))[i] = 1.0f;

        tensor_backward(yi, basis, false, false);
        if (!x->grad)
            return jac;
        Tensor* g = detach_copy(x->grad);
        if (!g)
            return jac;
        const float* gd = (const float*)tensor_data_ptr(g);
        for (size_t j = 0; j < nx; j++)
            jd[i * nx + j] = gd[j]; /* row i = d y_i / d x_j */
    }
    return jac;
}

typedef struct {
    CMLTransformFn f;
    void* ud;
    int argnum;
} HessCtx;

/**
 * Differentiable gradient of the scalar f w.r.t. inputs[argnum].
 * @param inputs (Tensor**)
 * @param n (int)
 * @param ud (void*)
 * @return (Tensor*)
 */
static Tensor* hess_grad_transform(Tensor** inputs, int n, void* ud) {
    HessCtx* h = (HessCtx*)ud;
    Tensor* x  = inputs[h->argnum];
    prep_input(x);
    Tensor* y = h->f(inputs, n, h->ud);
    if (!y)
        return NULL;
    tensor_backward(y, NULL, true, true); /* create_graph: grad must be re-differentiable */
    if (!x->grad || !x->grad->requires_grad)
        return NULL;
    return x->grad;
}

/**
 * Hessian of a scalar-output f w.r.t. inputs[argnum] as jacrev-of-grad.
 * @param f (CMLTransformFn)
 * @param ud (void*)
 * @param inputs (Tensor**)
 * @param n (int)
 * @param argnum (int)
 * @return (Tensor*)
 */
Tensor* cml_hessian(CMLTransformFn f, void* ud, Tensor** inputs, int n, int argnum) {
    if (!f || !inputs || argnum < 0 || argnum >= n)
        return NULL;
    if (!cml_autodiff_use_graph()) { /* reverse-over-reverse needs differentiable grads */
        LOG_ERROR("cml_hessian requires the graph autodiff engine (unset GRAD_MODE=eager); "
                  "reverse-over-reverse needs differentiable gradients");
        return NULL;
    }
    HessCtx h = {.f = f, .ud = ud, .argnum = argnum};
    return cml_jacrev(hess_grad_transform, &h, inputs, n, argnum);
}

/**
 * Product of shape[lo..hi).
 * @param shape (const int*)
 * @param lo (int)
 * @param hi (int)
 * @return (size_t)
 */
static size_t prod_range(const int* shape, int lo, int hi) {
    size_t p = 1;
    for (int d = lo; d < hi; d++)
        p *= (size_t)shape[d];
    return p;
}

/**
 * Select index idx along axis of a contiguous float32 tensor, dropping that
 * axis (a 1-D input collapses to a length-1 tensor).
 * @param src (Tensor*)
 * @param axis (int)
 * @param idx (int)
 * @return (Tensor*)
 */
static Tensor* select_along_axis(Tensor* src, int axis, int idx) {
    if (!src || tensor_ensure_executed(src) != 0)
        return NULL;
    int d         = src->ndim;
    const int* sh = src->shape;
    size_t outer  = prod_range(sh, 0, axis);
    size_t axdim  = (size_t)sh[axis];
    size_t inner  = prod_range(sh, axis + 1, d);
    if ((size_t)idx >= axdim)
        return NULL;

    int one  = 1;
    int* nsh = NULL;
    int nd   = d - 1;
    if (nd <= 0) {
        nsh = &one;
        nd  = 1;
    } else {
        nsh = (int*)malloc(sizeof(int) * (size_t)nd);
        if (!nsh)
            return NULL;
        int w = 0;
        for (int k = 0; k < d; k++)
            if (k != axis)
                nsh[w++] = sh[k];
    }
    TensorConfig cfg = cfg_of(src);
    Tensor* out      = cml_zeros(nsh, nd, &cfg);
    if (nd != 1 || nsh != &one)
        free(nsh);
    if (!out)
        return NULL;

    const float* s = (const float*)tensor_data_ptr(src);
    float* o       = (float*)tensor_data_ptr(out);
    for (size_t ov = 0; ov < outer; ov++)
        memcpy(o + ov * inner, s + (ov * axdim + (size_t)idx) * inner, inner * sizeof(float));
    return out;
}

/**
 * Reference (non-fused) vmap: slice each mapped input, apply f, stack results.
 * @param f (CMLTransformFn)
 * @param ud (void*)
 * @param inputs (Tensor**)
 * @param n (int)
 * @param in_axes (const int*)
 * @param out_axis (int)
 * @return (Tensor*)
 */
Tensor* cml_vmap(CMLTransformFn f, void* ud, Tensor** inputs, int n, const int* in_axes,
                 int out_axis) {
    if (!f || !inputs || !in_axes || n <= 0)
        return NULL;

    int batch = -1;
    for (int k = 0; k < n; k++) {
        if (in_axes[k] >= 0) {
            if (in_axes[k] >= inputs[k]->ndim)
                return NULL;
            int b = inputs[k]->shape[in_axes[k]];
            if (batch < 0)
                batch = b;
            else if (batch != b)
                return NULL; /* ragged mapped axes */
        }
    }
    if (batch <= 0)
        return NULL;

    Tensor** outs = (Tensor**)calloc((size_t)batch, sizeof(Tensor*));
    Tensor** slic = (Tensor**)malloc(sizeof(Tensor*) * (size_t)n);
    if (!outs || !slic) {
        free(outs);
        free(slic);
        return NULL;
    }

    Tensor* result      = NULL;
    size_t sample_numel = 0;
    int sndim           = 0;
    for (int b = 0; b < batch; b++) {
        for (int k = 0; k < n; k++)
            slic[k] = (in_axes[k] >= 0) ? select_along_axis(inputs[k], in_axes[k], b) : inputs[k];
        Tensor* yb = f(slic, n, ud);
        outs[b]    = detach_copy(yb);
        for (int k = 0; k < n; k++)
            if (in_axes[k] >= 0 && slic[k])
                tensor_free(slic[k]);
        if (!outs[b])
            goto done;
        if (b == 0) {
            sndim        = outs[0]->ndim;
            sample_numel = tensor_numel(outs[0]->shape, outs[0]->ndim);
        }
    }

    if (out_axis < 0 || out_axis > sndim)
        goto done;
    int rndim = sndim + 1;
    int* rsh  = (int*)malloc(sizeof(int) * (size_t)rndim);
    if (!rsh)
        goto done;
    for (int k = 0, w = 0; w < rndim; w++)
        rsh[w] = (w == out_axis) ? batch : outs[0]->shape[k++];

    TensorConfig cfg = cfg_of(outs[0]);
    result           = cml_zeros(rsh, rndim, &cfg);
    free(rsh);
    if (!result)
        goto done;

    {
        size_t outer = prod_range(outs[0]->shape, 0, out_axis);
        size_t inner = (outer == 0) ? 0 : sample_numel / outer;
        float* rd    = (float*)tensor_data_ptr(result);
        for (int b = 0; b < batch; b++) {
            const float* od = (const float*)tensor_data_ptr(outs[b]);
            for (size_t ov = 0; ov < outer; ov++)
                memcpy(rd + (ov * (size_t)batch + (size_t)b) * inner, od + ov * inner,
                       inner * sizeof(float));
        }
    }

done:
    for (int b = 0; b < batch; b++)
        if (outs[b])
            tensor_free(outs[b]);
    free(outs);
    free(slic);
    return result;
}

/**
 * Run module->forward(input) with params temporarily swapped in, then restore.
 * @param module (Module*)
 * @param params (const StateDict*)
 * @param input (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_functional_call(Module* module, const StateDict* params, Tensor* input) {
    if (!module || !params || !input)
        return NULL;

    Parameter** swapped = (Parameter**)malloc(sizeof(Parameter*) * (size_t)params->count);
    Tensor** saved      = (Tensor**)malloc(sizeof(Tensor*) * (size_t)params->count);
    if (!swapped || !saved) {
        free(swapped);
        free(saved);
        return NULL;
    }

    int ns = 0;
    for (int i = 0; i < params->count; i++) {
        Parameter* p = module_get_parameter(module, params->entries[i].key);
        if (!p || !params->entries[i].value)
            continue;
        swapped[ns] = p;
        saved[ns]   = p->tensor;
        p->tensor   = params->entries[i].value; /* pointer swap only, no free */
        ns++;
    }

    Tensor* out  = module_forward(module, input);
    Tensor* copy = detach_copy(out); /* realize while swapped params are live */

    for (int i = 0; i < ns; i++)
        swapped[i]->tensor = saved[i];
    free(swapped);
    free(saved);
    return copy;
}
