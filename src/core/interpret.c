#include "core/interpret.h"
#include "core/logging.h"
#include "alloc/cml_allocator.h"
#include "cml.h"
#include <math.h>
#include <string.h>

static const TensorConfig IFCFG = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

/** Build a one-hot gradient seed shaped like @p out, with 1 at target_class along the
 *  last dimension of every row. Returns an eager tensor (caller frees), or NULL.
 * @param out (Tensor*)
 * @param target_class (int)
 * @return (Tensor*) */
static Tensor* one_hot_seed(Tensor* out, int target_class) {
    if (!out || out->ndim < 1)
        return NULL;
    int classes = out->shape[out->ndim - 1];
    if (target_class < 0 || target_class >= classes)
        return NULL;
    size_t n     = out->numel;
    size_t outer = n / (size_t)classes;
    float* d     = cml_calloc(n, sizeof(float));
    if (!d)
        return NULL;
    for (size_t o = 0; o < outer; o++)
        d[o * (size_t)classes + (size_t)target_class] = 1.0f;
    Tensor* seed = tensor_from_data(d, out->shape, out->ndim, &IFCFG);
    cml_free(d);
    return seed;
}

/** Run model(input) differentiably and accumulate d output[target_class]/d input into
 *  @p accum (summed over the batch per input element). Returns 0 on success.
 * @param model (Module*)
 * @param input (Tensor*)
 * @param target_class (int)
 * @param accum (double*)
 * @return (int) */
static int grad_wrt_input(Module* model, Tensor* input, int target_class, double* accum) {
    tensor_set_requires_grad(input, true);
    tensor_zero_grad(input);
    Tensor* out = module_forward(model, input);
    if (!out)
        return -1;
    Tensor* seed = one_hot_seed(out, target_class);
    if (!seed) {
        cml_reset_ir_context();
        return -1;
    }
    cml_backward(out, seed, false, false);
    Tensor* g = tensor_get_grad(input);
    int rc    = -1;
    if (g) {
        float* gd = (float*)tensor_data_ptr(g);
        if (gd) {
            for (size_t i = 0; i < input->numel; i++)
                accum[i] += gd[i];
            rc = 0;
        }
    }
    tensor_free(seed);
    cml_reset_ir_context();
    return rc;
}

/** Saliency map: |d output[target_class] / d input|.
 * @param model (Module*)
 * @param input (Tensor*)
 * @param target_class (int)
 * @return (Tensor*)
 */
Tensor* cml_saliency(Module* model, Tensor* input, int target_class) {
    if (!model || !input)
        return NULL;
    size_t n      = input->numel;
    double* accum = cml_calloc(n, sizeof(double));
    if (!accum)
        return NULL;
    if (grad_wrt_input(model, input, target_class, accum) != 0) {
        cml_free(accum);
        return NULL;
    }
    Tensor* sal = tensor_empty(input->shape, input->ndim, &IFCFG);
    if (sal) {
        float* sd = (float*)tensor_data_ptr(sal);
        for (size_t i = 0; i < n; i++)
            sd[i] = (float)fabs(accum[i]);
    }
    cml_free(accum);
    return sal;
}

/** Integrated gradients attribution of output[target_class] to each input element.
 * @param model (Module*)
 * @param input (Tensor*)
 * @param baseline (Tensor*)
 * @param target_class (int)
 * @param steps (int)
 * @return (Tensor*)
 */
Tensor* cml_integrated_gradients(Module* model, Tensor* input, Tensor* baseline, int target_class,
                                 int steps) {
    if (!model || !input || steps < 1)
        return NULL;
    if (baseline && baseline->numel != input->numel) {
        LOG_ERROR("cml_integrated_gradients: baseline shape must match input");
        return NULL;
    }
    size_t n          = input->numel;
    const float* in   = (const float*)tensor_data_ptr(input);
    const float* base = baseline ? (const float*)tensor_data_ptr(baseline) : NULL;
    if (!in)
        return NULL;

    double* accum = cml_calloc(n, sizeof(double)); /* summed gradient along the path */
    float* xd     = cml_malloc(n * sizeof(float));
    if (!accum || !xd) {
        cml_free(accum);
        cml_free(xd);
        return NULL;
    }
    for (int s = 1; s <= steps; s++) {
        float alpha = (float)s / (float)steps;
        for (size_t i = 0; i < n; i++) {
            float b = base ? base[i] : 0.0f;
            xd[i]   = b + alpha * (in[i] - b);
        }
        Tensor* xs = tensor_from_data(xd, input->shape, input->ndim, &IFCFG);
        if (!xs)
            break;
        int rc = grad_wrt_input(model, xs, target_class, accum);
        tensor_free(xs);
        if (rc != 0) {
            cml_free(accum);
            cml_free(xd);
            return NULL;
        }
    }
    cml_free(xd);

    Tensor* ig = tensor_empty(input->shape, input->ndim, &IFCFG);
    if (ig) {
        float* igd = (float*)tensor_data_ptr(ig);
        for (size_t i = 0; i < n; i++) {
            float b = base ? base[i] : 0.0f;
            igd[i]  = (in[i] - b) * (float)(accum[i] / (double)steps);
        }
    }
    cml_free(accum);
    return ig;
}
