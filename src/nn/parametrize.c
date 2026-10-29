#include "nn/parametrize.h"

#include <math.h>
#include <string.h>

#include "alloc/cml_allocator.h"
#include "cml.h"

/**
 * Build a plain (grad-free) f32 tensor matching another tensor's device.
 * @param shape (int*)
 * @param ndim (int)
 * @param ref (Tensor*)
 * @return (Tensor*)
 */
static Tensor* const_like(int* shape, int ndim, Tensor* ref) {
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = ref->device, .has_dtype = true, .has_device = true};
    return cml_zeros(shape, ndim, &cfg);
}

Tensor* cml_nn_weight_norm(Tensor* v, Tensor* g, int dim) {
    if (!v || !g || dim < 0 || dim >= v->ndim)
        return NULL;

    Tensor* vt = (dim == 0) ? v : cml_transpose(v, 0, dim);
    if (!vt)
        return NULL;
    Tensor* vc = tensor_contiguous(vt);
    if (!vc)
        return NULL;

    int d          = vc->shape[0];
    size_t total   = tensor_numel(vc->shape, vc->ndim);
    int r          = (int)(total / (size_t)d);
    int flat_sh[2] = {d, r};
    Tensor* v2     = cml_reshape(vc, flat_sh, 2);
    if (!v2)
        return NULL;

    Tensor* norm = cml_sqrt(cml_sum(cml_mul(v2, v2), 1, true)); /* [d,1] */
    int g_sh[2]  = {d, 1};
    Tensor* g2   = cml_reshape(g, g_sh, 2);
    Tensor* w2   = cml_mul(v2, cml_div(g2, norm)); /* broadcast [d,r] */
    if (!w2)
        return NULL;

    Tensor* wt = cml_reshape(w2, vc->shape, vc->ndim);
    if (!wt)
        return NULL;
    return (dim == 0) ? wt : cml_transpose(wt, 0, dim);
}

Tensor* cml_nn_spectral_norm(Tensor* W, Tensor* u, int n_power_iterations) {
    if (!W || !u || W->ndim != 2)
        return NULL;
    int m = W->shape[0];
    int n = W->shape[1];
    if (u->ndim != 1 || u->shape[0] != m)
        return NULL;

    if (tensor_ensure_executed(W) != 0)
        return NULL;
    const float* wd = (const float*)tensor_data_ptr(W);
    float* ud       = (float*)tensor_data_ptr(u);
    if (!wd || !ud)
        return NULL;

    float* vbuf = (float*)cml_malloc((size_t)n * sizeof(float));
    if (!vbuf)
        return NULL;

    /* Power iteration refines u (left) and v (right) singular vectors; detached
     * from autograd so no gradient flows through the estimate itself. */
    for (int it = 0; it < n_power_iterations; it++) {
        float vn = 0.0f;
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int i = 0; i < m; i++)
                s += wd[(size_t)i * n + j] * ud[i];
            vbuf[j] = s;
            vn += s * s;
        }
        vn = sqrtf(vn) + 1e-12f;
        for (int j = 0; j < n; j++)
            vbuf[j] /= vn;

        float un = 0.0f;
        for (int i = 0; i < m; i++) {
            float s = 0.0f;
            for (int j = 0; j < n; j++)
                s += wd[(size_t)i * n + j] * vbuf[j];
            ud[i] = s;
            un += s * s;
        }
        un = sqrtf(un) + 1e-12f;
        for (int i = 0; i < m; i++)
            ud[i] /= un;
    }

    int v_sh[2]   = {n, 1};
    Tensor* v_col = const_like(v_sh, 2, W);
    int u_sh[2]   = {1, m};
    Tensor* u_row = const_like(u_sh, 2, W);
    if (!v_col || !u_row) {
        cml_free(vbuf);
        return NULL;
    }
    memcpy(tensor_data_ptr(v_col), vbuf, (size_t)n * sizeof(float));
    memcpy(tensor_data_ptr(u_row), ud, (size_t)m * sizeof(float));
    cml_free(vbuf);

    Tensor* wv    = cml_matmul(W, v_col);  /* [m,1] */
    Tensor* sigma = cml_matmul(u_row, wv); /* [1,1] = u^T W v */
    return cml_div(W, sigma);              /* broadcast scalar */
}
