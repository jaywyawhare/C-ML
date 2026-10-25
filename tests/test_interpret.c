/* Gradient attribution on a linear model, where the answers are exact: for y = Wx+b,
 * d y[c]/d x = W[c, :], so saliency = |W[c, :]| and integrated gradients satisfy the
 * completeness axiom sum(IG) = y[c](x) - y[c](baseline). */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "core/interpret.h"
#include <string.h>
#include "nn.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static TensorConfig CFG = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

/* Linear(3->2) with W row0={1,-2,3}, row1={4,5,-6}, bias={0.5,-0.5}. */
static Module* make_linear(void) {
    Module* m     = (Module*)cml_nn_linear(3, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
    Parameter** p = NULL;
    int np        = 0;
    module_collect_parameters(m, &p, &np, true);
    for (int i = 0; i < np; i++) {
        float* d = (float*)tensor_data_ptr(p[i]->tensor);
        if (p[i]->tensor->numel == 6) {
            float w[6] = {1, -2, 3, 4, 5, -6};
            memcpy(d, w, sizeof w);
        } else if (p[i]->tensor->numel == 2) {
            d[0] = 0.5f;
            d[1] = -0.5f;
        }
    }
    cml_free(p);
    return m;
}

static int test_saliency(void) {
    Module* m   = make_linear();
    float xd[3] = {2, 1, -1};
    int xs[2]   = {1, 3};
    Tensor* x   = tensor_from_data(xd, xs, 2, &CFG);
    Tensor* sal = cml_saliency(m, x, 0); /* |W[0,:]| = {1,2,3} */
    int ok      = sal != NULL;
    float* s    = sal ? (float*)tensor_data_ptr(sal) : NULL;
    ok &=
        s && fabsf(s[0] - 1.0f) < 1e-4f && fabsf(s[1] - 2.0f) < 1e-4f && fabsf(s[2] - 3.0f) < 1e-4f;
    tensor_free(sal);
    tensor_free(x);
    module_free(m);
    return ok;
}

static int test_integrated_gradients(void) {
    Module* m   = make_linear();
    float xd[3] = {2, 1, -1};
    int xs[2]   = {1, 3};
    Tensor* x   = tensor_from_data(xd, xs, 2, &CFG);
    Tensor* ig  = cml_integrated_gradients(m, x, NULL, 0, 32); /* baseline 0 */
    int ok      = ig != NULL;
    float* g    = ig ? (float*)tensor_data_ptr(ig) : NULL;
    /* IG = x .* W[0,:] = {2*1, 1*-2, -1*3} = {2,-2,-3} */
    ok &= g && fabsf(g[0] - 2.0f) < 1e-3f && fabsf(g[1] - (-2.0f)) < 1e-3f &&
          fabsf(g[2] - (-3.0f)) < 1e-3f;
    /* completeness: sum = y0(x) - y0(0) = W[0,:].x = -3 */
    float sum = g ? g[0] + g[1] + g[2] : 0;
    ok &= fabsf(sum - (-3.0f)) < 1e-3f;
    tensor_free(ig);
    tensor_free(x);
    module_free(m);
    return ok;
}

static int test_rejects_bad_args(void) {
    float xd[3] = {1, 2, 3};
    int xs[2]   = {1, 3};
    Tensor* x   = tensor_from_data(xd, xs, 2, &CFG);
    int ok =
        cml_saliency(NULL, x, 0) == NULL && cml_integrated_gradients(NULL, x, NULL, 0, 4) == NULL;
    tensor_free(x);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== gradient attribution ===\n");
    TEST(saliency);
    TEST(integrated_gradients);
    TEST(rejects_bad_args);
    cml_cleanup();
    return TEST_SUMMARY();
}
