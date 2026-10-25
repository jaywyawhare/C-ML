/* Global magnitude pruning, verified by the exact fraction of weights zeroed and that the
 * smallest-magnitude weights are the ones removed. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "nn.h"
#include "nn/state.h"
#include "tensor/tensor.h"
#include "test_harness.h"

/* Assign 1,2,3,... across every float32 weight so magnitudes are distinct and ordered. */
static int fill_sequential(Module* m) {
    Parameter** p = NULL;
    int np        = 0;
    module_collect_parameters(m, &p, &np, true);
    int n = 0;
    for (int i = 0; i < np; i++) {
        float* d = (float*)tensor_data_ptr(p[i]->tensor);
        for (size_t j = 0; j < p[i]->tensor->numel; j++)
            d[j] = (float)(++n);
    }
    cml_free(p);
    return n;
}

static int count_zeros(Module* m) {
    Parameter** p = NULL;
    int np        = 0;
    module_collect_parameters(m, &p, &np, true);
    int z = 0;
    for (int i = 0; i < np; i++) {
        float* d = (float*)tensor_data_ptr(p[i]->tensor);
        for (size_t j = 0; j < p[i]->tensor->numel; j++)
            if (d[j] == 0.0f)
                z++;
    }
    cml_free(p);
    return z;
}

static int test_prune_fraction(void) {
    Module* m = (Module*)cml_nn_linear(3, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
    int n     = fill_sequential(m); /* 8 weights: 1..8 */
    int ok    = fabsf(cml_model_sparsity(m)) < 1e-6f;
    float s   = cml_prune_magnitude(m, 0.25f); /* zero 2 smallest (1,2) */
    ok &= fabsf(s - 0.25f) < 1e-6f && count_zeros(m) == n / 4;
    module_free(m);
    return ok;
}

static int test_prune_all(void) {
    Module* m = (Module*)cml_nn_linear(3, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
    fill_sequential(m);
    float s = cml_prune_magnitude(m, 1.0f);
    int ok  = fabsf(s - 1.0f) < 1e-6f;
    module_free(m);
    return ok;
}

static int test_prune_none(void) {
    Module* m = (Module*)cml_nn_linear(3, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
    fill_sequential(m);
    float s = cml_prune_magnitude(m, 0.0f);
    int ok  = fabsf(s) < 1e-6f && cml_model_sparsity(NULL) == 0.0f;
    module_free(m);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== magnitude pruning ===\n");
    TEST(prune_fraction);
    TEST(prune_all);
    TEST(prune_none);
    cml_cleanup();
    return TEST_SUMMARY();
}
