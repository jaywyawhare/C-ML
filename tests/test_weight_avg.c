/* EMA / SWA weight averaging, verified exactly against the averaging formulas, with the
 * shadow kept independent of the live weights until copied back. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "nn.h"
#include "nn/state.h"
#include "tensor/tensor.h"
#include "test_harness.h"

/* Single linear layer; set all its parameter data to `v`. */
static void set_all(Module* m, float v) {
    Parameter** p = NULL;
    int np        = 0;
    module_collect_parameters(m, &p, &np, true);
    for (int i = 0; i < np; i++) {
        float* d = (float*)tensor_data_ptr(p[i]->tensor);
        for (size_t j = 0; j < p[i]->tensor->numel; j++)
            d[j] = v;
    }
    cml_free(p);
}

static float first_param(Module* m) {
    Parameter** p = NULL;
    int np        = 0;
    module_collect_parameters(m, &p, &np, true);
    float* d = (float*)tensor_data_ptr(p[0]->tensor);
    float v  = d ? d[0] : 0.0f;
    cml_free(p);
    return v;
}

static int test_ema(void) {
    Module* m = (Module*)cml_nn_linear(3, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
    set_all(m, 1.0f);
    CMLWeightAvg* wa = cml_weight_avg_create(m, 0.9f); /* shadow = 1.0 */
    set_all(m, 2.0f);
    cml_weight_avg_update_ema(wa, m); /* 0.9*1 + 0.1*2 = 1.1 */
    /* live weights unchanged by the update */
    int ok = fabsf(first_param(m) - 2.0f) < 1e-6f;
    cml_weight_avg_copy_to(wa, m);
    ok &= fabsf(first_param(m) - 1.1f) < 1e-5f;
    cml_weight_avg_free(wa);
    module_free(m);
    return ok;
}

static int test_swa(void) {
    Module* m = (Module*)cml_nn_linear(3, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
    set_all(m, 2.0f);
    CMLWeightAvg* wa = cml_weight_avg_create(m, 0.0f);
    set_all(m, 4.0f);
    cml_weight_avg_update_swa(wa, m); /* alpha=1 -> avg=4 (first snapshot) */
    set_all(m, 6.0f);
    cml_weight_avg_update_swa(wa, m); /* alpha=1/2 -> (4+6)/2 = 5 */
    set_all(m, 8.0f);
    cml_weight_avg_update_swa(wa, m); /* alpha=1/3 -> (2*5+8)/3 = 6 */
    cml_weight_avg_copy_to(wa, m);
    int ok = fabsf(first_param(m) - 6.0f) < 1e-5f;
    cml_weight_avg_free(wa);
    module_free(m);
    return ok;
}

static int test_rejects_null(void) {
    return cml_weight_avg_create(NULL, 0.9f) == NULL && cml_weight_avg_copy_to(NULL, NULL) == -1;
}

int main(void) {
    cml_init();
    printf("=== EMA / SWA weight averaging ===\n");
    TEST(ema);
    TEST(swa);
    TEST(rejects_null);
    cml_cleanup();
    return TEST_SUMMARY();
}
