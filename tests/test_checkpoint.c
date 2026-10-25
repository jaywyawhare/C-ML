/* Keep-best checkpoint: snapshots weights only when the monitored metric improves and
 * restores them exactly, for both maximize and minimize modes. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "nn.h"
#include "nn/state.h"
#include "tensor/tensor.h"
#include "test_harness.h"

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

static int test_minimize(void) {
    Module* m        = (Module*)cml_nn_linear(3, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
    CMLCheckpoint* c = cml_checkpoint_create(0); /* lower is better */

    set_all(m, 1.0f);
    int s1 = cml_checkpoint_update(c, m, 0.5f); /* first -> saved (weights=1) */
    set_all(m, 2.0f);
    int s2 = cml_checkpoint_update(c, m, 0.8f); /* worse -> not saved */
    set_all(m, 3.0f);
    int s3 = cml_checkpoint_update(c, m, 0.2f); /* better -> saved (weights=3) */
    set_all(m, 9.0f);                           /* scribble current weights */

    int ok = s1 == 1 && s2 == 0 && s3 == 1;
    ok &= cml_checkpoint_restore(c, m) == 0;
    ok &= fabsf(first_param(m) - 3.0f) < 1e-6f; /* best was at metric 0.2, weights=3 */
    cml_checkpoint_free(c);
    module_free(m);
    return ok;
}

static int test_maximize(void) {
    Module* m        = (Module*)cml_nn_linear(2, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
    CMLCheckpoint* c = cml_checkpoint_create(1); /* higher is better */
    set_all(m, 5.0f);
    cml_checkpoint_update(c, m, 0.6f);
    set_all(m, 7.0f);
    int improved = cml_checkpoint_update(c, m, 0.9f); /* better */
    set_all(m, 1.0f);
    cml_checkpoint_update(c, m, 0.1f); /* worse -> keep 7 */
    set_all(m, 0.0f);
    int ok =
        improved == 1 && cml_checkpoint_restore(c, m) == 0 && fabsf(first_param(m) - 7.0f) < 1e-6f;
    cml_checkpoint_free(c);
    module_free(m);
    return ok;
}

static int test_restore_without_best(void) {
    Module* m        = (Module*)cml_nn_linear(2, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
    CMLCheckpoint* c = cml_checkpoint_create(0);
    int ok           = cml_checkpoint_restore(c, m) == -1; /* nothing saved yet */
    cml_checkpoint_free(c);
    module_free(m);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== keep-best checkpoint ===\n");
    TEST(minimize);
    TEST(maximize);
    TEST(restore_without_best);
    cml_cleanup();
    return TEST_SUMMARY();
}
