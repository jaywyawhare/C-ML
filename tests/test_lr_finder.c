/* LR range test mechanics: it records num_steps geometrically-spaced LRs in range,
 * suggests an LR within range, and leaves the model weights and optimizer LR unchanged. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "core/dataset.h"
#include "core/lr_finder.h"
#include <string.h>
#include "nn.h"
#include "optim.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static int test_lr_range_find(void) {
    /* Tiny regression dataset + 2-layer MLP. */
    Dataset* ds = dataset_random_classification(64, 4, 1);
    if (!ds)
        return 0;
    DataLoader* loader = dataloader_create(ds, 16, true);

    Sequential* m = cml_nn_sequential();
    sequential_add(m, (Module*)cml_nn_linear(4, 8, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(m, (Module*)cml_nn_relu(false));
    sequential_add(m, (Module*)cml_nn_linear(8, 1, DTYPE_FLOAT32, DEVICE_CPU, true));

    Parameter** params = NULL;
    int np             = 0;
    module_collect_parameters((Module*)m, &params, &np, true);
    Optimizer* opt = cml_optim_sgd(params, np, 0.01f, 0.0f, 0.0f);

    /* Snapshot the first parameter to confirm restoration. */
    Tensor* p0 = params[0]->tensor;
    float before[64];
    size_t pn = p0->numel < 64 ? p0->numel : 64;
    float* pd = (float*)tensor_data_ptr(p0);
    memcpy(before, pd, pn * sizeof(float));

    CMLLRFindResult* r =
        cml_lr_range_find((Module*)m, loader, opt, cml_nn_mse_loss, 1e-4f, 1.0f, 12);
    int ok = r != NULL && r->count == 12;

    /* LRs are geometric and within [1e-4, 1]. */
    for (int i = 0; ok && i < r->count; i++)
        ok &= r->lrs[i] >= 1e-4f * 0.999f && r->lrs[i] <= 1.0f * 1.001f;
    for (int i = 1; ok && i < r->count; i++)
        ok &= r->lrs[i] > r->lrs[i - 1]; /* strictly increasing */
    ok &= r && r->suggested_lr >= 1e-4f * 0.999f && r->suggested_lr <= 1.0f * 1.001f;

    /* Model weights restored. */
    float* pa = (float*)tensor_data_ptr(p0);
    for (size_t i = 0; ok && i < pn; i++)
        ok &= fabsf(pa[i] - before[i]) < 1e-5f;

    /* Optimizer LR restored. */
    ok &= fabsf(opt->param_groups[0].lr - 0.01f) < 1e-7f;

    cml_lr_find_result_free(r);
    optimizer_free(opt);
    cml_free(params);
    module_free((Module*)m);
    dataloader_free(loader);
    dataset_free(ds);
    return ok;
}

static int test_rejects_bad_args(void) {
    return cml_lr_range_find(NULL, NULL, NULL, cml_nn_mse_loss, 1e-4f, 1.0f, 10) == NULL;
}

int main(void) {
    cml_init();
    printf("=== LR range finder ===\n");
    TEST(lr_range_find);
    TEST(rejects_bad_args);
    cml_cleanup();
    return TEST_SUMMARY();
}
