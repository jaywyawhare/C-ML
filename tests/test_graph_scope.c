/* Graph-view scoping: every IR node emitted inside a layer's forward should
 * carry that layer's module path, so the dashboard can collapse a decomposed
 * graph back into the layers it came from. VIZ must also only observe: the
 * same seeded training runs bit-identically with and without it. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cml.h"
#include "ops/ir/internal.h"
#include "ops/ir/export.h"
#include "alloc/cml_allocator.h"
#include "test_harness.h"

/* MSE backward scales by FILL(2), which the optimizer rewrites to an ADD with no input
 * tensors; optimizing the graph the step then ran made this loop crash under VIZ. */
static void train_losses(float* losses, int steps) {
    cml_seed(11);
    Sequential* model = cml_nn_sequential();
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(8, 16, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_tanh());
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(16, 3, DTYPE_FLOAT32, DEVICE_CPU, true));
    module_set_training((Module*)model, true);
    Parameter** params = NULL;
    int num_params     = 0;
    module_collect_parameters((Module*)model, &params, &num_params, true);
    Optimizer* opt = optim_sgd(params, num_params, 0.1f, 0.9f, 0.0f);

    float x[4 * 8], y[4 * 3];
    for (int i = 0; i < 4 * 8; i++)
        x[i] = sinf((float)i * 0.37f);
    for (int i = 0; i < 4 * 3; i++)
        y[i] = cosf((float)i * 0.21f);
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    for (int s = 0; s < steps; s++) {
        Tensor* X = cml_tensor(x, (int[]){4, 8}, 2, &cfg);
        Tensor* Y = cml_tensor(y, (int[]){4, 3}, 2, &cfg);
        optimizer_zero_grad(opt);
        Tensor* out  = module_forward((Module*)model, X);
        Tensor* loss = tensor_mse_loss(out, Y);
        tensor_backward(loss, NULL, false, false);
        losses[s] = tensor_get_float(loss, 0);
        optimizer_step(opt);
        tensor_free(loss);
        tensor_free(out);
        tensor_free(X);
        tensor_free(Y);
        cml_ir_reset_global_context();
    }
    optimizer_free(opt);
    cml_free(params);
    module_free((Module*)model);
}

int main(void) {
    /* Enable VIZ *after* cml_init: setting it beforehand makes cml_init spawn
     * the dashboard server and a browser, which a test must not do. */
    cml_init();
    int viz_was = cml_flag_push(CML_FLAG_VIZ, 1);
    printf("=== graph scope ===\n");

    CHECK("scope tracking enabled under VIZ", cml_ir_scope_enabled());

    /* Stack behaviour, independent of any model. */
    CHECK("scope starts empty", cml_ir_scope_current() == NULL);
    cml_ir_scope_push("Sequential");
    CHECK("single push",
          cml_ir_scope_current() && strcmp(cml_ir_scope_current(), "Sequential") == 0);
    cml_ir_scope_push("Linear");
    CHECK("nested push joins with '/'",
          cml_ir_scope_current() && strcmp(cml_ir_scope_current(), "Sequential/Linear") == 0);
    cml_ir_scope_pop();
    CHECK("pop restores parent",
          cml_ir_scope_current() && strcmp(cml_ir_scope_current(), "Sequential") == 0);
    cml_ir_scope_pop();
    CHECK("pop to empty", cml_ir_scope_current() == NULL);
    cml_ir_scope_pop(); /* underflow must not corrupt state */
    CHECK("extra pop is harmless", cml_ir_scope_current() == NULL);

    /* A real forward pass should tag its nodes without any layer opting in. */
    cml_reset_ir_context();

    Sequential* model = cml_nn_sequential();
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(8, 16, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));

    int xshape[]     = {2, 8};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* x   = cml_zeros(xshape, 2, &cfg);
    Tensor* out = module_forward((Module*)model, x);
    CHECK("forward produced output", out != NULL);

    CHECK("scope balanced after forward", cml_ir_scope_current() == NULL);

    char* json = cml_ir_export_graph_json(cml_ir_get_or_create_context());
    CHECK("graph json exported", json != NULL);
    if (json) {
        CHECK("nodes carry a scope field", strstr(json, "\"scope\"") != NULL);
        CHECK("scope names the container", strstr(json, "Sequential") != NULL);
        cml_free(json); /* project allocator, not libc malloc */
    }

    float plain[4], viz[4];
    cml_flag_push(CML_FLAG_VIZ, 0);
    train_losses(plain, 4);
    cml_flag_push(CML_FLAG_VIZ, 1);
    train_losses(viz, 4);
    cml_flag_pop(CML_FLAG_VIZ, viz_was);
    CHECK("VIZ leaves training numerics unchanged", memcmp(plain, viz, sizeof(plain)) == 0);

    return TEST_SUMMARY();
}
