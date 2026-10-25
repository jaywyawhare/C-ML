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
    /* A helper built after the model and freed before training: VIZ capture used to read
     * whichever Sequential was built last. */
    module_free((Module*)cml_nn_sequential());
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

/** Two-layer sigmoid MLP for the GAN parity check. */
static Sequential* gan_net(int in, int out) {
    Sequential* m = cml_nn_sequential();
    cml_nn_sequential_add(m, (Module*)cml_nn_linear(in, 16, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(m, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(m, (Module*)cml_nn_linear(16, out, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(m, (Module*)cml_nn_sigmoid());
    module_set_training((Module*)m, true);
    return m;
}

/* GAN-style steps run two backward passes in one graph: the second replaces grads the first
 * published, and the dashboard copy then read inputs that replacement had freed. */
static void gan_losses(float* losses, int steps) {
    cml_seed(42);
    Sequential* g = gan_net(8, 8);
    Sequential* d = gan_net(8, 1);
    Optimizer* og = optim_adam_for_model((Module*)g, 0.001f, 0.0f, 0.9f, 0.999f, 1e-8f);
    Optimizer* od = optim_adam_for_model((Module*)d, 0.001f, 0.0f, 0.9f, 0.999f, 1e-8f);
    float ones[4] = {1, 1, 1, 1}, zeros[4] = {0, 0, 0, 0}, real[32], noise[32];
    int xs[] = {4, 8}, ls[] = {4, 1};
    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < 32; i++)
            real[i] = 0.5f + 0.4f * sinf(0.3f * (float)i + (float)s);
        Tensor* one = cml_tensor(ones, ls, 2, NULL);
        Tensor* ldr =
            cml_nn_bce_loss(module_forward((Module*)d, cml_tensor(real, xs, 2, NULL)), one);
        for (int i = 0; i < 32; i++)
            noise[i] = cosf(0.7f * (float)i + 3.0f * (float)s);
        Tensor* fake = module_forward((Module*)g, cml_tensor(noise, xs, 2, NULL));
        Tensor* ldf =
            cml_nn_bce_loss(module_forward((Module*)d, fake), cml_tensor(zeros, ls, 2, NULL));
        Tensor* ld = cml_add(ldr, ldf);
        optimizer_zero_grad(od);
        tensor_backward(ld, NULL, false, false);
        optimizer_step(od);

        for (int i = 0; i < 32; i++)
            noise[i] = sinf(1.1f * (float)i + 5.0f * (float)s);
        Tensor* lg = cml_nn_bce_loss(
            module_forward((Module*)d, module_forward((Module*)g, cml_tensor(noise, xs, 2, NULL))),
            one);
        optimizer_zero_grad(og);
        tensor_backward(lg, NULL, false, false);
        optimizer_step(og);
        losses[2 * s]     = tensor_get_float(ld, 0);
        losses[2 * s + 1] = tensor_get_float(lg, 0);
        cml_ir_reset_global_context();
    }
    optimizer_free(og);
    optimizer_free(od);
    module_free((Module*)g);
    module_free((Module*)d);
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
    for (int i = 0; i < 40; i++)
        cml_ir_scope_push("L");
    for (int i = 0; i < 39; i++)
        cml_ir_scope_pop();
    CHECK("pushes past the depth limit pop back to the outermost",
          cml_ir_scope_current() && strcmp(cml_ir_scope_current(), "L") == 0);
    cml_ir_scope_pop();
    CHECK("deep pushes unwind to empty", cml_ir_scope_current() == NULL);

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

    float plain[4], viz[4], gan_plain[6], gan_viz[6];
    cml_flag_push(CML_FLAG_VIZ, 0);
    gan_losses(gan_plain, 3);
    train_losses(plain, 4);
    cml_flag_push(CML_FLAG_VIZ, 1);
    gan_losses(gan_viz, 3);
    train_losses(viz, 4);
    cml_flag_pop(CML_FLAG_VIZ, viz_was);
    CHECK("VIZ leaves training numerics unchanged", memcmp(plain, viz, sizeof(plain)) == 0);
    CHECK("VIZ leaves two-backward (GAN) steps unchanged",
          memcmp(gan_plain, gan_viz, sizeof(gan_plain)) == 0);

    return TEST_SUMMARY();
}
