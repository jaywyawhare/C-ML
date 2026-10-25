/* Tracked live bytes must stop moving once the caches warm up: a pinned
 * parameter grad whose graph-consumer refs teardown failed to drop leaked once
 * per step. Regression guard for that leak. */
#include "cml.h"
#include "alloc/cml_allocator.h"
#include "ops/ir/context.h"
#include "test_harness.h"

#include <stdio.h>

#define N 8
#define HW 6
#define CLASSES 3

static size_t live_bytes(void) {
    size_t live = 0, peak = 0, count = 0;
    cml_allocator_get_stats(&live, &peak, &count);
    return live;
}

static int test_live_bytes_flat_across_steps(void) {
    cml_seed(7);
    Sequential* m = nn_sequential();
    sequential_add(m, (Module*)nn_conv2d(1, 4, 3, 1, 1, 1, true, DTYPE_FLOAT32, DEVICE_CPU));
    sequential_add(m, (Module*)nn_relu(false));
    sequential_add(m, (Module*)nn_flatten(1, -1));
    sequential_add(m, (Module*)nn_linear(4 * HW * HW, CLASSES, DTYPE_FLOAT32, DEVICE_CPU, true));
    module_set_training((Module*)m, true);

    Parameter** params = NULL;
    int num_params     = 0;
    module_collect_parameters((Module*)m, &params, &num_params, true);
    Optimizer* opt = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);

    float x[N * HW * HW], y[N];
    for (int i = 0; i < N * HW * HW; i++)
        x[i] = (float)((i * 37) % 11) / 11.0f;
    for (int i = 0; i < N; i++)
        y[i] = (float)(i % CLASSES);
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    int xs[4] = {N, 1, HW, HW};

    const int warmup = 4, steps = 14;
    size_t after_warmup = 0;
    for (int s = 0; s < steps; s++) {
        Tensor* X = cml_tensor(x, xs, 4, &cfg);
        Tensor* Y = cml_tensor_1d(y, N);
        optimizer_zero_grad(opt);
        Tensor* out  = module_forward((Module*)m, X);
        Tensor* loss = tensor_sparse_cross_entropy_loss(out, Y);
        tensor_backward(loss, NULL, false, false);
        optimizer_step(opt);
        tensor_free(loss);
        tensor_free(out);
        tensor_free(X);
        tensor_free(Y);
        cml_ir_reset_global_context();
        if (s == warmup - 1)
            after_warmup = live_bytes();
    }
    size_t end  = live_bytes();
    long growth = (long)end - (long)after_warmup;
    int ok      = growth < 1024; /* the leak grew ~1.1KB per step */
    printf("    live after warmup=%zu end=%zu growth=%ld bytes over %d steps\n", after_warmup, end,
           growth, steps - warmup);

    optimizer_free(opt);
    cml_free(params);
    module_free((Module*)m);
    return ok;
}

/**
 * An interned node comes back from cml_ir_add_uop already shaped and its builder shapes it
 * again; every repeat of the same op used to leak the previous shape array.
 * @return (int)
 */
static int test_interned_ops_keep_live_bytes_flat(void) {
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    float d[6] = {1, 2, 3, 4, 5, 6};
    int s23[2] = {2, 3}, s32[2] = {3, 2};
    Tensor* a = cml_tensor(d, s23, 2, &cfg);
    Tensor* b = cml_tensor(d, s32, 2, &cfg);
    Tensor* e = tensor_exp(a);
    Tensor* m = tensor_matmul(a, b);

    size_t before = live_bytes();
    int ok        = 1;
    for (int i = 0; i < 16; i++)
        ok &= tensor_exp(a) == e && tensor_matmul(a, b) == m;
    long growth = (long)live_bytes() - (long)before;
    printf("    interned repeats=%d growth=%ld bytes\n", ok, growth);

    tensor_free(e);
    tensor_free(m);
    tensor_free(a);
    tensor_free(b);
    cml_ir_reset_global_context();
    return ok && growth == 0;
}

int main(void) {
    cml_init();
    cml_enable_grad();
    printf("=== training step steady-state memory ===\n");
    TEST(live_bytes_flat_across_steps);
    TEST(interned_ops_keep_live_bytes_flat);
    cml_cleanup();
    return TEST_SUMMARY();
}
