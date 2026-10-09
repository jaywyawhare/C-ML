/**
 * Regression guard: the eager in-place optimizer fast path must produce the
 * same parameter updates as the node-emitting IR path it replaces. Both are run
 * on an identically-initialised model with identical gradients; FUSE_OPTIM
 * forces the IR path (the in-place path is skipped under it), and the final
 * parameters must match. This locks the two paths together so a future change
 * to one cannot silently diverge from the other.
 */
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cml.h"
#include "core/cml_flags.h"

typedef Optimizer* (*OptFactory)(Parameter**, int);

/* Overwrite parameter buffers with a fixed pattern so both runs start identical
 * regardless of the (non-libc-seeded) weight initialiser. */
static void init_params(Parameter** params, int np) {
    for (int i = 0; i < np; i++) {
        Tensor* t = params[i] ? params[i]->tensor : NULL;
        if (!t || !t->data)
            continue;
        float* d = (float*)t->data;
        for (size_t j = 0; j < t->numel; j++)
            d[j] = 0.05f * (float)(((int)j * 7 + i * 3) % 11 - 5);
    }
}

static Optimizer* make_sgd(Parameter** p, int n) { return cml_optim_sgd(p, n, 0.05f, 0.0f, 0.0f); }
static Optimizer* make_sgd_mom(Parameter** p, int n) {
    return cml_optim_sgd(p, n, 0.05f, 0.9f, 0.001f);
}
static Optimizer* make_adam(Parameter** p, int n) {
    return cml_optim_adam(p, n, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
}
static Optimizer* make_adam_wd(Parameter** p, int n) {
    return cml_optim_adam(p, n, 0.01f, 0.01f, 0.9f, 0.999f, 1e-8f);
}
static Optimizer* make_adam_ams(Parameter** p, int n) {
    Optimizer* o = cml_optim_adam(p, n, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    optimizer_set_amsgrad(o, true);
    return o;
}

/* Train a fresh fixed-init model for `steps` real steps (forward/loss/backward/
 * step/reset -- the ordinary loop), optionally forcing the node-emitting IR
 * path, and flatten the final parameters into `out`. Fixed seed and fixed input
 * make the gradient sequence identical between the two runs. */
static void run(OptFactory make, int steps, bool force_ir, float* out, int* out_n) {
    srand(1234);
    Sequential* m = cml_nn_sequential();
    sequential_add(m, (Module*)cml_nn_linear(3, 5, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(m, (Module*)cml_nn_relu(false));
    sequential_add(m, (Module*)cml_nn_linear(5, 2, DTYPE_FLOAT32, DEVICE_CPU, true));

    Parameter** params = NULL;
    int np             = 0;
    module_collect_parameters((Module*)m, &params, &np, true);
    init_params(params, np);
    Optimizer* opt = make(params, np);

    const int batch = 4;
    float xd[12], yd[8];
    for (int i = 0; i < 12; i++)
        xd[i] = 0.1f * (float)((i % 5) - 2);
    for (int i = 0; i < 8; i++)
        yd[i] = 0.1f * (float)((i % 3) - 1);
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    int xs[] = {batch, 3}, ys[] = {batch, 2};

    int prev = force_ir ? cml_flag_push(CML_FLAG_FUSE_OPTIM, 1) : 0;
    for (int s = 0; s < steps; s++) {
        Tensor* X    = cml_tensor(xd, xs, 2, &cfg);
        Tensor* Y    = cml_tensor(yd, ys, 2, &cfg);
        Tensor* pred = cml_nn_sequential_forward(m, X);
        Tensor* loss = cml_nn_mse_loss(pred, Y);
        cml_optim_zero_grad(opt);
        cml_backward(loss, NULL, false, false);
        optimizer_step(opt);
        cml_reset_ir_context();
    }
    if (force_ir)
        cml_flag_pop(CML_FLAG_FUSE_OPTIM, prev);

    int k = 0;
    for (int i = 0; i < np; i++) {
        Tensor* t = params[i]->tensor;
        for (size_t j = 0; j < t->numel; j++)
            out[k++] = ((float*)t->data)[j];
    }
    *out_n = k;

    optimizer_free(opt);
    cml_free(params);
    module_free((Module*)m);
}

static int equiv(OptFactory make, const char* name) {
    float a[4096], b[4096];
    int na = 0, nb = 0;
    run(make, 5, false, a, &na); /* in-place fast path */
    run(make, 5, true, b, &nb);  /* node-emitting IR path */
    if (na != nb) {
        printf("  %-10s param-count mismatch (%d vs %d) FAIL\n", name, na, nb);
        return 0;
    }
    float maxdiff = 0.0f;
    for (int i = 0; i < na; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > maxdiff)
            maxdiff = d;
    }
    int pass = maxdiff < 1e-5f;
    printf("  %-10s maxdiff=%.3e %s\n", name, (double)maxdiff, pass ? "PASS" : "FAIL");
    return pass;
}

int main(void) {
    cml_init();
    printf("Optimizer in-place vs IR-path equivalence\n");

    int ok = 1;
    ok &= equiv(make_sgd, "sgd");
    ok &= equiv(make_sgd_mom, "sgd_mom");
    ok &= equiv(make_adam, "adam");
    ok &= equiv(make_adam_wd, "adam_wd");
    ok &= equiv(make_adam_ams, "adam_ams");

    printf(ok ? "All equivalence tests passed.\n" : "Equivalence FAILURES.\n");
    return ok ? 0 : 1;
}
