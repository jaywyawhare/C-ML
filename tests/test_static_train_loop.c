/* cml_train static_graph mode: build the fwd+bwd graph ONCE, then every batch
 * memcpys new data into fixed buffers, cml_ir_reexecute()s (no rebuild), and
 * applies an in-place SGD step. This test runs a single static training in its
 * own process (the static mode reuses the global graph/cache, so mixing it with
 * other full graph executions in the same process is not supported) and checks
 * that it converges on a small linear-regression problem. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cml.h"
#include "core/dataset.h"
#include "core/training_loop.h"
#include "ops/ir/context.h"

static float g_first = -1.0f, g_last = 0.0f;
static int on_batch(int epoch, int batch, float loss, void* ud) {
    (void)epoch;
    (void)ud;
    /* Track batch 0 only (shuffle=false, so it's the same data every epoch) — a
     * clean convergence signal, unlike comparing losses of different batches. */
    if (batch == 0) {
        if (g_first < 0.0f)
            g_first = loss;
        g_last = loss;
    }
    return 0;
}

int main(void) {
    cml_init();
    int N = 16, in = 3, out = 1;
    float X[48], Y[16];
    for (int s = 0; s < N; s++) {
        float acc = 0;
        for (int j = 0; j < in; j++) {
            X[s * in + j] = 0.1f * ((s * in + j) % 5) - 0.2f;
            acc += X[s * in + j] * (j + 1);
        }
        Y[s] = acc; /* y = x0 + 2x1 + 3x2 (exactly linear) */
    }
    cml_reset_ir_context();
    Dataset* ds    = dataset_from_arrays(X, Y, N, in, out);
    DataLoader* dl = dataloader_create(ds, 4, false); /* constant-shape batches */
    Module* model  = (Module*)cml_nn_linear(in, out, DTYPE_FLOAT32, DEVICE_CPU, true);
    module_set_training(model, true);
    Optimizer* opt = optim_sgd_for_model(model, 0.03f, 0.0f, 0.0f);

    TrainingConfig cfg;
    training_config_default(&cfg);
    cfg.epochs                 = 150;
    cfg.verbose                = false;
    cfg.use_progress_bar       = false;
    cfg.static_graph           = true; /* the feature under test */
    cfg.callbacks.on_batch_end = on_batch;

    printf("=== cml_train static_graph mode ===\n");
    cml_train(model, dl, opt, cml_nn_mse_loss, &cfg);
    printf("  first-batch loss=%.5f  final loss=%.5f\n", g_first, g_last);

    int fail = 0;
    if (!(g_last < g_first * 0.5f)) {
        printf("  FAIL  static training converges\n");
        fail = 1;
    } else {
        printf("  PASS  static training converges (zero-rebuild)\n");
    }

    optimizer_free(opt);
    module_free(model);
    dataloader_free(dl);
    dataset_free(ds);
    printf("\n%s\n", fail ? "STATIC TRAIN LOOP FAILED" : "static train loop passed");
    return fail;
}
