/* Train a small CNN (conv-relu-pool x2, linear) on full MNIST via cml_dataset_load
 * + DataLoader and report held-out test accuracy (~99%); CML_MNIST_MLP=1 uses an
 * MLP instead. Fetches MNIST over the network on first run (then cached); exits
 * cleanly with a skip notice when it is unavailable. */
#include "cml.h"
#include "datasets/datasets.h"
#include "core/dataset.h"
#include "ops/ir/context.h"

#include <stdio.h>
#include <stdlib.h>

#define IMG 28
#define FEAT (IMG * IMG)
#define N_TEST 10000 /* MNIST canonical test split (last samples after train) */

static int use_cnn(void) { return getenv("CML_MNIST_MLP") == NULL; }

/** Default conv stack: [N,1,28,28] -> 10 logits; MLP under CML_MNIST_MLP. */
static Sequential* build_model(void) {
    Sequential* m = nn_sequential();
    if (use_cnn()) {
        sequential_add(m, (Module*)nn_conv2d(1, 16, 3, 1, 1, 1, true, DTYPE_FLOAT32, DEVICE_CPU));
        sequential_add(m, (Module*)nn_relu(false));
        sequential_add(m, (Module*)nn_maxpool2d(2, 2, 0, 1, false)); /* 28 -> 14 */
        sequential_add(m, (Module*)nn_conv2d(16, 32, 3, 1, 1, 1, true, DTYPE_FLOAT32, DEVICE_CPU));
        sequential_add(m, (Module*)nn_relu(false));
        sequential_add(m, (Module*)nn_maxpool2d(2, 2, 0, 1, false)); /* 14 -> 7 */
        sequential_add(m, (Module*)nn_flatten(1, -1));
        sequential_add(m, (Module*)nn_linear(32 * 7 * 7, 10, DTYPE_FLOAT32, DEVICE_CPU, true));
        return m;
    }
    sequential_add(m, (Module*)nn_flatten(1, -1));
    sequential_add(m, (Module*)nn_linear(FEAT, 128, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(m, (Module*)nn_relu(false));
    sequential_add(m, (Module*)nn_linear(128, 10, DTYPE_FLOAT32, DEVICE_CPU, true));
    return m;
}

/** Fraction of `n` samples (features at `X`, class indices at `Y`) classified
 *  correctly, in fixed chunks to bound graph size. CNN reshapes to NCHW. */
static float evaluate(Sequential* model, const float* X, const float* Y, int n) {
    const int chunk  = 500;
    int correct      = 0;
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    for (int start = 0; start < n; start += chunk) {
        int b         = (start + chunk <= n) ? chunk : (n - start);
        int xs4[4]    = {b, 1, IMG, IMG};
        int xs2[2]    = {b, FEAT};
        Tensor* Xb    = use_cnn() ? cml_tensor((void*)&X[start * FEAT], xs4, 4, &cfg)
                                  : cml_tensor((void*)&X[start * FEAT], xs2, 2, &cfg);
        Tensor* logit = module_forward((Module*)model, Xb);
        float* lp     = (float*)tensor_data_ptr(logit);
        if (lp)
            for (int i = 0; i < b; i++) {
                int best    = 0;
                float bestv = lp[i * 10];
                for (int c = 1; c < 10; c++)
                    if (lp[i * 10 + c] > bestv) {
                        bestv = lp[i * 10 + c];
                        best  = c;
                    }
                if (best == (int)(Y[start + i] + 0.5f))
                    correct++;
            }
        tensor_free(logit);
        tensor_free(Xb);
        cml_ir_reset_global_context();
    }
    return (float)correct / (float)n;
}

int main(void) {
    cml_init();
    cml_seed(42);
    cml_enable_grad();

    printf("Loading MNIST (downloads + caches on first run)...\n");
    Dataset* full = cml_dataset_load("mnist");
    if (!full || !full->X || !full->y || full->num_samples < 1000) {
        printf("MNIST unavailable (no network / download failed). Skipping.\n");
        cml_cleanup();
        return 0;
    }

    int n_total       = full->num_samples; /* 70000 = 60k train + 10k test */
    int n_train       = n_total - N_TEST;
    const float* allX = (const float*)tensor_data_ptr(full->X);
    const float* allY = (const float*)tensor_data_ptr(full->y);
    printf("Loaded %d samples (%d train, %d test), %dx%d, pixels in [0,1], model=%s\n", n_total,
           n_train, N_TEST, IMG, IMG, use_cnn() ? "CNN" : "MLP");

    Dataset* train_ds = dataset_from_arrays((float*)allX, (float*)allY, n_train, FEAT, 1);
    if (!train_ds) {
        printf("Failed to build training dataset\n");
        cml_cleanup();
        return 1;
    }
    train_ds->num_classes = 10;
    DataLoader* loader    = dataloader_create(train_ds, 64, true);

    Sequential* model = build_model();
    module_set_training((Module*)model, true);
    Parameter** params = NULL;
    int num_params     = 0;
    module_collect_parameters((Module*)model, &params, &num_params, true);
    Optimizer* opt = optim_adam(params, num_params, 0.001f, 0.0f, 0.9f, 0.999f, 1e-8f);

    const int epochs = 3;
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

    for (int epoch = 0; epoch < epochs; epoch++) {
        dataloader_reset(loader);
        double running = 0.0;
        int steps      = 0;
        Batch* batch;
        while ((batch = dataloader_next_batch(loader)) != NULL) {
            int b      = batch->batch_size;
            int xs4[4] = {b, 1, IMG, IMG};
            int xs2[2] = {b, FEAT};
            Tensor* Xb = use_cnn() ? cml_tensor(tensor_data_ptr(batch->X), xs4, 4, &cfg)
                                   : cml_tensor(tensor_data_ptr(batch->X), xs2, 2, &cfg);
            Tensor* Yb = cml_tensor_1d((float*)tensor_data_ptr(batch->y), b);

            optimizer_zero_grad(opt);
            Tensor* out  = module_forward((Module*)model, Xb);
            Tensor* loss = tensor_sparse_cross_entropy_loss(out, Yb);
            float* lp    = (float*)tensor_data_ptr(loss);
            running += lp ? lp[0] : 0.0;
            steps++;

            tensor_backward(loss, NULL, false, false);
            optimizer_step(opt);

            tensor_free(loss);
            tensor_free(out);
            tensor_free(Xb);
            tensor_free(Yb);
            batch_free(batch);
            /* Reuse the warm graph/plan/kernel caches across steps: the graph
             * is structurally identical every step, so a full reset would
             * re-decompose and re-plan from scratch (~2x slower) for nothing. */
            cml_ir_reset_graph_only();

            if (steps % 200 == 0)
                printf("  epoch %d  step %4d  mean loss %.4f\n", epoch + 1, steps, running / steps);
        }
        printf("epoch %d done: mean train loss %.4f over %d steps\n", epoch + 1,
               steps ? running / steps : 0.0, steps);
    }

    module_set_training((Module*)model, false);
    float acc = evaluate(model, &allX[n_train * FEAT], &allY[n_train], N_TEST);
    printf("\nMNIST test accuracy = %.4f (%d samples, %s)\n", (double)acc, N_TEST,
           use_cnn() ? "CNN" : "MLP");

    dataloader_free(loader);
    dataset_free(train_ds);
    optimizer_free(opt);
    cml_free(params);
    module_free((Module*)model);
    cml_cleanup();
    return 0;
}
