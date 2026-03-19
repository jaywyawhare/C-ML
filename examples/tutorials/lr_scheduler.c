#include "cml.h"
#include "core/training_loop.h"
#include <stdio.h>

#define N_EPOCHS 50

static Sequential* make_model(int input_size) {
    Sequential* m = cml_nn_sequential();
    cml_nn_sequential_add(m, (Module*)cml_nn_linear(input_size, 16, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(m, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(m, (Module*)cml_nn_linear(16, 1, DTYPE_FLOAT32, DEVICE_CPU, true));
    return m;
}

static void train_with_scheduler(const char* name, Sequential* model,
                                  Tensor* X, Tensor* y,
                                  Optimizer* opt, LRScheduler* sched,
                                  int epochs) {
    printf("\n%s\n", name);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        Tensor* pred = cml_nn_sequential_forward(model, X);
        Tensor* loss = cml_nn_mse_loss(pred, y);
        cml_optim_zero_grad(opt);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(opt);

        float current_lr = lr_scheduler_get_lr(sched);
        if (epoch % 10 == 0 || epoch == 1)
            printf("  Epoch %3d  Loss: %.6f  LR: %.6f\n",
                   epoch, tensor_get_float(loss, 0), current_lr);

        lr_scheduler_update(sched, tensor_get_float(loss, 0));
    }
}

int main(void) {
    cml_init();
    printf("Example 14: Learning Rate Schedulers (Boston Housing)\n");

    Dataset* ds = cml_dataset_load("boston");
    if (!ds) { printf("Failed to load boston dataset\n"); return 1; }

    dataset_normalize(ds, "minmax");
    int nf = ds->input_size;

    printf("Samples: %d, Features: %d\n", ds->num_samples, nf);

    {
        Sequential* m = make_model(nf);
        Optimizer* opt = cml_optim_adam_for_model((Module*)m, 0.05f, 0.0f, 0.9f, 0.999f, 1e-8f);
        LRScheduler* sched = lr_scheduler_step(opt, 20, 0.5f);
        train_with_scheduler("StepLR (step=20, gamma=0.5)", m, ds->X, ds->y, opt, sched, N_EPOCHS);
        lr_scheduler_free(sched);
    }

    {
        Sequential* m = make_model(nf);
        Optimizer* opt = cml_optim_adam_for_model((Module*)m, 0.05f, 0.0f, 0.9f, 0.999f, 1e-8f);
        LRScheduler* sched = lr_scheduler_exponential(opt, 0.99f);
        train_with_scheduler("ExponentialLR (gamma=0.99)", m, ds->X, ds->y, opt, sched, N_EPOCHS);
        lr_scheduler_free(sched);
    }

    {
        Sequential* m = make_model(nf);
        Optimizer* opt = cml_optim_adam_for_model((Module*)m, 0.05f, 0.0f, 0.9f, 0.999f, 1e-8f);
        LRScheduler* sched = lr_scheduler_cosine(opt, N_EPOCHS, 0.001f);
        train_with_scheduler("CosineAnnealingLR (T_max=50, eta_min=0.001)", m, ds->X, ds->y, opt, sched, N_EPOCHS);
        lr_scheduler_free(sched);
    }

    {
        Sequential* m = make_model(nf);
        Optimizer* opt = cml_optim_adam_for_model((Module*)m, 0.05f, 0.0f, 0.9f, 0.999f, 1e-8f);
        LRScheduler* sched = lr_scheduler_reduce_on_plateau(opt, 0.5f, 5, 0.001f);
        train_with_scheduler("ReduceOnPlateau (factor=0.5, patience=5)", m, ds->X, ds->y, opt, sched, N_EPOCHS);
        lr_scheduler_free(sched);
    }

    printf("\nAll scheduler examples complete.\n");
    cml_cleanup();
    return 0;
}
