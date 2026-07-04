/*
 * torch_c_example.c — PyTorch-like C API training demo
 *
 * Mirrors the linear_regression.c tutorial but uses the torch_c façade
 * inspired by ExecuTorch's eager-mode Module API (Section 5.2).
 */
#include "torch/torch_c.h"
#include "cml.h"
#include <stdio.h>

int main(void) {
    torch_init();
    torch_manual_seed(42);
    torch_set_default_device(DEVICE_CPU);
    torch_set_default_dtype(DTYPE_FLOAT32);

    printf("Torch C API Example: Linear Regression\n\n");

    Dataset* ds = cml_dataset_load("boston");
    if (!ds) {
        printf("Failed to load boston dataset\n");
        torch_cleanup();
        return 1;
    }
    dataset_normalize(ds, "minmax");

    Dataset *train, *test;
    dataset_split(ds, 0.8f, &train, &test);
    printf("Train: %d, Test: %d, Features: %d\n\n", train->num_samples, test->num_samples,
           train->input_size);

    Sequential* model = torch_nn_sequential();
    torch_nn_sequential_add(model, (Module*)torch_nn_linear(train->input_size, 1, true));

    Optimizer* opt = torch_optim_adam((Module*)model, 0.05f, 0.0f, 0.9f, 0.999f, 1e-8f);

    for (int epoch = 1; epoch <= 100; epoch++) {
        Tensor* pred = torch_nn_sequential_forward(model, train->X);
        Tensor* loss = torch_nn_mse_loss(pred, train->y);

        torch_optim_zero_grad(opt);
        torch_backward(loss, NULL, false, false);
        torch_optim_step(opt);

        if (epoch % 20 == 0 || epoch == 1)
            printf("Epoch %3d  Loss: %.6f\n", epoch, torch_tensor_item_float(loss));

        torch_reset_ir();
    }

    torch_module_eval((Module*)model);
    Tensor* test_pred = torch_nn_sequential_forward(model, test->X);

    float mse = 0.0f;
    for (int i = 0; i < test->num_samples; i++) {
        float p = tensor_get_float(test_pred, i);
        float t = tensor_get_float(test->y, i);
        mse += (p - t) * (p - t);
    }
    printf("\nTest MSE: %.6f\n", mse / test->num_samples);

    torch_tensor_free(test_pred);
    torch_reset_ir();
    torch_optim_free(opt);
    module_free((Module*)model);
    torch_cleanup();
    return 0;
}
