#include "cml.h"
#include <stdio.h>

int main(void) {
    cml_init();
    printf("Example 02: Linear Regression (Boston Housing)\n\n");

    Dataset* ds = cml_dataset_load("boston");
    if (!ds) { printf("Failed to load boston dataset\n"); return 1; }

    dataset_normalize(ds, "minmax");

    /* Also normalize targets to [0,1] for stable training */
    float* y_data = (float*)tensor_data_ptr(ds->y);
    float y_min = y_data[0], y_max = y_data[0];
    for (int i = 1; i < ds->num_samples; i++) {
        if (y_data[i] < y_min) y_min = y_data[i];
        if (y_data[i] > y_max) y_max = y_data[i];
    }
    float y_range = y_max - y_min;
    if (y_range < 1e-8f) y_range = 1.0f;
    for (int i = 0; i < ds->num_samples; i++) {
        y_data[i] = (y_data[i] - y_min) / y_range;
    }

    Dataset *train, *test;
    dataset_split(ds, 0.8f, &train, &test);

    printf("Train: %d samples, Test: %d samples, Features: %d\n\n",
           train->num_samples, test->num_samples, train->input_size);

    Sequential* model = cml_nn_sequential();
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(train->input_size, 1,
                          DTYPE_FLOAT32, DEVICE_CPU, true));

    cml_summary((Module*)model);
    Optimizer* opt = cml_optim_adam_for_model((Module*)model, 0.05f, 0.0f, 0.9f, 0.999f, 1e-8f);

    for (int epoch = 1; epoch <= 100; epoch++) {
        Tensor* pred = cml_nn_sequential_forward(model, train->X);
        Tensor* loss = cml_nn_mse_loss(pred, train->y);
        cml_optim_zero_grad(opt);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(opt);
        cml_reset_ir_context();

        if (epoch % 20 == 0 || epoch == 1)
            printf("Epoch %3d  Loss: %.6f\n", epoch, tensor_get_float(loss, 0));
    }

    Tensor* test_pred = cml_nn_sequential_forward(model, test->X);
    float mse = 0;
    for (int i = 0; i < test->num_samples; i++) {
        float p = tensor_get_float(test_pred, i) * y_range + y_min;
        float t = tensor_get_float(test->y, i) * y_range + y_min;
        mse += (p - t) * (p - t);
    }
    printf("\nTest MSE: %.6f (on %d samples)\n", mse / test->num_samples, test->num_samples);

    printf("\nSample predictions:\n");
    for (int i = 0; i < 5 && i < test->num_samples; i++) {
        printf("  pred=%.3f  target=%.3f\n",
               tensor_get_float(test_pred, i) * y_range + y_min,
               tensor_get_float(test->y, i) * y_range + y_min);
    }

    cml_cleanup();
    return 0;
}
