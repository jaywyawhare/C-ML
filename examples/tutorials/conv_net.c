#include "cml.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    cml_init();
    printf("Example 06: Image Classifier on Digits 8x8\n\n");

    Dataset* ds = cml_dataset_load("digits");
    if (!ds) { printf("Failed to load digits dataset\n"); return 1; }

    dataset_normalize(ds, "minmax");

    int n = ds->num_samples;
    int nf = ds->input_size;
    float* X_all = (float*)tensor_data_ptr(ds->X);
    float* y_all = (float*)tensor_data_ptr(ds->y);

    srand(42);
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
            for (int f = 0; f < nf; f++) {
            float tmp = X_all[i * nf + f];
            X_all[i * nf + f] = X_all[j * nf + f];
            X_all[j * nf + f] = tmp;
        }
        float tmp = y_all[i];
        y_all[i] = y_all[j];
        y_all[j] = tmp;
    }

    int n_train = (int)(n * 0.8f);
    int n_test = n - n_train;
    printf("Train: %d, Test: %d\n\n", n_train, n_test);

    float* train_labels = malloc(sizeof(float) * n_train);
    float* test_labels = malloc(sizeof(float) * n_test);
    for (int i = 0; i < n_train; i++)
        train_labels[i] = (y_all[i] < 5.0f) ? 0.0f : 1.0f;
    for (int i = 0; i < n_test; i++)
        test_labels[i] = (y_all[n_train + i] < 5.0f) ? 0.0f : 1.0f;

    int x_shape[] = {n_train, nf};
    int y_shape[] = {n_train, 1};
    Tensor* X = cml_tensor(X_all, x_shape, 2, NULL);
    Tensor* y = cml_tensor(train_labels, y_shape, 2, NULL);

    Sequential* model = cml_nn_sequential();
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(nf, 32, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(32, 16, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(16, 1, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_sigmoid());

    printf("Model:\n");
    cml_summary((Module*)model);

    Optimizer* opt = cml_optim_adam_for_model((Module*)model, 0.005f, 0.0f, 0.9f, 0.999f, 1e-8f);

    for (int epoch = 1; epoch <= 50; epoch++) {
        Tensor* pred = cml_nn_sequential_forward(model, X);
        Tensor* loss = cml_nn_bce_loss(pred, y);
        cml_optim_zero_grad(opt);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(opt);

        if (epoch % 10 == 0)
            printf("Epoch %3d  Loss: %.6f\n", epoch, tensor_get_float(loss, 0));
    }

    int tx_shape[] = {n_test, nf};
    Tensor* X_test = cml_tensor(&X_all[n_train * nf], tx_shape, 2, NULL);
    Tensor* pred = cml_nn_sequential_forward(model, X_test);
    int correct = 0;
    for (int i = 0; i < n_test; i++) {
        float p = tensor_get_float(pred, i);
        int cls = p > 0.5f ? 1 : 0;
        correct += (cls == (int)test_labels[i]);
    }
    printf("\nTest accuracy (low vs high digits): %d/%d (%.1f%%)\n",
           correct, n_test, correct / (float)n_test * 100);

    free(train_labels);
    free(test_labels);
    cml_cleanup();
    return 0;
}
