/**
 * Example 04: Multi-Layer Perceptron (3-class classification)
 *
 * Architecture: Input(4) -> Dense(16) -> ReLU -> Dense(8) -> ReLU -> Dense(3) -> Sigmoid
 * Dataset: Iris (150 samples, 4 features, 3 classes).
 * Uses per-class BCE with one-hot targets (avoids cross-entropy gather backward).
 */
#include "cml.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    cml_init();
    printf("Example 04: MLP Classifier (Iris)\n\n");

    Dataset* ds = cml_dataset_load("iris");
    if (!ds) { printf("Failed to load iris dataset\n"); return 1; }

    dataset_normalize(ds, "minmax");
    Dataset *train, *test;
    dataset_split(ds, 0.8f, &train, &test);

    printf("Train: %d samples, Test: %d samples\n", train->num_samples, test->num_samples);
    printf("Features: %d, Classes: %d\n\n", train->input_size, ds->num_classes);

    int n_train = train->num_samples;
    int n_test = test->num_samples;
    int nc = 3;

    /* Create one-hot targets for BCE-based training */
    float* train_y_raw = (float*)tensor_data_ptr(train->y);
    float* onehot = calloc(n_train * nc, sizeof(float));
    for (int i = 0; i < n_train; i++) {
        int cls = (int)train_y_raw[i];
        if (cls >= 0 && cls < nc)
            onehot[i * nc + cls] = 1.0f;
    }
    int y_shape[] = {n_train, nc};
    Tensor* y_oh = cml_tensor(onehot, y_shape, 2, NULL);

    Sequential* model = cml_nn_sequential();
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(train->input_size, 16,
                          DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(16, 8, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(8, nc, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_sigmoid());

    printf("Model:\n");
    cml_summary((Module*)model);

    Optimizer* opt = cml_optim_adam_for_model((Module*)model, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);

    for (int epoch = 1; epoch <= 100; epoch++) {
        Tensor* pred = cml_nn_sequential_forward(model, train->X);
        Tensor* loss = cml_nn_bce_loss(pred, y_oh);
        cml_optim_zero_grad(opt);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(opt);

        if (epoch % 20 == 0)
            printf("Epoch %3d  Loss: %.6f\n", epoch, tensor_get_float(loss, 0));
    }

    /* Test evaluation: argmax over 3 outputs */
    float* test_y_raw = (float*)tensor_data_ptr(test->y);
    Tensor* test_pred = cml_nn_sequential_forward(model, test->X);
    tensor_ensure_executed(test_pred);
    int correct = 0;
    for (int i = 0; i < n_test; i++) {
        float max_val = -1e9f;
        int pred_cls = 0;
        int true_cls = (int)test_y_raw[i];
        for (int c = 0; c < nc; c++) {
            float v = tensor_get_float(test_pred, i * nc + c);
            if (v > max_val) { max_val = v; pred_cls = c; }
        }
        correct += (pred_cls == true_cls);
    }
    printf("\nTest accuracy: %d/%d (%.1f%%)\n", correct, n_test,
           correct / (float)n_test * 100);

    free(onehot);
    cml_cleanup();
    return 0;
}
