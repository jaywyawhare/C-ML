#include "cml.h"
#include <stdio.h>

#if defined(_WIN32)
#include <windows.h>
static void cml_sleep_ms(unsigned int ms) { Sleep(ms); }
#else
#define _XOPEN_SOURCE 700
#include <unistd.h>
static void cml_sleep_ms(unsigned int ms) { usleep(ms * 1000); }
#endif

static float compute_accuracy(Tensor* output, float* y_data, int n) {
    int correct = 0;
    for (int i = 0; i < n; i++) {
        float pred = tensor_get_float(output, i);
        float rounded = pred > 0.5f ? 1.0f : 0.0f;
        if (rounded == y_data[i]) correct++;
    }
    return (float)correct / (float)n;
}

int main() {
    printf("C-ML Simple XOR Example\n\n");

    cml_init();

    float X_data[] = {
        0.0f, 0.0f, // XOR(0,0) = 0
        0.0f, 1.0f, // XOR(0,1) = 1
        1.0f, 0.0f, // XOR(1,0) = 1
        1.0f, 1.0f  // XOR(1,1) = 0
    };
    float y_data[] = {0.0f, 1.0f, 1.0f, 0.0f};

    int X_shape[] = {4, 2};
    int y_shape[] = {4, 1};

    Tensor* X = cml_tensor(X_data, X_shape, 2, NULL);
    Tensor* y = cml_tensor(y_data, y_shape, 2, NULL);

    printf("Training data created: 4 samples\n");

    Sequential* model = cml_nn_sequential();
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(2, 8, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(8, 1, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_sigmoid());

    printf("\nModel architecture:\n");
    cml_summary((Module*)model);

    Optimizer* optimizer = cml_optim_adam_for_model((Module*)model,
                                                    0.01f,  // learning rate
                                                    0.0f,   // weight decay
                                                    0.9f,   // beta1
                                                    0.999f, // beta2
                                                    1e-8f   // epsilon
    );

    printf("\nStarting training...\n");
    printf("Epoch\t\tLoss\n");
    printf("-----\t\t----\n");

    int epochs = 1000;
    training_metrics_set_expected_epochs(epochs);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        Tensor* output = cml_nn_sequential_forward(model, X);
        Tensor* loss = cml_nn_mse_loss(output, y);

        float accuracy = compute_accuracy(output, y_data, 4);

        cml_optim_zero_grad(optimizer);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(optimizer);

        training_metrics_auto_capture_train_accuracy(accuracy);
        cml_sleep_ms(5);

        if (epoch % 100 == 0 || epoch == epochs) {
            printf("%d\t\t%.6f\n", epoch, tensor_get_float(loss, 0));

            Tensor* val_output      = cml_nn_sequential_forward(model, X);
            Tensor* val_loss_tensor = cml_nn_mse_loss(val_output, y);

            float val_accuracy = compute_accuracy(val_output, y_data, 4);
            float val_loss = tensor_get_float(val_loss_tensor, 0);

            training_metrics_auto_capture_validation(val_loss, val_accuracy);
        }
    }

    printf("\nTraining complete!\n\n");

    printf("Testing predictions:\n");
    printf("Input\t\tPrediction\tExpected\tCorrect?\n");
    printf("-----\t\t----------\t--------\t--------\n");

    Tensor* predictions      = cml_nn_sequential_forward(model, X);
    Tensor* test_loss_tensor = cml_nn_mse_loss(predictions, y);
    float test_loss          = tensor_get_float(test_loss_tensor, 0);

    for (int i = 0; i < 4; i++) {
        float pred     = tensor_get_float(predictions, i);
        float expected = y_data[i];
        float rounded  = pred > 0.5f ? 1.0f : 0.0f;
        int is_correct = (rounded == expected);

        printf("[%.0f, %.0f]\t\t%.4f\t\t%.0f\t\t%s\n", X_data[i * 2], X_data[i * 2 + 1], pred,
               expected, is_correct ? "PASS" : "FAIL");
    }

    float test_accuracy = compute_accuracy(predictions, y_data, 4);
    int correct = (int)(test_accuracy * 4.0f);
    printf("\nAccuracy: %d/4 (%.0f%%)\n", correct, test_accuracy * 100.0f);

    training_metrics_auto_capture_test(test_loss, test_accuracy);
    cml_cleanup();

    printf("\nExample Complete\n");
    return 0;
}
