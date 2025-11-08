/**
 * @file main.c
 * @brief Training example using C-ML
 *
 * This trains a simple neural network on the XOR dataset.
 */

#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

int main() {
    CleanupContext* cleanup = cleanup_context_create();
    if (!cleanup) {
        printf("Failed to create cleanup context\n");
        return 1;
    }

    if (cml_init() != 0) {
        printf("Failed to initialize C-ML library\n");
        cleanup_context_free(cleanup);
        return 1;
    }

    Sequential* model = nn_sequential();
    if (!model) {
        printf("Error: Failed to create Sequential model\n");
        goto cleanup;
    }
    cleanup_register_model(cleanup, (Module*)model);

    Linear* linear1   = nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true);
    ReLU* relu1       = nn_relu(false);
    Linear* linear2   = nn_linear(4, 4, DTYPE_FLOAT32, DEVICE_CPU, true);
    Tanh* tanh1       = nn_tanh();
    Linear* linear3   = nn_linear(4, 1, DTYPE_FLOAT32, DEVICE_CPU, true);
    Sigmoid* sigmoid1 = nn_sigmoid();

    if (!linear1 || !relu1 || !linear2 || !tanh1 || !linear3 || !sigmoid1) {
        printf("Error: Failed to create layers\n");
        goto cleanup;
    }

    sequential_add(model, (Module*)linear1);
    sequential_add(model, (Module*)relu1);
    sequential_add(model, (Module*)linear2);
    sequential_add(model, (Module*)tanh1);
    sequential_add(model, (Module*)linear3);
    sequential_add(model, (Module*)sigmoid1);

    printf("Model: Sequential(Linear(2->4), ReLU, Linear(4->4), Tanh, Linear(4->1), Sigmoid)\n\n");

    summary((Module*)model);
    training_metrics_register_model((Module*)model);

    module_set_training((Module*)model, true);

    Parameter** params = NULL;
    int num_params     = 0;
    if (module_collect_parameters((Module*)model, &params, &num_params, true) != 0) {
        printf("Error: Failed to collect model parameters\n");
        goto cleanup;
    }
    cleanup_register_params(cleanup, params);

    printf("Trainable parameters: %d\n", num_params);
    printf("Optimizer: Adam (lr=0.01)\n\n");

    Optimizer* optimizer = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    if (!optimizer) {
        printf("Error: Failed to create optimizer\n");
        goto cleanup;
    }
    cleanup_register_optimizer(cleanup, optimizer);

    int num_samples = 4;
    int input_size  = 2;
    int output_size = 1;

    int input_shape[] = {num_samples, input_size};
    Tensor* X         = tensor_empty(input_shape, 2, DTYPE_FLOAT32, DEVICE_CPU);
    if (!X) {
        printf("Error: Failed to create input tensor\n");
        goto cleanup;
    }
    cleanup_register_tensor(cleanup, X);

    int target_shape[] = {num_samples, output_size};
    Tensor* y          = tensor_empty(target_shape, 2, DTYPE_FLOAT32, DEVICE_CPU);
    if (!y) {
        printf("Error: Failed to create target tensor\n");
        goto cleanup;
    }
    cleanup_register_tensor(cleanup, y);

    float* X_data = (float*)tensor_data_ptr(X);
    float* y_data = (float*)tensor_data_ptr(y);

    X_data[0] = 0.0f;
    X_data[1] = 0.0f;
    y_data[0] = 0.0f;
    X_data[2] = 0.0f;
    X_data[3] = 1.0f;
    y_data[1] = 1.0f;
    X_data[4] = 1.0f;
    X_data[5] = 0.0f;
    y_data[2] = 1.0f;
    X_data[6] = 1.0f;
    X_data[7] = 1.0f;
    y_data[3] = 0.0f;

    printf("Dataset: XOR problem (%d samples)\n\n", num_samples);

    int num_epochs      = 500;
    float best_loss     = INFINITY;
    float best_accuracy = 0.0f;

    training_metrics_set_expected_epochs(num_epochs);

    printf("Training for %d epochs...\n\n", num_epochs);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int correct      = 0;

        optimizer_zero_grad(optimizer);

        Tensor* outputs = module_forward((Module*)model, X);
        if (!outputs) {
            printf("Error: Forward pass failed at epoch %d\n", epoch);
            break;
        }

        float* output_data = (float*)tensor_data_ptr(outputs);
        float* target_data = (float*)tensor_data_ptr(y);
        if (output_data && target_data) {
            for (int i = 0; i < num_samples; i++) {
                float pred   = output_data[i];
                float target = target_data[i];
                if ((target > 0.5f && pred > 0.5f) || (target <= 0.5f && pred <= 0.5f)) {
                    correct++;
                }
            }
        }
        float accuracy = num_samples > 0 ? (float)correct / num_samples : 0.0f;
        training_metrics_auto_capture_train_accuracy(accuracy);

        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
        }

        Tensor* loss = tensor_mse_loss(outputs, y);
        if (!loss) {
            printf("Error: Loss computation failed at epoch %d\n", epoch);
            tensor_free(outputs);
            break;
        }

        float* loss_data = (float*)tensor_data_ptr(loss);
        if (loss_data) {
            epoch_loss = loss_data[0];
            if (epoch_loss < best_loss) {
                best_loss = epoch_loss;
            }
        }

        tensor_backward(loss, NULL, false, false);
        optimizer_step(optimizer);

        if ((epoch + 1) % 50 == 0 || (epoch < 50 && (epoch + 1) % 10 == 0) || epoch == 0) {
            printf("Epoch %4d/%d - Loss: %.6f, Accuracy: %.2f%% (%d/%d)\n", epoch + 1, num_epochs,
                   epoch_loss, accuracy * 100.0f, correct, num_samples);
        }

        tensor_free(loss);
        tensor_free(outputs);
    }

    printf("\nTraining completed!\n");
    printf("  Best loss: %.6f\n", best_loss);
    printf("  Best accuracy: %.2f%%\n\n", best_accuracy * 100.0f);

    printf("Testing model on XOR dataset:\n");
    module_set_training((Module*)model, false);

    for (int i = 0; i < num_samples; i++) {
        int sample_shape[]   = {1, input_size};
        Tensor* sample_input = tensor_empty(sample_shape, 2, DTYPE_FLOAT32, DEVICE_CPU);
        if (!sample_input)
            continue;

        float* sample_data = (float*)tensor_data_ptr(sample_input);
        sample_data[0]     = X_data[i * input_size];
        sample_data[1]     = X_data[i * input_size + 1];

        Tensor* prediction = module_forward((Module*)model, sample_input);
        if (prediction) {
            float* pred_data  = (float*)tensor_data_ptr(prediction);
            float* target_val = &y_data[i];

            printf("  Input: [%.1f, %.1f] -> Target: %.1f, Prediction: %.4f\n", sample_data[0],
                   sample_data[1], *target_val, pred_data[0]);

            tensor_free(prediction);
        }

        tensor_free(sample_input);
    }

    printf("\n");

cleanup:
    cleanup_context_free(cleanup);
    cml_cleanup();

    return 0;
}
