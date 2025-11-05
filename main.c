/**
 * @file main.c
 * @brief Training example using C-ML
 *
 * This trains a simple neural network on the XOR dataset.
 */

#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

int main() {
    if (cml_init() != 0) {
        printf("Failed to initialize C-ML library\n");
        return 1;
    }

    Sequential* model = nn_sequential();
    if (!model) {
        printf("Error: Failed to create Sequential model\n");
        cml_cleanup();
        return 1;
    }

    Linear* linear1   = nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true);
    ReLU* relu1       = nn_relu(false);
    Linear* linear2   = nn_linear(4, 4, DTYPE_FLOAT32, DEVICE_CPU, true);
    Tanh* tanh1       = nn_tanh();
    Linear* linear3   = nn_linear(4, 1, DTYPE_FLOAT32, DEVICE_CPU, true);
    Sigmoid* sigmoid1 = nn_sigmoid();

    if (!linear1 || !relu1 || !linear2 || !tanh1 || !linear3 || !sigmoid1) {
        printf("Error: Failed to create layers\n");
        if (linear1)
            module_free((Module*)linear1);
        if (relu1)
            module_free((Module*)relu1);
        if (linear2)
            module_free((Module*)linear2);
        if (tanh1)
            module_free((Module*)tanh1);
        if (linear3)
            module_free((Module*)linear3);
        if (sigmoid1)
            module_free((Module*)sigmoid1);
        module_free((Module*)model);
        cml_cleanup();
        return 1;
    }

    sequential_add(model, (Module*)linear1);
    sequential_add(model, (Module*)relu1);
    sequential_add(model, (Module*)linear2);
    sequential_add(model, (Module*)tanh1);
    sequential_add(model, (Module*)linear3);
    sequential_add(model, (Module*)sigmoid1);

    printf("Model: Sequential(Linear(2->4), ReLU, Linear(4->4), Tanh, Linear(4->1), Sigmoid)\n\n");

    summary((Module*)model);

    module_set_training((Module*)model, true);

    Parameter** params = NULL;
    int num_params     = 0;
    if (module_collect_parameters((Module*)model, &params, &num_params, true) != 0) {
        printf("Error: Failed to collect model parameters\n");
        module_free((Module*)model);
        cml_cleanup();
        return 1;
    }

    printf("Trainable parameters: %d\n", num_params);
    printf("Optimizer: Adam (lr=0.01)\n\n");

    Optimizer* optimizer = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    if (!optimizer) {
        printf("Error: Failed to create optimizer\n");
        CM_FREE(params);
        module_free((Module*)model);
        cml_cleanup();
        return 1;
    }

    int num_samples = 4;
    int input_size  = 2;
    int output_size = 1;

    int input_shape[] = {num_samples, input_size};
    Tensor* X         = tensor_empty(input_shape, 2, DTYPE_FLOAT32, DEVICE_CPU);
    if (!X) {
        printf("Error: Failed to create input tensor\n");
        optimizer_free(optimizer);
        CM_FREE(params);
        module_free((Module*)model);
        cml_cleanup();
        return 1;
    }

    int target_shape[] = {num_samples, output_size};
    Tensor* y          = tensor_empty(target_shape, 2, DTYPE_FLOAT32, DEVICE_CPU);
    if (!y) {
        printf("Error: Failed to create target tensor\n");
        tensor_free(X);
        optimizer_free(optimizer);
        CM_FREE(params);
        module_free((Module*)model);
        cml_cleanup();
        return 1;
    }

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

    tensor_free(y);
    tensor_free(X);
    optimizer_free(optimizer);
    CM_FREE(params);
    module_free((Module*)model);

    cml_cleanup();

    return 0;
}
