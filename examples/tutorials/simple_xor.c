/**
 * Simple XOR Neural Network Example
 *
 * This example demonstrates how to:
 * - Create a neural network
 * - Train it on the XOR problem
 * - Make predictions
 *
 * Compile:
 *   gcc simple_xor.c -I/usr/local/include/cml -lcml -lm -o simple_xor
 *
 * Run:
 *   ./simple_xor
 *   VIZ=1 ./simple_xor  (with visualization)
 */

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

int main() {
    printf("=== C-ML Simple XOR Example ===\n\n");

    // Initialize C-ML library
    cml_init();

    // XOR training data
    // Input: [x1, x2], Output: [y]
    float X_data[] = {
        0.0f, 0.0f, // XOR(0,0) = 0
        0.0f, 1.0f, // XOR(0,1) = 1
        1.0f, 0.0f, // XOR(1,0) = 1
        1.0f, 1.0f  // XOR(1,1) = 0
    };
    float y_data[] = {0.0f, 1.0f, 1.0f, 0.0f};

    // Create tensors
    int X_shape[] = {4, 2}; // 4 samples, 2 features
    int y_shape[] = {4, 1}; // 4 samples, 1 output

    Tensor* X = cml_tensor(X_data, X_shape, 2, NULL);
    Tensor* y = cml_tensor(y_data, y_shape, 2, NULL);

    printf("Training data created: 4 samples\n");

    // Build neural network
    // Architecture: 2 -> 8 -> 1 (with ReLU and Sigmoid activations)
    Sequential* model = cml_nn_sequential();
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(2, 8, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(8, 1, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_sigmoid());

    printf("\nModel architecture:\n");
    cml_summary((Module*)model);

    // Create Adam optimizer
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

    // Set expected number of epochs for visualization
    int epochs = 1000;
    training_metrics_set_expected_epochs(epochs);

    // Training loop
    for (int epoch = 1; epoch <= epochs; epoch++) {
        // Forward pass
        Tensor* output = cml_nn_sequential_forward(model, X);

        // Compute loss (Mean Squared Error)
        Tensor* loss = cml_nn_mse_loss(output, y);

        // Compute accuracy
        float accuracy    = 0.0f;
        int correct_count = 0;
        for (int i = 0; i < 4; i++) {
            float pred     = tensor_get_float(output, i);
            float expected = y_data[i];
            float rounded  = pred > 0.5f ? 1.0f : 0.0f;
            if (rounded == expected) {
                correct_count++;
            }
        }
        accuracy = (float)correct_count / 4.0f;

        // Backward pass
        cml_optim_zero_grad(optimizer);
        cml_backward(loss, NULL, false, false);

        // Update weights
        cml_optim_step(optimizer);

        // Auto-capture accuracy for visualization
        training_metrics_auto_capture_train_accuracy(accuracy);

        // Slow down training slightly to make visualization observable (5ms per epoch)
        cml_sleep_ms(5);

        // Print progress every 100 epochs
        if (epoch % 100 == 0 || epoch == epochs) {
            printf("%d\t\t%.6f\n", epoch, tensor_get_float(loss, 0));

            // Run validation (using same data for this simple example)
            // In a real app, you would use a separate validation set
            Tensor* val_output      = cml_nn_sequential_forward(model, X);
            Tensor* val_loss_tensor = cml_nn_mse_loss(val_output, y);

            float val_accuracy    = 0.0f;
            int val_correct_count = 0;
            for (int i = 0; i < 4; i++) {
                float pred     = tensor_get_float(val_output, i);
                float expected = y_data[i];
                float rounded  = pred > 0.5f ? 1.0f : 0.0f;
                if (rounded == expected) {
                    val_correct_count++;
                }
            }
            val_accuracy   = (float)val_correct_count / 4.0f;
            float val_loss = tensor_get_float(val_loss_tensor, 0);

            // Capture validation metrics
            training_metrics_auto_capture_validation(val_loss, val_accuracy);

            // Cleanup validation tensors (since we don't call backward on them)
            // Note: In a full implementation, we'd want a proper graph cleanup here
            // For now, we rely on the memory pool or OS cleanup for this small example
        }
    }

    printf("\nTraining complete!\n\n");

    // Test the trained model
    printf("Testing predictions:\n");
    printf("Input\t\tPrediction\tExpected\tCorrect?\n");
    printf("-----\t\t----------\t--------\t--------\n");

    Tensor* predictions      = cml_nn_sequential_forward(model, X);
    Tensor* test_loss_tensor = cml_nn_mse_loss(predictions, y);
    float test_loss          = tensor_get_float(test_loss_tensor, 0);

    int correct = 0;
    for (int i = 0; i < 4; i++) {
        float pred     = tensor_get_float(predictions, i);
        float expected = y_data[i];
        float rounded  = pred > 0.5f ? 1.0f : 0.0f;
        int is_correct = (rounded == expected);
        correct += is_correct;

        printf("[%.0f, %.0f]\t\t%.4f\t\t%.0f\t\t%s\n", X_data[i * 2], X_data[i * 2 + 1], pred,
               expected, is_correct ? "PASS" : "FAIL");
    }

    float test_accuracy = (float)correct / 4.0f;
    printf("\nAccuracy: %d/4 (%.0f%%)\n", correct, test_accuracy * 100.0f);

    // Capture test metrics
    training_metrics_auto_capture_test(test_loss, test_accuracy);

    // Cleanup
    cml_cleanup();

    printf("\n=== Example Complete ===\n");
    return 0;
}
