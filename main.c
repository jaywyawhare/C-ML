#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

int main(void) {
    cml_init();
    cml_seed(42);

    Sequential* model = cml_nn_sequential();
    DeviceType device = cml_get_default_device();
    DType dtype       = cml_get_default_dtype();

    model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(2, 4, dtype, device, true));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(4, 4, dtype, device, true));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_tanh());
    model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(4, 1, dtype, device, true));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_sigmoid());

    cml_summary((Module*)model);
    cml_nn_module_set_training((Module*)model, true);

    Optimizer* optimizer =
        cml_optim_adam_for_model((Module*)model, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    printf("Optimizer: Adam (lr=0.01)\n\n");

    Dataset* dataset = dataset_xor();
    Tensor* X        = dataset->X;
    Tensor* y        = dataset->y;

    int num_samples = dataset->num_samples;
    printf("Dataset: XOR problem (%d samples)\n\n", num_samples);

    // Training
    int num_epochs      = 200;
    float best_loss     = INFINITY;
    float best_accuracy = 0.0f;

    training_metrics_set_expected_epochs(num_epochs);
    printf("\nTraining for %d epochs...\n\n", num_epochs);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        cml_optim_zero_grad(optimizer);

        Tensor* outputs = cml_nn_module_forward((Module*)model, X);
        if (!outputs) {
            printf("Error: Forward pass failed at epoch %d\n", epoch);
            break;
        }

        Tensor* loss = cml_nn_mse_loss(outputs, y);
        if (!loss) {
            printf("Error: Loss computation failed at epoch %d\n", epoch);
            tensor_free(outputs);
            break;
        }

        float epoch_loss = tensor_get_float(loss, 0);
        if (epoch_loss < best_loss) {
            best_loss = epoch_loss;
        }

        int correct        = 0;
        float* output_data = (float*)tensor_data_ptr(outputs);
        float* target_data = (float*)tensor_data_ptr(y);
        if (output_data && target_data) {
            for (int i = 0; i < num_samples; i++) {
                if ((output_data[i] > 0.5f) == (target_data[i] > 0.5f)) {
                    correct++;
                }
            }
        }
        float accuracy = num_samples > 0 ? (float)correct / num_samples : 0.0f;

        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
        }

        cml_backward(loss, NULL, false, false);

        cml_optim_step(optimizer);

        // Capture training metrics AFTER backward (so loss is captured by cml_backward)
        training_metrics_auto_capture_train_accuracy(accuracy);

        if ((epoch + 1) % 50 == 0 || (epoch < 50 && (epoch + 1) % 10 == 0) || epoch == 0) {
            printf("Epoch %4d/%d - Loss: %.6f, Accuracy: %.2f%% (%d/%d)\n", epoch + 1, num_epochs,
                   (double)epoch_loss, (double)(accuracy * 100.0f), correct, num_samples);
        }

        tensor_free(loss);
        tensor_free(outputs);
    }

    printf("\nTraining completed!\n");
    printf("  Best loss: %.6f\n", (double)best_loss);
    printf("  Best accuracy: %.2f%%\n\n", (double)(best_accuracy * 100.0f));

    printf("Testing model on XOR dataset:\n");
    cml_nn_module_eval((Module*)model);

    float* X_data = (float*)tensor_data_ptr(X);
    float* y_data = (float*)tensor_data_ptr(y);

    if (!X_data || !y_data) {
        printf("Error: Failed to get dataset data pointers\n");
        cml_cleanup();
        return CML_HAS_ERRORS() ? 1 : 0;
    }

    int input_size = 2;

    for (int i = 0; i < num_samples; i++) {
        Tensor* sample_input = cml_zeros_2d(1, input_size);
        if (!sample_input)
            continue;

        float* sample_data = (float*)tensor_data_ptr(sample_input);
        if (!sample_data) {
            printf("Error: Failed to get sample input data\n");
            tensor_free(sample_input);
            continue;
        }

        sample_data[0] = X_data[i * input_size];
        sample_data[1] = X_data[i * input_size + 1];

        Tensor* prediction = cml_nn_module_forward((Module*)model, sample_input);
        if (prediction) {
            float* pred_data = (float*)tensor_data_ptr(prediction);
            if (pred_data) {
                float* target_val = &y_data[i];
                printf("  Input: [%.1f, %.1f] -> Target: %.1f, Prediction: %.4f\n",
                       (double)sample_data[0], (double)sample_data[1], (double)*target_val,
                       (double)pred_data[0]);
            } else {
                printf("  Input: [%.1f, %.1f] -> Target: %.1f, Prediction: <failed to execute>\n",
                       (double)sample_data[0], (double)sample_data[1], (double)y_data[i]);
            }
            tensor_free(prediction);
        }

        tensor_free(sample_input);
    }

    printf("\n");

    cml_cleanup();

    return CML_HAS_ERRORS() ? 1 : 0;
}
