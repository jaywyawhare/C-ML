#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void generate_dataset(float* X, float* y, int num_samples, int input_size, int seed) {
    srand((unsigned int)seed);

    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < input_size; j++) {
            X[i * input_size + j] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        }

        float sum1 = 0.0f;
        float sum2 = 0.0f;
        for (int j = 0; j < input_size; j += 2) {
            float a = X[i * input_size + j];
            float b = (j + 1 < input_size) ? X[i * input_size + j + 1] : 0.0f;
            sum1 += (a * b > 0 ? 1.0f : -1.0f) * (a - b);
            sum2 += a * a - b * b;
        }

        float high_order = 1.0f;
        for (int j = 0; j < input_size && j < 4; j++) {
            high_order *= X[i * input_size + j];
        }

        float output_val = sum1 + 0.3f * sum2 + 2.0f * high_order;

        float threshold = 0.5f;
        y[i]            = (output_val > threshold) ? 1.0f : 0.0f;
    }
}

int main(void) {
    Sequential* model    = NULL;
    Parameter** params   = NULL;
    Optimizer* optimizer = NULL;
    Tensor* X            = NULL;
    Tensor* y            = NULL;

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

    model = nn_sequential();
    if (!model) {
        printf("Error: Failed to create Sequential model\n");
        goto cleanup;
    }
    cleanup_register_model(cleanup, (Module*)model);

    Linear* linear1  = nn_linear(8, 320, DTYPE_FLOAT32, DEVICE_CPU, true);
    ReLU* relu1      = nn_relu(false);
    Linear* linear2  = nn_linear(320, 32, DTYPE_FLOAT32, DEVICE_CPU, true);
    ReLU* relu2      = nn_relu(false);
    Linear* linear3  = nn_linear(32, 16, DTYPE_FLOAT32, DEVICE_CPU, true);
    ReLU* relu3      = nn_relu(false);
    Linear* linear4  = nn_linear(16, 1, DTYPE_FLOAT32, DEVICE_CPU, true);
    Sigmoid* sigmoid = nn_sigmoid();

    if (!linear1 || !relu1 || !linear2 || !relu2 || !linear3 || !relu3 || !linear4 || !sigmoid) {
        printf("Error: Failed to create layers\n");
        goto cleanup;
    }

    sequential_add(model, (Module*)linear1);
    sequential_add(model, (Module*)relu1);
    sequential_add(model, (Module*)linear2);
    sequential_add(model, (Module*)relu2);
    sequential_add(model, (Module*)linear3);
    sequential_add(model, (Module*)relu3);
    sequential_add(model, (Module*)linear4);
    sequential_add(model, (Module*)sigmoid);

    cml_summary((Module*)model);

    int num_params = 0;
    if (module_collect_parameters((Module*)model, &params, &num_params, true) != 0) {
        printf("Error: Failed to collect parameters\n");
        goto cleanup;
    }
    cleanup_register_params(cleanup, params);

    float initial_lr = 0.01f;
    optimizer        = optim_adam(params, num_params, initial_lr, 0.0f, 0.9f, 0.999f, 1e-8f);
    if (!optimizer) {
        printf("Error: Failed to create optimizer\n");
        goto cleanup;
    }
    cleanup_register_optimizer(cleanup, optimizer);

    int num_samples = 200;
    int input_size  = 8;
    int output_size = 1;

    int X_shape[] = {num_samples, input_size};
    int y_shape[] = {num_samples, output_size};

    TensorConfig config = (TensorConfig){
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    X = tensor_empty(X_shape, 2, &config);
    if (!X) {
        printf("Error: Failed to create input tensor\n");
        goto cleanup;
    }
    cleanup_register_tensor(cleanup, X);

    y = tensor_empty(y_shape, 2, &config);
    if (!y) {
        printf("Error: Failed to create target tensor\n");
        goto cleanup;
    }
    cleanup_register_tensor(cleanup, y);

    float* X_data = (float*)tensor_data_ptr(X);
    float* y_data = (float*)tensor_data_ptr(y);

    if (!X_data || !y_data) {
        printf("Error: Failed to get tensor data pointers\n");
        goto cleanup;
    }

    generate_dataset(X_data, y_data, num_samples, input_size, 42);

    int num_epochs        = 2000;
    int patience          = 15;
    float improvement_tol = 1e-5f;
    int lr_step_size      = 3;
    float lr_gamma        = 0.5f;

    float best_loss       = INFINITY;
    int no_improve_epochs = 0;
    int early_stopped_at  = -1;

    training_metrics_set_expected_epochs((size_t)num_epochs);

    TrainingMetrics* metrics = training_metrics_get_global();
    if (metrics) {
        training_metrics_set_learning_rate(metrics, initial_lr, "StepLR");
        char params_buf[128];
        snprintf(params_buf, sizeof(params_buf), "step_size=%d,gamma=%.2f", lr_step_size,
                 (double)lr_gamma);
        training_metrics_set_lr_schedule_params(metrics, params_buf);
    }

    printf("\nTraining Configuration\n");
    printf("Max epochs: %d\n", num_epochs);
    printf("Early stopping patience: %d\n", patience);
    printf("LR scheduler: StepLR (step_size=%d, gamma=%.2f)\n", lr_step_size, (double)lr_gamma);
    printf("Initial learning rate: %.4f\n", (double)initial_lr);
    printf("\nStarting training...\n\n");

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
        float accuracy = num_samples > 0 ? (float)correct / (float)num_samples : 0.0f;
        training_metrics_auto_capture_train_accuracy(accuracy);

        Tensor* loss = tensor_mse_loss(outputs, y);
        if (!loss) {
            printf("Error: Loss computation failed at epoch %d\n", epoch);
            tensor_free(outputs);
            break;
        }

        float* loss_data = (float*)tensor_data_ptr(loss);
        if (loss_data) {
            epoch_loss = loss_data[0];
        }

        tensor_backward(loss, NULL, false, false);
        optimizer_step(optimizer);

        if ((epoch + 1) % lr_step_size == 0) {
            float current_lr = optimizer_get_group_lr(optimizer, 0);
            float new_lr     = current_lr * lr_gamma;
            optimizer_set_lr(optimizer, new_lr);
            printf("  [Epoch %d] LR decayed: %.6f -> %.6f\n", epoch + 1, (double)current_lr,
                   (double)new_lr);
        }

        if (epoch_loss < best_loss - improvement_tol) {
            best_loss         = epoch_loss;
            no_improve_epochs = 0;
        } else {
            no_improve_epochs++;
            if (no_improve_epochs >= patience) {
                early_stopped_at = epoch + 1;
                printf("\nEarly Stopping Triggered\n");
                printf("Stopped at epoch %d (best loss: %.6f)\n", early_stopped_at,
                       (double)best_loss);
                printf("No improvement for %d epochs\n", patience);

                training_metrics_mark_early_stop((size_t)epoch);
                break;
            }
        }

        if ((epoch + 1) % 5 == 0 || epoch == 0) {
            float current_lr = optimizer_get_group_lr(optimizer, 0);
            printf("Epoch %3d/%d - Loss: %.6f, Acc: %.2f%%, LR: %.6f, No improve: %d/%d\n",
                   epoch + 1, num_epochs, (double)epoch_loss, (double)(accuracy * 100.0f),
                   (double)current_lr, no_improve_epochs, patience);
        }

        cml_reset_ir_context();
    }

    printf("\nTraining Summary\n");
    if (early_stopped_at > 0) {
        printf("Training stopped early at epoch %d (out of %d planned)\n", early_stopped_at,
               num_epochs);
        printf("Best loss achieved: %.6f\n", (double)best_loss);
    } else {
        printf("Training completed all %d epochs\n", num_epochs);
        printf("Final loss: %.6f\n", (double)best_loss);
    }

    metrics = training_metrics_get_global();
    if (metrics) {
        printf("Best accuracy: %.2f%%\n", (double)(metrics->best_accuracy * 100.0f));
        printf("Total training time: %.2f seconds\n", (double)metrics->total_time);
    }

cleanup:
    cml_cleanup();
    return 0;
}
