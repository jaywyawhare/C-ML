#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// Generate synthetic classification dataset with learnable pattern
void generate_dataset(float* X, float* y, int num_samples, int input_size, int seed) {
    srand(seed);

    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < input_size; j++) {
            X[i * input_size + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }

        float weighted_sum = 0.0f;
        for (int j = 0; j < input_size; j++) {
            float weight = (j % 2 == 0) ? 1.0f : -0.5f;
            weighted_sum += weight * X[i * input_size + j];
        }

        float threshold = 0.3f;
        y[i]            = (weighted_sum > threshold) ? 1.0f : 0.0f;
    }
}

int main() {
    Sequential* model    = NULL;
    Parameter** params   = NULL;
    Optimizer* optimizer = NULL;
    float* X_all         = NULL;
    float* y_all         = NULL;

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

    int input_size  = 16;
    int num_epochs  = 300;
    int num_samples = 500;
    int output_size = 1;

    Linear* linear1   = nn_linear(input_size, 64, DTYPE_FLOAT32, DEVICE_CPU, true);
    ReLU* relu1       = nn_relu(false);
    Linear* linear2   = nn_linear(64, 64, DTYPE_FLOAT32, DEVICE_CPU, true);
    ReLU* relu2       = nn_relu(false);
    Linear* linear3   = nn_linear(64, 128, DTYPE_FLOAT32, DEVICE_CPU, true);
    ReLU* relu3       = nn_relu(false);
    Linear* linear4   = nn_linear(128, 64, DTYPE_FLOAT32, DEVICE_CPU, true);
    Tanh* tanh1       = nn_tanh();
    Linear* linear5   = nn_linear(64, 32, DTYPE_FLOAT32, DEVICE_CPU, true);
    ReLU* relu4       = nn_relu(false);
    Linear* linear6   = nn_linear(32, 1, DTYPE_FLOAT32, DEVICE_CPU, true);
    Sigmoid* sigmoid1 = nn_sigmoid();

    if (!linear1 || !relu1 || !linear2 || !relu2 || !linear3 || !relu3 || !linear4 || !tanh1 ||
        !linear5 || !relu4 || !linear6 || !sigmoid1) {
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
    sequential_add(model, (Module*)tanh1);
    sequential_add(model, (Module*)linear5);
    sequential_add(model, (Module*)relu4);
    sequential_add(model, (Module*)linear6);
    sequential_add(model, (Module*)sigmoid1);

    summary((Module*)model);
    training_metrics_register_model((Module*)model);

    module_set_training((Module*)model, true);

    int num_params = 0;
    if (module_collect_parameters((Module*)model, &params, &num_params, true) != 0) {
        printf("Error: Failed to collect model parameters\n");
        goto cleanup;
    }
    cleanup_register_params(cleanup, params);

    optimizer = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    if (!optimizer) {
        printf("Error: Failed to create optimizer\n");
        goto cleanup;
    }
    cleanup_register_optimizer(cleanup, optimizer);

    X_all = CM_MALLOC(num_samples * input_size * sizeof(float));
    y_all = CM_MALLOC(num_samples * sizeof(float));

    if (!X_all || !y_all) {
        printf("Error: Failed to allocate dataset memory\n");
        goto cleanup;
    }
    cleanup_register_memory(cleanup, X_all);
    cleanup_register_memory(cleanup, y_all);

    generate_dataset(X_all, y_all, num_samples, input_size, 42);

    Dataset* full_dataset = dataset_from_arrays(X_all, y_all, num_samples, input_size, output_size);
    if (!full_dataset) {
        printf("Error: Failed to create dataset\n");
        goto cleanup;
    }
    cleanup_register_dataset(cleanup, full_dataset);

    float train_ratio      = 0.7f;
    float val_ratio        = 0.15f;
    Dataset* train_dataset = NULL;
    Dataset* val_dataset   = NULL;
    Dataset* test_dataset  = NULL;

    if (dataset_split_three(full_dataset, train_ratio, val_ratio, &train_dataset, &val_dataset,
                            &test_dataset) != 0) {
        printf("Error: Failed to split dataset\n");
        goto cleanup;
    }
    cleanup_register_dataset(cleanup, train_dataset);
    cleanup_register_dataset(cleanup, val_dataset);
    cleanup_register_dataset(cleanup, test_dataset);

    printf("Dataset split: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)\n\n",
           train_dataset->num_samples, train_ratio * 100.0f, val_dataset->num_samples,
           val_ratio * 100.0f, test_dataset->num_samples,
           (1.0f - train_ratio - val_ratio) * 100.0f);

    training_metrics_set_expected_epochs(num_epochs);

    printf("Training for %d epochs...\n\n", num_epochs);

    float best_loss     = INFINITY;
    float best_accuracy = 0.0f;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        optimizer_zero_grad(optimizer);

        Tensor* outputs = module_forward((Module*)model, train_dataset->X);
        if (!outputs) {
            printf("Error: Forward pass failed at epoch %d\n", epoch);
            break;
        }

        float* output_data = (float*)tensor_data_ptr(outputs);
        float* target_data = (float*)tensor_data_ptr(train_dataset->y);
        int correct        = 0;
        if (output_data && target_data) {
            for (int i = 0; i < train_dataset->num_samples; i++) {
                float pred   = output_data[i];
                float target = target_data[i];
                if ((target > 0.5f && pred > 0.5f) || (target <= 0.5f && pred <= 0.5f)) {
                    correct++;
                }
            }
        }
        float accuracy =
            train_dataset->num_samples > 0 ? (float)correct / train_dataset->num_samples : 0.0f;
        training_metrics_auto_capture_train_accuracy(accuracy);

        Tensor* loss = tensor_mse_loss(outputs, train_dataset->y);
        if (!loss) {
            printf("Error: Loss computation failed at epoch %d\n", epoch);
            tensor_free(outputs);
            break;
        }

        float* loss_data = (float*)tensor_data_ptr(loss);
        float epoch_loss = loss_data ? loss_data[0] : INFINITY;

        tensor_backward(loss, NULL, false, false);

        optimizer_step(optimizer);

        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
        }
        if (epoch_loss < best_loss) {
            best_loss = epoch_loss;
        }

        if (training_metrics_evaluate_dataset((Module*)model, val_dataset, tensor_mse_loss, true) !=
            0) {
            printf("Warning: Validation evaluation failed at epoch %d\n", epoch);
        }

        if ((epoch + 1) % 25 == 0 || (epoch < 25 && (epoch + 1) % 5 == 0) || epoch == 0) {
            TrainingMetrics* metrics = training_metrics_get_global();
            float val_loss = INFINITY, val_acc = 0.0f;
            if (metrics && metrics->epoch_validation_losses &&
                metrics->epoch_validation_accuracies) {
                val_loss = metrics->epoch_validation_losses[epoch];
                val_acc  = metrics->epoch_validation_accuracies[epoch];
            }
            printf("Epoch %4d/%d - Train Loss: %.6f, Train Acc: %.2f%%, Val Loss: %.6f, Val Acc: "
                   "%.2f%%\n",
                   epoch + 1, num_epochs, epoch_loss, accuracy * 100.0f,
                   val_loss != INFINITY ? val_loss : 0.0f,
                   val_acc != INFINITY ? val_acc * 100.0f : 0.0f);
        }

        tensor_free(loss);
        tensor_free(outputs);
    }

    printf("\nTraining completed!\n");
    printf("Best Training Loss: %.6f\n", best_loss);
    printf("Best Training Accuracy: %.2f%%\n", best_accuracy * 100.0f);

    printf("\n=== Final Evaluation ===\n");

    if (training_metrics_evaluate_dataset((Module*)model, train_dataset, tensor_mse_loss, false) ==
        0) {
        TrainingMetrics* metrics = training_metrics_get_global();
        if (metrics && metrics->epoch_training_losses && metrics->epoch_training_accuracies) {
            size_t last_epoch = num_epochs - 1;
            printf("Training Set   - Loss: %.6f, Accuracy: %.2f%%\n",
                   metrics->epoch_training_losses[last_epoch],
                   metrics->epoch_training_accuracies[last_epoch] * 100.0f);
        }
    }

    if (training_metrics_evaluate_dataset((Module*)model, val_dataset, tensor_mse_loss, true) ==
        0) {
        TrainingMetrics* metrics = training_metrics_get_global();
        if (metrics && metrics->epoch_validation_losses && metrics->epoch_validation_accuracies) {
            size_t last_epoch = num_epochs - 1;
            printf("Validation Set - Loss: %.6f, Accuracy: %.2f%%\n",
                   metrics->epoch_validation_losses[last_epoch],
                   metrics->epoch_validation_accuracies[last_epoch] * 100.0f);
        }
    }

    if (training_metrics_evaluate_dataset((Module*)model, test_dataset, tensor_mse_loss, false) ==
        0) {
        TrainingMetrics* metrics = training_metrics_get_global();
        if (metrics && metrics->epoch_testing_losses && metrics->epoch_testing_accuracies) {
            size_t last_epoch = num_epochs - 1;
            printf("Test Set       - Loss: %.6f, Accuracy: %.2f%%\n",
                   metrics->epoch_testing_losses[last_epoch],
                   metrics->epoch_testing_accuracies[last_epoch] * 100.0f);
        }
    }

cleanup:
    cleanup_context_free(cleanup);
    cml_cleanup();
    return 0;
}
