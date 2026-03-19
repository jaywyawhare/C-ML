#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    cml_init();
    cml_seed(42);

    int input_size  = 8;
    int num_epochs  = 50;
    int num_samples = 100;
    int output_size = 1;

    Sequential* model = nn_sequential();
    DeviceType device = cml_get_default_device();
    DType dtype       = cml_get_default_dtype();
    model = sequential_add_chain(model, (Module*)nn_linear(input_size, 32, dtype, device, true));
    model = sequential_add_chain(model, (Module*)nn_relu(false));
    model = sequential_add_chain(model, (Module*)nn_linear(32, 16, dtype, device, true));
    model = sequential_add_chain(model, (Module*)nn_relu(false));
    model = sequential_add_chain(model, (Module*)nn_linear(16, 1, dtype, device, true));
    model = sequential_add_chain(model, (Module*)nn_sigmoid());

    cml_summary((Module*)model);
    module_set_training((Module*)model, true);

    Optimizer* optimizer = optim_adam_for_model((Module*)model, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    if (!optimizer) {
        printf("Error: Failed to create optimizer\n");
        return 1;
    }

    Dataset* full_dataset = dataset_random_classification(num_samples, input_size, output_size);
    if (!full_dataset) {
        printf("Error: Failed to create dataset\n");
        return 1;
    }

    float train_ratio      = 0.7f;
    float val_ratio        = 0.15f;
    Dataset* train_dataset = NULL;
    Dataset* val_dataset   = NULL;
    Dataset* test_dataset  = NULL;

    if (dataset_split_three(full_dataset, train_ratio, val_ratio, &train_dataset, &val_dataset,
                            &test_dataset) != 0) {
        printf("Error: Failed to split dataset\n");
        return 1;
    }

    printf("Dataset split: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)\n\n",
           train_dataset->num_samples, (double)(train_ratio * 100.0f), val_dataset->num_samples,
           (double)(val_ratio * 100.0f), test_dataset->num_samples,
           (double)((1.0f - train_ratio - val_ratio) * 100.0f));

    training_metrics_set_expected_epochs((size_t)num_epochs);

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
        float accuracy = train_dataset->num_samples > 0
                             ? (float)correct / (float)train_dataset->num_samples
                             : 0.0f;
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

        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            TrainingMetrics* metrics = training_metrics_get_global();
            float val_loss = INFINITY, val_acc = 0.0f;
            if (metrics && metrics->epoch_validation_losses &&
                metrics->epoch_validation_accuracies) {
                val_loss = metrics->epoch_validation_losses[epoch];
                val_acc  = metrics->epoch_validation_accuracies[epoch];
            }
            printf("Epoch %4d/%d - Train Loss: %.6f, Train Acc: %.2f%%, Val Loss: %.6f, Val Acc: "
                   "%.2f%%\n",
                   epoch + 1, num_epochs, (double)epoch_loss, (double)(accuracy * 100.0f),
                   !isinf(val_loss) ? (double)val_loss : 0.0,
                   !isinf(val_acc) ? (double)(val_acc * 100.0f) : 0.0);
        }

        tensor_free(loss);
        tensor_free(outputs);
    }

    printf("\nTraining completed!\n");
    printf("Best Training Loss: %.6f\n", (double)best_loss);
    printf("Best Training Accuracy: %.2f%%\n", (double)(best_accuracy * 100.0f));

    printf("\nFinal Evaluation\n");

    if (training_metrics_evaluate_dataset((Module*)model, train_dataset, tensor_mse_loss, false) ==
        0) {
        TrainingMetrics* metrics = training_metrics_get_global();
        if (metrics && metrics->epoch_training_losses && metrics->epoch_training_accuracies) {
            size_t last_epoch = (size_t)(num_epochs - 1);
            printf("Training Set   - Loss: %.6f, Accuracy: %.2f%%\n",
                   (double)metrics->epoch_training_losses[last_epoch],
                   (double)(metrics->epoch_training_accuracies[last_epoch] * 100.0f));
        }
    }

    if (training_metrics_evaluate_dataset((Module*)model, val_dataset, tensor_mse_loss, true) ==
        0) {
        TrainingMetrics* metrics = training_metrics_get_global();
        if (metrics && metrics->epoch_validation_losses && metrics->epoch_validation_accuracies) {
            size_t last_epoch = (size_t)(num_epochs - 1);
            printf("Validation Set - Loss: %.6f, Accuracy: %.2f%%\n",
                   (double)metrics->epoch_validation_losses[last_epoch],
                   (double)(metrics->epoch_validation_accuracies[last_epoch] * 100.0f));
        }
    }

    if (training_metrics_evaluate_dataset((Module*)model, test_dataset, tensor_mse_loss, false) ==
        0) {
        TrainingMetrics* metrics = training_metrics_get_global();
        if (metrics && metrics->epoch_testing_losses && metrics->epoch_testing_accuracies) {
            size_t last_epoch = (size_t)(num_epochs - 1);
            printf("Test Set       - Loss: %.6f, Accuracy: %.2f%%\n",
                   (double)metrics->epoch_testing_losses[last_epoch],
                   (double)(metrics->epoch_testing_accuracies[last_epoch] * 100.0f));
        }
    }

    cml_cleanup();
    return 0;
}
