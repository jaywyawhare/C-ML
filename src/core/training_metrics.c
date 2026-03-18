#include "core/training_metrics.h"
#include "core/model_architecture.h"
#include "core/dataset.h"
#include "nn.h"
#include "autograd/autograd.h"
#include "optim.h"
#include "core/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static TrainingMetrics* g_global_metrics = NULL;
static size_t g_current_epoch            = 0;
static int g_optimizer_step_count        = 0;
static int g_last_epoch_step_count       = 0;
static double g_epoch_start_time         = 0.0;
static float g_last_loss                 = 0.0f;
static bool g_epoch_in_progress          = false;
static bool g_zero_grad_called           = false;
static Module* g_current_model           = NULL; 
static bool g_architecture_exported      = false;
static bool g_manual_epoch_control       = false;
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

TrainingMetrics* training_metrics_get_global(void) { return g_global_metrics; }
static int training_metrics_ensure_capacity(TrainingMetrics* metrics, size_t num_epochs) {
    if (!metrics)
        return -1;

    if (num_epochs <= metrics->num_epochs) {
        return 0;
    }

    size_t old_capacity = metrics->num_epochs;
    size_t new_capacity = num_epochs;
    metrics->epoch_training_losses =
        realloc(metrics->epoch_training_losses, (size_t)new_capacity * sizeof(float));
    metrics->epoch_training_accuracies =
        realloc(metrics->epoch_training_accuracies, (size_t)new_capacity * sizeof(float));
    metrics->epoch_times = realloc(metrics->epoch_times, (size_t)new_capacity * sizeof(float));
    metrics->epoch_learning_rates =
        realloc(metrics->epoch_learning_rates, (size_t)new_capacity * sizeof(float));
    if (metrics->epoch_validation_losses) {
        metrics->epoch_validation_losses =
            realloc(metrics->epoch_validation_losses, (size_t)new_capacity * sizeof(float));
        metrics->epoch_validation_accuracies =
            realloc(metrics->epoch_validation_accuracies, (size_t)new_capacity * sizeof(float));
    }
    if (metrics->epoch_testing_losses) {
        metrics->epoch_testing_losses =
            realloc(metrics->epoch_testing_losses, (size_t)new_capacity * sizeof(float));
        metrics->epoch_testing_accuracies =
            realloc(metrics->epoch_testing_accuracies, (size_t)new_capacity * sizeof(float));
    }

    if (!metrics->epoch_training_losses || !metrics->epoch_training_accuracies ||
        !metrics->epoch_times || !metrics->epoch_learning_rates) {
        return -1;
    }

    for (size_t i = old_capacity; i < new_capacity; i++) {
        metrics->epoch_training_losses[i]     = 0.0f;
        metrics->epoch_training_accuracies[i] = 0.0f;
        metrics->epoch_times[i]               = 0.0f;
        metrics->epoch_learning_rates[i]      = 0.0f;

        if (metrics->epoch_validation_losses) {
            metrics->epoch_validation_losses[i]     = INFINITY;
            metrics->epoch_validation_accuracies[i] = INFINITY;
        }

        if (metrics->epoch_testing_losses) {
            metrics->epoch_testing_losses[i]     = INFINITY;
            metrics->epoch_testing_accuracies[i] = INFINITY;
        }
    }

    metrics->num_epochs = new_capacity;
    return 0;
}

TrainingMetrics* training_metrics_create(size_t num_epochs) {
    TrainingMetrics* metrics = malloc(sizeof(TrainingMetrics));
    if (!metrics)
        return NULL;

    metrics->num_epochs = num_epochs;
    metrics->epoch_training_losses     = malloc(num_epochs * sizeof(float));
    metrics->epoch_training_accuracies = malloc(num_epochs * sizeof(float));
    metrics->epoch_testing_losses     = NULL;
    metrics->epoch_testing_accuracies = NULL;
    metrics->epoch_validation_losses     = NULL;
    metrics->epoch_validation_accuracies = NULL;

    metrics->epoch_times      = malloc(num_epochs * sizeof(float));
    metrics->total_time       = 0.0f;
    metrics->epoch_start_time = 0;
    metrics->epoch_learning_rates = malloc(num_epochs * sizeof(float));

    metrics->best_loss        = 0.0f;
    metrics->best_accuracy    = 0.0f;
    metrics->total_params     = 0;
    metrics->trainable_params = 0;
    metrics->model_summary    = NULL;
    metrics->learning_rate      = 0.0f;
    metrics->lr_schedule        = NULL;
    metrics->lr_schedule_params = NULL;
    metrics->gradient_norm      = 0.0f;
    metrics->loss_reduction_rate = 0.0f;
    metrics->loss_stability      = 0.0f;
    metrics->early_stopped   = false;
    metrics->expected_epochs = num_epochs;
    metrics->actual_epochs   = num_epochs; // Initially same as expected

    if (!metrics->epoch_training_losses || !metrics->epoch_training_accuracies ||
        !metrics->epoch_times || !metrics->epoch_learning_rates) {
        if (metrics->epoch_training_losses)
            free(metrics->epoch_training_losses);
        if (metrics->epoch_training_accuracies)
            free(metrics->epoch_training_accuracies);
        if (metrics->epoch_times)
            free(metrics->epoch_times);
        if (metrics->epoch_learning_rates)
            free(metrics->epoch_learning_rates);
        free(metrics);
        return NULL;
    }

    for (size_t i = 0; i < num_epochs; i++) {
        metrics->epoch_training_losses[i]     = 0.0f;
        metrics->epoch_training_accuracies[i] = 0.0f;
        metrics->epoch_times[i]               = 0.0f;
        metrics->epoch_learning_rates[i]      = 0.0f;
    }

    return metrics;
}

void training_metrics_start_epoch(TrainingMetrics* metrics) {
    if (!metrics)
        return;
    metrics->epoch_start_time = clock();
}

void training_metrics_record_epoch(TrainingMetrics* metrics, size_t epoch, float loss,
                                   float accuracy) {
    if (!metrics || epoch >= metrics->num_epochs)
        return;
    if (metrics->epoch_start_time > 0) {
        clock_t epoch_end     = clock();
        double epoch_time_sec = ((double)(epoch_end - metrics->epoch_start_time)) / CLOCKS_PER_SEC;
        metrics->epoch_times[epoch] = (float)epoch_time_sec;
        metrics->total_time += (float)epoch_time_sec;
    }
    metrics->epoch_training_losses[epoch]     = loss;
    metrics->epoch_training_accuracies[epoch] = accuracy;

    if (epoch == 0 || loss < metrics->best_loss) {
        metrics->best_loss = loss;
    }
    if (epoch == 0 || accuracy > metrics->best_accuracy) {
        metrics->best_accuracy = accuracy;
    }
}

void training_metrics_record_epoch_full(TrainingMetrics* metrics, size_t epoch, float train_loss,
                                        float train_accuracy, float test_loss, float test_accuracy,
                                        float val_loss, float val_accuracy) {
    if (!metrics || epoch >= metrics->num_epochs)
        return;

    metrics->epoch_training_losses[epoch]     = train_loss;
    metrics->epoch_training_accuracies[epoch] = train_accuracy;

    if (!isinf(test_loss) && !isinf(test_accuracy) && !isnan(test_loss) && !isnan(test_accuracy) &&
        test_loss >= 0.0f && test_accuracy >= 0.0f) {
        if (!metrics->epoch_testing_losses) {
            metrics->epoch_testing_losses     = malloc(metrics->num_epochs * sizeof(float));
            metrics->epoch_testing_accuracies = malloc(metrics->num_epochs * sizeof(float));
            if (metrics->epoch_testing_losses && metrics->epoch_testing_accuracies) {
                for (size_t i = 0; i < metrics->num_epochs; i++) {
                    metrics->epoch_testing_losses[i]     = INFINITY;
                    metrics->epoch_testing_accuracies[i] = INFINITY;
                }
            }
        }
        if (metrics->epoch_testing_losses && metrics->epoch_testing_accuracies) {
            metrics->epoch_testing_losses[epoch]     = test_loss;
            metrics->epoch_testing_accuracies[epoch] = test_accuracy;
        }
    }

    if (!isinf(val_loss) && !isinf(val_accuracy) && !isnan(val_loss) && !isnan(val_accuracy) &&
        val_loss >= 0.0f && val_accuracy >= 0.0f) {
        if (!metrics->epoch_validation_losses) {
            metrics->epoch_validation_losses     = malloc(metrics->num_epochs * sizeof(float));
            metrics->epoch_validation_accuracies = malloc(metrics->num_epochs * sizeof(float));
            if (metrics->epoch_validation_losses && metrics->epoch_validation_accuracies) {
                for (size_t i = 0; i < metrics->num_epochs; i++) {
                    metrics->epoch_validation_losses[i]     = INFINITY;
                    metrics->epoch_validation_accuracies[i] = INFINITY;
                }
            }
        }
        if (metrics->epoch_validation_losses && metrics->epoch_validation_accuracies) {
            metrics->epoch_validation_losses[epoch]     = val_loss;
            metrics->epoch_validation_accuracies[epoch] = val_accuracy;
        }
    }
    if (epoch == 0 || train_loss < metrics->best_loss) {
        metrics->best_loss = train_loss;
    }
    if (epoch == 0 || train_accuracy > metrics->best_accuracy) {
        metrics->best_accuracy = train_accuracy;
    }
}

void training_metrics_set_summary(TrainingMetrics* metrics, const char* summary) {
    if (!metrics)
        return;

    if (metrics->model_summary) {
        free(metrics->model_summary);
    }

    if (summary) {
        size_t len             = strlen(summary);
        metrics->model_summary = malloc(len + 1);
        if (metrics->model_summary) {
            memcpy(metrics->model_summary, summary, len);
            metrics->model_summary[len] = '\0';
        }
    } else {
        metrics->model_summary = NULL;
    }
}

void training_metrics_set_params(TrainingMetrics* metrics, int total, int trainable) {
    if (!metrics)
        return;
    metrics->total_params     = total;
    metrics->trainable_params = trainable;
}

void training_metrics_record_epoch_time(TrainingMetrics* metrics, size_t epoch,
                                        float time_seconds) {
    if (!metrics || epoch >= metrics->num_epochs)
        return;
    metrics->epoch_times[epoch] = time_seconds;
    metrics->total_time += time_seconds;
}

void training_metrics_set_learning_rate(TrainingMetrics* metrics, float lr, const char* scheduler) {
    if (!metrics)
        return;
    metrics->learning_rate = lr;

    if (metrics->lr_schedule) {
        free(metrics->lr_schedule);
    }

    if (scheduler) {
        size_t len           = strlen(scheduler);
        metrics->lr_schedule = malloc(len + 1);
        if (metrics->lr_schedule) {
            memcpy(metrics->lr_schedule, scheduler, len);
            metrics->lr_schedule[len] = '\0';
        }
    } else {
        metrics->lr_schedule = NULL;
    }
}

void training_metrics_set_lr_schedule_params(TrainingMetrics* metrics, const char* params) {
    if (!metrics)
        return;

    if (metrics->lr_schedule_params) {
        free(metrics->lr_schedule_params);
    }

    if (params) {
        size_t len                  = strlen(params);
        metrics->lr_schedule_params = malloc(len + 1);
        if (metrics->lr_schedule_params) {
            memcpy(metrics->lr_schedule_params, params, len);
            metrics->lr_schedule_params[len] = '\0';
        }
    } else {
        metrics->lr_schedule_params = NULL;
    }
}

void training_metrics_set_gradient_norm(TrainingMetrics* metrics, float grad_norm) {
    if (!metrics)
        return;
    metrics->gradient_norm = grad_norm;
}

float training_metrics_calculate_gradient_norm(TrainingMetrics* metrics, void** parameters,
                                               int num_parameters) {
    if (!metrics || !parameters || num_parameters == 0)
        return 0.0f;

    float grad_norm_squared = 0.0f;
    int params_with_grad    = 0;

    for (int i = 0; i < num_parameters; i++) {
        Parameter* param = (Parameter*)parameters[i];
        if (!param || !param->tensor)
            continue;

        Tensor* grad = tensor_get_grad(param->tensor);
        if (!grad)
            continue;

        float* grad_data = (float*)tensor_data_ptr(grad);
        if (!grad_data)
            continue;

        size_t num_elements = grad->numel; // Use numel field directly
        if (num_elements == 0 && grad->shape && grad->ndim > 0) {
            num_elements = 1;
            for (int d = 0; d < grad->ndim; d++) {
                num_elements *= (size_t)grad->shape[d];
            }
        }

        for (size_t j = 0; j < num_elements; j++) {
            float g = grad_data[j];
            grad_norm_squared += g * g;
        }
        params_with_grad++;
    }

    if (params_with_grad == 0)
        return 0.0f;

    float grad_norm        = sqrtf(grad_norm_squared);
    metrics->gradient_norm = grad_norm;
    return grad_norm;
}

static void write_json_escaped(FILE* f, const char* s) {
    if (!s) {
        fputs("null", f);
        return;
    }
    fputc('"', f);
    for (const char* p = s; *p; p++) {
        if (*p == '"' || *p == '\\') {
            fputc('\\', f);
            fputc(*p, f);
        } else if ((unsigned char)*p < 0x20) {
            fprintf(f, "\\u%04x", (unsigned char)*p);
        } else
            fputc(*p, f);
    }
    fputc('"', f);
}

int training_metrics_export_json(const TrainingMetrics* metrics, const char* path,
                                 bool incremental) {
    if (!metrics || !path)
        return -1;
    char tmp_path[1024];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);

    FILE* f = fopen(tmp_path, "wb");
    if (!f)
        return -2;

    fputs("{\n", f);
    fputs("  \"model_summary\": ", f);
    write_json_escaped(f, metrics->model_summary);
    fputs(",\n", f);

    fprintf(f, "  \"total_params\": %d,\n", metrics->total_params);
    fprintf(f, "  \"trainable_params\": %d,\n", metrics->trainable_params);
    fprintf(f, "  \"best_loss\": %.6f,\n", (double)metrics->best_loss);
    fprintf(f, "  \"best_accuracy\": %.6f,\n", (double)metrics->best_accuracy);
    size_t current_epoch = 0;
    for (size_t i = 0; i < metrics->num_epochs; i++) {
        if (metrics->epoch_training_losses && metrics->epoch_training_losses[i] > 0.0f) {
            current_epoch = i + 1;
        }
    }
    fprintf(f, "  \"early_stopped\": %s,\n", metrics->early_stopped ? "true" : "false");
    if (metrics->early_stopped) {
        fprintf(f, "  \"expected_epochs\": %zu,\n", metrics->expected_epochs);
        fprintf(f, "  \"actual_epochs\": %zu,\n", metrics->actual_epochs);
    } else {
        fprintf(f, "  \"expected_epochs\": %zu,\n", metrics->expected_epochs);
    }
    size_t display_num_epochs = metrics->num_epochs;
    if (metrics->expected_epochs > 0) {
        if (current_epoch <= metrics->expected_epochs) {
            display_num_epochs = metrics->expected_epochs;
        } else {
            display_num_epochs = current_epoch;
        }
    }
    fprintf(f, "  \"num_epochs\": %zu,\n", display_num_epochs);

    fprintf(f, "  \"current_epoch\": %zu,\n", current_epoch);
    fprintf(f, "  \"is_training\": %s,\n", incremental ? "true" : "false");

    size_t num_epochs_to_export = incremental ? metrics->num_epochs : metrics->num_epochs;
    for (size_t i = metrics->num_epochs; i > 0; i--) {
        if (metrics->epoch_training_losses && metrics->epoch_training_losses[i - 1] > 0.0f) {
            num_epochs_to_export = i;
            break;
        }
    }
    float loss_reduction_rate = 0.0f;
    float loss_stability      = 0.0f;

    if (num_epochs_to_export >= 2) {
        float last_loss = metrics->epoch_training_losses[num_epochs_to_export - 1];
        float prev_loss = metrics->epoch_training_losses[num_epochs_to_export - 2];
        if (prev_loss > 0.0f) {
            loss_reduction_rate = ((prev_loss - last_loss) / prev_loss) * 100.0f;
        }
    }
    if (num_epochs_to_export >= 5) {
        size_t window_size = (num_epochs_to_export >= 10) ? 10 : num_epochs_to_export;
        size_t start_idx   = num_epochs_to_export - window_size;

        float mean = 0.0f;
        for (size_t i = start_idx; i < num_epochs_to_export; i++) {
            mean += metrics->epoch_training_losses[i];
        }
        mean /= (float)window_size;

        float variance = 0.0f;
        for (size_t i = start_idx; i < num_epochs_to_export; i++) {
            float diff = metrics->epoch_training_losses[i] - mean;
            variance += diff * diff;
        }
        variance /= (float)window_size;
        loss_stability = sqrtf(variance);
    }
    float avg_epoch_time              = 0.0f;
    size_t completed_epochs_with_time = 0;
    for (size_t i = 0; i < num_epochs_to_export; i++) {
        if (metrics->epoch_times[i] > 0.0f) {
            avg_epoch_time += metrics->epoch_times[i];
            completed_epochs_with_time++;
        }
    }
    if (completed_epochs_with_time > 0) {
        avg_epoch_time /= (float)completed_epochs_with_time;
    }
    float epochs_per_hour = 0.0f;
    if (avg_epoch_time > 0.0f) {
        epochs_per_hour = 3600.0f / avg_epoch_time;
    }
    float estimated_remaining = 0.0f;
    if (avg_epoch_time > 0.0f && current_epoch < metrics->num_epochs) {
        estimated_remaining = avg_epoch_time * (float)(metrics->num_epochs - current_epoch);
    }
    fprintf(f, "  \"epoch_time\": %.6f,\n",
            (double)(avg_epoch_time > 0.0f ? avg_epoch_time : 0.0f));
    fprintf(f, "  \"time_per_epoch\": %.6f,\n",
            (double)(avg_epoch_time > 0.0f ? avg_epoch_time : 0.0f));
    fprintf(f, "  \"total_time\": %.6f,\n",
            (double)(metrics->total_time > 0.0f ? metrics->total_time : 0.0f));
    fprintf(f, "  \"elapsed_time\": %.6f,\n",
            (double)(metrics->total_time > 0.0f ? metrics->total_time : 0.0f));
    fprintf(f, "  \"estimated_remaining\": %.6f,\n",
            (double)(estimated_remaining > 0.0f ? estimated_remaining : 0.0f));
    fprintf(f, "  \"epochs_per_hour\": %.6f,\n",
            (double)(epochs_per_hour > 0.0f ? epochs_per_hour : 0.0f));
    fprintf(f, "  \"learning_rate\": %.8f,\n",
            (double)(metrics->learning_rate > 0.0f ? metrics->learning_rate : 0.0f));
    fprintf(f, "  \"lr\": %.8f,\n",
            (double)(metrics->learning_rate > 0.0f ? metrics->learning_rate : 0.0f));
    fputs("  \"epoch_learning_rates\": [", f);
    for (size_t i = 0; i < num_epochs_to_export; i++) {
        if (i > 0)
            fputs(", ", f);
        fprintf(f, "%.8f",
                (double)(metrics->epoch_learning_rates[i] > 0.0f ? metrics->epoch_learning_rates[i]
                                                                 : 0.0f));
    }
    fputs("],\n", f);
    if (metrics->lr_schedule) {
        fputs("  \"lr_schedule\": ", f);
        write_json_escaped(f, metrics->lr_schedule);
        fputs(",\n", f);
        fputs("  \"learning_rate_schedule\": ", f);
        write_json_escaped(f, metrics->lr_schedule);
        fputs(",\n", f);
        fputs("  \"scheduler\": ", f);
        write_json_escaped(f, metrics->lr_schedule);
        fputs(",\n", f);
    }
    if (metrics->lr_schedule_params) {
        fputs("  \"lr_schedule_params\": ", f);
        write_json_escaped(f, metrics->lr_schedule_params);
        fputs(",\n", f);
    }
    fprintf(f, "  \"gradient_norm\": %.8f,\n",
            (double)(metrics->gradient_norm > 0.0f ? metrics->gradient_norm : 0.0f));
    fprintf(f, "  \"grad_norm\": %.8f,\n",
            (double)(metrics->gradient_norm > 0.0f ? metrics->gradient_norm : 0.0f));
    fprintf(f, "  \"gradient_norm_avg\": %.8f,\n",
            (double)(metrics->gradient_norm > 0.0f ? metrics->gradient_norm : 0.0f));
    fprintf(f, "  \"loss_reduction_rate\": %.6f,\n", (double)loss_reduction_rate);
    fprintf(f, "  \"loss_stability\": %.8f,\n", (double)loss_stability);
    fputs("  \"epoch_training_losses\": [", f);
    for (size_t i = 0; i < num_epochs_to_export; i++) {
        if (i > 0)
            fputs(", ", f);
        fprintf(f, "%.6f", (double)metrics->epoch_training_losses[i]);
    }
    fputs("],\n", f);

    fputs("  \"epoch_training_accuracies\": [", f);
    for (size_t i = 0; i < num_epochs_to_export; i++) {
        if (i > 0)
            fputs(", ", f);
        fprintf(f, "%.6f", (double)metrics->epoch_training_accuracies[i]);
    }
    fputs("],\n", f);
    // Find the last non-INFINITY test values (test is typically evaluated once at the end)
    float test_loss     = INFINITY;
    float test_accuracy = INFINITY;
    bool has_test_data  = false;

    if (metrics->epoch_testing_losses) {
        // Find the last non-INFINITY value (check original values, not converted ones)
        // Test is typically evaluated once at the end, so check all epochs (not just
        // num_epochs_to_export) Search from the last epoch backwards to find test metrics
        for (size_t i = metrics->num_epochs; i > 0; i--) {
            size_t idx = i - 1;
            // Check if value is set (not INFINITY and not NaN)
            // Don't require > 0.0f since test loss could theoretically be 0
            if (!isinf(metrics->epoch_testing_losses[idx]) &&
                !isnan(metrics->epoch_testing_losses[idx])) {
                test_loss     = metrics->epoch_testing_losses[idx];
                has_test_data = true;
                if (metrics->epoch_testing_accuracies &&
                    !isinf(metrics->epoch_testing_accuracies[idx]) &&
                    !isnan(metrics->epoch_testing_accuracies[idx])) {
                    test_accuracy = metrics->epoch_testing_accuracies[idx];
                }
                break;
            }
        }

        // Export as single float values (preferred for UI)
        // Only export if test data exists (skip null fields)
        // Export both naming variants for UI compatibility (UI checks testing_loss first)
        if (has_test_data && !isinf(test_loss)) {
            fprintf(f, "  \"testing_loss\": %.6f,\n", (double)test_loss);
            fprintf(f, "  \"test_loss\": %.6f,\n", (double)test_loss);
        }

        if (has_test_data && !isinf(test_accuracy)) {
            fprintf(f, "  \"testing_accuracy\": %.6f,\n", (double)test_accuracy);
            fprintf(f, "  \"test_accuracy\": %.6f,\n", (double)test_accuracy);
        }

        // Only export arrays if they have actual data (not all zeros)
        // Skip arrays entirely if no test data exists to reduce JSON size
        if (has_test_data) {
            fputs("  \"epoch_testing_losses\": [", f);
            for (size_t i = 0; i < num_epochs_to_export; i++) {
                if (i > 0)
                    fputs(", ", f);
                float val = (isinf(metrics->epoch_testing_losses[i]) ||
                             isnan(metrics->epoch_testing_losses[i]))
                                ? 0.0f
                                : metrics->epoch_testing_losses[i];
                fprintf(f, "%.6f", (double)val);
            }
            fputs("],\n", f);

            fputs("  \"epoch_testing_accuracies\": [", f);
            for (size_t i = 0; i < num_epochs_to_export; i++) {
                if (i > 0)
                    fputs(", ", f);
                float val = (isinf(metrics->epoch_testing_accuracies[i]) ||
                             isnan(metrics->epoch_testing_accuracies[i]))
                                ? 0.0f
                                : metrics->epoch_testing_accuracies[i];
                fprintf(f, "%.6f", (double)val);
            }
            fputs("],\n", f);
        }
    }
    if (metrics->epoch_validation_losses) {
        fputs("  \"epoch_validation_losses\": [", f);
        for (size_t i = 0; i < num_epochs_to_export; i++) {
            if (i > 0)
                fputs(", ", f);
            float val = (isinf(metrics->epoch_validation_losses[i]) ||
                         isnan(metrics->epoch_validation_losses[i]))
                            ? 0.0f
                            : metrics->epoch_validation_losses[i];
            fprintf(f, "%.6f", (double)val);
        }
        fputs("],\n", f);

        fputs("  \"epoch_validation_accuracies\": [", f);
        for (size_t i = 0; i < num_epochs_to_export; i++) {
            if (i > 0)
                fputs(", ", f);
            float val = (isinf(metrics->epoch_validation_accuracies[i]) ||
                         isnan(metrics->epoch_validation_accuracies[i]))
                            ? 0.0f
                            : metrics->epoch_validation_accuracies[i];
            fprintf(f, "%.6f", (double)val);
        }
        fputs("],\n", f);
    }
    if (metrics->model_summary && strstr(metrics->model_summary, "\"layers\"")) {
        fputs("  \"architecture\": ", f);
        write_json_escaped(f, metrics->model_summary);
        fputs("\n", f);
    } else {
    }

    fputs("}\n", f);
    fclose(f);
    rename(tmp_path, path);

    return 0;
}

int training_metrics_export_epoch_update(const TrainingMetrics* metrics, size_t epoch,
                                         const char* path) {
    if (!metrics || !path || epoch >= metrics->num_epochs)
        return -1;
    return training_metrics_export_json(metrics, path, true);
}

int training_metrics_export_architecture(Module* module, const char* path) {
    if (!module || !path)
        return -1;

    ModelArchitecture* arch = model_architecture_create();
    if (!arch)
        return -1;

    if (model_architecture_extract(module, arch) != 0) {
        model_architecture_free(arch);
        return -1;
    }

    int result = model_architecture_export_json(arch, path);
    model_architecture_free(arch);

    return result;
}

void training_metrics_log(TrainingMetrics* metrics, size_t epoch) {
    if (!metrics)
        return;
    float epoch_loss = 0.0f;

    if (epoch < metrics->num_epochs) {
        epoch_loss = metrics->epoch_training_losses[epoch];
    }
    float avg_epoch_time    = 0.0f;
    size_t completed_epochs = 0;
    for (size_t i = 0; i <= epoch && i < metrics->num_epochs; i++) {
        if (metrics->epoch_times[i] > 0.0f) {
            avg_epoch_time += metrics->epoch_times[i];
            completed_epochs++;
        }
    }
    if (completed_epochs > 0) {
        avg_epoch_time /= (float)completed_epochs;
    }
    float epochs_per_hour = 0.0f;
    if (avg_epoch_time > 0.0f) {
        epochs_per_hour = 3600.0f / avg_epoch_time;
    }
    float estimated_remaining = 0.0f;
    if (avg_epoch_time > 0.0f && epoch < metrics->num_epochs) {
        estimated_remaining = avg_epoch_time * (float)(metrics->num_epochs - epoch - 1);
    }
    float loss_reduction_rate = 0.0f;
    if (epoch >= 1 && epoch < metrics->num_epochs) {
        float prev_loss = metrics->epoch_training_losses[epoch - 1];
        if (prev_loss > 0.0f) {
            loss_reduction_rate = ((prev_loss - epoch_loss) / prev_loss) * 100.0f;
        }
    }
    float loss_stability = 0.0f;
    if (epoch >= 4) {
        size_t window_size = (epoch >= 9) ? 10 : (epoch + 1);
        size_t start_idx   = epoch + 1 - window_size;

        float mean = 0.0f;
        for (size_t i = start_idx; i <= epoch && i < metrics->num_epochs; i++) {
            mean += metrics->epoch_training_losses[i];
        }
        mean /= (float)window_size;

        float variance = 0.0f;
        for (size_t i = start_idx; i <= epoch && i < metrics->num_epochs; i++) {
            float diff = metrics->epoch_training_losses[i] - mean;
            variance += diff * diff;
        }
        variance /= (float)window_size;
        loss_stability = sqrtf(variance);
    }
    const char* time_unit = "s";
    float time_display    = avg_epoch_time;
    if (avg_epoch_time >= 3600) {
        time_display = avg_epoch_time / 3600.0f;
        time_unit    = "hr";
    } else if (avg_epoch_time >= 60) {
        time_display = avg_epoch_time / 60.0f;
        time_unit    = "min";
    }
    const char* epochs_unit = "/hr";
    float epochs_display    = epochs_per_hour;
    if (epochs_per_hour < 1.0f && epochs_per_hour * 60 >= 1.0f) {
        epochs_display = epochs_per_hour * 60.0f;
        epochs_unit    = "/min";
    } else if (epochs_per_hour * 60 < 1.0f) {
        epochs_display = epochs_per_hour * 3600.0f;
        epochs_unit    = "/sec";
    }
    printf("\nTraining Metrics (Epoch %zu)\n", epoch + 1);
    printf("Time/Epoch:        %.2f %s\n", (double)time_display, time_unit);
    printf("Total Time:       %.2f min\n", (double)(metrics->total_time / 60.0f));
    printf("Est. Remaining:   %.2f min\n", (double)(estimated_remaining / 60.0f));
    printf("Epochs/Hour:      %.2f %s\n", (double)epochs_display, epochs_unit);
    if (metrics->learning_rate > 0.0f) {
        printf("Learning Rate:     %.8f", (double)metrics->learning_rate);
        if (metrics->lr_schedule) {
            printf(" (%s)", metrics->lr_schedule);
        }
        printf("\n");
    } else {
        printf("Learning Rate:     N/A\n");
    }
    if (metrics->gradient_norm > 0.0f) {
        printf("Gradient Norm:     %.8f\n", (double)metrics->gradient_norm);
    } else {
        printf("Gradient Norm:     N/A\n");
    }
    printf("Reduction Rate:    %.2f%%\n", (double)loss_reduction_rate);
    printf("Loss Stability (σ): %.8f\n", (double)loss_stability);
    printf("\n");
}

void training_metrics_free(TrainingMetrics* metrics) {
    if (!metrics)
        return;

    if (metrics->epoch_training_losses)
        free(metrics->epoch_training_losses);
    if (metrics->epoch_training_accuracies)
        free(metrics->epoch_training_accuracies);
    if (metrics->epoch_testing_losses)
        free(metrics->epoch_testing_losses);
    if (metrics->epoch_testing_accuracies)
        free(metrics->epoch_testing_accuracies);
    if (metrics->epoch_validation_losses)
        free(metrics->epoch_validation_losses);
    if (metrics->epoch_validation_accuracies)
        free(metrics->epoch_validation_accuracies);
    if (metrics->epoch_times)
        free(metrics->epoch_times);
    if (metrics->epoch_learning_rates)
        free(metrics->epoch_learning_rates);
    if (metrics->model_summary)
        free(metrics->model_summary);
    if (metrics->lr_schedule)
        free(metrics->lr_schedule);
    if (metrics->lr_schedule_params)
        free(metrics->lr_schedule_params);
    free(metrics);
}

int training_metrics_step(Module* model, Tensor* X, Tensor* y, Tensor* (*loss_fn)(Tensor*, Tensor*),
                          Optimizer* optimizer, size_t epoch, float* loss_out,
                          float* accuracy_out) {
    if (!model || !X || !y || !loss_fn || !optimizer || !loss_out || !accuracy_out) {
        LOG_ERROR("Invalid parameters for training_step");
        return -1;
    }
    TrainingMetrics* metrics = (TrainingMetrics*)optimizer->training_metrics;
    if (metrics) {
        training_metrics_start_epoch(metrics);
    }
    module_set_training(model, true);
    optimizer_zero_grad(optimizer);

    Tensor* outputs = module_forward(model, X);
    if (!outputs) {
        LOG_ERROR("Forward pass failed");
        return -1;
    }
    float* output_data = (float*)tensor_data_ptr(outputs);
    float* target_data = (float*)tensor_data_ptr(y);
    int correct        = 0;
    int num_samples    = X->shape[0];

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
    *accuracy_out  = accuracy;
    Tensor* loss = loss_fn(outputs, y);
    if (!loss) {
        LOG_ERROR("Loss computation failed");
        tensor_free(outputs);
        return -1;
    }

    float* loss_data = (float*)tensor_data_ptr(loss);
    *loss_out        = loss_data ? loss_data[0] : INFINITY;
    tensor_backward(loss, NULL, false, false);
    float grad_norm = 0.0f;
    if (metrics) {
        Parameter** params = NULL;
        int num_params     = 0;
        if (module_collect_parameters(model, &params, &num_params, true) == 0 && params) {
            grad_norm =
                training_metrics_calculate_gradient_norm(metrics, (void**)params, num_params);
            free(params);
        }
    }
    optimizer_step(optimizer);
    if (metrics) {
        training_metrics_record_epoch(metrics, epoch, *loss_out, accuracy);
        float current_lr = optimizer_get_group_lr(optimizer, 0);
        if (current_lr > 0.0f) {
            training_metrics_set_learning_rate(metrics, current_lr, "Constant");
        }
        if (grad_norm > 0.0f) {
            training_metrics_set_gradient_norm(metrics, grad_norm);
        }
        const char* viz     = getenv("CML_VIZ");
        const char* viz_env = getenv("VIZ");
        if ((viz && viz[0] != '\0') ||
            (viz_env && (viz_env[0] == '1' || strcmp(viz_env, "true") == 0))) {
            const char* metrics_path = "training.json";
            training_metrics_export_epoch_update(metrics, epoch, metrics_path);
        }
    }

    tensor_free(loss);
    tensor_free(outputs);

    return 0;
}
void training_metrics_mark_zero_grad(void) { g_zero_grad_called = true; }
static void training_metrics_auto_detect_epoch(Optimizer* optimizer) {
    if (!g_global_metrics || !optimizer)
        return;
    if (g_manual_epoch_control) {
        g_zero_grad_called = false; // Reset flag to prevent accumulation
        return;
    }

    if (g_zero_grad_called && g_epoch_in_progress) {
        if (g_epoch_start_time > 0.0) {
            double epoch_end      = get_time_sec();
            double epoch_time_sec = epoch_end - g_epoch_start_time;
            training_metrics_ensure_capacity(g_global_metrics, g_current_epoch + 1);
            if (g_current_epoch < g_global_metrics->num_epochs) {
                g_global_metrics->epoch_times[g_current_epoch] = (float)epoch_time_sec;
                g_global_metrics->total_time += (float)epoch_time_sec;
                const char* metrics_path = "training.json";
                training_metrics_export_json(g_global_metrics, metrics_path, true);
            }
        }

        g_current_epoch++;
        g_epoch_start_time = get_time_sec();
        g_zero_grad_called = false; // Reset flag
    } else if (g_zero_grad_called && !g_epoch_in_progress) {
        g_current_epoch     = 1;
        g_epoch_start_time  = get_time_sec();
        g_epoch_in_progress = true;
        g_zero_grad_called  = false;
    }
}
void training_metrics_set_expected_epochs(size_t num_epochs) {
    if (!g_global_metrics)
        return;
    g_manual_epoch_control = true;
    g_current_epoch        = 0; // Reset epoch counter for manual control
    if (num_epochs > g_global_metrics->num_epochs) {
        training_metrics_ensure_capacity(g_global_metrics, num_epochs);
    }
    g_global_metrics->expected_epochs = num_epochs;
    g_global_metrics->num_epochs = num_epochs;
    if (g_global_metrics->actual_epochs == 0 || g_global_metrics->actual_epochs > num_epochs) {
        g_global_metrics->actual_epochs = num_epochs;
    }
    const char* metrics_path = "training.json";
    training_metrics_export_json(g_global_metrics, metrics_path, true);
}

void training_metrics_mark_early_stop(size_t actual_epochs) {
    if (!g_global_metrics)
        return;
    g_global_metrics->early_stopped = true;
    g_global_metrics->actual_epochs = actual_epochs + 1;
    if (g_global_metrics->actual_epochs < g_global_metrics->num_epochs) {
        g_global_metrics->num_epochs = g_global_metrics->actual_epochs;
    }
    const char* metrics_path = "training.json";
    training_metrics_export_json(g_global_metrics, metrics_path, false); // Final export
}

void training_metrics_register_model(Module* model) {
    if (!model)
        return;
    g_current_model = model;
    if (!g_architecture_exported) {
        training_metrics_auto_export_architecture(model);
    }
}
void training_metrics_auto_export_architecture(Module* model) {
    if (!model)
        return;
    if (!g_current_model) {
        g_current_model = model;
    }

    if (g_architecture_exported)
        return;
    const char* arch_path = "model_architecture.json";
    if (training_metrics_export_architecture(model, arch_path) == 0) {
        g_architecture_exported = true;
        g_current_model         = model;
        if (g_global_metrics) {
            FILE* f = fopen(arch_path, "rb");
            if (f) {
                fseek(f, 0, SEEK_END);
                long size = ftell(f);
                fseek(f, 0, SEEK_SET);

                if (size > 0 && size < 1024 * 1024) { // Limit to 1MB
                    char* json_str = malloc((size_t)size + 1);
                    if (json_str) {
                        size_t read    = fread(json_str, 1, (size_t)size, f);
                        json_str[read] = '\0';
                        training_metrics_set_summary(g_global_metrics, json_str);

                        free(json_str);
                    }
                }
                fclose(f);
                const char* metrics_path = "training.json";
                training_metrics_export_json(g_global_metrics, metrics_path, true);
            }
        }
    }
}
void training_metrics_auto_capture_loss(Tensor* loss_tensor) {
    if (!g_global_metrics || !loss_tensor)
        return;
    if (g_current_model && !g_architecture_exported) {
        training_metrics_auto_export_architecture(g_current_model);
    }
    bool is_scalar = (loss_tensor->ndim == 0) ||
                     (loss_tensor->ndim == 1 && loss_tensor->shape[0] == 1) ||
                     (loss_tensor->numel == 1);

    if (is_scalar) {
        float loss_value = 0.0f;
        if (loss_tensor->data) {
            if (loss_tensor->dtype == DTYPE_FLOAT32) {
                loss_value = ((float*)loss_tensor->data)[0];
            }
        }

        training_metrics_ensure_capacity(g_global_metrics, g_current_epoch + 1);

        if (g_current_epoch < g_global_metrics->num_epochs) {
            g_global_metrics->epoch_training_losses[g_current_epoch] = loss_value;
            g_last_loss                                              = loss_value;
            if (g_current_epoch == 0 || loss_value < g_global_metrics->best_loss ||
                fabsf(g_global_metrics->best_loss) < 1e-6f) {
                g_global_metrics->best_loss = loss_value;
            }
            const char* metrics_path = "training.json";
            training_metrics_export_json(g_global_metrics, metrics_path, true);
        }
    }
}
void training_metrics_auto_capture_optimizer(Optimizer* optimizer) {
    if (!g_global_metrics || !optimizer)
        return;
    if (g_current_model && !g_architecture_exported) {
        training_metrics_auto_export_architecture(g_current_model);
    }
    training_metrics_auto_detect_epoch(optimizer);
    if (optimizer->num_param_groups > 0) {
        float lr = optimizer->param_groups[0].lr;
        if (lr > 0.0f) {
            g_global_metrics->learning_rate = lr;
            training_metrics_ensure_capacity(g_global_metrics, g_current_epoch + 1);
            if (g_current_epoch < g_global_metrics->num_epochs) {
                g_global_metrics->epoch_learning_rates[g_current_epoch] = lr;
            }
        }
    }
    if (optimizer->param_groups && optimizer->num_param_groups > 0) {
        ParameterGroup* group = &optimizer->param_groups[0];
        if (group->parameters && group->num_parameters > 0) {
            float grad_norm = training_metrics_calculate_gradient_norm(
                g_global_metrics, (void**)group->parameters, group->num_parameters);
            if (grad_norm > 0.0f) {
                g_global_metrics->gradient_norm = grad_norm;
            }
        }
    }
    const char* metrics_path = "training.json";
    training_metrics_export_json(g_global_metrics, metrics_path, true);
}
void training_metrics_init_global(void) {
    if (g_global_metrics)
        return; // Already initialized
    g_global_metrics = training_metrics_create(100);
    if (g_global_metrics) {
        g_current_epoch         = 0;
        g_optimizer_step_count  = 0;
        g_last_epoch_step_count = 0;
        g_epoch_start_time      = 0;
        g_last_loss             = 0.0f;
        g_epoch_in_progress     = false;
        g_current_model         = NULL;
        g_architecture_exported = false;
    }
}
void training_metrics_auto_capture_train_accuracy(float train_accuracy) {
    if (!g_global_metrics)
        return;
    training_metrics_ensure_capacity(g_global_metrics, g_current_epoch + 1);

    if (g_current_epoch < g_global_metrics->num_epochs) {
        g_global_metrics->epoch_training_accuracies[g_current_epoch] = train_accuracy;

        if (g_current_epoch == 0 || train_accuracy > g_global_metrics->best_accuracy) {
            g_global_metrics->best_accuracy = train_accuracy;
        }
        const char* metrics_path = "training.json";
        training_metrics_export_json(g_global_metrics, metrics_path, true);
    }
}
void training_metrics_auto_capture_validation(float val_loss, float val_accuracy) {
    if (!g_global_metrics)
        return;
    if (!g_global_metrics->epoch_validation_losses) {
        g_global_metrics->epoch_validation_losses =
            malloc(g_global_metrics->num_epochs * sizeof(float));
        g_global_metrics->epoch_validation_accuracies =
            malloc(g_global_metrics->num_epochs * sizeof(float));
        if (!g_global_metrics->epoch_validation_losses ||
            !g_global_metrics->epoch_validation_accuracies) {
            return;
        }
            for (size_t i = 0; i < g_global_metrics->num_epochs; i++) {
            g_global_metrics->epoch_validation_losses[i]     = INFINITY;
            g_global_metrics->epoch_validation_accuracies[i] = INFINITY;
        }
    }
    training_metrics_ensure_capacity(g_global_metrics, g_current_epoch + 1);
    size_t epoch_index = g_current_epoch > 0 ? g_current_epoch - 1 : 0;
    if (epoch_index < g_global_metrics->num_epochs) {
        g_global_metrics->epoch_validation_losses[epoch_index]     = val_loss;
        g_global_metrics->epoch_validation_accuracies[epoch_index] = val_accuracy;
        const char* metrics_path = "training.json";
        training_metrics_export_json(g_global_metrics, metrics_path, true);
    }
}
void training_metrics_auto_capture_test(float test_loss, float test_accuracy) {
    if (!g_global_metrics)
        return;
    if (!g_global_metrics->epoch_testing_losses) {
        g_global_metrics->epoch_testing_losses =
            malloc(g_global_metrics->num_epochs * sizeof(float));
        g_global_metrics->epoch_testing_accuracies =
            malloc(g_global_metrics->num_epochs * sizeof(float));
        if (!g_global_metrics->epoch_testing_losses ||
            !g_global_metrics->epoch_testing_accuracies) {
            return;
        }
            for (size_t i = 0; i < g_global_metrics->num_epochs; i++) {
            g_global_metrics->epoch_testing_losses[i]     = INFINITY;
            g_global_metrics->epoch_testing_accuracies[i] = INFINITY;
        }
    }
    training_metrics_ensure_capacity(g_global_metrics, g_global_metrics->num_epochs);

    size_t last_epoch = g_global_metrics->num_epochs > 0 ? g_global_metrics->num_epochs - 1 : 0;

    if (last_epoch < g_global_metrics->num_epochs) {
        g_global_metrics->epoch_testing_losses[last_epoch]     = test_loss;
        g_global_metrics->epoch_testing_accuracies[last_epoch] = test_accuracy;
        const char* metrics_path = "training.json";
        training_metrics_export_json(g_global_metrics, metrics_path, true);
    }
}

int training_metrics_evaluate_dataset(Module* model, Dataset* dataset,
                                      Tensor* (*loss_fn)(Tensor*, Tensor*), bool is_validation) {
    if (!model || !dataset || !loss_fn || !dataset->X || !dataset->y) {
        return -1;
    }
    bool was_training = module_is_training(model);
    module_set_training(model, false);
    Tensor* outputs = module_forward(model, dataset->X);
    if (!outputs) {
        module_set_training(model, was_training);
        return -1;
    }
    Tensor* loss = loss_fn(outputs, dataset->y);
    if (!loss) {
        tensor_free(outputs);
        module_set_training(model, was_training);
        return -1;
    }
    float loss_value = INFINITY;
    float* loss_data = (float*)tensor_data_ptr(loss);
    if (loss_data) {
        loss_value = loss_data[0];
    }
    float* output_data = (float*)tensor_data_ptr(outputs);
    float* target_data = (float*)tensor_data_ptr(dataset->y);
    int correct        = 0;
    int num_samples    = dataset->num_samples;

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
    if (is_validation) {
        training_metrics_auto_capture_validation(loss_value, accuracy);
    } else {
        training_metrics_auto_capture_test(loss_value, accuracy);
    }
    tensor_free(loss);
    tensor_free(outputs);
    module_set_training(model, was_training);

    return 0;
}
void training_metrics_complete_epoch(void) {
    if (!g_global_metrics)
        return;
    size_t completed = g_current_epoch + 1;
    if (completed > g_global_metrics->num_epochs) {
        training_metrics_ensure_capacity(g_global_metrics, completed);
    }
    if (g_global_metrics->num_epochs < completed) {
        g_global_metrics->num_epochs = completed;
    }
    g_global_metrics->actual_epochs = completed;
    g_current_epoch++;
    const char* metrics_path = "training.json";
    training_metrics_export_json(g_global_metrics, metrics_path, false);
}

void training_metrics_cleanup_global(void) {
    if (g_global_metrics) {
        const char* metrics_path = "training.json";
        training_metrics_export_json(g_global_metrics, metrics_path, false);

        training_metrics_free(g_global_metrics);
        g_global_metrics        = NULL;
        g_current_epoch         = 0;
        g_optimizer_step_count  = 0;
        g_last_epoch_step_count = 0;
        g_epoch_start_time      = 0;
        g_last_loss             = 0.0f;
        g_epoch_in_progress     = false;
        g_current_model         = NULL;
        g_architecture_exported = false;
        g_manual_epoch_control  = false;
    }
}
