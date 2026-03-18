#ifndef CML_TRAINING_METRICS_H
#define CML_TRAINING_METRICS_H

#include <stddef.h>
#include <stdbool.h>
#include <time.h>

typedef struct Module Module;
typedef struct Optimizer Optimizer;
typedef struct Tensor Tensor;
typedef struct Dataset Dataset;

typedef struct {
    float* epoch_training_losses;     // Array of training loss values per epoch
    float* epoch_training_accuracies; // Array of training accuracy values per epoch

    float* epoch_testing_losses;     // Array of testing loss values per epoch
    float* epoch_testing_accuracies; // Array of testing accuracy values per epoch

    float* epoch_validation_losses;     // Array of validation loss values per epoch
    float* epoch_validation_accuracies; // Array of validation accuracy values per epoch

    size_t num_epochs;    // Number of epochs
    float best_loss;      // Best loss achieved
    float best_accuracy;  // Best accuracy achieved
    int total_params;     // Total number of parameters
    int trainable_params; // Number of trainable parameters
    char* model_summary;  // Model architecture summary

    float* epoch_times;      // Array of time per epoch (in seconds)
    float total_time;        // Total training time (in seconds)
    double epoch_start_time; // Start time of current epoch (in seconds, monotonic)

    float learning_rate;         // Current learning rate
    float* epoch_learning_rates; // Array of learning rate values per epoch (for LR schedule
                                 // visualization)
    char* lr_schedule;           // Learning rate scheduler name (e.g., "StepLR", "CosineAnnealing")
    char* lr_schedule_params;    // LR scheduler parameters (e.g., "step_size=30,gamma=0.5")
    float gradient_norm;         // Current gradient norm (for gradient health)

    float loss_reduction_rate; // Percentage reduction in loss
    float loss_stability;      // Standard deviation of recent losses

    bool early_stopped;     // Whether training was stopped early
    size_t expected_epochs; // Original expected number of epochs (before early stopping)
    size_t actual_epochs;   // Actual number of epochs completed (after early stopping)
} TrainingMetrics;

TrainingMetrics* training_metrics_create(size_t num_epochs);
void training_metrics_start_epoch(TrainingMetrics* metrics);
void training_metrics_record_epoch(TrainingMetrics* metrics, size_t epoch, float loss,
                                   float accuracy);

/* Pass 0.0f for optional test/val metrics if not available */
void training_metrics_record_epoch_full(TrainingMetrics* metrics, size_t epoch, float train_loss,
                                        float train_accuracy, float test_loss, float test_accuracy,
                                        float val_loss, float val_accuracy);

void training_metrics_set_summary(TrainingMetrics* metrics, const char* summary);
void training_metrics_set_params(TrainingMetrics* metrics, int total, int trainable);
void training_metrics_record_epoch_time(TrainingMetrics* metrics, size_t epoch, float time_seconds);
void training_metrics_set_learning_rate(TrainingMetrics* metrics, float lr, const char* scheduler);
void training_metrics_set_lr_schedule_params(TrainingMetrics* metrics, const char* params);
void training_metrics_set_gradient_norm(TrainingMetrics* metrics, float grad_norm);
float training_metrics_calculate_gradient_norm(TrainingMetrics* metrics, void** parameters,
                                               int num_parameters);

/* incremental=true exports only completed epochs (for real-time updates) */
int training_metrics_export_json(const TrainingMetrics* metrics, const char* path,
                                 bool incremental);
int training_metrics_export_epoch_update(const TrainingMetrics* metrics, size_t epoch,
                                         const char* path);
int training_metrics_export_architecture(Module* module, const char* path);
void training_metrics_free(TrainingMetrics* metrics);
void training_metrics_log(TrainingMetrics* metrics, size_t epoch);

/*
 * Training step helper: forward pass, loss, backward, optimizer step,
 * and automatic metrics recording.
 */
int training_metrics_step(Module* model, Tensor* X, Tensor* y, Tensor* (*loss_fn)(Tensor*, Tensor*),
                          Optimizer* optimizer, size_t epoch, float* loss_out, float* accuracy_out);

TrainingMetrics* training_metrics_get_global(void);
void training_metrics_auto_capture_loss(Tensor* loss_tensor);
void training_metrics_auto_capture_optimizer(Optimizer* optimizer);
void training_metrics_init_global(void);
void training_metrics_cleanup_global(void);
void training_metrics_mark_zero_grad(void);
void training_metrics_set_expected_epochs(size_t num_epochs);
void training_metrics_register_model(Module* model);
void training_metrics_auto_export_architecture(Module* model);
void training_metrics_auto_capture_train_accuracy(float train_accuracy);
void training_metrics_auto_capture_validation(float val_loss, float val_accuracy);
void training_metrics_auto_capture_test(float test_loss, float test_accuracy);
int training_metrics_evaluate_dataset(Module* model, Dataset* dataset,
                                      Tensor* (*loss_fn)(Tensor*, Tensor*), bool is_validation);
void training_metrics_mark_early_stop(size_t actual_epochs);

/*
 * Call at end of training to ensure final epoch data is captured.
 * Needed because epoch tracking increments on the NEXT zero_grad call.
 */
void training_metrics_complete_epoch(void);

#endif // CML_TRAINING_METRICS_H
