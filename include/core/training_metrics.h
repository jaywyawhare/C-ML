/**
 * @file training_metrics.h
 * @brief Training metrics export for visualization
 */

#ifndef CML_TRAINING_METRICS_H
#define CML_TRAINING_METRICS_H

#include <stddef.h>
#include <stdbool.h>
#include <time.h>

// Forward declarations
typedef struct Module Module;
typedef struct Optimizer Optimizer;
typedef struct Tensor Tensor;
typedef struct Dataset Dataset;

/**
 * @brief Training metrics structure
 */
typedef struct {
    // Training metrics
    float* epoch_training_losses;     // Array of training loss values per epoch
    float* epoch_training_accuracies; // Array of training accuracy values per epoch

    // Testing metrics (optional)
    float* epoch_testing_losses;     // Array of testing loss values per epoch
    float* epoch_testing_accuracies; // Array of testing accuracy values per epoch

    // Validation metrics (optional)
    float* epoch_validation_losses;     // Array of validation loss values per epoch
    float* epoch_validation_accuracies; // Array of validation accuracy values per epoch

    size_t num_epochs;    // Number of epochs
    float best_loss;      // Best loss achieved
    float best_accuracy;  // Best accuracy achieved
    int total_params;     // Total number of parameters
    int trainable_params; // Number of trainable parameters
    char* model_summary;  // Model architecture summary

    // Time metrics
    float* epoch_times;      // Array of time per epoch (in seconds)
    float total_time;        // Total training time (in seconds)
    double epoch_start_time; // Start time of current epoch (in seconds, monotonic)

    // Training configuration
    float learning_rate;         // Current learning rate
    float* epoch_learning_rates; // Array of learning rate values per epoch (for LR schedule
                                 // visualization)
    char* lr_schedule;           // Learning rate scheduler name (e.g., "StepLR", "CosineAnnealing")
    char* lr_schedule_params;    // LR scheduler parameters (e.g., "step_size=30,gamma=0.5")
    float gradient_norm;         // Current gradient norm (for gradient health)

    // Computed metrics (updated on export)
    float loss_reduction_rate; // Percentage reduction in loss
    float loss_stability;      // Standard deviation of recent losses (σ)

    // Early stopping
    bool early_stopped;     // Whether training was stopped early
    size_t expected_epochs; // Original expected number of epochs (before early stopping)
    size_t actual_epochs;   // Actual number of epochs completed (after early stopping)
} TrainingMetrics;

/**
 * @brief Initialize training metrics structure
 */
TrainingMetrics* training_metrics_create(size_t num_epochs);

/**
 * @brief Start epoch timing (call before epoch starts)
 */
void training_metrics_start_epoch(TrainingMetrics* metrics);

/**
 * @brief Record metrics for an epoch
 * Automatically calculates and records epoch time if start_epoch was called
 * @param metrics Training metrics structure
 * @param epoch Epoch number (0-indexed)
 * @param loss Training loss value
 * @param accuracy Training accuracy value
 */
void training_metrics_record_epoch(TrainingMetrics* metrics, size_t epoch, float loss,
                                   float accuracy);

/**
 * @brief Record training metrics for an epoch
 * @param metrics Training metrics structure
 * @param epoch Epoch number (0-indexed)
 * @param train_loss Training loss value
 * @param train_accuracy Training accuracy value
 * @param test_loss Testing loss value (optional, pass 0.0f if not available)
 * @param test_accuracy Testing accuracy value (optional, pass 0.0f if not available)
 * @param val_loss Validation loss value (optional, pass 0.0f if not available)
 * @param val_accuracy Validation accuracy value (optional, pass 0.0f if not available)
 */
void training_metrics_record_epoch_full(TrainingMetrics* metrics, size_t epoch, float train_loss,
                                        float train_accuracy, float test_loss, float test_accuracy,
                                        float val_loss, float val_accuracy);

/**
 * @brief Set model summary
 */
void training_metrics_set_summary(TrainingMetrics* metrics, const char* summary);

/**
 * @brief Set parameter counts
 */
void training_metrics_set_params(TrainingMetrics* metrics, int total, int trainable);

/**
 * @brief Record epoch time
 */
void training_metrics_record_epoch_time(TrainingMetrics* metrics, size_t epoch, float time_seconds);

/**
 * @brief Set learning rate and scheduler
 */
void training_metrics_set_learning_rate(TrainingMetrics* metrics, float lr, const char* scheduler);

/**
 * @brief Set learning rate scheduler parameters (e.g., "step_size=30,gamma=0.5")
 */
void training_metrics_set_lr_schedule_params(TrainingMetrics* metrics, const char* params);

/**
 * @brief Set gradient norm (for gradient health monitoring)
 */
void training_metrics_set_gradient_norm(TrainingMetrics* metrics, float grad_norm);

/**
 * @brief Calculate and set gradient norm from parameters
 * @param metrics Training metrics structure
 * @param parameters Array of parameters
 * @param num_parameters Number of parameters
 * @return Calculated gradient norm, or 0.0f if calculation failed
 */
float training_metrics_calculate_gradient_norm(TrainingMetrics* metrics, void** parameters,
                                               int num_parameters);

/**
 * @brief Export training metrics to JSON file
 * @param incremental If true, only export completed epochs (for real-time updates)
 */
int training_metrics_export_json(const TrainingMetrics* metrics, const char* path,
                                 bool incremental);

/**
 * @brief Export single epoch update (for real-time streaming like wandb)
 */
int training_metrics_export_epoch_update(const TrainingMetrics* metrics, size_t epoch,
                                         const char* path);

/**
 * @brief Export model architecture from a Module
 *
 * Extracts architecture information from a C-ML Module and exports it to JSON.
 * This can be used to visualize the model structure.
 *
 * @param module The module to extract architecture from
 * @param path Output file path for architecture JSON
 * @return 0 on success, negative value on failure
 */
int training_metrics_export_architecture(Module* module, const char* path);

/**
 * @brief Free training metrics structure
 */
void training_metrics_free(TrainingMetrics* metrics);

/**
 * @brief Log all training metrics to console
 * @param metrics Training metrics structure
 * @param epoch Current epoch number
 */
void training_metrics_log(TrainingMetrics* metrics, size_t epoch);

/**
 * @brief Training step helper - automatically records metrics
 * This function handles forward pass, loss calculation, backward pass, optimizer step,
 * and automatic metrics recording if metrics are registered with the optimizer.
 *
 * @param model Model to train
 * @param X Input tensor
 * @param y Target tensor
 * @param loss_fn Loss function to use (e.g., tensor_mse_loss)
 * @param optimizer Optimizer instance (with optional metrics registered)
 * @param epoch Current epoch number
 * @param loss_out Output parameter for loss value
 * @param accuracy_out Output parameter for accuracy value
 * @return 0 on success, negative value on failure
 */
int training_metrics_step(Module* model, Tensor* X, Tensor* y, Tensor* (*loss_fn)(Tensor*, Tensor*),
                          Optimizer* optimizer, size_t epoch, float* loss_out, float* accuracy_out);

/**
 * @brief Get global metrics instance (for automatic capture)
 * @return Global TrainingMetrics instance, or NULL if not initialized
 */
TrainingMetrics* training_metrics_get_global(void);

/**
 * @brief Auto-capture loss from tensor_backward (called automatically)
 * @param loss_tensor Loss tensor to capture
 */
void training_metrics_auto_capture_loss(Tensor* loss_tensor);

/**
 * @brief Auto-capture metrics from optimizer_step (called automatically)
 * @param optimizer Optimizer instance
 */
void training_metrics_auto_capture_optimizer(Optimizer* optimizer);

/**
 * @brief Initialize global metrics (called from cml_init)
 */
void training_metrics_init_global(void);

/**
 * @brief Cleanup global metrics (called from cml_cleanup)
 */
void training_metrics_cleanup_global(void);

/**
 * @brief Mark that zero_grad was called (for epoch detection)
 */
void training_metrics_mark_zero_grad(void);

/**
 * @brief Set expected number of epochs (for UI display)
 * @param num_epochs Expected number of epochs
 */
void training_metrics_set_expected_epochs(size_t num_epochs);

/**
 * @brief Register model for automatic architecture export
 * @param model Model to register
 */
void training_metrics_register_model(Module* model);

/**
 * @brief Auto-export model architecture (called automatically when training starts)
 * @param model Model to export architecture for
 */
void training_metrics_auto_export_architecture(Module* model);

/**
 * @brief Auto-capture training accuracy (called from training loop)
 * @param train_accuracy Training accuracy value
 */
void training_metrics_auto_capture_train_accuracy(float train_accuracy);

/**
 * @brief Auto-capture validation metrics (called automatically during evaluation)
 * @param val_loss Validation loss value
 * @param val_accuracy Validation accuracy value
 */
void training_metrics_auto_capture_validation(float val_loss, float val_accuracy);

/**
 * @brief Auto-capture test metrics (called automatically during evaluation)
 * @param test_loss Test loss value
 * @param test_accuracy Test accuracy value
 */
void training_metrics_auto_capture_test(float test_loss, float test_accuracy);

/**
 * @brief Evaluate model on dataset and automatically record metrics
 * @param model Model to evaluate
 * @param dataset Dataset to evaluate on
 * @param loss_fn Loss function to use (e.g., tensor_mse_loss)
 * @param is_validation If true, records as validation metrics; if false, records as test metrics
 * @return 0 on success, negative value on failure
 */
int training_metrics_evaluate_dataset(Module* model, Dataset* dataset,
                                      Tensor* (*loss_fn)(Tensor*, Tensor*), bool is_validation);

/**
 * @brief Mark that training was stopped early
 * @param actual_epochs Actual number of epochs completed (0-indexed, so pass epoch number)
 * This should be called when early stopping is triggered in the training loop.
 * It updates the metrics to reflect the actual number of epochs completed.
 */
void training_metrics_mark_early_stop(size_t actual_epochs);

/**
 * @brief Manually complete the current epoch
 * This should be called at the end of training to ensure the final epoch's data is captured.
 * Needed because epoch tracking increments on the NEXT zero_grad call.
 */
void training_metrics_complete_epoch(void);

#endif // CML_TRAINING_METRICS_H
