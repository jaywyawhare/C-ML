#ifndef CML_TRAINING_METRICS_H
#define CML_TRAINING_METRICS_H

#include <stddef.h>
#include <stdbool.h>
#include <time.h>
#include "core/cml_flags.h"

typedef struct Module Module;
typedef struct Optimizer Optimizer;
typedef struct Tensor Tensor;
typedef struct Dataset Dataset;

/** Per-epoch shape of a parameter population (all weights, or all gradients).
 * A single gradient-norm scalar hides the failure modes that matter -- dead
 * units, a few exploding channels, a layer that stopped moving -- because they
 * all average out. Percentiles come from a fixed-bin histogram over [min,max]
 * rather than a sort, which would mean ordering every parameter in the model
 * once per epoch for a chart that only needs quartiles. */
typedef struct {
    float min, p25, p50, p75, max;
    float mean, std;
    float frac_zero; /* share of exactly-zero entries: dead ReLUs, pruned weights */
} DistributionSummary;

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
    DistributionSummary* epoch_grad_dist;   // Per-epoch gradient distribution
    DistributionSummary* epoch_weight_dist; // Per-epoch weight distribution
    bool has_distributions;                 // Set once distributions are recorded

    float loss_reduction_rate; // Percentage reduction in loss
    float loss_stability;      // Standard deviation of recent losses

    bool early_stopped;     // Whether training was stopped early
    size_t expected_epochs; // Original expected number of epochs (before early stopping)
    size_t actual_epochs;   // Actual number of epochs completed (after early stopping)
} TrainingMetrics;

/**
 * Materialised float data of `t` plus its element count; `numel` is recomputed
 * from the shape when the tensor has not cached it. NULL when `t` is not
 * float32 or has no data.
 * @param t (Tensor*)
 * @param num_elements (size_t*)
 * @return (float*)
 */
float* cml_tensor_float_buffer(Tensor* t, size_t* num_elements);

TrainingMetrics* training_metrics_create(size_t num_epochs);
void training_metrics_start_epoch(TrainingMetrics* metrics);
void training_metrics_record_epoch(TrainingMetrics* metrics, size_t epoch, float loss,
                                   float accuracy);

/**
 * Record train plus optional test/val loss and accuracy for `epoch`. A split is recorded only
 * when its loss and accuracy are both finite and non-negative: pass NAN for one that is absent.
 * @param metrics (TrainingMetrics*)
 * @param epoch (size_t)
 * @param train_loss (float)
 * @param train_accuracy (float)
 * @param test_loss (float)
 * @param test_accuracy (float)
 * @param val_loss (float)
 * @param val_accuracy (float)
 * @return (void)
 */
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

/**
 * Summarise the weight and gradient populations for `epoch`. Call once per
 * epoch, after backward; skipped cheaply when the arrays were never allocated.
 * @param metrics (TrainingMetrics*)
 * @param epoch (size_t)
 * @param parameters (void**)
 * @param num_parameters (int)
 * @return (void)
 */
void training_metrics_record_distributions(TrainingMetrics* metrics, size_t epoch,
                                           void** parameters, int num_parameters);

/**
 * Write the metrics as JSON to `path`; `incremental` marks the run as still training. Returns 0
 * on success (and under NO_EXPORT), negative on error.
 * @param metrics (const TrainingMetrics*)
 * @param path (const char*)
 * @param incremental (bool)
 * @return (int)
 */
int training_metrics_export_json(const TrainingMetrics* metrics, const char* path,
                                 bool incremental);
int training_metrics_export_epoch_update(const TrainingMetrics* metrics, size_t epoch,
                                         const char* path);
int training_metrics_export_architecture(Module* module, const char* path);
void training_metrics_free(TrainingMetrics* metrics);
void training_metrics_log(TrainingMetrics* metrics, size_t epoch);

/**
 * Training step helper: forward pass, loss, backward, optimizer step,
 * and automatic metrics recording.
 * @param model (Module*)
 * @param X (Tensor*)
 * @param y (Tensor*)
 * @param loss_fn (Tensor* (*)(Tensor*, Tensor*))
 * @param optimizer (Optimizer*)
 * @param epoch (size_t)
 * @param loss_out (float*)
 * @param accuracy_out (float*)
 * @return (int)
 */
int training_metrics_step(Module* model, Tensor* X, Tensor* y, Tensor* (*loss_fn)(Tensor*, Tensor*),
                          Optimizer* optimizer, size_t epoch, float* loss_out, float* accuracy_out);

TrainingMetrics* training_metrics_get_global(void);
void training_metrics_auto_capture_loss(Tensor* loss_tensor);
void training_metrics_auto_capture_optimizer(Optimizer* optimizer);

/**
 * Classification losses call this with (prediction, target) so the metrics layer
 * can auto-compute per-epoch accuracy during loss capture. The pointers are kept
 * until the next loss capture reads them, or until either tensor is freed.
 * @param prediction (Tensor*)
 * @param target (Tensor*)
 * @return (void)
 */
void training_metrics_note_prediction(Tensor* prediction, Tensor* target);
void training_metrics_init_global(void);
/**
 * Drop any capture state pointing at @p obj (a Module or Tensor) before it is freed.
 * @param obj (const void*)
 * @return (void)
 */
void training_metrics_forget(const void* obj);
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

/**
 * Close the current epoch: records it as completed, advances the epoch index and
 * exports. Call at the end of every epoch when using manual epoch control
 * (which training_metrics_set_expected_epochs turns on) -- otherwise the index
 * never moves and every epoch overwrites the first slot. Under automatic epoch
 * detection, call it once at the end of training, since tracking there
 * increments on the NEXT zero_grad call and the final epoch would be missed.
 * @return (void)
 */
void training_metrics_complete_epoch(void);

#endif // CML_TRAINING_METRICS_H
