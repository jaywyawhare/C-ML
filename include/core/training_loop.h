/**
 * @file training_loop.h
 * @brief Training loop utilities with callbacks and LR schedulers
 *
 * Provides:
 * - Learning rate schedulers
 * - Training callbacks and hooks
 * - One-line training function
 * - Progress bars
 */

#ifndef CML_CORE_TRAINING_LOOP_H
#define CML_CORE_TRAINING_LOOP_H

#include "nn.h"
#include "optim.h"
#include "core/dataset.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Learning rate scheduler types
 */
typedef enum {
    LR_SCHEDULER_NONE,              // No scheduling
    LR_SCHEDULER_STEP,              // StepLR: lr *= gamma every step_size epochs
    LR_SCHEDULER_REDUCE_ON_PLATEAU, // ReduceLROnPlateau: reduce when metric plateaus
    LR_SCHEDULER_EXPONENTIAL,       // ExponentialLR: lr *= gamma every epoch
    LR_SCHEDULER_COSINE,            // CosineAnnealingLR: cosine annealing
} LRSchedulerType;

/**
 * @brief Learning rate scheduler structure
 */
typedef struct LRScheduler {
    LRSchedulerType type;
    Optimizer* optimizer;

    // StepLR parameters
    int step_size;
    float gamma;

    // ReduceLROnPlateau parameters
    float factor;
    int patience;
    float min_lr;
    float threshold;
    int cooldown;
    int best_epoch;
    float best_metric;
    int plateau_count;

    // ExponentialLR parameters
    float exp_gamma;

    // CosineAnnealingLR parameters
    int T_max;
    float eta_min;
    float initial_lr; // Initial learning rate (for cosine annealing)

    // Internal state
    int last_epoch;
    float current_lr;
} LRScheduler;

/**
 * @brief Create StepLR scheduler
 *
 * @param optimizer Optimizer to schedule
 * @param step_size Number of epochs between LR reductions
 * @param gamma Multiplicative factor for LR reduction
 * @return LRScheduler, or NULL on failure
 */
LRScheduler* lr_scheduler_step(Optimizer* optimizer, int step_size, float gamma);

/**
 * @brief Create ReduceLROnPlateau scheduler
 *
 * @param optimizer Optimizer to schedule
 * @param factor Multiplicative factor for LR reduction
 * @param patience Number of epochs to wait before reducing LR
 * @param min_lr Minimum learning rate
 * @return LRScheduler, or NULL on failure
 */
LRScheduler* lr_scheduler_reduce_on_plateau(Optimizer* optimizer, float factor, int patience,
                                            float min_lr);

/**
 * @brief Create ExponentialLR scheduler
 *
 * @param optimizer Optimizer to schedule
 * @param gamma Multiplicative factor for LR reduction
 * @return LRScheduler, or NULL on failure
 */
LRScheduler* lr_scheduler_exponential(Optimizer* optimizer, float gamma);

/**
 * @brief Create CosineAnnealingLR scheduler
 *
 * Learning rate follows cosine curve from initial_lr to eta_min over T_max epochs.
 * Formula: lr = eta_min + (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
 *
 * @param optimizer Optimizer to schedule
 * @param T_max Maximum number of epochs for one cycle
 * @param eta_min Minimum learning rate (default 0)
 * @return LRScheduler, or NULL on failure
 */
LRScheduler* lr_scheduler_cosine(Optimizer* optimizer, int T_max, float eta_min);

/**
 * @brief Step the learning rate scheduler
 *
 * @param scheduler LRScheduler
 * @param metric Current metric value (for ReduceLROnPlateau)
 * @return New learning rate
 */
float lr_scheduler_update(LRScheduler* scheduler, float metric);

/**
 * @brief Get current learning rate from scheduler
 *
 * @param scheduler LRScheduler
 * @return Current learning rate
 */
float lr_scheduler_get_lr(LRScheduler* scheduler);

/**
 * @brief Free learning rate scheduler
 *
 * @param scheduler LRScheduler to free
 */
void lr_scheduler_free(LRScheduler* scheduler);

/**
 * @brief Training callback function types
 */
typedef int (*OnEpochBeginCallback)(int epoch, void* user_data);
typedef int (*OnEpochEndCallback)(int epoch, float train_loss, float train_acc, void* user_data);
typedef int (*OnBatchBeginCallback)(int epoch, int batch, void* user_data);
typedef int (*OnBatchEndCallback)(int epoch, int batch, float loss, void* user_data);
typedef int (*OnTrainingBeginCallback)(void* user_data);
typedef int (*OnTrainingEndCallback)(void* user_data);

/**
 * @brief Training callbacks structure
 */
typedef struct TrainingCallbacks {
    OnEpochBeginCallback on_epoch_begin;
    OnEpochEndCallback on_epoch_end;
    OnBatchBeginCallback on_batch_begin;
    OnBatchEndCallback on_batch_end;
    OnTrainingBeginCallback on_training_begin;
    OnTrainingEndCallback on_training_end;
    void* user_data;
} TrainingCallbacks;

/**
 * @brief Create empty training callbacks
 *
 * @return TrainingCallbacks with all callbacks set to NULL
 */
void training_callbacks_create(TrainingCallbacks* callbacks);

/**
 * @brief Training configuration structure
 */
typedef struct TrainingConfig {
    int epochs;
    bool verbose;
    bool use_progress_bar;
    LRScheduler* scheduler;
    TrainingCallbacks callbacks;
    float grad_clip_norm;
    bool early_stopping;
    int early_stopping_patience;
    float early_stopping_min_delta;
} TrainingConfig;

/**
 * @brief Create default training configuration
 *
 * @return TrainingConfig with sensible defaults
 */
void training_config_default(TrainingConfig* config);

/**
 * @brief Train a model
 *
 * Trains a model with automatic metrics tracking, logging, and cleanup.
 *
 * @param model Model to train
 * @param train_loader Training DataLoader
 * @param optimizer Optimizer
 * @param loss_fn Loss function (e.g., tensor_mse_loss)
 * @param config Training configuration (NULL for defaults)
 * @return 0 on success, negative value on failure
 */
int cml_train(Module* model, DataLoader* train_loader, Optimizer* optimizer,
              Tensor* (*loss_fn)(Tensor*, Tensor*), TrainingConfig* config);

/**
 * @brief Train a model with validation
 *
 * @param model Model to train
 * @param train_loader Training DataLoader
 * @param val_loader Validation DataLoader
 * @param optimizer Optimizer
 * @param loss_fn Loss function
 * @param config Training configuration (NULL for defaults)
 * @return 0 on success, negative value on failure
 */
int cml_train_with_validation(Module* model, DataLoader* train_loader, DataLoader* val_loader,
                              Optimizer* optimizer, Tensor* (*loss_fn)(Tensor*, Tensor*),
                              TrainingConfig* config);

/**
 * @brief Progress callback function type
 *
 * @param percent Progress percentage (0-100)
 * @param user_data User data
 */
typedef void (*ProgressCallback)(float percent, void* user_data);

/**
 * @brief Set progress callback
 *
 * @param callback Progress callback function (NULL to disable)
 * @param user_data User data to pass to callback
 */
void cml_set_progress_callback(ProgressCallback callback, void* user_data);

/**
 * @brief Print text progress bar
 *
 * @param percent Progress percentage (0-100)
 * @param message Optional message to display
 */
void cml_print_progress_bar(float percent, const char* message);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_TRAINING_LOOP_H
