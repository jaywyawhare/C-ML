#ifndef CML_CORE_TRAINING_LOOP_H
#define CML_CORE_TRAINING_LOOP_H

#include "nn.h"
#include "optim.h"
#include "core/dataset.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LR_SCHEDULER_NONE,              // No scheduling
    LR_SCHEDULER_STEP,              // StepLR: lr *= gamma every step_size epochs
    LR_SCHEDULER_REDUCE_ON_PLATEAU, // ReduceLROnPlateau: reduce when metric plateaus
    LR_SCHEDULER_EXPONENTIAL,       // ExponentialLR: lr *= gamma every epoch
    LR_SCHEDULER_COSINE,            // CosineAnnealingLR: cosine annealing
    LR_SCHEDULER_ONE_CYCLE,         // OneCycleLR: ramp up then ramp down
    LR_SCHEDULER_MULTI_STEP,        // MultiStepLR: decay at specified milestones
    LR_SCHEDULER_POLYNOMIAL,        // PolynomialLR: polynomial decay
    LR_SCHEDULER_WARMUP,            // Warmup wrapper: linear warmup then delegate
} LRSchedulerType;

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

    // OneCycleLR parameters
    float max_lr;
    int total_steps;
    float pct_start;
    float div_factor;
    float final_div_factor;

    // MultiStepLR parameters
    int* milestones;
    int num_milestones;

    // PolynomialLR parameters
    int total_iters;
    float power;

    // Warmup parameters
    struct LRScheduler* inner_scheduler;
    int warmup_steps;
    float warmup_start_factor;

    // Internal state
    int last_epoch;
    float current_lr;
} LRScheduler;

LRScheduler* lr_scheduler_step(Optimizer* optimizer, int step_size, float gamma);
LRScheduler* lr_scheduler_reduce_on_plateau(Optimizer* optimizer, float factor, int patience,
                                            float min_lr);
LRScheduler* lr_scheduler_exponential(Optimizer* optimizer, float gamma);

/* lr = eta_min + (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2 */
LRScheduler* lr_scheduler_cosine(Optimizer* optimizer, int T_max, float eta_min);

LRScheduler* lr_scheduler_one_cycle(Optimizer* optimizer, float max_lr, int total_steps,
                                     float pct_start, float div_factor, float final_div_factor);
LRScheduler* lr_scheduler_multi_step(Optimizer* optimizer, int* milestones, int num_milestones,
                                      float gamma);

/* lr = (initial_lr - min_lr) * (1 - epoch/total_iters)^power + min_lr */
LRScheduler* lr_scheduler_polynomial(Optimizer* optimizer, int total_iters, float power,
                                      float min_lr);

/* Linear warmup for first N steps, then delegates to inner scheduler */
LRScheduler* lr_scheduler_warmup(LRScheduler* inner, int warmup_steps, float warmup_start_factor);

float lr_scheduler_update(LRScheduler* scheduler, float metric);
float lr_scheduler_get_lr(LRScheduler* scheduler);
void lr_scheduler_free(LRScheduler* scheduler);

typedef int (*OnEpochBeginCallback)(int epoch, void* user_data);
typedef int (*OnEpochEndCallback)(int epoch, float train_loss, float train_acc, void* user_data);
typedef int (*OnBatchBeginCallback)(int epoch, int batch, void* user_data);
typedef int (*OnBatchEndCallback)(int epoch, int batch, float loss, void* user_data);
typedef int (*OnTrainingBeginCallback)(void* user_data);
typedef int (*OnTrainingEndCallback)(void* user_data);

typedef struct TrainingCallbacks {
    OnEpochBeginCallback on_epoch_begin;
    OnEpochEndCallback on_epoch_end;
    OnBatchBeginCallback on_batch_begin;
    OnBatchEndCallback on_batch_end;
    OnTrainingBeginCallback on_training_begin;
    OnTrainingEndCallback on_training_end;
    void* user_data;
} TrainingCallbacks;

void training_callbacks_create(TrainingCallbacks* callbacks);

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
    bool use_checkpointing;           // Enable gradient checkpointing
    int checkpoint_every_n_layers;    // Checkpoint every N layers (0 = auto)
} TrainingConfig;

void training_config_default(TrainingConfig* config);

int cml_train(Module* model, DataLoader* train_loader, Optimizer* optimizer,
              Tensor* (*loss_fn)(Tensor*, Tensor*), TrainingConfig* config);
int cml_train_with_validation(Module* model, DataLoader* train_loader, DataLoader* val_loader,
                              Optimizer* optimizer, Tensor* (*loss_fn)(Tensor*, Tensor*),
                              TrainingConfig* config);

typedef void (*ProgressCallback)(float percent, void* user_data);
void cml_set_progress_callback(ProgressCallback callback, void* user_data);
void cml_print_progress_bar(float percent, const char* message);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_TRAINING_LOOP_H
