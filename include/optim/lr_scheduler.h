/**
 * @file lr_scheduler.h
 * @brief Learning rate scheduler implementations
 */

#ifndef CML_OPTIM_LR_SCHEDULER_H
#define CML_OPTIM_LR_SCHEDULER_H

#include "optim.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LR_SCHEDULER_STEP,
    LR_SCHEDULER_EXPONENTIAL,
    LR_SCHEDULER_COSINE_ANNEALING,
    LR_SCHEDULER_REDUCE_ON_PLATEAU,
    LR_SCHEDULER_MULTI_STEP,
} LRSchedulerType;

typedef struct LRScheduler {
    LRSchedulerType type;
    Optimizer* optimizer;
    float initial_lr;
    int last_epoch;

    // StepLR params
    int step_size;
    float gamma;

    // CosineAnnealing params
    int T_max;
    float eta_min;

    // ReduceOnPlateau params
    float factor;
    int patience;
    float threshold;
    int num_bad_epochs;
    float best_metric;
    bool mode_min; // true = minimize, false = maximize

    // MultiStepLR params
    int* milestones;
    int num_milestones;
} LRScheduler;

LRScheduler* lr_scheduler_step(Optimizer* optimizer, int step_size, float gamma);
LRScheduler* lr_scheduler_exponential(Optimizer* optimizer, float gamma);
LRScheduler* lr_scheduler_cosine_annealing(Optimizer* optimizer, int T_max, float eta_min);
LRScheduler* lr_scheduler_reduce_on_plateau(Optimizer* optimizer, float factor, int patience,
                                             float threshold, bool mode_min);
LRScheduler* lr_scheduler_multi_step(Optimizer* optimizer, int* milestones, int num_milestones,
                                      float gamma);

void lr_scheduler_step_epoch(LRScheduler* scheduler);
void lr_scheduler_step_metric(LRScheduler* scheduler, float metric);
float lr_scheduler_get_lr(LRScheduler* scheduler);
void lr_scheduler_free(LRScheduler* scheduler);

#ifdef __cplusplus
}
#endif

#endif // CML_OPTIM_LR_SCHEDULER_H
