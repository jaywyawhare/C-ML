/**
 * @file lr_scheduler.h
 * @brief Extended learning rate scheduler API
 *
 * This header provides additional LR scheduler constructors and stepping
 * functions beyond those in core/training_loop.h.  It re-exports the
 * canonical LRScheduler type from training_loop.h so that users who
 * include either header get the same struct definition.
 *
 * Extra schedulers provided here:
 *   - lr_scheduler_cosine_annealing()  (alias for lr_scheduler_cosine)
 *   - lr_scheduler_multi_step()
 *
 * Extra stepping helpers:
 *   - lr_scheduler_step_epoch()   (epoch-based, no metric)
 *   - lr_scheduler_step_metric()  (metric-based, for ReduceOnPlateau)
 */

#ifndef CML_OPTIM_LR_SCHEDULER_H
#define CML_OPTIM_LR_SCHEDULER_H

#include "core/training_loop.h"

#ifdef __cplusplus
extern "C" {
#endif

// LR_SCHEDULER_MULTI_STEP and LR_SCHEDULER_ONE_CYCLE are now in the
// LRSchedulerType enum defined in core/training_loop.h

/**
 * @brief Create a CosineAnnealingLR scheduler (alias)
 *
 * Equivalent to lr_scheduler_cosine() from training_loop.h.
 */
LRScheduler* lr_scheduler_cosine_annealing(Optimizer* optimizer, int T_max, float eta_min);

/**
 * @brief Create a MultiStepLR scheduler
 *
 * Decays the learning rate by gamma at each milestone epoch.
 */
LRScheduler* lr_scheduler_multi_step(Optimizer* optimizer, int* milestones, int num_milestones,
                                      float gamma);

/**
 * @brief Advance scheduler by one epoch (no metric)
 *
 * Calls lr_scheduler_update(scheduler, 0.0f) internally.
 */
void lr_scheduler_step_epoch(LRScheduler* scheduler);

/**
 * @brief Update scheduler with a metric value
 *
 * Calls lr_scheduler_update(scheduler, metric) internally.
 */
void lr_scheduler_step_metric(LRScheduler* scheduler, float metric);

/**
 * @brief Create a OneCycleLR scheduler
 *
 * Implements the 1cycle policy: linearly ramps up LR from initial_lr/div_factor to max_lr,
 * then cosine anneals down to max_lr/final_div_factor.
 *
 * @param optimizer Optimizer to schedule
 * @param max_lr Maximum learning rate
 * @param total_steps Total number of training steps
 * @param pct_start Percentage of training spent increasing LR (default: 0.3)
 * @param div_factor Initial lr = max_lr / div_factor (default: 25.0)
 * @param final_div_factor Final lr = max_lr / final_div_factor (default: 1e4)
 * @return LRScheduler, or NULL on failure
 */
LRScheduler* lr_scheduler_one_cycle(Optimizer* optimizer, float max_lr, int total_steps,
                                     float pct_start, float div_factor, float final_div_factor);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPTIM_LR_SCHEDULER_H */
