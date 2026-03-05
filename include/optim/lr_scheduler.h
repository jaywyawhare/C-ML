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

#define LR_SCHEDULER_MULTI_STEP 10

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

#ifdef __cplusplus
}
#endif

#endif /* CML_OPTIM_LR_SCHEDULER_H */
