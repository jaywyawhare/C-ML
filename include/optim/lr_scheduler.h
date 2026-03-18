#ifndef CML_OPTIM_LR_SCHEDULER_H
#define CML_OPTIM_LR_SCHEDULER_H

#include "core/training_loop.h"

#ifdef __cplusplus
extern "C" {
#endif

// LR_SCHEDULER_MULTI_STEP and LR_SCHEDULER_ONE_CYCLE are now in the
// LRSchedulerType enum defined in core/training_loop.h

LRScheduler* lr_scheduler_cosine_annealing(Optimizer* optimizer, int T_max, float eta_min);

LRScheduler* lr_scheduler_multi_step(Optimizer* optimizer, int* milestones, int num_milestones,
                                      float gamma);

void lr_scheduler_step_epoch(LRScheduler* scheduler);

void lr_scheduler_step_metric(LRScheduler* scheduler, float metric);

LRScheduler* lr_scheduler_one_cycle(Optimizer* optimizer, float max_lr, int total_steps,
                                     float pct_start, float div_factor, float final_div_factor);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPTIM_LR_SCHEDULER_H */
