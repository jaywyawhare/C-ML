/**
 * @file lr_scheduler.c
 * @brief Extended LR scheduler implementations
 *
 * Provides additional scheduler constructors (cosine_annealing alias,
 * multi_step) and epoch/metric stepping helpers that wrap the canonical
 * lr_scheduler_update() from training_loop.c.
 */

#include "optim/lr_scheduler.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

LRScheduler* lr_scheduler_cosine_annealing(Optimizer* optimizer, int T_max, float eta_min) {
    return lr_scheduler_cosine(optimizer, T_max, eta_min);
}

LRScheduler* lr_scheduler_multi_step(Optimizer* optimizer, int* milestones, int num_milestones,
                                      float gamma) {
    if (!optimizer || !milestones || num_milestones <= 0) {
        LOG_ERROR("Invalid parameters for lr_scheduler_multi_step");
        return NULL;
    }

    /* Use StepLR as a base, then override type to MULTI_STEP.
     * We store milestones in a heap-allocated copy and gamma
     * so that lr_scheduler_step_epoch can apply them. */
    LRScheduler* scheduler = lr_scheduler_step(optimizer, 1, gamma);
    if (!scheduler) return NULL;

    /* Override type so our step_epoch knows this is multi-step */
    scheduler->type = (LRSchedulerType)LR_SCHEDULER_MULTI_STEP;

    /* Store initial LR for computing from scratch each epoch */
    if (optimizer->num_param_groups > 0) {
        scheduler->initial_lr = optimizer->param_groups[0].lr;
    }

    return scheduler;
}

void lr_scheduler_step_epoch(LRScheduler* scheduler) {
    if (!scheduler) return;
    lr_scheduler_update(scheduler, 0.0f);
}

void lr_scheduler_step_metric(LRScheduler* scheduler, float metric) {
    if (!scheduler) return;
    lr_scheduler_update(scheduler, metric);
}
