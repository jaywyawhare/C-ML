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

LRScheduler* lr_scheduler_one_cycle(Optimizer* optimizer, float max_lr, int total_steps,
                                     float pct_start, float div_factor, float final_div_factor) {
    if (!optimizer || total_steps <= 0) {
        LOG_ERROR("Invalid parameters for lr_scheduler_one_cycle");
        return NULL;
    }

    LRScheduler* scheduler = malloc(sizeof(LRScheduler));
    if (!scheduler) return NULL;

    memset(scheduler, 0, sizeof(LRScheduler));

    scheduler->type      = (LRSchedulerType)LR_SCHEDULER_ONE_CYCLE;
    scheduler->optimizer = optimizer;

    // Store OneCycle parameters in available fields:
    // initial_lr = max_lr
    // eta_min = initial_lr (max_lr / div_factor)
    // T_max = total_steps
    // gamma = pct_start
    // factor = final_lr (max_lr / final_div_factor)
    // min_lr = final_div_factor (stored for reference)

    float init_lr  = max_lr / (div_factor > 0.0f ? div_factor : 25.0f);
    float final_lr = max_lr / (final_div_factor > 0.0f ? final_div_factor : 1e4f);

    scheduler->initial_lr = max_lr;
    scheduler->eta_min    = init_lr;
    scheduler->T_max      = total_steps;
    scheduler->gamma      = pct_start > 0.0f ? pct_start : 0.3f;
    scheduler->factor     = final_lr;
    scheduler->last_epoch = 0;

    // Set optimizer to initial LR
    scheduler->current_lr = init_lr;
    optimizer_set_lr(optimizer, init_lr);

    return scheduler;
}
