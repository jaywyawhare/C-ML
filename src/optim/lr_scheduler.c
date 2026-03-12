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

    LRScheduler* scheduler = malloc(sizeof(LRScheduler));
    if (!scheduler) return NULL;

    memset(scheduler, 0, sizeof(LRScheduler));

    scheduler->type           = LR_SCHEDULER_MULTI_STEP;
    scheduler->optimizer      = optimizer;
    scheduler->gamma          = gamma;
    scheduler->num_milestones = num_milestones;
    scheduler->last_epoch     = 0;

    // Copy milestones array
    scheduler->milestones = malloc(sizeof(int) * (size_t)num_milestones);
    if (!scheduler->milestones) {
        free(scheduler);
        return NULL;
    }
    memcpy(scheduler->milestones, milestones, sizeof(int) * (size_t)num_milestones);

    if (optimizer->num_param_groups > 0) {
        scheduler->initial_lr = optimizer->param_groups[0].lr;
        scheduler->current_lr = scheduler->initial_lr;
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

    scheduler->type             = LR_SCHEDULER_ONE_CYCLE;
    scheduler->optimizer        = optimizer;
    scheduler->max_lr           = max_lr;
    scheduler->total_steps      = total_steps;
    scheduler->pct_start        = pct_start > 0.0f ? pct_start : 0.3f;
    scheduler->div_factor       = div_factor > 0.0f ? div_factor : 25.0f;
    scheduler->final_div_factor = final_div_factor > 0.0f ? final_div_factor : 1e4f;
    scheduler->last_epoch       = 0;

    // Set initial learning rate = max_lr / div_factor
    float init_lr = max_lr / scheduler->div_factor;
    scheduler->initial_lr = init_lr;
    scheduler->current_lr = init_lr;
    optimizer_set_lr(optimizer, init_lr);

    return scheduler;
}
