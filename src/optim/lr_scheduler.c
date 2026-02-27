#include "optim/lr_scheduler.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Helper to create and initialize a base LRScheduler
 */
static LRScheduler* lr_scheduler_create_base(LRSchedulerType type, Optimizer* optimizer) {
    if (!optimizer) {
        LOG_ERROR("Cannot create LR scheduler: optimizer is NULL");
        return NULL;
    }

    LRScheduler* scheduler = (LRScheduler*)calloc(1, sizeof(LRScheduler));
    if (!scheduler) {
        LOG_ERROR("Failed to allocate LR scheduler");
        return NULL;
    }

    scheduler->type = type;
    scheduler->optimizer = optimizer;
    scheduler->initial_lr = optimizer_get_group_lr(optimizer, 0);
    scheduler->last_epoch = 0;
    scheduler->milestones = NULL;
    scheduler->num_milestones = 0;

    return scheduler;
}

/**
 * @brief Create a StepLR scheduler
 *
 * Decays the learning rate by gamma every step_size epochs.
 * new_lr = initial_lr * gamma^(epoch / step_size)
 *
 * @param optimizer Target optimizer
 * @param step_size Period of learning rate decay
 * @param gamma Multiplicative factor of learning rate decay
 * @return New StepLR scheduler, or NULL on failure
 */
LRScheduler* lr_scheduler_step(Optimizer* optimizer, int step_size, float gamma) {
    LRScheduler* scheduler = lr_scheduler_create_base(LR_SCHEDULER_STEP, optimizer);
    if (!scheduler) return NULL;

    scheduler->step_size = step_size;
    scheduler->gamma = gamma;

    return scheduler;
}

/**
 * @brief Create an ExponentialLR scheduler
 *
 * Decays the learning rate by gamma every epoch.
 * new_lr = initial_lr * gamma^epoch
 *
 * @param optimizer Target optimizer
 * @param gamma Multiplicative factor of learning rate decay
 * @return New ExponentialLR scheduler, or NULL on failure
 */
LRScheduler* lr_scheduler_exponential(Optimizer* optimizer, float gamma) {
    LRScheduler* scheduler = lr_scheduler_create_base(LR_SCHEDULER_EXPONENTIAL, optimizer);
    if (!scheduler) return NULL;

    scheduler->gamma = gamma;

    return scheduler;
}

/**
 * @brief Create a CosineAnnealingLR scheduler
 *
 * Anneals the learning rate using a cosine schedule.
 * new_lr = eta_min + (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
 *
 * @param optimizer Target optimizer
 * @param T_max Maximum number of iterations
 * @param eta_min Minimum learning rate
 * @return New CosineAnnealingLR scheduler, or NULL on failure
 */
LRScheduler* lr_scheduler_cosine_annealing(Optimizer* optimizer, int T_max, float eta_min) {
    LRScheduler* scheduler = lr_scheduler_create_base(LR_SCHEDULER_COSINE_ANNEALING, optimizer);
    if (!scheduler) return NULL;

    scheduler->T_max = T_max;
    scheduler->eta_min = eta_min;

    return scheduler;
}

/**
 * @brief Create a ReduceLROnPlateau scheduler
 *
 * Reduces learning rate when a metric has stopped improving.
 * new_lr = lr * factor when metric doesn't improve for patience epochs.
 *
 * @param optimizer Target optimizer
 * @param factor Factor by which the learning rate will be reduced
 * @param patience Number of epochs with no improvement after which LR is reduced
 * @param threshold Threshold for measuring improvement
 * @param mode_min If true, LR is reduced when metric stops decreasing; if false, when it stops increasing
 * @return New ReduceLROnPlateau scheduler, or NULL on failure
 */
LRScheduler* lr_scheduler_reduce_on_plateau(Optimizer* optimizer, float factor, int patience,
                                             float threshold, bool mode_min) {
    LRScheduler* scheduler = lr_scheduler_create_base(LR_SCHEDULER_REDUCE_ON_PLATEAU, optimizer);
    if (!scheduler) return NULL;

    scheduler->factor = factor;
    scheduler->patience = patience;
    scheduler->threshold = threshold;
    scheduler->mode_min = mode_min;
    scheduler->num_bad_epochs = 0;
    scheduler->best_metric = mode_min ? INFINITY : -INFINITY;

    return scheduler;
}

/**
 * @brief Create a MultiStepLR scheduler
 *
 * Decays the learning rate by gamma once the epoch reaches one of the milestones.
 * new_lr = initial_lr * gamma^(number of milestones passed)
 *
 * @param optimizer Target optimizer
 * @param milestones Array of epoch indices at which to decay LR
 * @param num_milestones Number of milestones
 * @param gamma Multiplicative factor of learning rate decay
 * @return New MultiStepLR scheduler, or NULL on failure
 */
LRScheduler* lr_scheduler_multi_step(Optimizer* optimizer, int* milestones, int num_milestones,
                                      float gamma) {
    LRScheduler* scheduler = lr_scheduler_create_base(LR_SCHEDULER_MULTI_STEP, optimizer);
    if (!scheduler) return NULL;

    scheduler->gamma = gamma;
    scheduler->num_milestones = num_milestones;

    if (num_milestones > 0 && milestones != NULL) {
        scheduler->milestones = (int*)malloc(sizeof(int) * num_milestones);
        if (!scheduler->milestones) {
            LOG_ERROR("Failed to allocate milestones array");
            free(scheduler);
            return NULL;
        }
        memcpy(scheduler->milestones, milestones, sizeof(int) * num_milestones);
    }

    return scheduler;
}

/**
 * @brief Compute the new learning rate based on scheduler type and current epoch
 */
static float compute_lr(LRScheduler* scheduler) {
    int epoch = scheduler->last_epoch;

    switch (scheduler->type) {
        case LR_SCHEDULER_STEP: {
            int num_decays = epoch / scheduler->step_size;
            return scheduler->initial_lr * powf(scheduler->gamma, (float)num_decays);
        }

        case LR_SCHEDULER_EXPONENTIAL: {
            return scheduler->initial_lr * powf(scheduler->gamma, (float)epoch);
        }

        case LR_SCHEDULER_COSINE_ANNEALING: {
            return scheduler->eta_min +
                   (scheduler->initial_lr - scheduler->eta_min) *
                   (1.0f + cosf((float)M_PI * (float)epoch / (float)scheduler->T_max)) / 2.0f;
        }

        case LR_SCHEDULER_MULTI_STEP: {
            int num_decays = 0;
            for (int i = 0; i < scheduler->num_milestones; i++) {
                if (epoch >= scheduler->milestones[i]) {
                    num_decays++;
                }
            }
            return scheduler->initial_lr * powf(scheduler->gamma, (float)num_decays);
        }

        case LR_SCHEDULER_REDUCE_ON_PLATEAU:
            // ReduceOnPlateau does not use epoch-based computation;
            // LR is updated in lr_scheduler_step_metric instead.
            return optimizer_get_group_lr(scheduler->optimizer, 0);

        default:
            return optimizer_get_group_lr(scheduler->optimizer, 0);
    }
}

/**
 * @brief Advance the scheduler by one epoch and update the learning rate
 *
 * For all scheduler types except ReduceOnPlateau, this increments the
 * epoch counter and computes the new learning rate.
 *
 * @param scheduler Target scheduler
 */
void lr_scheduler_step_epoch(LRScheduler* scheduler) {
    if (!scheduler) return;

    scheduler->last_epoch++;

    if (scheduler->type == LR_SCHEDULER_REDUCE_ON_PLATEAU) {
        // ReduceOnPlateau should use lr_scheduler_step_metric() instead
        return;
    }

    float new_lr = compute_lr(scheduler);
    optimizer_set_lr(scheduler->optimizer, new_lr);
}

/**
 * @brief Update the scheduler with a metric value (for ReduceOnPlateau)
 *
 * Compares the given metric against the best seen so far. If no improvement
 * is observed for patience epochs, the learning rate is reduced by factor.
 *
 * @param scheduler Target scheduler
 * @param metric The metric value to evaluate
 */
void lr_scheduler_step_metric(LRScheduler* scheduler, float metric) {
    if (!scheduler) return;

    if (scheduler->type != LR_SCHEDULER_REDUCE_ON_PLATEAU) {
        // Only ReduceOnPlateau uses metric-based stepping
        return;
    }

    bool is_better;
    if (scheduler->mode_min) {
        is_better = metric < (scheduler->best_metric - scheduler->threshold);
    } else {
        is_better = metric > (scheduler->best_metric + scheduler->threshold);
    }

    if (is_better) {
        scheduler->best_metric = metric;
        scheduler->num_bad_epochs = 0;
    } else {
        scheduler->num_bad_epochs++;
    }

    if (scheduler->num_bad_epochs >= scheduler->patience) {
        float current_lr = optimizer_get_group_lr(scheduler->optimizer, 0);
        float new_lr = current_lr * scheduler->factor;
        optimizer_set_lr(scheduler->optimizer, new_lr);
        scheduler->num_bad_epochs = 0;
    }
}

/**
 * @brief Get the current learning rate from the scheduler's optimizer
 *
 * @param scheduler Target scheduler
 * @return Current learning rate
 */
float lr_scheduler_get_lr(LRScheduler* scheduler) {
    if (!scheduler || !scheduler->optimizer) return 0.0f;

    return optimizer_get_group_lr(scheduler->optimizer, 0);
}

/**
 * @brief Free a learning rate scheduler and its resources
 *
 * @param scheduler Scheduler to free
 */
void lr_scheduler_free(LRScheduler* scheduler) {
    if (!scheduler) return;

    if (scheduler->milestones) {
        free(scheduler->milestones);
        scheduler->milestones = NULL;
    }

    free(scheduler);
}
