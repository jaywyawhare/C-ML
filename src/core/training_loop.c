#include "core/training_loop.h"
#include "core/logging.h"
#include "core/training_metrics.h"
#include "core/dataset.h"
#include "nn.h"
#include "optim.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include "autograd/loss_functions.h"
#include "autograd/autograd.h"
#include "ops/ir/context.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

LRScheduler* lr_scheduler_step(Optimizer* optimizer, int step_size, float gamma) {
    if (!optimizer) {
        return NULL;
    }

    LRScheduler* scheduler = malloc(sizeof(LRScheduler));
    if (!scheduler) {
        return NULL;
    }

    scheduler->type       = LR_SCHEDULER_STEP;
    scheduler->optimizer  = optimizer;
    scheduler->step_size  = step_size;
    scheduler->gamma      = gamma;
    scheduler->last_epoch = 0;
    scheduler->current_lr = 0.0f;

    // Initialize other fields
    scheduler->factor        = 0.0f;
    scheduler->patience      = 0;
    scheduler->min_lr        = 0.0f;
    scheduler->threshold     = 0.0f;
    scheduler->cooldown      = 0;
    scheduler->best_epoch    = 0;
    scheduler->best_metric   = INFINITY;
    scheduler->plateau_count = 0;
    scheduler->exp_gamma     = 0.0f;
    scheduler->T_max         = 0;
    scheduler->eta_min       = 0.0f;
    scheduler->initial_lr    = 0.0f;

    return scheduler;
}

LRScheduler* lr_scheduler_reduce_on_plateau(Optimizer* optimizer, float factor, int patience,
                                            float min_lr) {
    if (!optimizer) {
        return NULL;
    }

    LRScheduler* scheduler = malloc(sizeof(LRScheduler));
    if (!scheduler) {
        return NULL;
    }

    scheduler->type          = LR_SCHEDULER_REDUCE_ON_PLATEAU;
    scheduler->optimizer     = optimizer;
    scheduler->factor        = factor;
    scheduler->patience      = patience;
    scheduler->min_lr        = min_lr;
    scheduler->last_epoch    = 0;
    scheduler->current_lr    = 0.0f;
    scheduler->best_metric   = INFINITY;
    scheduler->plateau_count = 0;

    // Initialize other fields
    scheduler->step_size  = 0;
    scheduler->gamma      = 0.0f;
    scheduler->threshold  = 0.0f;
    scheduler->cooldown   = 0;
    scheduler->best_epoch = 0;
    scheduler->exp_gamma  = 0.0f;
    scheduler->T_max      = 0;
    scheduler->eta_min    = 0.0f;
    scheduler->initial_lr = 0.0f;

    return scheduler;
}

LRScheduler* lr_scheduler_exponential(Optimizer* optimizer, float gamma) {
    if (!optimizer) {
        return NULL;
    }

    LRScheduler* scheduler = malloc(sizeof(LRScheduler));
    if (!scheduler) {
        return NULL;
    }

    scheduler->type       = LR_SCHEDULER_EXPONENTIAL;
    scheduler->optimizer  = optimizer;
    scheduler->exp_gamma  = gamma;
    scheduler->last_epoch = 0;
    scheduler->current_lr = 0.0f;

    // Initialize other fields
    scheduler->step_size     = 0;
    scheduler->gamma         = 0.0f;
    scheduler->factor        = 0.0f;
    scheduler->patience      = 0;
    scheduler->min_lr        = 0.0f;
    scheduler->threshold     = 0.0f;
    scheduler->cooldown      = 0;
    scheduler->best_epoch    = 0;
    scheduler->best_metric   = INFINITY;
    scheduler->plateau_count = 0;
    scheduler->T_max         = 0;
    scheduler->eta_min       = 0.0f;
    scheduler->initial_lr    = 0.0f;

    return scheduler;
}

LRScheduler* lr_scheduler_cosine(Optimizer* optimizer, int T_max, float eta_min) {
    if (!optimizer || T_max <= 0) {
        return NULL;
    }

    LRScheduler* scheduler = malloc(sizeof(LRScheduler));
    if (!scheduler) {
        return NULL;
    }

    scheduler->type       = LR_SCHEDULER_COSINE;
    scheduler->optimizer  = optimizer;
    scheduler->T_max      = T_max;
    scheduler->eta_min    = eta_min;
    scheduler->last_epoch = 0;
    scheduler->current_lr = 0.0f;

    // Store initial learning rate for cosine annealing
    if (optimizer->num_param_groups > 0) {
        scheduler->initial_lr = optimizer->param_groups[0].lr;
    } else {
        scheduler->initial_lr = 0.001f; // Default fallback
    }

    // Initialize other fields
    scheduler->step_size     = 0;
    scheduler->gamma         = 0.0f;
    scheduler->factor        = 0.0f;
    scheduler->patience      = 0;
    scheduler->min_lr        = eta_min;
    scheduler->threshold     = 0.0f;
    scheduler->cooldown      = 0;
    scheduler->best_epoch    = 0;
    scheduler->best_metric   = INFINITY;
    scheduler->plateau_count = 0;
    scheduler->exp_gamma     = 0.0f;

    return scheduler;
}

// Note: lr_scheduler_one_cycle and lr_scheduler_multi_step are defined in optim/lr_scheduler.c

LRScheduler* lr_scheduler_polynomial(Optimizer* optimizer, int total_iters, float power,
                                      float min_lr) {
    if (!optimizer || total_iters <= 0) {
        return NULL;
    }

    LRScheduler* scheduler = calloc(1, sizeof(LRScheduler));
    if (!scheduler) {
        return NULL;
    }

    scheduler->type        = LR_SCHEDULER_POLYNOMIAL;
    scheduler->optimizer   = optimizer;
    scheduler->total_iters = total_iters;
    scheduler->power       = power;
    scheduler->min_lr      = min_lr;
    scheduler->last_epoch  = 0;

    if (optimizer->num_param_groups > 0) {
        scheduler->initial_lr = optimizer->param_groups[0].lr;
        scheduler->current_lr = scheduler->initial_lr;
    }

    return scheduler;
}

LRScheduler* lr_scheduler_warmup(LRScheduler* inner, int warmup_steps, float warmup_start_factor) {
    if (!inner || warmup_steps <= 0) {
        return NULL;
    }

    LRScheduler* scheduler = calloc(1, sizeof(LRScheduler));
    if (!scheduler) {
        return NULL;
    }

    scheduler->type                = LR_SCHEDULER_WARMUP;
    scheduler->optimizer           = inner->optimizer;
    scheduler->inner_scheduler     = inner;
    scheduler->warmup_steps        = warmup_steps;
    scheduler->warmup_start_factor = warmup_start_factor;
    scheduler->last_epoch          = 0;

    if (scheduler->optimizer && scheduler->optimizer->num_param_groups > 0) {
        scheduler->initial_lr = scheduler->optimizer->param_groups[0].lr;
        scheduler->current_lr = scheduler->initial_lr * warmup_start_factor;
        // Set initial warmup LR
        for (int i = 0; i < scheduler->optimizer->num_param_groups; i++) {
            scheduler->optimizer->param_groups[i].lr = scheduler->current_lr;
        }
    }

    return scheduler;
}

float lr_scheduler_update(LRScheduler* scheduler, float metric) {
    if (!scheduler || !scheduler->optimizer) {
        return 0.0f;
    }

    scheduler->last_epoch++;

    switch (scheduler->type) {
    case LR_SCHEDULER_STEP: {
        if (scheduler->step_size > 0 && scheduler->last_epoch % scheduler->step_size == 0) {
            // Reduce learning rate
            for (int i = 0; i < scheduler->optimizer->num_param_groups; i++) {
                ParameterGroup* pg = &scheduler->optimizer->param_groups[i];
                pg->lr *= scheduler->gamma;
                if (scheduler->min_lr > 0.0f && pg->lr < scheduler->min_lr) {
                    pg->lr = scheduler->min_lr;
                }
            }
        }
        break;
    }
    case LR_SCHEDULER_REDUCE_ON_PLATEAU: {
        if (metric < scheduler->best_metric) {
            scheduler->best_metric   = metric;
            scheduler->best_epoch    = scheduler->last_epoch;
            scheduler->plateau_count = 0;
        } else {
            scheduler->plateau_count++;
            if (scheduler->plateau_count >= scheduler->patience) {
                // Reduce learning rate
                for (int i = 0; i < scheduler->optimizer->num_param_groups; i++) {
                    ParameterGroup* pg = &scheduler->optimizer->param_groups[i];
                    pg->lr *= scheduler->factor;
                    if (pg->lr < scheduler->min_lr) {
                        pg->lr = scheduler->min_lr;
                    }
                }
                scheduler->plateau_count = 0;
            }
        }
        break;
    }
    case LR_SCHEDULER_EXPONENTIAL: {
        // Reduce learning rate exponentially
        for (int i = 0; i < scheduler->optimizer->num_param_groups; i++) {
            ParameterGroup* pg = &scheduler->optimizer->param_groups[i];
            pg->lr *= scheduler->exp_gamma;
            if (scheduler->min_lr > 0.0f && pg->lr < scheduler->min_lr) {
                pg->lr = scheduler->min_lr;
            }
        }
        break;
    }
    case LR_SCHEDULER_COSINE: {
        // CosineAnnealingLR: lr = eta_min + (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max))
        // / 2
        float progress = (float)scheduler->last_epoch / (float)scheduler->T_max;
        // Clamp progress to [0, 1] for safety
        if (progress > 1.0f)
            progress = 1.0f;
        if (progress < 0.0f)
            progress = 0.0f;

        float cosine_factor = (1.0f + cosf((float)M_PI * progress)) / 2.0f;
        float new_lr =
            scheduler->eta_min + (scheduler->initial_lr - scheduler->eta_min) * cosine_factor;

        // Apply to all parameter groups
        for (int i = 0; i < scheduler->optimizer->num_param_groups; i++) {
            scheduler->optimizer->param_groups[i].lr = new_lr;
        }
        break;
    }
    case LR_SCHEDULER_ONE_CYCLE: {
        float init_lr  = scheduler->max_lr / scheduler->div_factor;
        float final_lr = init_lr / scheduler->final_div_factor;
        int warmup_end = (int)(scheduler->pct_start * (float)scheduler->total_steps);
        int epoch      = scheduler->last_epoch;

        float new_lr;
        if (epoch <= warmup_end) {
            // Linear warmup: init_lr -> max_lr
            float t = warmup_end > 0 ? (float)epoch / (float)warmup_end : 1.0f;
            new_lr  = init_lr + (scheduler->max_lr - init_lr) * t;
        } else {
            // Cosine annealing: max_lr -> final_lr
            int anneal_steps = scheduler->total_steps - warmup_end;
            float t = anneal_steps > 0
                          ? (float)(epoch - warmup_end) / (float)anneal_steps
                          : 1.0f;
            if (t > 1.0f) t = 1.0f;
            new_lr = final_lr + (scheduler->max_lr - final_lr) * (1.0f + cosf((float)M_PI * t)) / 2.0f;
        }

        for (int i = 0; i < scheduler->optimizer->num_param_groups; i++) {
            scheduler->optimizer->param_groups[i].lr = new_lr;
        }
        break;
    }
    case LR_SCHEDULER_MULTI_STEP: {
        // Check if current epoch is a milestone
        for (int m = 0; m < scheduler->num_milestones; m++) {
            if (scheduler->last_epoch == scheduler->milestones[m]) {
                for (int i = 0; i < scheduler->optimizer->num_param_groups; i++) {
                    scheduler->optimizer->param_groups[i].lr *= scheduler->gamma;
                }
                break;
            }
        }
        break;
    }
    case LR_SCHEDULER_POLYNOMIAL: {
        // lr = (initial_lr - min_lr) * (1 - epoch/total_iters)^power + min_lr
        float progress = (float)scheduler->last_epoch / (float)scheduler->total_iters;
        if (progress > 1.0f) progress = 1.0f;

        float decay  = powf(1.0f - progress, scheduler->power);
        float new_lr = (scheduler->initial_lr - scheduler->min_lr) * decay + scheduler->min_lr;

        for (int i = 0; i < scheduler->optimizer->num_param_groups; i++) {
            scheduler->optimizer->param_groups[i].lr = new_lr;
        }
        break;
    }
    case LR_SCHEDULER_WARMUP: {
        if (scheduler->last_epoch <= scheduler->warmup_steps) {
            // Linear warmup: start_factor -> 1.0
            float t = (float)scheduler->last_epoch / (float)scheduler->warmup_steps;
            float factor = scheduler->warmup_start_factor + (1.0f - scheduler->warmup_start_factor) * t;
            float new_lr = scheduler->initial_lr * factor;

            for (int i = 0; i < scheduler->optimizer->num_param_groups; i++) {
                scheduler->optimizer->param_groups[i].lr = new_lr;
            }
        } else if (scheduler->inner_scheduler) {
            // Delegate to inner scheduler
            lr_scheduler_update(scheduler->inner_scheduler, metric);
        }
        break;
    }
    case LR_SCHEDULER_NONE:
    default:
        break;
    }

    // Get current learning rate
    if (scheduler->optimizer->num_param_groups > 0) {
        scheduler->current_lr = scheduler->optimizer->param_groups[0].lr;
    }

    return scheduler->current_lr;
}

float lr_scheduler_get_lr(LRScheduler* scheduler) {
    if (!scheduler || !scheduler->optimizer) {
        return 0.0f;
    }

    if (scheduler->optimizer->num_param_groups > 0) {
        return scheduler->optimizer->param_groups[0].lr;
    }

    return 0.0f;
}

void lr_scheduler_free(LRScheduler* scheduler) {
    if (!scheduler) {
        return;
    }
    if (scheduler->milestones) {
        free(scheduler->milestones);
    }
    if (scheduler->inner_scheduler) {
        lr_scheduler_free(scheduler->inner_scheduler);
    }
    free(scheduler);
}

void training_callbacks_create(TrainingCallbacks* callbacks) {
    if (!callbacks)
        return;
    *callbacks = (TrainingCallbacks){0};
}

void training_config_default(TrainingConfig* config) {
    if (!config)
        return;
    *config = (TrainingConfig){.epochs                    = 10,
                               .verbose                   = true,
                               .use_progress_bar          = true,
                               .scheduler                 = NULL,
                               .callbacks                 = (TrainingCallbacks){0},
                               .grad_clip_norm            = 0.0f,
                               .early_stopping            = false,
                               .early_stopping_patience   = 5,
                               .early_stopping_min_delta  = 0.0f,
                               .use_checkpointing         = false,
                               .checkpoint_every_n_layers = 0};
}

static ProgressCallback g_progress_callback = NULL;
static void* g_progress_user_data           = NULL;

void cml_set_progress_callback(ProgressCallback callback, void* user_data) {
    g_progress_callback  = callback;
    g_progress_user_data = user_data;
}

void cml_print_progress_bar(float percent, const char* message) {
    if (percent < 0.0f)
        percent = 0.0f;
    if (percent > 100.0f)
        percent = 100.0f;

    int bar_width = 50;
    int pos       = (int)((float)bar_width * percent / 100.0f);

    printf("\r[");
    for (int i = 0; i < bar_width; i++) {
        if (i < pos) {
            printf("=");
        } else if (i == pos) {
            printf(">");
        } else {
            printf(" ");
        }
    }
    printf("] %.1f%%", (double)percent);

    if (message) {
        printf(" %s", message);
    }

    fflush(stdout);

    // Call user callback if set
    if (g_progress_callback) {
        g_progress_callback(percent, g_progress_user_data);
    }
}

int cml_train(Module* model, DataLoader* train_loader, Optimizer* optimizer,
              Tensor* (*loss_fn)(Tensor*, Tensor*), TrainingConfig* config) {
    if (!model || !train_loader || !optimizer || !loss_fn) {
        LOG_ERROR("Invalid arguments to cml_train");
        return -1;
    }

    // Automatically register model for metrics tracking
    training_metrics_register_model(model);

    // Use default config if not provided
    TrainingConfig default_config = {.epochs                    = 10,
                                     .verbose                   = true,
                                     .use_progress_bar          = true,
                                     .scheduler                 = NULL,
                                     .callbacks                 = (TrainingCallbacks){0},
                                     .grad_clip_norm            = 0.0f,
                                     .early_stopping            = false,
                                     .early_stopping_patience   = 5,
                                     .early_stopping_min_delta  = 0.0f,
                                     .use_checkpointing         = false,
                                     .checkpoint_every_n_layers = 0};
    if (!config) {
        config = &default_config;
    }

    int epochs                  = config->epochs;
    bool verbose                = config->verbose;
    bool use_progress_bar       = config->use_progress_bar;
    LRScheduler* scheduler      = config->scheduler;
    TrainingCallbacks callbacks = config->callbacks;
    float grad_clip_norm        = config->grad_clip_norm;

    // Initialize metrics if available
    TrainingMetrics* metrics = NULL;
    if (optimizer->training_metrics) {
        metrics = (TrainingMetrics*)optimizer->training_metrics;
    }

    // Call training begin callback
    if (callbacks.on_training_begin) {
        callbacks.on_training_begin(callbacks.user_data);
    }

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Call epoch begin callback
        if (callbacks.on_epoch_begin) {
            callbacks.on_epoch_begin(epoch, callbacks.user_data);
        }

        // Reset dataloader for new epoch
        dataloader_reset(train_loader);

        float epoch_loss = 0.0f;
        int num_batches  = 0;

        // Iterate over batches
        Batch* batch = NULL;
        while ((batch = dataloader_next_batch(train_loader)) != NULL) {
            // Call batch begin callback
            if (callbacks.on_batch_begin) {
                callbacks.on_batch_begin(epoch, batch->batch_index, callbacks.user_data);
            }

            // Reset IR context at the start of each batch to prevent accumulation
            cml_ir_reset_global_context();

            // Forward pass
            Tensor* output = module_forward(model, batch->X);
            if (!output) {
                LOG_ERROR("Forward pass failed");
                batch_free(batch);
                return -1;
            }

            // Compute loss
            Tensor* loss = loss_fn(output, batch->y);
            if (!loss) {
                LOG_ERROR("Loss computation failed");
                tensor_free(output);
                batch_free(batch);
                return -1;
            }

            // Get loss value
            float loss_value = tensor_get_float(loss, 0);
            epoch_loss += loss_value;
            num_batches++;

            // Zero gradients
            optimizer->zero_grad(optimizer);

            // Backward pass
            tensor_backward(loss, NULL, false, false);

            // Gradient clipping (if enabled)
            if (grad_clip_norm > 0.0f) {
                // Collect all parameters from the model
                Parameter** params = NULL;
                int num_params     = 0;
                if (module_collect_parameters(model, &params, &num_params, true) == 0 && params) {
                    // Calculate gradient norm
                    float grad_norm_squared = 0.0f;
                    int params_with_grad    = 0;

                    for (int i = 0; i < num_params; i++) {
                        if (!params[i] || !params[i]->tensor)
                            continue;

                        Tensor* grad = tensor_get_grad(params[i]->tensor);
                        if (!grad)
                            continue;

                        float* grad_data = (float*)tensor_data_ptr(grad);
                        if (!grad_data)
                            continue;

                        size_t num_elements = grad->numel;
                        if (num_elements == 0 && grad->shape && grad->ndim > 0) {
                            num_elements = 1;
                            for (int d = 0; d < grad->ndim; d++) {
                                num_elements *= (size_t)grad->shape[d];
                            }
                        }

                        for (size_t j = 0; j < num_elements; j++) {
                            float g = grad_data[j];
                            grad_norm_squared += g * g;
                        }
                        params_with_grad++;
                    }

                    if (params_with_grad > 0) {
                        float grad_norm = sqrtf(grad_norm_squared);

                        // Clip gradients if norm exceeds threshold
                        if (grad_norm > grad_clip_norm) {
                            float clip_factor = grad_clip_norm / grad_norm;

                            for (int i = 0; i < num_params; i++) {
                                if (!params[i] || !params[i]->tensor)
                                    continue;

                                Tensor* grad = tensor_get_grad(params[i]->tensor);
                                if (!grad)
                                    continue;

                                float* grad_data = (float*)tensor_data_ptr(grad);
                                if (!grad_data)
                                    continue;

                                size_t num_elements = grad->numel;
                                if (num_elements == 0 && grad->shape && grad->ndim > 0) {
                                    num_elements = 1;
                                    for (int d = 0; d < grad->ndim; d++) {
                                        num_elements *= (size_t)grad->shape[d];
                                    }
                                }

                                // Scale gradient by clip factor
                                for (size_t j = 0; j < num_elements; j++) {
                                    grad_data[j] *= clip_factor;
                                }
                            }

                            LOG_DEBUG("Gradient clipped: norm=%.6f, threshold=%.6f, factor=%.6f",
                                      (double)grad_norm, (double)grad_clip_norm,
                                      (double)clip_factor);
                        }
                    }

                    free(params);
                }
            }

            // Optimizer step
            optimizer->step(optimizer);

            // Update progress bar
            if (use_progress_bar && num_batches % 10 == 0) {
                float progress = 100.0f *
                                 (float)(epoch * train_loader->total_batches + batch->batch_index) /
                                 (float)(epochs * train_loader->total_batches);
                char msg[256];
                snprintf(msg, sizeof(msg), "Epoch %d/%d, Batch %d/%d, Loss: %.4f", epoch + 1,
                         epochs, batch->batch_index + 1, train_loader->total_batches,
                         (double)loss_value);
                cml_print_progress_bar(progress, msg);
            }

            // Call batch end callback
            if (callbacks.on_batch_end) {
                callbacks.on_batch_end(epoch, batch->batch_index, loss_value, callbacks.user_data);
            }

            // Cleanup
            tensor_free(loss);
            tensor_free(output);
            batch_free(batch);
        }

        // Calculate average loss for epoch
        float avg_loss = num_batches > 0 ? epoch_loss / (float)num_batches : 0.0f;

        // Update metrics
        if (metrics) {
            training_metrics_record_epoch(metrics, (size_t)epoch, avg_loss, 0.0f);
        }

        // Step learning rate scheduler
        if (scheduler) {
            lr_scheduler_update(scheduler, avg_loss);
        }

        // Print epoch summary
        if (verbose) {
            float current_lr =
                scheduler
                    ? lr_scheduler_get_lr(scheduler)
                    : (optimizer->num_param_groups > 0 ? optimizer->param_groups[0].lr : 0.0f);
            printf("Epoch %d/%d: Loss: %.4f, LR: %.6f\n", epoch + 1, epochs, (double)avg_loss,
                   (double)current_lr);
        }

        // Call epoch end callback
        if (callbacks.on_epoch_end) {
            callbacks.on_epoch_end(epoch, avg_loss, 0.0f,
                                   callbacks.user_data); // Accuracy not computed
        }

        // Update progress bar for epoch completion
        if (use_progress_bar) {
            float progress = 100.0f * (float)(epoch + 1) / (float)epochs;
            char msg[256];
            snprintf(msg, sizeof(msg), "Epoch %d/%d completed, Loss: %.4f", epoch + 1, epochs,
                     (double)avg_loss);
            cml_print_progress_bar(progress, msg);
        }
    }

    // Final progress bar update
    if (use_progress_bar) {
        cml_print_progress_bar(100.0f, "Training completed");
        printf("\n");
    }

    // Call training end callback
    if (callbacks.on_training_end) {
        callbacks.on_training_end(callbacks.user_data);
    }

    return 0;
}

int cml_train_with_validation(Module* model, DataLoader* train_loader, DataLoader* val_loader,
                              Optimizer* optimizer, Tensor* (*loss_fn)(Tensor*, Tensor*),
                              TrainingConfig* config) {
    if (!model || !train_loader || !val_loader || !optimizer || !loss_fn) {
        LOG_ERROR("Invalid arguments to cml_train_with_validation");
        return -1;
    }

    // Automatically register model for metrics tracking
    training_metrics_register_model(model);

    // Use default config if not provided
    TrainingConfig default_config = {.epochs                    = 10,
                                     .verbose                   = true,
                                     .use_progress_bar          = true,
                                     .scheduler                 = NULL,
                                     .callbacks                 = (TrainingCallbacks){0},
                                     .grad_clip_norm            = 0.0f,
                                     .early_stopping            = false,
                                     .early_stopping_patience   = 5,
                                     .early_stopping_min_delta  = 0.0f,
                                     .use_checkpointing         = false,
                                     .checkpoint_every_n_layers = 0};
    if (!config) {
        config = &default_config;
    }

    int epochs                     = config->epochs;
    bool verbose                   = config->verbose;
    bool use_progress_bar          = config->use_progress_bar;
    LRScheduler* scheduler         = config->scheduler;
    TrainingCallbacks callbacks    = config->callbacks;
    float grad_clip_norm           = config->grad_clip_norm;
    bool early_stopping            = config->early_stopping;
    int early_stopping_patience    = config->early_stopping_patience;
    float early_stopping_min_delta = config->early_stopping_min_delta;

    // Initialize metrics if available
    TrainingMetrics* metrics = NULL;
    if (optimizer->training_metrics) {
        metrics = (TrainingMetrics*)optimizer->training_metrics;
    }

    // Early stopping state
    float best_val_loss  = INFINITY;
    int patience_counter = 0;

    // Call training begin callback
    if (callbacks.on_training_begin) {
        callbacks.on_training_begin(callbacks.user_data);
    }

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Call epoch begin callback
        if (callbacks.on_epoch_begin) {
            callbacks.on_epoch_begin(epoch, callbacks.user_data);
        }

        // Reset dataloader for new epoch
        dataloader_reset(train_loader);

        float train_loss      = 0.0f;
        int num_train_batches = 0;

        // Iterate over training batches
        Batch* batch = NULL;
        while ((batch = dataloader_next_batch(train_loader)) != NULL) {
            // Call batch begin callback
            if (callbacks.on_batch_begin) {
                callbacks.on_batch_begin(epoch, batch->batch_index, callbacks.user_data);
            }

            // Reset IR context at the start of each batch to prevent accumulation
            cml_ir_reset_global_context();

            // Forward pass
            Tensor* output = module_forward(model, batch->X);
            if (!output) {
                LOG_ERROR("Forward pass failed");
                batch_free(batch);
                return -1;
            }

            // Compute loss
            Tensor* loss = loss_fn(output, batch->y);
            if (!loss) {
                LOG_ERROR("Loss computation failed");
                tensor_free(output);
                batch_free(batch);
                return -1;
            }

            // Get loss value
            float loss_value = tensor_get_float(loss, 0);
            train_loss += loss_value;
            num_train_batches++;

            // Zero gradients
            optimizer->zero_grad(optimizer);

            // Backward pass
            tensor_backward(loss, NULL, false, false);

            // Gradient clipping (if enabled)
            if (grad_clip_norm > 0.0f) {
                // Collect all parameters from the model
                Parameter** params = NULL;
                int num_params     = 0;
                if (module_collect_parameters(model, &params, &num_params, true) == 0 && params) {
                    // Calculate gradient norm
                    float grad_norm_squared = 0.0f;
                    int params_with_grad    = 0;

                    for (int i = 0; i < num_params; i++) {
                        if (!params[i] || !params[i]->tensor)
                            continue;

                        Tensor* grad = tensor_get_grad(params[i]->tensor);
                        if (!grad)
                            continue;

                        float* grad_data = (float*)tensor_data_ptr(grad);
                        if (!grad_data)
                            continue;

                        size_t num_elements = grad->numel;
                        if (num_elements == 0 && grad->shape && grad->ndim > 0) {
                            num_elements = 1;
                            for (int d = 0; d < grad->ndim; d++) {
                                num_elements *= (size_t)grad->shape[d];
                            }
                        }

                        for (size_t j = 0; j < num_elements; j++) {
                            float g = grad_data[j];
                            grad_norm_squared += g * g;
                        }
                        params_with_grad++;
                    }

                    if (params_with_grad > 0) {
                        float grad_norm = sqrtf(grad_norm_squared);

                        // Clip gradients if norm exceeds threshold
                        if (grad_norm > grad_clip_norm) {
                            float clip_factor = grad_clip_norm / grad_norm;

                            for (int i = 0; i < num_params; i++) {
                                if (!params[i] || !params[i]->tensor)
                                    continue;

                                Tensor* grad = tensor_get_grad(params[i]->tensor);
                                if (!grad)
                                    continue;

                                float* grad_data = (float*)tensor_data_ptr(grad);
                                if (!grad_data)
                                    continue;

                                size_t num_elements = grad->numel;
                                if (num_elements == 0 && grad->shape && grad->ndim > 0) {
                                    num_elements = 1;
                                    for (int d = 0; d < grad->ndim; d++) {
                                        num_elements *= (size_t)grad->shape[d];
                                    }
                                }

                                // Scale gradient by clip factor
                                for (size_t j = 0; j < num_elements; j++) {
                                    grad_data[j] *= clip_factor;
                                }
                            }

                            LOG_DEBUG("Gradient clipped: norm=%.6f, threshold=%.6f, factor=%.6f",
                                      (double)grad_norm, (double)grad_clip_norm,
                                      (double)clip_factor);
                        }
                    }

                    free(params);
                }
            }

            // Optimizer step
            optimizer->step(optimizer);

            // Call batch end callback
            if (callbacks.on_batch_end) {
                callbacks.on_batch_end(epoch, batch->batch_index, loss_value, callbacks.user_data);
            }

            // Cleanup
            tensor_free(loss);
            tensor_free(output);
            batch_free(batch);
        }

        // Calculate average training loss
        float avg_train_loss = num_train_batches > 0 ? train_loss / (float)num_train_batches : 0.0f;

        // Update metrics
        if (metrics) {
            training_metrics_record_epoch_full(metrics, (size_t)epoch, avg_train_loss, 0.0f, 0.0f,
                                               0.0f, 0.0f, 0.0f);
        }

        // Set model to evaluation mode (if supported)
        // For now, we'll just run validation

        dataloader_reset(val_loader);

        float val_loss      = 0.0f;
        int num_val_batches = 0;

        // Iterate over validation batches
        while ((batch = dataloader_next_batch(val_loader)) != NULL) {
            // Forward pass (no gradients)
            Tensor* output = module_forward(model, batch->X);
            if (!output) {
                LOG_ERROR("Validation forward pass failed");
                batch_free(batch);
                return -1;
            }

            // Compute loss
            Tensor* loss = loss_fn(output, batch->y);
            if (!loss) {
                LOG_ERROR("Validation loss computation failed");
                tensor_free(output);
                batch_free(batch);
                return -1;
            }

            // Get loss value
            float loss_value = tensor_get_float(loss, 0);
            val_loss += loss_value;
            num_val_batches++;

            // Cleanup
            tensor_free(loss);
            tensor_free(output);
            batch_free(batch);
        }

        // Calculate average validation loss
        float avg_val_loss = num_val_batches > 0 ? val_loss / (float)num_val_batches : 0.0f;

        // Update metrics with validation loss
        if (metrics) {
            training_metrics_record_epoch_full(metrics, (size_t)epoch, avg_train_loss, 0.0f, 0.0f,
                                               0.0f, avg_val_loss, 0.0f);
        }

        // Step learning rate scheduler (using validation loss)
        if (scheduler) {
            lr_scheduler_update(scheduler, avg_val_loss);
        }

        // Early stopping check
        if (early_stopping) {
            if (avg_val_loss < best_val_loss - early_stopping_min_delta) {
                best_val_loss    = avg_val_loss;
                patience_counter = 0;
            } else {
                patience_counter++;
                if (patience_counter >= early_stopping_patience) {
                    if (verbose) {
                        printf("Early stopping triggered at epoch %d\n", epoch + 1);
                    }
                    break;
                }
            }
        }

        // Print epoch summary
        if (verbose) {
            float current_lr =
                scheduler
                    ? lr_scheduler_get_lr(scheduler)
                    : (optimizer->num_param_groups > 0 ? optimizer->param_groups[0].lr : 0.0f);
            printf("Epoch %d/%d: Train Loss: %.4f, Val Loss: %.4f, LR: %.6f\n", epoch + 1, epochs,
                   (double)avg_train_loss, (double)avg_val_loss, (double)current_lr);
        }

        // Call epoch end callback
        if (callbacks.on_epoch_end) {
            callbacks.on_epoch_end(epoch, avg_train_loss, avg_val_loss, callbacks.user_data);
        }

        // Update progress bar
        if (use_progress_bar) {
            float progress = 100.0f * (float)(epoch + 1) / (float)epochs;
            char msg[256];
            snprintf(msg, sizeof(msg), "Epoch %d/%d: Train: %.4f, Val: %.4f", epoch + 1, epochs,
                     (double)avg_train_loss, (double)avg_val_loss);
            cml_print_progress_bar(progress, msg);
        }
    }

    // Final progress bar update
    if (use_progress_bar) {
        cml_print_progress_bar(100.0f, "Training completed");
        printf("\n");
    }

    // Call training end callback
    if (callbacks.on_training_end) {
        callbacks.on_training_end(callbacks.user_data);
    }

    return 0;
}
