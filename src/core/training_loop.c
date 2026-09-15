#include "core/training_loop.h"
#include "core/logging.h"
#include "core/training_metrics.h"
#include "core/dataset.h"
#include "nn.h"
#include "optim.h"
#include "tensor/tensor.h"
#include "tensor/realize.h"
#include "autograd/forward_ops.h"
#include "autograd/loss_functions.h"
#include "autograd/autograd.h"
#include "ops/ir/context.h"
#include "ops/ir/execution.h"   /* cml_ir_reexecute for the static-graph mode */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "alloc/cml_allocator.h"

LRScheduler* lr_scheduler_step(Optimizer* optimizer, int step_size, float gamma) {
    if (!optimizer) {
        return NULL;
    }

    LRScheduler* scheduler = cml_calloc(1, sizeof(LRScheduler));
    if (!scheduler) {
        return NULL;
    }

    scheduler->type       = LR_SCHEDULER_STEP;
    scheduler->optimizer  = optimizer;
    scheduler->step_size  = step_size;
    scheduler->gamma      = gamma;
    scheduler->best_metric = INFINITY;

    return scheduler;
}

LRScheduler* lr_scheduler_reduce_on_plateau(Optimizer* optimizer, float factor, int patience,
                                            float min_lr) {
    if (!optimizer) {
        return NULL;
    }

    LRScheduler* scheduler = cml_calloc(1, sizeof(LRScheduler));
    if (!scheduler) {
        return NULL;
    }

    scheduler->type          = LR_SCHEDULER_REDUCE_ON_PLATEAU;
    scheduler->optimizer     = optimizer;
    scheduler->factor        = factor;
    scheduler->patience      = patience;
    scheduler->min_lr        = min_lr;
    scheduler->best_metric   = INFINITY;

    return scheduler;
}

LRScheduler* lr_scheduler_exponential(Optimizer* optimizer, float gamma) {
    if (!optimizer) {
        return NULL;
    }

    LRScheduler* scheduler = cml_calloc(1, sizeof(LRScheduler));
    if (!scheduler) {
        return NULL;
    }

    scheduler->type       = LR_SCHEDULER_EXPONENTIAL;
    scheduler->optimizer  = optimizer;
    scheduler->exp_gamma  = gamma;
    scheduler->best_metric = INFINITY;

    return scheduler;
}

LRScheduler* lr_scheduler_cosine(Optimizer* optimizer, int T_max, float eta_min) {
    if (!optimizer || T_max <= 0) {
        return NULL;
    }

    LRScheduler* scheduler = cml_calloc(1, sizeof(LRScheduler));
    if (!scheduler) {
        return NULL;
    }

    scheduler->type       = LR_SCHEDULER_COSINE;
    scheduler->optimizer  = optimizer;
    scheduler->T_max      = T_max;
    scheduler->eta_min    = eta_min;
    scheduler->min_lr     = eta_min;
    scheduler->best_metric = INFINITY;
    if (optimizer->num_param_groups > 0) {
        scheduler->initial_lr = optimizer->param_groups[0].lr;
    } else {
        scheduler->initial_lr = 0.001f;
    }

    return scheduler;
}

LRScheduler* lr_scheduler_polynomial(Optimizer* optimizer, int total_iters, float power,
                                      float min_lr) {
    if (!optimizer || total_iters <= 0) {
        return NULL;
    }

    LRScheduler* scheduler = cml_calloc(1, sizeof(LRScheduler));
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

    LRScheduler* scheduler = cml_calloc(1, sizeof(LRScheduler));
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
        float progress = (float)scheduler->last_epoch / (float)scheduler->T_max;
        if (progress > 1.0f)
            progress = 1.0f;
        if (progress < 0.0f)
            progress = 0.0f;

        float cosine_factor = (1.0f + cosf((float)M_PI * progress)) / 2.0f;
        float new_lr =
            scheduler->eta_min + (scheduler->initial_lr - scheduler->eta_min) * cosine_factor;
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
            float t = warmup_end > 0 ? (float)epoch / (float)warmup_end : 1.0f;
            new_lr  = init_lr + (scheduler->max_lr - init_lr) * t;
        } else {
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
            float t = (float)scheduler->last_epoch / (float)scheduler->warmup_steps;
            float factor = scheduler->warmup_start_factor + (1.0f - scheduler->warmup_start_factor) * t;
            float new_lr = scheduler->initial_lr * factor;

            for (int i = 0; i < scheduler->optimizer->num_param_groups; i++) {
                scheduler->optimizer->param_groups[i].lr = new_lr;
            }
        } else if (scheduler->inner_scheduler) {
            lr_scheduler_update(scheduler->inner_scheduler, metric);
        }
        break;
    }
    case LR_SCHEDULER_NONE:
    default:
        break;
    }
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
        cml_free(scheduler->milestones);
    }
    if (scheduler->inner_scheduler) {
        lr_scheduler_free(scheduler->inner_scheduler);
    }
    cml_free(scheduler);
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
    if (g_progress_callback) {
        g_progress_callback(percent, g_progress_user_data);
    }
}

/* Default hyperparameters for the training-loop entry points. */
static TrainingConfig default_training_config(void) {
    return (TrainingConfig){.epochs                    = 10,
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

/* Clip `model`'s gradients in place so their global L2 norm ≤ max_norm. No-op
 * when max_norm ≤ 0 or the norm is already within bounds. */

static void clip_gradients_by_norm(Module* model, float max_norm) {
    if (max_norm <= 0.0f)
        return;
    Parameter** params = NULL;
    int num_params     = 0;
    if (module_collect_parameters(model, &params, &num_params, true) != 0 || !params)
        return;

    float grad_norm_squared = 0.0f;
    int params_with_grad    = 0;
    for (int i = 0; i < num_params; i++) {
        if (!params[i] || !params[i]->tensor)
            continue;
        Tensor* grad = tensor_get_grad(params[i]->tensor);
        if (!grad)
            continue;
        size_t num_elements;
        float* grad_data = cml_tensor_float_buffer(grad, &num_elements);
        if (!grad_data)
            continue;
        for (size_t j = 0; j < num_elements; j++)
            grad_norm_squared += grad_data[j] * grad_data[j];
        params_with_grad++;
    }

    if (params_with_grad > 0) {
        float grad_norm = sqrtf(grad_norm_squared);
        if (grad_norm > max_norm) {
            float clip_factor = max_norm / grad_norm;
            for (int i = 0; i < num_params; i++) {
                if (!params[i] || !params[i]->tensor)
                    continue;
                Tensor* grad = tensor_get_grad(params[i]->tensor);
                if (!grad)
                    continue;
                size_t num_elements;
                float* grad_data = cml_tensor_float_buffer(grad, &num_elements);
                if (!grad_data)
                    continue;
                for (size_t j = 0; j < num_elements; j++)
                    grad_data[j] *= clip_factor;
            }
            LOG_DEBUG("Gradient clipped: norm=%.6f, threshold=%.6f, factor=%.6f",
                      (double)grad_norm, (double)max_norm, (double)clip_factor);
        }
    }
    cml_free(params);
}

int cml_train(Module* model, DataLoader* train_loader, Optimizer* optimizer,
              Tensor* (*loss_fn)(Tensor*, Tensor*), TrainingConfig* config) {
    if (!model || !train_loader || !optimizer || !loss_fn) {
        LOG_ERROR("Invalid arguments to cml_train");
        return -1;
    }
    training_metrics_register_model(model);
    TrainingConfig default_config = default_training_config();
    if (!config) {
        config = &default_config;
    }

    int epochs                  = config->epochs;
    bool verbose                = config->verbose;
    bool use_progress_bar       = config->use_progress_bar;
    LRScheduler* scheduler      = config->scheduler;
    TrainingCallbacks callbacks = config->callbacks;
    float grad_clip_norm        = config->grad_clip_norm;
    TrainingMetrics* metrics = NULL;
    if (optimizer->training_metrics) {
        metrics = (TrainingMetrics*)optimizer->training_metrics;
    }
    if (callbacks.on_training_begin) {
        callbacks.on_training_begin(callbacks.user_data);
    }

    /* Zero-rebuild static-graph mode: the fwd+bwd graph is built ONCE (first
     * batch) around fixed input buffers, then every later batch of the same shape
     * memcpys new data into those buffers, cml_ir_reexecute()s the graph (no
     * rebuild, no node allocation), and applies an in-place SGD step. */
    bool  sg_mode     = config->static_graph;
    bool  sg_captured = false;
    CMLGraph_t sg_ir  = NULL;
    Tensor* sg_X = NULL, *sg_Y = NULL, *sg_loss = NULL;
    size_t  sg_xn = 0, sg_yn = 0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        if (callbacks.on_epoch_begin) {
            callbacks.on_epoch_begin(epoch, callbacks.user_data);
        }
        dataloader_reset(train_loader);

        float epoch_loss = 0.0f;
        int num_batches  = 0;
        Batch* batch = NULL;
        while ((batch = dataloader_next_batch(train_loader)) != NULL) {
            if (callbacks.on_batch_begin) {
                callbacks.on_batch_begin(epoch, batch->batch_index, callbacks.user_data);
            }

            /* ---- static-graph fast path (after the first batch is captured) ---- */
            if (sg_mode && sg_captured) {
                if (batch->X) tensor_realize(batch->X);
                if (batch->y) tensor_realize(batch->y);
                if (!batch->X || !batch->y ||
                    batch->X->numel != sg_xn || batch->y->numel != sg_yn ||
                    !batch->X->data || !batch->y->data) {
                    /* shape changed (e.g. a smaller trailing batch) — skip it */
                    batch_free(batch);
                    continue;
                }
                memcpy(sg_X->data, batch->X->data, sg_xn * sizeof(float));
                memcpy(sg_Y->data, batch->y->data, sg_yn * sizeof(float));
                cml_ir_reexecute(sg_ir);                 /* re-run fwd+bwd, no rebuild */
                float loss_value = tensor_get_float(sg_loss, 0);
                epoch_loss += loss_value;
                num_batches++;
                optimizer_step_inplace(optimizer);       /* in-place SGD, no nodes */
                if (callbacks.on_batch_end)
                    callbacks.on_batch_end(epoch, batch->batch_index, loss_value, callbacks.user_data);
                batch_free(batch);
                continue;
            }

            /* Realize batch tensors so they survive the graph reset, then
             * clear the previous iteration's computation graph. */
            if (batch->X) tensor_realize(batch->X);
            if (batch->y) tensor_realize(batch->y);
            if (!(sg_mode && !sg_captured))
                cml_ir_reset_global_context();
            /* In static mode the first batch builds the graph around persistent
             * clones of X/y so it can be reused (batch tensors are freed below). */
            Tensor* fwd_input = batch->X;
            Tensor* loss_target = batch->y;
            if (sg_mode && !sg_captured) {
                cml_ir_reset_global_context();
                sg_X = tensor_clone(batch->X);
                sg_Y = tensor_clone(batch->y);
                if (!sg_X || !sg_Y) { LOG_ERROR("static_graph: clone failed"); batch_free(batch); return -1; }
                sg_xn = batch->X->numel; sg_yn = batch->y->numel;
                fwd_input = sg_X; loss_target = sg_Y;
            }
            Tensor* output = module_forward(model, fwd_input);
            if (!output) {
                LOG_ERROR("Forward pass failed");
                batch_free(batch);
                return -1;
            }
            Tensor* loss = loss_fn(output, loss_target);
            if (!loss) {
                LOG_ERROR("Loss computation failed");
                tensor_free(output);
                batch_free(batch);
                return -1;
            }
            float loss_value = tensor_get_float(loss, 0);
            epoch_loss += loss_value;
            num_batches++;
            optimizer->zero_grad(optimizer);
            tensor_backward(loss, NULL, false, false);
            clip_gradients_by_norm(model, grad_clip_norm);
            /* static mode: in-place SGD (no IR nodes), then CAPTURE the graph so
             * every later batch reuses it. The optimizer's normal step would emit
             * uop_sgd_step nodes into this graph — breaking the static reuse — so
             * we use the in-place update on the first batch too. */
            if (sg_mode && !sg_captured) {
                optimizer_step_inplace(optimizer);
                sg_ir     = cml_ir_get_or_create_context();
                sg_loss   = loss;              /* kept: part of the static graph */
                sg_captured = true;
            } else {
                optimizer->step(optimizer);
            }
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
            if (callbacks.on_batch_end) {
                callbacks.on_batch_end(epoch, batch->batch_index, loss_value, callbacks.user_data);
            }
            /* In static mode the loss/output tensors ARE the persistent graph — keep them. */
            if (!sg_mode) {
                tensor_free(loss);
                tensor_free(output);
            }
            batch_free(batch);
        }
        float avg_loss = num_batches > 0 ? epoch_loss / (float)num_batches : 0.0f;
        if (metrics) {
            training_metrics_record_epoch(metrics, (size_t)epoch, avg_loss, 0.0f);
        }
        if (scheduler) {
            lr_scheduler_update(scheduler, avg_loss);
        }
        if (verbose) {
            float current_lr =
                scheduler
                    ? lr_scheduler_get_lr(scheduler)
                    : (optimizer->num_param_groups > 0 ? optimizer->param_groups[0].lr : 0.0f);
            printf("Epoch %d/%d: Loss: %.4f, LR: %.6f\n", epoch + 1, epochs, (double)avg_loss,
                   (double)current_lr);
        }
        if (callbacks.on_epoch_end) {
            callbacks.on_epoch_end(epoch, avg_loss, 0.0f,
                                   callbacks.user_data);
        }
        if (use_progress_bar) {
            float progress = 100.0f * (float)(epoch + 1) / (float)epochs;
            char msg[256];
            snprintf(msg, sizeof(msg), "Epoch %d/%d completed, Loss: %.4f", epoch + 1, epochs,
                     (double)avg_loss);
            cml_print_progress_bar(progress, msg);
        }
    }
    if (use_progress_bar) {
        cml_print_progress_bar(100.0f, "Training completed");
        printf("\n");
    }
    if (callbacks.on_training_end) {
        callbacks.on_training_end(callbacks.user_data);
    }

    /* Static mode: free the persistent input clones and drop the reused graph
     * (frees the kept output/loss tensors with it). */
    if (sg_mode && sg_captured) {
        cml_ir_reset_global_context();
        if (sg_X) tensor_free(sg_X);
        if (sg_Y) tensor_free(sg_Y);
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
    training_metrics_register_model(model);
    TrainingConfig default_config = default_training_config();
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
    TrainingMetrics* metrics = NULL;
    if (optimizer->training_metrics) {
        metrics = (TrainingMetrics*)optimizer->training_metrics;
    }
    float best_val_loss  = INFINITY;
    int patience_counter = 0;
    if (callbacks.on_training_begin) {
        callbacks.on_training_begin(callbacks.user_data);
    }
    for (int epoch = 0; epoch < epochs; epoch++) {
        if (callbacks.on_epoch_begin) {
            callbacks.on_epoch_begin(epoch, callbacks.user_data);
        }
        dataloader_reset(train_loader);

        float train_loss      = 0.0f;
        int num_train_batches = 0;
        Batch* batch = NULL;
        while ((batch = dataloader_next_batch(train_loader)) != NULL) {
            if (callbacks.on_batch_begin) {
                callbacks.on_batch_begin(epoch, batch->batch_index, callbacks.user_data);
            }
            /* Realize batch tensors so they survive the graph reset, then
             * clear the previous iteration's computation graph. */
            if (batch->X) tensor_realize(batch->X);
            if (batch->y) tensor_realize(batch->y);
            cml_ir_reset_global_context();
            Tensor* output = module_forward(model, batch->X);
            if (!output) {
                LOG_ERROR("Forward pass failed");
                batch_free(batch);
                return -1;
            }
            Tensor* loss = loss_fn(output, batch->y);
            if (!loss) {
                LOG_ERROR("Loss computation failed");
                tensor_free(output);
                batch_free(batch);
                return -1;
            }
            float loss_value = tensor_get_float(loss, 0);
            train_loss += loss_value;
            num_train_batches++;
            optimizer->zero_grad(optimizer);
            tensor_backward(loss, NULL, false, false);
            clip_gradients_by_norm(model, grad_clip_norm);
            optimizer->step(optimizer);
            if (callbacks.on_batch_end) {
                callbacks.on_batch_end(epoch, batch->batch_index, loss_value, callbacks.user_data);
            }
            tensor_free(loss);
            tensor_free(output);
            batch_free(batch);
        }
        float avg_train_loss = num_train_batches > 0 ? train_loss / (float)num_train_batches : 0.0f;
        if (metrics) {
            training_metrics_record_epoch_full(metrics, (size_t)epoch, avg_train_loss, 0.0f, 0.0f,
                                               0.0f, 0.0f, 0.0f);
        }

        dataloader_reset(val_loader);

        /* Validation must run in eval mode (dropout off, batchnorm using
         * running stats) and without building training state: realize batch
         * tensors and reset the graph per batch exactly like the train loop,
         * so graph state and pool memory cannot accumulate across epochs. */
        module_set_training(model, false);
        float val_loss      = 0.0f;
        int num_val_batches = 0;
        while ((batch = dataloader_next_batch(val_loader)) != NULL) {
            if (batch->X) tensor_realize(batch->X);
            if (batch->y) tensor_realize(batch->y);
            cml_ir_reset_global_context();
            Tensor* output = module_forward(model, batch->X);
            if (!output) {
                LOG_ERROR("Validation forward pass failed");
                module_set_training(model, true);
                batch_free(batch);
                return -1;
            }
            Tensor* loss = loss_fn(output, batch->y);
            if (!loss) {
                LOG_ERROR("Validation loss computation failed");
                tensor_free(output);
                module_set_training(model, true);
                batch_free(batch);
                return -1;
            }
            float loss_value = tensor_get_float(loss, 0);
            val_loss += loss_value;
            num_val_batches++;
            tensor_free(loss);
            tensor_free(output);
            batch_free(batch);
        }
        cml_ir_reset_global_context();
        module_set_training(model, true);
        float avg_val_loss = num_val_batches > 0 ? val_loss / (float)num_val_batches : 0.0f;
        if (metrics) {
            training_metrics_record_epoch_full(metrics, (size_t)epoch, avg_train_loss, 0.0f, 0.0f,
                                               0.0f, avg_val_loss, 0.0f);
        }
        if (scheduler) {
            lr_scheduler_update(scheduler, avg_val_loss);
        }
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
        if (verbose) {
            float current_lr =
                scheduler
                    ? lr_scheduler_get_lr(scheduler)
                    : (optimizer->num_param_groups > 0 ? optimizer->param_groups[0].lr : 0.0f);
            printf("Epoch %d/%d: Train Loss: %.4f, Val Loss: %.4f, LR: %.6f\n", epoch + 1, epochs,
                   (double)avg_train_loss, (double)avg_val_loss, (double)current_lr);
        }
        if (callbacks.on_epoch_end) {
            callbacks.on_epoch_end(epoch, avg_train_loss, avg_val_loss, callbacks.user_data);
        }
        if (use_progress_bar) {
            float progress = 100.0f * (float)(epoch + 1) / (float)epochs;
            char msg[256];
            snprintf(msg, sizeof(msg), "Epoch %d/%d: Train: %.4f, Val: %.4f", epoch + 1, epochs,
                     (double)avg_train_loss, (double)avg_val_loss);
            cml_print_progress_bar(progress, msg);
        }
    }
    if (use_progress_bar) {
        cml_print_progress_bar(100.0f, "Training completed");
        printf("\n");
    }
    if (callbacks.on_training_end) {
        callbacks.on_training_end(callbacks.user_data);
    }

    return 0;
}
