/**
 * @file optim.c
 * @brief Optimizer implementation for neural network training - PRODUCTION-READY
 *
 * This file provides optimizer implementations (SGD, Adam, etc.) that work
 * with the training API. The implementations support:
 * - **Per-group hyperparameters** (learning rate, weight decay)
 * - Parameter groups with different learning rates
 * - Momentum and adaptive learning rates
 * - Weight decay (L2 regularization)
 * - State management for advanced optimizers like Adam
 * - Dynamic parameter group storage with automatic resizing
 *
 * - Each parameter group can have its own learning rate and weight decay
 * - Groups are stored in a dynamically allocated array
 * - Efficient parameter management across multiple groups
 * - Memory-safe implementation with proper cleanup
 *
 * Additional optimizer variants (RMSprop, AdamW, etc.) can be added as needed.
 */

#include "optim/optimizer.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include "tensor/tensor.h"
#include "autograd/autograd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Optimizer Implementation (Simplified)

int optimizer_init(Optimizer* optimizer, const char* name, StepFn step, ZeroGradFn zero_grad) {
    if (!optimizer || !name || !step || !zero_grad)
        return -1;

    optimizer->name                   = name;
    optimizer->step                   = step;
    optimizer->zero_grad              = zero_grad;
    optimizer->param_groups           = NULL;
    optimizer->num_param_groups       = 0;
    optimizer->param_groups_capacity  = 0;
    optimizer->use_amp                = false;
    optimizer->grad_clip_norm         = 0.0f;
    optimizer->amsgrad                = false;
    optimizer->lr_scheduler_factor    = 1.0f;
    optimizer->lr_scheduler_step_size = 0;
    optimizer->lr_scheduler_gamma     = 1.0f;
    optimizer->version                = "1.0.0";
    optimizer->description            = "Optimizer";

    return 0;
}

Optimizer* optimizer_create(const char* name, StepFn step, ZeroGradFn zero_grad) {
    Optimizer* optimizer = CM_MALLOC(sizeof(Optimizer));
    if (!optimizer)
        return NULL;

    if (optimizer_init(optimizer, name, step, zero_grad) != 0) {
        CM_FREE(optimizer);
        return NULL;
    }

    return optimizer;
}

// Forward declarations for optimizer state structures
typedef struct SGDMomentumState {
    Tensor* momentum_buffer;
} SGDMomentumState;

typedef struct AdamState {
    Tensor* exp_avg;
    Tensor* exp_avg_sq;
    Tensor* max_exp_avg_sq;
} AdamState;

typedef struct RMSpropState {
    Tensor* square_avg;
} RMSpropState;

typedef struct AdagradState {
    Tensor* sum_sq_grad;
} AdagradState;

void optimizer_free(Optimizer* optimizer) {
    if (!optimizer)
        return;

    // Free each parameter group
    if (optimizer->param_groups) {
        for (int i = 0; i < optimizer->num_param_groups; i++) {
            ParameterGroup* group = &optimizer->param_groups[i];
            // Free parameters array (just the array, not the parameters themselves)
            if (group->parameters) {
                CM_FREE(group->parameters);
            }
            // Free optimizer-specific state if it exists
            if (group->state) {
                // Free state based on optimizer type
                if (strcmp(optimizer->name, "SGD") == 0) {
                    // Free SGD momentum states
                    SGDMomentumState** states = (SGDMomentumState**)group->state;
                    for (int j = 0; j < group->num_parameters; j++) {
                        if (states[j]) {
                            if (states[j]->momentum_buffer) {
                                tensor_free(states[j]->momentum_buffer);
                            }
                            CM_FREE(states[j]);
                        }
                    }
                    CM_FREE(states);
                } else if (strcmp(optimizer->name, "Adam") == 0) {
                    // Free Adam states
                    AdamState** states = (AdamState**)group->state;
                    for (int j = 0; j < group->num_parameters; j++) {
                        if (states[j]) {
                            if (states[j]->exp_avg)
                                tensor_free(states[j]->exp_avg);
                            if (states[j]->exp_avg_sq)
                                tensor_free(states[j]->exp_avg_sq);
                            if (states[j]->max_exp_avg_sq)
                                tensor_free(states[j]->max_exp_avg_sq);
                            CM_FREE(states[j]);
                        }
                    }
                    CM_FREE(states);
                } else if (strcmp(optimizer->name, "RMSprop") == 0) {
                    // Free RMSprop states
                    RMSpropState** states = (RMSpropState**)group->state;
                    for (int j = 0; j < group->num_parameters; j++) {
                        if (states[j]) {
                            if (states[j]->square_avg)
                                tensor_free(states[j]->square_avg);
                            CM_FREE(states[j]);
                        }
                    }
                    CM_FREE(states);
                } else if (strcmp(optimizer->name, "Adagrad") == 0) {
                    // Free Adagrad states
                    AdagradState** states = (AdagradState**)group->state;
                    for (int j = 0; j < group->num_parameters; j++) {
                        if (states[j]) {
                            if (states[j]->sum_sq_grad)
                                tensor_free(states[j]->sum_sq_grad);
                            CM_FREE(states[j]);
                        }
                    }
                    CM_FREE(states);
                } else {
                    // Generic state cleanup
                    CM_FREE(group->state);
                }
            }
        }
        CM_FREE(optimizer->param_groups);
    }

    // Free the optimizer itself
    CM_FREE(optimizer);
}

int optimizer_add_param_group(Optimizer* optimizer, Parameter** parameters, int num_parameters,
                              float lr, float weight_decay) {
    if (!optimizer || !parameters || num_parameters <= 0) {
        LOG_ERROR("Invalid arguments to optimizer_add_param_group");
        return -1;
    }

    // Resize array if needed
    if (optimizer->num_param_groups >= optimizer->param_groups_capacity) {
        int new_capacity =
            optimizer->param_groups_capacity == 0 ? 4 : optimizer->param_groups_capacity * 2;
        ParameterGroup* new_groups =
            CM_REALLOC(optimizer->param_groups, new_capacity * sizeof(ParameterGroup));
        if (!new_groups) {
            LOG_ERROR("Failed to allocate memory for parameter groups");
            return -1;
        }
        optimizer->param_groups          = new_groups;
        optimizer->param_groups_capacity = new_capacity;
    }

    // Initialize new parameter group
    ParameterGroup* group = &optimizer->param_groups[optimizer->num_param_groups];

    // Allocate parameters array
    group->parameters = CM_MALLOC(num_parameters * sizeof(Parameter*));
    if (!group->parameters) {
        LOG_ERROR("Failed to allocate memory for parameter group parameters");
        return -1;
    }

    // Copy parameter pointers
    for (int i = 0; i < num_parameters; i++) {
        group->parameters[i] = parameters[i];
    }
    group->num_parameters = num_parameters;

    // Set hyperparameters
    group->lr           = lr;
    group->weight_decay = weight_decay;
    group->momentum     = 0.0f;   // Default, can be set later
    group->beta1        = 0.9f;   // Adam default
    group->beta2        = 0.999f; // Adam default
    group->epsilon      = 1e-8f;  // Adam default

    // Initialize state
    group->state      = NULL;
    group->step_count = 0;

    optimizer->num_param_groups++;

    LOG_DEBUG("Added parameter group %d with %d parameters (lr=%.6f, wd=%.6f)",
              optimizer->num_param_groups - 1, num_parameters, lr, weight_decay);

    return 0;
}

int optimizer_get_param_groups(Optimizer* optimizer, ParameterGroup** groups, int* num_groups) {
    if (!optimizer)
        return -1;

    if (num_groups) {
        *num_groups = optimizer->num_param_groups;
    }

    if (groups && optimizer->num_param_groups > 0) {
        *groups = optimizer->param_groups;
    }

    return 0;
}

ParameterGroup* optimizer_get_param_group(Optimizer* optimizer, int index) {
    if (!optimizer || index < 0 || index >= optimizer->num_param_groups) {
        return NULL;
    }

    return &optimizer->param_groups[index];
}

void optimizer_step(Optimizer* optimizer) {
    if (!optimizer || !optimizer->step)
        return;

    // Call the step function
    optimizer->step(optimizer);
}

void optimizer_zero_grad(Optimizer* optimizer) {
    if (!optimizer || !optimizer->zero_grad)
        return;

    // Call the zero_grad function
    optimizer->zero_grad(optimizer);
}

int optimizer_get_step_count(Optimizer* optimizer) {
    if (!optimizer || optimizer->num_param_groups == 0)
        return 0;

    // Return step count from first parameter group
    return optimizer->param_groups[0].step_count;
}

void optimizer_set_lr(Optimizer* optimizer, float lr) {
    if (!optimizer)
        return;

    // Set learning rate for all parameter groups
    for (int i = 0; i < optimizer->num_param_groups; i++) {
        optimizer->param_groups[i].lr = lr;
    }

    LOG_DEBUG("Set learning rate to %.6f for all %d parameter groups", lr,
              optimizer->num_param_groups);
}

void optimizer_set_group_lr(Optimizer* optimizer, int group_index, float lr) {
    if (!optimizer || group_index < 0 || group_index >= optimizer->num_param_groups) {
        LOG_WARNING("Invalid group index %d for optimizer with %d groups", group_index,
                    optimizer ? optimizer->num_param_groups : 0);
        return;
    }

    optimizer->param_groups[group_index].lr = lr;

    LOG_DEBUG("Set learning rate to %.6f for parameter group %d", lr, group_index);
}

float optimizer_get_group_lr(Optimizer* optimizer, int group_index) {
    if (!optimizer || group_index < 0 || group_index >= optimizer->num_param_groups) {
        LOG_WARNING("Invalid group index %d for optimizer with %d groups", group_index,
                    optimizer ? optimizer->num_param_groups : 0);
        return 0.0f;
    }

    return optimizer->param_groups[group_index].lr;
}

void optimizer_set_lr_scheduler(Optimizer* optimizer, int step_size, float gamma) {
    if (!optimizer)
        return;

    optimizer->lr_scheduler_step_size = step_size;
    optimizer->lr_scheduler_gamma     = gamma;
}

void optimizer_set_amp(Optimizer* optimizer, bool use_amp) {
    if (optimizer) {
        optimizer->use_amp = use_amp;
    }
}

void optimizer_set_grad_clip_norm(Optimizer* optimizer, float norm) {
    if (optimizer) {
        optimizer->grad_clip_norm = norm;
    }
}

void optimizer_set_amsgrad(Optimizer* optimizer, bool amsgrad) {
    if (optimizer) {
        optimizer->amsgrad = amsgrad;
    }
}

const char* optimizer_get_name(Optimizer* optimizer) { return optimizer ? optimizer->name : NULL; }

int optimizer_get_total_parameters(Optimizer* optimizer) {
    if (!optimizer)
        return 0;

    int total = 0;
    for (int i = 0; i < optimizer->num_param_groups; i++) {
        total += optimizer->param_groups[i].num_parameters;
    }

    return total;
}

void optimizer_print_summary(Optimizer* optimizer, int indent) {
    if (!optimizer)
        return;

    for (int i = 0; i < indent; i++)
        printf("  ");
    printf("Optimizer: %s (Parameter Groups: %d, Total Parameters: %d)\n", optimizer->name,
           optimizer->num_param_groups, optimizer_get_total_parameters(optimizer));

    // Print each parameter group
    for (int i = 0; i < optimizer->num_param_groups; i++) {
        ParameterGroup* group = &optimizer->param_groups[i];
        for (int j = 0; j < indent + 1; j++)
            printf("  ");
        printf("Group %d: %d parameters, lr=%.6f, wd=%.6f\n", i, group->num_parameters, group->lr,
               group->weight_decay);
    }
}

// Utility Functions

bool optimizer_supports_lr_scheduling(Optimizer* optimizer) {
    if (!optimizer)
        return false;
    return optimizer->lr_scheduler_step_size > 0;
}

bool optimizer_supports_grad_clipping(Optimizer* optimizer) {
    if (!optimizer)
        return false;
    return optimizer->grad_clip_norm > 0.0f;
}

// SGD Step Function
static void sgd_step(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g = 0; g < optimizer->num_param_groups; g++) {
        ParameterGroup* group = &optimizer->param_groups[g];
        float lr              = group->lr;
        float weight_decay    = group->weight_decay;
        float momentum        = group->momentum;

        // Initialize momentum state if needed
        if (momentum > 0.0f && !group->state) {
            // Allocate state array for all parameters
            SGDMomentumState** states =
                CM_MALLOC(group->num_parameters * sizeof(SGDMomentumState*));
            if (!states) {
                LOG_ERROR("Failed to allocate SGD momentum state");
                continue;
            }

            for (int i = 0; i < group->num_parameters; i++) {
                Parameter* param = group->parameters[i];
                if (!param || !param->tensor)
                    continue;

                Tensor* tensor          = param->tensor;
                SGDMomentumState* state = CM_MALLOC(sizeof(SGDMomentumState));
                if (!state)
                    continue;

                // Initialize momentum buffer to zeros
                state->momentum_buffer =
                    tensor_zeros(tensor->shape, tensor->ndim, tensor->dtype, tensor->device);
                states[i] = state;
            }

            group->state = states;
        }

        SGDMomentumState** states = momentum > 0.0f ? (SGDMomentumState**)group->state : NULL;

        for (int i = 0; i < group->num_parameters; i++) {
            Parameter* param = group->parameters[i];
            if (!param || !param->tensor || !param->requires_grad)
                continue;

            Tensor* tensor = param->tensor;
            Tensor* grad   = tensor_get_grad(tensor);

            if (!grad)
                continue;

            float* param_data = (float*)tensor_data_ptr(tensor);
            float* grad_data  = (float*)tensor_data_ptr(grad);

            if (!param_data || !grad_data)
                continue;

            size_t numel = tensor->numel;

            // Weight decay (L2 regularization)
            if (weight_decay > 0.0f) {
                for (size_t j = 0; j < numel; j++) {
                    grad_data[j] += weight_decay * param_data[j];
                }
            }

            // Momentum update
            if (momentum > 0.0f && states && states[i] && states[i]->momentum_buffer) {
                float* momentum_data = (float*)tensor_data_ptr(states[i]->momentum_buffer);
                if (momentum_data) {
                    // Update momentum buffer: v = momentum * v - lr * grad
                    for (size_t j = 0; j < numel; j++) {
                        momentum_data[j] = momentum * momentum_data[j] - lr * grad_data[j];
                    }
                    // Update parameter: param = param + v
                    for (size_t j = 0; j < numel; j++) {
                        param_data[j] += momentum_data[j];
                    }
                } else {
                    // Fallback to simple SGD if momentum buffer allocation failed
                    for (size_t j = 0; j < numel; j++) {
                        param_data[j] -= lr * grad_data[j];
                    }
                }
            } else {
                // Simple SGD without momentum: param = param - lr * grad
                for (size_t j = 0; j < numel; j++) {
                    param_data[j] -= lr * grad_data[j];
                }
            }
        }

        group->step_count++;
    }
}

// SGD Zero Grad Function
static void sgd_zero_grad(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g = 0; g < optimizer->num_param_groups; g++) {
        ParameterGroup* group = &optimizer->param_groups[g];

        for (int i = 0; i < group->num_parameters; i++) {
            Parameter* param = group->parameters[i];
            if (!param || !param->tensor)
                continue;

            tensor_zero_grad(param->tensor);
        }
    }
}

// Adam Step Function
static void adam_step(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g = 0; g < optimizer->num_param_groups; g++) {
        ParameterGroup* group = &optimizer->param_groups[g];
        float lr              = group->lr;
        float weight_decay    = group->weight_decay;
        float beta1           = group->beta1;
        float beta2           = group->beta2;
        float epsilon         = group->epsilon;

        // Initialize state if needed
        if (!group->state) {
            // Allocate state array for all parameters
            AdamState** states = CM_MALLOC(group->num_parameters * sizeof(AdamState*));
            if (!states) {
                LOG_ERROR("Failed to allocate Adam state");
                continue;
            }

            for (int i = 0; i < group->num_parameters; i++) {
                Parameter* param = group->parameters[i];
                if (!param || !param->tensor)
                    continue;

                Tensor* tensor   = param->tensor;
                AdamState* state = CM_MALLOC(sizeof(AdamState));
                if (!state)
                    continue;

                // Initialize moment estimates
                state->exp_avg =
                    tensor_zeros(tensor->shape, tensor->ndim, tensor->dtype, tensor->device);
                state->exp_avg_sq =
                    tensor_zeros(tensor->shape, tensor->ndim, tensor->dtype, tensor->device);
                state->max_exp_avg_sq =
                    optimizer->amsgrad
                        ? tensor_zeros(tensor->shape, tensor->ndim, tensor->dtype, tensor->device)
                        : NULL;

                states[i] = state;
            }

            group->state = states;
        }

        AdamState** states = (AdamState**)group->state;
        int step           = group->step_count + 1;

        // Bias correction factors
        float bias_correction1 = 1.0f - powf(beta1, step);
        float bias_correction2 = 1.0f - powf(beta2, step);

        for (int i = 0; i < group->num_parameters; i++) {
            Parameter* param = group->parameters[i];
            if (!param || !param->tensor || !param->requires_grad)
                continue;

            Tensor* tensor = param->tensor;
            Tensor* grad   = tensor_get_grad(tensor);

            if (!grad || !states[i])
                continue;

            float* param_data      = (float*)tensor_data_ptr(tensor);
            float* grad_data       = (float*)tensor_data_ptr(grad);
            float* exp_avg_data    = (float*)tensor_data_ptr(states[i]->exp_avg);
            float* exp_avg_sq_data = (float*)tensor_data_ptr(states[i]->exp_avg_sq);

            if (!param_data || !grad_data || !exp_avg_data || !exp_avg_sq_data)
                continue;

            size_t numel = tensor->numel;

            // Update moment estimates
            for (size_t j = 0; j < numel; j++) {
                float g = grad_data[j];

                // Weight decay
                if (weight_decay > 0.0f) {
                    g += weight_decay * param_data[j];
                }

                // Update biased first moment estimate
                exp_avg_data[j] = beta1 * exp_avg_data[j] + (1.0f - beta1) * g;

                // Update biased second raw moment estimate
                exp_avg_sq_data[j] = beta2 * exp_avg_sq_data[j] + (1.0f - beta2) * g * g;

                // Compute bias-corrected estimates
                float m = exp_avg_data[j] / bias_correction1;
                float v = exp_avg_sq_data[j] / bias_correction2;

                // AMSGrad variant
                if (optimizer->amsgrad && states[i]->max_exp_avg_sq) {
                    float* max_exp_avg_sq_data = (float*)tensor_data_ptr(states[i]->max_exp_avg_sq);
                    if (max_exp_avg_sq_data) {
                        max_exp_avg_sq_data[j] = fmaxf(max_exp_avg_sq_data[j], v);
                        v                      = max_exp_avg_sq_data[j];
                    }
                }

                // Update parameter: param = param - lr * m / (sqrt(v) + epsilon)
                param_data[j] -= lr * m / (sqrtf(v) + epsilon);
            }
        }

        group->step_count++;
    }
}

// Adam Zero Grad Function
static void adam_zero_grad(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g = 0; g < optimizer->num_param_groups; g++) {
        ParameterGroup* group = &optimizer->param_groups[g];

        for (int i = 0; i < group->num_parameters; i++) {
            Parameter* param = group->parameters[i];
            if (!param || !param->tensor)
                continue;

            tensor_zero_grad(param->tensor);
        }
    }
}

// RMSprop Step Function
static void rmsprop_step(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g = 0; g < optimizer->num_param_groups; g++) {
        ParameterGroup* group = &optimizer->param_groups[g];
        float lr              = group->lr;
        float weight_decay    = group->weight_decay;
        float alpha           = group->beta1; // Reuse beta1 for alpha (decay rate)
        float epsilon         = group->epsilon;

        // Initialize state if needed
        if (!group->state) {
            // Allocate state array for all parameters
            RMSpropState** states = CM_MALLOC(group->num_parameters * sizeof(RMSpropState*));
            if (!states) {
                LOG_ERROR("Failed to allocate RMSprop state");
                continue;
            }

            for (int i = 0; i < group->num_parameters; i++) {
                Parameter* param = group->parameters[i];
                if (!param || !param->tensor)
                    continue;

                Tensor* tensor      = param->tensor;
                RMSpropState* state = CM_MALLOC(sizeof(RMSpropState));
                if (!state)
                    continue;

                // Initialize square average to zeros
                state->square_avg =
                    tensor_zeros(tensor->shape, tensor->ndim, tensor->dtype, tensor->device);
                states[i] = state;
            }

            group->state = states;
        }

        RMSpropState** states = (RMSpropState**)group->state;

        for (int i = 0; i < group->num_parameters; i++) {
            Parameter* param = group->parameters[i];
            if (!param || !param->tensor || !param->requires_grad)
                continue;

            Tensor* tensor = param->tensor;
            Tensor* grad   = tensor_get_grad(tensor);

            if (!grad || !states[i])
                continue;

            float* param_data      = (float*)tensor_data_ptr(tensor);
            float* grad_data       = (float*)tensor_data_ptr(grad);
            float* square_avg_data = (float*)tensor_data_ptr(states[i]->square_avg);

            if (!param_data || !grad_data || !square_avg_data)
                continue;

            size_t numel = tensor->numel;

            // Update square average: square_avg = alpha * square_avg + (1 - alpha) * grad^2
            for (size_t j = 0; j < numel; j++) {
                float g = grad_data[j];

                // Weight decay
                if (weight_decay > 0.0f) {
                    g += weight_decay * param_data[j];
                }

                // Update square average
                square_avg_data[j] = alpha * square_avg_data[j] + (1.0f - alpha) * g * g;

                // Update parameter: param = param - lr * grad / sqrt(square_avg + epsilon)
                param_data[j] -= lr * g / (sqrtf(square_avg_data[j]) + epsilon);
            }
        }

        group->step_count++;
    }
}

// RMSprop Zero Grad Function
static void rmsprop_zero_grad(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g = 0; g < optimizer->num_param_groups; g++) {
        ParameterGroup* group = &optimizer->param_groups[g];

        for (int i = 0; i < group->num_parameters; i++) {
            Parameter* param = group->parameters[i];
            if (!param || !param->tensor)
                continue;

            tensor_zero_grad(param->tensor);
        }
    }
}

// Adagrad Step Function
static void adagrad_step(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g = 0; g < optimizer->num_param_groups; g++) {
        ParameterGroup* group = &optimizer->param_groups[g];
        float lr              = group->lr;
        float weight_decay    = group->weight_decay;
        float epsilon         = group->epsilon;

        // Initialize state if needed
        if (!group->state) {
            // Allocate state array for all parameters
            AdagradState** states = CM_MALLOC(group->num_parameters * sizeof(AdagradState*));
            if (!states) {
                LOG_ERROR("Failed to allocate Adagrad state");
                continue;
            }

            for (int i = 0; i < group->num_parameters; i++) {
                Parameter* param = group->parameters[i];
                if (!param || !param->tensor)
                    continue;

                Tensor* tensor      = param->tensor;
                AdagradState* state = CM_MALLOC(sizeof(AdagradState));
                if (!state)
                    continue;

                // Initialize sum of squared gradients to zeros
                state->sum_sq_grad =
                    tensor_zeros(tensor->shape, tensor->ndim, tensor->dtype, tensor->device);
                states[i] = state;
            }

            group->state = states;
        }

        AdagradState** states = (AdagradState**)group->state;

        for (int i = 0; i < group->num_parameters; i++) {
            Parameter* param = group->parameters[i];
            if (!param || !param->tensor || !param->requires_grad)
                continue;

            Tensor* tensor = param->tensor;
            Tensor* grad   = tensor_get_grad(tensor);

            if (!grad || !states[i])
                continue;

            float* param_data       = (float*)tensor_data_ptr(tensor);
            float* grad_data        = (float*)tensor_data_ptr(grad);
            float* sum_sq_grad_data = (float*)tensor_data_ptr(states[i]->sum_sq_grad);

            if (!param_data || !grad_data || !sum_sq_grad_data)
                continue;

            size_t numel = tensor->numel;

            // Update sum of squared gradients and parameters
            for (size_t j = 0; j < numel; j++) {
                float g = grad_data[j];

                // Weight decay
                if (weight_decay > 0.0f) {
                    g += weight_decay * param_data[j];
                }

                // Accumulate squared gradients: sum_sq_grad += grad^2
                sum_sq_grad_data[j] += g * g;

                // Update parameter: param = param - lr * grad / sqrt(sum_sq_grad + epsilon)
                param_data[j] -= lr * g / (sqrtf(sum_sq_grad_data[j]) + epsilon);
            }
        }

        group->step_count++;
    }
}

// Adagrad Zero Grad Function
static void adagrad_zero_grad(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g = 0; g < optimizer->num_param_groups; g++) {
        ParameterGroup* group = &optimizer->param_groups[g];

        for (int i = 0; i < group->num_parameters; i++) {
            Parameter* param = group->parameters[i];
            if (!param || !param->tensor)
                continue;

            tensor_zero_grad(param->tensor);
        }
    }
}

// Optimizer Creation Functions

/**
 * @brief Create SGD optimizer
 */
Optimizer* optim_sgd(Parameter** parameters, int num_parameters, float lr, float momentum,
                     float weight_decay) {
    Optimizer* optimizer = optimizer_create("SGD", sgd_step, sgd_zero_grad);
    if (!optimizer)
        return NULL;

    if (optimizer_add_param_group(optimizer, parameters, num_parameters, lr, weight_decay) != 0) {
        optimizer_free(optimizer);
        return NULL;
    }

    // Set momentum
    if (optimizer->num_param_groups > 0) {
        optimizer->param_groups[0].momentum = momentum;
    }

    return optimizer;
}

/**
 * @brief Create Adam optimizer
 */
Optimizer* optim_adam(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                      float beta1, float beta2, float epsilon) {
    Optimizer* optimizer = optimizer_create("Adam", adam_step, adam_zero_grad);
    if (!optimizer)
        return NULL;

    if (optimizer_add_param_group(optimizer, parameters, num_parameters, lr, weight_decay) != 0) {
        optimizer_free(optimizer);
        return NULL;
    }

    // Set Adam hyperparameters
    if (optimizer->num_param_groups > 0) {
        ParameterGroup* group = &optimizer->param_groups[0];
        group->beta1          = beta1 > 0.0f ? beta1 : 0.9f;
        group->beta2          = beta2 > 0.0f ? beta2 : 0.999f;
        group->epsilon        = epsilon > 0.0f ? epsilon : 1e-8f;
    }

    return optimizer;
}

/**
 * @brief Create RMSprop optimizer
 */
Optimizer* optim_rmsprop(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                         float alpha, float epsilon) {
    Optimizer* optimizer = optimizer_create("RMSprop", rmsprop_step, rmsprop_zero_grad);
    if (!optimizer)
        return NULL;

    if (optimizer_add_param_group(optimizer, parameters, num_parameters, lr, weight_decay) != 0) {
        optimizer_free(optimizer);
        return NULL;
    }

    // Set RMSprop hyperparameters
    if (optimizer->num_param_groups > 0) {
        ParameterGroup* group = &optimizer->param_groups[0];
        group->beta1          = alpha > 0.0f ? alpha : 0.99f; // Reuse beta1 for alpha
        group->epsilon        = epsilon > 0.0f ? epsilon : 1e-8f;
    }

    return optimizer;
}

/**
 * @brief Create Adagrad optimizer
 */
Optimizer* optim_adagrad(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                         float epsilon) {
    Optimizer* optimizer = optimizer_create("Adagrad", adagrad_step, adagrad_zero_grad);
    if (!optimizer)
        return NULL;

    if (optimizer_add_param_group(optimizer, parameters, num_parameters, lr, weight_decay) != 0) {
        optimizer_free(optimizer);
        return NULL;
    }

    // Set Adagrad hyperparameters
    if (optimizer->num_param_groups > 0) {
        ParameterGroup* group = &optimizer->param_groups[0];
        group->epsilon        = epsilon > 0.0f ? epsilon : 1e-8f;
    }

    return optimizer;
}
