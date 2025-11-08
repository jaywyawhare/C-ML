/**
 * @file optimizer.h
 * @brief Base optimizer class for parameter optimization
 *
 * This header defines the base Optimizer class that all optimization
 * algorithms inherit from. It provides a unified interface for
 * parameter updates, learning rate management, and optimizer state.
 */

#ifndef CML_OPTIM_OPTIMIZER_H
#define CML_OPTIM_OPTIMIZER_H

#include "nn/module.h"
#include "Core/logging.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct Optimizer;
struct ParameterGroup;

// Optimizer function signatures
typedef void (*StepFn)(struct Optimizer* optimizer);
typedef void (*ZeroGradFn)(struct Optimizer* optimizer);

/**
 * @brief Parameter group configuration
 *
 * Groups parameters with the same optimization settings
 * (learning rate, weight decay, etc.)
 */
typedef struct ParameterGroup {
    Parameter** parameters; // Array of parameters to optimize
    int num_parameters;     // Number of parameters

    // Optimization hyperparameters
    float lr;           // Learning rate
    float weight_decay; // Weight decay (L2 regularization)
    float momentum;     // Momentum factor (for SGD)

    // Adam-specific parameters
    float beta1;   // First moment decay rate
    float beta2;   // Second moment decay rate
    float epsilon; // Numerical stability constant

    // Internal state
    void* state;    // Optimizer-specific state
    int step_count; // Step counter
} ParameterGroup;

/**
 * @brief Base optimizer structure
 *
 * All optimization algorithms inherit from this base structure.
 * It provides:
 * - Parameter group management
 * - Learning rate scheduling
 * - Gradient accumulation
 * - Optimizer state management
 */
typedef struct Optimizer {
    const char* name;     // Optimizer name
    StepFn step;          // Step function
    ZeroGradFn zero_grad; // Zero gradients function

    ParameterGroup* param_groups; // Array of parameter groups
    int num_param_groups;         // Number of parameter groups
    int param_groups_capacity;    // Capacity of param_groups array

    // Optimizer configuration
    bool use_amp;         // Automatic mixed precision
    float grad_clip_norm; // Gradient clipping norm
    bool amsgrad;         // AMSGrad variant (for Adam)

    // Learning rate scheduling
    float lr_scheduler_factor;  // Learning rate multiplier
    int lr_scheduler_step_size; // Step size for LR scheduling
    float lr_scheduler_gamma;   // LR decay factor

    // Optimizer metadata
    const char* version;     // Optimizer version
    const char* description; // Optimizer description

    void* training_metrics; // TrainingMetrics* (void* to avoid circular dependency)
} Optimizer;

// Optimizer Creation and Management

/**
 * @brief Initialize an optimizer with basic settings
 *
 * @param optimizer Optimizer to initialize
 * @param name Optimizer name
 * @param step Step function
 * @param zero_grad Zero gradients function
 * @return 0 on success, negative value on failure
 */
int optimizer_init(Optimizer* optimizer, const char* name, StepFn step, ZeroGradFn zero_grad);

/**
 * @brief Create a new optimizer
 *
 * @param name Optimizer name
 * @param step Step function
 * @param zero_grad Zero gradients function
 * @return New optimizer, or NULL on failure
 */
Optimizer* optimizer_create(const char* name, StepFn step, ZeroGradFn zero_grad);

/**
 * @brief Free an optimizer and all its resources
 *
 * @param optimizer Optimizer to free
 */
void optimizer_free(Optimizer* optimizer);

// Parameter Group Management

/**
 * @brief Add a parameter group to an optimizer
 *
 * @param optimizer Target optimizer
 * @param parameters Array of parameters
 * @param num_parameters Number of parameters
 * @param lr Learning rate for this group
 * @param weight_decay Weight decay for this group
 * @return 0 on success, negative value on failure
 */
int optimizer_add_param_group(Optimizer* optimizer, Parameter** parameters, int num_parameters,
                              float lr, float weight_decay);

/**
 * @brief Get optimizer parameter groups
 *
 * @param optimizer Target optimizer
 * @param groups Array to store parameter groups (can be NULL to just get count)
 * @param num_groups Pointer to store group count
 * @return 0 on success, negative value on failure
 */
int optimizer_get_param_groups(Optimizer* optimizer, ParameterGroup** groups, int* num_groups);

/**
 * @brief Get parameter group by index
 *
 * @param optimizer Target optimizer
 * @param index Group index
 * @return Parameter group if found, NULL otherwise
 */
ParameterGroup* optimizer_get_param_group(Optimizer* optimizer, int index);

// Optimizer Operations

/**
 * @brief Perform optimization step
 *
 * @param optimizer Target optimizer
 */
void optimizer_step(Optimizer* optimizer);

/**
 * @brief Set training metrics for automatic recording
 * @param optimizer Optimizer instance
 * @param metrics Training metrics structure (can be NULL to disable)
 */
void optimizer_set_metrics(Optimizer* optimizer, void* metrics);

/**
 * @brief Zero all parameter gradients
 *
 * @param optimizer Target optimizer
 */
void optimizer_zero_grad(Optimizer* optimizer);

/**
 * @brief Get optimizer step count
 *
 * @param optimizer Target optimizer
 * @return Current step count
 */
int optimizer_get_step_count(Optimizer* optimizer);

// Learning Rate Management

/**
 * @brief Set learning rate for all parameter groups
 *
 * @param optimizer Target optimizer
 * @param lr New learning rate
 */
void optimizer_set_lr(Optimizer* optimizer, float lr);

/**
 * @brief Set learning rate for specific parameter group
 *
 * @param optimizer Target optimizer
 * @param group_index Parameter group index
 * @param lr New learning rate
 */
void optimizer_set_group_lr(Optimizer* optimizer, int group_index, float lr);

/**
 * @brief Get learning rate for specific parameter group
 *
 * @param optimizer Target optimizer
 * @param group_index Parameter group index
 * @return Learning rate
 */
float optimizer_get_group_lr(Optimizer* optimizer, int group_index);

/**
 * @brief Set learning rate scheduler
 *
 * @param optimizer Target optimizer
 * @param step_size Step size for LR scheduling
 * @param gamma LR decay factor
 */
void optimizer_set_lr_scheduler(Optimizer* optimizer, int step_size, float gamma);

// Optimizer Configuration

/**
 * @brief Enable or disable automatic mixed precision
 *
 * @param optimizer Target optimizer
 * @param use_amp Whether to use AMP
 */
void optimizer_set_amp(Optimizer* optimizer, bool use_amp);

/**
 * @brief Set gradient clipping norm
 *
 * @param optimizer Target optimizer
 * @param norm Gradient clipping norm
 */
void optimizer_set_grad_clip_norm(Optimizer* optimizer, float norm);

/**
 * @brief Enable or disable AMSGrad variant
 *
 * @param optimizer Target optimizer
 * @param amsgrad Whether to use AMSGrad
 */
void optimizer_set_amsgrad(Optimizer* optimizer, bool amsgrad);

// Optimizer Information

/**
 * @brief Get optimizer name
 *
 * @param optimizer Target optimizer
 * @return Optimizer name
 */
const char* optimizer_get_name(Optimizer* optimizer);

/**
 * @brief Get total parameter count across all groups
 *
 * @param optimizer Target optimizer
 * @return Total number of parameters
 */
int optimizer_get_total_parameters(Optimizer* optimizer);

/**
 * @brief Print optimizer summary
 *
 * @param optimizer Target optimizer
 * @param indent Indentation level
 */
void optimizer_print_summary(Optimizer* optimizer, int indent);

// Utility Functions

/**
 * @brief Check if optimizer supports learning rate scheduling
 *
 * @param optimizer Target optimizer
 * @return true if LR scheduling is supported, false otherwise
 */
bool optimizer_supports_lr_scheduling(Optimizer* optimizer);

/**
 * @brief Check if optimizer supports gradient clipping
 *
 * @param optimizer Target optimizer
 * @return true if gradient clipping is supported, false otherwise
 */
bool optimizer_supports_grad_clipping(Optimizer* optimizer);

// Optimizer Creation Functions

/**
 * @brief Create SGD optimizer
 *
 * @param parameters Array of parameters to optimize
 * @param num_parameters Number of parameters
 * @param lr Learning rate
 * @param momentum Momentum factor (0.0 = no momentum)
 * @param weight_decay Weight decay (L2 regularization)
 * @return New SGD optimizer, or NULL on failure
 */
Optimizer* optim_sgd(Parameter** parameters, int num_parameters, float lr, float momentum,
                     float weight_decay);

/**
 * @brief Create Adam optimizer
 *
 * @param parameters Array of parameters to optimize
 * @param num_parameters Number of parameters
 * @param lr Learning rate
 * @param weight_decay Weight decay (L2 regularization)
 * @param beta1 First moment decay rate (default: 0.9)
 * @param beta2 Second moment decay rate (default: 0.999)
 * @param epsilon Numerical stability constant (default: 1e-8)
 * @return New Adam optimizer, or NULL on failure
 */
Optimizer* optim_adam(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                      float beta1, float beta2, float epsilon);

/**
 * @brief Create RMSprop optimizer
 *
 * @param parameters Array of parameters to optimize
 * @param num_parameters Number of parameters
 * @param lr Learning rate
 * @param weight_decay Weight decay (L2 regularization)
 * @param alpha Smoothing constant (default: 0.99)
 * @param epsilon Numerical stability constant (default: 1e-8)
 * @return New RMSprop optimizer, or NULL on failure
 */
Optimizer* optim_rmsprop(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                         float alpha, float epsilon);

/**
 * @brief Create Adagrad optimizer
 *
 * @param parameters Array of parameters to optimize
 * @param num_parameters Number of parameters
 * @param lr Learning rate
 * @param weight_decay Weight decay (L2 regularization)
 * @param epsilon Numerical stability constant (default: 1e-8)
 * @return New Adagrad optimizer, or NULL on failure
 */
Optimizer* optim_adagrad(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                         float epsilon);

#ifdef __cplusplus
}
#endif

#endif // CML_OPTIM_OPTIMIZER_H
