#ifndef CML_OPTIM_OPTIMIZER_H
#define CML_OPTIM_OPTIMIZER_H

#include "nn.h"
#include "core/logging.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Optimizer;
struct ParameterGroup;

typedef void (*StepFn)(struct Optimizer* optimizer);
typedef void (*ZeroGradFn)(struct Optimizer* optimizer);

typedef struct ParameterGroup {
    Parameter** parameters; // Array of parameters to optimize
    int num_parameters;     // Number of parameters

    float lr;           // Learning rate
    float weight_decay; // Weight decay (L2 regularization)
    float momentum;     // Momentum factor (for SGD)

    float beta1;   // First moment decay rate
    float beta2;   // Second moment decay rate
    float epsilon; // Numerical stability constant

    void* state;    // Optimizer-specific state
    int step_count; // Step counter
} ParameterGroup;

typedef struct Optimizer {
    const char* name;     // Optimizer name
    StepFn step;          // Step function
    ZeroGradFn zero_grad; // Zero gradients function

    ParameterGroup* param_groups; // Array of parameter groups
    int num_param_groups;         // Number of parameter groups
    int param_groups_capacity;    // Capacity of param_groups array

    bool use_amp;         // Automatic mixed precision
    float grad_clip_norm; // Gradient clipping norm
    bool amsgrad;         // AMSGrad variant (for Adam)

    float lr_scheduler_factor;  // Learning rate multiplier
    int lr_scheduler_step_size; // Step size for LR scheduling
    float lr_scheduler_gamma;   // LR decay factor

    const char* version;     // Optimizer version
    const char* description; // Optimizer description

    void* training_metrics; // TrainingMetrics* (void* to avoid circular dependency)
} Optimizer;

int optimizer_init(Optimizer* optimizer, const char* name, StepFn step, ZeroGradFn zero_grad);

Optimizer* optimizer_create(const char* name, StepFn step, ZeroGradFn zero_grad);

void optimizer_free(Optimizer* optimizer);

int optimizer_add_param_group(Optimizer* optimizer, Parameter** parameters, int num_parameters,
                              float lr, float weight_decay);

int optimizer_get_param_groups(Optimizer* optimizer, ParameterGroup** groups, int* num_groups);

ParameterGroup* optimizer_get_param_group(Optimizer* optimizer, int index);

void optimizer_step(Optimizer* optimizer);

void optimizer_set_metrics(Optimizer* optimizer, void* metrics);

void optimizer_zero_grad(Optimizer* optimizer);

int optimizer_get_step_count(Optimizer* optimizer);

void optimizer_set_lr(Optimizer* optimizer, float lr);

void optimizer_set_group_lr(Optimizer* optimizer, int group_index, float lr);

float optimizer_get_group_lr(Optimizer* optimizer, int group_index);

void optimizer_set_lr_scheduler(Optimizer* optimizer, int step_size, float gamma);

void optimizer_set_amp(Optimizer* optimizer, bool use_amp);

void optimizer_set_grad_clip_norm(Optimizer* optimizer, float norm);

void optimizer_set_amsgrad(Optimizer* optimizer, bool amsgrad);

const char* optimizer_get_name(Optimizer* optimizer);

int optimizer_get_total_parameters(Optimizer* optimizer);

void optimizer_print_summary(Optimizer* optimizer, int indent);

bool optimizer_supports_lr_scheduling(Optimizer* optimizer);

bool optimizer_supports_grad_clipping(Optimizer* optimizer);

Optimizer* optim_sgd(Parameter** parameters, int num_parameters, float lr, float momentum,
                     float weight_decay);

Optimizer* optim_adam(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                      float beta1, float beta2, float epsilon);

Optimizer* optim_rmsprop(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                         float alpha, float epsilon);

Optimizer* optim_adagrad(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                         float epsilon);

Optimizer* optim_adamw(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                       float beta1, float beta2, float epsilon);

Optimizer* optim_adadelta(Parameter** parameters, int num_parameters, float rho, float weight_decay,
                          float epsilon);

Optimizer* optim_adam_for_model(Module* model, float lr, float weight_decay, float beta1,
                                float beta2, float eps);

Optimizer* optim_sgd_for_model(Module* model, float lr, float momentum, float weight_decay);

Optimizer* optim_lamb(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                      float beta1, float beta2, float epsilon);

Optimizer* optim_lars(Parameter** parameters, int num_parameters, float lr, float momentum,
                      float weight_decay, float trust_coefficient);

Optimizer* optim_muon(Parameter** parameters, int num_parameters, float lr, float momentum,
                      float weight_decay, bool nesterov);

Optimizer* optim_nadam(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                       float beta1, float beta2, float epsilon);

Optimizer* optim_adamax(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                        float beta1, float beta2, float epsilon);

#ifdef __cplusplus
}
#endif

#endif // CML_OPTIM_OPTIMIZER_H
