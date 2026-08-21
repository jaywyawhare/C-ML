#ifndef CML_NN_MODULE_H
#define CML_NN_MODULE_H

#include "tensor/tensor.h"
#include "core/logging.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Module;
struct Parameter;

typedef Tensor* (*ForwardFn)(struct Module* module, Tensor* input);
typedef void (*FreeFn)(struct Module* module);

typedef struct Parameter {
    Tensor* tensor;     // The parameter tensor
    bool requires_grad; // Whether to compute gradients
    char* name;         // Parameter name for identification (owned)
} Parameter;

typedef struct Module {
    char* name;        // Module name for identification (owned)
    ForwardFn forward; // Forward pass function
    FreeFn free;       // Cleanup function

    Parameter** parameters;  // Array of trainable parameters
    int num_parameters;      // Number of parameters
    int parameters_capacity; // Capacity of parameters array

    struct Module* next; // Next module in sequence (for containers)

    bool training;   // Training mode flag
    void* user_data; // User-defined data
    void* backward_hooks; /* ModuleHookList*, owned by autograd */

    const char* version;     // Module version
    const char* description; // Module description
} Module;

int module_init(Module* module, const char* name, ForwardFn forward, FreeFn free);

Module* module_create(const char* name, ForwardFn forward, FreeFn free);

void module_free(Module* module);

int module_add_parameter(Module* module, Tensor* tensor, const char* name, bool requires_grad);

/* Create a zero-initialised 1-D "bias" parameter of length `size`, optionally
 * post-initialised by `init`, register it on `module`, and return the stored
 * Parameter. On any failure, frees `module` and returns NULL (so callers can
 * `return NULL` directly). Shared by the linear/conv layer constructors. */
Parameter* nn_add_bias_param(Module* module, int size, DType dtype, DeviceType device,
                             void (*init)(Tensor*, int));

/* Register an already-created and initialised `weight` on `module` and return
 * the stored Parameter. On failure frees both and returns NULL, so callers can
 * `return NULL` directly. Mirrors nn_add_bias_param. */
Parameter* nn_add_weight_param(Module* module, Tensor* weight);

/* Create a normalisation layer's affine pair — gamma "weight" (ones) and beta
 * "bias" (zeros), both length `size`. Stores them into *weight_out/*bias_out.
 * On failure, frees `module` and returns -1; returns 0 on success. */
int nn_add_affine_params(Module* module, int size, DType dtype, DeviceType device,
                         Parameter** weight_out, Parameter** bias_out);

/* Create a normalisation layer's running statistics buffers — running_mean
 * (zeros) and running_var (ones), both length `size`. On failure frees `module`
 * and returns -1; returns 0 on success. */
int nn_add_running_stats(Module* module, int size, DType dtype, DeviceType device,
                         Tensor** mean_out, Tensor** var_out);

/* Shared state of the BatchNorm{1,2,3}d family. The three public layer types
 * are aliases of this struct, so the whole family runs one forward pass that
 * only differs in the input rank it accepts. */
typedef struct BatchNormState {
    Module base;

    int num_features;
    int input_ndim; /* rank this layer accepts: 2 (1d), 4 (2d), 5 (3d) */
    float eps;
    float momentum;
    bool affine;
    bool track_running_stats;

    Parameter* weight;
    Parameter* bias;

    Tensor* running_mean;
    Tensor* running_var;

    Tensor* current_mean;
    Tensor* current_var;
} BatchNormState;

/* Allocate a BatchNorm layer named `name` that accepts rank-`input_ndim` input
 * with channels on dimension 1. Returns NULL on failure, having freed whatever
 * it had built. */
BatchNormState* nn_batchnorm_new(const char* name, int input_ndim, int num_features, float eps,
                                 float momentum, bool affine, bool track_running_stats,
                                 DType dtype, DeviceType device);

/* Apply a normalisation layer's affine pair to `x`, broadcasting the 1-D gamma
 * and beta along `channel_dim` of `shape`/`ndim`. Returns `x` untouched when
 * either parameter is absent, NULL on failure. */
Tensor* nn_norm_affine(Tensor* x, const Parameter* weight, const Parameter* bias,
                       const int* shape, int ndim, int channel_dim);

/* Standardise `x` over its innermost axis, viewing it as [rows, cols]:
 * (x - mean) / sqrt(var + eps). The result keeps the [rows, cols] shape.
 * Shared by InstanceNorm2d (rows = N*C) and LayerNorm2d (rows = N). */
Tensor* nn_norm_rowwise(Tensor* x, int rows, int cols, float eps);

void nn_tensor_param_alias(Tensor* t);

/* Outermost module of the most recent forward pass, i.e. the model. Recorded
 * only under VIZ; NULL otherwise. Lets the dashboard summarise parameters
 * without every training loop having to register its model. */
void nn_viz_set_root_module(Module* module);
Module* nn_viz_root_module(void);

int module_get_parameters(Module* module, Parameter** params, int* num_parameters);

Parameter* module_get_parameter(Module* module, const char* name);

int module_set_parameter(Module* module, const char* name, Tensor* tensor);

Tensor* module_forward(Module* module, Tensor* input);

void module_set_training(Module* module, bool training);

bool module_is_training(Module* module);

void module_zero_grad(Module* module);

const char* module_get_name(Module* module);

int module_get_parameter_count(Module* module);

void module_print_summary(Module* module, int indent);

int module_get_total_parameters(Module* module);

int module_collect_parameters(Module* module, Parameter*** params_out, int* num_params_out,
                              bool recursive);

int module_chain(Module* first, Module* second);

Module* module_get_next(Module* module);

void module_set_next(Module* module, Module* next);

int module_to_device(Module* module, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_MODULE_H
