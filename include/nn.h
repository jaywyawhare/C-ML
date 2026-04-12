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

    const char* version;     // Module version
    const char* description; // Module description
} Module;

int module_init(Module* module, const char* name, ForwardFn forward, FreeFn free);

Module* module_create(const char* name, ForwardFn forward, FreeFn free);

void module_free(Module* module);

int module_add_parameter(Module* module, Tensor* tensor, const char* name, bool requires_grad);

void nn_tensor_param_alias(Tensor* t);

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
