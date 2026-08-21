#define _POSIX_C_SOURCE 200809L
#include "nn.h"
#include "cml.h"
#include "core/logging.h"
#include "core/error_stack.h"
#include "backend/device.h"
#include "tensor/tensor.h"
#include "tensor/realize.h"
#include "autograd/autograd.h"
#include "ops/ir/internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "alloc/cml_allocator.h"

static Module* g_viz_root_module = NULL;

void nn_viz_set_root_module(Module* module) { g_viz_root_module = module; }
Module* nn_viz_root_module(void) { return g_viz_root_module; }

void nn_tensor_param_alias(Tensor* t) {
    if (t)
        t->ref_count++;
}

int module_init(Module* module, const char* name, ForwardFn forward, FreeFn free) {
    if (!module || !name)
        return -1;

    module->name                = cml_strdup(name);
    module->forward             = forward;
    module->free                = free;
    module->parameters          = NULL;
    module->num_parameters      = 0;
    module->parameters_capacity = 0;
    module->next                = NULL;
    module->training            = false;
    module->user_data           = NULL;
    module->backward_hooks      = NULL;
    module->version             = "1.0.0";
    module->description         = "Module";

    return 0;
}

Module* module_create(const char* name, ForwardFn forward, FreeFn free) {
    Module* module = cml_malloc(sizeof(Module));
    if (!module)
        return NULL;

    if (module_init(module, name, forward, free) != 0) {
        cml_free(module);
        return NULL;
    }

    return module;
}

void module_free(Module* module) {
    if (!module)
        return;

    
    cml_untrack_module(module);

    
    
    void (*specialized_free)(Module*) = module->free;

    
    module->free = NULL;

    if (module->name) {
        cml_free(module->name);
        module->name = NULL;
    }

    
    if (module->parameters) {
        for (int i = 0; i < module->num_parameters; i++) {
            if (module->parameters[i]) {
                Parameter* p = module->parameters[i];
                if (p->name) {
                    cml_free(p->name);
                    p->name = NULL;
                }
                if (p->tensor) {
                    tensor_free(p->tensor);
                    p->tensor = NULL;
                }
                cml_free(p);
                module->parameters[i] = NULL;
            }
        }
        cml_free(module->parameters);
        module->parameters = NULL;
    }

    autograd_free_module_hooks(module);

    if (specialized_free) {
        specialized_free(module);
    } else {
        
        cml_free(module);
    }
}

int module_add_parameter(Module* module, Tensor* tensor, const char* name, bool requires_grad) {
    if (!module || !tensor || !name) {
        LOG_ERROR("Invalid arguments to module_add_parameter");
        return -1;
    }

    /* Parameters must be leaf tensors (realized, detached from the IR graph)
     * so they survive across cml_ir_free() calls between training steps. */
    if (tensor->ir_node) {
        if (tensor_realize(tensor) != 0) {
            LOG_ERROR("module_add_parameter: failed to realize lazy tensor '%s'", name);
            return -1;
        }
    }

    if (module_get_parameter(module, name) != NULL) {
        LOG_WARNING("Parameter '%s' already exists in module '%s'", name, module->name);
        return -1;
    }

    if (module->num_parameters >= module->parameters_capacity) {
        int new_capacity = module->parameters_capacity == 0 ? 8 : module->parameters_capacity * 2;
        Parameter** new_params =
            cml_realloc(module->parameters, (size_t)new_capacity * sizeof(Parameter*));
        if (!new_params) {
            LOG_ERROR("Failed to allocate memory for parameters");
            return -1;
        }
        module->parameters          = new_params;
        module->parameters_capacity = new_capacity;
    }

    Parameter* param = cml_malloc(sizeof(Parameter));
    if (!param) {
        LOG_ERROR("Failed to allocate memory for parameter");
        return -1;
    }

    param->tensor        = tensor;
    param->requires_grad = requires_grad;
    param->name          = cml_strdup(name);

    if (!param->name) {
        LOG_ERROR("Failed to duplicate parameter name");
        cml_free(param);
        return -1;
    }

    tensor->requires_grad = requires_grad;

    module->parameters[module->num_parameters] = param;
    module->num_parameters++;

    LOG_DEBUG("Added parameter '%s' to module '%s' (total: %d)", name, module->name,
              module->num_parameters);

    return 0;
}

int module_get_parameters(Module* module, Parameter** params, int* num_parameters) {
    if (!module)
        return -1;

    if (num_parameters) {
        *num_parameters = module->num_parameters;
    }

    if (params && module->num_parameters > 0) {
        for (int i = 0; i < module->num_parameters; i++) {
            params[i] = module->parameters[i];
        }
    }

    return 0;
}

Parameter* module_get_parameter(Module* module, const char* name) {
    if (!module || !name)
        return NULL;

    for (int i = 0; i < module->num_parameters; i++) {
        if (module->parameters[i] && module->parameters[i]->name) {
            if (strcmp(module->parameters[i]->name, name) == 0) {
                return module->parameters[i];
            }
        }
    }

    return NULL;
}

Parameter* nn_add_bias_param(Module* module, int size, DType dtype, DeviceType device,
                             void (*init)(Tensor*, int)) {
    int shape[] = {size};
    TensorConfig cfg = {
        .dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* bias = tensor_zeros(shape, 1, &cfg);
    if (!bias) {
        module_free(module);
        return NULL;
    }
    if (init)
        init(bias, size);
    if (module_add_parameter(module, bias, "bias", true) != 0) {
        tensor_free(bias);
        module_free(module);
        return NULL;
    }
    return module_get_parameter(module, "bias");
}

Parameter* nn_add_weight_param(Module* module, Tensor* weight) {
    if (!weight) {
        module_free(module);
        return NULL;
    }
    if (module_add_parameter(module, weight, "weight", true) != 0) {
        tensor_free(weight);
        module_free(module);
        return NULL;
    }
    return module_get_parameter(module, "weight");
}

int nn_add_affine_params(Module* module, int size, DType dtype, DeviceType device,
                         Parameter** weight_out, Parameter** bias_out) {
    int shape[] = {size};
    TensorConfig cfg = {
        .dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* weight = tensor_ones(shape, 1, &cfg);
    if (!weight) {
        module_free(module);
        return -1;
    }
    if (module_add_parameter(module, weight, "weight", true) != 0) {
        tensor_free(weight);
        module_free(module);
        return -1;
    }
    *weight_out = module_get_parameter(module, "weight");
    *bias_out = nn_add_bias_param(module, size, dtype, device, NULL);
    return *bias_out ? 0 : -1;
}

int nn_add_running_stats(Module* module, int size, DType dtype, DeviceType device,
                         Tensor** mean_out, Tensor** var_out) {
    int shape[] = {size};
    TensorConfig cfg = {
        .dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    *mean_out = tensor_zeros(shape, 1, &cfg);
    *var_out  = tensor_ones(shape, 1, &cfg);
    if (!*mean_out || !*var_out) {
        if (*mean_out) tensor_free(*mean_out);
        if (*var_out)  tensor_free(*var_out);
        module_free(module);
        return -1;
    }
    return 0;
}

int module_set_parameter(Module* module, const char* name, Tensor* tensor) {
    if (!module || !name || !tensor)
        return -1;

    Parameter* param = module_get_parameter(module, name);
    if (!param) {
        LOG_WARNING("Parameter '%s' not found in module '%s'", name, module->name);
        return -1;
    }

    if (param->tensor && param->tensor != tensor)
        tensor_free(param->tensor);

    param->tensor         = tensor;
    tensor->requires_grad = param->requires_grad;

    LOG_DEBUG("Updated parameter '%s' in module '%s'", name, module->name);

    return 0;
}

Tensor* module_forward(Module* module, Tensor* input) {
    if (!module) {
        return NULL;
    }
    if (!module->forward) {
        return NULL;
    }
    /* Tag every IR node this layer emits with the module path, so the graph
     * view can collapse thousands of primitives back into the layers they came
     * from. Nested containers nest naturally (Sequential/Linear). No-op unless
     * VIZ is set. */
    /* The outermost forward in a pass is the model itself. Remembering it lets
     * the metrics layer summarise weights/gradients for any training loop,
     * including hand-written ones that never call cml_train. */
    if (cml_ir_scope_enabled() && cml_ir_scope_current() == NULL)
        nn_viz_set_root_module(module);

    cml_ir_scope_push(module->name);
    Tensor* out = module->forward(module, input);
    cml_ir_scope_pop();
    return out;
}

void module_set_training(Module* module, bool training) {
    if (module) {
        module->training = training;
    }
}

bool module_is_training(Module* module) { return module && module->training; }

void module_zero_grad(Module* module) {
    if (!module)
        return;

    for (int i = 0; i < module->num_parameters; i++) {
        if (module->parameters[i] && module->parameters[i]->tensor) {
            Tensor* tensor = module->parameters[i]->tensor;

            if (tensor->grad) {
                tensor_free(tensor->grad);
                tensor->grad = NULL;
            }

            if (tensor->requires_grad) {
                TensorConfig config = (TensorConfig){.dtype      = tensor->dtype,
                                                     .device     = tensor->device,
                                                     .has_dtype  = true,
                                                     .has_device = true};
                tensor->grad        = tensor_zeros(tensor->shape, tensor->ndim, &config);
            }
        }
    }

    LOG_DEBUG("Zeroed gradients for module '%s'", module->name);
}

const char* module_get_name(Module* module) { return module ? module->name : NULL; }

int module_get_parameter_count(Module* module) { return module ? module->num_parameters : 0; }

void module_print_summary(Module* module, int indent) {
    if (!module)
        return;

    for (int i = 0; i < indent; i++)
        printf("  ");
    printf("Module: %s (Parameters: %d)\n", module->name, module->num_parameters);
}

int module_get_total_parameters(Module* module) { return module ? module->num_parameters : 0; }

int module_chain(Module* first, Module* second) {
    if (!first || !second)
        return -1;

    first->next = second;
    return 0;
}

Module* module_get_next(Module* module) { return module ? module->next : NULL; }

void module_set_next(Module* module, Module* next) {
    if (module) {
        module->next = next;
    }
}


int module_collect_parameters(Module* module, Parameter*** params_out, int* num_params_out,
                              bool recursive) {
    if (!module || !params_out || !num_params_out)
        return -1;

    
    int total_params = module->num_parameters;
    if (recursive) {
        
        Module* current = module->next;
        while (current) {
            total_params += current->num_parameters;
            current = current->next;
        }

        
        
        
    }

    if (total_params == 0) {
        *params_out     = NULL;
        *num_params_out = 0;
        return 0;
    }

    Parameter** params = cml_malloc((size_t)total_params * sizeof(Parameter*));
    if (!params) {
        LOG_ERROR("Failed to allocate memory for parameter collection");
        return -1;
    }

    
    int idx = 0;

    
    for (int i = 0; i < module->num_parameters; i++) {
        if (module->parameters[i]) {
            params[idx++] = module->parameters[i];
        }
    }

    
    if (recursive) {
        Module* current = module->next;
        while (current) {
            for (int i = 0; i < current->num_parameters; i++) {
                if (current->parameters[i]) {
                    params[idx++] = current->parameters[i];
                }
            }
            current = current->next;
        }
    }

    *params_out     = params;
    *num_params_out = idx;

    LOG_DEBUG("Collected %d parameters from module '%s' (recursive=%d)", idx, module->name,
              recursive);

    return 0;
}

int module_to_device(Module* module, DeviceType device) {
    if (!module) {
        return -1;
    }

    
    Parameter** params = NULL;
    int num_params     = 0;
    if (module_collect_parameters(module, &params, &num_params, true) != 0) {
        return -1;
    }

    
    for (int i = 0; i < num_params; i++) {
        if (params[i] && params[i]->tensor) {
            if (device_move_tensor(params[i]->tensor, device) != 0) {
                cml_free(params);
                return -1;
            }
        }
    }

    cml_free(params);
    return 0;
}
