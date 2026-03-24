#define _POSIX_C_SOURCE 200809L
#include "nn.h"
#include "cml.h"
#include "core/logging.h"
#include "core/error_stack.h"
#include "backend/device.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int module_init(Module* module, const char* name, ForwardFn forward, FreeFn free) {
    if (!module || !name)
        return -1;

    module->name                = strdup(name);
    module->forward             = forward;
    module->free                = free;
    module->parameters          = NULL;
    module->num_parameters      = 0;
    module->parameters_capacity = 0;
    module->next                = NULL;
    module->training            = false;
    module->user_data           = NULL;
    module->version             = "1.0.0";
    module->description         = "Module";

    return 0;
}

Module* module_create(const char* name, ForwardFn forward, FreeFn free) {
    Module* module = malloc(sizeof(Module));
    if (!module)
        return NULL;

    if (module_init(module, name, forward, free) != 0) {
        free(module);
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
        free(module->name);
        module->name = NULL;
    }

    
    if (module->parameters) {
        for (int i = 0; i < module->num_parameters; i++) {
            if (module->parameters[i]) {
                if (module->parameters[i]->name) {
                    free(module->parameters[i]->name);
                    module->parameters[i]->name = NULL;
                }
                free(module->parameters[i]);
                module->parameters[i] = NULL;
            }
        }
        free(module->parameters);
        module->parameters = NULL;
    }

    
    
    
    
    if (specialized_free) {
        specialized_free(module);
    } else {
        
        free(module);
    }
}

int module_add_parameter(Module* module, Tensor* tensor, const char* name, bool requires_grad) {
    if (!module || !tensor || !name) {
        LOG_ERROR("Invalid arguments to module_add_parameter");
        return -1;
    }

    if (module_get_parameter(module, name) != NULL) {
        LOG_WARNING("Parameter '%s' already exists in module '%s'", name, module->name);
        return -1;
    }

    if (module->num_parameters >= module->parameters_capacity) {
        int new_capacity = module->parameters_capacity == 0 ? 8 : module->parameters_capacity * 2;
        Parameter** new_params =
            realloc(module->parameters, (size_t)new_capacity * sizeof(Parameter*));
        if (!new_params) {
            LOG_ERROR("Failed to allocate memory for parameters");
            return -1;
        }
        module->parameters          = new_params;
        module->parameters_capacity = new_capacity;
    }

    Parameter* param = malloc(sizeof(Parameter));
    if (!param) {
        LOG_ERROR("Failed to allocate memory for parameter");
        return -1;
    }

    param->tensor        = tensor;
    param->requires_grad = requires_grad;
    param->name          = strdup(name);

    if (!param->name) {
        LOG_ERROR("Failed to duplicate parameter name");
        free(param);
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

int module_set_parameter(Module* module, const char* name, Tensor* tensor) {
    if (!module || !name || !tensor)
        return -1;

    Parameter* param = module_get_parameter(module, name);
    if (!param) {
        LOG_WARNING("Parameter '%s' not found in module '%s'", name, module->name);
        return -1;
    }

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
    return module->forward(module, input);
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

static int collect_container_params(Module* container, Parameter*** params_out, int* num_params_out,
                                   bool recursive);

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

    Parameter** params = malloc((size_t)total_params * sizeof(Parameter*));
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
                free(params);
                return -1;
            }
        }
    }

    free(params);
    return 0;
}
