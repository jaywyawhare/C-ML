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
#include "nn/layers/sequential.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "alloc/cml_allocator.h"

static Module* g_viz_root_module = NULL;

/** Record the top-level module so the viz/metrics layer can summarize the model. */
void nn_viz_set_root_module(Module* module) { g_viz_root_module = module; }
/** Return the module last set as the viz root (may be NULL). */
Module* nn_viz_root_module(void) { return g_viz_root_module; }

/** Bump a tensor's refcount to mark it aliased as a module parameter. */
void nn_tensor_param_alias(Tensor* t) {
    if (t)
        t->ref_count++;
}

/** Register a non-trainable buffer (e.g. running stats) by name; the tensor is
 *  borrowed, the name is copied. -1 if the name already exists. */
int module_add_buffer(Module* module, Tensor* tensor, const char* name) {
    if (!module || !tensor || !name)
        return -1;

    if (module_get_buffer(module, name)) {
        LOG_WARNING("Buffer '%s' already exists in module '%s'", name, module->name);
        return -1;
    }

    if (module->num_buffers >= module->buffers_capacity) {
        int new_capacity     = module->buffers_capacity == 0 ? 4 : module->buffers_capacity * 2;
        Tensor** new_buffers = cml_realloc(module->buffers, (size_t)new_capacity * sizeof(Tensor*));
        char** new_names = cml_realloc(module->buffer_names, (size_t)new_capacity * sizeof(char*));
        if (!new_buffers || !new_names) {
            LOG_ERROR("Failed to grow buffer registry for module '%s'", module->name);
            return -1;
        }
        module->buffers          = new_buffers;
        module->buffer_names     = new_names;
        module->buffers_capacity = new_capacity;
    }

    char* copy = cml_strdup(name);
    if (!copy)
        return -1;

    module->buffers[module->num_buffers]      = tensor;
    module->buffer_names[module->num_buffers] = copy;
    module->num_buffers++;
    return 0;
}

/** Look up a registered buffer by name; NULL if absent. */
Tensor* module_get_buffer(Module* module, const char* name) {
    if (!module || !name)
        return NULL;
    for (int i = 0; i < module->num_buffers; i++) {
        if (module->buffer_names[i] && strcmp(module->buffer_names[i], name) == 0)
            return module->buffers[i];
    }
    return NULL;
}

/** Initialize an already-allocated module with its name, forward/free hooks, and
 *  empty parameter/buffer registries. */
int module_init(Module* module, const char* name, ForwardFn forward, FreeFn free) {
    if (!module || !name)
        return -1;

    module->name                = cml_strdup(name);
    module->forward             = forward;
    module->free                = free;
    module->parameters          = NULL;
    module->num_parameters      = 0;
    module->parameters_capacity = 0;
    module->buffers             = NULL;
    module->buffer_names        = NULL;
    module->num_buffers         = 0;
    module->buffers_capacity    = 0;
    module->next                = NULL;
    module->training            = false;
    module->user_data           = NULL;
    module->backward_hooks      = NULL;
    module->version             = "1.0.0";
    module->description         = "Module";

    return 0;
}

/** Allocate and initialize a new module with the given name and forward/free hooks. */
Module* module_create(const char* name, ForwardFn forward, FreeFn free) {
    /* calloc: Module structs are recycled by the allocator; every field must
     * start zeroed. module_init below sets the standard fields explicitly,
     * but future additions are covered by the zeroing. */
    Module* module = cml_calloc(1, sizeof(Module));
    if (!module)
        return NULL;

    if (module_init(module, name, forward, free) != 0) {
        cml_free(module);
        return NULL;
    }

    return module;
}

/** Free a module: untracks it, releases parameters (names + tensors) and buffer
 *  names, then delegates to the layer's specialized free hook (or frees the struct). */
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

    /* Buffer names are owned; the Tensors are not (layers free their own). */
    if (module->buffer_names) {
        for (int i = 0; i < module->num_buffers; i++)
            if (module->buffer_names[i])
                cml_free(module->buffer_names[i]);
        cml_free(module->buffer_names);
        module->buffer_names = NULL;
    }
    if (module->buffers) {
        cml_free(module->buffers);
        module->buffers = NULL;
    }
    module->num_buffers = 0;

    autograd_free_module_hooks(module);

    if (specialized_free) {
        specialized_free(module);
    } else {

        cml_free(module);
    }
}

/** Register a trainable parameter by name. Lazy tensors are realized first so they
 *  survive IR teardown between steps; -1 if the name already exists or on error. */
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

/** Copy the module's parameter pointers into `params` and/or report the count. */
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

/** Look up a parameter by name; NULL if absent. */
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

/** Create a zero-initialized "bias" parameter (optionally run `init`) and add it;
 *  frees the module and returns NULL on failure. */
Parameter* nn_add_bias_param(Module* module, int size, DType dtype, DeviceType device,
                             void (*init)(Tensor*, int)) {
    int shape[]      = {size};
    TensorConfig cfg = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* bias     = tensor_zeros(shape, 1, &cfg);
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

/** Add an existing tensor as the "weight" parameter; frees the module on failure. */
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

/** Add the affine pair for norm layers: ones "weight" and zeros "bias"; -1 on failure. */
int nn_add_affine_params(Module* module, int size, DType dtype, DeviceType device,
                         Parameter** weight_out, Parameter** bias_out) {
    int shape[]      = {size};
    TensorConfig cfg = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* weight   = tensor_ones(shape, 1, &cfg);
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
    *bias_out   = nn_add_bias_param(module, size, dtype, device, NULL);
    return *bias_out ? 0 : -1;
}

/** Allocate running-mean (zeros) and running-var (ones) buffers for a norm layer;
 *  frees the module and returns -1 on failure. */
int nn_add_running_stats(Module* module, int size, DType dtype, DeviceType device,
                         Tensor** mean_out, Tensor** var_out) {
    int shape[]      = {size};
    TensorConfig cfg = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    *mean_out        = tensor_zeros(shape, 1, &cfg);
    *var_out         = tensor_ones(shape, 1, &cfg);
    if (!*mean_out || !*var_out) {
        if (*mean_out)
            tensor_free(*mean_out);
        if (*var_out)
            tensor_free(*var_out);
        module_free(module);
        return -1;
    }
    return 0;
}

/** Replace a named parameter's tensor, freeing the previous one and syncing requires_grad. */
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

/** Run a module's forward pass, pushing an IR scope named after the module so emitted
 *  nodes can be grouped back into layers; also remembers the outermost module as viz root. */
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

/** Set the module's train/eval flag. */
void module_set_training(Module* module, bool training) {
    if (module) {
        module->training = training;
    }
}

/** True if the module is in training mode. */
bool module_is_training(Module* module) { return module && module->training; }

/** Reset each parameter's gradient: free any existing grad and re-zero it for
 *  parameters that require grad. */
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

/** Return the module's name, or NULL. */
const char* module_get_name(Module* module) { return module ? module->name : NULL; }

/** Number of directly-owned parameters. */
int module_get_parameter_count(Module* module) { return module ? module->num_parameters : 0; }

/** Print a one-line indented summary of the module and its parameter count. */
void module_print_summary(Module* module, int indent) {
    if (!module)
        return;

    for (int i = 0; i < indent; i++)
        printf("  ");
    printf("Module: %s (Parameters: %d)\n", module->name, module->num_parameters);
}

/** Total parameter count for the module. */
int module_get_total_parameters(Module* module) { return module ? module->num_parameters : 0; }

/** Link `second` after `first` in the module chain. */
int module_chain(Module* first, Module* second) {
    if (!first || !second)
        return -1;

    first->next = second;
    return 0;
}

/** Return the next module in the chain, or NULL. */
Module* module_get_next(Module* module) { return module ? module->next : NULL; }

/** Set the next module in the chain. */
void module_set_next(Module* module, Module* next) {
    if (module) {
        module->next = next;
    }
}

/** Gather the module's parameters (and, if recursive, those of chained modules) into
 *  a newly allocated array the caller frees. Sequential already flattens children, so
 *  recursion walks the chain without double-counting aliases. */
int module_collect_parameters(Module* module, Parameter*** params_out, int* num_params_out,
                              bool recursive) {
    if (!module || !params_out || !num_params_out)
        return -1;

    /* sequential_add flattens container children's parameters into the
     * container's own array, so this walk is complete for Sequential models
     * without recursing (recursing would double-count the aliases). */
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

    for (int i = 0; i < module->num_parameters; i++)
        if (module->parameters[i])
            params[idx++] = module->parameters[i];

    Module* current = recursive ? module->next : NULL;
    while (current) {
        for (int i = 0; i < current->num_parameters; i++)
            if (current->parameters[i])
                params[idx++] = current->parameters[i];
        current = current->next;
    }

    *params_out     = params;
    *num_params_out = idx;

    LOG_DEBUG("Collected %d parameters from module '%s' (recursive=%d)", idx, module->name,
              recursive);
    return 0;
}

/** Move all of a module's parameter tensors to the given device; -1 on failure. */
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
