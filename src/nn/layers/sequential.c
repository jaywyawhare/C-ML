/**
 * @file sequential.c
 * @brief Sequential container implementation
 */

#include "nn/layers/sequential.h"
#include "nn/module.h"
#include "tensor/tensor.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static Tensor* sequential_forward(Module* module, Tensor* input) {
    Sequential* seq = (Sequential*)module;

    if (!seq || !input)
        return NULL;

    Tensor* output = input;

    // Forward through all modules sequentially
    // NOTE: We do NOT free intermediate outputs here because they are part of
    // the computation graph and needed for the backward pass. They will be
    // freed when the computation graph is cleaned up during backward pass.
    for (int i = 0; i < seq->num_modules; i++) {
        if (!seq->modules[i])
            continue;

        Tensor* next_output = module_forward(seq->modules[i], output);

        if (!next_output) {
            LOG_ERROR("Forward pass failed at module %d in Sequential", i);
            // Only free output if it's not the input (avoid double-free)
            if (output != input) {
                tensor_free(output);
            }
            return NULL;
        }

        // Don't free output here - it's part of the computation graph
        // The backward pass will handle cleanup
        output = next_output;
    }

    return output;
}

static void sequential_free(Module* module) {
    Sequential* seq = (Sequential*)module;
    if (!seq)
        return;

    // Free all modules
    if (seq->modules) {
        for (int i = 0; i < seq->num_modules; i++) {
            if (seq->modules[i]) {
                module_free(seq->modules[i]);
            }
        }
        CM_FREE(seq->modules);
    }

    CM_FREE(seq);
}

Sequential* nn_sequential(void) {
    Sequential* seq = CM_MALLOC(sizeof(Sequential));
    if (!seq)
        return NULL;

    if (module_init((Module*)seq, "Sequential", sequential_forward, sequential_free) != 0) {
        CM_FREE(seq);
        return NULL;
    }

    seq->modules     = NULL;
    seq->num_modules = 0;
    seq->capacity    = 0;

    return seq;
}

int sequential_add(Sequential* seq, Module* module) {
    if (!seq || !module)
        return -1;

    // Resize array if needed
    if (seq->num_modules >= seq->capacity) {
        int new_capacity     = seq->capacity == 0 ? 8 : seq->capacity * 2;
        Module** new_modules = CM_REALLOC(seq->modules, new_capacity * sizeof(Module*));
        if (!new_modules) {
            LOG_ERROR("Failed to allocate memory for Sequential modules");
            return -1;
        }
        seq->modules  = new_modules;
        seq->capacity = new_capacity;
    }

    seq->modules[seq->num_modules] = module;
    int module_index               = seq->num_modules; // Store index before incrementing
    seq->num_modules++;

    Parameter** params = NULL;
    int num_params     = 0;
    if (module_collect_parameters(module, &params, &num_params, true) == 0) {
        // Add parameters to Sequential with unique names based on module index
        for (int i = 0; i < num_params; i++) {
            if (params[i]) {
                // Create unique parameter name: "module_index.module_name.param_name"
                // This ensures each layer's parameters have unique names
                char param_name[256];
                snprintf(param_name, sizeof(param_name), "%d.%s.%s", module_index, module->name,
                         params[i]->name ? params[i]->name : "unnamed");
                int result = module_add_parameter((Module*)seq, params[i]->tensor, param_name,
                                                  params[i]->requires_grad);
                if (result != 0) {
                    LOG_WARNING(
                        "Failed to add parameter '%s' from module '%s' (index %d) to Sequential",
                        param_name, module->name, module_index);
                }
            }
        }
        if (params)
            CM_FREE(params);
    }

    LOG_DEBUG(
        "Added module '%s' (index %d) to Sequential (total modules: %d, total parameters: %d)",
        module->name, module_index, seq->num_modules, ((Module*)seq)->num_parameters);

    return 0;
}

Module* sequential_get(Sequential* seq, int index) {
    if (!seq || index < 0 || index >= seq->num_modules) {
        return NULL;
    }

    return seq->modules[index];
}

int sequential_get_length(Sequential* seq) { return seq ? seq->num_modules : 0; }
