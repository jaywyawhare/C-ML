/**
 * @file sequential.c
 * @brief Sequential container implementation
 */

#include "nn/layers/sequential.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "ops/ir/context.h"
#include "ops/ir/graph_cache.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/simd_math.h"
#include "backend/blas.h"
#include "core/logging.h"
#include "core/error_stack.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>

static void free_cached_graph(CachedModelGraph* cache) {
    if (!cache)
        return;

    if (cache->plan) {
        cml_free_execution_plan(cache->plan);
    }
    if (cache->input_shape) {
        free(cache->input_shape);
    }
    if (cache->input_buffer) {
        free(cache->input_buffer);
    }
    if (cache->output_buffer) {
        free(cache->output_buffer);
    }
    free(cache);
}

static bool shapes_match(CachedModelGraph* cache, Tensor* input) {
    if (!cache || !cache->valid || !input)
        return false;
    if (cache->input_ndim != input->ndim)
        return false;

    for (int i = 0; i < input->ndim; i++) {
        if (cache->input_shape[i] != input->shape[i])
            return false;
    }
    return true;
}

static CachedModelGraph* create_cached_graph(Tensor* input, Tensor* output, CMLGraph_t ir) {
    if (!input || !output || !ir)
        return NULL;

    CachedModelGraph* cache = calloc(1, sizeof(CachedModelGraph));
    if (!cache)
        return NULL;

    // Store input shape
    cache->input_ndim  = input->ndim;
    cache->input_shape = malloc(sizeof(int) * input->ndim);
    if (!cache->input_shape) {
        free(cache);
        return NULL;
    }
    memcpy(cache->input_shape, input->shape, sizeof(int) * input->ndim);
    cache->input_numel = input->numel;

    // Allocate input buffer
    cache->input_buffer = aligned_alloc(32, input->numel * sizeof(float));
    if (!cache->input_buffer) {
        free(cache->input_shape);
        free(cache);
        return NULL;
    }

    // Store output info and allocate buffer
    cache->output_numel  = output->numel;
    cache->output_buffer = aligned_alloc(32, output->numel * sizeof(float));
    if (!cache->output_buffer) {
        free(cache->input_buffer);
        free(cache->input_shape);
        free(cache);
        return NULL;
    }

    // Create execution plan
    cache->plan = cml_create_execution_plan(ir);
    if (!cache->plan) {
        free(cache->output_buffer);
        free(cache->input_buffer);
        free(cache->input_shape);
        free(cache);
        return NULL;
    }

    cache->valid = true;
    LOG_DEBUG("Created cached graph: input[%zu], output[%zu], %zu nodes", cache->input_numel,
              cache->output_numel, cache->plan->num_nodes);

    return cache;
}

// Execute using cached graph
// The key insight is that we need to:
// 1. Copy new input data into the cached graph's first node inputs
// 2. Make intermediate nodes read from plan buffers (set by previous nodes)
// 3. Execute nodes in order, writing to plan buffers
static Tensor* execute_cached_forward(Sequential* seq, Tensor* input) {
    if (!seq || !seq->cached_graph || !seq->cached_graph->plan) {
        return NULL;
    }

    CachedModelGraph* cache = seq->cached_graph;
    CMLExecutionPlan* plan  = cache->plan;

    if (!plan->valid || plan->num_nodes == 0) {
        return NULL;
    }

    // Copy input data to the input buffer (this is the new input for this forward pass)
    memcpy(cache->input_buffer, input->data, cache->input_numel * sizeof(float));

    // Execute each node in order, using pre-allocated buffers
    for (size_t i = 0; i < plan->num_nodes; i++) {
        struct IRNode* node = plan->nodes[i];
        float* out_buf      = plan->buffers[i];

        if (!node || !out_buf) {
            LOG_ERROR("Invalid node or buffer at index %zu", i);
            return NULL;
        }

        // For each input of this node, we need to determine where to read from:
        // - If it's the original input tensor, read from cache->input_buffer
        // - If it's an intermediate result, read from the appropriate plan->buffers[j]
        for (int inp_idx = 0; inp_idx < node->num_inputs; inp_idx++) {
            if (!node->inputs || !node->inputs[inp_idx])
                continue;

            Tensor* inp_tensor = node->inputs[inp_idx];

            // Check if this input is the original input tensor (compare numel and shape)
            if (inp_tensor->numel == cache->input_numel) {
                // This might be the input tensor - use our cached input buffer
                // Note: we can't compare pointers because the original tensor is gone,
                // but we can check if this tensor's data was never set to a plan buffer
                bool is_plan_buffer = false;
                for (size_t j = 0; j < i; j++) {
                    if (inp_tensor->data == plan->buffers[j]) {
                        is_plan_buffer = true;
                        break;
                    }
                }
                if (!is_plan_buffer) {
                    inp_tensor->data = cache->input_buffer;
                }
            }
            // If data pointer matches a plan buffer, it's already pointing to an intermediate
            // result
        }

        // Execute the node (this reads from node->inputs[*]->data and writes to out_buf)
        // Re-implement core execution logic inline for reliability
        float* in1   = NULL;
        float* in2   = NULL;
        size_t in1_n = 0, in2_n = 0;
        size_t out_n = node->output ? node->output->numel : 0;

        if (node->num_inputs >= 1 && node->inputs && node->inputs[0]) {
            in1   = (float*)node->inputs[0]->data;
            in1_n = node->inputs[0]->numel;
        }
        if (node->num_inputs >= 2 && node->inputs && node->inputs[1]) {
            in2   = (float*)node->inputs[1]->data;
            in2_n = node->inputs[1]->numel;
        }

        // Execute based on operation type (simplified inline execution)
        int exec_result = -1;
        switch (node->type) {
        case UOP_ADD:
            if (in1 && in2) {
                if (in1_n == in2_n && in1_n == out_n) {
                    simd_add_f32(in1, in2, out_buf, out_n);
                } else {
                    for (size_t j = 0; j < out_n; j++) {
                        out_buf[j] =
                            in1[in1_n == 1 ? 0 : j % in1_n] + in2[in2_n == 1 ? 0 : j % in2_n];
                    }
                }
                exec_result = 0;
            }
            break;

        case UOP_MUL:
            if (in1 && in2) {
                if (in1_n == in2_n && in1_n == out_n) {
                    simd_mul_f32(in1, in2, out_buf, out_n);
                } else {
                    for (size_t j = 0; j < out_n; j++) {
                        out_buf[j] =
                            in1[in1_n == 1 ? 0 : j % in1_n] * in2[in2_n == 1 ? 0 : j % in2_n];
                    }
                }
                exec_result = 0;
            }
            break;

        case UOP_MATMUL: {
            if (in1 && in2) {
                Tensor* a = node->inputs[0];
                Tensor* b = node->inputs[1];
                if (a->ndim >= 2 && b->ndim >= 2) {
                    int M = a->shape[a->ndim - 2];
                    int K = a->shape[a->ndim - 1];
                    int N = b->shape[b->ndim - 1];
                    memset(out_buf, 0, out_n * sizeof(float));
                    CMLBlasContext* blas = cml_blas_get_context();
                    if (blas && blas->initialized) {
                        cml_blas_sgemm(blas, in1, in2, out_buf, M, N, K, 1.0f, 0.0f);
                    } else {
                        // Fallback naive matmul
                        for (int m = 0; m < M; m++) {
                            for (int k = 0; k < K; k++) {
                                float a_mk = in1[m * K + k];
                                for (int n = 0; n < N; n++) {
                                    out_buf[m * N + n] += a_mk * in2[k * N + n];
                                }
                            }
                        }
                    }
                    exec_result = 0;
                }
            }
            break;
        }

        case UOP_MAX:
            if (in1 && in2) {
                for (size_t j = 0; j < out_n; j++) {
                    float v1   = in1[in1_n == 1 ? 0 : j % in1_n];
                    float v2   = in2[in2_n == 1 ? 0 : j % in2_n];
                    out_buf[j] = v1 > v2 ? v1 : v2;
                }
                exec_result = 0;
            }
            break;

        case UOP_SIGMOID:
            if (in1) {
                simd_sigmoid_f32(in1, out_buf, out_n);
                exec_result = 0;
            }
            break;

        case UOP_TANH:
            if (in1) {
                simd_tanh_f32(in1, out_buf, out_n);
                exec_result = 0;
            }
            break;

        case UOP_EXP:
            if (in1) {
                simd_exp_f32(in1, out_buf, out_n);
                exec_result = 0;
            }
            break;

        case UOP_NEG:
            if (in1) {
                simd_neg_f32(in1, out_buf, out_n);
                exec_result = 0;
            }
            break;

        case UOP_SUB:
            if (in1 && in2) {
                if (in1_n == in2_n && in1_n == out_n) {
                    simd_sub_f32(in1, in2, out_buf, out_n);
                } else {
                    for (size_t j = 0; j < out_n; j++) {
                        out_buf[j] =
                            in1[in1_n == 1 ? 0 : j % in1_n] - in2[in2_n == 1 ? 0 : j % in2_n];
                    }
                }
                exec_result = 0;
            }
            break;

        case UOP_DIV:
            if (in1 && in2) {
                if (in1_n == in2_n && in1_n == out_n) {
                    simd_div_f32(in1, in2, out_buf, out_n);
                } else {
                    for (size_t j = 0; j < out_n; j++) {
                        float denom = in2[in2_n == 1 ? 0 : j % in2_n];
                        out_buf[j] =
                            in1[in1_n == 1 ? 0 : j % in1_n] / (denom != 0.0f ? denom : 1e-8f);
                    }
                }
                exec_result = 0;
            }
            break;

        case UOP_PERMUTE: {
            // 2D transpose: out[j,i] = in[i,j]
            if (in1 && node->inputs[0]->ndim == 2) {
                int rows = node->inputs[0]->shape[0];
                int cols = node->inputs[0]->shape[1];
                simd_transpose_f32(in1, out_buf, rows, cols);
                exec_result = 0;
            } else if (in1) {
                // Fallback: just copy for other cases
                memcpy(out_buf, in1, out_n * sizeof(float));
                exec_result = 0;
            }
            break;
        }

        case UOP_RESHAPE:
        case UOP_EXPAND:
        case UOP_STRIDE:
        case UOP_SLICE:
            // Movement ops - copy data
            if (in1) {
                if (in1_n == out_n) {
                    memcpy(out_buf, in1, out_n * sizeof(float));
                } else {
                    for (size_t j = 0; j < out_n; j++) {
                        out_buf[j] = in1[j % in1_n];
                    }
                }
                exec_result = 0;
            }
            break;

        default:
            // Unsupported operation - fall back to non-cached path
            LOG_WARNING("Unsupported op %d in cached execution", node->type);
            return NULL;
        }

        if (exec_result != 0) {
            LOG_ERROR("Cached execution failed at node %zu (type %d)", i, node->type);
            return NULL;
        }

        // Update the node's output data pointer to our buffer
        // This is important so that subsequent nodes can read from it
        if (node->output) {
            node->output->data = out_buf;
        }
    }

    // Copy final result to output buffer
    struct IRNode* last_node = plan->nodes[plan->num_nodes - 1];
    if (last_node && last_node->output) {
        memcpy(cache->output_buffer, plan->buffers[plan->num_nodes - 1],
               cache->output_numel * sizeof(float));
    }

    // Create output tensor pointing to our cached output buffer
    // The tensor doesn't own the data - we manage it
    Tensor* output = tensor_empty(last_node->output->shape, last_node->output->ndim, NULL);
    if (output) {
        output->data      = cache->output_buffer;
        output->owns_data = false; // We manage the buffer
    }

    return output;
}

static Tensor* sequential_forward(Module* module, Tensor* input) {
    Sequential* seq = (Sequential*)module;

    if (!seq || !input)
        return NULL;

    // FAST PATH: Try cached execution if enabled and shapes match
    if (seq->enable_graph_cache && seq->cached_graph && shapes_match(seq->cached_graph, input)) {
        Tensor* cached_output = execute_cached_forward(seq, input);
        if (cached_output) {
            return cached_output; // Cache hit!
        }
        // Cache execution failed - invalidate and fall through to normal path
        LOG_WARNING("Cached execution failed, rebuilding graph");
        sequential_invalidate_cache(seq);
    }

    // NORMAL PATH: Build IR graph and execute
    // Reset global IR context to prevent accumulation
    cml_ir_reset_global_context();

    Tensor* output = input;

    // Forward through all modules sequentially
    for (int i = 0; i < seq->num_modules; i++) {
        if (!seq->modules[i])
            continue;

        Tensor* next_output = module_forward(seq->modules[i], output);

        if (!next_output) {
            LOG_ERROR("Forward pass failed at module %d in Sequential", i);
            if (output != input) {
                tensor_free(output);
            }
            return NULL;
        }

        output = next_output;
    }

    // If graph caching is enabled but we don't have a cache, create one
    if (seq->enable_graph_cache && !seq->cached_graph) {
        CMLGraph_t ir = cml_ir_get_or_create_context();
        if (ir) {
            // Force execution first to ensure graph is complete
            float* out_data = (float*)tensor_data_ptr(output);
            (void)out_data;

            seq->cached_graph = create_cached_graph(input, output, ir);
            if (seq->cached_graph) {
                LOG_DEBUG("Graph cached for Sequential model");
            }
        }
    }

    return output;
}

static void sequential_free(Module* module) {
    Sequential* seq = (Sequential*)module;
    if (!seq)
        return;

    // Free cached graph if present
    if (seq->cached_graph) {
        free_cached_graph(seq->cached_graph);
        seq->cached_graph = NULL;
    }

    // Free all child modules (they are NOT tracked by global cleanup individually)
    if (seq->modules) {
        for (int i = 0; i < seq->num_modules; i++) {
            if (seq->modules[i]) {
                module_free(seq->modules[i]);
            }
        }
        free(seq->modules);
    }

    free(seq);
}

Sequential* nn_sequential(void) {
    Sequential* seq = malloc(sizeof(Sequential));
    if (!seq) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR,
                         "Failed to allocate memory for Sequential module", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    if (module_init((Module*)seq, "Sequential", sequential_forward, sequential_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize Sequential module", __FILE__,
                         __LINE__, __func__);
        free(seq);
        return NULL;
    }

    seq->modules            = NULL;
    seq->num_modules        = 0;
    seq->capacity           = 0;
    seq->cached_graph       = NULL;
    seq->enable_graph_cache = false; // Disabled by default

    // Automatically track for cleanup
    extern void cml_track_module(Module*);
    cml_track_module((Module*)seq);

    // Automatically register for metrics tracking
    extern void training_metrics_register_model(Module*);
    training_metrics_register_model((Module*)seq);

    return seq;
}

int sequential_add(Sequential* seq, Module* module) {
    if (!seq || !module)
        return -1;

    // Resize array if needed
    if (seq->num_modules >= seq->capacity) {
        int new_capacity     = seq->capacity == 0 ? 8 : seq->capacity * 2;
        Module** new_modules = realloc(seq->modules, (size_t)new_capacity * sizeof(Module*));
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
            free(params);
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

Sequential* sequential_add_chain(Sequential* seq, Module* module) {
    if (!seq || !module) {
        return seq;
    }
    sequential_add(seq, module);
    return seq;
}

Sequential* nn_sequentialv(int num_layers, ...) {
    Sequential* seq = nn_sequential();
    if (!seq) {
        return NULL;
    }

    va_list args;
    va_start(args, num_layers);

    for (int i = 0; i < num_layers; i++) {
        Module* module = va_arg(args, Module*);
        if (module) {
            sequential_add(seq, module);
        }
    }

    va_end(args);
    return seq;
}

void sequential_enable_graph_cache(Sequential* seq, bool enable) {
    if (!seq)
        return;
    seq->enable_graph_cache = enable;
    if (!enable && seq->cached_graph) {
        sequential_invalidate_cache(seq);
    }
    LOG_DEBUG("Graph caching %s for Sequential model", enable ? "enabled" : "disabled");
}

void sequential_invalidate_cache(Sequential* seq) {
    if (!seq || !seq->cached_graph)
        return;
    LOG_DEBUG("Invalidating cached graph for Sequential model");
    free_cached_graph(seq->cached_graph);
    seq->cached_graph = NULL;
}
