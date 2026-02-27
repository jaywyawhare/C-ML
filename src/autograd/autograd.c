#include "autograd/autograd.h"
#include "core/training_metrics.h"
#include "nn.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

AutogradEngine* global_autograd_engine = NULL;

void autograd_init(void) {
    if (global_autograd_engine) {
        LOG_WARNING("Autograd engine already initialized");
        return;
    }

    global_autograd_engine = malloc(sizeof(AutogradEngine));
    if (!global_autograd_engine) {
        LOG_ERROR("Failed to initialize autograd engine");
        return;
    }

    global_autograd_engine->enabled           = true;
    global_autograd_engine->grad_mode         = true;
    global_autograd_engine->anomaly_detection = false;
    global_autograd_engine->deterministic     = false;
    global_autograd_engine->create_graph      = false;
    global_autograd_engine->accumulate_grad   = true;

    LOG_INFO("Autograd engine initialized successfully");
}

void autograd_shutdown(void) {
    if (global_autograd_engine) {
        free(global_autograd_engine);
        global_autograd_engine = NULL;
        LOG_INFO("Autograd engine shut down");
    }
}

AutogradEngine* autograd_get_engine(void) {
    if (!global_autograd_engine) {
        autograd_init();
    }
    return global_autograd_engine;
}

void autograd_set_grad_mode(bool enabled) {
    AutogradEngine* engine = autograd_get_engine();
    engine->grad_mode      = enabled;
    LOG_DEBUG("Gradient mode set to %s", enabled ? "enabled" : "disabled");
}

bool autograd_is_grad_enabled(void) {
    AutogradEngine* engine = autograd_get_engine();
    return engine->grad_mode && engine->enabled;
}

void autograd_no_grad_enter(void) { autograd_set_grad_mode(false); }

void autograd_no_grad_exit(void) { autograd_set_grad_mode(true); }

void autograd_set_anomaly_detection(bool enabled) {
    AutogradEngine* engine    = autograd_get_engine();
    engine->anomaly_detection = enabled;
    LOG_INFO("Anomaly detection %s", enabled ? "enabled" : "disabled");
}

bool tensor_requires_grad(Tensor* t) { return t && t->requires_grad; }

void tensor_set_requires_grad(Tensor* t, bool requires_grad) {
    if (!t)
        return;
    t->requires_grad = requires_grad;
    LOG_DEBUG("Set requires_grad=%s for tensor %p", requires_grad ? "true" : "false", (void*)t);
}

bool tensor_is_leaf(Tensor* t) { return t && (t->ir_node == NULL); }

Tensor* tensor_detach(Tensor* t) {
    if (!t)
        return NULL;

    LOG_DEBUG("Detaching tensor %p", (void*)t);

    // Create a new tensor that shares data but has no IR node
    Tensor* detached = tensor_clone(t);
    if (!detached)
        return NULL;

    detached->requires_grad = false;
    detached->ir_node       = NULL;
    detached->ir_context    = NULL;
    detached->grad          = NULL;

    return detached;
}

void tensor_detach_inplace(Tensor* t) {
    if (!t)
        return;

    LOG_DEBUG("In-place detaching tensor %p", (void*)t);

    t->requires_grad = false;
    t->ir_node       = NULL;
    t->ir_context    = NULL;
}

void tensor_retain_grad(Tensor* t) {
    if (!t)
        return;

    LOG_DEBUG("Retaining gradient for tensor %p", (void*)t);
    t->requires_grad = true;
}

#define MAX_HOOKS 16

typedef struct {
    TensorBackwardHook hooks[MAX_HOOKS];
    int num_hooks;
} TensorHookList;

static TensorHookList* get_tensor_hooks(Tensor* t) {
    if (!t->user_data) {
        t->user_data = calloc(1, sizeof(TensorHookList));
    }
    return (TensorHookList*)t->user_data;
}

int tensor_register_backward_hook(Tensor* t, TensorBackwardHook hook) {
    if (!t || !hook) {
        LOG_ERROR("Invalid arguments to tensor_register_backward_hook");
        return -1;
    }

    TensorHookList* hooks = get_tensor_hooks(t);
    if (hooks->num_hooks >= MAX_HOOKS) {
        LOG_ERROR("Maximum number of hooks (%d) reached", MAX_HOOKS);
        return -1;
    }

    hooks->hooks[hooks->num_hooks++] = hook;
    LOG_DEBUG("Registered backward hook for tensor %p", (void*)t);
    return 0;
}

void tensor_remove_hooks(Tensor* t) {
    if (!t || !t->user_data) {
        return;
    }

    TensorHookList* hooks = (TensorHookList*)t->user_data;
    hooks->num_hooks      = 0;
    LOG_DEBUG("Removed all hooks for tensor %p", (void*)t);
}

int module_register_backward_hook(struct Module* module, ModuleBackwardHook hook) {
    if (!module || !hook) {
        LOG_ERROR("Invalid arguments to module_register_backward_hook");
        return -1;
    }

    // Store hook in module's user_data
    if (!module->user_data) {
        module->user_data = malloc(sizeof(ModuleBackwardHook));
        if (!module->user_data) {
            LOG_ERROR("Failed to allocate memory for module hook");
            return -1;
        }
    }

    *(ModuleBackwardHook*)module->user_data = hook;
    LOG_DEBUG("Registered backward hook for module %s", module->name ? module->name : "unknown");
    return 0;
}

void tensor_zero_grad(Tensor* tensor) {
    if (!tensor)
        return;

    LOG_DEBUG("Zeroing gradient for tensor %p", (void*)tensor);

    if (tensor->grad) {
        tensor_free(tensor->grad);
        tensor->grad = NULL;
    }
}

void tensor_accumulate_grad(Tensor* tensor, Tensor* new_grad) {
    if (!tensor || !new_grad)
        return;

    // Only accumulate gradients for tensors that require them
    if (!tensor->requires_grad) {
        LOG_DEBUG("Skipping gradient accumulation for tensor %p (requires_grad=false)",
                  (void*)tensor);
        return;
    }

    if (!tensor->grad) {
        // First gradient, just assign it
        tensor->grad = tensor_clone(new_grad);
        LOG_DEBUG("Initialized gradient for tensor %p", (void*)tensor);
    } else {
        // Accumulate gradient (add new_grad to existing grad)
        LOG_DEBUG("Accumulating gradient for tensor %p", (void*)tensor);

        // Element-wise addition
        for (size_t i = 0; i < tensor->grad->numel && i < new_grad->numel; i++) {
            float old_val = tensor_get_float(tensor->grad, i);
            float new_val = tensor_get_float(new_grad, i);
            tensor_set_float(tensor->grad, i, old_val + new_val);
        }
    }

    // Check for anomalies if enabled
    if (autograd_get_engine()->anomaly_detection) {
        autograd_check_anomaly(tensor->grad, "gradient accumulation");
    }

    // CRITICAL: If this tensor is an IR facade (has ir_node and ir_context),
    // also propagate the gradient to ir_node->output so the next backward node can find it
    if (tensor->ir_node && tensor->ir_node->output && tensor->ir_node->output != tensor) {
        fflush(stdout);
        if (!tensor->ir_node->output->grad) {
            tensor->ir_node->output->grad = tensor_clone(tensor->grad);
        }
        // Note: we don't accumulate here since we just set tensor->grad above
    }
}

Tensor* tensor_get_grad(Tensor* tensor) { return tensor ? tensor->grad : NULL; }

void tensor_backward(Tensor* tensor, Tensor* gradient, bool retain_graph, bool create_graph) {
    (void)retain_graph; // Reserved for future use
    (void)create_graph; // Reserved for future use

    if (!tensor) {
        LOG_ERROR("Cannot compute gradients for NULL tensor");
        return;
    }

    if (!autograd_is_grad_enabled()) {
        LOG_WARNING("Gradient computation is disabled");
        return;
    }

    if (!tensor->requires_grad) {
        LOG_WARNING("Tensor does not require gradients");
        return;
    }

    // Ensure tensor is executed before backward pass
    // This is critical for lazy execution: we need the forward pass values to compute gradients!

    if (tensor_ensure_executed(tensor) != 0) {
        LOG_ERROR("Failed to execute tensor for backward pass");
        return;
    }

    // Auto-capture loss (if this is a scalar loss tensor)
    // Must be called AFTER tensor_ensure_executed so the loss value is available
    training_metrics_auto_capture_loss(tensor);

    LOG_INFO("Starting backward pass for tensor %p", (void*)tensor);

    // Initialize gradient if not provided
    if (!gradient) {
        // For scalar tensors, initialize gradient to ones
        TensorConfig config = (TensorConfig){.dtype      = tensor->dtype,
                                             .device     = tensor->device,
                                             .has_dtype  = true,
                                             .has_device = true};
        gradient            = tensor_ones(tensor->shape, tensor->ndim, &config);
        if (!gradient) {
            LOG_ERROR("Failed to initialize gradient");
            return;
        }
    }

    // Set root gradient
    if (!tensor->grad) {
        tensor->grad = tensor_clone(gradient);
    } else {
        tensor_accumulate_grad(tensor, gradient);
    }

    // Build backward graph in IR
    if (cml_ir_build_backward(tensor->ir_context, tensor->ir_node) != 0) {
        LOG_ERROR("Failed to build backward graph");
        return;
    }

    // Execute backward pass
    if (cml_ir_execute_backward(tensor->ir_context) != 0) {
        LOG_ERROR("Failed to execute backward pass");
        LOG_ERROR("Aborting training run to avoid repeated failing executions");
        // Hard fail so main/training_loop don't keep looping on the same error.
        exit(1);
    }
    LOG_INFO("Backward pass completed using IR for tensor %p", (void*)tensor);

    const char* viz     = getenv("CML_VIZ");
    const char* viz_env = getenv("VIZ");
    if ((viz && viz[0] != '\0') ||
        (viz_env && (viz_env[0] == '1' || strcmp(viz_env, "true") == 0))) {

        // CRITICAL: Optimize IR before exporting to show fusion and dead code elimination
        if (tensor->ir_context) {
            cml_ir_optimize(tensor->ir_context);
        }

        const char* out_path = "graph.json";
        int rc               = autograd_export_json(tensor, out_path);
        if (rc != 0) {
            LOG_WARNING("CML_VIZ export failed rc=%d", rc);
        } else {
            LOG_INFO("CML_VIZ exported graph to %s", out_path);
        }

        // Export kernel analysis for both raw and optimized views
        if (tensor->ir_context) {
            char* kernel_json_raw = cml_ir_export_kernel_analysis(tensor->ir_context, false);
            char* kernel_json_opt = cml_ir_export_kernel_analysis(tensor->ir_context, true);

            if (kernel_json_raw) {
                FILE* f = fopen("kernels.json", "w");
                if (f) {
                    fprintf(f, "{\"unoptimized\":%s,\"optimized\":%s}", kernel_json_raw,
                            kernel_json_opt ? kernel_json_opt : "{}");
                    fclose(f);
                    LOG_INFO("CML_VIZ exported kernels to kernels.json");
                }
                free(kernel_json_raw);
            }
            if (kernel_json_opt) {
                free(kernel_json_opt);
            }

            // Ensure loss tensor data is materialized before resetting IR
            tensor_ensure_executed(tensor);

            // Ensure gradients are materialized before resetting IR
            cml_ir_ensure_gradients_executed(tensor->ir_context);

            tensor->ir_context = NULL;
        }
    } else {
        if (tensor->ir_context) {
            tensor->ir_context = NULL;
        }
    }
}

/**
 * @brief Check if two shapes can be broadcast together (NumPy-style rules)
 *
 * NumPy broadcasting rules:
 * 1. Shapes are aligned from the right (trailing dimensions)
 * 2. Two dimensions are compatible if:
 *    - They are equal, OR
 *    - One of them is 1
 * 3. Empty dimensions (for scalars) are treated as size 1
 *
 * @param shape1 First shape array
 * @param ndim1 Number of dimensions in shape1
 * @param shape2 Second shape array
 * @param ndim2 Number of dimensions in shape2
 * @return true if shapes can be broadcast, false otherwise
 */
bool tensor_can_broadcast_shapes(int* shape1, int ndim1, int* shape2, int ndim2) {
    if (!shape1 || !shape2)
        return false;

    // Handle scalar cases (0D tensors)
    if (ndim1 == 0 && ndim2 == 0)
        return true;
    if (ndim1 == 0 || ndim2 == 0)
        return true; // Scalar can broadcast with any shape

    int max_ndim = ndim1 > ndim2 ? ndim1 : ndim2;

    // Check from right to left (trailing dimensions)
    for (int i = 0; i < max_ndim; i++) {
        // Get dimensions from right (trailing)
        int idx1 = ndim1 - 1 - i;
        int idx2 = ndim2 - 1 - i;

        int dim1 = (idx1 >= 0 && idx1 < ndim1) ? shape1[idx1] : 1;
        int dim2 = (idx2 >= 0 && idx2 < ndim2) ? shape2[idx2] : 1;

        // Check compatibility: equal OR one is 1
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;
        }

        // Check for invalid dimensions (negative or zero sizes)
        if (dim1 < 0 || dim2 < 0) {
            return false;
        }
    }

    return true;
}

/**
 * @brief Compute output shape after broadcasting (NumPy-style)
 *
 * The output shape has the maximum number of dimensions, and each dimension
 * is the maximum of the corresponding input dimensions.
 *
 * @param shape1 First shape array
 * @param ndim1 Number of dimensions in shape1
 * @param shape2 Second shape array
 * @param ndim2 Number of dimensions in shape2
 * @param out_ndim Output: number of dimensions in result
 * @return Output shape array (caller must free), or NULL on failure
 */
int* broadcast_shapes(int* shape1, int ndim1, int* shape2, int ndim2, int* out_ndim) {
    if (!shape1 || !shape2 || !out_ndim)
        return NULL;

    // Handle scalar cases
    if (ndim1 == 0 && ndim2 == 0) {
        *out_ndim = 0;
        return NULL; // Scalar result
    }
    if (ndim1 == 0) {
        // Broadcast scalar to shape2
        int* result = tensor_shape_copy(shape2, ndim2);
        if (result)
            *out_ndim = ndim2;
        return result;
    }
    if (ndim2 == 0) {
        // Broadcast scalar to shape1
        int* result = tensor_shape_copy(shape1, ndim1);
        if (result)
            *out_ndim = ndim1;
        return result;
    }

    int max_ndim = ndim1 > ndim2 ? ndim1 : ndim2;
    int* result  = malloc((size_t)max_ndim * sizeof(int));
    if (!result)
        return NULL;

    // Compute output shape from right to left
    for (int i = 0; i < max_ndim; i++) {
        int idx1 = ndim1 - 1 - i;
        int idx2 = ndim2 - 1 - i;

        int dim1 = (idx1 >= 0 && idx1 < ndim1) ? shape1[idx1] : 1;
        int dim2 = (idx2 >= 0 && idx2 < ndim2) ? shape2[idx2] : 1;

        // Output dimension is max of the two (NumPy rule)
        result[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
    }

    *out_ndim = max_ndim;
    return result;
}

/**
 * @brief Broadcast multiple shapes together (NumPy-style)
 *
 * @param shapes Array of shape arrays
 * @param ndims Array of dimension counts
 * @param num_shapes Number of shapes to broadcast
 * @param out_ndim Output: number of dimensions in result
 * @return Output shape array (caller must free), or NULL on failure
 */
int* broadcast_multi_shapes(int** shapes, int* ndims, int num_shapes, int* out_ndim) {
    if (!shapes || !ndims || num_shapes <= 0 || !out_ndim)
        return NULL;

    if (num_shapes == 1) {
        return tensor_shape_copy(shapes[0], ndims[0]);
    }

    // Start with first shape
    int* result      = tensor_shape_copy(shapes[0], ndims[0]);
    int current_ndim = ndims[0];

    if (!result)
        return NULL;

    // Broadcast each subsequent shape
    for (int i = 1; i < num_shapes; i++) {
        int* new_result =
            broadcast_shapes(result, current_ndim, shapes[i], ndims[i], &current_ndim);
        if (!new_result) {
            free(result);
            return NULL;
        }
        free(result);
        result = new_result;
    }

    *out_ndim = current_ndim;
    return result;
}

void tensor_compute_grad_for_broadcast(Tensor* grad_output, int* original_shape, int ndim,
                                       Tensor** grad_input) {
    if (!grad_output || !original_shape || !grad_input)
        return;

    LOG_DEBUG("Computing gradient for broadcast: grad_output ndim=%d, original ndim=%d",
              grad_output->ndim, ndim);

    // Check if shapes match (no broadcasting occurred)
    if (grad_output->ndim == ndim) {
        bool shapes_match = true;
        for (int i = 0; i < ndim; i++) {
            if (grad_output->shape[i] != original_shape[i]) {
                shapes_match = false;
                break;
            }
        }
        if (shapes_match) {
            *grad_input = tensor_clone(grad_output);
            return;
        }
    }

    // Create gradient tensor with original shape
    TensorConfig config = (TensorConfig){.dtype      = grad_output->dtype,
                                         .device     = grad_output->device,
                                         .has_dtype  = true,
                                         .has_device = true};
    *grad_input         = tensor_zeros(original_shape, ndim, &config);
    if (!*grad_input) {
        LOG_ERROR("Failed to create gradient tensor for broadcast");
        return;
    }

    float* grad_out_data = (float*)grad_output->data;
    float* grad_in_data  = (float*)(*grad_input)->data;

    // Compute strides for both tensors
    size_t* out_strides = malloc((size_t)grad_output->ndim * sizeof(size_t));
    size_t* in_strides  = malloc((size_t)ndim * sizeof(size_t));

    if (!out_strides || !in_strides) {
        if (out_strides)
            free(out_strides);
        if (in_strides)
            free(in_strides);
        return;
    }

    // Calculate strides
    out_strides[grad_output->ndim - 1] = 1;
    for (int i = grad_output->ndim - 2; i >= 0; i--) {
        out_strides[i] = out_strides[i + 1] * (size_t)grad_output->shape[i + 1];
    }

    in_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        in_strides[i] = in_strides[i + 1] * (size_t)original_shape[i + 1];
    }

    // Iterate over all elements of grad_output
    for (size_t i = 0; i < grad_output->numel; i++) {
        // Calculate multi-dimensional index for grad_output
        size_t temp   = i;
        size_t in_idx = 0;

        for (int d = grad_output->ndim - 1; d >= 0; d--) {
            size_t coord = temp % (size_t)grad_output->shape[d];
            temp /= (size_t)grad_output->shape[d];

            // Map to input dimension (handle dimension mismatch)
            int in_d = d - (grad_output->ndim - ndim);
            if (in_d >= 0 && in_d < ndim) {
                // If input dimension is 1, always use index 0
                if (original_shape[in_d] == 1) {
                    // Don't add anything (stays at 0)
                } else if (original_shape[in_d] == grad_output->shape[d]) {
                    in_idx += coord * in_strides[in_d];
                }
            }
        }

        // Accumulate gradient
        grad_in_data[in_idx] += grad_out_data[i];
    }

    free(out_strides);
    free(in_strides);

    LOG_DEBUG("Unbroadcast gradient: output numel=%zu, input numel=%zu", grad_output->numel,
              (*grad_input)->numel);
}

void autograd_check_anomaly(Tensor* tensor, const char* operation) {
    if (!tensor || !tensor->data)
        return;

    bool has_nan = false;
    bool has_inf = false;

    for (size_t i = 0; i < tensor->numel; i++) {
        float val = tensor_get_float(tensor, i);
        if (isnan(val))
            has_nan = true;
        if (isinf(val))
            has_inf = true;
    }

    if (has_nan) {
        LOG_ERROR("NaN detected in %s", operation);
    }
    if (has_inf) {
        LOG_ERROR("Inf detected in %s", operation);
    }
}

void autograd_print_graph(Tensor* tensor) {
    if (!tensor)
        return;

    printf("\n=== Autograd Graph ===\n");
    printf("Tensor: %p\n", (void*)tensor);
    printf("Requires grad: %s\n", tensor->requires_grad ? "Yes" : "No");
    printf("Is leaf: %s\n", tensor_is_leaf(tensor) ? "Yes" : "No");
    printf("Has grad: %s\n", tensor->grad ? "Yes" : "No");

    if (tensor->ir_node) {
        const char* op_name = uop_type_to_string(tensor->ir_node->type);
        printf("IR node type: %s\n", op_name ? op_name : "UNKNOWN");
        printf("Number of inputs: %d\n", tensor->ir_node->num_inputs);
    }

    printf("=====================\n\n");
}

#include <inttypes.h>

// simple map using parallel arrays for small graphs
typedef struct {
    const void** keys; // Stores const pointers for comparison only
    int* ids;
    int size;
    int cap;
} PtrIdMap;

static void map_init(PtrIdMap* m) {
    m->keys = NULL;
    m->ids  = NULL;
    m->size = 0;
    m->cap  = 0;
}
static void map_free(PtrIdMap* m) {
    if (m->keys)
        free(m->keys);
    if (m->ids)
        free(m->ids);
}
static int map_get_or_insert(PtrIdMap* m, const void* key, int next_id) {
    for (int i = 0; i < m->size; i++)
        if (m->keys[i] == key)
            return m->ids[i];
    if (m->size >= m->cap) {
        int ncap           = m->cap ? m->cap * 2 : 64;
        const void** nkeys = realloc(m->keys, (size_t)ncap * sizeof(const void*));
        int* nids          = realloc(m->ids, (size_t)ncap * sizeof(int));
        if (!nkeys || !nids)
            return -1;
        m->keys = nkeys;
        m->ids  = nids;
        m->cap  = ncap;
    }
    m->keys[m->size] = key;
    m->ids[m->size]  = next_id;
    m->size++;
    return next_id;
}

static void write_json_escaped(FILE* f, const char* s) {
    if (!s) {
        fputs("null", f);
        return;
    }
    fputc('"', f);
    for (const char* p = s; *p; p++) {
        if (*p == '"' || *p == '\\') {
            fputc('\\', f);
            fputc(*p, f);
        } else if ((unsigned char)*p < 0x20) {
            fprintf(f, "\\u%04x", (unsigned char)*p);
        } else
            fputc(*p, f);
    }
    fputc('"', f);
}

int autograd_export_json(Tensor* root, const char* path) {
    if (!root || !path)
        return -1;

    if (!root->ir_context || !root->ir_node) {
        LOG_WARNING("Cannot export graph: tensor has no IR node (leaf tensor)");
        return -2;
    }

    FILE* f = fopen(path, "wb");
    if (!f)
        return -3;

    PtrIdMap idmap;
    map_init(&idmap);

    // Map to track which nodes are reachable from output (for dead code detection)
    PtrIdMap reachable;
    map_init(&reachable);

    // First pass: assign IDs to ALL nodes in the IR context (including dead code)
    int next_id         = 1;
    struct IRNode* node = root->ir_context->head;
    while (node) {
        map_get_or_insert(&idmap, (const void*)node, next_id++);
        node = node->next;
    }

    // Second pass: mark nodes reachable from output (backward traversal via DFS)
    int stack_cap         = 256;
    struct IRNode** stack = malloc(stack_cap * sizeof(struct IRNode*));
    int stack_size        = 0;
    if (stack) {
        stack[stack_size++] = root->ir_node;
        map_get_or_insert(&reachable, (const void*)root->ir_node, 1);

        while (stack_size > 0) {
            struct IRNode* n = stack[--stack_size];
            for (int j = 0; j < n->num_inputs; j++) {
                if (!n->inputs[j] || !n->inputs[j]->ir_node)
                    continue;
                struct IRNode* input = n->inputs[j]->ir_node;
                int already          = map_get_or_insert(&reachable, (const void*)input, 1);
                if (already == 1) {
                    // Newly added - push to stack
                    if (stack_size >= stack_cap) {
                        stack_cap *= 2;
                        stack = realloc(stack, stack_cap * sizeof(struct IRNode*));
                    }
                    if (stack)
                        stack[stack_size++] = input;
                }
            }
        }
        free(stack);
    }

    // Third pass: export all nodes
    fputs("{\n", f);
    bool first_node   = true;
    bool is_optimized = root->ir_context->is_optimized;

    node = root->ir_context->head;
    while (node) {
        int my_id = map_get_or_insert(&idmap, (const void*)node, 0);

        if (!first_node)
            fputs(",\n", f);
        first_node = false;

        fprintf(f, "  \"%d\": { ", my_id);

        // label
        fputs("\"label\": ", f);
        const char* name = uop_type_to_string(node->type);
        write_json_escaped(f, name);

        // flags - mark as dead if:
        // 1. After optimization: not used flag
        // 2. Before optimization: not reachable from output
        bool is_dead;
        if (is_optimized) {
            is_dead = !node->is_used && node->use_count == 0;
        } else {
            // Check if node is reachable from output
            int reach_check = map_get_or_insert(&reachable, (const void*)node, 0);
            is_dead         = (reach_check == 0); // Not in reachable set = dead
        }

        fprintf(f, ", \"is_dead\": %s", is_dead ? "true" : "false");
        fprintf(f, ", \"is_fused\": %s", node->is_fused ? "true" : "false");

        // Export fusedKernelId for grouping in UI
        fprintf(f, ", \"fusedKernelId\": \"%p\"", (void*)node->fused_kernel);

        // src edges
        fputs(", \"src\": [", f);
        bool first_edge = true;
        for (int j = 0; j < node->num_inputs; j++) {
            if (!node->inputs[j] || !node->inputs[j]->ir_node)
                continue;

            struct IRNode* input_node = node->inputs[j]->ir_node;
            int src_id                = map_get_or_insert(&idmap, (const void*)input_node, 0);

            // Only add edge if input node exists in our map
            if (src_id > 0) {
                if (!first_edge) {
                    fputs(", ", f);
                }
                first_edge = false;
                fprintf(f, "[%d, \"%d\"]", j, src_id);
            }
        }
        fputs("] }", f);

        node = node->next;
    }

    fputs("\n}\n", f);

    fclose(f);
    map_free(&idmap);
    map_free(&reachable);
    return 0;
}

#define MAX_TENSOR_HOOKS 1024

typedef struct TensorHookEntry {
    Tensor* tensor;
    TensorHookFn hook_fn;
    bool active;
} TensorHookEntry;

static TensorHookEntry g_tensor_hooks[MAX_TENSOR_HOOKS];
static int g_tensor_hook_count = 0;

void tensor_register_hook(Tensor* tensor, TensorHookFn hook_fn) {
    if (!tensor || !hook_fn) {
        LOG_ERROR("Cannot register hook: NULL tensor or hook function");
        return;
    }

    if (g_tensor_hook_count >= MAX_TENSOR_HOOKS) {
        LOG_WARNING("Maximum tensor hooks reached (%d), cannot register more", MAX_TENSOR_HOOKS);
        return;
    }

    // Add hook to registry
    g_tensor_hooks[g_tensor_hook_count].tensor  = tensor;
    g_tensor_hooks[g_tensor_hook_count].hook_fn = hook_fn;
    g_tensor_hooks[g_tensor_hook_count].active  = true;
    g_tensor_hook_count++;

    LOG_DEBUG("Registered tensor hook for tensor %p (total hooks: %d)", (void*)tensor,
              g_tensor_hook_count);
}
