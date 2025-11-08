#include "autograd/autograd.h"
#include "Core/training_metrics.h"
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

    global_autograd_engine = CM_MALLOC(sizeof(AutogradEngine));
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
        CM_FREE(global_autograd_engine);
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

AutogradContext* autograd_context_create(void) {
    AutogradContext* ctx = CM_MALLOC(sizeof(AutogradContext));
    if (!ctx) {
        LOG_ERROR("Failed to allocate autograd context");
        return NULL;
    }

    ctx->saved_tensors     = NULL;
    ctx->num_saved_tensors = 0;
    ctx->capacity          = 0;
    ctx->saved_data        = NULL;
    ctx->saved_data_size   = 0;
    ctx->saved_shape       = NULL;
    ctx->saved_ndim        = 0;
    ctx->custom_data       = NULL;
    ctx->custom_free_fn    = NULL;

    return ctx;
}

void autograd_context_free(AutogradContext* ctx) {
    if (!ctx)
        return;

    if (ctx->saved_tensors) {
        CM_FREE(ctx->saved_tensors);
    }

    if (ctx->saved_data) {
        CM_FREE(ctx->saved_data);
    }

    if (ctx->saved_shape) {
        CM_FREE(ctx->saved_shape);
    }

    if (ctx->custom_data && ctx->custom_free_fn) {
        ctx->custom_free_fn(ctx->custom_data);
    }

    CM_FREE(ctx);
}

void autograd_context_save_for_backward(AutogradContext* ctx, Tensor** tensors, int num_tensors) {
    if (!ctx || !tensors || num_tensors <= 0)
        return;

    ctx->saved_tensors = CM_MALLOC(num_tensors * sizeof(Tensor*));
    if (!ctx->saved_tensors) {
        LOG_ERROR("Failed to allocate saved tensors array");
        return;
    }

    memcpy(ctx->saved_tensors, tensors, num_tensors * sizeof(Tensor*));
    ctx->num_saved_tensors = num_tensors;
    ctx->capacity          = num_tensors;

    LOG_DEBUG("Saved %d tensors for backward", num_tensors);
}

void autograd_context_save_shape(AutogradContext* ctx, int* shape, int ndim) {
    if (!ctx || !shape || ndim <= 0)
        return;

    ctx->saved_shape = CM_MALLOC(ndim * sizeof(int));
    if (!ctx->saved_shape) {
        LOG_ERROR("Failed to allocate saved shape");
        return;
    }

    memcpy(ctx->saved_shape, shape, ndim * sizeof(int));
    ctx->saved_ndim = ndim;
}

void autograd_context_save_data(AutogradContext* ctx, void* data, size_t size) {
    if (!ctx || !data || size == 0)
        return;

    ctx->saved_data = CM_MALLOC(size);
    if (!ctx->saved_data) {
        LOG_ERROR("Failed to allocate saved data");
        return;
    }

    memcpy(ctx->saved_data, data, size);
    ctx->saved_data_size = size;
}

Tensor* autograd_context_get_saved_tensor(AutogradContext* ctx, int index) {
    if (!ctx || index < 0 || index >= ctx->num_saved_tensors) {
        LOG_ERROR("Invalid saved tensor index: %d", index);
        return NULL;
    }
    return ctx->saved_tensors[index];
}

static int next_sequence_nr = 0;

Function* autograd_function_create(OpType op_type, const char* name) {
    Function* fn = CM_MALLOC(sizeof(Function));
    if (!fn) {
        LOG_ERROR("Failed to allocate Function");
        return NULL;
    }

    fn->op_type     = op_type;
    fn->op_name     = name ? strdup(name) : NULL;
    fn->inputs      = NULL;
    fn->num_inputs  = 0;
    fn->ctx         = autograd_context_create();
    fn->backward_fn = NULL;
    fn->sequence_nr = next_sequence_nr++;
    fn->ref_count   = 1;

    memset(fn->needs_input_grad, 0, sizeof(fn->needs_input_grad));

    LOG_DEBUG("Created function '%s' (op_type=%d, seq=%d)", name ? name : "unnamed", op_type,
              fn->sequence_nr);

    return fn;
}

void autograd_function_free(Function* fn) {
    if (!fn)
        return;

    fn->ref_count--;
    if (fn->ref_count > 0)
        return;

    LOG_DEBUG("Freeing function '%s' (seq=%d)", fn->op_name ? fn->op_name : "unnamed",
              fn->sequence_nr);

    if (fn->op_name) {
        free(fn->op_name);
    }

    if (fn->inputs) {
        CM_FREE(fn->inputs);
    }

    if (fn->ctx) {
        autograd_context_free(fn->ctx);
    }

    CM_FREE(fn);
}

void autograd_function_set_backward(Function* fn, BackwardFn backward_fn) {
    if (fn) {
        fn->backward_fn = backward_fn;
    }
}

void autograd_function_set_inputs(Function* fn, Tensor** inputs, int num_inputs) {
    if (!fn || !inputs || num_inputs <= 0)
        return;

    fn->inputs = CM_MALLOC(num_inputs * sizeof(Tensor*));
    if (!fn->inputs) {
        LOG_ERROR("Failed to allocate inputs array");
        return;
    }

    memcpy(fn->inputs, inputs, num_inputs * sizeof(Tensor*));
    fn->num_inputs = num_inputs;

    // Determine which inputs need gradients
    for (int i = 0; i < num_inputs && i < 8; i++) {
        fn->needs_input_grad[i] = inputs[i] && inputs[i]->requires_grad;
    }
}

bool tensor_requires_grad(Tensor* t) { return t && t->requires_grad; }

void tensor_set_requires_grad(Tensor* t, bool requires_grad) {
    if (!t)
        return;
    t->requires_grad = requires_grad;
    LOG_DEBUG("Set requires_grad=%s for tensor %p", requires_grad ? "true" : "false", (void*)t);
}

bool tensor_is_leaf(Tensor* t) { return t && (t->grad_fn == NULL); }

Tensor* tensor_detach(Tensor* t) {
    if (!t)
        return NULL;

    LOG_DEBUG("Detaching tensor %p", (void*)t);

    // Create a new tensor that shares data but has no grad_fn
    Tensor* detached = tensor_clone(t);
    if (!detached)
        return NULL;

    detached->requires_grad = false;
    detached->grad_fn       = NULL;
    detached->grad          = NULL;

    return detached;
}

void tensor_detach_(Tensor* t) {
    if (!t)
        return;

    LOG_DEBUG("In-place detaching tensor %p", (void*)t);

    t->requires_grad = false;
    if (t->grad_fn) {
        autograd_function_free(t->grad_fn);
        t->grad_fn = NULL;
    }
}

void tensor_retain_grad(Tensor* t) {
    if (!t)
        return;

    // For non-leaf tensors, we need to keep gradients
    // This is done by marking requires_grad and optionally registering hooks
    LOG_DEBUG("Retaining grad for tensor %p", (void*)t);

    // Mark tensor to retain gradients
    t->requires_grad = true;

    // Note: If you need to preserve intermediate gradients during backward,
    // you can register a hook using tensor_register_hook() to capture them
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
}

Tensor* tensor_get_grad(Tensor* tensor) { return tensor ? tensor->grad : NULL; }

BackwardGraph* backward_graph_create(void) {
    BackwardGraph* graph = CM_MALLOC(sizeof(BackwardGraph));
    if (!graph) {
        LOG_ERROR("Failed to allocate backward graph");
        return NULL;
    }

    graph->nodes     = NULL;
    graph->num_nodes = 0;
    graph->capacity  = 0;

    return graph;
}

void backward_graph_free(BackwardGraph* graph) {
    if (!graph)
        return;

    if (graph->nodes) {
        for (int i = 0; i < graph->num_nodes; i++) {
            if (graph->nodes[i]) {
                CM_FREE(graph->nodes[i]);
            }
        }
        CM_FREE(graph->nodes);
    }

    CM_FREE(graph);
}

void backward_graph_add_node(BackwardGraph* graph, Tensor* tensor, Tensor* grad) {
    if (!graph || !tensor)
        return;

    // Expand capacity if needed
    if (graph->num_nodes >= graph->capacity) {
        int new_capacity         = graph->capacity == 0 ? 16 : graph->capacity * 2;
        BackwardNode** new_nodes = CM_REALLOC(graph->nodes, new_capacity * sizeof(BackwardNode*));
        if (!new_nodes) {
            LOG_ERROR("Failed to expand backward graph capacity");
            return;
        }
        graph->nodes    = new_nodes;
        graph->capacity = new_capacity;
    }

    BackwardNode* node = CM_MALLOC(sizeof(BackwardNode));
    if (!node) {
        LOG_ERROR("Failed to allocate backward node");
        return;
    }

    node->tensor  = tensor;
    node->grad    = grad;
    node->grad_fn = tensor->grad_fn;
    node->depth   = 0;
    node->visited = false;
    node->next    = NULL;

    graph->nodes[graph->num_nodes++] = node;
}

static int compare_depth(const void* a, const void* b) {
    BackwardNode* node_a = *(BackwardNode**)a;
    BackwardNode* node_b = *(BackwardNode**)b;
    return node_b->depth - node_a->depth; // Descending order
}

static void build_backward_graph_recursive(BackwardGraph* graph, Tensor* tensor, int depth) {
    if (!tensor)
        return;

    // Stop at leaf tensors (no grad_fn) - these are parameters or inputs
    // But we still need to process them if they have requires_grad
    if (!tensor->grad_fn) {
        // This is a leaf tensor (parameter or input)
        // Check if already visited
        for (int i = 0; i < graph->num_nodes; i++) {
            if (graph->nodes[i]->tensor == tensor) {
                return;
            }
        }
        // Add leaf tensor to graph if it requires gradients
        if (tensor->requires_grad) {
            backward_graph_add_node(graph, tensor, NULL);
            if (graph->num_nodes > 0) {
                graph->nodes[graph->num_nodes - 1]->depth = depth;
            }
        }
        return;
    }

    // Check if already visited
    for (int i = 0; i < graph->num_nodes; i++) {
        if (graph->nodes[i]->tensor == tensor) {
            // Update depth if this path is deeper
            if (depth > graph->nodes[i]->depth) {
                graph->nodes[i]->depth = depth;
            }
            return;
        }
    }

    // Add this tensor to graph
    backward_graph_add_node(graph, tensor, NULL);
    if (graph->num_nodes > 0) {
        graph->nodes[graph->num_nodes - 1]->depth = depth;
    }

    // Recursively add parent tensors
    Function* fn = tensor->grad_fn;
    if (fn && fn->inputs) {
        LOG_DEBUG("Building backward graph: tensor %p has grad_fn '%s' with %d inputs",
                  (void*)tensor, fn->op_name ? fn->op_name : "unnamed", fn->num_inputs);
        for (int i = 0; i < fn->num_inputs; i++) {
            if (fn->inputs[i] && fn->inputs[i]->requires_grad) {
                LOG_DEBUG("  Adding input %d: tensor %p (requires_grad=%d, has_grad_fn=%s)", i,
                          (void*)fn->inputs[i], fn->inputs[i]->requires_grad,
                          fn->inputs[i]->grad_fn ? "yes" : "no");
                build_backward_graph_recursive(graph, fn->inputs[i], depth + 1);
            } else if (fn->inputs[i]) {
                LOG_DEBUG("  Skipping input %d: tensor %p (requires_grad=%d)", i,
                          (void*)fn->inputs[i], fn->inputs[i]->requires_grad);
            }
        }
    }
}

void backward_graph_execute(BackwardGraph* graph, bool retain_graph) {
    if (!graph || graph->num_nodes == 0)
        return;

    (void)retain_graph; // Reserved for future use
    LOG_DEBUG("Executing backward graph with %d nodes", graph->num_nodes);

    // Sort nodes by depth (topological order) - deepest first
    qsort(graph->nodes, graph->num_nodes, sizeof(BackwardNode*), compare_depth);

    // Track which nodes have been processed
    bool* processed = CM_CALLOC(graph->num_nodes, sizeof(bool));
    if (!processed) {
        LOG_ERROR("Failed to allocate processed tracking array");
        return;
    }

    // Iterative execution: keep processing until no more gradients are propagated
    bool progress      = true;
    int iterations     = 0;
    int max_iterations = graph->num_nodes * graph->num_nodes + 10; // Safety limit (more generous)

    while (progress && iterations < max_iterations) {
        progress = false;
        iterations++;

        LOG_INFO("Backward pass iteration %d", iterations);

        // Process all nodes that have gradients and haven't been processed
        for (int i = 0; i < graph->num_nodes; i++) {
            if (processed[i])
                continue;

            BackwardNode* node = graph->nodes[i];
            if (!node || !node->tensor) {
                processed[i] = true;
                continue;
            }

            // Leaf tensors (parameters) don't have grad_fn - they just receive gradients
            if (!node->grad_fn) {
                // This is a leaf tensor (parameter) - check if it has a gradient
                if (node->tensor->grad) {
                    LOG_INFO("Leaf tensor %p has gradient (parameter)", (void*)node->tensor);
                } else {
                    LOG_INFO("Leaf tensor %p does NOT have gradient yet (parameter)",
                             (void*)node->tensor);
                }
                processed[i] = true;
                continue;
            }

            // Check if this node has a gradient
            if (!node->tensor->grad) {
                continue; // Wait for gradient to be set by children
            }

            Function* fn = node->grad_fn;

            LOG_INFO("Processing node %d: op='%s' (seq=%d) tensor=%p", i,
                     fn->op_name ? fn->op_name : "unnamed", fn->sequence_nr, (void*)node->tensor);

            // Call backward function - this will accumulate gradients into inputs
            if (fn->backward_fn) {
                fn->backward_fn(fn, node->tensor->grad);
                processed[i] = true;
                progress     = true;

                LOG_DEBUG("Processed backward for '%s', marked progress",
                          fn->op_name ? fn->op_name : "unnamed");

                // After backward, check if any input tensors now have gradients
                // and need to be processed (this handles chains like matmul -> transpose -> weight)
                if (fn->inputs) {
                    for (int j = 0; j < fn->num_inputs; j++) {
                        if (fn->inputs[j] && fn->inputs[j]->requires_grad && fn->inputs[j]->grad) {
                            // Check if this input tensor is in the graph
                            bool found = false;
                            for (int k = 0; k < graph->num_nodes; k++) {
                                if (graph->nodes[k]->tensor == fn->inputs[j]) {
                                    found = true;
                                    // If it has a grad_fn and hasn't been processed, ensure it will
                                    // be processed
                                    if (fn->inputs[j]->grad_fn && !processed[k]) {
                                        LOG_INFO(
                                            "Input tensor %p (from '%s') now has gradient and "
                                            "grad_fn '%s', will be processed in next iteration",
                                            (void*)fn->inputs[j],
                                            fn->op_name ? fn->op_name : "unnamed",
                                            fn->inputs[j]->grad_fn->op_name
                                                ? fn->inputs[j]->grad_fn->op_name
                                                : "unnamed");
                                        // Don't mark as processed yet - let it be processed in next
                                        // iteration
                                        progress = true; // Ensure we continue processing
                                    }
                                    break;
                                }
                            }
                            if (!found && fn->inputs[j]->grad_fn) {
                                // This tensor is not in the graph but has grad_fn - add it
                                LOG_INFO(
                                    "Adding missing tensor %p to backward graph (has grad_fn '%s')",
                                    (void*)fn->inputs[j],
                                    fn->inputs[j]->grad_fn->op_name
                                        ? fn->inputs[j]->grad_fn->op_name
                                        : "unnamed");
                                // Recursively add it and its inputs to the graph
                                build_backward_graph_recursive(graph, fn->inputs[j],
                                                               node->depth + 1);
                                // Re-sort graph by depth after adding new nodes
                                qsort(graph->nodes, graph->num_nodes, sizeof(BackwardNode*),
                                      compare_depth);
                                // Reallocate processed array
                                bool* new_processed =
                                    CM_REALLOC(processed, graph->num_nodes * sizeof(bool));
                                if (new_processed) {
                                    processed = new_processed;
                                    // Initialize new entries to false
                                    for (int k = graph->num_nodes - 1; k >= 0; k--) {
                                        // Find which node was just added by checking tensor pointer
                                        if (graph->nodes[k]->tensor == fn->inputs[j]) {
                                            processed[k] = false; // Mark new node as unprocessed
                                            break;
                                        }
                                    }
                                }
                                progress = true; // Ensure we continue processing
                            }
                        }
                    }
                }
            } else {
                LOG_WARNING("Node %d has no backward function", i);
                processed[i] = true;
            }
        }
    }

    if (iterations >= max_iterations) {
        LOG_WARNING("Backward pass reached maximum iterations (%d), may be incomplete",
                    max_iterations);
    }

    CM_FREE(processed);
    LOG_INFO("Backward pass completed in %d iterations", iterations);
}

void tensor_backward(Tensor* tensor, Tensor* gradient, bool retain_graph, bool create_graph) {
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

    // Auto-capture loss (if this is a scalar loss tensor)
    training_metrics_auto_capture_loss(tensor);

    LOG_INFO("Starting backward pass for tensor %p", (void*)tensor);

    // Initialize gradient if not provided
    if (!gradient) {
        // For scalar tensors, initialize gradient to ones
        gradient = tensor_ones(tensor->shape, tensor->ndim, tensor->dtype, tensor->device);
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

    // Build backward graph
    BackwardGraph* graph = backward_graph_create();
    if (!graph) {
        LOG_ERROR("Failed to create backward graph");
        return;
    }

    build_backward_graph_recursive(graph, tensor, 0);

    LOG_INFO("Built backward graph with %d nodes", graph->num_nodes);
    for (int i = 0; i < graph->num_nodes; i++) {
        BackwardNode* node = graph->nodes[i];
        if (node && node->tensor) {
            LOG_INFO("  Node %d: tensor %p, has_grad_fn=%s, requires_grad=%d, depth=%d", i,
                     (void*)node->tensor, node->grad_fn ? "yes" : "no", node->tensor->requires_grad,
                     node->depth);
        }
    }

    // Execute backward pass
    AutogradEngine* engine = autograd_get_engine();
    bool prev_create_graph = engine->create_graph;
    engine->create_graph   = create_graph;

    backward_graph_execute(graph, retain_graph);

    engine->create_graph = prev_create_graph;

    // Clean up
    backward_graph_free(graph);

    LOG_INFO("Backward pass completed for tensor %p", (void*)tensor);

    const char* viz     = getenv("CML_VIZ");
    const char* viz_env = getenv("VIZ");
    if ((viz && viz[0] != '\0') ||
        (viz_env && (viz_env[0] == '1' || strcmp(viz_env, "true") == 0))) {
        const char* out_path = "viz-ui/public/graph.json";
        int rc               = autograd_export_json(tensor, out_path);
        if (rc != 0) {
            LOG_WARNING("CML_VIZ export failed rc=%d", rc);
        } else {
            LOG_INFO("CML_VIZ exported graph to %s", out_path);
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
bool can_broadcast_shapes(int* shape1, int ndim1, int* shape2, int ndim2) {
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
    int* result  = CM_MALLOC(max_ndim * sizeof(int));
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
            CM_FREE(result);
            return NULL;
        }
        CM_FREE(result);
        result = new_result;
    }

    *out_ndim = current_ndim;
    return result;
}

void compute_grad_for_broadcast(Tensor* grad_output, int* original_shape, int ndim,
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
    *grad_input = tensor_zeros(original_shape, ndim, grad_output->dtype, grad_output->device);
    if (!*grad_input) {
        LOG_ERROR("Failed to create gradient tensor for broadcast");
        return;
    }

    float* grad_out_data = (float*)grad_output->data;
    float* grad_in_data  = (float*)(*grad_input)->data;

    // Compute strides for both tensors
    size_t* out_strides = CM_MALLOC(grad_output->ndim * sizeof(size_t));
    size_t* in_strides  = CM_MALLOC(ndim * sizeof(size_t));

    if (!out_strides || !in_strides) {
        if (out_strides)
            CM_FREE(out_strides);
        if (in_strides)
            CM_FREE(in_strides);
        return;
    }

    // Calculate strides
    out_strides[grad_output->ndim - 1] = 1;
    for (int i = grad_output->ndim - 2; i >= 0; i--) {
        out_strides[i] = out_strides[i + 1] * grad_output->shape[i + 1];
    }

    in_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        in_strides[i] = in_strides[i + 1] * original_shape[i + 1];
    }

    // Iterate over all elements of grad_output
    for (size_t i = 0; i < grad_output->numel; i++) {
        // Calculate multi-dimensional index for grad_output
        size_t temp   = i;
        size_t in_idx = 0;

        for (int d = grad_output->ndim - 1; d >= 0; d--) {
            size_t coord = temp % grad_output->shape[d];
            temp /= grad_output->shape[d];

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

    CM_FREE(out_strides);
    CM_FREE(in_strides);

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

const char* op_type_to_string(OpType op) {
    switch (op) {
    case OP_NONE:
        return "None";
    case OP_ADD:
        return "Add";
    case OP_SUB:
        return "Sub";
    case OP_MUL:
        return "Mul";
    case OP_DIV:
        return "Div";
    case OP_POW:
        return "Pow";
    case OP_MATMUL:
        return "MatMul";
    case OP_NEG:
        return "Neg";
    case OP_EXP:
        return "Exp";
    case OP_LOG:
        return "Log";
    case OP_SQRT:
        return "Sqrt";
    case OP_SIN:
        return "Sin";
    case OP_COS:
        return "Cos";
    case OP_TAN:
        return "Tan";
    case OP_TANH:
        return "Tanh";
    case OP_RELU:
        return "ReLU";
    case OP_SIGMOID:
        return "Sigmoid";
    case OP_SOFTMAX:
        return "Softmax";
    case OP_SUM:
        return "Sum";
    case OP_MEAN:
        return "Mean";
    case OP_MAX:
        return "Max";
    case OP_MIN:
        return "Min";
    case OP_MSE_LOSS:
        return "MSELoss";
    case OP_MAE_LOSS:
        return "MAELoss";
    case OP_BCE_LOSS:
        return "BCELoss";
    case OP_CROSS_ENTROPY_LOSS:
        return "CrossEntropyLoss";
    case OP_CLONE:
        return "Clone";
    case OP_DETACH:
        return "Detach";
    default:
        return "Unknown";
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

    if (tensor->grad_fn) {
        Function* fn = tensor->grad_fn;
        printf("Grad function: %s (seq=%d)\n",
               fn->op_name ? fn->op_name : op_type_to_string(fn->op_type), fn->sequence_nr);
        printf("Number of inputs: %d\n", fn->num_inputs);
    }

    printf("=====================\n\n");
}

#include <inttypes.h>
static const char* op_color(OpType op) {
    switch (op) {
    case OP_ADD:
    case OP_SUB:
    case OP_MUL:
    case OP_DIV:
    case OP_POW:
        return "#f6ccff"; // ALU
    case OP_MATMUL:
        return "#ffc0c0"; // GEMM-ish
    case OP_RELU:
    case OP_SIGMOID:
    case OP_TANH:
        return "#cef263"; // activation
    case OP_SUM:
    case OP_MEAN:
        return "#FF5B5B"; // reduce
    default:
        return "#e0e0e0";
    }
}

// simple map using parallel arrays for small graphs
typedef struct {
    void** keys;
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
        CM_FREE(m->keys);
    if (m->ids)
        CM_FREE(m->ids);
}
static int map_get_or_insert(PtrIdMap* m, const void* key, int next_id) {
    for (int i = 0; i < m->size; i++)
        if (m->keys[i] == key)
            return m->ids[i];
    if (m->size >= m->cap) {
        int ncap     = m->cap ? m->cap * 2 : 64;
        void** nkeys = CM_REALLOC(m->keys, ncap * sizeof(void*));
        int* nids    = CM_REALLOC(m->ids, ncap * sizeof(int));
        if (!nkeys || !nids)
            return -1;
        m->keys = nkeys;
        m->ids  = nids;
        m->cap  = ncap;
    }
    m->keys[m->size] = (void*)key;
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
    BackwardGraph* graph = backward_graph_create();
    if (!graph)
        return -2;
    build_backward_graph_recursive(graph, root, 0);
    FILE* f = fopen(path, "wb");
    if (!f) {
        backward_graph_free(graph);
        return -3;
    }

    PtrIdMap idmap;
    map_init(&idmap);
    // assign ids to each tensor node via its grad_fn (function focus)
    int next_id = 1;
    for (int i = 0; i < graph->num_nodes; i++) {
        BackwardNode* n = graph->nodes[i];
        if (n && n->tensor && n->tensor->grad_fn) {
            map_get_or_insert(&idmap, (const void*)n->tensor->grad_fn, next_id++);
        }
    }

    fputs("{\n", f);
    bool first = true;
    for (int i = 0; i < graph->num_nodes; i++) {
        BackwardNode* n = graph->nodes[i];
        if (!n || !n->tensor || !n->tensor->grad_fn)
            continue;
        Function* fn = n->tensor->grad_fn;
        int myid     = map_get_or_insert(&idmap, (const void*)fn, next_id++);
        if (!first)
            fputs(",\n", f);
        first = false;
        fprintf(f, "  \"%d\": { ", myid);
        // label
        fputs("\"label\": ", f);
        const char* name = fn->op_name ? fn->op_name : op_type_to_string(fn->op_type);
        write_json_escaped(f, name);
        // color
        fprintf(f, ", \"color\": ");
        write_json_escaped(f, op_color(fn->op_type));
        // src edges
        fputs(", \"src\": [", f);
        bool first_edge = true;
        for (int j = 0; j < fn->num_inputs; j++) {
            Tensor* inp = fn->inputs[j];
            if (!inp)
                continue;
            if (!inp->grad_fn)
                continue; // leaf -> omit for now
            int srcid = map_get_or_insert(&idmap, (const void*)inp->grad_fn, next_id++);
            if (!first_edge) {
                fputs(", ", f);
            }
            first_edge = false;
            fprintf(f, "[%d, \"%d\"]", j, srcid);
        }
        fputs("] }", f);
    }
    fputs("\n}\n", f);

    fclose(f);
    map_free(&idmap);
    backward_graph_free(graph);
    return 0;
}

#define MAX_TENSOR_HOOKS 1024
#define MAX_FUNCTION_HOOKS 512

typedef struct TensorHookEntry {
    Tensor* tensor;
    TensorHookFn hook_fn;
    bool active;
} TensorHookEntry;

typedef struct FunctionHookEntry {
    Function* fn;
    BackwardHookFn hook_fn;
    bool active;
} FunctionHookEntry;

static TensorHookEntry g_tensor_hooks[MAX_TENSOR_HOOKS];
static FunctionHookEntry g_function_hooks[MAX_FUNCTION_HOOKS];
static int g_tensor_hook_count   = 0;
static int g_function_hook_count = 0;

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

void tensor_remove_hooks(Tensor* tensor) {
    if (!tensor) {
        LOG_WARNING("Cannot remove hooks: NULL tensor");
        return;
    }

    int removed = 0;

    // Deactivate all hooks for this tensor
    for (int i = 0; i < g_tensor_hook_count; i++) {
        if (g_tensor_hooks[i].active && g_tensor_hooks[i].tensor == tensor) {
            g_tensor_hooks[i].active = false;
            removed++;
        }
    }

    // Compact the array by removing inactive hooks
    int write_idx = 0;
    for (int read_idx = 0; read_idx < g_tensor_hook_count; read_idx++) {
        if (g_tensor_hooks[read_idx].active) {
            if (write_idx != read_idx) {
                g_tensor_hooks[write_idx] = g_tensor_hooks[read_idx];
            }
            write_idx++;
        }
    }
    g_tensor_hook_count = write_idx;

    LOG_DEBUG("Removed %d tensor hooks for tensor %p (remaining hooks: %d)", removed, (void*)tensor,
              g_tensor_hook_count);
}

void function_register_hook(Function* fn, BackwardHookFn hook_fn) {
    if (!fn || !hook_fn) {
        LOG_ERROR("Cannot register hook: NULL function or hook function");
        return;
    }

    if (g_function_hook_count >= MAX_FUNCTION_HOOKS) {
        LOG_WARNING("Maximum function hooks reached (%d), cannot register more",
                    MAX_FUNCTION_HOOKS);
        return;
    }

    // Add hook to registry
    g_function_hooks[g_function_hook_count].fn      = fn;
    g_function_hooks[g_function_hook_count].hook_fn = hook_fn;
    g_function_hooks[g_function_hook_count].active  = true;
    g_function_hook_count++;

    LOG_DEBUG("Registered function hook for function %p (total hooks: %d)", (void*)fn,
              g_function_hook_count);
}

void autograd_execute_tensor_hooks(Tensor* tensor, Tensor* grad) {
    if (!tensor || !grad)
        return;

    for (int i = 0; i < g_tensor_hook_count; i++) {
        if (g_tensor_hooks[i].active && g_tensor_hooks[i].tensor == tensor) {
            LOG_DEBUG("Executing tensor hook %d for tensor %p", i, (void*)tensor);
            g_tensor_hooks[i].hook_fn(grad);
        }
    }
}

// Helper function to execute function hooks
void autograd_execute_function_hooks(Function* fn, Tensor* grad_input, Tensor* grad_output) {
    if (!fn)
        return;

    for (int i = 0; i < g_function_hook_count; i++) {
        if (g_function_hooks[i].active && g_function_hooks[i].fn == fn) {
            LOG_DEBUG("Executing function hook %d for function %p", i, (void*)fn);
            g_function_hooks[i].hook_fn(fn, grad_input, grad_output);
        }
    }
}
