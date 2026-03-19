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

static pthread_once_t g_autograd_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_hook_lock = PTHREAD_MUTEX_INITIALIZER;

static void autograd_init_once(void) {
    if (global_autograd_engine) return;

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
    global_autograd_engine->lock_initialized  = false;

    pthread_mutex_init(&global_autograd_engine->lock, NULL);
    global_autograd_engine->lock_initialized = true;

}

void autograd_init(void) {
    pthread_once(&g_autograd_once, autograd_init_once);
}

void autograd_shutdown(void) {
    if (global_autograd_engine) {
        if (global_autograd_engine->lock_initialized) {
            pthread_mutex_destroy(&global_autograd_engine->lock);
            global_autograd_engine->lock_initialized = false;
        }
        free(global_autograd_engine);
        global_autograd_engine = NULL;
        /* Reset pthread_once so re-init is possible (e.g., in tests) */
        g_autograd_once = (pthread_once_t)PTHREAD_ONCE_INIT;
    }
}

AutogradEngine* autograd_get_engine(void) {
    pthread_once(&g_autograd_once, autograd_init_once);
    return global_autograd_engine;
}

void autograd_set_grad_mode(bool enabled) {
    AutogradEngine* engine = autograd_get_engine();
    if (engine->lock_initialized) pthread_mutex_lock(&engine->lock);
    engine->grad_mode = enabled;
    if (engine->lock_initialized) pthread_mutex_unlock(&engine->lock);
}

bool autograd_is_grad_enabled(void) {
    AutogradEngine* engine = autograd_get_engine();
    return engine->grad_mode && engine->enabled;
}

void autograd_no_grad_enter(void) { autograd_set_grad_mode(false); }

void autograd_no_grad_exit(void) { autograd_set_grad_mode(true); }

void autograd_set_anomaly_detection(bool enabled) {
    AutogradEngine* engine = autograd_get_engine();
    if (engine->lock_initialized) pthread_mutex_lock(&engine->lock);
    engine->anomaly_detection = enabled;
    if (engine->lock_initialized) pthread_mutex_unlock(&engine->lock);
    LOG_INFO("Anomaly detection %s", enabled ? "enabled" : "disabled");
}

bool tensor_requires_grad(Tensor* t) { return t && t->requires_grad; }

void tensor_set_requires_grad(Tensor* t, bool requires_grad) {
    if (!t)
        return;
    t->requires_grad = requires_grad;
}

bool tensor_is_leaf(Tensor* t) { return t && (t->ir_node == NULL); }

Tensor* tensor_detach(Tensor* t) {
    if (!t)
        return NULL;

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

    t->requires_grad = false;
    t->ir_node       = NULL;
    t->ir_context    = NULL;
}

void tensor_retain_grad(Tensor* t) {
    if (!t)
        return;

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
    return 0;
}

void tensor_remove_hooks(Tensor* t) {
    if (!t || !t->user_data) {
        return;
    }

    TensorHookList* hooks = (TensorHookList*)t->user_data;
    hooks->num_hooks      = 0;
}

int module_register_backward_hook(struct Module* module, ModuleBackwardHook hook) {
    if (!module || !hook) {
        LOG_ERROR("Invalid arguments to module_register_backward_hook");
        return -1;
    }

    if (!module->user_data) {
        module->user_data = malloc(sizeof(ModuleBackwardHook));
        if (!module->user_data) {
            LOG_ERROR("Failed to allocate memory for module hook");
            return -1;
        }
    }

    *(ModuleBackwardHook*)module->user_data = hook;
    return 0;
}

void tensor_zero_grad(Tensor* tensor) {
    if (!tensor)
        return;

    if (tensor->grad) {
        tensor_free(tensor->grad);
        tensor->grad = NULL;
    }
}

void tensor_accumulate_grad(Tensor* tensor, Tensor* new_grad) {
    if (!tensor || !new_grad)
        return;

    if (!tensor->requires_grad)
        return;

    if (!tensor->grad) {
        tensor->grad = tensor_clone(new_grad);
    } else {
        for (size_t i = 0; i < tensor->grad->numel && i < new_grad->numel; i++) {
            float old_val = tensor_get_float(tensor->grad, i);
            float new_val = tensor_get_float(new_grad, i);
            tensor_set_float(tensor->grad, i, old_val + new_val);
        }
    }

    if (autograd_get_engine()->anomaly_detection) {
        autograd_check_anomaly(tensor->grad, "gradient accumulation");
    }

    /* Propagate gradient to ir_node->output so the next backward node can find it */
    if (tensor->ir_node && tensor->ir_node->output && tensor->ir_node->output != tensor) {
        fflush(stdout);
        if (!tensor->ir_node->output->grad) {
            tensor->ir_node->output->grad = tensor_clone(tensor->grad);
        }
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

    /* Lazy execution: forward pass values needed before computing gradients */
    if (tensor_ensure_executed(tensor) != 0) {
        LOG_ERROR("Failed to execute tensor for backward pass");
        return;
    }

    training_metrics_auto_capture_loss(tensor);

    if (!gradient) {
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

    if (!tensor->grad) {
        tensor->grad = tensor_clone(gradient);
    } else {
        tensor_accumulate_grad(tensor, gradient);
    }

    if (cml_ir_build_backward(tensor->ir_context, tensor->ir_node) != 0) {
        LOG_ERROR("Failed to build backward graph");
        return;
    }

    if (cml_ir_execute_backward(tensor->ir_context) != 0) {
        LOG_ERROR("Failed to execute backward pass");
        LOG_ERROR("Aborting training run to avoid repeated failing executions");
        exit(1);
    }

    const char* viz     = getenv("CML_VIZ");
    const char* viz_env = getenv("VIZ");
    if ((viz && viz[0] != '\0') ||
        (viz_env && (viz_env[0] == '1' || strcmp(viz_env, "true") == 0))) {
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
            tensor_ensure_executed(tensor);
            cml_ir_ensure_gradients_executed(tensor->ir_context);

            tensor->ir_context = NULL;
        }
    } else {
        if (tensor->ir_context) {
            tensor->ir_context = NULL;
        }
    }
}

/* NumPy-style broadcasting: shapes aligned from the right, dims compatible if equal or 1 */
bool tensor_can_broadcast_shapes(int* shape1, int ndim1, int* shape2, int ndim2) {
    if (!shape1 || !shape2)
        return false;

    if (ndim1 == 0 && ndim2 == 0)
        return true;
    if (ndim1 == 0 || ndim2 == 0)
        return true;

    int max_ndim = ndim1 > ndim2 ? ndim1 : ndim2;

    for (int i = 0; i < max_ndim; i++) {
        int idx1 = ndim1 - 1 - i;
        int idx2 = ndim2 - 1 - i;

        int dim1 = (idx1 >= 0 && idx1 < ndim1) ? shape1[idx1] : 1;
        int dim2 = (idx2 >= 0 && idx2 < ndim2) ? shape2[idx2] : 1;

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;
        }

        if (dim1 < 0 || dim2 < 0) {
            return false;
        }
    }

    return true;
}

int* broadcast_shapes(int* shape1, int ndim1, int* shape2, int ndim2, int* out_ndim) {
    if (!shape1 || !shape2 || !out_ndim)
        return NULL;

    if (ndim1 == 0 && ndim2 == 0) {
        *out_ndim = 0;
        return NULL;
    }
    if (ndim1 == 0) {
        int* result = tensor_shape_copy(shape2, ndim2);
        if (result)
            *out_ndim = ndim2;
        return result;
    }
    if (ndim2 == 0) {
        int* result = tensor_shape_copy(shape1, ndim1);
        if (result)
            *out_ndim = ndim1;
        return result;
    }

    int max_ndim = ndim1 > ndim2 ? ndim1 : ndim2;
    int* result  = malloc((size_t)max_ndim * sizeof(int));
    if (!result)
        return NULL;

    for (int i = 0; i < max_ndim; i++) {
        int idx1 = ndim1 - 1 - i;
        int idx2 = ndim2 - 1 - i;

        int dim1 = (idx1 >= 0 && idx1 < ndim1) ? shape1[idx1] : 1;
        int dim2 = (idx2 >= 0 && idx2 < ndim2) ? shape2[idx2] : 1;

        result[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
    }

    *out_ndim = max_ndim;
    return result;
}

int* broadcast_multi_shapes(int** shapes, int* ndims, int num_shapes, int* out_ndim) {
    if (!shapes || !ndims || num_shapes <= 0 || !out_ndim)
        return NULL;

    if (num_shapes == 1) {
        return tensor_shape_copy(shapes[0], ndims[0]);
    }

    int* result      = tensor_shape_copy(shapes[0], ndims[0]);
    int current_ndim = ndims[0];

    if (!result)
        return NULL;

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

    size_t* out_strides = malloc((size_t)grad_output->ndim * sizeof(size_t));
    size_t* in_strides  = malloc((size_t)ndim * sizeof(size_t));

    if (!out_strides || !in_strides) {
        if (out_strides)
            free(out_strides);
        if (in_strides)
            free(in_strides);
        return;
    }

    out_strides[grad_output->ndim - 1] = 1;
    for (int i = grad_output->ndim - 2; i >= 0; i--) {
        out_strides[i] = out_strides[i + 1] * (size_t)grad_output->shape[i + 1];
    }

    in_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        in_strides[i] = in_strides[i + 1] * (size_t)original_shape[i + 1];
    }

    for (size_t i = 0; i < grad_output->numel; i++) {
        size_t temp   = i;
        size_t in_idx = 0;

        for (int d = grad_output->ndim - 1; d >= 0; d--) {
            size_t coord = temp % (size_t)grad_output->shape[d];
            temp /= (size_t)grad_output->shape[d];

            int in_d = d - (grad_output->ndim - ndim);
            if (in_d >= 0 && in_d < ndim) {
                if (original_shape[in_d] == 1) {
                    /* broadcast dim: accumulate at index 0 */
                } else if (original_shape[in_d] == grad_output->shape[d]) {
                    in_idx += coord * in_strides[in_d];
                }
            }
        }

        grad_in_data[in_idx] += grad_out_data[i];
    }

    free(out_strides);
    free(in_strides);

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

    printf("\nAutograd Graph\n");
    printf("Tensor: %p\n", (void*)tensor);
    printf("Requires grad: %s\n", tensor->requires_grad ? "Yes" : "No");
    printf("Is leaf: %s\n", tensor_is_leaf(tensor) ? "Yes" : "No");
    printf("Has grad: %s\n", tensor->grad ? "Yes" : "No");

    if (tensor->ir_node) {
        const char* op_name = uop_type_to_string(tensor->ir_node->type);
        printf("IR node type: %s\n", op_name ? op_name : "UNKNOWN");
        printf("Number of inputs: %d\n", tensor->ir_node->num_inputs);
    }

    printf("\n");
}

#include <inttypes.h>

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

    PtrIdMap reachable;
    map_init(&reachable);

    int next_id         = 1;
    struct IRNode* node = root->ir_context->head;
    while (node) {
        map_get_or_insert(&idmap, (const void*)node, next_id++);
        node = node->next;
    }

    int stack_cap         = 256;
    struct IRNode** stack = malloc(stack_cap * sizeof(struct IRNode*));
    int stack_size        = 0;
    if (stack) {
        stack[stack_size++] = root->ir_node;
        map_get_or_insert(&reachable, (const void*)root->ir_node, 1);

        bool stack_ok = true;
        while (stack_size > 0 && stack_ok) {
            struct IRNode* n = stack[--stack_size];
            for (int j = 0; j < n->num_inputs; j++) {
                if (!n->inputs[j] || !n->inputs[j]->ir_node)
                    continue;
                struct IRNode* input = n->inputs[j]->ir_node;
                int already          = map_get_or_insert(&reachable, (const void*)input, 1);
                if (already == 1) {
                    if (stack_size >= stack_cap) {
                        stack_cap *= 2;
                        struct IRNode** new_stack = realloc(stack, stack_cap * sizeof(struct IRNode*));
                        if (!new_stack) {
                            stack_ok = false;
                            break;
                        }
                        stack = new_stack;
                    }
                    stack[stack_size++] = input;
                }
            }
        }
        free(stack);
    }

    fputs("{\n", f);
    bool first_node   = true;
    bool is_optimized = root->ir_context->is_optimized;

    node = root->ir_context->head;
    while (node) {
        int my_id = map_get_or_insert(&idmap, (const void*)node, 0);
        if (my_id < 0) {
            LOG_ERROR("map_get_or_insert failed in autograd_export_json");
            fclose(f);
            map_free(&idmap);
            map_free(&reachable);
            return -4;
        }

        if (!first_node)
            fputs(",\n", f);
        first_node = false;

        fprintf(f, "  \"%d\": { ", my_id);

        fputs("\"label\": ", f);
        const char* name = uop_type_to_string(node->type);
        write_json_escaped(f, name);

        bool is_dead;
        if (is_optimized) {
            is_dead = !node->is_used && node->use_count == 0;
        } else {
            int reach_check = map_get_or_insert(&reachable, (const void*)node, 0);
            is_dead         = (reach_check == 0);
        }

        fprintf(f, ", \"is_dead\": %s", is_dead ? "true" : "false");
        fprintf(f, ", \"is_fused\": %s", node->is_fused ? "true" : "false");

        fprintf(f, ", \"fusedKernelId\": \"%p\"", (void*)node->fused_kernel);

        fputs(", \"src\": [", f);
        bool first_edge = true;
        for (int j = 0; j < node->num_inputs; j++) {
            if (!node->inputs[j] || !node->inputs[j]->ir_node)
                continue;

            struct IRNode* input_node = node->inputs[j]->ir_node;
            int src_id                = map_get_or_insert(&idmap, (const void*)input_node, 0);

            if (src_id >= 0 && src_id > 0) {
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

    pthread_mutex_lock(&g_hook_lock);

    if (g_tensor_hook_count >= MAX_TENSOR_HOOKS) {
        pthread_mutex_unlock(&g_hook_lock);
        LOG_WARNING("Maximum tensor hooks reached (%d), cannot register more", MAX_TENSOR_HOOKS);
        return;
    }

    g_tensor_hooks[g_tensor_hook_count].tensor  = tensor;
    g_tensor_hooks[g_tensor_hook_count].hook_fn = hook_fn;
    g_tensor_hooks[g_tensor_hook_count].active  = true;
    g_tensor_hook_count++;

    pthread_mutex_unlock(&g_hook_lock);
}
