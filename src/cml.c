#include "cml.h"
#include "core/logging.h"
#include "core/training_metrics.h"
#include "core/error_stack.h"
#include "core/cleanup.h"
#include "core/model_architecture.h"
#include "backend/device.h"
#include "core/config.h"
#include "core/computation_graph.h"
#include "core/graph_context.h"
#include "nn.h"
#include "nn/layers.h"
#include "nn/layers/sequential.h"
#include "nn/layers/linear.h"
#include "nn/layers/instancenorm.h"
#include "tensor/tensor.h"
#include "tensor/tensor_manipulation.h"
#include "autograd/forward_ops.h"
#include "optim.h"
#include "autograd/autograd.h"
#include "autograd/loss_functions.h"
#include "autograd/amp.h"
#include "autograd/checkpointing.h"
#include "tensor/sparse_tensor.h"
#include "nn/layers/rnn.h"
#include "nn/layers/conv_transpose3d.h"
#include "nn/layers/upsample.h"
#include "nn/layers/pixel_shuffle.h"
#include "ops/ir/context.h"
#include "ops/ir/execution.h"
#include "ops/uops.h"
#include "core/gguf.h"
#include "core/safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#define F_OK 0
#define access _access
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

static bool g_cml_initialized       = false;
static int g_cml_init_count         = 0;
static bool g_cml_atexit_registered = false;

static CleanupContext** g_cleanup_contexts = NULL;
static size_t g_num_cleanup_contexts       = 0;
static size_t g_cleanup_contexts_capacity  = 0;

static Module** g_tracked_modules = NULL;
static size_t g_num_modules       = 0;
static size_t g_modules_capacity  = 0;

static Optimizer** g_tracked_optimizers = NULL;
static size_t g_num_optimizers          = 0;
static size_t g_optimizers_capacity     = 0;

static Dataset** g_tracked_datasets = NULL;
static size_t g_num_datasets        = 0;
static size_t g_datasets_capacity   = 0;

void cml_track_module(Module* module) {
    if (!module || !g_cml_initialized)
        return;

    if (g_num_modules >= g_modules_capacity) {
        size_t new_capacity  = g_modules_capacity == 0 ? 16 : g_modules_capacity * 2;
        Module** new_modules = realloc(g_tracked_modules, (size_t)new_capacity * sizeof(Module*));
        if (!new_modules)
            return;
        g_tracked_modules  = new_modules;
        g_modules_capacity = new_capacity;
    }
    g_tracked_modules[g_num_modules++] = module;
}

void cml_untrack_module(Module* module) {
    if (!module || !g_tracked_modules)
        return;

    for (size_t i = 0; i < g_num_modules; i++) {
        if (g_tracked_modules[i] == module) {
            g_tracked_modules[i] = NULL;
            return;
        }
    }
}

void cml_track_optimizer(Optimizer* optimizer) {
    if (!optimizer || !g_cml_initialized)
        return;

    if (g_num_optimizers >= g_optimizers_capacity) {
        size_t new_capacity = g_optimizers_capacity == 0 ? 16 : g_optimizers_capacity * 2;
        Optimizer** new_optimizers =
            realloc(g_tracked_optimizers, (size_t)new_capacity * sizeof(Optimizer*));
        if (!new_optimizers)
            return;
        g_tracked_optimizers  = new_optimizers;
        g_optimizers_capacity = new_capacity;
    }
    g_tracked_optimizers[g_num_optimizers++] = optimizer;
}

void cml_untrack_optimizer(Optimizer* optimizer) {
    if (!optimizer || !g_tracked_optimizers)
        return;

    for (size_t i = 0; i < g_num_optimizers; i++) {
        if (g_tracked_optimizers[i] == optimizer) {
            g_tracked_optimizers[i] = NULL;
            return;
        }
    }
}

void cml_track_dataset(Dataset* dataset) {
    if (!dataset || !g_cml_initialized)
        return;

    if (g_num_datasets >= g_datasets_capacity) {
        size_t new_capacity = g_datasets_capacity == 0 ? 16 : g_datasets_capacity * 2;
        Dataset** new_datasets =
            realloc(g_tracked_datasets, (size_t)new_capacity * sizeof(Dataset*));
        if (!new_datasets)
            return;
        g_tracked_datasets  = new_datasets;
        g_datasets_capacity = new_capacity;
    }
    g_tracked_datasets[g_num_datasets++] = dataset;
}

static void cml_auto_cleanup(void) {
    if (!g_cml_initialized) {
        return;
    }

    // FIRST: Reset global IR context to detach all tensors from IR
    // This must happen before freeing any tensors to prevent dangling pointers
    cml_ir_reset_global_context();

    if (g_cleanup_contexts) {
        for (size_t i = 0; i < g_num_cleanup_contexts; i++) {
            if (g_cleanup_contexts[i]) {
                cleanup_context_free(g_cleanup_contexts[i]);
            }
        }
        free(g_cleanup_contexts);
        g_cleanup_contexts          = NULL;
        g_num_cleanup_contexts      = 0;
        g_cleanup_contexts_capacity = 0;
    }

    if (g_tracked_datasets) {
        for (size_t i = 0; i < g_num_datasets; i++) {
            if (g_tracked_datasets[i]) {
                dataset_free(g_tracked_datasets[i]);
            }
        }
        free(g_tracked_datasets);
        g_tracked_datasets  = NULL;
        g_num_datasets      = 0;
        g_datasets_capacity = 0;
    }

    if (g_tracked_optimizers) {
        for (size_t i = 0; i < g_num_optimizers; i++) {
            if (g_tracked_optimizers[i]) {
                optimizer_free(g_tracked_optimizers[i]);
            }
        }
        free(g_tracked_optimizers);
        g_tracked_optimizers  = NULL;
        g_num_optimizers      = 0;
        g_optimizers_capacity = 0;
    }

    if (g_tracked_modules) {
        for (size_t i = 0; i < g_num_modules; i++) {
            if (g_tracked_modules[i]) {
                module_free(g_tracked_modules[i]);
            }
        }
        free(g_tracked_modules);
        g_tracked_modules  = NULL;
        g_num_modules      = 0;
        g_modules_capacity = 0;
    }

    device_cleanup();

    cml_graph_context_cleanup();

    training_metrics_cleanup_global();

    autograd_shutdown();

    if (CML_HAS_ERRORS()) {
        printf("\nErrors occurred during execution\n");
        error_stack_print_all();
        printf("Last error: %s (code: %d)\n", CML_LAST_ERROR(), CML_LAST_ERROR_CODE());
    } else {
        printf("\nExecution completed successfully\n");
    }

    error_stack_cleanup();

    cml_cleanup_buffer_cache();

    g_cml_initialized = false;
    g_cml_init_count  = 0;
}

void cml_register_cleanup_context(CleanupContext* ctx) {
    if (!ctx)
        return;

    if (g_num_cleanup_contexts >= g_cleanup_contexts_capacity) {
        size_t new_capacity =
            g_cleanup_contexts_capacity == 0 ? 16 : g_cleanup_contexts_capacity * 2;
        CleanupContext** new_contexts =
            realloc(g_cleanup_contexts, (size_t)new_capacity * sizeof(CleanupContext*));
        if (!new_contexts)
            return;
        g_cleanup_contexts          = new_contexts;
        g_cleanup_contexts_capacity = new_capacity;
    }

    g_cleanup_contexts[g_num_cleanup_contexts++] = ctx;
}

/* Constructor disabled: call check_and_launch_viz() from cml_init() instead */
static void check_and_launch_viz(void) {
    /* Quick exit: only activate when VIZ env var is explicitly set */
    const char* viz = getenv("VIZ");
    if (!viz || viz[0] == '\0') {
        return;
    }
    if (viz[0] != '1' && strcmp(viz, "true") != 0) {
        return;
    }

    const char* viz_launched = getenv("CML_VIZ_LAUNCHED");
    if (viz_launched && viz_launched[0] != '\0') {
        return;
    }

#ifdef _WIN32
    const char* try_paths[] = {"scripts/viz.py", "../scripts/viz.py", getenv("CML_VIZ_SCRIPT"),
                               NULL};
#else
    const char* try_paths[] = {
        "scripts/viz.py",        "../scripts/viz.py",      "/usr/local/share/cml/viz.py",
        "/usr/share/cml/viz.py", getenv("CML_VIZ_SCRIPT"), NULL};
#endif

    const char* script_path = NULL;
    for (int i = 0; try_paths[i]; i++) {
        if (!try_paths[i])
            continue;
        if (access(try_paths[i], F_OK) == 0) {
            script_path = try_paths[i];
            break;
        }
    }

    if (!script_path) {
        return;
    }

    char exe_path[1024] = {0};

#ifdef _WIN32
    GetModuleFileName(NULL, exe_path, sizeof(exe_path));
    _putenv_s("CML_VIZ_LAUNCHED", "1");
    _putenv_s("CML_VIZ", "1");

    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "python \"%s\" \"%s\"", script_path, exe_path);
    system(cmd);

    // Note: On Windows, system() waits for the command to finish.
    // If we want async, we'd need CreateProcess.
    // For viz, blocking might be annoying if it doesn't return.
    // However, the python script spawns subprocesses, so it might be okay
    // if it returns quickly or if we want to block until viz is closed.

#elif defined(__APPLE__)
    uint32_t size = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &size) != 0) {
        return;
    }

    setenv("CML_VIZ_LAUNCHED", "1", 1);
    setenv("CML_VIZ", "1", 1);

    char script_buf[1024];
    strncpy(script_buf, script_path, sizeof(script_buf) - 1);
    script_buf[sizeof(script_buf) - 1] = '\0';
    char python_cmd[]                  = "python3";
    char* viz_argv[]                   = {python_cmd, script_buf, exe_path, NULL};

    // Fork and exec to avoid blocking the main process if possible,
    // or just use execvp if we want to replace (but we are in a constructor...)
    // Wait, this is a constructor. We shouldn't replace the process!
    execvp("python3", viz_argv);

#else
    char cmdline[4096] = {0};
    char* args[64]     = {0}; // Max 64 arguments
    int argc           = 0;

    FILE* f = fopen("/proc/self/cmdline", "r");
    if (f) {
        size_t cmdlen = fread(cmdline, 1, sizeof(cmdline) - 1, f);
        fclose(f);

        char* p   = cmdline;
        char* end = cmdline + cmdlen;
        while (p < end && argc < 62) { // Leave room for python3 and script
            if (*p) {
                args[argc++] = p;
                p += strlen(p);
            }
            p++;
        }
    }

    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len == -1 || len >= (ssize_t)sizeof(exe_path)) {
        const char* env_exe = getenv("_");
        if (env_exe && env_exe[0] != '\0') {
            strncpy(exe_path, env_exe, sizeof(exe_path) - 1);
            exe_path[sizeof(exe_path) - 1] = '\0';
        } else if (argc > 0) {
            strncpy(exe_path, args[0], sizeof(exe_path) - 1);
        } else {
            strncpy(exe_path, "./build/main", sizeof(exe_path) - 1);
        }
    } else {
        exe_path[len] = '\0';
    }

    setenv("CML_VIZ_LAUNCHED", "1", 1);
    setenv("CML_VIZ", "1", 1);

    char python_cmd[] = "python3";
    char script_buf[1024];
    strncpy(script_buf, script_path, sizeof(script_buf) - 1);
    script_buf[sizeof(script_buf) - 1] = '\0';
    char* viz_argv[68]                 = {0};
    viz_argv[0]                        = python_cmd;
    viz_argv[1]                        = script_buf;
    viz_argv[2]                        = exe_path;

    int vi = 3;
    for (int i = 1; i < argc && vi < 66; i++) {
        viz_argv[vi++] = args[i];
    }
    viz_argv[vi] = NULL;

    execvp("python3", viz_argv);
#endif
}

static const char* g_build_info = "C-ML Library\n"
                                  "Features: autograd, nn, optim, logging, memory_management";

int cml_init(void) {
    if (g_cml_initialized) {
        g_cml_init_count++;
        LOG_DEBUG("C-ML library already initialized, reference count: %d", g_cml_init_count);
        return 0;
    }

    LOG_INFO("Initializing C-ML Library");

    int result = 0;

    error_stack_init();

    cml_set_log_level(LOG_LEVEL_ERROR);

    cml_set_default_device(DEVICE_CPU);
    cml_set_default_dtype(DTYPE_FLOAT32);

    cml_random_seed();

    autograd_init();

    training_metrics_init_global();

    cml_graph_context_init();

    if (!g_cml_atexit_registered) {
        atexit(cml_auto_cleanup);
        g_cml_atexit_registered = true;
    }

    if (result == 0) {
        g_cml_initialized = true;
        g_cml_init_count  = 1;
        LOG_INFO("C-ML Library initialized successfully");
        check_and_launch_viz();
    } else {
        LOG_ERROR("Failed to initialize C-ML Library");
    }

    return result;
}

int cml_cleanup(void) {
    if (!g_cml_initialized) {
        LOG_WARNING("C-ML library not initialized, nothing to cleanup");
        return 0;
    }

    g_cml_init_count--;

    if (g_cml_init_count > 0) {
        LOG_DEBUG("C-ML library cleanup requested, but reference count > 0: %d", g_cml_init_count);
        return 0;
    }

    LOG_INFO("Cleaning up C-ML Library");

    g_cml_initialized = false;
    g_cml_init_count  = 0;

    // Reset global IR context FIRST to detach all tensors from IR
    // This must happen before freeing any tensors to prevent dangling pointers
    cml_ir_reset_global_context();

    cml_graph_context_cleanup();

    if (g_tracked_datasets) {
        for (size_t i = 0; i < g_num_datasets; i++) {
            if (g_tracked_datasets[i]) {
                dataset_free(g_tracked_datasets[i]);
                g_tracked_datasets[i] = NULL;
            }
        }
        free(g_tracked_datasets);
        g_tracked_datasets  = NULL;
        g_num_datasets      = 0;
        g_datasets_capacity = 0;
    }

    if (g_tracked_optimizers) {
        for (size_t i = 0; i < g_num_optimizers; i++) {
            if (g_tracked_optimizers[i]) {
                optimizer_free(g_tracked_optimizers[i]);
                g_tracked_optimizers[i] = NULL;
            }
        }
        free(g_tracked_optimizers);
        g_tracked_optimizers  = NULL;
        g_num_optimizers      = 0;
        g_optimizers_capacity = 0;
    }

    if (g_tracked_modules) {
        for (size_t i = 0; i < g_num_modules; i++) {
            if (g_tracked_modules[i]) {
                module_free(g_tracked_modules[i]);
                g_tracked_modules[i] = NULL;
            }
        }
        free(g_tracked_modules);
        g_tracked_modules  = NULL;
        g_num_modules      = 0;
        g_modules_capacity = 0;
    }

    device_cleanup();

    training_metrics_cleanup_global();

    autograd_shutdown();

    if (CML_HAS_ERRORS()) {
        printf("\nErrors occurred during execution\n");
        error_stack_print_all();
        printf("Last error: %s (code: %d)\n", CML_LAST_ERROR(), CML_LAST_ERROR_CODE());
    } else {
        printf("\nExecution completed successfully\n");
    }

    error_stack_cleanup();

    cml_cleanup_buffer_cache();

    LOG_INFO("C-ML Library cleanup completed");
    return 0;
}

static CMLGlobalErrorHandler g_error_handler = NULL;

void cml_set_error_handler(CMLGlobalErrorHandler handler) { g_error_handler = handler; }

CMLGlobalErrorHandler cml_get_error_handler(void) { return g_error_handler; }

void cml_get_version(int* major, int* minor, int* patch, const char** version_string) {
    if (major)
        *major = CML_VERSION_MAJOR;
    if (minor)
        *minor = CML_VERSION_MINOR;
    if (patch)
        *patch = CML_VERSION_PATCH;
    if (version_string)
        *version_string = CML_VERSION_STRING;
}

int cml_version(void) { return CML_VERSION; }

const char* cml_version_string(void) { return CML_VERSION_STRING; }

const char* cml_get_build_info(void) { return g_build_info; }

bool cml_is_initialized(void) { return g_cml_initialized; }

int cml_get_init_count(void) { return g_cml_init_count; }

int cml_force_cleanup(void) {
    if (!g_cml_initialized) {
        return 0;
    }

    LOG_WARNING("Forcing C-ML library cleanup (reference count was %d)", g_cml_init_count);

    g_cml_init_count = 0;
    return cml_cleanup();
}

static void print_layer_summary(Module* module, int indent, int* layer_num) {
    if (!module)
        return;

    if (strcmp(module->name, "Sequential") == 0) {
        Sequential* seq = (Sequential*)module;
        int num_modules = sequential_get_length(seq);
        for (int i = 0; i < num_modules; i++) {
            Module* child = sequential_get(seq, i);
            if (child) {
                print_layer_summary(child, indent, layer_num);
            }
        }
        return;
    }

    const char* layer_type = module->name;

    int layer_params = 0;
    for (int i = 0; i < module->num_parameters; i++) {
        if (module->parameters[i] && module->parameters[i]->tensor) {
            layer_params += (int)module->parameters[i]->tensor->numel;
        }
    }

    char layer_desc[64] = {0};
    if (strcmp(module->name, "Linear") == 0) {
        Linear* linear   = (Linear*)module;
        int in_features  = linear_get_in_features(linear);
        int out_features = linear_get_out_features(linear);
        bool use_bias    = linear_get_use_bias(linear);
        snprintf(layer_desc, sizeof(layer_desc), "%s (%d->%d, bias=%s)", layer_type, in_features,
                 out_features, use_bias ? "True" : "False");
    } else {
        snprintf(layer_desc, sizeof(layer_desc), "%s", layer_type);
    }

    if (indent > 0) {
        for (int i = 0; i < indent; i++)
            printf("  ");
    }
    printf("%-5d %-35s %15d\n", (*layer_num)++, layer_desc, layer_params);
}

void cml_summary(Module* module) {
    if (!module) {
        printf("Model Summary: (empty)\n");
        return;
    }

    printf("\n");
    for (int i = 0; i < 60; i++)
        printf("=");
    printf("\nModel Summary\n");
    for (int i = 0; i < 60; i++)
        printf("=");
    printf("\n");

    Parameter** params  = NULL;
    int num_params      = 0;
    int total_trainable = 0;

    if (module_collect_parameters(module, &params, &num_params, true) == 0) {
        for (int i = 0; i < num_params; i++) {
            if (params[i] && params[i]->tensor && params[i]->requires_grad) {
                total_trainable += (int)params[i]->tensor->numel;
            }
        }
        if (params)
            free(params);
    }

    printf("%-5s %-35s %15s\n", "Layer", "Type", "Parameters");
    for (int i = 0; i < 60; i++)
        printf("-");
    printf("\n");

    int layer_num = 1;
    print_layer_summary(module, 0, &layer_num);

    for (int i = 0; i < 60; i++)
        printf("-");
    printf("\n");
    printf("Total params: %d\n", total_trainable);
    printf("Trainable params: %d\n", total_trainable);
    printf("Non-trainable params: 0\n");
    for (int i = 0; i < 60; i++)
        printf("=");
    printf("\n\n");

    const char* viz_env     = getenv("VIZ");
    const char* cml_viz_env = getenv("CML_VIZ");
    bool viz_enabled = (viz_env && (strcmp(viz_env, "1") == 0 || strcmp(viz_env, "true") == 0)) ||
                       (cml_viz_env && (strcmp(cml_viz_env, "1") == 0));

    if (viz_enabled) {
        ModelArchitecture* arch = model_architecture_create();
        if (arch) {
            if (model_architecture_extract(module, arch) == 0) {
                model_architecture_export_json(arch, "model_architecture.json");
                LOG_INFO("Exported model architecture to model_architecture.json");
            }
            model_architecture_free(arch);
        }
    }
}

Tensor* cml_empty(int* shape, int ndim, const TensorConfig* config) {
    return tensor_empty(shape, ndim, config);
}

Tensor* cml_zeros(int* shape, int ndim, const TensorConfig* config) {
    return tensor_zeros(shape, ndim, config);
}

Tensor* cml_ones(int* shape, int ndim, const TensorConfig* config) {
    return tensor_ones(shape, ndim, config);
}

Tensor* cml_full(int* shape, int ndim, const TensorConfig* config, float value) {
    return tensor_full(shape, ndim, config, value);
}

Tensor* cml_tensor(void* data, int* shape, int ndim, const TensorConfig* config) {
    return tensor_from_data(data, shape, ndim, config);
}

Tensor* cml_zeros_2d(int rows, int cols) { return tensor_zeros_2d(rows, cols); }

Tensor* cml_ones_2d(int rows, int cols) { return tensor_ones_2d(rows, cols); }

Tensor* cml_empty_2d(int rows, int cols) { return tensor_empty_2d(rows, cols); }

Tensor* cml_tensor_2d(const float* data, int rows, int cols) {
    return tensor_from_array_2d(data, rows, cols);
}

Tensor* cml_zeros_1d(int size) {
    int shape[] = {size};
    return tensor_zeros(shape, 1, NULL);
}

Tensor* cml_ones_1d(int size) {
    int shape[] = {size};
    return tensor_ones(shape, 1, NULL);
}

Tensor* cml_empty_1d(int size) {
    int shape[] = {size};
    return tensor_empty(shape, 1, NULL);
}

Tensor* cml_tensor_1d(const float* data, int size) {
    int shape[] = {size};
    return tensor_from_data(data, shape, 1, NULL);
}

Tensor* cml_add(Tensor* a, Tensor* b) { return tensor_add(a, b); }
Tensor* cml_sub(Tensor* a, Tensor* b) { return tensor_sub(a, b); }
Tensor* cml_mul(Tensor* a, Tensor* b) { return tensor_mul(a, b); }
Tensor* cml_div(Tensor* a, Tensor* b) { return tensor_div(a, b); }
Tensor* cml_exp(Tensor* a) { return tensor_exp(a); }
Tensor* cml_log(Tensor* a) { return tensor_log(a); }
Tensor* cml_sqrt(Tensor* a) { return tensor_sqrt(a); }
Tensor* cml_sin(Tensor* a) { return tensor_sin(a); }
Tensor* cml_cos(Tensor* a) { return tensor_cos(a); }
Tensor* cml_tan(Tensor* a) { return tensor_tan(a); }
Tensor* cml_pow(Tensor* a, Tensor* b) { return tensor_pow(a, b); }
Tensor* cml_relu(Tensor* a) { return tensor_relu(a); }
Tensor* cml_sigmoid(Tensor* a) { return tensor_sigmoid(a); }
Tensor* cml_tanh(Tensor* a) { return tensor_tanh(a); }
Tensor* cml_softmax(Tensor* a, int dim) { return tensor_softmax(a, dim); }
Tensor* cml_elu(Tensor* x, float alpha) { return tensor_elu(x, alpha); }
Tensor* cml_selu(Tensor* x) { return tensor_selu(x); }
Tensor* cml_mish(Tensor* x) { return tensor_mish(x); }
Tensor* cml_silu(Tensor* x) { return tensor_silu(x); }
Tensor* cml_hardswish(Tensor* x) { return tensor_hardswish(x); }
Tensor* cml_leaky_relu(Tensor* x, float negative_slope) { return tensor_leaky_relu(x, negative_slope); }
Tensor* cml_sum(Tensor* a, int dim, bool keepdim) { return tensor_sum(a, dim, keepdim); }
Tensor* cml_mean(Tensor* a, int dim, bool keepdim) { return tensor_mean(a, dim, keepdim); }
Tensor* cml_max(Tensor* a, int dim, bool keepdim) { return tensor_max(a, dim, keepdim); }
Tensor* cml_min(Tensor* a, int dim, bool keepdim) { return tensor_min(a, dim, keepdim); }
Tensor* cml_matmul(Tensor* a, Tensor* b) { return tensor_matmul(a, b); }
Tensor* cml_transpose(Tensor* a, int dim1, int dim2) { return tensor_transpose(a, dim1, dim2); }
Tensor* cml_reshape(Tensor* a, int* new_shape, int new_ndim) {
    return tensor_reshape(a, new_shape, new_ndim);
}
Tensor* cml_clone(Tensor* a) { return tensor_clone(a); }
Tensor* cml_detach(Tensor* a) { return tensor_detach(a); }
Tensor* cml_concat(Tensor** tensors, int num_tensors, int dim) {
    return tensor_concat(tensors, num_tensors, dim);
}
Tensor* cml_stack(Tensor** tensors, int num_tensors, int dim) {
    return tensor_stack(tensors, num_tensors, dim);
}

Sequential* cml_nn_sequential(void) { return nn_sequential(); }
Sequential* cml_nn_sequential_add(Sequential* seq, Module* layer) {
    return sequential_add_chain(seq, layer);
}
Tensor* cml_nn_sequential_forward(Sequential* seq, Tensor* input) {
    return module_forward((Module*)seq, input);
}
Linear* cml_nn_linear(int in_features, int out_features, DType dtype, DeviceType device,
                      bool bias) {
    return nn_linear(in_features, out_features, dtype, device, bias);
}
ReLU* cml_nn_relu(bool inplace) { return nn_relu(inplace); }
Sigmoid* cml_nn_sigmoid(void) { return nn_sigmoid(); }
Tanh* cml_nn_tanh(void) { return nn_tanh(); }
LeakyReLU* cml_nn_leaky_relu(float negative_slope, bool inplace) {
    return nn_leaky_relu(negative_slope, inplace);
}
Dropout* cml_nn_dropout(float p, bool inplace) { return nn_dropout(p, inplace); }
Conv2d* cml_nn_conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                      int dilation, bool bias, DType dtype, DeviceType device) {
    return nn_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, dtype,
                     device);
}
BatchNorm2d* cml_nn_batchnorm2d(int num_features, float eps, float momentum, bool affine,
                                bool track_running_stats, DType dtype, DeviceType device) {
    return nn_batchnorm2d(num_features, eps, momentum, affine, track_running_stats, dtype, device);
}
LayerNorm* cml_nn_layernorm(int normalized_shape, float eps, bool affine, DType dtype,
                            DeviceType device) {
    return nn_layernorm(normalized_shape, eps, affine, dtype, device);
}
MaxPool2d* cml_nn_maxpool2d(int kernel_size, int stride, int padding, int dilation,
                            bool ceil_mode) {
    return nn_maxpool2d(kernel_size, stride, padding, dilation, ceil_mode);
}
AvgPool2d* cml_nn_avgpool2d(int kernel_size, int stride, int padding, bool ceil_mode,
                            bool count_include_pad) {
    return nn_avgpool2d(kernel_size, stride, padding, ceil_mode, count_include_pad);
}

Conv1d* cml_nn_conv1d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                      int dilation, bool use_bias, DType dtype, DeviceType device) {
    return nn_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias,
                     dtype, device);
}
Conv3d* cml_nn_conv3d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                      int dilation, bool use_bias, DType dtype, DeviceType device) {
    return nn_conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias,
                     dtype, device);
}
Embedding* cml_nn_embedding(int num_embeddings, int embedding_dim, int padding_idx, DType dtype,
                            DeviceType device) {
    return nn_embedding(num_embeddings, embedding_dim, padding_idx, dtype, device);
}
GroupNorm* cml_nn_groupnorm(int num_groups, int num_channels, float eps, bool affine, DType dtype,
                            DeviceType device) {
    return nn_groupnorm(num_groups, num_channels, eps, affine, dtype, device);
}
RNNCell* cml_nn_rnn_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                         DeviceType device) {
    return nn_rnn_cell(input_size, hidden_size, use_bias, dtype, device);
}
LSTMCell* cml_nn_lstm_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                           DeviceType device) {
    return nn_lstm_cell(input_size, hidden_size, use_bias, dtype, device);
}
GRUCell* cml_nn_gru_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                         DeviceType device) {
    return nn_gru_cell(input_size, hidden_size, use_bias, dtype, device);
}
MultiHeadAttention* cml_nn_multihead_attention(int embed_dim, int num_heads, float dropout,
                                               DType dtype, DeviceType device) {
    return nn_multihead_attention(embed_dim, num_heads, dropout, dtype, device);
}
TransformerEncoderLayer* cml_nn_transformer_encoder_layer(int d_model, int nhead,
                                                          int dim_feedforward, float dropout,
                                                          DType dtype, DeviceType device) {
    return nn_transformer_encoder_layer(d_model, nhead, dim_feedforward, dropout, dtype, device);
}
ModuleList* cml_nn_module_list(void) { return nn_module_list(); }
ModuleDict* cml_nn_module_dict(void) { return nn_module_dict(); }

Optimizer* cml_optim_adam(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                          float beta1, float beta2, float eps) {
    return optim_adam(parameters, num_parameters, lr, weight_decay, beta1, beta2, eps);
}
Optimizer* cml_optim_sgd(Parameter** parameters, int num_parameters, float lr, float momentum,
                         float weight_decay) {
    return optim_sgd(parameters, num_parameters, lr, momentum, weight_decay);
}
Optimizer* cml_optim_rmsprop(Parameter** parameters, int num_parameters, float lr,
                             float weight_decay, float alpha, float eps) {
    return optim_rmsprop(parameters, num_parameters, lr, weight_decay, alpha, eps);
}
Optimizer* cml_optim_adagrad(Parameter** parameters, int num_parameters, float lr,
                             float weight_decay, float eps) {
    return optim_adagrad(parameters, num_parameters, lr, weight_decay, eps);
}
Optimizer* cml_optim_adam_for_model(Module* model, float lr, float weight_decay, float beta1,
                                    float beta2, float eps) {
    return optim_adam_for_model(model, lr, weight_decay, beta1, beta2, eps);
}
Optimizer* cml_optim_sgd_for_model(Module* model, float lr, float momentum, float weight_decay) {
    return optim_sgd_for_model(model, lr, momentum, weight_decay);
}
void cml_optim_zero_grad(Optimizer* optimizer) { optimizer_zero_grad(optimizer); }
void cml_optim_step(Optimizer* optimizer) { optimizer_step(optimizer); }

Tensor* cml_nn_mse_loss(Tensor* input, Tensor* target) { return tensor_mse_loss(input, target); }
Tensor* cml_nn_mae_loss(Tensor* input, Tensor* target) { return tensor_mae_loss(input, target); }
Tensor* cml_nn_bce_loss(Tensor* input, Tensor* target) { return tensor_bce_loss(input, target); }
Tensor* cml_nn_cross_entropy_loss(Tensor* input, Tensor* target) {
    return tensor_cross_entropy_loss(input, target);
}
Tensor* cml_nn_huber_loss(Tensor* input, Tensor* target, float delta) {
    return tensor_huber_loss(input, target, delta);
}
Tensor* cml_nn_kl_div_loss(Tensor* input, Tensor* target) {
    return tensor_kl_div_loss(input, target);
}
Tensor* cml_nn_sparse_cross_entropy_loss(Tensor* input, Tensor* target) {
    return tensor_sparse_cross_entropy_loss(input, target);
}
Tensor* cml_nn_triplet_margin_loss(Tensor* anchor, Tensor* positive, Tensor* negative,
                                   float margin) {
    return tensor_triplet_margin_loss(anchor, positive, negative, margin);
}
Tensor* cml_nn_cosine_embedding_loss(Tensor* x1, Tensor* x2, Tensor* target, float margin) {
    return tensor_cosine_embedding_loss(x1, x2, target, margin);
}
Tensor* cml_nn_nll_loss(Tensor* log_probs, Tensor* targets) {
    return tensor_nll_loss(log_probs, targets);
}

void cml_backward(Tensor* tensor, Tensor* gradient, bool retain_graph, bool create_graph) {
    tensor_backward(tensor, gradient, retain_graph, create_graph);
}
void cml_zero_grad(Tensor* tensor) { tensor_zero_grad(tensor); }
void cml_no_grad(void) { autograd_no_grad_enter(); }
void cml_enable_grad(void) { autograd_set_grad_mode(true); }
bool cml_is_grad_enabled(void) { return autograd_is_grad_enabled(); }
bool cml_requires_grad(Tensor* t) { return tensor_requires_grad(t); }
void cml_set_requires_grad(Tensor* t, bool requires_grad) {
    tensor_set_requires_grad(t, requires_grad);
}
bool cml_is_leaf(Tensor* t) { return tensor_is_leaf(t); }
void cml_reset_ir_context(void) { cml_ir_reset_global_context(); }

struct CMLKernelCache;
struct CMLKernelCache* cml_kernel_cache_get_default(void);
void cml_kernel_cache_clear_impl(struct CMLKernelCache* cache);
void cml_kernel_cache_stats_impl(struct CMLKernelCache* cache, size_t* hits, size_t* misses,
                                 size_t* count, size_t* memory);
double cml_kernel_cache_hit_rate_impl(struct CMLKernelCache* cache);
void cml_kernel_cache_print_stats_impl(struct CMLKernelCache* cache);

void cml_kernel_cache_clear(void) {
    struct CMLKernelCache* cache = cml_kernel_cache_get_default();
    if (cache) {
        cml_kernel_cache_clear_impl(cache);
    }
}

void cml_kernel_cache_stats(size_t* hits, size_t* misses, size_t* count, size_t* memory) {
    struct CMLKernelCache* cache = cml_kernel_cache_get_default();
    if (cache) {
        cml_kernel_cache_stats_impl(cache, hits, misses, count, memory);
    } else {
        if (hits)
            *hits = 0;
        if (misses)
            *misses = 0;
        if (count)
            *count = 0;
        if (memory)
            *memory = 0;
    }
}

double cml_kernel_cache_hit_rate(void) {
    struct CMLKernelCache* cache = cml_kernel_cache_get_default();
    if (cache) {
        return cml_kernel_cache_hit_rate_impl(cache);
    }
    return 0.0;
}

void cml_kernel_cache_print_stats(void) {
    struct CMLKernelCache* cache = cml_kernel_cache_get_default();
    if (cache) {
        cml_kernel_cache_print_stats_impl(cache);
    } else {
        printf("Kernel Cache: not initialized\n");
    }
}

Tensor* cml_nn_module_forward(Module* module, Tensor* input) {
    return module_forward(module, input);
}
void cml_nn_module_set_training(Module* module, bool training) {
    module_set_training(module, training);
}
bool cml_nn_module_is_training(Module* module) { return module_is_training(module); }
void cml_nn_module_eval(Module* module) { module_set_training(module, false); }
void cml_nn_module_train(Module* module) { module_set_training(module, true); }

Tensor* cml_sign(Tensor* a) { return uop_sign(a); }
Tensor* cml_floor(Tensor* a) { return uop_floor(a); }
Tensor* cml_ceil(Tensor* a) { return uop_ceil(a); }
Tensor* cml_round(Tensor* a) { return uop_round(a); }
Tensor* cml_log2(Tensor* a) { return uop_log2(a); }
Tensor* cml_exp2(Tensor* a) { return uop_exp2(a); }
Tensor* cml_asin(Tensor* a) { return uop_asin(a); }
Tensor* cml_acos(Tensor* a) { return uop_acos(a); }
Tensor* cml_atan(Tensor* a) { return uop_atan(a); }
Tensor* cml_square(Tensor* a) { return uop_square(a); }
Tensor* cml_rsqrt(Tensor* a) { return uop_rsqrt(a); }
Tensor* cml_erf(Tensor* a) { return uop_erf(a); }

Tensor* cml_clamp(Tensor* a, float min_val, float max_val) {
    return uop_clamp(a, min_val, max_val);
}

Tensor* cml_prod(Tensor* a, int dim, bool keepdim) {
    int dims[] = {dim};
    ReduceParams params = {.dims = dims, .num_dims = 1, .keepdim = keepdim};
    return uop_prod(a, &params);
}

Tensor* cml_argmax(Tensor* a, int dim) { return tensor_argmax(a, dim); }
Tensor* cml_argmin(Tensor* a, int dim) { return tensor_argmin(a, dim); }

Tensor* cml_cumsum(Tensor* a, int dim) {
    return uop_cumsum(a, dim);
}

Tensor* cml_var(Tensor* a, int dim, bool unbiased, bool keepdim) {
    return tensor_var(a, dim, unbiased, keepdim);
}

Tensor* cml_std(Tensor* a, int dim, bool unbiased, bool keepdim) {
    return tensor_std(a, dim, unbiased, keepdim);
}

Tensor* cml_squeeze(Tensor* a, int dim) { return tensor_squeeze(a, dim); }
Tensor* cml_unsqueeze(Tensor* a, int dim) { return tensor_unsqueeze(a, dim); }
Tensor* cml_flip(Tensor* a, int dim) { return tensor_flip(a, dim); }
Tensor* cml_repeat(Tensor* a, int* repeats, int num_repeats) {
    return tensor_repeat(a, repeats, num_repeats);
}
Tensor** cml_split(Tensor* a, int num_splits, int dim, int* out_count) {
    return tensor_split(a, num_splits, dim, out_count);
}
Tensor** cml_chunk(Tensor* a, int chunks, int dim, int* out_count) {
    return tensor_chunk(a, chunks, dim, out_count);
}

Tensor* cml_triu(Tensor* a, int diagonal) {
    return uop_triu(a, diagonal);
}

Tensor* cml_tril(Tensor* a, int diagonal) {
    return uop_tril(a, diagonal);
}

Tensor* cml_pad(Tensor* a, int* pad_widths, int num_dims, float value) {
    return uop_pad(a, pad_widths, num_dims, value);
}
Tensor* cml_pad_reflect(Tensor* a, int* pad_widths, int num_dims) {
    return uop_pad_reflect(a, pad_widths, num_dims);
}
Tensor* cml_pad_replicate(Tensor* a, int* pad_widths, int num_dims) {
    return uop_pad_replicate(a, pad_widths, num_dims);
}

Tensor* cml_arange(float start, float end, float step, const TensorConfig* config) {
    return tensor_arange(start, end, step, config);
}
Tensor* cml_linspace(float start, float end, int steps, const TensorConfig* config) {
    return tensor_linspace(start, end, steps, config);
}
Tensor* cml_eye(int n, const TensorConfig* config) { return tensor_eye(n, config); }
Tensor* cml_rand(int* shape, int ndim, const TensorConfig* config) {
    return tensor_rand(shape, ndim, config);
}
Tensor* cml_randn(int* shape, int ndim, const TensorConfig* config) {
    return tensor_randn(shape, ndim, config);
}
Tensor* cml_randint(int low, int high, int* shape, int ndim, const TensorConfig* config) {
    return tensor_randint(low, high, shape, ndim, config);
}
void cml_manual_seed(uint64_t seed) { tensor_manual_seed(seed); }
Tensor* cml_zeros_like(Tensor* a) { return tensor_zeros_like(a); }
Tensor* cml_ones_like(Tensor* a) { return tensor_ones_like(a); }
Tensor* cml_rand_like(Tensor* a) { return tensor_rand_like(a); }
Tensor* cml_randn_like(Tensor* a) { return tensor_randn_like(a); }
Tensor* cml_full_like(Tensor* a, float value) { return tensor_full_like(a, value); }

Tensor* cml_kaiming_uniform(int* shape, int ndim, int fan_in, const TensorConfig* config) {
    return tensor_kaiming_uniform(shape, ndim, fan_in, config);
}
Tensor* cml_kaiming_normal(int* shape, int ndim, int fan_in, const TensorConfig* config) {
    return tensor_kaiming_normal(shape, ndim, fan_in, config);
}
Tensor* cml_glorot_uniform(int* shape, int ndim, int fan_in, int fan_out,
                            const TensorConfig* config) {
    return tensor_glorot_uniform(shape, ndim, fan_in, fan_out, config);
}
Tensor* cml_xavier_normal(int* shape, int ndim, int fan_in, int fan_out,
                           const TensorConfig* config) {
    return tensor_xavier_normal(shape, ndim, fan_in, fan_out, config);
}

Optimizer* cml_optim_lamb(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                           float beta1, float beta2, float epsilon) {
    return optim_lamb(parameters, num_parameters, lr, weight_decay, beta1, beta2, epsilon);
}
Optimizer* cml_optim_lars(Parameter** parameters, int num_parameters, float lr, float momentum,
                           float weight_decay, float trust_coefficient) {
    return optim_lars(parameters, num_parameters, lr, momentum, weight_decay, trust_coefficient);
}

InstanceNorm2d* cml_nn_instancenorm2d(int num_features, float eps, bool affine, DType dtype,
                                       DeviceType device) {
    return nn_instancenorm2d(num_features, eps, affine, dtype, device);
}
ConvTranspose1d* cml_nn_conv_transpose1d(int in_channels, int out_channels, int kernel_size,
                                          int stride, int padding, int output_padding,
                                          bool use_bias, DType dtype, DeviceType device) {
    return nn_conv_transpose1d(in_channels, out_channels, kernel_size, stride, padding,
                               output_padding, use_bias, dtype, device);
}
BatchNorm3d* cml_nn_batchnorm3d(int num_features, float eps, float momentum, bool affine,
                                 bool track_running_stats, DType dtype, DeviceType device) {
    return nn_batchnorm3d(num_features, eps, momentum, affine, track_running_stats, dtype, device);
}
LayerNorm2d* cml_nn_layernorm2d(int num_channels, float eps, bool affine, DType dtype,
                                 DeviceType device) {
    return nn_layernorm2d(num_channels, eps, affine, dtype, device);
}

Optimizer* cml_optim_muon(Parameter** parameters, int num_parameters, float lr, float momentum,
                           float weight_decay, bool nesterov) {
    return optim_muon(parameters, num_parameters, lr, momentum, weight_decay, nesterov);
}
Optimizer* cml_optim_adamw(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                            float beta1, float beta2, float epsilon) {
    return optim_adamw(parameters, num_parameters, lr, weight_decay, beta1, beta2, epsilon);
}
Optimizer* cml_optim_adadelta(Parameter** parameters, int num_parameters, float rho,
                               float weight_decay, float epsilon) {
    return optim_adadelta(parameters, num_parameters, rho, weight_decay, epsilon);
}

Tensor* cml_unfold(Tensor* a, int kernel_size, int stride) {
    return uop_unfold(a, kernel_size, stride);
}

Tensor* cml_cast(Tensor* a, DType dtype) { return tensor_cast(a, dtype); }
Tensor* cml_contiguous(Tensor* a) { return tensor_contiguous(a); }
Tensor* cml_from_blob(void* data, int* shape, int ndim, const TensorConfig* config) {
    return tensor_from_blob(data, shape, ndim, config);
}
Tensor* cml_randperm(int n, const TensorConfig* config) { return tensor_randperm(n, config); }
Tensor* cml_half(Tensor* a) { return tensor_half(a); }
Tensor* cml_double(Tensor* a) { return tensor_double(a); }
Tensor* cml_int_(Tensor* a) { return tensor_int(a); }
Tensor* cml_long(Tensor* a) { return tensor_long(a); }
Tensor* cml_short(Tensor* a) { return tensor_short(a); }
Tensor* cml_bool_(Tensor* a) { return tensor_bool(a); }
Tensor* cml_bfloat16(Tensor* a) { return tensor_bfloat16(a); }

Tensor* cml_interpolate(Tensor* a, int* output_size, int num_dims, InterpMode mode) {
    return tensor_interpolate(a, output_size, num_dims, mode);
}

Tensor* cml_dot(Tensor* a, Tensor* b) { return tensor_dot(a, b); }

MaxPool1d* cml_nn_maxpool1d(int kernel_size, int stride, int padding, int dilation, bool ceil_mode) {
    return nn_maxpool1d(kernel_size, stride, padding, dilation, ceil_mode);
}
AvgPool1d* cml_nn_avgpool1d(int kernel_size, int stride, int padding, bool ceil_mode,
                            bool count_include_pad) {
    return nn_avgpool1d(kernel_size, stride, padding, ceil_mode, count_include_pad);
}
AdaptiveAvgPool2d* cml_nn_adaptive_avgpool2d(int output_h, int output_w) {
    return nn_adaptive_avgpool2d(output_h, output_w);
}
AdaptiveAvgPool1d* cml_nn_adaptive_avgpool1d(int output_size) {
    return nn_adaptive_avgpool1d(output_size);
}

MaxPool3d* cml_nn_maxpool3d(int kernel_size, int stride, int padding, int dilation, bool ceil_mode) {
    return nn_maxpool3d(kernel_size, stride, padding, dilation, ceil_mode);
}
AvgPool3d* cml_nn_avgpool3d(int kernel_size, int stride, int padding, bool ceil_mode,
                            bool count_include_pad) {
    return nn_avgpool3d(kernel_size, stride, padding, ceil_mode, count_include_pad);
}
AdaptiveMaxPool2d* cml_nn_adaptive_maxpool2d(int output_h, int output_w) {
    return nn_adaptive_maxpool2d(output_h, output_w);
}
AdaptiveMaxPool1d* cml_nn_adaptive_maxpool1d(int output_size) {
    return nn_adaptive_maxpool1d(output_size);
}

Tensor* cml_scatter_reduce(Tensor* self, int dim, Tensor* index, Tensor* src, ScatterReduceMode mode) {
    return tensor_scatter_reduce(self, dim, index, src, mode);
}
Tensor* cml_bitcast(Tensor* a, DType target_dtype) {
    return tensor_bitcast(a, target_dtype);
}

QRResult cml_qr(Tensor* a) { return tensor_qr(a); }
SVDResult cml_svd(Tensor* a) { return tensor_svd(a); }

Tensor* cml_from_url(const char* url) { return tensor_from_url(url); }

GGUFContext* cml_gguf_open_read(const char* p) { return gguf_open_read(p); }
GGUFContext* cml_gguf_open_write(const char* p) { return gguf_open_write(p); }
void cml_gguf_close(GGUFContext* c) { gguf_close(c); }
int cml_gguf_write_tensor(GGUFContext* c, const char* n, Tensor* t) { return gguf_write_tensor(c, n, t); }
Tensor* cml_gguf_read_tensor(GGUFContext* c, const char* n) { return gguf_read_tensor(c, n); }
int cml_module_save_gguf(Module* m, const char* p) { return module_save_gguf(m, p); }
int cml_module_load_gguf(Module* m, const char* p) { return module_load_gguf(m, p); }

SafeTensorsContext* cml_safetensors_open_read(const char* p) { return safetensors_open_read(p); }
SafeTensorsContext* cml_safetensors_open_write(const char* p) { return safetensors_open_write(p); }
void cml_safetensors_close(SafeTensorsContext* c) { safetensors_close(c); }
int cml_safetensors_write_tensor(SafeTensorsContext* c, const char* n, Tensor* t) { return safetensors_write_tensor(c, n, t); }
Tensor* cml_safetensors_read_tensor(SafeTensorsContext* c, const char* n) { return safetensors_read_tensor(c, n); }
int cml_module_save_safetensors(Module* m, const char* p) { return module_save_safetensors(m, p); }
int cml_module_load_safetensors(Module* m, const char* p) { return module_load_safetensors(m, p); }

TransformerEncoder* cml_nn_transformer_encoder(int d_model, int nhead, int dim_feedforward,
                                                float dropout, int num_layers,
                                                DType dtype, DeviceType device) {
    return nn_transformer_encoder(d_model, nhead, dim_feedforward, dropout, num_layers, dtype, device);
}
TransformerDecoderLayer* cml_nn_transformer_decoder_layer(int d_model, int nhead, int dim_feedforward,
                                                           float dropout, DType dtype, DeviceType device) {
    return nn_transformer_decoder_layer(d_model, nhead, dim_feedforward, dropout, dtype, device);
}
TransformerDecoder* cml_nn_transformer_decoder(int d_model, int nhead, int dim_feedforward,
                                                float dropout, int num_layers,
                                                DType dtype, DeviceType device) {
    return nn_transformer_decoder(d_model, nhead, dim_feedforward, dropout, num_layers, dtype, device);
}

Optimizer* cml_optim_nadam(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                            float beta1, float beta2, float epsilon) {
    return optim_nadam(parameters, num_parameters, lr, weight_decay, beta1, beta2, epsilon);
}
Optimizer* cml_optim_adamax(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                             float beta1, float beta2, float epsilon) {
    return optim_adamax(parameters, num_parameters, lr, weight_decay, beta1, beta2, epsilon);
}

RNN* cml_nn_rnn(int input_size, int hidden_size, int num_layers, bool bidirectional,
                bool batch_first, float dropout, bool use_bias, DType dtype, DeviceType device) {
    return nn_rnn(input_size, hidden_size, num_layers, bidirectional, batch_first, dropout,
                  use_bias, dtype, device);
}
LSTM* cml_nn_lstm(int input_size, int hidden_size, int num_layers, bool bidirectional,
                  bool batch_first, float dropout, bool use_bias, DType dtype, DeviceType device) {
    return nn_lstm(input_size, hidden_size, num_layers, bidirectional, batch_first, dropout,
                   use_bias, dtype, device);
}
GRU* cml_nn_gru(int input_size, int hidden_size, int num_layers, bool bidirectional,
                bool batch_first, float dropout, bool use_bias, DType dtype, DeviceType device) {
    return nn_gru(input_size, hidden_size, num_layers, bidirectional, batch_first, dropout,
                  use_bias, dtype, device);
}
ConvTranspose3d* cml_nn_conv_transpose3d(int in_channels, int out_channels, int kernel_size,
                                          int stride, int padding, int output_padding,
                                          bool use_bias, DType dtype, DeviceType device) {
    return nn_conv_transpose3d(in_channels, out_channels, kernel_size, stride, padding,
                               output_padding, use_bias, dtype, device);
}
Upsample* cml_nn_upsample(float scale_factor, const int* output_size, int num_output_dims,
                           UpsampleMode mode, bool align_corners) {
    return nn_upsample(scale_factor, output_size, num_output_dims, mode, align_corners);
}
PixelShuffle* cml_nn_pixel_shuffle(int upscale_factor) {
    return nn_pixel_shuffle(upscale_factor);
}
PixelUnshuffle* cml_nn_pixel_unshuffle(int downscale_factor) {
    return nn_pixel_unshuffle(downscale_factor);
}

Flatten* cml_nn_flatten(int start_dim, int end_dim) {
    return nn_flatten(start_dim, end_dim);
}
Identity* cml_nn_identity(void) {
    return nn_identity();
}
BatchNorm1d* cml_nn_batchnorm1d(int num_features, float eps, float momentum, bool affine,
                                 bool track_running_stats, DType dtype, DeviceType device) {
    return nn_batchnorm1d(num_features, eps, momentum, affine, track_running_stats, dtype, device);
}
PReLU* cml_nn_prelu(int num_parameters, float init, DType dtype, DeviceType device) {
    return nn_prelu(num_parameters, init, dtype, device);
}

Tensor* cml_f_interpolate(Tensor* input, int* output_size, int num_dims,
                           UpsampleMode mode, bool align_corners) {
    return f_interpolate(input, output_size, num_dims, mode, align_corners);
}
Tensor* cml_f_pixel_shuffle(Tensor* input, int upscale_factor) {
    return f_pixel_shuffle(input, upscale_factor);
}
Tensor* cml_f_pixel_unshuffle(Tensor* input, int downscale_factor) {
    return f_pixel_unshuffle(input, downscale_factor);
}

void cml_autocast_enter(DType target_dtype) { autocast_enter(target_dtype); }
void cml_autocast_exit(void) { autocast_exit(); }
bool cml_autocast_is_enabled(void) { return autocast_is_enabled(); }
GradScaler* cml_grad_scaler_create(float init_scale, float growth_factor,
                                     float backoff_factor, int growth_interval) {
    return grad_scaler_create(init_scale, growth_factor, backoff_factor, growth_interval);
}
void cml_grad_scaler_free(GradScaler* scaler) { grad_scaler_free(scaler); }
Tensor* cml_grad_scaler_scale(GradScaler* scaler, Tensor* loss) {
    return grad_scaler_scale(scaler, loss);
}
void cml_grad_scaler_unscale(GradScaler* scaler, Parameter** params, int num_params) {
    grad_scaler_unscale(scaler, params, num_params);
}
void cml_grad_scaler_step(GradScaler* scaler, void (*step_fn)(void*), void* optimizer) {
    grad_scaler_step(scaler, step_fn, optimizer);
}
void cml_grad_scaler_update(GradScaler* scaler) { grad_scaler_update(scaler); }

SparseCOOData* cml_sparse_coo_tensor(Tensor* indices, Tensor* values,
                                      const int* dense_shape, int dense_ndim) {
    return sparse_coo_tensor(indices, values, dense_shape, dense_ndim);
}
SparseCOOData* cml_sparse_from_dense(Tensor* dense) { return sparse_from_dense(dense); }
Tensor* cml_sparse_to_dense(SparseCOOData* sparse, const TensorConfig* config) {
    return sparse_to_dense(sparse, config);
}
Tensor* cml_sparse_matmul(SparseCOOData* sparse, Tensor* dense) {
    return sparse_matmul(sparse, dense);
}
SparseCOOData* cml_sparse_coalesce(SparseCOOData* sparse) { return sparse_coalesce(sparse); }
void cml_sparse_free(SparseCOOData* sparse) { sparse_free(sparse); }

Tensor* cml_sort(Tensor* a, int dim, bool descending) { return tensor_sort(a, dim, descending); }
Tensor* cml_topk(Tensor* a, int k, int dim, bool largest, bool sorted) {
    return tensor_topk(a, k, dim, largest, sorted);
}
Tensor* cml_masked_select(Tensor* a, Tensor* mask) { return tensor_masked_select(a, mask); }
Tensor** cml_meshgrid(Tensor** tensors, int num_tensors, int* num_outputs) {
    return tensor_meshgrid(tensors, num_tensors, num_outputs);
}
Tensor* cml_diagonal(Tensor* a, int offset, int dim1, int dim2) {
    return tensor_diagonal(a, offset, dim1, dim2);
}
Tensor* cml_lerp(Tensor* a, Tensor* b, float weight) { return tensor_lerp(a, b, weight); }
Tensor* cml_idiv(Tensor* a, Tensor* b) { return tensor_idiv(a, b); }
Tensor* cml_mod(Tensor* a, Tensor* b) { return tensor_mod(a, b); }

LRScheduler* cml_lr_scheduler_step(Optimizer* opt, int step_size, float gamma) {
    return lr_scheduler_step(opt, step_size, gamma);
}
LRScheduler* cml_lr_scheduler_reduce_on_plateau(Optimizer* opt, float factor, int patience,
                                                  float min_lr) {
    return lr_scheduler_reduce_on_plateau(opt, factor, patience, min_lr);
}
LRScheduler* cml_lr_scheduler_exponential(Optimizer* opt, float gamma) {
    return lr_scheduler_exponential(opt, gamma);
}
LRScheduler* cml_lr_scheduler_cosine(Optimizer* opt, int T_max, float eta_min) {
    return lr_scheduler_cosine(opt, T_max, eta_min);
}
LRScheduler* cml_lr_scheduler_one_cycle(Optimizer* opt, float max_lr, int total_steps,
                                         float pct_start, float div_factor, float final_div_factor) {
    return lr_scheduler_one_cycle(opt, max_lr, total_steps, pct_start, div_factor, final_div_factor);
}
LRScheduler* cml_lr_scheduler_multi_step(Optimizer* opt, int* milestones, int num_milestones,
                                          float gamma) {
    return lr_scheduler_multi_step(opt, milestones, num_milestones, gamma);
}
LRScheduler* cml_lr_scheduler_polynomial(Optimizer* opt, int total_iters, float power,
                                          float min_lr) {
    return lr_scheduler_polynomial(opt, total_iters, power, min_lr);
}
LRScheduler* cml_lr_scheduler_warmup(LRScheduler* inner, int warmup_steps,
                                      float warmup_start_factor) {
    return lr_scheduler_warmup(inner, warmup_steps, warmup_start_factor);
}
float cml_lr_scheduler_update(LRScheduler* scheduler, float metric) {
    return lr_scheduler_update(scheduler, metric);
}
float cml_lr_scheduler_get_lr(LRScheduler* scheduler) {
    return lr_scheduler_get_lr(scheduler);
}
void cml_lr_scheduler_free(LRScheduler* scheduler) {
    lr_scheduler_free(scheduler);
}
