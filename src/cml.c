/**
 * @file cml.c
 * @brief C-ML Library main implementation
 *
 * This file contains the main library initialization, cleanup,
 * and utility functions for the C-ML library.
 */

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
#include "tensor/tensor.h"
#include "tensor/tensor_manipulation.h"
#include "autograd/forward_ops.h"
#include "optim.h"
#include "autograd/autograd.h"
#include "autograd/loss_functions.h"
#include "ops/ir/context.h"
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

// Global cleanup context list for automatic cleanup
static CleanupContext** g_cleanup_contexts = NULL;
static size_t g_num_cleanup_contexts       = 0;
static size_t g_cleanup_contexts_capacity  = 0;

// Global resource tracking for automatic cleanup
static Module** g_tracked_modules = NULL;
static size_t g_num_modules       = 0;
static size_t g_modules_capacity  = 0;

static Optimizer** g_tracked_optimizers = NULL;
static size_t g_num_optimizers          = 0;
static size_t g_optimizers_capacity     = 0;

static Dataset** g_tracked_datasets = NULL;
static size_t g_num_datasets        = 0;
static size_t g_datasets_capacity   = 0;

// Helper functions to track resources for automatic cleanup
// Made non-static so they can be called from other files
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

    // Find and remove the module from the tracking list
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

    // Find and remove the optimizer from the tracking list
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

// Automatic cleanup function registered with atexit
static void cml_auto_cleanup(void) {
    // Only cleanup if not already cleaned up
    if (!g_cml_initialized) {
        return;
    }

    // FIRST: Reset global IR context to detach all tensors from IR
    // This must happen before freeing any tensors to prevent dangling pointers
    cml_ir_reset_global_context();

    // Cleanup all registered cleanup contexts
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

    // Cleanup all tracked datasets
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

    // Cleanup all tracked optimizers
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

    // Cleanup all tracked modules
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

    // Call device_cleanup
    device_cleanup();

    // Cleanup graph context
    cml_graph_context_cleanup();

    // Call cml_cleanup which handles error printing
    // Note: We call it directly here, bypassing reference counting for atexit
    training_metrics_cleanup_global();

    autograd_shutdown();

    // Check and print errors
    if (CML_HAS_ERRORS()) {
        printf("\n=== Errors occurred during execution ===\n");
        error_stack_print_all();
        printf("Last error: %s (code: %d)\n", CML_LAST_ERROR(), CML_LAST_ERROR_CODE());
    } else {
        printf("\n=== Execution completed successfully ===\n");
    }

    // Cleanup error stack
    error_stack_cleanup();

    g_cml_initialized = false;
    g_cml_init_count  = 0;
}

// Function to register cleanup context for automatic cleanup
void cml_register_cleanup_context(CleanupContext* ctx) {
    if (!ctx)
        return;

    // Resize array if needed
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

static void check_and_launch_viz(void) __attribute__((constructor));
static void check_and_launch_viz(void) {
    const char* viz = getenv("VIZ");
    if (!viz || (viz[0] != '1' && strcmp(viz, "true") != 0)) {
        return;
    }

    const char* viz_launched = getenv("CML_VIZ_LAUNCHED");
    if (viz_launched && viz_launched[0] != '\0') {
        return;
    }

    // Platform-specific paths
#ifdef _WIN32
    const char* try_paths[] = {"scripts/viz.py", "../scripts/viz.py", getenv("CML_VIZ_SCRIPT"),
                               NULL};
    // On Windows, we might look in Program Files or similar if we had a standard install location
    // For now, rely on local scripts or env var
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
    // Windows implementation
    GetModuleFileName(NULL, exe_path, sizeof(exe_path));
    _putenv_s("CML_VIZ_LAUNCHED", "1");
    _putenv_s("CML_VIZ", "1");

    // Construct command for Windows
    // We use python directly
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "python \"%s\" \"%s\"", script_path, exe_path);
    system(cmd);

    // Note: On Windows, system() waits for the command to finish.
    // If we want async, we'd need CreateProcess.
    // For viz, blocking might be annoying if it doesn't return.
    // However, the python script spawns subprocesses, so it might be okay
    // if it returns quickly or if we want to block until viz is closed.
    // For now, system() is simple.

#elif defined(__APPLE__)
    // macOS implementation
    uint32_t size = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &size) != 0) {
        // Buffer too small
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
    // The original code used execvp which REPLACES the current process.
    // That means the actual model code wouldn't run!
    // Ah, the original code was:
    // char* viz_argv[] = {"python3", (char*)script_path, exe_path, NULL};
    // execvp("python3", viz_argv);

    // If we execvp, the C program is gone. The python script runs the C program as a subprocess.
    // So yes, we DO want to replace the current process with the python wrapper.
    execvp("python3", viz_argv);

#else
    // Linux implementation - read full command line from /proc/self/cmdline
    char cmdline[4096] = {0};
    char* args[64]     = {0}; // Max 64 arguments
    int argc           = 0;

    FILE* f = fopen("/proc/self/cmdline", "r");
    if (f) {
        size_t cmdlen = fread(cmdline, 1, sizeof(cmdline) - 1, f);
        fclose(f);

        // Parse null-separated arguments
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

    // Get executable path
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

    // Build argv for Python: python3 script.py exe_path [original_args...]
    char python_cmd[] = "python3";
    char script_buf[1024];
    strncpy(script_buf, script_path, sizeof(script_buf) - 1);
    script_buf[sizeof(script_buf) - 1] = '\0';
    char* viz_argv[68]                 = {0};
    viz_argv[0]                        = python_cmd;
    viz_argv[1]                        = script_buf;
    viz_argv[2]                        = exe_path;

    // Add original arguments (skip argv[0] which is the exe itself)
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

/**
 * @brief Initialize the C-ML library
 *
 * This function should be called before using any C-ML functionality.
 * It initializes internal systems, sets up logging, and prepares
 * the library for use.
 *
 * @return 0 on success, negative value on failure
 */
int cml_init(void) {
    if (g_cml_initialized) {
        g_cml_init_count++;
        LOG_DEBUG("C-ML library already initialized, reference count: %d", g_cml_init_count);
        return 0;
    }

    LOG_INFO("Initializing C-ML Library");

    int result = 0;

    // Initialize error stack first
    error_stack_init();

    cml_set_log_level(LOG_LEVEL_ERROR);

    // Initialize global configuration
    // Set defaults: device=CPU, dtype=FLOAT32
    cml_set_default_device(DEVICE_CPU);
    cml_set_default_dtype(DTYPE_FLOAT32);

    // Seed RNG
    cml_random_seed();

    autograd_init();

    training_metrics_init_global();

    // Initialize graph context
    cml_graph_context_init();

    // Register automatic cleanup with atexit (only once)
    if (!g_cml_atexit_registered) {
        atexit(cml_auto_cleanup);
        g_cml_atexit_registered = true;
    }

    if (result == 0) {
        g_cml_initialized = true;
        g_cml_init_count  = 1;
        LOG_INFO("C-ML Library initialized successfully");
    } else {
        LOG_ERROR("Failed to initialize C-ML Library");
    }

    return result;
}

/**
 * @brief Cleanup the C-ML library
 *
 * This function should be called when the library is no longer needed.
 * It cleans up internal resources and ensures proper shutdown.
 *
 * @return 0 on success, negative value on failure
 */
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

    // Mark as not initialized FIRST to prevent atexit from double-freeing
    g_cml_initialized = false;
    g_cml_init_count  = 0;

    // Reset global IR context FIRST to detach all tensors from IR
    // This must happen before freeing any tensors to prevent dangling pointers
    cml_ir_reset_global_context();

    // Cleanup graph context
    cml_graph_context_cleanup();

    // Cleanup all tracked datasets
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

    // Cleanup all tracked optimizers
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

    // Cleanup all tracked modules
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

    // Note: device_cleanup is called automatically by cml_auto_cleanup via atexit
    // But we can also call it here if cleanup is done manually
    device_cleanup();

    training_metrics_cleanup_global();

    autograd_shutdown();

    // Check and print errors before cleanup
    if (CML_HAS_ERRORS()) {
        printf("\n=== Errors occurred during execution ===\n");
        error_stack_print_all();
        printf("Last error: %s (code: %d)\n", CML_LAST_ERROR(), CML_LAST_ERROR_CODE());
    } else {
        printf("\n=== Execution completed successfully ===\n");
    }

    // Cleanup error stack last
    error_stack_cleanup();

    LOG_INFO("C-ML Library cleanup completed");
    return 0;
}

static CMLGlobalErrorHandler g_error_handler = NULL;

void cml_set_error_handler(CMLGlobalErrorHandler handler) { g_error_handler = handler; }

CMLGlobalErrorHandler cml_get_error_handler(void) { return g_error_handler; }

/**
 * @brief Get library version information
 *
 * @param major Pointer to store major version
 * @param minor Pointer to store minor version
 * @param patch Pointer to store patch version
 * @param version_string Pointer to store version string
 */
void cml_get_version(int* major, int* minor, int* patch, const char** version_string) {
    // Version managed by release process - return NULL/0 for now
    if (major)
        *major = 0;
    if (minor)
        *minor = 0;
    if (patch)
        *patch = 0;
    if (version_string)
        *version_string = NULL;
}

/**
 * @brief Get library build information
 *
 * @return String containing build information (compiler, flags, etc.)
 */
const char* cml_get_build_info(void) { return g_build_info; }

/**
 * @brief Check if C-ML library is initialized
 *
 * @return true if initialized, false otherwise
 */
bool cml_is_initialized(void) { return g_cml_initialized; }

/**
 * @brief Get C-ML library initialization count
 *
 * @return Current initialization reference count
 */
int cml_get_init_count(void) { return g_cml_init_count; }

/**
 * @brief Force cleanup of C-ML library (ignores reference count)
 *
 * This function should be used with caution as it forces cleanup
 * regardless of the reference count.
 *
 * @return 0 on success, negative value on failure
 */
int cml_force_cleanup(void) {
    if (!g_cml_initialized) {
        return 0;
    }

    LOG_WARNING("Forcing C-ML library cleanup (reference count was %d)", g_cml_init_count);

    g_cml_init_count = 0;
    return cml_cleanup();
}

/**
 * @brief Print model summary (TensorFlow/Keras style)
 *
 * Prints a summary of the model architecture, showing layers,
 * parameter counts, and total trainable parameters.
 *
 * @param module The module to summarize (can be Sequential, Linear, etc.)
 */
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

    // Export architecture to JSON if VIZ is enabled
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

// Tensor Operations
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

// Neural Network Layers
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

// Optimizers
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

// Loss Functions
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

// Autograd
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

// ============================================================================
// JIT Kernel Cache Management
// ============================================================================

// Forward declarations for kernel cache functions (defined in mlir_kernel_cache.h)
#ifdef CML_HAS_MLIR
struct CMLKernelCache;
struct CMLKernelCache* cml_kernel_cache_get_default(void);
void cml_kernel_cache_clear_impl(struct CMLKernelCache* cache);
void cml_kernel_cache_stats_impl(struct CMLKernelCache* cache, size_t* hits, size_t* misses,
                                 size_t* count, size_t* memory);
double cml_kernel_cache_hit_rate_impl(struct CMLKernelCache* cache);
void cml_kernel_cache_print_stats_impl(struct CMLKernelCache* cache);
#endif

void cml_kernel_cache_clear(void) {
#ifdef CML_HAS_MLIR
    struct CMLKernelCache* cache = cml_kernel_cache_get_default();
    if (cache) {
        cml_kernel_cache_clear_impl(cache);
    }
#endif
}

void cml_kernel_cache_stats(size_t* hits, size_t* misses, size_t* count, size_t* memory) {
#ifdef CML_HAS_MLIR
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
#else
    if (hits)
        *hits = 0;
    if (misses)
        *misses = 0;
    if (count)
        *count = 0;
    if (memory)
        *memory = 0;
#endif
}

double cml_kernel_cache_hit_rate(void) {
#ifdef CML_HAS_MLIR
    struct CMLKernelCache* cache = cml_kernel_cache_get_default();
    if (cache) {
        return cml_kernel_cache_hit_rate_impl(cache);
    }
#endif
    return 0.0;
}

void cml_kernel_cache_print_stats(void) {
#ifdef CML_HAS_MLIR
    struct CMLKernelCache* cache = cml_kernel_cache_get_default();
    if (cache) {
        cml_kernel_cache_print_stats_impl(cache);
    } else {
        printf("Kernel Cache: not initialized\n");
    }
#else
    printf("Kernel Cache: MLIR not compiled in\n");
#endif
}

// Module Operations
Tensor* cml_nn_module_forward(Module* module, Tensor* input) {
    return module_forward(module, input);
}
void cml_nn_module_set_training(Module* module, bool training) {
    module_set_training(module, training);
}
bool cml_nn_module_is_training(Module* module) { return module_is_training(module); }
void cml_nn_module_eval(Module* module) { module_set_training(module, false); }
void cml_nn_module_train(Module* module) { module_set_training(module, true); }
