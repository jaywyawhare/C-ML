#include "cml.h"
#include "tensor/realize.h"
#include "core/logging.h"
#include "core/cml_flags.h"
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
#include "backend/threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
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

/**
 * Register a module for automatic teardown at cleanup/at-exit.
 * @param module (Module*)
 * @return (void)
 */
void cml_track_module(Module* module) {
    if (!module || !g_cml_initialized)
        return;

    if (g_num_modules >= g_modules_capacity) {
        size_t new_capacity = g_modules_capacity == 0 ? 16 : g_modules_capacity * 2;
        Module** new_modules =
            cml_realloc(g_tracked_modules, (size_t)new_capacity * sizeof(Module*));
        if (!new_modules)
            return;
        g_tracked_modules  = new_modules;
        g_modules_capacity = new_capacity;
    }
    g_tracked_modules[g_num_modules++] = module;
}

/**
 * Drop a module from the tracking table so cleanup won't double-free it.
 * @param module (Module*)
 * @return (void)
 */
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

/**
 * Register an optimizer for automatic teardown at cleanup/at-exit.
 * @param optimizer (Optimizer*)
 * @return (void)
 */
void cml_track_optimizer(Optimizer* optimizer) {
    if (!optimizer || !g_cml_initialized)
        return;

    if (g_num_optimizers >= g_optimizers_capacity) {
        size_t new_capacity = g_optimizers_capacity == 0 ? 16 : g_optimizers_capacity * 2;
        Optimizer** new_optimizers =
            cml_realloc(g_tracked_optimizers, (size_t)new_capacity * sizeof(Optimizer*));
        if (!new_optimizers)
            return;
        g_tracked_optimizers  = new_optimizers;
        g_optimizers_capacity = new_capacity;
    }
    g_tracked_optimizers[g_num_optimizers++] = optimizer;
}

/**
 * Drop an optimizer from the tracking table so cleanup won't double-free it.
 * @param optimizer (Optimizer*)
 * @return (void)
 */
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

/**
 * Register a dataset for automatic teardown at cleanup/at-exit.
 * @param dataset (Dataset*)
 * @return (void)
 */
void cml_track_dataset(Dataset* dataset) {
    if (!dataset || !g_cml_initialized)
        return;

    if (g_num_datasets >= g_datasets_capacity) {
        size_t new_capacity = g_datasets_capacity == 0 ? 16 : g_datasets_capacity * 2;
        Dataset** new_datasets =
            cml_realloc(g_tracked_datasets, (size_t)new_capacity * sizeof(Dataset*));
        if (!new_datasets)
            return;
        g_tracked_datasets  = new_datasets;
        g_datasets_capacity = new_capacity;
    }
    g_tracked_datasets[g_num_datasets++] = dataset;
}

/**
 * Counterpart to cml_track_dataset, called from dataset_free. Without it a
 * dataset the caller frees itself stays in the tracking table, and the
 * at-exit sweep frees it a second time.
 * @param dataset (Dataset*)
 * @return (void)
 */
void cml_untrack_dataset(Dataset* dataset) {
    if (!dataset || !g_tracked_datasets)
        return;

    for (size_t i = 0; i < g_num_datasets; i++) {
        if (g_tracked_datasets[i] == dataset) {
            g_tracked_datasets[i] = NULL;
            return;
        }
    }
}

/**
 * atexit handler: detach the IR context, then free every tracked resource.
 * @return (void)
 */
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
        cml_free(g_cleanup_contexts);
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
        cml_free(g_tracked_datasets);
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
        cml_free(g_tracked_optimizers);
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
        cml_free(g_tracked_modules);
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
    cml_exec_pool_shutdown();

    g_cml_initialized = false;
    g_cml_init_count  = 0;
}

/**
 * Register a cleanup context to be freed during the at-exit sweep.
 * @param ctx (CleanupContext*)
 * @return (void)
 */
void cml_register_cleanup_context(CleanupContext* ctx) {
    if (!ctx)
        return;

    if (g_num_cleanup_contexts >= g_cleanup_contexts_capacity) {
        size_t new_capacity =
            g_cleanup_contexts_capacity == 0 ? 16 : g_cleanup_contexts_capacity * 2;
        CleanupContext** new_contexts =
            cml_realloc(g_cleanup_contexts, (size_t)new_capacity * sizeof(CleanupContext*));
        if (!new_contexts)
            return;
        g_cleanup_contexts          = new_contexts;
        g_cleanup_contexts_capacity = new_capacity;
    }

    g_cleanup_contexts[g_num_cleanup_contexts++] = ctx;
}

/**
 * Constructor disabled: call check_and_launch_viz() from cml_init() instead
 * @return (void)
 */
static void check_and_launch_viz(void) {
    if (!cml_flag_enabled(CML_FLAG_VIZ))
        return;

    const char* viz_launched = getenv("VIZ_LAUNCHED");
    if (viz_launched && viz_launched[0] != '\0') {
        return;
    }

#ifdef _WIN32
    const char* try_paths[] = {"scripts/viz.py", "../scripts/viz.py", getenv("VIZ_SCRIPT"), NULL};
#else
    const char* try_paths[] = {
        "scripts/viz.py",        "../scripts/viz.py",  "/usr/local/share/cml/viz.py",
        "/usr/share/cml/viz.py", getenv("VIZ_SCRIPT"), NULL};
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
    _putenv_s("VIZ_LAUNCHED", "1");
    _putenv_s("VIZ", "1");

    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "python \"%s\" \"%s\"", script_path, exe_path);
    if (system(cmd) != 0) { /* best-effort launch of the viz script */
    }

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

    setenv("VIZ_LAUNCHED", "1", 1);
    setenv("VIZ", "1", 1);

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

    setenv("VIZ_LAUNCHED", "1", 1);
    setenv("VIZ", "1", 1);

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

/**
 * Map the DEFAULT_FLOAT env var (FLOAT32/FLOAT/HALF/FLOAT16/BFLOAT16/FLOAT64/
 * DOUBLE) to a DType, mirroring tinygrad's DEFAULT_FLOAT. Falls back to
 * DTYPE_FLOAT32 when unset or unrecognized.
 * @return (DType)
 */
static DType cml_default_float_from_env(void) {
    const char* v = getenv("DEFAULT_FLOAT");
    if (!v || !*v)
        return DTYPE_FLOAT32;
    if (strcasecmp(v, "HALF") == 0 || strcasecmp(v, "FLOAT16") == 0)
        return DTYPE_FLOAT16;
    if (strcasecmp(v, "BFLOAT16") == 0)
        return DTYPE_BFLOAT16;
    if (strcasecmp(v, "FLOAT64") == 0 || strcasecmp(v, "DOUBLE") == 0)
        return DTYPE_FLOAT64;
    if (strcasecmp(v, "FLOAT32") == 0 || strcasecmp(v, "FLOAT") == 0)
        return DTYPE_FLOAT32;
    LOG_WARNING("Unrecognized DEFAULT_FLOAT='%s', using FLOAT32", v);
    return DTYPE_FLOAT32;
}

/**
 * Initialize the library (flags, logging, device, RNG, autograd, thread pool);
 * reference-counted and idempotent. Returns 0 on success, -1 on error.
 * @return (int)
 */
int cml_init(void) {
    if (g_cml_initialized) {
        g_cml_init_count++;
        LOG_DEBUG("C-ML library already initialized, reference count: %d", g_cml_init_count);
        return 0;
    }

    LOG_INFO("Initializing C-ML Library");

    int result = 0;

    error_stack_init();

    /* Read the central flag registry from the environment before anything else
     * so DEBUG/NOOPT/etc. take effect for the rest of initialization. */
    cml_flags_init();

    /* VIZ and NO_EXPORT ask for opposite things: one wants the dashboard files,
     * the other guarantees none are written. Honouring a precedence rule would
     * mean a run started with VIZ=1 silently producing nothing, with no hint
     * why -- so reject the combination instead of quietly picking a winner. */
    if (cml_flag_enabled(CML_FLAG_VIZ) && cml_flag_enabled(CML_FLAG_NO_EXPORT)) {
        LOG_ERROR("VIZ=1 and NO_EXPORT=1 are mutually exclusive: VIZ asks for the "
                  "dashboard exports, NO_EXPORT guarantees no files are written. "
                  "Set exactly one.");
        error_stack_push(CM_INVALID_ARGUMENT, "VIZ and NO_EXPORT are mutually exclusive", __FILE__,
                         __LINE__, __func__);
        return -1;
    }

    /* DEBUG=n raises the log verbosity (default is ERROR-only):
     *   0 -> ERROR (quiet), 1-2 -> INFO, >=3 -> DEBUG (everything). */
    int debug = cml_flag(CML_FLAG_DEBUG);
    cml_set_log_level(debug >= 3 ? LOG_LEVEL_DEBUG : debug >= 1 ? LOG_LEVEL_INFO : LOG_LEVEL_ERROR);
    if (debug >= 1)
        cml_flags_dump(stderr);

    cml_set_default_device(DEVICE_CPU);

    /* DEFAULT_FLOAT overrides the default float dtype (tinygrad-style). */
    cml_set_default_dtype(cml_default_float_from_env());

    cml_random_seed();

    autograd_init();

    training_metrics_init_global();

    cml_graph_context_init();

    /* Global worker pool for parallel elementwise/reduction kernels
     * (simd_*_parallel). Thread count: CML_THREADS, else auto-detect. */
    {
        const char* threads_env = getenv("CML_THREADS");
        size_t threads   = (threads_env && atoi(threads_env) > 0) ? (size_t)atoi(threads_env) : 0;
        ThreadPool* pool = threadpool_create(threads);
        if (pool) {
            threadpool_set_global(pool);
            LOG_DEBUG("C-ML thread pool started (%zu workers)", threadpool_get_num_threads(pool));
        } else {
            LOG_WARNING("Thread pool unavailable; elementwise kernels run serial");
        }
    }

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

/**
 * Reference-counted teardown; frees all tracked resources once the count
 * reaches zero. Returns 0 (no-op if not initialized).
 * @return (int)
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

    g_cml_initialized = false;
    g_cml_init_count  = 0;

    /* Stop workers before any teardown touches shared state.
     * threadpool_set_global(NULL) destroys the previous pool. */
    threadpool_set_global(NULL);

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
        cml_free(g_tracked_datasets);
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
        cml_free(g_tracked_optimizers);
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
        cml_free(g_tracked_modules);
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
    cml_exec_pool_shutdown();

    LOG_INFO("C-ML Library cleanup completed");
    return 0;
}

static CMLGlobalErrorHandler g_error_handler = NULL;

/**
 * Error-stack notify shim: forwards pushed errors to the user error handler.
 * @param error_code (int)
 * @param error_msg (const char*)
 * @param context (void*)
 * @return (void)
 */
static void cml_error_stack_notify(int error_code, const char* error_msg, void* context) {
    (void)context;
    if (g_error_handler)
        (void)g_error_handler(error_code, error_msg, NULL);
}

/**
 * Install (or clear, with NULL) the global callback invoked on each error.
 * @param handler (CMLGlobalErrorHandler)
 * @return (void)
 */
void cml_set_error_handler(CMLGlobalErrorHandler handler) {
    g_error_handler = handler;
    error_stack_set_notify(handler ? cml_error_stack_notify : NULL, NULL);
}

/**
 * Return the currently installed global error handler (or NULL).
 * @return (CMLGlobalErrorHandler)
 */
CMLGlobalErrorHandler cml_get_error_handler(void) { return g_error_handler; }

/**
 * Write the library version components into any non-NULL out-parameters.
 * @param major (int*)
 * @param minor (int*)
 * @param patch (int*)
 * @param version_string (const char**)
 * @return (void)
 */
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

/**
 * Return the packed integer version (CML_VERSION).
 * @return (int)
 */
int cml_version(void) { return CML_VERSION; }

/**
 * Return the human-readable version string.
 * @return (const char*)
 */
const char* cml_version_string(void) { return CML_VERSION_STRING; }

/**
 * Return the static build-info banner (library name and feature list).
 * @return (const char*)
 */
const char* cml_get_build_info(void) { return g_build_info; }

/**
 * True if the library has been initialized.
 * @return (bool)
 */
bool cml_is_initialized(void) { return g_cml_initialized; }

/**
 * Current init/cleanup reference count.
 * @return (int)
 */
int cml_get_init_count(void) { return g_cml_init_count; }

/**
 * Force teardown regardless of reference count (resets it to zero first).
 * @return (int)
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
 * Recursively print one summary row per layer (flattening Sequential), with its
 * parameter count; helper for cml_summary.
 * @param module (Module*)
 * @param indent (int)
 * @param layer_num (int*)
 * @return (void)
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

/**
 * Print a Keras-style model summary (layers, types, parameter totals); also
 * exports the architecture JSON when visualization is enabled.
 * @param module (Module*)
 * @return (void)
 */
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
            cml_free(params);
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

    if (cml_viz_enabled()) {
        ModelArchitecture* arch = model_architecture_create();
        if (arch) {
            if (model_architecture_extract(module, arch) == 0) {
                if (!cml_flag_enabled(CML_FLAG_NO_EXPORT)) {
                    model_architecture_export_json(arch, "model_architecture.json");
                    LOG_INFO("Exported model architecture to model_architecture.json");
                }
            }
            model_architecture_free(arch);
        }
    }
}

/**
 * Public API: uninitialized tensor of the given shape (-> tensor_empty).
 * @param shape (int*)
 * @param ndim (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_empty(int* shape, int ndim, const TensorConfig* config) {
    return tensor_empty(shape, ndim, config);
}

/**
 * Public API: zero-filled tensor of the given shape (-> tensor_zeros).
 * @param shape (int*)
 * @param ndim (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_zeros(int* shape, int ndim, const TensorConfig* config) {
    return tensor_zeros(shape, ndim, config);
}

/**
 * Public API: ones-filled tensor of the given shape (-> tensor_ones).
 * @param shape (int*)
 * @param ndim (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_ones(int* shape, int ndim, const TensorConfig* config) {
    return tensor_ones(shape, ndim, config);
}

/**
 * Public API: tensor filled with a constant value (-> tensor_full).
 * @param shape (int*)
 * @param ndim (int)
 * @param config (const TensorConfig*)
 * @param value (float)
 * @return (Tensor*)
 */
Tensor* cml_full(int* shape, int ndim, const TensorConfig* config, float value) {
    return tensor_full(shape, ndim, config, value);
}

/**
 * Public API: tensor copied from a raw data buffer (-> tensor_from_data).
 * @param data (void*)
 * @param shape (int*)
 * @param ndim (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_tensor(void* data, int* shape, int ndim, const TensorConfig* config) {
    return tensor_from_data(data, shape, ndim, config);
}

/**
 * Public API: 2-D zero-filled tensor (-> tensor_zeros_2d).
 * @param rows (int)
 * @param cols (int)
 * @return (Tensor*)
 */
Tensor* cml_zeros_2d(int rows, int cols) { return tensor_zeros_2d(rows, cols); }

/**
 * Public API: 2-D ones-filled tensor (-> tensor_ones_2d).
 * @param rows (int)
 * @param cols (int)
 * @return (Tensor*)
 */
Tensor* cml_ones_2d(int rows, int cols) { return tensor_ones_2d(rows, cols); }

/**
 * Public API: 2-D uninitialized tensor (-> tensor_empty_2d).
 * @param rows (int)
 * @param cols (int)
 * @return (Tensor*)
 */
Tensor* cml_empty_2d(int rows, int cols) { return tensor_empty_2d(rows, cols); }

/**
 * Public API: 2-D tensor copied from a row-major array (-> tensor_from_array_2d).
 * @param data (const float*)
 * @param rows (int)
 * @param cols (int)
 * @return (Tensor*)
 */
Tensor* cml_tensor_2d(const float* data, int rows, int cols) {
    return tensor_from_array_2d(data, rows, cols);
}

/**
 * Public API: 1-D zero-filled tensor (-> tensor_zeros).
 * @param size (int)
 * @return (Tensor*)
 */
Tensor* cml_zeros_1d(int size) {
    int shape[] = {size};
    return tensor_zeros(shape, 1, NULL);
}

/**
 * Public API: 1-D ones-filled tensor (-> tensor_ones).
 * @param size (int)
 * @return (Tensor*)
 */
Tensor* cml_ones_1d(int size) {
    int shape[] = {size};
    return tensor_ones(shape, 1, NULL);
}

/**
 * Public API: 1-D uninitialized tensor (-> tensor_empty).
 * @param size (int)
 * @return (Tensor*)
 */
Tensor* cml_empty_1d(int size) {
    int shape[] = {size};
    return tensor_empty(shape, 1, NULL);
}

/**
 * Public API: 1-D tensor copied from an array (-> tensor_from_data).
 * @param data (const float*)
 * @param size (int)
 * @return (Tensor*)
 */
Tensor* cml_tensor_1d(const float* data, int size) {
    int shape[] = {size};
    return tensor_from_data(data, shape, 1, NULL);
}

/**
 * Public API: elementwise add (-> tensor_add).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_add(Tensor* a, Tensor* b) { return tensor_add(a, b); }
/**
 * Public API: elementwise subtract (-> tensor_sub).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_sub(Tensor* a, Tensor* b) { return tensor_sub(a, b); }
/**
 * Public API: elementwise multiply (-> tensor_mul).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_mul(Tensor* a, Tensor* b) { return tensor_mul(a, b); }
/**
 * Public API: elementwise divide (-> tensor_div).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_div(Tensor* a, Tensor* b) { return tensor_div(a, b); }
/**
 * Public API: in-place elementwise add (-> tensor_add_).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_add_(Tensor* a, Tensor* b) { return tensor_add_(a, b); }
/**
 * Public API: in-place elementwise subtract (-> tensor_sub_).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_sub_(Tensor* a, Tensor* b) { return tensor_sub_(a, b); }
/**
 * Public API: in-place elementwise multiply (-> tensor_mul_).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_mul_(Tensor* a, Tensor* b) { return tensor_mul_(a, b); }
/**
 * Public API: in-place elementwise divide (-> tensor_div_).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_div_(Tensor* a, Tensor* b) { return tensor_div_(a, b); }
/**
 * Public API: elementwise exponential (-> tensor_exp).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_exp(Tensor* a) { return tensor_exp(a); }
/**
 * Public API: elementwise natural log (-> tensor_log).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_log(Tensor* a) { return tensor_log(a); }
/**
 * Public API: elementwise square root (-> tensor_sqrt).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_sqrt(Tensor* a) { return tensor_sqrt(a); }
/**
 * Public API: elementwise sine (-> tensor_sin).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_sin(Tensor* a) { return tensor_sin(a); }
/**
 * Public API: elementwise cosine (-> tensor_cos).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_cos(Tensor* a) { return tensor_cos(a); }
/**
 * Public API: elementwise tangent (-> tensor_tan).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_tan(Tensor* a) { return tensor_tan(a); }
/**
 * Public API: elementwise power (-> tensor_pow).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_pow(Tensor* a, Tensor* b) { return tensor_pow(a, b); }
/**
 * Public API: ReLU activation (-> tensor_relu).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_relu(Tensor* a) { return tensor_relu(a); }
/**
 * Public API: sigmoid activation (-> tensor_sigmoid).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_sigmoid(Tensor* a) { return tensor_sigmoid(a); }
/**
 * Public API: tanh activation (-> tensor_tanh).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_tanh(Tensor* a) { return tensor_tanh(a); }
/**
 * Public API: softmax over a dim (-> tensor_softmax).
 * @param a (Tensor*)
 * @param dim (int)
 * @return (Tensor*)
 */
Tensor* cml_softmax(Tensor* a, int dim) { return tensor_softmax(a, dim); }
/**
 * Public API: ELU activation (-> tensor_elu).
 * @param x (Tensor*)
 * @param alpha (float)
 * @return (Tensor*)
 */
Tensor* cml_elu(Tensor* x, float alpha) { return tensor_elu(x, alpha); }
/**
 * Public API: SELU activation (-> tensor_selu).
 * @param x (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_selu(Tensor* x) { return tensor_selu(x); }
/**
 * Public API: Mish activation (-> tensor_mish).
 * @param x (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_mish(Tensor* x) { return tensor_mish(x); }
/**
 * Public API: SiLU/swish activation (-> tensor_silu).
 * @param x (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_silu(Tensor* x) { return tensor_silu(x); }
/**
 * Public API: hard-swish activation (-> tensor_hardswish).
 * @param x (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_hardswish(Tensor* x) { return tensor_hardswish(x); }
/**
 * Public API: leaky ReLU activation (-> tensor_leaky_relu).
 * @param x (Tensor*)
 * @param negative_slope (float)
 * @return (Tensor*)
 */
Tensor* cml_leaky_relu(Tensor* x, float negative_slope) {
    return tensor_leaky_relu(x, negative_slope);
}
/**
 * Public API: sum reduction over a dim (-> tensor_sum).
 * @param a (Tensor*)
 * @param dim (int)
 * @param keepdim (bool)
 * @return (Tensor*)
 */
Tensor* cml_sum(Tensor* a, int dim, bool keepdim) { return tensor_sum(a, dim, keepdim); }
/**
 * Public API: mean reduction over a dim (-> tensor_mean).
 * @param a (Tensor*)
 * @param dim (int)
 * @param keepdim (bool)
 * @return (Tensor*)
 */
Tensor* cml_mean(Tensor* a, int dim, bool keepdim) { return tensor_mean(a, dim, keepdim); }
/**
 * Public API: max reduction over a dim (-> tensor_max).
 * @param a (Tensor*)
 * @param dim (int)
 * @param keepdim (bool)
 * @return (Tensor*)
 */
Tensor* cml_max(Tensor* a, int dim, bool keepdim) { return tensor_max(a, dim, keepdim); }
/**
 * Public API: min reduction over a dim (-> tensor_min).
 * @param a (Tensor*)
 * @param dim (int)
 * @param keepdim (bool)
 * @return (Tensor*)
 */
Tensor* cml_min(Tensor* a, int dim, bool keepdim) { return tensor_min(a, dim, keepdim); }
/**
 * Public API: matrix multiply (-> tensor_matmul).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_matmul(Tensor* a, Tensor* b) { return tensor_matmul(a, b); }
/**
 * Public API: swap two dims (-> tensor_transpose).
 * @param a (Tensor*)
 * @param dim1 (int)
 * @param dim2 (int)
 * @return (Tensor*)
 */
Tensor* cml_transpose(Tensor* a, int dim1, int dim2) { return tensor_transpose(a, dim1, dim2); }
/**
 * Public API: reshape as a differentiable graph node (-> uop_reshape).
 * @param a (Tensor*)
 * @param new_shape (int*)
 * @param new_ndim (int)
 * @return (Tensor*)
 */
Tensor* cml_reshape(Tensor* a, int* new_shape, int new_ndim) {
    if (!a || !new_shape || new_ndim <= 0)
        return NULL;
    /* Route through uop_reshape so the reshape is a differentiable graph node
     * (or an autograd-attached view), not a bare tensor_reshape view that
     * severs the backward graph -- gradients would not reach producers of the
     * reshaped tensor (e.g. attention/transformer blocks did not train). */
    ReshapeParams p = {.new_shape = new_shape, .new_ndim = new_ndim};
    return uop_reshape(a, &p);
}
/**
 * Public API: deep copy of a tensor (-> tensor_clone).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_clone(Tensor* a) { return tensor_clone(a); }
/**
 * Public API: detach from the autograd graph (-> tensor_detach).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_detach(Tensor* a) { return tensor_detach(a); }
/**
 * Public API: concatenate tensors along a dim (-> tensor_concat).
 * @param tensors (Tensor**)
 * @param num_tensors (int)
 * @param dim (int)
 * @return (Tensor*)
 */
Tensor* cml_concat(Tensor** tensors, int num_tensors, int dim) {
    return tensor_concat(tensors, num_tensors, dim);
}
/**
 * Public API: stack tensors along a new dim (-> tensor_stack).
 * @param tensors (Tensor**)
 * @param num_tensors (int)
 * @param dim (int)
 * @return (Tensor*)
 */
Tensor* cml_stack(Tensor** tensors, int num_tensors, int dim) {
    return tensor_stack(tensors, num_tensors, dim);
}
/**
 * Public API: elementwise select by condition (-> tensor_where).
 * @param condition (Tensor*)
 * @param x (Tensor*)
 * @param y (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_where(Tensor* condition, Tensor* x, Tensor* y) { return tensor_where(condition, x, y); }
/**
 * Public API: Einstein-summation contraction (-> tensor_einsum).
 * @param equation (const char*)
 * @param tensors (Tensor**)
 * @param num_tensors (int)
 * @return (Tensor*)
 */
Tensor* cml_einsum(const char* equation, Tensor** tensors, int num_tensors) {
    return tensor_einsum(equation, tensors, num_tensors);
}
/**
 * Public API: circular shift along an axis (-> tensor_roll).
 * @param a (Tensor*)
 * @param shift (int)
 * @param axis (int)
 * @return (Tensor*)
 */
Tensor* cml_roll(Tensor* a, int shift, int axis) { return tensor_roll(a, shift, axis); }
/**
 * Public API: copy sign of b onto magnitude of a (-> tensor_copysign).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_copysign(Tensor* a, Tensor* b) { return tensor_copysign(a, b); }
/**
 * Public API: log(exp(a)+exp(b)) elementwise (-> tensor_logaddexp).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_logaddexp(Tensor* a, Tensor* b) { return tensor_logaddexp(a, b); }
/**
 * Public API: one-hot encode integer indices (-> tensor_one_hot).
 * @param indices (Tensor*)
 * @param num_classes (int)
 * @return (Tensor*)
 */
Tensor* cml_one_hot(Tensor* indices, int num_classes) {
    return tensor_one_hot(indices, num_classes);
}

/**
 * Public API: create an empty Sequential container (-> nn_sequential).
 * @return (Sequential*)
 */
Sequential* cml_nn_sequential(void) { return nn_sequential(); }
/**
 * Public API: append a layer to a Sequential, returning it (-> sequential_add_chain).
 * @param seq (Sequential*)
 * @param layer (Module*)
 * @return (Sequential*)
 */
Sequential* cml_nn_sequential_add(Sequential* seq, Module* layer) {
    return sequential_add_chain(seq, layer);
}
/**
 * Public API: run a Sequential's forward pass (-> module_forward).
 * @param seq (Sequential*)
 * @param input (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_nn_sequential_forward(Sequential* seq, Tensor* input) {
    return module_forward((Module*)seq, input);
}
/**
 * Public API: create a Linear (fully-connected) layer (-> nn_linear).
 * @param in_features (int)
 * @param out_features (int)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @param bias (bool)
 * @return (Linear*)
 */
Linear* cml_nn_linear(int in_features, int out_features, DType dtype, DeviceType device,
                      bool bias) {
    return nn_linear(in_features, out_features, dtype, device, bias);
}
/**
 * Public API: create a ReLU layer (-> nn_relu).
 * @param inplace (bool)
 * @return (ReLU*)
 */
ReLU* cml_nn_relu(bool inplace) { return nn_relu(inplace); }
/**
 * Public API: create a Sigmoid layer (-> nn_sigmoid).
 * @return (Sigmoid*)
 */
Sigmoid* cml_nn_sigmoid(void) { return nn_sigmoid(); }
/**
 * Public API: create a Tanh layer (-> nn_tanh).
 * @return (Tanh*)
 */
Tanh* cml_nn_tanh(void) { return nn_tanh(); }
/**
 * Public API: create a LeakyReLU layer (-> nn_leaky_relu).
 * @param negative_slope (float)
 * @param inplace (bool)
 * @return (LeakyReLU*)
 */
LeakyReLU* cml_nn_leaky_relu(float negative_slope, bool inplace) {
    return nn_leaky_relu(negative_slope, inplace);
}
/**
 * Public API: create a Dropout layer (-> nn_dropout).
 * @param p (float)
 * @param inplace (bool)
 * @return (Dropout*)
 */
Dropout* cml_nn_dropout(float p, bool inplace) { return nn_dropout(p, inplace); }
/**
 * Public API: create a 2-D convolution layer (-> nn_conv2d).
 * @param in_channels (int)
 * @param out_channels (int)
 * @param kernel_size (int)
 * @param stride (int)
 * @param padding (int)
 * @param dilation (int)
 * @param bias (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (Conv2d*)
 */
Conv2d* cml_nn_conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                      int dilation, bool bias, DType dtype, DeviceType device) {
    return nn_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, dtype,
                     device);
}
/**
 * Public API: create a 2-D batch-norm layer (-> nn_batchnorm2d).
 * @param num_features (int)
 * @param eps (float)
 * @param momentum (float)
 * @param affine (bool)
 * @param track_running_stats (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (BatchNorm2d*)
 */
BatchNorm2d* cml_nn_batchnorm2d(int num_features, float eps, float momentum, bool affine,
                                bool track_running_stats, DType dtype, DeviceType device) {
    return nn_batchnorm2d(num_features, eps, momentum, affine, track_running_stats, dtype, device);
}
/**
 * Public API: create a LayerNorm layer (-> nn_layernorm).
 * @param normalized_shape (int)
 * @param eps (float)
 * @param affine (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (LayerNorm*)
 */
LayerNorm* cml_nn_layernorm(int normalized_shape, float eps, bool affine, DType dtype,
                            DeviceType device) {
    return nn_layernorm(normalized_shape, eps, affine, dtype, device);
}
/**
 * Public API: create a 2-D max-pool layer (-> nn_maxpool2d).
 * @param kernel_size (int)
 * @param stride (int)
 * @param padding (int)
 * @param dilation (int)
 * @param ceil_mode (bool)
 * @return (MaxPool2d*)
 */
MaxPool2d* cml_nn_maxpool2d(int kernel_size, int stride, int padding, int dilation,
                            bool ceil_mode) {
    return nn_maxpool2d(kernel_size, stride, padding, dilation, ceil_mode);
}
/**
 * Public API: create a 2-D average-pool layer (-> nn_avgpool2d).
 * @param kernel_size (int)
 * @param stride (int)
 * @param padding (int)
 * @param ceil_mode (bool)
 * @param count_include_pad (bool)
 * @return (AvgPool2d*)
 */
AvgPool2d* cml_nn_avgpool2d(int kernel_size, int stride, int padding, bool ceil_mode,
                            bool count_include_pad) {
    return nn_avgpool2d(kernel_size, stride, padding, ceil_mode, count_include_pad);
}

/**
 * Public API: create a 1-D convolution layer (-> nn_conv1d).
 * @param in_channels (int)
 * @param out_channels (int)
 * @param kernel_size (int)
 * @param stride (int)
 * @param padding (int)
 * @param dilation (int)
 * @param use_bias (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (Conv1d*)
 */
Conv1d* cml_nn_conv1d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                      int dilation, bool use_bias, DType dtype, DeviceType device) {
    return nn_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias,
                     dtype, device);
}
/**
 * Public API: create a 3-D convolution layer (-> nn_conv3d).
 * @param in_channels (int)
 * @param out_channels (int)
 * @param kernel_size (int)
 * @param stride (int)
 * @param padding (int)
 * @param dilation (int)
 * @param use_bias (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (Conv3d*)
 */
Conv3d* cml_nn_conv3d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                      int dilation, bool use_bias, DType dtype, DeviceType device) {
    return nn_conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias,
                     dtype, device);
}
/**
 * Public API: create an Embedding lookup layer (-> nn_embedding).
 * @param num_embeddings (int)
 * @param embedding_dim (int)
 * @param padding_idx (int)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (Embedding*)
 */
Embedding* cml_nn_embedding(int num_embeddings, int embedding_dim, int padding_idx, DType dtype,
                            DeviceType device) {
    return nn_embedding(num_embeddings, embedding_dim, padding_idx, dtype, device);
}
/**
 * Public API: create a GroupNorm layer (-> nn_groupnorm).
 * @param num_groups (int)
 * @param num_channels (int)
 * @param eps (float)
 * @param affine (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (GroupNorm*)
 */
GroupNorm* cml_nn_groupnorm(int num_groups, int num_channels, float eps, bool affine, DType dtype,
                            DeviceType device) {
    return nn_groupnorm(num_groups, num_channels, eps, affine, dtype, device);
}
/**
 * Public API: create a vanilla RNN cell (-> nn_rnn_cell).
 * @param input_size (int)
 * @param hidden_size (int)
 * @param use_bias (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (RNNCell*)
 */
RNNCell* cml_nn_rnn_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                         DeviceType device) {
    return nn_rnn_cell(input_size, hidden_size, use_bias, dtype, device);
}
/**
 * Public API: create an LSTM cell (-> nn_lstm_cell).
 * @param input_size (int)
 * @param hidden_size (int)
 * @param use_bias (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (LSTMCell*)
 */
LSTMCell* cml_nn_lstm_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                           DeviceType device) {
    return nn_lstm_cell(input_size, hidden_size, use_bias, dtype, device);
}
/**
 * Public API: create a GRU cell (-> nn_gru_cell).
 * @param input_size (int)
 * @param hidden_size (int)
 * @param use_bias (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (GRUCell*)
 */
GRUCell* cml_nn_gru_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                         DeviceType device) {
    return nn_gru_cell(input_size, hidden_size, use_bias, dtype, device);
}
/**
 * Public API: create a multi-head attention layer (-> nn_multihead_attention).
 * @param embed_dim (int)
 * @param num_heads (int)
 * @param dropout (float)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (MultiHeadAttention*)
 */
MultiHeadAttention* cml_nn_multihead_attention(int embed_dim, int num_heads, float dropout,
                                               DType dtype, DeviceType device) {
    return nn_multihead_attention(embed_dim, num_heads, dropout, dtype, device);
}
/**
 * Public API: create a transformer encoder layer (-> nn_transformer_encoder_layer).
 * @param d_model (int)
 * @param nhead (int)
 * @param dim_feedforward (int)
 * @param dropout (float)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (TransformerEncoderLayer*)
 */
TransformerEncoderLayer* cml_nn_transformer_encoder_layer(int d_model, int nhead,
                                                          int dim_feedforward, float dropout,
                                                          DType dtype, DeviceType device) {
    return nn_transformer_encoder_layer(d_model, nhead, dim_feedforward, dropout, dtype, device);
}
/**
 * Public API: create an empty ModuleList container (-> nn_module_list).
 * @return (ModuleList*)
 */
ModuleList* cml_nn_module_list(void) { return nn_module_list(); }
/**
 * Public API: create an empty ModuleDict container (-> nn_module_dict).
 * @return (ModuleDict*)
 */
ModuleDict* cml_nn_module_dict(void) { return nn_module_dict(); }

/**
 * Public API: create an Adam optimizer over a parameter list (-> optim_adam).
 * @param parameters (Parameter**)
 * @param num_parameters (int)
 * @param lr (float)
 * @param weight_decay (float)
 * @param beta1 (float)
 * @param beta2 (float)
 * @param eps (float)
 * @return (Optimizer*)
 */
Optimizer* cml_optim_adam(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                          float beta1, float beta2, float eps) {
    return optim_adam(parameters, num_parameters, lr, weight_decay, beta1, beta2, eps);
}
/**
 * Public API: create an SGD optimizer over a parameter list (-> optim_sgd).
 * @param parameters (Parameter**)
 * @param num_parameters (int)
 * @param lr (float)
 * @param momentum (float)
 * @param weight_decay (float)
 * @return (Optimizer*)
 */
Optimizer* cml_optim_sgd(Parameter** parameters, int num_parameters, float lr, float momentum,
                         float weight_decay) {
    return optim_sgd(parameters, num_parameters, lr, momentum, weight_decay);
}
/**
 * Public API: create an RMSprop optimizer over a parameter list (-> optim_rmsprop).
 * @param parameters (Parameter**)
 * @param num_parameters (int)
 * @param lr (float)
 * @param weight_decay (float)
 * @param alpha (float)
 * @param eps (float)
 * @return (Optimizer*)
 */
Optimizer* cml_optim_rmsprop(Parameter** parameters, int num_parameters, float lr,
                             float weight_decay, float alpha, float eps) {
    return optim_rmsprop(parameters, num_parameters, lr, weight_decay, alpha, eps);
}
/**
 * Public API: create an Adagrad optimizer over a parameter list (-> optim_adagrad).
 * @param parameters (Parameter**)
 * @param num_parameters (int)
 * @param lr (float)
 * @param weight_decay (float)
 * @param eps (float)
 * @return (Optimizer*)
 */
Optimizer* cml_optim_adagrad(Parameter** parameters, int num_parameters, float lr,
                             float weight_decay, float eps) {
    return optim_adagrad(parameters, num_parameters, lr, weight_decay, eps);
}
/**
 * Public API: create an Adam optimizer over a model's parameters (-> optim_adam_for_model).
 * @param model (Module*)
 * @param lr (float)
 * @param weight_decay (float)
 * @param beta1 (float)
 * @param beta2 (float)
 * @param eps (float)
 * @return (Optimizer*)
 */
Optimizer* cml_optim_adam_for_model(Module* model, float lr, float weight_decay, float beta1,
                                    float beta2, float eps) {
    return optim_adam_for_model(model, lr, weight_decay, beta1, beta2, eps);
}
/**
 * Public API: create an SGD optimizer over a model's parameters (-> optim_sgd_for_model).
 * @param model (Module*)
 * @param lr (float)
 * @param momentum (float)
 * @param weight_decay (float)
 * @return (Optimizer*)
 */
Optimizer* cml_optim_sgd_for_model(Module* model, float lr, float momentum, float weight_decay) {
    return optim_sgd_for_model(model, lr, momentum, weight_decay);
}
/**
 * Public API: zero all parameter gradients (-> optimizer_zero_grad).
 * @param optimizer (Optimizer*)
 * @return (void)
 */
void cml_optim_zero_grad(Optimizer* optimizer) { optimizer_zero_grad(optimizer); }
/**
 * Public API: apply one optimizer update step (-> optimizer_step).
 * @param optimizer (Optimizer*)
 * @return (void)
 */
void cml_optim_step(Optimizer* optimizer) { optimizer_step(optimizer); }

/**
 * Public API: mean-squared-error loss (-> tensor_mse_loss).
 * @param input (Tensor*)
 * @param target (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_nn_mse_loss(Tensor* input, Tensor* target) { return tensor_mse_loss(input, target); }
/**
 * Public API: mean-absolute-error loss (-> tensor_mae_loss).
 * @param input (Tensor*)
 * @param target (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_nn_mae_loss(Tensor* input, Tensor* target) { return tensor_mae_loss(input, target); }
/**
 * Public API: binary cross-entropy loss (-> tensor_bce_loss).
 * @param input (Tensor*)
 * @param target (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_nn_bce_loss(Tensor* input, Tensor* target) { return tensor_bce_loss(input, target); }
/**
 * Public API: cross-entropy loss (-> tensor_cross_entropy_loss).
 * @param input (Tensor*)
 * @param target (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_nn_cross_entropy_loss(Tensor* input, Tensor* target) {
    return tensor_cross_entropy_loss(input, target);
}
/**
 * Public API: Huber loss (-> tensor_huber_loss).
 * @param input (Tensor*)
 * @param target (Tensor*)
 * @param delta (float)
 * @return (Tensor*)
 */
Tensor* cml_nn_huber_loss(Tensor* input, Tensor* target, float delta) {
    return tensor_huber_loss(input, target, delta);
}
/**
 * Public API: focal loss (-> tensor_focal_loss).
 * @param input (Tensor*)
 * @param target (Tensor*)
 * @param alpha (float)
 * @param gamma (float)
 * @return (Tensor*)
 */
Tensor* cml_nn_focal_loss(Tensor* input, Tensor* target, float alpha, float gamma) {
    return tensor_focal_loss(input, target, alpha, gamma);
}
/**
 * Public API: soft Dice loss (-> tensor_dice_loss).
 * @param input (Tensor*)
 * @param target (Tensor*)
 * @param eps (float)
 * @return (Tensor*)
 */
Tensor* cml_nn_dice_loss(Tensor* input, Tensor* target, float eps) {
    return tensor_dice_loss(input, target, eps);
}
/**
 * Public API: KL-divergence loss (-> tensor_kl_div_loss).
 * @param input (Tensor*)
 * @param target (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_nn_kl_div_loss(Tensor* input, Tensor* target) {
    return tensor_kl_div_loss(input, target);
}
/**
 * Public API: sparse (integer-target) cross-entropy loss (-> tensor_sparse_cross_entropy_loss).
 * @param input (Tensor*)
 * @param target (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_nn_sparse_cross_entropy_loss(Tensor* input, Tensor* target) {
    return tensor_sparse_cross_entropy_loss(input, target);
}
/**
 * Public API: triplet margin loss (-> tensor_triplet_margin_loss).
 * @param anchor (Tensor*)
 * @param positive (Tensor*)
 * @param negative (Tensor*)
 * @param margin (float)
 * @return (Tensor*)
 */
Tensor* cml_nn_triplet_margin_loss(Tensor* anchor, Tensor* positive, Tensor* negative,
                                   float margin) {
    return tensor_triplet_margin_loss(anchor, positive, negative, margin);
}
/**
 * Public API: cosine embedding loss (-> tensor_cosine_embedding_loss).
 * @param x1 (Tensor*)
 * @param x2 (Tensor*)
 * @param target (Tensor*)
 * @param margin (float)
 * @return (Tensor*)
 */
Tensor* cml_nn_cosine_embedding_loss(Tensor* x1, Tensor* x2, Tensor* target, float margin) {
    return tensor_cosine_embedding_loss(x1, x2, target, margin);
}
/**
 * Public API: negative log-likelihood loss (-> tensor_nll_loss).
 * @param log_probs (Tensor*)
 * @param targets (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_nn_nll_loss(Tensor* log_probs, Tensor* targets) {
    return tensor_nll_loss(log_probs, targets);
}

/**
 * Public API: run backpropagation from a tensor (-> tensor_backward).
 * @param tensor (Tensor*)
 * @param gradient (Tensor*)
 * @param retain_graph (bool)
 * @param create_graph (bool)
 * @return (void)
 */
void cml_backward(Tensor* tensor, Tensor* gradient, bool retain_graph, bool create_graph) {
    tensor_backward(tensor, gradient, retain_graph, create_graph);
}

/* End-of-training-step boundary. Detaches `keep` (typically the loss) from the
 * autograd graph - materializing its data so callers can still read it - and
 * then discards the accumulated forward/backward graph. Without this, a hand-
 * written training loop that reuses the same parameters piles every step's graph
 * onto them, so each backward re-traverses all prior steps (O(n^2) blow-up and
 * compounding gradients). The C convergence tests get this implicitly by freeing
 * the loss each step; this gives the same guarantee explicitly and is safe for
 * language bindings that keep the loss object alive (it's detached first, so the
 * reset never frees a tensor the caller still references). */
/**
 * Declared in ops/ir/graph_cache.c - drops cached execution plans (whose tensor
 * pointers dangle once the graph is freed) without freeing the pooled buffers.
 * @return (void)
 */
void cml_graph_cache_reset_global(void);

void cml_autograd_step_end(Tensor* keep) {
    if (keep)
        tensor_realize(keep); /* materialize + detach so the reset won't free it */
    cml_autograd_reset_after_step();
}

/**
 * Post-optimizer-step graph reset. Frees the accumulated autograd graph AND
 * drops the execution-plan cache, which would otherwise keep pointers into this
 * step's freed intermediate buffers and be replayed (crash) for a differently-
 * allocated tensor of the same shape - e.g. a second model in the same process.
 * The pooled buffers themselves and the parameters' realized data are kept, so
 * this is safe to call every step (unlike the full cml_reset_ir_context).
 * @return (void)
 */
void cml_autograd_reset_after_step(void) { cml_ir_reset_graph_only(); }

/**
 * Materialize every parameter (and its gradient) into owned storage, so they no
 * longer borrow data from execution-plan buffers. Called before the plan cache
 * is dropped in a training step, which would otherwise free the buffers the
 * parameters point at.
 * @param opt (Optimizer*)
 * @return (void)
 */
void cml_optim_realize_params(Optimizer* opt) {
    if (!opt)
        return;
    for (int g = 0; g < opt->num_param_groups; g++) {
        ParameterGroup* pg = &opt->param_groups[g];
        for (int i = 0; i < pg->num_parameters; i++) {
            Parameter* p = pg->parameters[i];
            if (p && p->tensor) {
                tensor_realize(p->tensor);
                if (p->tensor->grad)
                    tensor_realize(p->tensor->grad);
            }
        }
    }
}
/**
 * Public API: clear a tensor's accumulated gradient (-> tensor_zero_grad).
 * @param tensor (Tensor*)
 * @return (void)
 */
void cml_zero_grad(Tensor* tensor) { tensor_zero_grad(tensor); }
/**
 * Public API: enter a no-grad scope (-> autograd_no_grad_enter).
 * @return (void)
 */
void cml_no_grad(void) { autograd_no_grad_enter(); }
/**
 * Public API: re-enable gradient tracking (-> autograd_set_grad_mode).
 * @return (void)
 */
void cml_enable_grad(void) { autograd_set_grad_mode(true); }
/**
 * Public API: whether gradient tracking is currently on (-> autograd_is_grad_enabled).
 * @return (bool)
 */
bool cml_is_grad_enabled(void) { return autograd_is_grad_enabled(); }
/**
 * Public API: whether a tensor requires gradients (-> tensor_requires_grad).
 * @param t (Tensor*)
 * @return (bool)
 */
bool cml_requires_grad(Tensor* t) { return tensor_requires_grad(t); }
/**
 * Public API: set whether a tensor requires gradients (-> tensor_set_requires_grad).
 * @param t (Tensor*)
 * @param requires_grad (bool)
 * @return (void)
 */
void cml_set_requires_grad(Tensor* t, bool requires_grad) {
    tensor_set_requires_grad(t, requires_grad);
}
/**
 * Public API: whether a tensor is a graph leaf (-> tensor_is_leaf).
 * @param t (Tensor*)
 * @return (bool)
 */
bool cml_is_leaf(Tensor* t) { return tensor_is_leaf(t); }
/**
 * Public API: reset the global IR context, detaching all tensors
 * (-> cml_ir_reset_global_context).
 * @return (void)
 */
void cml_reset_ir_context(void) { cml_ir_reset_global_context(); }
/**
 * Public API: reset only the IR graph, keeping buffers (-> cml_ir_reset_graph_only).
 * @return (void)
 */
void cml_reset_ir_graph_only(void) { cml_ir_reset_graph_only(); }

struct CMLKernelCache;
struct CMLKernelCache* cml_kernel_cache_get_default(void);
void cml_kernel_cache_clear_impl(struct CMLKernelCache* cache);
void cml_kernel_cache_stats_impl(struct CMLKernelCache* cache, size_t* hits, size_t* misses,
                                 size_t* count, size_t* memory);
double cml_kernel_cache_hit_rate_impl(struct CMLKernelCache* cache);
void cml_kernel_cache_print_stats_impl(struct CMLKernelCache* cache);

/**
 * Public API: clear the default kernel cache (-> cml_kernel_cache_clear_impl).
 * @return (void)
 */
void cml_kernel_cache_clear(void) {
    struct CMLKernelCache* cache = cml_kernel_cache_get_default();
    if (cache) {
        cml_kernel_cache_clear_impl(cache);
    }
}

/**
 * Public API: read kernel-cache counters, zeroed if no cache (-> cml_kernel_cache_stats_impl).
 * @param hits (size_t*)
 * @param misses (size_t*)
 * @param count (size_t*)
 * @param memory (size_t*)
 * @return (void)
 */
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

/**
 * Public API: kernel-cache hit rate, 0 if no cache (-> cml_kernel_cache_hit_rate_impl).
 * @return (double)
 */
double cml_kernel_cache_hit_rate(void) {
    struct CMLKernelCache* cache = cml_kernel_cache_get_default();
    if (cache) {
        return cml_kernel_cache_hit_rate_impl(cache);
    }
    return 0.0;
}

/**
 * Public API: print kernel-cache stats (-> cml_kernel_cache_print_stats_impl).
 * @return (void)
 */
void cml_kernel_cache_print_stats(void) {
    struct CMLKernelCache* cache = cml_kernel_cache_get_default();
    if (cache) {
        cml_kernel_cache_print_stats_impl(cache);
    } else {
        printf("Kernel Cache: not initialized\n");
    }
}

/**
 * Public API: run a module's forward pass (-> module_forward).
 * @param module (Module*)
 * @param input (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_nn_module_forward(Module* module, Tensor* input) {
    return module_forward(module, input);
}
/**
 * Public API: set a module's train/eval flag (-> module_set_training).
 * @param module (Module*)
 * @param training (bool)
 * @return (void)
 */
void cml_nn_module_set_training(Module* module, bool training) {
    module_set_training(module, training);
}
/**
 * Public API: whether a module is in training mode (-> module_is_training).
 * @param module (Module*)
 * @return (bool)
 */
bool cml_nn_module_is_training(Module* module) { return module_is_training(module); }
/**
 * Public API: switch a module to eval mode (-> module_set_training).
 * @param module (Module*)
 * @return (void)
 */
void cml_nn_module_eval(Module* module) { module_set_training(module, false); }
/**
 * Public API: switch a module to training mode (-> module_set_training).
 * @param module (Module*)
 * @return (void)
 */
void cml_nn_module_train(Module* module) { module_set_training(module, true); }

/**
 * Public API: elementwise sign (-> uop_sign).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_sign(Tensor* a) { return uop_sign(a); }
/**
 * Public API: elementwise floor (-> uop_floor).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_floor(Tensor* a) { return uop_floor(a); }
/**
 * Public API: elementwise ceil (-> uop_ceil).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_ceil(Tensor* a) { return uop_ceil(a); }
/**
 * Public API: elementwise round (-> uop_round).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_round(Tensor* a) { return uop_round(a); }
/**
 * Public API: elementwise base-2 log (-> uop_log2).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_log2(Tensor* a) { return uop_log2(a); }
/**
 * Public API: elementwise base-2 exponential (-> uop_exp2).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_exp2(Tensor* a) { return uop_exp2(a); }
/**
 * Public API: elementwise arcsine (-> uop_asin).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_asin(Tensor* a) { return uop_asin(a); }
/**
 * Public API: elementwise arccosine (-> uop_acos).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_acos(Tensor* a) { return uop_acos(a); }
/**
 * Public API: elementwise arctangent (-> uop_atan).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_atan(Tensor* a) { return uop_atan(a); }
/**
 * Public API: elementwise square (-> uop_square).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_square(Tensor* a) { return uop_square(a); }
/**
 * Public API: elementwise reciprocal square root (-> uop_rsqrt).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_rsqrt(Tensor* a) { return uop_rsqrt(a); }
/**
 * Public API: elementwise error function (-> uop_erf).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_erf(Tensor* a) { return uop_erf(a); }

/**
 * Public API: clamp values into [min, max] (-> uop_clamp).
 * @param a (Tensor*)
 * @param min_val (float)
 * @param max_val (float)
 * @return (Tensor*)
 */
Tensor* cml_clamp(Tensor* a, float min_val, float max_val) {
    return uop_clamp(a, min_val, max_val);
}

/**
 * Public API: product reduction over a dim (-> uop_prod).
 * @param a (Tensor*)
 * @param dim (int)
 * @param keepdim (bool)
 * @return (Tensor*)
 */
Tensor* cml_prod(Tensor* a, int dim, bool keepdim) {
    int dims[]          = {dim};
    ReduceParams params = {.dims = dims, .num_dims = 1, .keepdim = keepdim};
    return uop_prod(a, &params);
}

/**
 * Public API: index of the max along a dim (-> tensor_argmax).
 * @param a (Tensor*)
 * @param dim (int)
 * @return (Tensor*)
 */
Tensor* cml_argmax(Tensor* a, int dim) { return tensor_argmax(a, dim); }
/**
 * Public API: index of the min along a dim (-> tensor_argmin).
 * @param a (Tensor*)
 * @param dim (int)
 * @return (Tensor*)
 */
Tensor* cml_argmin(Tensor* a, int dim) { return tensor_argmin(a, dim); }

/**
 * Public API: cumulative sum along a dim (-> uop_cumsum).
 * @param a (Tensor*)
 * @param dim (int)
 * @return (Tensor*)
 */
Tensor* cml_cumsum(Tensor* a, int dim) { return uop_cumsum(a, dim); }

/**
 * Public API: cumulative product along a dim (-> uop_cumprod).
 * @param a (Tensor*)
 * @param dim (int)
 * @return (Tensor*)
 */
Tensor* cml_cumprod(Tensor* a, int dim) { return uop_cumprod(a, dim); }

/**
 * Public API: cumulative log-sum-exp along a dim (-> uop_logcumsumexp).
 * @param a (Tensor*)
 * @param dim (int)
 * @return (Tensor*)
 */
Tensor* cml_logcumsumexp(Tensor* a, int dim) { return uop_logcumsumexp(a, dim); }

/**
 * Public API: indices that sort along a dim (-> uop_argsort).
 * @param a (Tensor*)
 * @param dim (int)
 * @param descending (bool)
 * @return (Tensor*)
 */
Tensor* cml_argsort(Tensor* a, int dim, bool descending) { return uop_argsort(a, dim, descending); }

/**
 * Public API: variance over a dim (-> tensor_var).
 * @param a (Tensor*)
 * @param dim (int)
 * @param unbiased (bool)
 * @param keepdim (bool)
 * @return (Tensor*)
 */
Tensor* cml_var(Tensor* a, int dim, bool unbiased, bool keepdim) {
    return tensor_var(a, dim, unbiased, keepdim);
}

/**
 * Public API: standard deviation over a dim (-> tensor_std).
 * @param a (Tensor*)
 * @param dim (int)
 * @param unbiased (bool)
 * @param keepdim (bool)
 * @return (Tensor*)
 */
Tensor* cml_std(Tensor* a, int dim, bool unbiased, bool keepdim) {
    return tensor_std(a, dim, unbiased, keepdim);
}

/**
 * squeeze/unsqueeze are size-1-dim reshapes; route through cml_reshape so they
 * build a differentiable graph node rather than a bare view (which severed the
 * backward graph, as raw reshape did before its fix).
 * @param a (Tensor*)
 * @param dim (int)
 * @return (Tensor*)
 */
Tensor* cml_squeeze(Tensor* a, int dim) {
    if (!a || a->ndim < 1)
        return tensor_squeeze(a, dim);
    int nd = a->ndim;
    if (dim < 0)
        dim += nd;
    if (dim < 0 || dim >= nd || a->shape[dim] != 1)
        return cml_reshape(a, a->shape, nd); /* nothing to drop: identity reshape */
    int shape[16], j = 0;
    for (int i = 0; i < nd; i++)
        if (i != dim)
            shape[j++] = a->shape[i];
    return cml_reshape(a, shape, nd - 1);
}
Tensor* cml_unsqueeze(Tensor* a, int dim) {
    if (!a || a->ndim + 1 > 16)
        return tensor_unsqueeze(a, dim);
    int nd = a->ndim;
    if (dim < 0)
        dim += nd + 1;
    if (dim < 0 || dim > nd)
        return tensor_unsqueeze(a, dim);
    int shape[16], j = 0;
    for (int i = 0; i < nd + 1; i++)
        shape[i] = (i == dim) ? 1 : a->shape[j++];
    return cml_reshape(a, shape, nd + 1);
}
/**
 * Route through uop_flip so flip is a differentiable graph node (grad = flip
 * back), not a bare tensor_flip view that severed the backward graph.
 * @param a (Tensor*)
 * @param dim (int)
 * @return (Tensor*)
 */
Tensor* cml_flip(Tensor* a, int dim) { return uop_flip(a, dim); }
/**
 * Public API: tile a tensor along each dim (-> tensor_repeat).
 * @param a (Tensor*)
 * @param repeats (int*)
 * @param num_repeats (int)
 * @return (Tensor*)
 */
Tensor* cml_repeat(Tensor* a, int* repeats, int num_repeats) {
    return tensor_repeat(a, repeats, num_repeats);
}
/**
 * Public API: split a tensor into equal parts along a dim (-> tensor_split).
 * @param a (Tensor*)
 * @param num_splits (int)
 * @param dim (int)
 * @param out_count (int*)
 * @return (Tensor**)
 */
Tensor** cml_split(Tensor* a, int num_splits, int dim, int* out_count) {
    return tensor_split(a, num_splits, dim, out_count);
}
/**
 * Public API: split a tensor into N chunks along a dim (-> tensor_chunk).
 * @param a (Tensor*)
 * @param chunks (int)
 * @param dim (int)
 * @param out_count (int*)
 * @return (Tensor**)
 */
Tensor** cml_chunk(Tensor* a, int chunks, int dim, int* out_count) {
    return tensor_chunk(a, chunks, dim, out_count);
}

/**
 * Public API: upper-triangular part of a matrix (-> uop_triu).
 * @param a (Tensor*)
 * @param diagonal (int)
 * @return (Tensor*)
 */
Tensor* cml_triu(Tensor* a, int diagonal) { return uop_triu(a, diagonal); }

/**
 * Public API: lower-triangular part of a matrix (-> uop_tril).
 * @param a (Tensor*)
 * @param diagonal (int)
 * @return (Tensor*)
 */
Tensor* cml_tril(Tensor* a, int diagonal) { return uop_tril(a, diagonal); }

/**
 * Public API: constant-value padding (-> uop_pad).
 * @param a (Tensor*)
 * @param pad_widths (int*)
 * @param num_dims (int)
 * @param value (float)
 * @return (Tensor*)
 */
Tensor* cml_pad(Tensor* a, int* pad_widths, int num_dims, float value) {
    return uop_pad(a, pad_widths, num_dims, value);
}
/**
 * Public API: reflection padding (-> uop_pad_reflect).
 * @param a (Tensor*)
 * @param pad_widths (int*)
 * @param num_dims (int)
 * @return (Tensor*)
 */
Tensor* cml_pad_reflect(Tensor* a, int* pad_widths, int num_dims) {
    return uop_pad_reflect(a, pad_widths, num_dims);
}
/**
 * Public API: replication (edge) padding (-> uop_pad_replicate).
 * @param a (Tensor*)
 * @param pad_widths (int*)
 * @param num_dims (int)
 * @return (Tensor*)
 */
Tensor* cml_pad_replicate(Tensor* a, int* pad_widths, int num_dims) {
    return uop_pad_replicate(a, pad_widths, num_dims);
}

/**
 * Public API: evenly spaced values in [start, end) by step (-> tensor_arange).
 * @param start (float)
 * @param end (float)
 * @param step (float)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_arange(float start, float end, float step, const TensorConfig* config) {
    return tensor_arange(start, end, step, config);
}
/**
 * Public API: `steps` evenly spaced values in [start, end] (-> tensor_linspace).
 * @param start (float)
 * @param end (float)
 * @param steps (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_linspace(float start, float end, int steps, const TensorConfig* config) {
    return tensor_linspace(start, end, steps, config);
}
/**
 * Public API: n-by-n identity matrix (-> tensor_eye).
 * @param n (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_eye(int n, const TensorConfig* config) { return tensor_eye(n, config); }
/**
 * Public API: uniform [0,1) random tensor (-> tensor_rand).
 * @param shape (int*)
 * @param ndim (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_rand(int* shape, int ndim, const TensorConfig* config) {
    return tensor_rand(shape, ndim, config);
}
/**
 * Public API: standard-normal random tensor (-> tensor_randn).
 * @param shape (int*)
 * @param ndim (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_randn(int* shape, int ndim, const TensorConfig* config) {
    return tensor_randn(shape, ndim, config);
}
/**
 * Public API: random integers in [low, high) (-> tensor_randint).
 * @param low (int)
 * @param high (int)
 * @param shape (int*)
 * @param ndim (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_randint(int low, int high, int* shape, int ndim, const TensorConfig* config) {
    return tensor_randint(low, high, shape, ndim, config);
}
/**
 * Public API: seed the global RNG (-> tensor_manual_seed).
 * @param seed (uint64_t)
 * @return (void)
 */
void cml_manual_seed(uint64_t seed) { tensor_manual_seed(seed); }
/**
 * Public API: message of the most recent error (-> error_stack_get_last_message).
 * @return (const char*)
 */
const char* cml_get_last_error(void) { return error_stack_get_last_message(); }
/**
 * Public API: code of the most recent error (-> error_stack_get_last_code).
 * @return (int)
 */
int cml_get_last_error_code(void) { return error_stack_get_last_code(); }
/**
 * Public API: clear the error stack (-> error_stack_clear).
 * @return (void)
 */
void cml_clear_last_error(void) { error_stack_clear(); }
/**
 * Public API: zeros matching another tensor's shape/dtype (-> tensor_zeros_like).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_zeros_like(Tensor* a) { return tensor_zeros_like(a); }
/**
 * Public API: ones matching another tensor's shape/dtype (-> tensor_ones_like).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_ones_like(Tensor* a) { return tensor_ones_like(a); }
/**
 * Public API: uniform random matching another tensor (-> tensor_rand_like).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_rand_like(Tensor* a) { return tensor_rand_like(a); }
/**
 * Public API: normal random matching another tensor (-> tensor_randn_like).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_randn_like(Tensor* a) { return tensor_randn_like(a); }
/**
 * Public API: constant-filled tensor matching another (-> tensor_full_like).
 * @param a (Tensor*)
 * @param value (float)
 * @return (Tensor*)
 */
Tensor* cml_full_like(Tensor* a, float value) { return tensor_full_like(a, value); }

/**
 * Public API: Kaiming/He uniform initialization (-> tensor_kaiming_uniform).
 * @param shape (int*)
 * @param ndim (int)
 * @param fan_in (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_kaiming_uniform(int* shape, int ndim, int fan_in, const TensorConfig* config) {
    return tensor_kaiming_uniform(shape, ndim, fan_in, config);
}
/**
 * Public API: Kaiming/He normal initialization (-> tensor_kaiming_normal).
 * @param shape (int*)
 * @param ndim (int)
 * @param fan_in (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_kaiming_normal(int* shape, int ndim, int fan_in, const TensorConfig* config) {
    return tensor_kaiming_normal(shape, ndim, fan_in, config);
}
/**
 * Public API: Glorot/Xavier uniform initialization (-> tensor_glorot_uniform).
 * @param shape (int*)
 * @param ndim (int)
 * @param fan_in (int)
 * @param fan_out (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_glorot_uniform(int* shape, int ndim, int fan_in, int fan_out,
                           const TensorConfig* config) {
    return tensor_glorot_uniform(shape, ndim, fan_in, fan_out, config);
}
/**
 * Public API: Xavier normal initialization (-> tensor_xavier_normal).
 * @param shape (int*)
 * @param ndim (int)
 * @param fan_in (int)
 * @param fan_out (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_xavier_normal(int* shape, int ndim, int fan_in, int fan_out,
                          const TensorConfig* config) {
    return tensor_xavier_normal(shape, ndim, fan_in, fan_out, config);
}

/**
 * Public API: create a LAMB optimizer (-> optim_lamb).
 * @param parameters (Parameter**)
 * @param num_parameters (int)
 * @param lr (float)
 * @param weight_decay (float)
 * @param beta1 (float)
 * @param beta2 (float)
 * @param epsilon (float)
 * @return (Optimizer*)
 */
Optimizer* cml_optim_lamb(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                          float beta1, float beta2, float epsilon) {
    return optim_lamb(parameters, num_parameters, lr, weight_decay, beta1, beta2, epsilon);
}
/**
 * Public API: create a LARS optimizer (-> optim_lars).
 * @param parameters (Parameter**)
 * @param num_parameters (int)
 * @param lr (float)
 * @param momentum (float)
 * @param weight_decay (float)
 * @param trust_coefficient (float)
 * @return (Optimizer*)
 */
Optimizer* cml_optim_lars(Parameter** parameters, int num_parameters, float lr, float momentum,
                          float weight_decay, float trust_coefficient) {
    return optim_lars(parameters, num_parameters, lr, momentum, weight_decay, trust_coefficient);
}

/**
 * Public API: create a 2-D instance-norm layer (-> nn_instancenorm2d).
 * @param num_features (int)
 * @param eps (float)
 * @param affine (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (InstanceNorm2d*)
 */
InstanceNorm2d* cml_nn_instancenorm2d(int num_features, float eps, bool affine, DType dtype,
                                      DeviceType device) {
    return nn_instancenorm2d(num_features, eps, affine, dtype, device);
}
/**
 * Public API: create a 1-D transposed convolution layer (-> nn_conv_transpose1d).
 * @param in_channels (int)
 * @param out_channels (int)
 * @param kernel_size (int)
 * @param stride (int)
 * @param padding (int)
 * @param output_padding (int)
 * @param use_bias (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (ConvTranspose1d*)
 */
ConvTranspose1d* cml_nn_conv_transpose1d(int in_channels, int out_channels, int kernel_size,
                                         int stride, int padding, int output_padding, bool use_bias,
                                         DType dtype, DeviceType device) {
    return nn_conv_transpose1d(in_channels, out_channels, kernel_size, stride, padding,
                               output_padding, use_bias, dtype, device);
}
/**
 * Public API: create a 2-D transposed convolution layer (-> nn_conv_transpose2d).
 * @param in_channels (int)
 * @param out_channels (int)
 * @param kernel_size (int)
 * @param stride (int)
 * @param padding (int)
 * @param output_padding (int)
 * @param use_bias (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (ConvTranspose2d*)
 */
ConvTranspose2d* cml_nn_conv_transpose2d(int in_channels, int out_channels, int kernel_size,
                                         int stride, int padding, int output_padding, bool use_bias,
                                         DType dtype, DeviceType device) {
    return nn_conv_transpose2d(in_channels, out_channels, kernel_size, stride, padding,
                               output_padding, use_bias, dtype, device);
}
/**
 * Public API: create a 3-D batch-norm layer (-> nn_batchnorm3d).
 * @param num_features (int)
 * @param eps (float)
 * @param momentum (float)
 * @param affine (bool)
 * @param track_running_stats (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (BatchNorm3d*)
 */
BatchNorm3d* cml_nn_batchnorm3d(int num_features, float eps, float momentum, bool affine,
                                bool track_running_stats, DType dtype, DeviceType device) {
    return nn_batchnorm3d(num_features, eps, momentum, affine, track_running_stats, dtype, device);
}
/**
 * Public API: create a channels-first 2-D LayerNorm layer (-> nn_layernorm2d).
 * @param num_channels (int)
 * @param eps (float)
 * @param affine (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (LayerNorm2d*)
 */
LayerNorm2d* cml_nn_layernorm2d(int num_channels, float eps, bool affine, DType dtype,
                                DeviceType device) {
    return nn_layernorm2d(num_channels, eps, affine, dtype, device);
}

/**
 * Public API: create a Muon optimizer (-> optim_muon).
 * @param parameters (Parameter**)
 * @param num_parameters (int)
 * @param lr (float)
 * @param momentum (float)
 * @param weight_decay (float)
 * @param nesterov (bool)
 * @return (Optimizer*)
 */
Optimizer* cml_optim_muon(Parameter** parameters, int num_parameters, float lr, float momentum,
                          float weight_decay, bool nesterov) {
    return optim_muon(parameters, num_parameters, lr, momentum, weight_decay, nesterov);
}
/**
 * Public API: create an AdamW optimizer (-> optim_adamw).
 * @param parameters (Parameter**)
 * @param num_parameters (int)
 * @param lr (float)
 * @param weight_decay (float)
 * @param beta1 (float)
 * @param beta2 (float)
 * @param epsilon (float)
 * @return (Optimizer*)
 */
Optimizer* cml_optim_adamw(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                           float beta1, float beta2, float epsilon) {
    return optim_adamw(parameters, num_parameters, lr, weight_decay, beta1, beta2, epsilon);
}
/**
 * Public API: create an Adadelta optimizer (-> optim_adadelta).
 * @param parameters (Parameter**)
 * @param num_parameters (int)
 * @param rho (float)
 * @param weight_decay (float)
 * @param epsilon (float)
 * @return (Optimizer*)
 */
Optimizer* cml_optim_adadelta(Parameter** parameters, int num_parameters, float rho,
                              float weight_decay, float epsilon) {
    return optim_adadelta(parameters, num_parameters, rho, weight_decay, epsilon);
}

/**
 * Public API: extract sliding local blocks (im2col) (-> uop_unfold).
 * @param a (Tensor*)
 * @param kernel_size (int)
 * @param stride (int)
 * @return (Tensor*)
 */
Tensor* cml_unfold(Tensor* a, int kernel_size, int stride) {
    return uop_unfold(a, kernel_size, stride);
}

/**
 * Public API: cast a tensor to another dtype (-> tensor_cast).
 * @param a (Tensor*)
 * @param dtype (DType)
 * @return (Tensor*)
 */
Tensor* cml_cast(Tensor* a, DType dtype) { return tensor_cast(a, dtype); }
/**
 * Public API: return a contiguous copy/view (-> tensor_contiguous).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_contiguous(Tensor* a) { return tensor_contiguous(a); }
/**
 * Public API: wrap an external buffer without copying (-> tensor_from_blob).
 * @param data (void*)
 * @param shape (int*)
 * @param ndim (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_from_blob(void* data, int* shape, int ndim, const TensorConfig* config) {
    return tensor_from_blob(data, shape, ndim, config);
}
/**
 * Public API: random permutation of 0..n-1 (-> tensor_randperm).
 * @param n (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_randperm(int n, const TensorConfig* config) { return tensor_randperm(n, config); }
/**
 * Public API: cast to float16 (-> tensor_half).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_half(Tensor* a) { return tensor_half(a); }
/**
 * Public API: cast to float64 (-> tensor_double).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_double(Tensor* a) { return tensor_double(a); }
/**
 * Public API: cast to int32 (-> tensor_int).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_int_(Tensor* a) { return tensor_int(a); }
/**
 * Public API: cast to int64 (-> tensor_long).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_long(Tensor* a) { return tensor_long(a); }
/**
 * Public API: cast to int16 (-> tensor_short).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_short(Tensor* a) { return tensor_short(a); }
/**
 * Public API: cast to bool (-> tensor_bool).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_bool_(Tensor* a) { return tensor_bool(a); }
/**
 * Public API: cast to bfloat16 (-> tensor_bfloat16).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_bfloat16(Tensor* a) { return tensor_bfloat16(a); }

/**
 * Public API: resize/interpolate a tensor (-> tensor_interpolate).
 * @param a (Tensor*)
 * @param output_size (int*)
 * @param num_dims (int)
 * @param mode (InterpMode)
 * @return (Tensor*)
 */
Tensor* cml_interpolate(Tensor* a, int* output_size, int num_dims, InterpMode mode) {
    return tensor_interpolate(a, output_size, num_dims, mode);
}

/**
 * Public API: 1-D dot product (-> tensor_dot).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_dot(Tensor* a, Tensor* b) { return tensor_dot(a, b); }

/**
 * Public API: create a 1-D max-pool layer (-> nn_maxpool1d).
 * @param kernel_size (int)
 * @param stride (int)
 * @param padding (int)
 * @param dilation (int)
 * @param ceil_mode (bool)
 * @return (MaxPool1d*)
 */
MaxPool1d* cml_nn_maxpool1d(int kernel_size, int stride, int padding, int dilation,
                            bool ceil_mode) {
    return nn_maxpool1d(kernel_size, stride, padding, dilation, ceil_mode);
}
/**
 * Public API: create a 1-D average-pool layer (-> nn_avgpool1d).
 * @param kernel_size (int)
 * @param stride (int)
 * @param padding (int)
 * @param ceil_mode (bool)
 * @param count_include_pad (bool)
 * @return (AvgPool1d*)
 */
AvgPool1d* cml_nn_avgpool1d(int kernel_size, int stride, int padding, bool ceil_mode,
                            bool count_include_pad) {
    return nn_avgpool1d(kernel_size, stride, padding, ceil_mode, count_include_pad);
}
/**
 * Public API: create a 2-D adaptive average-pool layer (-> nn_adaptive_avgpool2d).
 * @param output_h (int)
 * @param output_w (int)
 * @return (AdaptiveAvgPool2d*)
 */
AdaptiveAvgPool2d* cml_nn_adaptive_avgpool2d(int output_h, int output_w) {
    return nn_adaptive_avgpool2d(output_h, output_w);
}
/**
 * Public API: create a 1-D adaptive average-pool layer (-> nn_adaptive_avgpool1d).
 * @param output_size (int)
 * @return (AdaptiveAvgPool1d*)
 */
AdaptiveAvgPool1d* cml_nn_adaptive_avgpool1d(int output_size) {
    return nn_adaptive_avgpool1d(output_size);
}

/**
 * Public API: create a 3-D max-pool layer (-> nn_maxpool3d).
 * @param kernel_size (int)
 * @param stride (int)
 * @param padding (int)
 * @param dilation (int)
 * @param ceil_mode (bool)
 * @return (MaxPool3d*)
 */
MaxPool3d* cml_nn_maxpool3d(int kernel_size, int stride, int padding, int dilation,
                            bool ceil_mode) {
    return nn_maxpool3d(kernel_size, stride, padding, dilation, ceil_mode);
}
/**
 * Public API: create a 3-D average-pool layer (-> nn_avgpool3d).
 * @param kernel_size (int)
 * @param stride (int)
 * @param padding (int)
 * @param ceil_mode (bool)
 * @param count_include_pad (bool)
 * @return (AvgPool3d*)
 */
AvgPool3d* cml_nn_avgpool3d(int kernel_size, int stride, int padding, bool ceil_mode,
                            bool count_include_pad) {
    return nn_avgpool3d(kernel_size, stride, padding, ceil_mode, count_include_pad);
}
/**
 * Public API: create a 2-D adaptive max-pool layer (-> nn_adaptive_maxpool2d).
 * @param output_h (int)
 * @param output_w (int)
 * @return (AdaptiveMaxPool2d*)
 */
AdaptiveMaxPool2d* cml_nn_adaptive_maxpool2d(int output_h, int output_w) {
    return nn_adaptive_maxpool2d(output_h, output_w);
}
/**
 * Public API: create a 1-D adaptive max-pool layer (-> nn_adaptive_maxpool1d).
 * @param output_size (int)
 * @return (AdaptiveMaxPool1d*)
 */
AdaptiveMaxPool1d* cml_nn_adaptive_maxpool1d(int output_size) {
    return nn_adaptive_maxpool1d(output_size);
}

/**
 * Public API: scatter values with a reduction (-> tensor_scatter_reduce).
 * @param self (Tensor*)
 * @param dim (int)
 * @param index (Tensor*)
 * @param src (Tensor*)
 * @param mode (ScatterReduceMode)
 * @return (Tensor*)
 */
Tensor* cml_scatter_reduce(Tensor* self, int dim, Tensor* index, Tensor* src,
                           ScatterReduceMode mode) {
    return tensor_scatter_reduce(self, dim, index, src, mode);
}
/**
 * Public API: reinterpret raw bits as another dtype (-> tensor_bitcast).
 * @param a (Tensor*)
 * @param target_dtype (DType)
 * @return (Tensor*)
 */
Tensor* cml_bitcast(Tensor* a, DType target_dtype) { return tensor_bitcast(a, target_dtype); }

/**
 * Public API: QR decomposition (-> tensor_qr).
 * @param a (Tensor*)
 * @return (QRResult)
 */
QRResult cml_qr(Tensor* a) { return tensor_qr(a); }
/**
 * Public API: singular value decomposition (-> tensor_svd).
 * @param a (Tensor*)
 * @return (SVDResult)
 */
SVDResult cml_svd(Tensor* a) { return tensor_svd(a); }
/**
 * Public API: Cholesky factor of an SPD matrix (-> tensor_cholesky).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_cholesky(Tensor* a) { return tensor_cholesky(a); }
/**
 * Public API: solve A x = b (-> tensor_solve).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_solve(Tensor* a, Tensor* b) { return tensor_solve(a, b); }
/**
 * Public API: matrix inverse (-> tensor_inverse).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_inverse(Tensor* a) { return tensor_inverse(a); }
/**
 * Public API: matrix determinant (-> tensor_det).
 * @param a (Tensor*)
 * @return (float)
 */
float cml_det(Tensor* a) { return tensor_det(a); }
/**
 * Public API: least-squares solve (-> tensor_lstsq).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_lstsq(Tensor* a, Tensor* b) { return tensor_lstsq(a, b); }
/**
 * Public API: Frobenius norm (-> tensor_matrix_norm).
 * @param a (Tensor*)
 * @return (float)
 */
float cml_matrix_norm(Tensor* a) { return tensor_matrix_norm(a); }
/**
 * Public API: numerical matrix rank (-> tensor_matrix_rank).
 * @param a (Tensor*)
 * @param tol (float)
 * @return (int)
 */
int cml_matrix_rank(Tensor* a, float tol) { return tensor_matrix_rank(a, tol); }
/**
 * Public API: Moore-Penrose pseudo-inverse (-> tensor_pinv).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_pinv(Tensor* a) { return tensor_pinv(a); }
/**
 * Public API: symmetric eigendecomposition (-> tensor_eigh).
 * @param a (Tensor*)
 * @return (EigResult)
 */
EigResult cml_eigh(Tensor* a) { return tensor_eigh(a); }
/**
 * Public API: trace of a 2D matrix (-> tensor_trace).
 * @param a (Tensor*)
 * @return (float)
 */
float cml_trace(Tensor* a) { return tensor_trace(a); }
/**
 * Public API: Kronecker product (-> tensor_kron).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_kron(Tensor* a, Tensor* b) { return tensor_kron(a, b); }
/**
 * Public API: cross product of two length-3 vectors (-> tensor_cross).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_cross(Tensor* a, Tensor* b) { return tensor_cross(a, b); }
/**
 * Public API: sign and log-absolute determinant (-> tensor_slogdet).
 * @param a (Tensor*)
 * @return (SlogdetResult)
 */
SlogdetResult cml_slogdet(Tensor* a) { return tensor_slogdet(a); }
/**
 * Public API: median of all elements (-> tensor_median).
 * @param a (Tensor*)
 * @return (float)
 */
float cml_median(Tensor* a) { return tensor_median(a); }
/**
 * Public API: q-quantile over all elements (-> tensor_quantile).
 * @param a (Tensor*)
 * @param q (float)
 * @return (float)
 */
float cml_quantile(Tensor* a, float q) { return tensor_quantile(a, q); }
/**
 * Public API: p-th percentile over all elements (-> tensor_percentile).
 * @param a (Tensor*)
 * @param p (float)
 * @return (float)
 */
float cml_percentile(Tensor* a, float p) { return tensor_percentile(a, p); }
/**
 * Public API: covariance matrix (-> tensor_cov).
 * @param a (Tensor*)
 * @param rowvar (bool)
 * @return (Tensor*)
 */
Tensor* cml_cov(Tensor* a, bool rowvar) { return tensor_cov(a, rowvar); }
/**
 * Public API: correlation matrix (-> tensor_corrcoef).
 * @param a (Tensor*)
 * @param rowvar (bool)
 * @return (Tensor*)
 */
Tensor* cml_corrcoef(Tensor* a, bool rowvar) { return tensor_corrcoef(a, rowvar); }
/**
 * Public API: n-th discrete difference (-> tensor_diff).
 * @param a (Tensor*)
 * @param n (int)
 * @return (Tensor*)
 */
Tensor* cml_diff(Tensor* a, int n) { return tensor_diff(a, n); }
/**
 * Public API: vector outer product (-> tensor_outer).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_outer(Tensor* a, Tensor* b) { return tensor_outer(a, b); }
/**
 * Public API: tile a 1-D tensor (-> tensor_tile).
 * @param a (Tensor*)
 * @param reps (int)
 * @return (Tensor*)
 */
Tensor* cml_tile(Tensor* a, int reps) { return tensor_tile(a, reps); }
/**
 * Public API: sorted unique values (-> tensor_unique).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_unique(Tensor* a) { return tensor_unique(a); }
/**
 * Public API: trapezoidal integral (-> tensor_trapz).
 * @param y (Tensor*)
 * @param dx (float)
 * @return (float)
 */
float cml_trapz(Tensor* y, float dx) { return tensor_trapz(y, dx); }
/**
 * Public API: Vandermonde matrix (-> tensor_vander).
 * @param x (Tensor*)
 * @param N (int)
 * @param increasing (bool)
 * @return (Tensor*)
 */
Tensor* cml_vander(Tensor* x, int N, bool increasing) { return tensor_vander(x, N, increasing); }
/**
 * Public API: least-squares polynomial fit (-> tensor_polyfit).
 * @param x (Tensor*)
 * @param y (Tensor*)
 * @param deg (int)
 * @return (Tensor*)
 */
Tensor* cml_polyfit(Tensor* x, Tensor* y, int deg) { return tensor_polyfit(x, y, deg); }
/**
 * Public API: polynomial evaluation (-> tensor_polyval).
 * @param coeffs (Tensor*)
 * @param x (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_polyval(Tensor* coeffs, Tensor* x) { return tensor_polyval(coeffs, x); }
/**
 * Public API: skewness (-> tensor_skewness).
 * @param a (Tensor*)
 * @return (float)
 */
float cml_skewness(Tensor* a) { return tensor_skewness(a); }
/**
 * Public API: excess kurtosis (-> tensor_kurtosis).
 * @param a (Tensor*)
 * @return (float)
 */
float cml_kurtosis(Tensor* a) { return tensor_kurtosis(a); }
/**
 * Public API: element-wise z-score (-> tensor_zscore).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_zscore(Tensor* a) { return tensor_zscore(a); }
/**
 * Public API: NaN-ignoring mean (-> tensor_nanmean).
 * @param a (Tensor*)
 * @return (float)
 */
float cml_nanmean(Tensor* a) { return tensor_nanmean(a); }
/**
 * Public API: NaN-ignoring sum (-> tensor_nansum).
 * @param a (Tensor*)
 * @return (float)
 */
float cml_nansum(Tensor* a) { return tensor_nansum(a); }
/**
 * Public API: per-column standardization (-> tensor_standard_scale).
 * @param x (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_standard_scaler(Tensor* x) { return tensor_standard_scale(x); }
/**
 * Public API: per-column min-max scaling (-> tensor_minmax_scale).
 * @param x (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_minmax_scaler(Tensor* x) { return tensor_minmax_scale(x); }
/**
 * Public API: per-column robust scaling (-> tensor_robust_scale).
 * @param x (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_robust_scaler(Tensor* x) { return tensor_robust_scale(x); }
/**
 * Public API: per-row L2 normalization (-> tensor_normalize_l2).
 * @param x (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_normalize_l2(Tensor* x) { return tensor_normalize_l2(x); }
/**
 * Public API: cosine similarity (-> tensor_cosine_similarity).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (float)
 */
float cml_cosine_similarity(Tensor* a, Tensor* b) { return tensor_cosine_similarity(a, b); }
/**
 * Public API: Euclidean distance (-> tensor_euclidean_distance).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (float)
 */
float cml_euclidean_distance(Tensor* a, Tensor* b) { return tensor_euclidean_distance(a, b); }
/**
 * Public API: Manhattan distance (-> tensor_manhattan_distance).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (float)
 */
float cml_manhattan_distance(Tensor* a, Tensor* b) { return tensor_manhattan_distance(a, b); }
/**
 * Public API: pairwise Euclidean distance matrix (-> tensor_cdist).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_cdist(Tensor* a, Tensor* b) { return tensor_cdist(a, b); }
/**
 * Public API: principal component analysis (-> tensor_pca).
 * @param x (Tensor*)
 * @param n_components (int)
 * @return (PCAResult)
 */
PCAResult cml_pca(Tensor* x, int n_components) { return tensor_pca(x, n_components); }
/**
 * Public API: sorted intersection of two 1-D tensors (-> tensor_intersect1d).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_intersect1d(Tensor* a, Tensor* b) { return tensor_intersect1d(a, b); }
/**
 * Public API: sorted union of two 1-D tensors (-> tensor_union1d).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_union1d(Tensor* a, Tensor* b) { return tensor_union1d(a, b); }
/**
 * Public API: sorted set difference (-> tensor_setdiff1d).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_setdiff1d(Tensor* a, Tensor* b) { return tensor_setdiff1d(a, b); }
/**
 * Public API: membership mask (-> tensor_isin).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_isin(Tensor* a, Tensor* b) { return tensor_isin(a, b); }
/**
 * Public API: peak-to-peak range (-> tensor_ptp).
 * @param a (Tensor*)
 * @return (float)
 */
float cml_ptp(Tensor* a) { return tensor_ptp(a); }
/**
 * Public API: interquartile range (-> tensor_iqr).
 * @param a (Tensor*)
 * @return (float)
 */
float cml_iqr(Tensor* a) { return tensor_iqr(a); }
/**
 * Public API: statistical mode (-> tensor_mode).
 * @param a (Tensor*)
 * @return (float)
 */
float cml_mode(Tensor* a) { return tensor_mode(a); }
/**
 * Public API: Shannon entropy (-> tensor_entropy).
 * @param a (Tensor*)
 * @param base (float)
 * @return (float)
 */
float cml_entropy(Tensor* a, float base) { return tensor_entropy(a, base); }
/**
 * Public API: geometric mean (-> tensor_gmean).
 * @param a (Tensor*)
 * @return (float)
 */
float cml_gmean(Tensor* a) { return tensor_gmean(a); }
/**
 * Public API: harmonic mean (-> tensor_hmean).
 * @param a (Tensor*)
 * @return (float)
 */
float cml_hmean(Tensor* a) { return tensor_hmean(a); }
/**
 * Public API: rolling window sum (-> tensor_rolling_sum).
 * @param a (Tensor*)
 * @param window (int)
 * @return (Tensor*)
 */
Tensor* cml_rolling_sum(Tensor* a, int window) { return tensor_rolling_sum(a, window); }
/**
 * Public API: rolling window mean (-> tensor_rolling_mean).
 * @param a (Tensor*)
 * @param window (int)
 * @return (Tensor*)
 */
Tensor* cml_rolling_mean(Tensor* a, int window) { return tensor_rolling_mean(a, window); }
/**
 * Public API: exponentially weighted moving average (-> tensor_ewma).
 * @param a (Tensor*)
 * @param alpha (float)
 * @return (Tensor*)
 */
Tensor* cml_ewma(Tensor* a, float alpha) { return tensor_ewma(a, alpha); }
/**
 * Public API: cumulative maximum (-> tensor_cummax).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_cummax(Tensor* a) { return tensor_cummax(a); }
/**
 * Public API: cumulative minimum (-> tensor_cummin).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_cummin(Tensor* a) { return tensor_cummin(a); }
/**
 * Public API: integer matrix power (-> tensor_matrix_power).
 * @param a (Tensor*)
 * @param p (int)
 * @return (Tensor*)
 */
Tensor* cml_matrix_power(Tensor* a, int p) { return tensor_matrix_power(a, p); }
/**
 * Public API: 2-norm condition number (-> tensor_cond).
 * @param a (Tensor*)
 * @return (float)
 */
float cml_cond(Tensor* a) { return tensor_cond(a); }
/**
 * Public API: triangular solve (-> tensor_solve_triangular).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @param lower (bool)
 * @return (Tensor*)
 */
Tensor* cml_solve_triangular(Tensor* a, Tensor* b, bool lower) {
    return tensor_solve_triangular(a, b, lower);
}
/**
 * Public API: matrix exponential (-> tensor_expm).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_expm(Tensor* a) { return tensor_expm(a); }
/**
 * Public API: complementary error function (-> tensor_erfc).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_erfc(Tensor* a) { return tensor_erfc(a); }
/**
 * Public API: log-gamma (-> tensor_lgamma).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_lgamma(Tensor* a) { return tensor_lgamma(a); }
/**
 * Public API: digamma (-> tensor_digamma).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_digamma(Tensor* a) { return tensor_digamma(a); }
/**
 * Public API: logit (-> tensor_logit).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_logit(Tensor* a) { return tensor_logit(a); }
/**
 * Public API: Toeplitz matrix (-> tensor_toeplitz).
 * @param c (Tensor*)
 * @param r (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_toeplitz(Tensor* c, Tensor* r) { return tensor_toeplitz(c, r); }
/**
 * Public API: circulant matrix (-> tensor_circulant).
 * @param c (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_circulant(Tensor* c) { return tensor_circulant(c); }
/**
 * Public API: Hankel matrix (-> tensor_hankel).
 * @param c (Tensor*)
 * @param r (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_hankel(Tensor* c, Tensor* r) { return tensor_hankel(c, r); }
/**
 * Public API: detrend a 1-D signal (-> tensor_detrend).
 * @param y (Tensor*)
 * @param type (int)
 * @return (Tensor*)
 */
Tensor* cml_detrend(Tensor* y, int type) { return tensor_detrend(y, type); }
/**
 * Public API: median absolute deviation (-> tensor_mad).
 * @param a (Tensor*)
 * @return (float)
 */
float cml_median_abs_deviation(Tensor* a) { return tensor_mad(a); }
/**
 * Public API: trimmed mean (-> tensor_trim_mean).
 * @param a (Tensor*)
 * @param proportion (float)
 * @return (float)
 */
float cml_trim_mean(Tensor* a, float proportion) { return tensor_trim_mean(a, proportion); }
/**
 * Public API: standard error of the mean (-> tensor_sem).
 * @param a (Tensor*)
 * @return (float)
 */
float cml_sem(Tensor* a) { return tensor_sem(a); }
/**
 * Public API: winsorize (-> tensor_winsorize).
 * @param a (Tensor*)
 * @param lower (float)
 * @param upper (float)
 * @return (Tensor*)
 */
Tensor* cml_winsorize(Tensor* a, float lower, float upper) {
    return tensor_winsorize(a, lower, upper);
}
/**
 * Public API: Minkowski distance (-> tensor_minkowski_distance).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @param p (float)
 * @return (float)
 */
float cml_minkowski_distance(Tensor* a, Tensor* b, float p) {
    return tensor_minkowski_distance(a, b, p);
}
/**
 * Public API: Chebyshev distance (-> tensor_chebyshev_distance).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (float)
 */
float cml_chebyshev_distance(Tensor* a, Tensor* b) { return tensor_chebyshev_distance(a, b); }
/**
 * Public API: Hamming distance (-> tensor_hamming_distance).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (float)
 */
float cml_hamming_distance(Tensor* a, Tensor* b) { return tensor_hamming_distance(a, b); }
/**
 * Public API: Mahalanobis distance (-> tensor_mahalanobis_distance).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @param vi (Tensor*)
 * @return (float)
 */
float cml_mahalanobis_distance(Tensor* a, Tensor* b, Tensor* vi) {
    return tensor_mahalanobis_distance(a, b, vi);
}
/**
 * Public API: k-means clustering (-> tensor_kmeans).
 * @param x (Tensor*)
 * @param k (int)
 * @param max_iter (int)
 * @return (KMeansResult)
 */
KMeansResult cml_kmeans(Tensor* x, int k, int max_iter) { return tensor_kmeans(x, k, max_iter); }

/**
 * Public API: load a tensor from a URL (-> tensor_from_url).
 * @param url (const char*)
 * @return (Tensor*)
 */
Tensor* cml_from_url(const char* url) { return tensor_from_url(url); }

/**
 * Public API: open a GGUF file for reading (-> gguf_open_read).
 * @param p (const char*)
 * @return (GGUFContext*)
 */
GGUFContext* cml_gguf_open_read(const char* p) { return gguf_open_read(p); }
/**
 * Public API: open a GGUF file for writing (-> gguf_open_write).
 * @param p (const char*)
 * @return (GGUFContext*)
 */
GGUFContext* cml_gguf_open_write(const char* p) { return gguf_open_write(p); }
/**
 * Public API: close a GGUF context (-> gguf_close).
 * @param c (GGUFContext*)
 * @return (void)
 */
void cml_gguf_close(GGUFContext* c) { gguf_close(c); }
/**
 * Public API: write a named tensor to a GGUF file (-> gguf_write_tensor).
 * @param c (GGUFContext*)
 * @param n (const char*)
 * @param t (Tensor*)
 * @return (int)
 */
int cml_gguf_write_tensor(GGUFContext* c, const char* n, Tensor* t) {
    return gguf_write_tensor(c, n, t);
}
/**
 * Public API: read a named tensor from a GGUF file (-> gguf_read_tensor).
 * @param c (GGUFContext*)
 * @param n (const char*)
 * @return (Tensor*)
 */
Tensor* cml_gguf_read_tensor(GGUFContext* c, const char* n) { return gguf_read_tensor(c, n); }
/**
 * Public API: save a module's weights to GGUF (-> module_save_gguf).
 * @param m (Module*)
 * @param p (const char*)
 * @return (int)
 */
int cml_module_save_gguf(Module* m, const char* p) { return module_save_gguf(m, p); }
/**
 * Public API: load a module's weights from GGUF (-> module_load_gguf).
 * @param m (Module*)
 * @param p (const char*)
 * @return (int)
 */
int cml_module_load_gguf(Module* m, const char* p) { return module_load_gguf(m, p); }

/**
 * Public API: open a safetensors file for reading (-> safetensors_open_read).
 * @param p (const char*)
 * @return (SafeTensorsContext*)
 */
SafeTensorsContext* cml_safetensors_open_read(const char* p) { return safetensors_open_read(p); }
/**
 * Public API: open a safetensors file for writing (-> safetensors_open_write).
 * @param p (const char*)
 * @return (SafeTensorsContext*)
 */
SafeTensorsContext* cml_safetensors_open_write(const char* p) { return safetensors_open_write(p); }
/**
 * Public API: close a safetensors context (-> safetensors_close).
 * @param c (SafeTensorsContext*)
 * @return (void)
 */
void cml_safetensors_close(SafeTensorsContext* c) { safetensors_close(c); }
/**
 * Public API: write a named tensor to a safetensors file (-> safetensors_write_tensor).
 * @param c (SafeTensorsContext*)
 * @param n (const char*)
 * @param t (Tensor*)
 * @return (int)
 */
int cml_safetensors_write_tensor(SafeTensorsContext* c, const char* n, Tensor* t) {
    return safetensors_write_tensor(c, n, t);
}
/**
 * Public API: read a named tensor from a safetensors file (-> safetensors_read_tensor).
 * @param c (SafeTensorsContext*)
 * @param n (const char*)
 * @return (Tensor*)
 */
Tensor* cml_safetensors_read_tensor(SafeTensorsContext* c, const char* n) {
    return safetensors_read_tensor(c, n);
}
/**
 * Public API: save a module's weights to safetensors (-> module_save_safetensors).
 * @param m (Module*)
 * @param p (const char*)
 * @return (int)
 */
int cml_module_save_safetensors(Module* m, const char* p) { return module_save_safetensors(m, p); }
/**
 * Public API: load a module's weights from safetensors (-> module_load_safetensors).
 * @param m (Module*)
 * @param p (const char*)
 * @return (int)
 */
int cml_module_load_safetensors(Module* m, const char* p) { return module_load_safetensors(m, p); }

/**
 * Public API: create a stacked transformer encoder (-> nn_transformer_encoder).
 * @param d_model (int)
 * @param nhead (int)
 * @param dim_feedforward (int)
 * @param dropout (float)
 * @param num_layers (int)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (TransformerEncoder*)
 */
TransformerEncoder* cml_nn_transformer_encoder(int d_model, int nhead, int dim_feedforward,
                                               float dropout, int num_layers, DType dtype,
                                               DeviceType device) {
    return nn_transformer_encoder(d_model, nhead, dim_feedforward, dropout, num_layers, dtype,
                                  device);
}
/**
 * Public API: create a transformer decoder layer (-> nn_transformer_decoder_layer).
 * @param d_model (int)
 * @param nhead (int)
 * @param dim_feedforward (int)
 * @param dropout (float)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (TransformerDecoderLayer*)
 */
TransformerDecoderLayer* cml_nn_transformer_decoder_layer(int d_model, int nhead,
                                                          int dim_feedforward, float dropout,
                                                          DType dtype, DeviceType device) {
    return nn_transformer_decoder_layer(d_model, nhead, dim_feedforward, dropout, dtype, device);
}
/**
 * Public API: create a stacked transformer decoder (-> nn_transformer_decoder).
 * @param d_model (int)
 * @param nhead (int)
 * @param dim_feedforward (int)
 * @param dropout (float)
 * @param num_layers (int)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (TransformerDecoder*)
 */
TransformerDecoder* cml_nn_transformer_decoder(int d_model, int nhead, int dim_feedforward,
                                               float dropout, int num_layers, DType dtype,
                                               DeviceType device) {
    return nn_transformer_decoder(d_model, nhead, dim_feedforward, dropout, num_layers, dtype,
                                  device);
}

/**
 * Public API: create a NAdam optimizer (-> optim_nadam).
 * @param parameters (Parameter**)
 * @param num_parameters (int)
 * @param lr (float)
 * @param weight_decay (float)
 * @param beta1 (float)
 * @param beta2 (float)
 * @param epsilon (float)
 * @return (Optimizer*)
 */
Optimizer* cml_optim_nadam(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                           float beta1, float beta2, float epsilon) {
    return optim_nadam(parameters, num_parameters, lr, weight_decay, beta1, beta2, epsilon);
}
/**
 * Public API: create an Adamax optimizer (-> optim_adamax).
 * @param parameters (Parameter**)
 * @param num_parameters (int)
 * @param lr (float)
 * @param weight_decay (float)
 * @param beta1 (float)
 * @param beta2 (float)
 * @param epsilon (float)
 * @return (Optimizer*)
 */
Optimizer* cml_optim_adamax(Parameter** parameters, int num_parameters, float lr,
                            float weight_decay, float beta1, float beta2, float epsilon) {
    return optim_adamax(parameters, num_parameters, lr, weight_decay, beta1, beta2, epsilon);
}

/**
 * Public API: create a multi-layer RNN (-> nn_rnn).
 * @param input_size (int)
 * @param hidden_size (int)
 * @param num_layers (int)
 * @param bidirectional (bool)
 * @param batch_first (bool)
 * @param dropout (float)
 * @param use_bias (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (RNN*)
 */
RNN* cml_nn_rnn(int input_size, int hidden_size, int num_layers, bool bidirectional,
                bool batch_first, float dropout, bool use_bias, DType dtype, DeviceType device) {
    return nn_rnn(input_size, hidden_size, num_layers, bidirectional, batch_first, dropout,
                  use_bias, dtype, device);
}
/**
 * Public API: create a multi-layer LSTM (-> nn_lstm).
 * @param input_size (int)
 * @param hidden_size (int)
 * @param num_layers (int)
 * @param bidirectional (bool)
 * @param batch_first (bool)
 * @param dropout (float)
 * @param use_bias (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (LSTM*)
 */
LSTM* cml_nn_lstm(int input_size, int hidden_size, int num_layers, bool bidirectional,
                  bool batch_first, float dropout, bool use_bias, DType dtype, DeviceType device) {
    return nn_lstm(input_size, hidden_size, num_layers, bidirectional, batch_first, dropout,
                   use_bias, dtype, device);
}
/**
 * Public API: create a multi-layer GRU (-> nn_gru).
 * @param input_size (int)
 * @param hidden_size (int)
 * @param num_layers (int)
 * @param bidirectional (bool)
 * @param batch_first (bool)
 * @param dropout (float)
 * @param use_bias (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (GRU*)
 */
GRU* cml_nn_gru(int input_size, int hidden_size, int num_layers, bool bidirectional,
                bool batch_first, float dropout, bool use_bias, DType dtype, DeviceType device) {
    return nn_gru(input_size, hidden_size, num_layers, bidirectional, batch_first, dropout,
                  use_bias, dtype, device);
}
/**
 * Public API: create a 3-D transposed convolution layer (-> nn_conv_transpose3d).
 * @param in_channels (int)
 * @param out_channels (int)
 * @param kernel_size (int)
 * @param stride (int)
 * @param padding (int)
 * @param output_padding (int)
 * @param use_bias (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (ConvTranspose3d*)
 */
ConvTranspose3d* cml_nn_conv_transpose3d(int in_channels, int out_channels, int kernel_size,
                                         int stride, int padding, int output_padding, bool use_bias,
                                         DType dtype, DeviceType device) {
    return nn_conv_transpose3d(in_channels, out_channels, kernel_size, stride, padding,
                               output_padding, use_bias, dtype, device);
}
/**
 * Public API: create an Upsample layer (-> nn_upsample).
 * @param scale_factor (float)
 * @param output_size (const int*)
 * @param num_output_dims (int)
 * @param mode (UpsampleMode)
 * @param align_corners (bool)
 * @return (Upsample*)
 */
Upsample* cml_nn_upsample(float scale_factor, const int* output_size, int num_output_dims,
                          UpsampleMode mode, bool align_corners) {
    return nn_upsample(scale_factor, output_size, num_output_dims, mode, align_corners);
}
/**
 * Public API: create a PixelShuffle layer (-> nn_pixel_shuffle).
 * @param upscale_factor (int)
 * @return (PixelShuffle*)
 */
PixelShuffle* cml_nn_pixel_shuffle(int upscale_factor) { return nn_pixel_shuffle(upscale_factor); }
/**
 * Public API: create a PixelUnshuffle layer (-> nn_pixel_unshuffle).
 * @param downscale_factor (int)
 * @return (PixelUnshuffle*)
 */
PixelUnshuffle* cml_nn_pixel_unshuffle(int downscale_factor) {
    return nn_pixel_unshuffle(downscale_factor);
}

/**
 * Public API: create a Flatten layer (-> nn_flatten).
 * @param start_dim (int)
 * @param end_dim (int)
 * @return (Flatten*)
 */
Flatten* cml_nn_flatten(int start_dim, int end_dim) { return nn_flatten(start_dim, end_dim); }
/**
 * Public API: create an Identity (pass-through) layer (-> nn_identity).
 * @return (Identity*)
 */
Identity* cml_nn_identity(void) { return nn_identity(); }
/**
 * Public API: create a 1-D batch-norm layer (-> nn_batchnorm1d).
 * @param num_features (int)
 * @param eps (float)
 * @param momentum (float)
 * @param affine (bool)
 * @param track_running_stats (bool)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (BatchNorm1d*)
 */
BatchNorm1d* cml_nn_batchnorm1d(int num_features, float eps, float momentum, bool affine,
                                bool track_running_stats, DType dtype, DeviceType device) {
    return nn_batchnorm1d(num_features, eps, momentum, affine, track_running_stats, dtype, device);
}
/**
 * Public API: create a PReLU layer (-> nn_prelu).
 * @param num_parameters (int)
 * @param init (float)
 * @param dtype (DType)
 * @param device (DeviceType)
 * @return (PReLU*)
 */
PReLU* cml_nn_prelu(int num_parameters, float init, DType dtype, DeviceType device) {
    return nn_prelu(num_parameters, init, dtype, device);
}

/**
 * Public API: functional interpolate/resize (-> f_interpolate).
 * @param input (Tensor*)
 * @param output_size (int*)
 * @param num_dims (int)
 * @param mode (UpsampleMode)
 * @param align_corners (bool)
 * @return (Tensor*)
 */
Tensor* cml_f_interpolate(Tensor* input, int* output_size, int num_dims, UpsampleMode mode,
                          bool align_corners) {
    return f_interpolate(input, output_size, num_dims, mode, align_corners);
}
/**
 * Public API: functional pixel shuffle (-> f_pixel_shuffle).
 * @param input (Tensor*)
 * @param upscale_factor (int)
 * @return (Tensor*)
 */
Tensor* cml_f_pixel_shuffle(Tensor* input, int upscale_factor) {
    return f_pixel_shuffle(input, upscale_factor);
}
/**
 * Public API: functional pixel unshuffle (-> f_pixel_unshuffle).
 * @param input (Tensor*)
 * @param downscale_factor (int)
 * @return (Tensor*)
 */
Tensor* cml_f_pixel_unshuffle(Tensor* input, int downscale_factor) {
    return f_pixel_unshuffle(input, downscale_factor);
}

/**
 * Public API: enter a mixed-precision autocast scope (-> autocast_enter).
 * @param target_dtype (DType)
 * @return (void)
 */
void cml_autocast_enter(DType target_dtype) { autocast_enter(target_dtype); }
/**
 * Public API: exit the autocast scope (-> autocast_exit).
 * @return (void)
 */
void cml_autocast_exit(void) { autocast_exit(); }
/**
 * Public API: whether autocast is active (-> autocast_is_enabled).
 * @return (bool)
 */
bool cml_autocast_is_enabled(void) { return autocast_is_enabled(); }
/**
 * Public API: the default autocast dtype (-> autocast_default_dtype).
 * @return (DType)
 */
DType cml_autocast_default_dtype(void) { return autocast_default_dtype(); }
/**
 * Public API: set the autocast dtype (-> autocast_set_dtype).
 * @param dtype (DType)
 * @return (void)
 */
void cml_autocast_set_dtype(DType dtype) { autocast_set_dtype(dtype); }
/**
 * Public API: get the current autocast dtype (-> autocast_get_dtype).
 * @return (DType)
 */
DType cml_autocast_get_dtype(void) { return autocast_get_dtype(); }
/**
 * Public API: create a gradient scaler for AMP (-> grad_scaler_create).
 * @param init_scale (float)
 * @param growth_factor (float)
 * @param backoff_factor (float)
 * @param growth_interval (int)
 * @return (GradScaler*)
 */
GradScaler* cml_grad_scaler_create(float init_scale, float growth_factor, float backoff_factor,
                                   int growth_interval) {
    return grad_scaler_create(init_scale, growth_factor, backoff_factor, growth_interval);
}
/**
 * Public API: free a gradient scaler (-> grad_scaler_free).
 * @param scaler (GradScaler*)
 * @return (void)
 */
void cml_grad_scaler_free(GradScaler* scaler) { grad_scaler_free(scaler); }
/**
 * Public API: scale a loss before backward (-> grad_scaler_scale).
 * @param scaler (GradScaler*)
 * @param loss (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_grad_scaler_scale(GradScaler* scaler, Tensor* loss) {
    return grad_scaler_scale(scaler, loss);
}
/**
 * Public API: unscale gradients in place (-> grad_scaler_unscale).
 * @param scaler (GradScaler*)
 * @param params (Parameter**)
 * @param num_params (int)
 * @return (void)
 */
void cml_grad_scaler_unscale(GradScaler* scaler, Parameter** params, int num_params) {
    grad_scaler_unscale(scaler, params, num_params);
}
/**
 * Public API: conditionally run the optimizer step under AMP (-> grad_scaler_step).
 * @param scaler (GradScaler*)
 * @param step_fn (void (*)(void*))
 * @param optimizer (void*)
 * @return (void)
 */
void cml_grad_scaler_step(GradScaler* scaler, void (*step_fn)(void*), void* optimizer) {
    grad_scaler_step(scaler, step_fn, optimizer);
}
/**
 * Public API: update the loss scale for next step (-> grad_scaler_update).
 * @param scaler (GradScaler*)
 * @return (void)
 */
void cml_grad_scaler_update(GradScaler* scaler) { grad_scaler_update(scaler); }

/**
 * Public API: build a sparse COO tensor from indices/values (-> sparse_coo_tensor).
 * @param indices (Tensor*)
 * @param values (Tensor*)
 * @param dense_shape (const int*)
 * @param dense_ndim (int)
 * @return (SparseCOOData*)
 */
SparseCOOData* cml_sparse_coo_tensor(Tensor* indices, Tensor* values, const int* dense_shape,
                                     int dense_ndim) {
    return sparse_coo_tensor(indices, values, dense_shape, dense_ndim);
}
/**
 * Public API: convert a dense tensor to sparse COO (-> sparse_from_dense).
 * @param dense (Tensor*)
 * @return (SparseCOOData*)
 */
SparseCOOData* cml_sparse_from_dense(Tensor* dense) { return sparse_from_dense(dense); }
/**
 * Public API: convert a sparse COO tensor to dense (-> sparse_to_dense).
 * @param sparse (SparseCOOData*)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_sparse_to_dense(SparseCOOData* sparse, const TensorConfig* config) {
    return sparse_to_dense(sparse, config);
}
/**
 * Public API: sparse-by-dense matrix multiply (-> sparse_matmul).
 * @param sparse (SparseCOOData*)
 * @param dense (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_sparse_matmul(SparseCOOData* sparse, Tensor* dense) {
    return sparse_matmul(sparse, dense);
}
/**
 * Public API: coalesce duplicate COO entries (-> sparse_coalesce).
 * @param sparse (SparseCOOData*)
 * @return (SparseCOOData*)
 */
SparseCOOData* cml_sparse_coalesce(SparseCOOData* sparse) { return sparse_coalesce(sparse); }
/**
 * Public API: free a sparse COO tensor (-> sparse_free).
 * @param sparse (SparseCOOData*)
 * @return (void)
 */
void cml_sparse_free(SparseCOOData* sparse) { sparse_free(sparse); }

/**
 * Public API: sorted values along a dim (-> tensor_sort).
 * @param a (Tensor*)
 * @param dim (int)
 * @param descending (bool)
 * @return (Tensor*)
 */
Tensor* cml_sort(Tensor* a, int dim, bool descending) { return tensor_sort(a, dim, descending); }
/**
 * Public API: top-k values along a dim (-> tensor_topk).
 * @param a (Tensor*)
 * @param k (int)
 * @param dim (int)
 * @param largest (bool)
 * @param sorted (bool)
 * @return (Tensor*)
 */
Tensor* cml_topk(Tensor* a, int k, int dim, bool largest, bool sorted) {
    return tensor_topk(a, k, dim, largest, sorted);
}
/**
 * Public API: top-k values plus their indices (-> uop_topk).
 * @param a (Tensor*)
 * @param k (int)
 * @param dim (int)
 * @param largest (bool)
 * @param indices_out (Tensor**)
 * @return (Tensor*)
 */
Tensor* cml_topk_with_indices(Tensor* a, int k, int dim, bool largest, Tensor** indices_out) {
    if (!a || !indices_out)
        return uop_topk(a, k, dim, largest, NULL);
    *indices_out = NULL;
    return uop_topk(a, k, dim, largest, indices_out);
}
/**
 * Route through uop_masked_select so it builds a differentiable node (VJP
 * scatters grad back to selected positions) rather than a bare non-diff copy.
 * @param a (Tensor*)
 * @param mask (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_masked_select(Tensor* a, Tensor* mask) { return uop_masked_select(a, mask); }
/**
 * Public API: coordinate grids from 1-D tensors (-> tensor_meshgrid).
 * @param tensors (Tensor**)
 * @param num_tensors (int)
 * @param num_outputs (int*)
 * @return (Tensor**)
 */
Tensor** cml_meshgrid(Tensor** tensors, int num_tensors, int* num_outputs) {
    return tensor_meshgrid(tensors, num_tensors, num_outputs);
}
/**
 * Public API: extract a diagonal (-> tensor_diagonal).
 * @param a (Tensor*)
 * @param offset (int)
 * @param dim1 (int)
 * @param dim2 (int)
 * @return (Tensor*)
 */
Tensor* cml_diagonal(Tensor* a, int offset, int dim1, int dim2) {
    return tensor_diagonal(a, offset, dim1, dim2);
}
/**
 * Public API: linear interpolation between a and b (-> tensor_lerp).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @param weight (float)
 * @return (Tensor*)
 */
Tensor* cml_lerp(Tensor* a, Tensor* b, float weight) { return tensor_lerp(a, b, weight); }
/**
 * Public API: integer (floor) division (-> tensor_idiv).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_idiv(Tensor* a, Tensor* b) { return tensor_idiv(a, b); }
/**
 * Public API: elementwise modulo (-> tensor_mod).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_mod(Tensor* a, Tensor* b) { return tensor_mod(a, b); }

/**
 * Public API: step LR scheduler (-> lr_scheduler_step).
 * @param opt (Optimizer*)
 * @param step_size (int)
 * @param gamma (float)
 * @return (LRScheduler*)
 */
LRScheduler* cml_lr_scheduler_step(Optimizer* opt, int step_size, float gamma) {
    return lr_scheduler_step(opt, step_size, gamma);
}
/**
 * Public API: reduce-on-plateau LR scheduler (-> lr_scheduler_reduce_on_plateau).
 * @param opt (Optimizer*)
 * @param factor (float)
 * @param patience (int)
 * @param min_lr (float)
 * @return (LRScheduler*)
 */
LRScheduler* cml_lr_scheduler_reduce_on_plateau(Optimizer* opt, float factor, int patience,
                                                float min_lr) {
    return lr_scheduler_reduce_on_plateau(opt, factor, patience, min_lr);
}
/**
 * Public API: exponential-decay LR scheduler (-> lr_scheduler_exponential).
 * @param opt (Optimizer*)
 * @param gamma (float)
 * @return (LRScheduler*)
 */
LRScheduler* cml_lr_scheduler_exponential(Optimizer* opt, float gamma) {
    return lr_scheduler_exponential(opt, gamma);
}
/**
 * Public API: cosine-annealing LR scheduler (-> lr_scheduler_cosine).
 * @param opt (Optimizer*)
 * @param T_max (int)
 * @param eta_min (float)
 * @return (LRScheduler*)
 */
LRScheduler* cml_lr_scheduler_cosine(Optimizer* opt, int T_max, float eta_min) {
    return lr_scheduler_cosine(opt, T_max, eta_min);
}
/**
 * Public API: one-cycle LR scheduler (-> lr_scheduler_one_cycle).
 * @param opt (Optimizer*)
 * @param max_lr (float)
 * @param total_steps (int)
 * @param pct_start (float)
 * @param div_factor (float)
 * @param final_div_factor (float)
 * @return (LRScheduler*)
 */
LRScheduler* cml_lr_scheduler_one_cycle(Optimizer* opt, float max_lr, int total_steps,
                                        float pct_start, float div_factor, float final_div_factor) {
    return lr_scheduler_one_cycle(opt, max_lr, total_steps, pct_start, div_factor,
                                  final_div_factor);
}
/**
 * Public API: multi-step (milestones) LR scheduler (-> lr_scheduler_multi_step).
 * @param opt (Optimizer*)
 * @param milestones (int*)
 * @param num_milestones (int)
 * @param gamma (float)
 * @return (LRScheduler*)
 */
LRScheduler* cml_lr_scheduler_multi_step(Optimizer* opt, int* milestones, int num_milestones,
                                         float gamma) {
    return lr_scheduler_multi_step(opt, milestones, num_milestones, gamma);
}
/**
 * Public API: polynomial-decay LR scheduler (-> lr_scheduler_polynomial).
 * @param opt (Optimizer*)
 * @param total_iters (int)
 * @param power (float)
 * @param min_lr (float)
 * @return (LRScheduler*)
 */
LRScheduler* cml_lr_scheduler_polynomial(Optimizer* opt, int total_iters, float power,
                                         float min_lr) {
    return lr_scheduler_polynomial(opt, total_iters, power, min_lr);
}
/**
 * Public API: warmup wrapper around another scheduler (-> lr_scheduler_warmup).
 * @param inner (LRScheduler*)
 * @param warmup_steps (int)
 * @param warmup_start_factor (float)
 * @return (LRScheduler*)
 */
LRScheduler* cml_lr_scheduler_warmup(LRScheduler* inner, int warmup_steps,
                                     float warmup_start_factor) {
    return lr_scheduler_warmup(inner, warmup_steps, warmup_start_factor);
}
/**
 * Public API: advance the scheduler and return the new LR (-> lr_scheduler_update).
 * @param scheduler (LRScheduler*)
 * @param metric (float)
 * @return (float)
 */
float cml_lr_scheduler_update(LRScheduler* scheduler, float metric) {
    return lr_scheduler_update(scheduler, metric);
}
/**
 * Public API: current learning rate (-> lr_scheduler_get_lr).
 * @param scheduler (LRScheduler*)
 * @return (float)
 */
float cml_lr_scheduler_get_lr(LRScheduler* scheduler) { return lr_scheduler_get_lr(scheduler); }
/**
 * Public API: free an LR scheduler (-> lr_scheduler_free).
 * @param scheduler (LRScheduler*)
 * @return (void)
 */
void cml_lr_scheduler_free(LRScheduler* scheduler) { lr_scheduler_free(scheduler); }
