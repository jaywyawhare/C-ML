/**
 * @file dispatch.c
 * @brief Unified dispatch layer implementation
 *
 * Routes IR execution to the appropriate backend: LLVM JIT, CUDA, ROCm, or CPU fallback.
 */

#include "ops/ir/dispatch.h"
#include "ops/ir/kernel_cache.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/ir/execution.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include "backend/device.h"
#include "backend/blas.h"

#ifdef CML_HAS_LLVM_BACKEND
#include "ops/ir/llvm/llvm_backend.h"
#include "ops/ir/gpu/gpu_codegen.h"
#endif

#include "ops/ir/gpu/cuda_backend.h"
#include "ops/ir/gpu/rocm_backend.h"

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>

static CMLCUDABackend* g_cuda_backend = NULL;
static bool g_cuda_init_attempted = false;
static CMLROCmBackend* g_rocm_backend = NULL;
static bool g_rocm_init_attempted = false;

static CMLDispatchContext* g_dispatch_ctx = NULL;

static const char* backend_names[] = {
    "CPU (Interpreter)",
    "CPU (LLVM JIT)",
    "CUDA",
    "ROCm",
};

static const char* backend_descriptions[] = {
    "CPU interpreter fallback (no JIT)",
    "LLVM IR to JIT compilation",
    "NVIDIA CUDA GPU acceleration",
    "AMD ROCm GPU acceleration",
};

CMLDispatchContext* cml_dispatch_create(void) {
    CMLDispatchContext* ctx = (CMLDispatchContext*)calloc(1, sizeof(CMLDispatchContext));
    if (!ctx) {
        LOG_ERROR("Failed to allocate dispatch context");
        return NULL;
    }

    for (int i = 0; i < CML_BACKEND_COUNT; i++) {
        ctx->backends[i].type                 = (CMLBackendType)i;
        ctx->backends[i].status               = CML_BACKEND_STATUS_UNAVAILABLE;
        ctx->backends[i].name                 = backend_names[i];
        ctx->backends[i].description          = backend_descriptions[i];
        ctx->backends[i].device_count         = 0;
        ctx->backends[i].total_memory         = 0;
        ctx->backends[i].supports_async       = false;
        ctx->backends[i].supports_unified_mem = false;
    }

    // CPU fallback is always available
    ctx->backends[CML_BACKEND_CPU_FALLBACK].status       = CML_BACKEND_STATUS_AVAILABLE;
    ctx->backends[CML_BACKEND_CPU_FALLBACK].device_count = 1;

    // Default fallback chain
    ctx->fallback_chain[0] = CML_BACKEND_CUDA;
    ctx->fallback_chain[1] = CML_BACKEND_ROCM;
    ctx->fallback_chain[2] = CML_BACKEND_CPU_LLVM;
    ctx->fallback_chain[3] = CML_BACKEND_CPU_FALLBACK;
    ctx->fallback_count    = 4;

    ctx->preferred   = CML_BACKEND_CPU_LLVM;
    ctx->active      = CML_BACKEND_CPU_FALLBACK;
    ctx->initialized = false;

    return ctx;
}

int cml_dispatch_init(CMLDispatchContext* ctx) {
    if (!ctx)
        return -1;

    cml_dispatch_detect_backends(ctx);

    // Set active backend to best available
    ctx->active = cml_dispatch_get_best_backend(ctx);
    ctx->initialized = true;

    // Check env var
    cml_dispatch_set_from_env(ctx);

    LOG_INFO("Dispatch initialized. Active backend: %s", backend_names[ctx->active]);
    return 0;
}

void cml_dispatch_free(CMLDispatchContext* ctx) {
    if (!ctx)
        return;

    if (ctx->cache) {
        cml_kernel_cache_free((CMLKernelCache*)ctx->cache);
        ctx->cache = NULL;
    }

    if (g_cuda_backend) {
        cml_cuda_backend_free(g_cuda_backend);
        g_cuda_backend = NULL;
    }
    if (g_rocm_backend) {
        cml_rocm_backend_free(g_rocm_backend);
        g_rocm_backend = NULL;
    }

    free(ctx);
    if (ctx == g_dispatch_ctx)
        g_dispatch_ctx = NULL;
}

CMLDispatchContext* cml_dispatch_get_global(void) {
    if (!g_dispatch_ctx) {
        g_dispatch_ctx = cml_dispatch_create();
        if (g_dispatch_ctx) {
            cml_dispatch_init(g_dispatch_ctx);
        }
    }
    return g_dispatch_ctx;
}

int cml_dispatch_detect_backends(CMLDispatchContext* ctx) {
    if (!ctx)
        return 0;

    int available = 1; // CPU fallback always available

#ifdef CML_HAS_LLVM_BACKEND
    ctx->backends[CML_BACKEND_CPU_LLVM].status       = CML_BACKEND_STATUS_AVAILABLE;
    ctx->backends[CML_BACKEND_CPU_LLVM].device_count  = 1;
    available++;
    LOG_INFO("LLVM JIT backend available");
#endif

    // CUDA detection (only attempt once)
    if (!g_cuda_init_attempted && cml_cuda_available()) {
        g_cuda_init_attempted = true;
        g_cuda_backend = cml_cuda_backend_create();
        if (g_cuda_backend && cml_cuda_backend_init(g_cuda_backend, 0) == 0) {
            ctx->backends[CML_BACKEND_CUDA].status       = CML_BACKEND_STATUS_INITIALIZED;
            ctx->backends[CML_BACKEND_CUDA].device_count  = 1;
            ctx->backends[CML_BACKEND_CUDA].supports_async = true;
            ctx->backend_contexts[CML_BACKEND_CUDA]       = g_cuda_backend;
            available++;
            LOG_INFO("CUDA backend initialized");
        } else if (g_cuda_backend) {
            cml_cuda_backend_free(g_cuda_backend);
            g_cuda_backend = NULL;
        }
    } else if (g_cuda_backend) {
        ctx->backends[CML_BACKEND_CUDA].status       = CML_BACKEND_STATUS_INITIALIZED;
        ctx->backends[CML_BACKEND_CUDA].device_count  = 1;
        ctx->backends[CML_BACKEND_CUDA].supports_async = true;
        ctx->backend_contexts[CML_BACKEND_CUDA]       = g_cuda_backend;
        available++;
    }

    // ROCm detection (only attempt once)
    if (!g_rocm_init_attempted && cml_rocm_available()) {
        g_rocm_init_attempted = true;
        g_rocm_backend = cml_rocm_backend_create();
        if (g_rocm_backend && cml_rocm_backend_init(g_rocm_backend, 0) == 0) {
            ctx->backends[CML_BACKEND_ROCM].status       = CML_BACKEND_STATUS_INITIALIZED;
            ctx->backends[CML_BACKEND_ROCM].device_count  = 1;
            ctx->backends[CML_BACKEND_ROCM].supports_async = true;
            ctx->backend_contexts[CML_BACKEND_ROCM]       = g_rocm_backend;
            available++;
            LOG_INFO("ROCm backend initialized");
        } else if (g_rocm_backend) {
            cml_rocm_backend_free(g_rocm_backend);
            g_rocm_backend = NULL;
        }
    } else if (g_rocm_backend) {
        ctx->backends[CML_BACKEND_ROCM].status       = CML_BACKEND_STATUS_INITIALIZED;
        ctx->backends[CML_BACKEND_ROCM].device_count  = 1;
        ctx->backends[CML_BACKEND_ROCM].supports_async = true;
        ctx->backend_contexts[CML_BACKEND_ROCM]       = g_rocm_backend;
        available++;
    }

    LOG_INFO("Detected %d available backends", available);
    return available;
}

int cml_dispatch_set_preferred(CMLDispatchContext* ctx, CMLBackendType backend) {
    if (!ctx || backend >= CML_BACKEND_COUNT)
        return -1;
    ctx->preferred = backend;
    if (cml_dispatch_backend_available(ctx, backend)) {
        ctx->active = backend;
    }
    return 0;
}

const CMLBackendInfo* cml_dispatch_get_backend_info(CMLDispatchContext* ctx,
                                                    CMLBackendType backend) {
    if (!ctx || backend >= CML_BACKEND_COUNT)
        return NULL;
    return &ctx->backends[backend];
}

bool cml_dispatch_backend_available(CMLDispatchContext* ctx, CMLBackendType backend) {
    if (!ctx || backend >= CML_BACKEND_COUNT)
        return false;
    return ctx->backends[backend].status >= CML_BACKEND_STATUS_AVAILABLE;
}

CMLBackendType cml_dispatch_get_best_backend(CMLDispatchContext* ctx) {
    if (!ctx)
        return CML_BACKEND_CPU_FALLBACK;

    // Prefer GPU backends
    if (cml_dispatch_backend_available(ctx, CML_BACKEND_CUDA))
        return CML_BACKEND_CUDA;
    if (cml_dispatch_backend_available(ctx, CML_BACKEND_ROCM))
        return CML_BACKEND_ROCM;
    if (cml_dispatch_backend_available(ctx, CML_BACKEND_CPU_LLVM))
        return CML_BACKEND_CPU_LLVM;
    return CML_BACKEND_CPU_FALLBACK;
}

const char* cml_dispatch_backend_name(CMLBackendType backend) {
    if (backend >= CML_BACKEND_COUNT)
        return "Unknown";
    return backend_names[backend];
}

// Execute on a specific backend
int cml_dispatch_execute_on(CMLDispatchContext* ctx, CMLBackendType backend, CMLGraph_t ir,
                            Tensor** inputs, int nin, Tensor** outputs, int nout) {
    (void)inputs; (void)nin; (void)outputs; (void)nout;

    if (!ctx || !ir)
        return -1;

    switch (backend) {
    case CML_BACKEND_CPU_FALLBACK:
        return cpu_execute_ir(ir);

#ifdef CML_HAS_LLVM_BACKEND
    case CML_BACKEND_CPU_LLVM: {
        extern CMLLLVMBackend* cml_get_llvm_backend(void);
        CMLLLVMBackend* llvm = cml_get_llvm_backend();
        if (llvm) {
            int result = cml_llvm_execute(llvm, ir);
            if (result == 0) {
                ctx->executions_total++;
                return 0;
            }
        }
        return -1;
    }
#endif

    case CML_BACKEND_CUDA:
    case CML_BACKEND_ROCM: {
#ifdef CML_HAS_LLVM_BACKEND
        static CMLGPUCodegen* g_gpu_cuda = NULL;
        static CMLGPUCodegen* g_gpu_rocm = NULL;
        CMLGPUCodegen** cg_ptr = (backend == CML_BACKEND_CUDA) ? &g_gpu_cuda : &g_gpu_rocm;
        if (!*cg_ptr) {
            GPUTarget tgt = (backend == CML_BACKEND_CUDA) ? GPU_TARGET_CUDA : GPU_TARGET_ROCM;
            *cg_ptr = cml_gpu_codegen_create(tgt, ctx->backend_contexts[backend]);
        }
        if (*cg_ptr) {
            int r = cml_gpu_execute(*cg_ptr, ir);
            if (r == 0) { ctx->executions_total++; return 0; }
        }
        return -1;
#else
        LOG_DEBUG("GPU backend %s: requires LLVM backend", backend_names[backend]);
        return -1;
#endif
    }

    default:
        return -1;
    }
}

int cml_dispatch_execute(CMLDispatchContext* ctx, CMLGraph_t ir, Tensor** inputs, int nin,
                         Tensor** outputs, int nout) {
    if (!ctx)
        ctx = cml_dispatch_get_global();
    if (!ctx || !ir) {
        LOG_ERROR("Invalid dispatch arguments");
        return -1;
    }

    // Try active backend first
    int result = cml_dispatch_execute_on(ctx, ctx->active, ir, inputs, nin, outputs, nout);
    if (result == 0)
        return 0;

    // Fallback through the chain
    for (int i = 0; i < ctx->fallback_count; i++) {
        CMLBackendType fallback = ctx->fallback_chain[i];
        if (fallback == ctx->active)
            continue;
        if (!cml_dispatch_backend_available(ctx, fallback))
            continue;

        LOG_INFO("Falling back to: %s", backend_names[fallback]);
        ctx->fallbacks_used++;

        result = cml_dispatch_execute_on(ctx, fallback, ir, inputs, nin, outputs, nout);
        if (result == 0)
            return 0;
    }

    LOG_ERROR("All backends failed");
    return -1;
}

CMLBackendType cml_dispatch_select_backend(CMLDispatchContext* ctx, CMLGraph_t ir) {
    if (!ctx || !ir)
        return CML_BACKEND_CPU_FALLBACK;

    struct IRNode* node = ir->head;
    bool has_cuda_tensors = false;
    bool has_rocm_tensors = false;

    while (node) {
        if (node->output && node->output->device == DEVICE_CUDA)
            has_cuda_tensors = true;
        else if (node->output && node->output->device == DEVICE_ROCM)
            has_rocm_tensors = true;
        node = node->next;
    }

    if (has_cuda_tensors && cml_dispatch_backend_available(ctx, CML_BACKEND_CUDA))
        return CML_BACKEND_CUDA;
    if (has_rocm_tensors && cml_dispatch_backend_available(ctx, CML_BACKEND_ROCM))
        return CML_BACKEND_ROCM;

    if (cml_dispatch_backend_available(ctx, ctx->preferred))
        return ctx->preferred;

    return cml_dispatch_get_best_backend(ctx);
}

int cml_dispatch_enable_cache(CMLDispatchContext* ctx, size_t max_entries) {
    if (!ctx)
        return -1;
    if (ctx->cache)
        cml_kernel_cache_free((CMLKernelCache*)ctx->cache);
    ctx->cache = (struct CMLKernelCache*)cml_kernel_cache_create(max_entries);
    return ctx->cache ? 0 : -1;
}

void cml_dispatch_disable_cache(CMLDispatchContext* ctx) {
    if (!ctx || !ctx->cache)
        return;
    cml_kernel_cache_free((CMLKernelCache*)ctx->cache);
    ctx->cache = NULL;
}

void cml_dispatch_clear_cache(CMLDispatchContext* ctx) {
    if (!ctx || !ctx->cache)
        return;
    kernel_cache_clear((CMLKernelCache*)ctx->cache);
}

void cml_dispatch_cache_stats(CMLDispatchContext* ctx, size_t* hits, size_t* misses, size_t* size) {
    if (!ctx)
        return;
    if (ctx->cache) {
        size_t count = 0;
        kernel_cache_stats((CMLKernelCache*)ctx->cache, hits, misses, &count, size);
    } else {
        if (hits) *hits = ctx->cache_hits;
        if (misses) *misses = ctx->cache_misses;
        if (size) *size = 0;
    }
}

void cml_dispatch_print_status(CMLDispatchContext* ctx) {
    if (!ctx) {
        printf("Dispatch context: NULL\n");
        return;
    }

    printf("\n=== CML Dispatch Status ===\n");
    printf("Initialized: %s\n", ctx->initialized ? "Yes" : "No");
    printf("Preferred: %s\n", backend_names[ctx->preferred]);
    printf("Active: %s\n", backend_names[ctx->active]);
    printf("\nBackends:\n");

    for (int i = 0; i < CML_BACKEND_COUNT; i++) {
        const char* status;
        switch (ctx->backends[i].status) {
        case CML_BACKEND_STATUS_UNAVAILABLE: status = "Unavailable"; break;
        case CML_BACKEND_STATUS_AVAILABLE:   status = "Available"; break;
        case CML_BACKEND_STATUS_INITIALIZED: status = "Initialized"; break;
        case CML_BACKEND_STATUS_ERROR:       status = "Error"; break;
        default:                             status = "Unknown"; break;
        }
        printf("  [%d] %-20s: %s", i, backend_names[i], status);
        if (ctx->backends[i].device_count > 0)
            printf(" (%d devices)", ctx->backends[i].device_count);
        printf("\n");
    }

    printf("\nStatistics:\n");
    printf("  Total executions: %zu\n", ctx->executions_total);
    printf("  Cache hits: %zu\n", ctx->cache_hits);
    printf("  Cache misses: %zu\n", ctx->cache_misses);
    printf("  Fallbacks used: %zu\n", ctx->fallbacks_used);
    printf("===========================\n\n");
}

void cml_dispatch_synchronize(CMLDispatchContext* ctx) {
    if (!ctx)
        return;
    if (ctx->backends[CML_BACKEND_CUDA].status == CML_BACKEND_STATUS_INITIALIZED && g_cuda_backend)
        cml_cuda_synchronize(g_cuda_backend);
    if (ctx->backends[CML_BACKEND_ROCM].status == CML_BACKEND_STATUS_INITIALIZED && g_rocm_backend)
        cml_rocm_synchronize(g_rocm_backend);
}

int cml_dispatch_set_from_env(CMLDispatchContext* ctx) {
    if (!ctx)
        return -1;

    const char* env = getenv("CML_BACKEND");
    if (!env)
        return 0;

    CMLBackendType backend = CML_BACKEND_CPU_FALLBACK;

    if (strcasecmp(env, "cpu") == 0 || strcasecmp(env, "llvm") == 0 ||
        strcasecmp(env, "cpu_llvm") == 0) {
        backend = CML_BACKEND_CPU_LLVM;
    } else if (strcasecmp(env, "fallback") == 0 || strcasecmp(env, "interpreter") == 0) {
        backend = CML_BACKEND_CPU_FALLBACK;
    } else if (strcasecmp(env, "cuda") == 0 || strcasecmp(env, "nvidia") == 0) {
        backend = CML_BACKEND_CUDA;
    } else if (strcasecmp(env, "rocm") == 0 || strcasecmp(env, "amd") == 0 ||
               strcasecmp(env, "hip") == 0) {
        backend = CML_BACKEND_ROCM;
    } else {
        LOG_WARNING("Unknown CML_BACKEND value: %s", env);
        return -1;
    }

    LOG_INFO("CML_BACKEND=%s -> %s", env, backend_names[backend]);
    return cml_dispatch_set_preferred(ctx, backend);
}
