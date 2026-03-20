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
#include "ops/ir/gpu/ptx_codegen.h"

#ifdef CML_HAS_VULKAN
#include "ops/ir/gpu/vulkan_backend.h"
#endif

#ifdef CML_HAS_NV_DRIVER
#include "ops/ir/gpu/nv_driver.h"
#endif

#ifdef CML_HAS_AM_DRIVER
#include "ops/ir/gpu/am_driver.h"
#endif

#ifdef CML_HAS_NIR
#include "ops/ir/nir_compiler.h"
#endif

#ifdef CML_HAS_METAL
#include "ops/ir/gpu/metal_backend.h"
#endif

#ifdef CML_HAS_WEBGPU
#include "ops/ir/gpu/webgpu_backend.h"
#endif

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>

static CMLCUDABackend* g_cuda_backend = NULL;
static bool g_cuda_init_attempted = false;
static CMLROCmBackend* g_rocm_backend = NULL;
static bool g_rocm_init_attempted = false;

#ifdef CML_HAS_VULKAN
static CMLVulkanBackend* g_vulkan_backend = NULL;
static bool g_vulkan_init_attempted = false;
#endif

#ifdef CML_HAS_NV_DRIVER
static CMLNVDriver* g_nv_driver = NULL;
static bool g_nv_init_attempted = false;
#endif

#ifdef CML_HAS_AM_DRIVER
static CMLAMDriver* g_am_driver = NULL;
static bool g_am_init_attempted = false;
#endif

static CMLDispatchContext* g_dispatch_ctx = NULL;

static const char* backend_names[] = {
    "CPU (Interpreter)",
    "CPU (LLVM JIT)",
    "NV (Userspace)",
    "CUDA",
    "AM (Userspace)",
    "ROCm",
    "NIR (Mesa)",
    "Metal",
    "Vulkan",
    "WebGPU",
};

static const char* backend_descriptions[] = {
    "CPU interpreter fallback (no JIT)",
    "LLVM IR to JIT compilation",
    "NVIDIA userspace driver (direct ioctl)",
    "NVIDIA CUDA GPU acceleration",
    "AMD userspace driver (KFD ioctl)",
    "AMD ROCm GPU acceleration",
    "Mesa NIR multi-vendor GPU",
    "Apple Metal GPU acceleration",
    "Vulkan/SPIR-V compute",
    "WebGPU via wgpu-native",
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

    ctx->backends[CML_BACKEND_CPU_FALLBACK].status       = CML_BACKEND_STATUS_AVAILABLE;
    ctx->backends[CML_BACKEND_CPU_FALLBACK].device_count = 1;

    ctx->fallback_chain[0] = CML_BACKEND_NV;
    ctx->fallback_chain[1] = CML_BACKEND_CUDA;
    ctx->fallback_chain[2] = CML_BACKEND_AM;
    ctx->fallback_chain[3] = CML_BACKEND_ROCM;
    ctx->fallback_chain[4] = CML_BACKEND_NIR;
    ctx->fallback_chain[5] = CML_BACKEND_METAL;
    ctx->fallback_chain[6] = CML_BACKEND_VULKAN;
    ctx->fallback_chain[7] = CML_BACKEND_WEBGPU;
    ctx->fallback_chain[8] = CML_BACKEND_CPU_LLVM;
    ctx->fallback_chain[9] = CML_BACKEND_CPU_FALLBACK;
    ctx->fallback_count    = 10;

    ctx->preferred   = CML_BACKEND_CPU_LLVM;
    ctx->active      = CML_BACKEND_CPU_FALLBACK;
    ctx->initialized = false;

    return ctx;
}

int cml_dispatch_init(CMLDispatchContext* ctx) {
    if (!ctx)
        return -1;

    cml_dispatch_detect_backends(ctx);

    ctx->active = cml_dispatch_get_best_backend(ctx);
    ctx->initialized = true;

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

#ifdef CML_HAS_VULKAN
    if (g_vulkan_backend) {
        cml_vulkan_backend_free(g_vulkan_backend);
        g_vulkan_backend = NULL;
    }
#endif

#ifdef CML_HAS_NV_DRIVER
    if (g_nv_driver) {
        cml_nv_driver_free(g_nv_driver);
        g_nv_driver = NULL;
    }
#endif

#ifdef CML_HAS_AM_DRIVER
    if (g_am_driver) {
        cml_am_driver_free(g_am_driver);
        g_am_driver = NULL;
    }
#endif

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

#ifdef CML_HAS_VULKAN
    if (!g_vulkan_init_attempted && cml_vulkan_available()) {
        g_vulkan_init_attempted = true;
        g_vulkan_backend = cml_vulkan_backend_create();
        if (g_vulkan_backend && cml_vulkan_backend_init(g_vulkan_backend) == 0) {
            ctx->backends[CML_BACKEND_VULKAN].status        = CML_BACKEND_STATUS_INITIALIZED;
            ctx->backends[CML_BACKEND_VULKAN].device_count   = 1;
            ctx->backends[CML_BACKEND_VULKAN].supports_async = true;
            ctx->backend_contexts[CML_BACKEND_VULKAN]        = g_vulkan_backend;
            available++;
            LOG_INFO("Vulkan backend initialized: %s", g_vulkan_backend->device_name);
        } else if (g_vulkan_backend) {
            cml_vulkan_backend_free(g_vulkan_backend);
            g_vulkan_backend = NULL;
        }
    } else if (g_vulkan_backend) {
        ctx->backends[CML_BACKEND_VULKAN].status        = CML_BACKEND_STATUS_INITIALIZED;
        ctx->backends[CML_BACKEND_VULKAN].device_count   = 1;
        ctx->backends[CML_BACKEND_VULKAN].supports_async = true;
        ctx->backend_contexts[CML_BACKEND_VULKAN]        = g_vulkan_backend;
        available++;
    }
#endif

#ifdef CML_HAS_NV_DRIVER
    if (!g_nv_init_attempted && cml_nv_driver_available()) {
        g_nv_init_attempted = true;
        g_nv_driver = cml_nv_driver_create();
        if (g_nv_driver && cml_nv_driver_init(g_nv_driver) == 0) {
            ctx->backends[CML_BACKEND_NV].status        = CML_BACKEND_STATUS_INITIALIZED;
            ctx->backends[CML_BACKEND_NV].device_count   = 1;
            ctx->backends[CML_BACKEND_NV].supports_async = true;
            ctx->backend_contexts[CML_BACKEND_NV]        = g_nv_driver;
            available++;
            LOG_INFO("NV userspace driver initialized");
        } else if (g_nv_driver) {
            cml_nv_driver_free(g_nv_driver);
            g_nv_driver = NULL;
        }
    } else if (g_nv_driver) {
        ctx->backends[CML_BACKEND_NV].status        = CML_BACKEND_STATUS_INITIALIZED;
        ctx->backend_contexts[CML_BACKEND_NV]        = g_nv_driver;
        available++;
    }
#endif

#ifdef CML_HAS_AM_DRIVER
    if (!g_am_init_attempted && cml_am_driver_available()) {
        g_am_init_attempted = true;
        g_am_driver = cml_am_driver_create();
        if (g_am_driver && cml_am_driver_init(g_am_driver) == 0) {
            ctx->backends[CML_BACKEND_AM].status        = CML_BACKEND_STATUS_INITIALIZED;
            ctx->backends[CML_BACKEND_AM].device_count   = 1;
            ctx->backends[CML_BACKEND_AM].supports_async = true;
            ctx->backend_contexts[CML_BACKEND_AM]        = g_am_driver;
            available++;
            LOG_INFO("AM userspace driver initialized");
        } else if (g_am_driver) {
            cml_am_driver_free(g_am_driver);
            g_am_driver = NULL;
        }
    } else if (g_am_driver) {
        ctx->backends[CML_BACKEND_AM].status        = CML_BACKEND_STATUS_INITIALIZED;
        ctx->backend_contexts[CML_BACKEND_AM]        = g_am_driver;
        available++;
    }
#endif

#ifdef CML_HAS_NIR
    if (cml_nir_available()) {
        ctx->backends[CML_BACKEND_NIR].status        = CML_BACKEND_STATUS_AVAILABLE;
        ctx->backends[CML_BACKEND_NIR].device_count   = 1;
        available++;
        LOG_INFO("NIR/Mesa backend available");
    }
#endif

#ifdef CML_HAS_METAL
    if (cml_metal_available()) {
        CMLMetalBackend* metal = cml_metal_backend_create();
        if (metal && cml_metal_backend_init(metal) == 0) {
            ctx->backends[CML_BACKEND_METAL].status          = CML_BACKEND_STATUS_INITIALIZED;
            ctx->backends[CML_BACKEND_METAL].device_count    = 1;
            ctx->backends[CML_BACKEND_METAL].supports_async  = true;
            ctx->backends[CML_BACKEND_METAL].supports_unified_mem = true;
            ctx->backend_contexts[CML_BACKEND_METAL]         = metal;
            available++;
            LOG_INFO("Metal backend initialized");
        } else {
            if (metal) cml_metal_backend_free(metal);
            ctx->backends[CML_BACKEND_METAL].status       = CML_BACKEND_STATUS_AVAILABLE;
            ctx->backends[CML_BACKEND_METAL].device_count  = 1;
            ctx->backends[CML_BACKEND_METAL].supports_async = true;
            available++;
            LOG_INFO("Metal backend available (init deferred)");
        }
    }
#endif

#ifdef CML_HAS_WEBGPU
    if (cml_webgpu_available()) {
        CMLWebGPUBackend* wgpu = cml_webgpu_backend_create();
        if (wgpu && cml_webgpu_backend_init(wgpu) == 0) {
            ctx->backends[CML_BACKEND_WEBGPU].status       = CML_BACKEND_STATUS_INITIALIZED;
            ctx->backends[CML_BACKEND_WEBGPU].device_count  = 1;
            ctx->backends[CML_BACKEND_WEBGPU].supports_async = true;
            ctx->backend_contexts[CML_BACKEND_WEBGPU]       = wgpu;
            available++;
            LOG_INFO("WebGPU backend initialized");
        } else {
            if (wgpu) cml_webgpu_backend_free(wgpu);
            ctx->backends[CML_BACKEND_WEBGPU].status       = CML_BACKEND_STATUS_AVAILABLE;
            ctx->backends[CML_BACKEND_WEBGPU].device_count  = 1;
            ctx->backends[CML_BACKEND_WEBGPU].supports_async = true;
            available++;
            LOG_INFO("WebGPU backend available (init deferred)");
        }
    }
#endif

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

    /*
     * For CPU tensors, prefer CPU backends (LLVM JIT > BLAS fallback).
     * GPU backends (Vulkan, etc.) add dispatch overhead that exceeds
     * the compute time for CPU-resident data.  Only use GPU backends
     * when tensors are on a GPU device.
     */

    /* GPU backends — only beneficial when data is already on device */
    if (cml_dispatch_backend_available(ctx, CML_BACKEND_NV))
        return CML_BACKEND_NV;
    if (cml_dispatch_backend_available(ctx, CML_BACKEND_CUDA))
        return CML_BACKEND_CUDA;
    if (cml_dispatch_backend_available(ctx, CML_BACKEND_AM))
        return CML_BACKEND_AM;
    if (cml_dispatch_backend_available(ctx, CML_BACKEND_ROCM))
        return CML_BACKEND_ROCM;
    if (cml_dispatch_backend_available(ctx, CML_BACKEND_NIR))
        return CML_BACKEND_NIR;
    if (cml_dispatch_backend_available(ctx, CML_BACKEND_METAL))
        return CML_BACKEND_METAL;

    /* CPU backends — prefer LLVM JIT over software Vulkan/WebGPU */
    if (cml_dispatch_backend_available(ctx, CML_BACKEND_CPU_LLVM))
        return CML_BACKEND_CPU_LLVM;

    /* Software GPU backends — last resort before pure fallback */
    if (cml_dispatch_backend_available(ctx, CML_BACKEND_VULKAN))
        return CML_BACKEND_VULKAN;
    if (cml_dispatch_backend_available(ctx, CML_BACKEND_WEBGPU))
        return CML_BACKEND_WEBGPU;

    return CML_BACKEND_CPU_FALLBACK;
}

const char* cml_dispatch_backend_name(CMLBackendType backend) {
    if (backend >= CML_BACKEND_COUNT)
        return "Unknown";
    return backend_names[backend];
}

int cml_dispatch_execute_on(CMLDispatchContext* ctx, CMLBackendType backend, CMLGraph_t ir,
                            Tensor** inputs, int nin, Tensor** outputs, int nout) {
    (void)inputs; (void)nin; (void)outputs; (void)nout;

    if (!ctx || !ir)
        return -1;

    switch (backend) {
    case CML_BACKEND_CPU_FALLBACK: {
        int r = cpu_execute_ir(ir);
        if (r == 0) ctx->executions_total++;
        return r;
    }

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
        /* LLVM GPU codegen failed; fall through to PTX string codegen */
#endif
        /* PTX string codegen fallback (no LLVM dependency) */
        if (backend == CML_BACKEND_CUDA && g_cuda_backend && g_cuda_backend->initialized) {
            static CMLPTXCodegen* g_ptx_cg = NULL;
            if (!g_ptx_cg) {
                int sm = g_cuda_backend->compute_capability_major * 10
                       + g_cuda_backend->compute_capability_minor;
                g_ptx_cg = cml_ptx_codegen_create(sm, g_cuda_backend);
            }
            if (g_ptx_cg) {
                int r = cml_ptx_execute_graph(g_ptx_cg, ir);
                if (r == 0) { ctx->executions_total++; return 0; }
            }
            LOG_DEBUG("PTX string codegen: graph execution unavailable");
        }
        LOG_DEBUG("GPU backend %s: no codegen path available", backend_names[backend]);
        return -1;
    }

    case CML_BACKEND_VULKAN:
#ifdef CML_HAS_VULKAN
    {
        CMLVulkanBackend* vk =
            (CMLVulkanBackend*)ctx->backend_contexts[CML_BACKEND_VULKAN];
        if (vk) {
            int r = cml_vulkan_execute_graph(vk, ir);
            if (r == 0) { ctx->executions_total++; return 0; }
        }
        LOG_DEBUG("Vulkan backend execution failed");
        return -1;
    }
#else
        LOG_DEBUG("Vulkan backend not compiled");
        return -1;
#endif

    case CML_BACKEND_NV:
#ifdef CML_HAS_NV_DRIVER
    {
        CMLNVDriver* nv = (CMLNVDriver*)ctx->backend_contexts[CML_BACKEND_NV];
        if (nv) {
            int r = cml_nv_execute_graph(nv, ir);
            if (r == 0) { ctx->executions_total++; return 0; }
        }
        LOG_DEBUG("NV driver execution failed");
        return -1;
    }
#else
        LOG_DEBUG("NV driver not compiled");
        return -1;
#endif

    case CML_BACKEND_AM:
#ifdef CML_HAS_AM_DRIVER
    {
        CMLAMDriver* am = (CMLAMDriver*)ctx->backend_contexts[CML_BACKEND_AM];
        if (am) {
            int r = cml_am_execute_graph(am, ir);
            if (r == 0) { ctx->executions_total++; return 0; }
        }
        LOG_DEBUG("AM driver execution failed");
        return -1;
    }
#else
        LOG_DEBUG("AM driver not compiled");
        return -1;
#endif

    case CML_BACKEND_NIR:
#ifdef CML_HAS_NIR
    {
        /* NIR compiles to SPIR-V, then executes via Vulkan */
        CMLNIRCompiler* nir = cml_nir_compiler_create(NIR_TARGET_NVK);
        if (nir && cml_nir_compile(nir, ir) == 0) {
            /* Execute via Vulkan backend if available */
#ifdef CML_HAS_VULKAN
            CMLVulkanBackend* vk =
                (CMLVulkanBackend*)ctx->backend_contexts[CML_BACKEND_VULKAN];
            if (vk) {
                int r = cml_vulkan_execute_graph(vk, ir);
                cml_nir_compiler_free(nir);
                if (r == 0) { ctx->executions_total++; return 0; }
            }
#endif
            cml_nir_compiler_free(nir);
        } else if (nir) {
            cml_nir_compiler_free(nir);
        }
        LOG_DEBUG("NIR backend execution failed");
        return -1;
    }
#else
        LOG_DEBUG("NIR backend not compiled");
        return -1;
#endif

    case CML_BACKEND_METAL:
#ifdef CML_HAS_METAL
    {
        CMLMetalBackend* metal =
            (CMLMetalBackend*)ctx->backend_contexts[CML_BACKEND_METAL];
        if (metal) {
            int r = cml_metal_execute_graph(metal, ir);
            if (r == 0) { ctx->executions_total++; return 0; }
        }
        LOG_DEBUG("Metal backend execution failed");
        return -1;
    }
#else
        LOG_DEBUG("Metal backend not compiled");
        return -1;
#endif

    case CML_BACKEND_WEBGPU:
#ifdef CML_HAS_WEBGPU
    {
        CMLWebGPUBackend* wgpu =
            (CMLWebGPUBackend*)ctx->backend_contexts[CML_BACKEND_WEBGPU];
        if (wgpu) {
            int r = cml_webgpu_execute_graph(wgpu, ir);
            if (r == 0) { ctx->executions_total++; return 0; }
        }
        LOG_DEBUG("WebGPU backend execution failed");
        return -1;
    }
#else
        LOG_DEBUG("WebGPU backend not compiled");
        return -1;
#endif

    default:
        return -1;
    }
}

int cml_dispatch_execute_async(CMLDispatchContext* ctx, CMLGraph_t ir,
                               Tensor** inputs, int num_inputs,
                               Tensor** outputs, int num_outputs) {
    /* Async execution: currently dispatches synchronously.
     * With HCQ integration, this will submit to a hardware command queue
     * and return immediately, allowing overlapped compute/transfer. */
    return cml_dispatch_execute(ctx, ir, inputs, num_inputs, outputs, num_outputs);
}

int cml_dispatch_execute(CMLDispatchContext* ctx, CMLGraph_t ir, Tensor** inputs, int nin,
                         Tensor** outputs, int nout) {
    if (!ctx)
        ctx = cml_dispatch_get_global();
    if (!ctx || !ir) {
        LOG_ERROR("Invalid dispatch arguments");
        return -1;
    }

    int result = cml_dispatch_execute_on(ctx, ctx->active, ir, inputs, nin, outputs, nout);
    if (result == 0)
        return 0;

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

    printf("\nCML Dispatch Status\n");
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
    printf("\n");
}

void cml_dispatch_synchronize(CMLDispatchContext* ctx) {
    if (!ctx)
        return;
    if (ctx->backends[CML_BACKEND_CUDA].status == CML_BACKEND_STATUS_INITIALIZED && g_cuda_backend)
        cml_cuda_synchronize(g_cuda_backend);
    if (ctx->backends[CML_BACKEND_ROCM].status == CML_BACKEND_STATUS_INITIALIZED && g_rocm_backend)
        cml_rocm_synchronize(g_rocm_backend);
#ifdef CML_HAS_VULKAN
    if (ctx->backends[CML_BACKEND_VULKAN].status == CML_BACKEND_STATUS_INITIALIZED && g_vulkan_backend)
        cml_vulkan_synchronize(g_vulkan_backend);
#endif
}

struct CMLCUDABackend* cml_dispatch_get_cuda_backend(void) {
    return g_cuda_backend;
}

struct CMLVulkanBackend* cml_dispatch_get_vulkan_backend(void) {
#ifdef CML_HAS_VULKAN
    return g_vulkan_backend;
#else
    return NULL;
#endif
}

struct CMLNVDriver* cml_dispatch_get_nv_driver(void) {
#ifdef CML_HAS_NV_DRIVER
    return g_nv_driver;
#else
    return NULL;
#endif
}

struct CMLAMDriver* cml_dispatch_get_am_driver(void) {
#ifdef CML_HAS_AM_DRIVER
    return g_am_driver;
#else
    return NULL;
#endif
}

int cml_dispatch_execute_jit(CMLDispatchContext* ctx, CMLGraph_t ir,
                             Tensor** inputs, int num_inputs,
                             Tensor** outputs, int num_outputs) {
    /*
     * Route execution through the TinyJit layer.  On the first call for a
     * given graph structure the graph is executed normally and the launch
     * sequence is recorded.  Subsequent calls with the same structure replay
     * the cached trace, skipping scheduling and code generation.
     *
     * Falls back to regular dispatch if the TinyJit path is unavailable.
     */
    (void)inputs; (void)num_inputs; (void)outputs; (void)num_outputs;

    if (!ctx) ctx = cml_dispatch_get_global();
    if (!ctx || !ir) return -1;

    /* Try traced execution (capture-and-replay) */
    extern int cml_ir_execute_traced(CMLGraph_t ir);
    int rc = cml_ir_execute_traced(ir);
    if (rc == 0) {
        ctx->executions_total++;
        return 0;
    }

    /* Fall back to normal dispatch */
    return cml_dispatch_execute(ctx, ir, inputs, num_inputs, outputs, num_outputs);
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
    } else if (strcasecmp(env, "metal") == 0) {
        backend = CML_BACKEND_METAL;
    } else if (strcasecmp(env, "vulkan") == 0 || strcasecmp(env, "vk") == 0) {
        backend = CML_BACKEND_VULKAN;
    } else if (strcasecmp(env, "nv") == 0 || strcasecmp(env, "nv_driver") == 0) {
        backend = CML_BACKEND_NV;
    } else if (strcasecmp(env, "am") == 0 || strcasecmp(env, "am_driver") == 0 ||
               strcasecmp(env, "kfd") == 0) {
        backend = CML_BACKEND_AM;
    } else if (strcasecmp(env, "nir") == 0 || strcasecmp(env, "mesa") == 0) {
        backend = CML_BACKEND_NIR;
    } else if (strcasecmp(env, "webgpu") == 0 || strcasecmp(env, "wgpu") == 0) {
        backend = CML_BACKEND_WEBGPU;
    } else {
        LOG_WARNING("Unknown CML_BACKEND value: %s", env);
        return -1;
    }

    LOG_INFO("CML_BACKEND=%s -> %s", env, backend_names[backend]);
    return cml_dispatch_set_preferred(ctx, backend);
}
