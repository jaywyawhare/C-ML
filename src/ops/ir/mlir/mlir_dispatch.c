/**
 * @file mlir_dispatch.c
 * @brief Unified dispatch layer implementation
 */

#include "ops/ir/mlir/mlir_dispatch.h"
#include "ops/ir/mlir/mlir_kernel_cache.h"
#include "ops/ir/mlir/mlir_codegen.h"
#include "ops/ir/mlir/backends/vulkan_backend.h"
#include "ops/ir/mlir/backends/cuda_backend.h"
#include "ops/ir/mlir/backends/rocm_backend.h"
#include "ops/ir/mlir/backends/metal_backend.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/ir/execution.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include "backend/device.h"
#include "backend/blas.h"

#ifdef CML_HAS_MLIR
#include "ops/ir/mlir/mlir_internal.h"
#include "ops/ir/mlir/mlir_uops_builder.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <strings.h> // For strcasecmp
#include <stdio.h>

// BLAS context for matmul optimization
static CMLBlasContext* g_blas_ctx = NULL;

// Global backend instances (lazily initialized)
static CMLCUDABackend* g_cuda_backend     = NULL;
static CMLROCmBackend* g_rocm_backend     = NULL;
static CMLMetalBackend* g_metal_backend   = NULL;
static CMLVulkanBackend* g_vulkan_backend = NULL;

// Global dispatch context
static CMLDispatchContext* g_dispatch_ctx = NULL;

// Backend name strings (indexed by CMLBackendType enum)
static const char* backend_names[] = {
    "CPU (Interpreter)", // CML_BACKEND_CPU_FALLBACK = 0
    "CPU (LLVM JIT)",    // CML_BACKEND_CPU_LLVM = 1
    "CUDA",              // CML_BACKEND_CUDA = 2
    "ROCm",              // CML_BACKEND_ROCM = 3
    "Metal",             // CML_BACKEND_METAL = 4
    "Vulkan",            // CML_BACKEND_VULKAN = 5
};

// Backend descriptions (indexed by CMLBackendType enum)
static const char* backend_descriptions[] = {
    "CPU interpreter fallback (no JIT)", // CML_BACKEND_CPU_FALLBACK = 0
    "MLIR to LLVM JIT compilation",      // CML_BACKEND_CPU_LLVM = 1
    "NVIDIA CUDA GPU acceleration",      // CML_BACKEND_CUDA = 2
    "AMD ROCm GPU acceleration",         // CML_BACKEND_ROCM = 3
    "Apple Metal GPU acceleration",      // CML_BACKEND_METAL = 4
    "Vulkan cross-platform GPU",         // CML_BACKEND_VULKAN = 5
};

// ============================================================================
// Forward declarations for backend-specific execution
// ============================================================================

// MLIR JIT execution (when MLIR available)
// Note: cml_mlir_init() and other MLIR functions are declared in mlir_internal.h
#ifdef CML_HAS_MLIR
extern int cml_mlir_execute(void* ctx, Tensor** inputs, int nin, Tensor** outputs, int nout);
extern int cml_mlir_optimize(const void* module, const void* context);
#endif

// ============================================================================
// Initialization and Cleanup
// ============================================================================

CMLDispatchContext* cml_dispatch_create(void) {
    CMLDispatchContext* ctx = (CMLDispatchContext*)calloc(1, sizeof(CMLDispatchContext));
    if (!ctx) {
        LOG_ERROR("Failed to allocate dispatch context");
        return NULL;
    }

    // Initialize all backends as unavailable
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

    // Default fallback chain: try GPU first, then CPU LLVM, then fallback
    ctx->fallback_chain[0] = CML_BACKEND_CUDA;
    ctx->fallback_chain[1] = CML_BACKEND_METAL;
    ctx->fallback_chain[2] = CML_BACKEND_ROCM;
    ctx->fallback_chain[3] = CML_BACKEND_VULKAN;
    ctx->fallback_chain[4] = CML_BACKEND_CPU_LLVM;
    ctx->fallback_chain[5] = CML_BACKEND_CPU_FALLBACK;
    ctx->fallback_count    = 6;

    ctx->preferred   = CML_BACKEND_CPU_LLVM; // Default to LLVM JIT
    ctx->active      = CML_BACKEND_CPU_FALLBACK;
    ctx->cache       = NULL;
    ctx->initialized = false;

    return ctx;
}

int cml_dispatch_init(CMLDispatchContext* ctx) {
    if (!ctx)
        return -1;

    if (ctx->initialized) {
        LOG_DEBUG("Dispatch context already initialized");
        return 0;
    }

    // Detect available backends
    int num_backends = cml_dispatch_detect_backends(ctx);
    LOG_INFO("Detected %d available backends", num_backends);

    // Try to set from environment variable
    cml_dispatch_set_from_env(ctx);

    // Select best available backend
    ctx->active = cml_dispatch_get_best_backend(ctx);
    LOG_INFO("Active backend: %s", backend_names[ctx->active]);

    ctx->initialized = true;
    return 0;
}

void cml_dispatch_free(CMLDispatchContext* ctx) {
    if (!ctx)
        return;

    // Free backend contexts with backend-specific cleanup
    for (int i = 0; i < CML_BACKEND_COUNT; i++) {
        if (ctx->backend_contexts[i]) {
            switch (i) {
            case CML_BACKEND_CUDA:
                cml_cuda_backend_free((CMLCUDABackend*)ctx->backend_contexts[i]);
                if (g_cuda_backend == (CMLCUDABackend*)ctx->backend_contexts[i]) {
                    g_cuda_backend = NULL;
                }
                break;
            case CML_BACKEND_ROCM:
                cml_rocm_backend_free((CMLROCmBackend*)ctx->backend_contexts[i]);
                if (g_rocm_backend == (CMLROCmBackend*)ctx->backend_contexts[i]) {
                    g_rocm_backend = NULL;
                }
                break;
            case CML_BACKEND_METAL:
                cml_metal_backend_free((CMLMetalBackend*)ctx->backend_contexts[i]);
                if (g_metal_backend == (CMLMetalBackend*)ctx->backend_contexts[i]) {
                    g_metal_backend = NULL;
                }
                break;
            case CML_BACKEND_VULKAN:
                cml_vulkan_backend_free((CMLVulkanBackend*)ctx->backend_contexts[i]);
                if (g_vulkan_backend == (CMLVulkanBackend*)ctx->backend_contexts[i]) {
                    g_vulkan_backend = NULL;
                }
                break;
            default:
                // CPU backends don't need special cleanup
                break;
            }
            ctx->backend_contexts[i] = NULL;
        }
    }

    // Free kernel cache if present
    if (ctx->cache) {
        cml_kernel_cache_free((CMLKernelCache*)ctx->cache);
        ctx->cache = NULL;
    }

    if (ctx == g_dispatch_ctx) {
        g_dispatch_ctx = NULL;
    }

    free(ctx);
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

// ============================================================================
// Backend Detection
// ============================================================================

int cml_dispatch_detect_backends(CMLDispatchContext* ctx) {
    if (!ctx)
        return 0;

    int count = 1; // CPU fallback always available

    // Check MLIR availability
#ifdef CML_HAS_MLIR
    ctx->backends[CML_BACKEND_CPU_LLVM].status       = CML_BACKEND_STATUS_AVAILABLE;
    ctx->backends[CML_BACKEND_CPU_LLVM].device_count = 1;
    count++;
    LOG_DEBUG("MLIR/LLVM JIT backend available");
#else
    LOG_DEBUG("MLIR not compiled in, LLVM JIT unavailable");
#endif

    // Check CUDA availability
    if (device_cuda_available()) {
        ctx->backends[CML_BACKEND_CUDA].status         = CML_BACKEND_STATUS_AVAILABLE;
        int cuda_count                                 = device_cuda_get_count();
        ctx->backends[CML_BACKEND_CUDA].device_count   = cuda_count > 0 ? cuda_count : 1;
        ctx->backends[CML_BACKEND_CUDA].supports_async = true;
        count++;
        LOG_DEBUG("CUDA backend available with %d device(s)",
                  ctx->backends[CML_BACKEND_CUDA].device_count);
    }

    // Check ROCm availability
    if (device_rocm_available()) {
        ctx->backends[CML_BACKEND_ROCM].status         = CML_BACKEND_STATUS_AVAILABLE;
        int rocm_count                                 = device_rocm_get_count();
        ctx->backends[CML_BACKEND_ROCM].device_count   = rocm_count > 0 ? rocm_count : 1;
        ctx->backends[CML_BACKEND_ROCM].supports_async = true;
        count++;
        LOG_DEBUG("ROCm backend available with %d device(s)",
                  ctx->backends[CML_BACKEND_ROCM].device_count);
    }

    // Check Metal availability
    if (device_metal_available()) {
        ctx->backends[CML_BACKEND_METAL].status               = CML_BACKEND_STATUS_AVAILABLE;
        ctx->backends[CML_BACKEND_METAL].device_count         = 1;
        ctx->backends[CML_BACKEND_METAL].supports_unified_mem = true; // Apple Silicon
        ctx->backends[CML_BACKEND_METAL].supports_async       = true;
        count++;
        LOG_DEBUG("Metal backend available");
    }

    // Check Vulkan availability
    if (cml_vulkan_available()) {
        ctx->backends[CML_BACKEND_VULKAN].status         = CML_BACKEND_STATUS_AVAILABLE;
        ctx->backends[CML_BACKEND_VULKAN].device_count   = 1;
        ctx->backends[CML_BACKEND_VULKAN].supports_async = true;
        count++;
        LOG_DEBUG("Vulkan backend available");
    }

    return count;
}

int cml_dispatch_set_preferred(CMLDispatchContext* ctx, CMLBackendType backend) {
    if (!ctx || backend >= CML_BACKEND_COUNT)
        return -1;

    if (ctx->backends[backend].status == CML_BACKEND_STATUS_UNAVAILABLE) {
        LOG_WARNING("Backend %s not available, keeping current: %s", backend_names[backend],
                    backend_names[ctx->preferred]);
        return -1;
    }

    ctx->preferred = backend;
    ctx->active    = backend;
    LOG_INFO("Set preferred backend to: %s", backend_names[backend]);
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
    return ctx->backends[backend].status != CML_BACKEND_STATUS_UNAVAILABLE;
}

CMLBackendType cml_dispatch_get_best_backend(CMLDispatchContext* ctx) {
    if (!ctx)
        return CML_BACKEND_CPU_FALLBACK;

    // MLIR JIT still has issues - use CPU fallback with BLAS/SIMD optimizations
    // TODO: Fix remaining MLIR lowering issues for full JIT support
    //
    // To enable MLIR JIT (experimental), explicitly call:
    //   cml_dispatch_set_preferred(ctx, CML_BACKEND_CPU_LLVM);
    //
    // if (cml_dispatch_backend_available(ctx, CML_BACKEND_CPU_LLVM)) {
    //     return CML_BACKEND_CPU_LLVM;
    // }

    // Use CPU fallback with BLAS/SIMD optimizations
    return CML_BACKEND_CPU_FALLBACK;
}

const char* cml_dispatch_backend_name(CMLBackendType backend) {
    if (backend >= CML_BACKEND_COUNT)
        return "Unknown";
    return backend_names[backend];
}

// ============================================================================
// BLAS Optimization
// ============================================================================

/**
 * @brief Check if IR can be handled by BLAS (simple matmul)
 *
 * For BLAS to handle the operation, the IR must:
 * 1. Consist of a single MATMUL operation
 * 2. Have 2D contiguous input tensors
 * 3. Be on CPU device
 */
static bool can_use_blas(CMLIR_t ir, Tensor** inputs, int nin) {
    if (!ir || !inputs || nin != 2)
        return false;

    // Check if IR is a single matmul
    if (ir->node_count != 1)
        return false;

    struct IRNode* node = ir->head;
    if (!node || node->type != UOP_MATMUL)
        return false;

    // Check input tensors
    for (int i = 0; i < nin; i++) {
        Tensor* t = inputs[i];
        if (!t || !t->data)
            return false;

        // Must be 2D
        if (t->ndim != 2)
            return false;

        // Must be on CPU
        if (t->device != DEVICE_CPU)
            return false;

        // Must be contiguous (row-major, strides = [dim1, 1])
        if (t->ndim >= 2) {
            size_t expected_stride = 1;
            for (int d = t->ndim - 1; d >= 0; d--) {
                if (t->strides[d] != expected_stride)
                    return false;
                expected_stride *= (size_t)t->shape[d];
            }
        }
    }

    return true;
}

/**
 * @brief Execute matmul via BLAS
 *
 * Uses cblas_sgemm for optimized matrix multiplication.
 */
static int dispatch_execute_blas_matmul(CMLIR_t ir, Tensor** inputs, int nin, Tensor** outputs,
                                        int nout) {
    (void)ir; // IR validated in can_use_blas

    if (nin != 2 || nout != 1) {
        LOG_ERROR("BLAS matmul requires 2 inputs and 1 output");
        return -1;
    }

    // Initialize BLAS context if needed
    if (!g_blas_ctx) {
        g_blas_ctx = cml_blas_init();
        if (!g_blas_ctx) {
            LOG_WARNING("Failed to initialize BLAS context");
            return -1;
        }
    }

    Tensor* A = inputs[0];  // [M x K]
    Tensor* B = inputs[1];  // [K x N]
    Tensor* C = outputs[0]; // [M x N]

    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];

    // Validate dimensions
    if (B->shape[0] != K) {
        LOG_ERROR("BLAS matmul: dimension mismatch A[%d x %d] @ B[%d x %d]", M, K, B->shape[0],
                  B->shape[1]);
        return -1;
    }

    if (C->shape[0] != M || C->shape[1] != N) {
        LOG_ERROR("BLAS matmul: output dimension mismatch");
        return -1;
    }

    LOG_DEBUG("BLAS sgemm: [%d x %d] @ [%d x %d] -> [%d x %d]", M, K, K, N, M, N);

    // Execute BLAS sgemm: C = 1.0 * A @ B + 0.0 * C
    int result = cml_blas_sgemm(g_blas_ctx, (const float*)A->data, (const float*)B->data,
                                (float*)C->data, M, N, K, 1.0f, 0.0f);

    if (result == 0) {
        LOG_DEBUG("BLAS matmul completed successfully");
    } else {
        LOG_ERROR("BLAS matmul failed");
    }

    return result;
}

// ============================================================================
// Cached Kernel Execution
// ============================================================================

/**
 * @brief Execute a cached kernel
 *
 * This function executes a previously compiled and cached kernel without
 * recompilation. It dispatches to the appropriate backend based on the
 * kernel's backend type.
 *
 * @param cached The cached kernel entry
 * @param inputs Input tensors
 * @param nin Number of inputs
 * @param outputs Output tensors
 * @param nout Number of outputs
 * @return 0 on success, -1 on failure
 */
static int execute_cached_kernel(CMLKernelEntry* cached, Tensor** inputs, int nin, Tensor** outputs,
                                 int nout) {
    if (!cached || !cached->compiled) {
        LOG_ERROR("Invalid cached kernel entry");
        return -1;
    }

    LOG_DEBUG("Executing cached kernel, backend=%d, hash=0x%016llx", cached->backend,
              (unsigned long long)cached->hash);

    switch (cached->backend) {
    case CML_KERNEL_CUDA: {
#ifdef CML_HAS_MLIR
        if (!g_cuda_backend) {
            LOG_ERROR("CUDA backend not initialized for cached kernel");
            return -1;
        }
        CMLCUDAKernel* kernel = (CMLCUDAKernel*)cached->compiled;

        // Allocate GPU memory and copy inputs
        CUdeviceptr* d_inputs  = (CUdeviceptr*)calloc(nin, sizeof(CUdeviceptr));
        CUdeviceptr* d_outputs = (CUdeviceptr*)calloc(nout, sizeof(CUdeviceptr));
        int result             = -1;

        if (!d_inputs || !d_outputs) {
            free(d_inputs);
            free(d_outputs);
            return -1;
        }

        // Copy inputs to GPU
        for (int i = 0; i < nin; i++) {
            size_t size = tensor_numel(inputs[i]->shape, inputs[i]->ndim) * sizeof(float);
            d_inputs[i] = cml_cuda_malloc(g_cuda_backend, size);
            if (!d_inputs[i] ||
                cml_cuda_memcpy_h2d(g_cuda_backend, d_inputs[i], inputs[i]->data, size) != 0) {
                goto cuda_cache_cleanup;
            }
        }

        // Allocate output buffers
        for (int i = 0; i < nout; i++) {
            size_t size  = tensor_numel(outputs[i]->shape, outputs[i]->ndim) * sizeof(float);
            d_outputs[i] = cml_cuda_malloc(g_cuda_backend, size);
            if (!d_outputs[i]) {
                goto cuda_cache_cleanup;
            }
        }

        // Build kernel arguments
        if (nin < 0 || nout < 0 || nin + nout <= 0) {
            goto cuda_cache_cleanup;
        }
        void** args = (void**)calloc((size_t)(nin + nout), sizeof(void*));
        if (!args) {
            goto cuda_cache_cleanup;
        }
        for (int i = 0; i < nin; i++)
            args[i] = &d_inputs[i];
        for (int i = 0; i < nout; i++)
            args[nin + i] = &d_outputs[i];

        // Launch cached kernel
        result = cml_cuda_launch_kernel(g_cuda_backend, kernel, args, nin + nout);
        free(args);

        if (result == 0) {
            cml_cuda_synchronize(g_cuda_backend);
            // Copy outputs back
            for (int i = 0; i < nout; i++) {
                size_t size = tensor_numel(outputs[i]->shape, outputs[i]->ndim) * sizeof(float);
                if (cml_cuda_memcpy_d2h(g_cuda_backend, outputs[i]->data, d_outputs[i], size) !=
                    0) {
                    result = -1;
                    break;
                }
            }
        }

    cuda_cache_cleanup:
        for (int i = 0; i < nin; i++)
            if (d_inputs[i])
                cml_cuda_free(g_cuda_backend, d_inputs[i]);
        for (int i = 0; i < nout; i++)
            if (d_outputs[i])
                cml_cuda_free(g_cuda_backend, d_outputs[i]);
        free(d_inputs);
        free(d_outputs);
        return result;
#else
        return -1;
#endif
    }

    case CML_KERNEL_ROCM: {
#ifdef CML_HAS_MLIR
        if (!g_rocm_backend)
            return -1;
        CMLROCmKernel* kernel = (CMLROCmKernel*)cached->compiled;

        hipDeviceptr_t* d_inputs  = (hipDeviceptr_t*)calloc(nin, sizeof(hipDeviceptr_t));
        hipDeviceptr_t* d_outputs = (hipDeviceptr_t*)calloc(nout, sizeof(hipDeviceptr_t));
        int result                = -1;

        if (!d_inputs || !d_outputs) {
            free(d_inputs);
            free(d_outputs);
            return -1;
        }

        for (int i = 0; i < nin; i++) {
            size_t size = tensor_numel(inputs[i]->shape, inputs[i]->ndim) * sizeof(float);
            d_inputs[i] = cml_rocm_malloc(g_rocm_backend, size);
            if (!d_inputs[i] ||
                cml_rocm_memcpy_h2d(g_rocm_backend, d_inputs[i], inputs[i]->data, size) != 0) {
                goto rocm_cache_cleanup;
            }
        }

        for (int i = 0; i < nout; i++) {
            size_t size  = tensor_numel(outputs[i]->shape, outputs[i]->ndim) * sizeof(float);
            d_outputs[i] = cml_rocm_malloc(g_rocm_backend, size);
            if (!d_outputs[i])
                goto rocm_cache_cleanup;
        }

        if (nin < 0 || nout < 0 || nin + nout <= 0) {
            goto rocm_cache_cleanup;
        }
        void** args = (void**)calloc((size_t)(nin + nout), sizeof(void*));
        if (!args) {
            goto rocm_cache_cleanup;
        }
        for (int i = 0; i < nin; i++)
            args[i] = &d_inputs[i];
        for (int i = 0; i < nout; i++)
            args[nin + i] = &d_outputs[i];

        result = cml_rocm_launch_kernel(g_rocm_backend, kernel, args, nin + nout);
        free(args);

        if (result == 0) {
            cml_rocm_synchronize(g_rocm_backend);
            for (int i = 0; i < nout; i++) {
                size_t size = tensor_numel(outputs[i]->shape, outputs[i]->ndim) * sizeof(float);
                if (cml_rocm_memcpy_d2h(g_rocm_backend, outputs[i]->data, d_outputs[i], size) !=
                    0) {
                    result = -1;
                    break;
                }
            }
        }

    rocm_cache_cleanup:
        for (int i = 0; i < nin; i++)
            if (d_inputs[i])
                cml_rocm_free(g_rocm_backend, d_inputs[i]);
        for (int i = 0; i < nout; i++)
            if (d_outputs[i])
                cml_rocm_free(g_rocm_backend, d_outputs[i]);
        free(d_inputs);
        free(d_outputs);
        return result;
#else
        return -1;
#endif
    }

    case CML_KERNEL_METAL: {
#ifdef CML_HAS_MLIR
        if (!g_metal_backend)
            return -1;
        CMLMetalKernel* kernel = (CMLMetalKernel*)cached->compiled;

        // Metal uses unified memory - pass host pointers directly
        void** args = (void**)calloc(nin + nout, sizeof(void*));
        if (!args)
            return -1;

        for (int i = 0; i < nin; i++)
            args[i] = inputs[i]->data;
        for (int i = 0; i < nout; i++)
            args[nin + i] = outputs[i]->data;

        int result = cml_metal_launch_kernel(g_metal_backend, kernel, args, nin + nout);
        free(args);

        if (result == 0) {
            cml_metal_synchronize(g_metal_backend);
        }
        return result;
#else
        return -1;
#endif
    }

    case CML_KERNEL_VULKAN: {
#ifdef CML_HAS_MLIR
        if (!g_vulkan_backend)
            return -1;
        CMLVulkanKernel* kernel = (CMLVulkanKernel*)cached->compiled;

        VkBuffer* buffers        = (VkBuffer*)calloc(nin + nout, sizeof(VkBuffer));
        VkDeviceMemory* memories = (VkDeviceMemory*)calloc(nin + nout, sizeof(VkDeviceMemory));
        int result               = -1;

        if (!buffers || !memories) {
            free(buffers);
            free(memories);
            return -1;
        }

        for (int i = 0; i < nin; i++) {
            size_t size = tensor_numel(inputs[i]->shape, inputs[i]->ndim) * sizeof(float);
            buffers[i]  = cml_vulkan_create_buffer(g_vulkan_backend, size, &memories[i]);
            if (!buffers[i] ||
                cml_vulkan_upload_data(g_vulkan_backend, memories[i], inputs[i]->data, size) != 0) {
                goto vulkan_cache_cleanup;
            }
        }

        for (int i = 0; i < nout; i++) {
            size_t size      = tensor_numel(outputs[i]->shape, outputs[i]->ndim) * sizeof(float);
            buffers[nin + i] = cml_vulkan_create_buffer(g_vulkan_backend, size, &memories[nin + i]);
            if (!buffers[nin + i])
                goto vulkan_cache_cleanup;
        }

        uint32_t gx = outputs[0]->shape[0];
        uint32_t gy = (outputs[0]->ndim > 1) ? outputs[0]->shape[1] : 1;

        result = cml_vulkan_launch_kernel(g_vulkan_backend, kernel, buffers, nin + nout, gx, gy, 1);
        if (result == 0) {
            cml_vulkan_synchronize(g_vulkan_backend);
            for (int i = 0; i < nout; i++) {
                size_t size = tensor_numel(outputs[i]->shape, outputs[i]->ndim) * sizeof(float);
                if (cml_vulkan_download_data(g_vulkan_backend, memories[nin + i], outputs[i]->data,
                                             size) != 0) {
                    result = -1;
                    break;
                }
            }
        }

    vulkan_cache_cleanup:
        for (int i = 0; i < nin + nout; i++) {
            if (buffers[i])
                cml_vulkan_destroy_buffer(g_vulkan_backend, buffers[i], memories[i]);
        }
        free(buffers);
        free(memories);
        return result;
#else
        return -1;
#endif
    }

    case CML_KERNEL_CPU_LLVM:
    case CML_KERNEL_CPU_FALLBACK:
    default:
        // CPU cached execution would require storing MLIR ExecutionEngine
        // For now, fall through to normal execution
        LOG_DEBUG("Cached CPU execution not yet implemented");
        return -1;
    }
}

// ============================================================================
// Execution
// ============================================================================

// Execute on CPU fallback interpreter
static int dispatch_execute_cpu_fallback(CMLIR_t ir) {
    LOG_DEBUG("Executing on CPU fallback interpreter");
    return cpu_execute_ir(ir);
}

// Execute on CPU with LLVM JIT
static int dispatch_execute_cpu_llvm(CMLDispatchContext* ctx, CMLIR_t ir, Tensor** inputs, int nin,
                                     Tensor** outputs, int nout) {
#ifdef CML_HAS_MLIR
    LOG_DEBUG("Executing on CPU LLVM JIT");

    // Initialize MLIR context if needed
    if (!ir->mlir_ctx) {
        ir->mlir_ctx = cml_mlir_init();
        if (!ir->mlir_ctx) {
            LOG_ERROR("Failed to initialize MLIR context");
            return -1;
        }
    }

    // Build MLIR module from IR
    if (!cml_mlir_build_from_ir(ir->mlir_ctx, ir)) {
        LOG_ERROR("Failed to build MLIR module");
        return -1;
    }

    // Get MLIR context for optimization - use proper struct type
    CMLMLIRContext* mlir_ctx = (CMLMLIRContext*)ir->mlir_ctx;

    // Optimize
    if (cml_mlir_optimize(mlir_ctx->module.ptr, mlir_ctx->context.ptr) != 0) {
        LOG_WARNING("MLIR optimization failed, continuing anyway");
    }

    // Execute
    return cml_mlir_execute(ir->mlir_ctx, inputs, nin, outputs, nout);
#else
    LOG_WARNING("MLIR not available, falling back to CPU interpreter");
    (void)ctx;
    (void)inputs;
    (void)nin;
    (void)outputs;
    (void)nout;
    return dispatch_execute_cpu_fallback(ir);
#endif
}

// Execute on CUDA
static int dispatch_execute_cuda(CMLDispatchContext* ctx, CMLIR_t ir, Tensor** inputs, int nin,
                                 Tensor** outputs, int nout) {
#ifdef CML_HAS_MLIR
    LOG_DEBUG("Executing on CUDA backend");
    int result = -1; // Initialize to failure, set to 0 on success

    // Validate inputs
    if (nin > 0 && !inputs) {
        LOG_WARNING("CUDA dispatch: nin=%d but inputs is NULL, falling back", nin);
        return -1;
    }
    if (nout > 0 && !outputs) {
        LOG_WARNING("CUDA dispatch: nout=%d but outputs is NULL, falling back", nout);
        return -1;
    }

    // Initialize CUDA backend if needed
    if (!g_cuda_backend) {
        g_cuda_backend = (CMLCUDABackend*)calloc(1, sizeof(CMLCUDABackend));
        if (!g_cuda_backend || cml_cuda_backend_init(g_cuda_backend, 0) != 0) {
            LOG_WARNING("Failed to initialize CUDA backend");
            free(g_cuda_backend);
            g_cuda_backend = NULL;
            return -1;
        }
    }

    // Build MLIR module from IR if needed
    if (!ir->mlir_ctx) {
        ir->mlir_ctx = cml_mlir_init();
        if (!ir->mlir_ctx || !cml_mlir_build_from_ir(ir->mlir_ctx, ir)) {
            LOG_ERROR("Failed to build MLIR module for CUDA");
            return -1;
        }
    }

    // Get MLIR module - use proper struct type
    CMLMLIRContext* mlir_ctx = (CMLMLIRContext*)ir->mlir_ctx;
    if (!mlir_ctx || mlirModuleIsNull(mlir_ctx->module)) {
        LOG_ERROR("No MLIR module available");
        return -1;
    }

    // Generate PTX code
    char* ptx = cml_mlir_gen_ptx(mlir_ctx->module.ptr);
    if (!ptx) {
        LOG_WARNING("Failed to generate PTX, falling back");
        return -1;
    }

    // Compile PTX to CUDA kernel
    CMLCUDAKernel* kernel = cml_cuda_compile_ptx(g_cuda_backend, ptx, "main_kernel");
    free(ptx);
    if (!kernel) {
        LOG_WARNING("Failed to compile PTX kernel");
        return -1;
    }

    // Allocate GPU memory and copy inputs
    CUdeviceptr* d_inputs  = calloc(nin, sizeof(CUdeviceptr));
    CUdeviceptr* d_outputs = calloc(nout, sizeof(CUdeviceptr));
    if (!d_inputs || !d_outputs) {
        cml_cuda_kernel_free(g_cuda_backend, kernel);
        free(d_inputs);
        free(d_outputs);
        return -1;
    }

    // Copy inputs to GPU
    for (int i = 0; i < nin; i++) {
        size_t size = tensor_numel(inputs[i]->shape, inputs[i]->ndim) * sizeof(float);
        d_inputs[i] = cml_cuda_malloc(g_cuda_backend, size);
        if (!d_inputs[i] ||
            cml_cuda_memcpy_h2d(g_cuda_backend, d_inputs[i], inputs[i]->data, size) != 0) {
            LOG_ERROR("Failed to copy input %d to GPU", i);
            goto cuda_cleanup;
        }
    }

    // Allocate output buffers on GPU
    for (int i = 0; i < nout; i++) {
        size_t size  = tensor_numel(outputs[i]->shape, outputs[i]->ndim) * sizeof(float);
        d_outputs[i] = cml_cuda_malloc(g_cuda_backend, size);
        if (!d_outputs[i]) {
            LOG_ERROR("Failed to allocate output %d on GPU", i);
            goto cuda_cleanup;
        }
    }

    // Set kernel launch config based on output dimensions
    int grid_x = outputs[0]->shape[0];
    int grid_y = (outputs[0]->ndim > 1) ? outputs[0]->shape[1] : 1;
    cml_cuda_kernel_set_launch_config(kernel, grid_x, grid_y, 1, 1, 1, 1);

    // Build kernel arguments
    void** args = calloc(nin + nout, sizeof(void*));
    for (int i = 0; i < nin; i++)
        args[i] = &d_inputs[i];
    for (int i = 0; i < nout; i++)
        args[nin + i] = &d_outputs[i];

    // Launch kernel
    result = cml_cuda_launch_kernel(g_cuda_backend, kernel, args, nin + nout);
    free(args);

    if (result != 0) {
        LOG_ERROR("CUDA kernel launch failed");
        goto cuda_cleanup;
    }

    // Synchronize
    cml_cuda_synchronize(g_cuda_backend);

    // Copy outputs back
    for (int i = 0; i < nout; i++) {
        size_t size = tensor_numel(outputs[i]->shape, outputs[i]->ndim) * sizeof(float);
        if (cml_cuda_memcpy_d2h(g_cuda_backend, outputs[i]->data, d_outputs[i], size) != 0) {
            LOG_ERROR("Failed to copy output %d from GPU", i);
            goto cuda_cleanup;
        }
    }

    result = 0; // Success if we get here

cuda_cleanup:
    for (int i = 0; i < nin; i++)
        if (d_inputs[i])
            cml_cuda_free(g_cuda_backend, d_inputs[i]);
    for (int i = 0; i < nout; i++)
        if (d_outputs[i])
            cml_cuda_free(g_cuda_backend, d_outputs[i]);
    free(d_inputs);
    free(d_outputs);
    cml_cuda_kernel_free(g_cuda_backend, kernel);

    ctx->executions_total++;
    return result;
#else
    (void)ctx;
    (void)ir;
    (void)inputs;
    (void)nin;
    (void)outputs;
    (void)nout;
    LOG_WARNING("CUDA backend requires MLIR support");
    return -1;
#endif
}

// Execute on ROCm
static int dispatch_execute_rocm(CMLDispatchContext* ctx, CMLIR_t ir, Tensor** inputs, int nin,
                                 Tensor** outputs, int nout) {
#ifdef CML_HAS_MLIR
    LOG_DEBUG("Executing on ROCm backend");

    // Validate inputs
    if (nin > 0 && !inputs) {
        LOG_WARNING("ROCm dispatch: nin=%d but inputs is NULL, falling back", nin);
        return -1;
    }
    if (nout > 0 && !outputs) {
        LOG_WARNING("ROCm dispatch: nout=%d but outputs is NULL, falling back", nout);
        return -1;
    }

    // Initialize ROCm backend if needed
    if (!g_rocm_backend) {
        g_rocm_backend = (CMLROCmBackend*)calloc(1, sizeof(CMLROCmBackend));
        if (!g_rocm_backend || cml_rocm_backend_init(g_rocm_backend, 0) != 0) {
            LOG_WARNING("Failed to initialize ROCm backend");
            free(g_rocm_backend);
            g_rocm_backend = NULL;
            return -1;
        }
    }

    // Build MLIR module from IR if needed
    if (!ir->mlir_ctx) {
        ir->mlir_ctx = cml_mlir_init();
        if (!ir->mlir_ctx || !cml_mlir_build_from_ir(ir->mlir_ctx, ir)) {
            LOG_ERROR("Failed to build MLIR module for ROCm");
            return -1;
        }
    }

    // Get MLIR module - use proper struct type
    CMLMLIRContext* mlir_ctx = (CMLMLIRContext*)ir->mlir_ctx;
    if (!mlir_ctx || mlirModuleIsNull(mlir_ctx->module)) {
        LOG_ERROR("No MLIR module available");
        return -1;
    }

    // For ROCm, we need AMDGCN/HSACO code
    // Currently we don't have direct HSACO codegen, so we use PTX via HIP's CUDA compatibility
    // Note: This requires HIP to be built with CUDA support, which may not always be available
    // TODO: Add proper AMDGCN codegen via MLIR rocdl dialect
    char* ptx = cml_mlir_gen_ptx(mlir_ctx->module.ptr);
    if (!ptx) {
        LOG_WARNING("Failed to generate PTX for ROCm compatibility layer");
        return -1;
    }

    // Try to compile PTX via HIP's CUDA compatibility (may not work on all setups)
    CMLROCmKernel* kernel = cml_rocm_compile_hsaco(g_rocm_backend, ptx, "main_kernel");
    free(ptx);
    if (!kernel) {
        LOG_WARNING("Failed to compile kernel for ROCm (HSACO/PTX compilation failed)");
        return -1;
    }

    // Allocate GPU memory and copy inputs
    hipDeviceptr_t* d_inputs  = (hipDeviceptr_t*)calloc(nin, sizeof(hipDeviceptr_t));
    hipDeviceptr_t* d_outputs = (hipDeviceptr_t*)calloc(nout, sizeof(hipDeviceptr_t));
    int result                = -1;

    if (!d_inputs || !d_outputs) {
        cml_rocm_kernel_free(g_rocm_backend, kernel);
        free(d_inputs);
        free(d_outputs);
        return -1;
    }

    // Copy inputs to GPU
    for (int i = 0; i < nin; i++) {
        size_t size = tensor_numel(inputs[i]->shape, inputs[i]->ndim) * sizeof(float);
        d_inputs[i] = cml_rocm_malloc(g_rocm_backend, size);
        if (!d_inputs[i] ||
            cml_rocm_memcpy_h2d(g_rocm_backend, d_inputs[i], inputs[i]->data, size) != 0) {
            LOG_ERROR("Failed to copy input %d to ROCm device", i);
            goto rocm_cleanup;
        }
    }

    // Allocate output buffers on GPU
    for (int i = 0; i < nout; i++) {
        size_t size  = tensor_numel(outputs[i]->shape, outputs[i]->ndim) * sizeof(float);
        d_outputs[i] = cml_rocm_malloc(g_rocm_backend, size);
        if (!d_outputs[i]) {
            LOG_ERROR("Failed to allocate output %d on ROCm device", i);
            goto rocm_cleanup;
        }
    }

    // Set kernel launch config based on output dimensions
    kernel->grid_dim[0]  = outputs[0]->shape[0];
    kernel->grid_dim[1]  = (outputs[0]->ndim > 1) ? outputs[0]->shape[1] : 1;
    kernel->grid_dim[2]  = 1;
    kernel->block_dim[0] = 1;
    kernel->block_dim[1] = 1;
    kernel->block_dim[2] = 1;

    // Build kernel arguments
    void** args = (void**)calloc(nin + nout, sizeof(void*));
    for (int i = 0; i < nin; i++)
        args[i] = &d_inputs[i];
    for (int i = 0; i < nout; i++)
        args[nin + i] = &d_outputs[i];

    // Launch kernel
    result = cml_rocm_launch_kernel(g_rocm_backend, kernel, args, nin + nout);
    free(args);

    if (result != 0) {
        LOG_ERROR("ROCm kernel launch failed");
        goto rocm_cleanup;
    }

    // Synchronize
    cml_rocm_synchronize(g_rocm_backend);

    // Copy outputs back
    for (int i = 0; i < nout; i++) {
        size_t size = tensor_numel(outputs[i]->shape, outputs[i]->ndim) * sizeof(float);
        if (cml_rocm_memcpy_d2h(g_rocm_backend, outputs[i]->data, d_outputs[i], size) != 0) {
            LOG_ERROR("Failed to copy output %d from ROCm device", i);
            goto rocm_cleanup;
        }
    }

    result = 0; // Success

rocm_cleanup:
    for (int i = 0; i < nin; i++)
        if (d_inputs[i])
            cml_rocm_free(g_rocm_backend, d_inputs[i]);
    for (int i = 0; i < nout; i++)
        if (d_outputs[i])
            cml_rocm_free(g_rocm_backend, d_outputs[i]);
    free(d_inputs);
    free(d_outputs);
    cml_rocm_kernel_free(g_rocm_backend, kernel);

    ctx->executions_total++;
    return result;
#else
    (void)ctx;
    (void)ir;
    (void)inputs;
    (void)nin;
    (void)outputs;
    (void)nout;
    LOG_WARNING("ROCm backend requires MLIR support");
    return -1;
#endif
}

// Execute on Metal
static int dispatch_execute_metal(CMLDispatchContext* ctx, CMLIR_t ir, Tensor** inputs, int nin,
                                  Tensor** outputs, int nout) {
#ifdef CML_HAS_MLIR
    LOG_DEBUG("Executing on Metal backend");

    // Validate inputs
    if (nin > 0 && !inputs) {
        LOG_WARNING("Metal dispatch: nin=%d but inputs is NULL, falling back", nin);
        return -1;
    }
    if (nout > 0 && !outputs) {
        LOG_WARNING("Metal dispatch: nout=%d but outputs is NULL, falling back", nout);
        return -1;
    }

    // Initialize Metal backend if needed
    if (!g_metal_backend) {
        g_metal_backend = cml_metal_backend_create();
        if (!g_metal_backend || cml_metal_backend_init(g_metal_backend) != 0) {
            LOG_WARNING("Failed to initialize Metal backend");
            cml_metal_backend_free(g_metal_backend);
            g_metal_backend = NULL;
            return -1;
        }
    }

    // Build MLIR module from IR if needed
    if (!ir->mlir_ctx) {
        ir->mlir_ctx = cml_mlir_init();
        if (!ir->mlir_ctx || !cml_mlir_build_from_ir(ir->mlir_ctx, ir)) {
            LOG_ERROR("Failed to build MLIR module for Metal");
            return -1;
        }
    }

    // Get MLIR module - use proper struct type
    CMLMLIRContext* mlir_ctx = (CMLMLIRContext*)ir->mlir_ctx;
    if (!mlir_ctx || mlirModuleIsNull(mlir_ctx->module)) {
        LOG_ERROR("No MLIR module available");
        return -1;
    }

    // Generate Metal Shading Language (MSL) code
    char* msl_code = cml_mlir_gen_metal(mlir_ctx->module.ptr);
    if (!msl_code) {
        LOG_WARNING("Failed to generate Metal shader code");
        return -1;
    }

    // Compile MSL to Metal kernel
    CMLMetalKernel* kernel = cml_metal_compile_source(g_metal_backend, msl_code, "main_kernel");
    free(msl_code);
    if (!kernel) {
        LOG_WARNING("Failed to compile Metal shader");
        return -1;
    }

    // Metal uses unified memory on Apple Silicon - no explicit copy needed
    // We can pass host pointers directly (they'll be visible to GPU)
    // For discrete GPUs, we'd need explicit buffer management

    // Build kernel arguments (host pointers for unified memory)
    void** args = (void**)calloc(nin + nout, sizeof(void*));
    int result  = -1;

    if (!args) {
        cml_metal_kernel_free(g_metal_backend, kernel);
        return -1;
    }

    for (int i = 0; i < nin; i++) {
        args[i] = inputs[i]->data;
    }
    for (int i = 0; i < nout; i++) {
        args[nin + i] = outputs[i]->data;
    }

    // Set kernel launch config (grid and threadgroup sizes)
    kernel->grid_dim[0]          = outputs[0]->shape[0];
    kernel->grid_dim[1]          = (outputs[0]->ndim > 1) ? outputs[0]->shape[1] : 1;
    kernel->grid_dim[2]          = 1;
    kernel->threads_per_group[0] = 1;
    kernel->threads_per_group[1] = 1;
    kernel->threads_per_group[2] = 1;

    // Launch kernel
    result = cml_metal_launch_kernel(g_metal_backend, kernel, args, nin + nout);
    free(args);

    if (result != 0) {
        LOG_ERROR("Metal kernel launch failed");
        cml_metal_kernel_free(g_metal_backend, kernel);
        return -1;
    }

    // Synchronize (wait for GPU to finish)
    cml_metal_synchronize(g_metal_backend);

    cml_metal_kernel_free(g_metal_backend, kernel);
    ctx->executions_total++;
    return 0;
#else
    (void)ctx;
    (void)ir;
    (void)inputs;
    (void)nin;
    (void)outputs;
    (void)nout;
    LOG_WARNING("Metal backend requires MLIR support");
    return -1;
#endif
}

// Execute on Vulkan
static int dispatch_execute_vulkan(CMLDispatchContext* ctx, CMLIR_t ir, Tensor** inputs, int nin,
                                   Tensor** outputs, int nout) {
#ifdef CML_HAS_MLIR
    LOG_DEBUG("Executing on Vulkan backend");

    // Validate inputs - if we have input count but no input array, skip Vulkan
    if (nin > 0 && !inputs) {
        LOG_WARNING("Vulkan dispatch: nin=%d but inputs is NULL, falling back", nin);
        return -1; // Fall back to another backend
    }
    if (nout > 0 && !outputs) {
        LOG_WARNING("Vulkan dispatch: nout=%d but outputs is NULL, falling back", nout);
        return -1; // Fall back to another backend
    }

    // Initialize Vulkan backend if needed
    if (!g_vulkan_backend) {
        g_vulkan_backend = cml_vulkan_backend_create();
        if (!g_vulkan_backend || cml_vulkan_backend_init(g_vulkan_backend) != 0) {
            LOG_WARNING("Failed to initialize Vulkan backend");
            cml_vulkan_backend_free(g_vulkan_backend);
            g_vulkan_backend = NULL;
            return -1;
        }
    }

    // Build MLIR module from IR if needed
    if (!ir->mlir_ctx) {
        ir->mlir_ctx = cml_mlir_init();
        if (!ir->mlir_ctx || !cml_mlir_build_from_ir(ir->mlir_ctx, ir)) {
            LOG_ERROR("Failed to build MLIR module for Vulkan");
            return -1;
        }
    }

    // Get MLIR module using the proper struct type
#ifdef CML_HAS_MLIR
    CMLMLIRContext* mlir_ctx = (CMLMLIRContext*)ir->mlir_ctx;
    if (!mlir_ctx || mlirModuleIsNull(mlir_ctx->module)) {
        LOG_ERROR("No MLIR module available");
        return -1;
    }

    // Generate SPIR-V binary code - pass the module's internal pointer
    size_t spirv_size    = 0;
    uint32_t* spirv_code = cml_mlir_gen_spirv(mlir_ctx->module.ptr, &spirv_size);
#else
    LOG_ERROR("MLIR support not compiled in");
    return -1;
    size_t spirv_size    = 0;
    uint32_t* spirv_code = NULL;
#endif
    if (!spirv_code || spirv_size == 0) {
        LOG_WARNING("Failed to generate SPIR-V code");
        return -1;
    }

    // Compile SPIR-V to Vulkan kernel
    CMLVulkanKernel* kernel =
        cml_vulkan_compile_spirv(g_vulkan_backend, spirv_code, spirv_size, "main");
    free(spirv_code);
    if (!kernel) {
        LOG_WARNING("Failed to compile Vulkan shader");
        return -1;
    }

    // Allocate Vulkan buffers and copy inputs
    VkBuffer* buffers        = (VkBuffer*)calloc(nin + nout, sizeof(VkBuffer));
    VkDeviceMemory* memories = (VkDeviceMemory*)calloc(nin + nout, sizeof(VkDeviceMemory));
    int result               = -1;

    if (!buffers || !memories) {
        cml_vulkan_kernel_free(g_vulkan_backend, kernel);
        free(buffers);
        free(memories);
        return -1;
    }

    // Create input buffers and upload data
    for (int i = 0; i < nin; i++) {
        size_t size = tensor_numel(inputs[i]->shape, inputs[i]->ndim) * sizeof(float);
        buffers[i]  = cml_vulkan_create_buffer(g_vulkan_backend, size, &memories[i]);
        if (!buffers[i]) {
            LOG_ERROR("Failed to create Vulkan buffer for input %d", i);
            goto vulkan_cleanup;
        }
        if (cml_vulkan_upload_data(g_vulkan_backend, memories[i], inputs[i]->data, size) != 0) {
            LOG_ERROR("Failed to upload data for input %d", i);
            goto vulkan_cleanup;
        }
    }

    // Create output buffers
    for (int i = 0; i < nout; i++) {
        size_t size      = tensor_numel(outputs[i]->shape, outputs[i]->ndim) * sizeof(float);
        buffers[nin + i] = cml_vulkan_create_buffer(g_vulkan_backend, size, &memories[nin + i]);
        if (!buffers[nin + i]) {
            LOG_ERROR("Failed to create Vulkan buffer for output %d", i);
            goto vulkan_cleanup;
        }
    }

    // Calculate dispatch dimensions
    uint32_t group_count_x = outputs[0]->shape[0];
    uint32_t group_count_y = (outputs[0]->ndim > 1) ? outputs[0]->shape[1] : 1;
    uint32_t group_count_z = 1;

    // Launch kernel
    result = cml_vulkan_launch_kernel(g_vulkan_backend, kernel, buffers, nin + nout, group_count_x,
                                      group_count_y, group_count_z);
    if (result != 0) {
        LOG_ERROR("Vulkan kernel launch failed");
        goto vulkan_cleanup;
    }

    // Synchronize
    cml_vulkan_synchronize(g_vulkan_backend);

    // Download output data
    for (int i = 0; i < nout; i++) {
        size_t size = tensor_numel(outputs[i]->shape, outputs[i]->ndim) * sizeof(float);
        if (cml_vulkan_download_data(g_vulkan_backend, memories[nin + i], outputs[i]->data, size) !=
            0) {
            LOG_ERROR("Failed to download data for output %d", i);
            goto vulkan_cleanup;
        }
    }

    result = 0; // Success

vulkan_cleanup:
    // Free all buffers
    for (int i = 0; i < nin + nout; i++) {
        if (buffers[i]) {
            cml_vulkan_destroy_buffer(g_vulkan_backend, buffers[i], memories[i]);
        }
    }
    free(buffers);
    free(memories);
    cml_vulkan_kernel_free(g_vulkan_backend, kernel);

    ctx->executions_total++;
    return result;
#else
    (void)ctx;
    (void)ir;
    (void)inputs;
    (void)nin;
    (void)outputs;
    (void)nout;
    LOG_WARNING("Vulkan backend requires MLIR support");
    return -1;
#endif
}

int cml_dispatch_execute_on(CMLDispatchContext* ctx, CMLBackendType backend, CMLIR_t ir,
                            Tensor** inputs, int nin, Tensor** outputs, int nout) {
    if (!ctx || !ir) {
        LOG_ERROR("Invalid dispatch arguments");
        return -1;
    }

    ctx->executions_total++;

    switch (backend) {
    case CML_BACKEND_CPU_FALLBACK:
        return dispatch_execute_cpu_fallback(ir);

    case CML_BACKEND_CPU_LLVM:
        return dispatch_execute_cpu_llvm(ctx, ir, inputs, nin, outputs, nout);

    case CML_BACKEND_CUDA:
        return dispatch_execute_cuda(ctx, ir, inputs, nin, outputs, nout);

    case CML_BACKEND_ROCM:
        return dispatch_execute_rocm(ctx, ir, inputs, nin, outputs, nout);

    case CML_BACKEND_METAL:
        return dispatch_execute_metal(ctx, ir, inputs, nin, outputs, nout);

    case CML_BACKEND_VULKAN:
        return dispatch_execute_vulkan(ctx, ir, inputs, nin, outputs, nout);

    default:
        LOG_ERROR("Unknown backend: %d", backend);
        return -1;
    }
}

// Map dispatch backend to kernel cache backend
static CMLKernelBackend dispatch_to_kernel_backend(CMLBackendType backend) {
    switch (backend) {
    case CML_BACKEND_CPU_FALLBACK:
        return CML_KERNEL_CPU_FALLBACK;
    case CML_BACKEND_CPU_LLVM:
        return CML_KERNEL_CPU_LLVM;
    case CML_BACKEND_CUDA:
        return CML_KERNEL_CUDA;
    case CML_BACKEND_ROCM:
        return CML_KERNEL_ROCM;
    case CML_BACKEND_METAL:
        return CML_KERNEL_METAL;
    case CML_BACKEND_VULKAN:
        return CML_KERNEL_VULKAN;
    default:
        return CML_KERNEL_CPU_FALLBACK;
    }
}

int cml_dispatch_execute(CMLDispatchContext* ctx, CMLIR_t ir, Tensor** inputs, int nin,
                         Tensor** outputs, int nout) {
    if (!ctx) {
        ctx = cml_dispatch_get_global();
    }
    if (!ctx || !ir) {
        LOG_ERROR("Invalid dispatch arguments");
        return -1;
    }

    // Check if BLAS can handle this operation (optimized matmul)
    // BLAS is often faster than JIT for simple matmul on CPU
    if (cml_blas_available() && can_use_blas(ir, inputs, nin)) {
        LOG_DEBUG("Using BLAS for optimized matmul execution");
        int blas_result = dispatch_execute_blas_matmul(ir, inputs, nin, outputs, nout);
        if (blas_result == 0) {
            ctx->executions_total++;
            return 0;
        }
        // Fall through to normal dispatch if BLAS fails
        LOG_DEBUG("BLAS execution failed, trying other backends");
    }

    // Check kernel cache first
    CMLKernelEntry* cached          = NULL;
    uint64_t cache_hash             = 0;
    CMLKernelBackend kernel_backend = dispatch_to_kernel_backend(ctx->active);

    if (ctx->cache) {
        cache_hash = cml_kernel_cache_compute_hash(ir, inputs, nin, kernel_backend);
        cached     = cml_kernel_cache_lookup((CMLKernelCache*)ctx->cache, cache_hash);
        if (cached && cached->compiled) {
            ctx->cache_hits++;
            LOG_DEBUG("Kernel cache hit, hash=0x%016llx", (unsigned long long)cache_hash);
            // Try executing the cached kernel
            int cache_result = execute_cached_kernel(cached, inputs, nin, outputs, nout);
            if (cache_result == 0) {
                ctx->executions_total++;
                return 0;
            }
            // If cached execution fails, fall through to normal execution
            LOG_DEBUG("Cached kernel execution failed, recompiling");
        } else {
            ctx->cache_misses++;
        }
    }

    // Try active backend first
    int result = cml_dispatch_execute_on(ctx, ctx->active, ir, inputs, nin, outputs, nout);
    if (result == 0) {
        // Cache the successful execution (if cache enabled and not already cached)
        if (ctx->cache && !cached) {
            // Note: For now we don't actually cache the compiled kernel
            // This would need proper integration with the MLIR execution engine
            // to extract and cache the compiled function pointer
            LOG_DEBUG("Would cache kernel hash=0x%016llx (not yet implemented)",
                      (unsigned long long)cache_hash);
        }
        return 0;
    }

    // Fallback through the chain
    for (int i = 0; i < ctx->fallback_count; i++) {
        CMLBackendType fallback = ctx->fallback_chain[i];

        // Skip already tried or unavailable backends
        if (fallback == ctx->active)
            continue;
        if (!cml_dispatch_backend_available(ctx, fallback))
            continue;

        LOG_INFO("Falling back to: %s", backend_names[fallback]);
        ctx->fallbacks_used++;

        result = cml_dispatch_execute_on(ctx, fallback, ir, inputs, nin, outputs, nout);
        if (result == 0) {
            return 0;
        }
    }

    LOG_ERROR("All backends failed");
    return -1;
}

CMLBackendType cml_dispatch_select_backend(CMLDispatchContext* ctx, CMLIR_t ir) {
    if (!ctx || !ir)
        return CML_BACKEND_CPU_FALLBACK;

    // Check tensor device affinity
    struct IRNode* node    = ir->head;
    bool has_cuda_tensors  = false;
    bool has_metal_tensors = false;
    bool has_rocm_tensors  = false;

    while (node) {
        if (node->output && node->output->device == DEVICE_CUDA) {
            has_cuda_tensors = true;
        } else if (node->output && node->output->device == DEVICE_METAL) {
            has_metal_tensors = true;
        } else if (node->output && node->output->device == DEVICE_ROCM) {
            has_rocm_tensors = true;
        }
        node = node->next;
    }

    // Prefer backend matching tensor device
    if (has_cuda_tensors && cml_dispatch_backend_available(ctx, CML_BACKEND_CUDA)) {
        return CML_BACKEND_CUDA;
    }
    if (has_metal_tensors && cml_dispatch_backend_available(ctx, CML_BACKEND_METAL)) {
        return CML_BACKEND_METAL;
    }
    if (has_rocm_tensors && cml_dispatch_backend_available(ctx, CML_BACKEND_ROCM)) {
        return CML_BACKEND_ROCM;
    }

    // Fall back to preferred or best
    if (cml_dispatch_backend_available(ctx, ctx->preferred)) {
        return ctx->preferred;
    }

    return cml_dispatch_get_best_backend(ctx);
}

// ============================================================================
// Cache Management
// ============================================================================

int cml_dispatch_enable_cache(CMLDispatchContext* ctx, size_t max_entries) {
    if (!ctx)
        return -1;

    // Free existing cache if any
    if (ctx->cache) {
        cml_kernel_cache_free((CMLKernelCache*)ctx->cache);
    }

    ctx->cache = (struct CMLKernelCache*)cml_kernel_cache_create(max_entries);
    if (!ctx->cache) {
        LOG_ERROR("Failed to create kernel cache");
        return -1;
    }

    LOG_INFO("Enabled kernel cache with max_entries=%zu", max_entries);
    return 0;
}

void cml_dispatch_disable_cache(CMLDispatchContext* ctx) {
    if (!ctx || !ctx->cache)
        return;

    cml_kernel_cache_free((CMLKernelCache*)ctx->cache);
    ctx->cache = NULL;
    LOG_INFO("Disabled kernel cache");
}

void cml_dispatch_clear_cache(CMLDispatchContext* ctx) {
    if (!ctx || !ctx->cache)
        return;

    kernel_cache_clear((CMLKernelCache*)ctx->cache);
    LOG_DEBUG("Cleared kernel cache");
}

void cml_dispatch_cache_stats(CMLDispatchContext* ctx, size_t* hits, size_t* misses, size_t* size) {
    if (!ctx)
        return;

    if (ctx->cache) {
        size_t count = 0;
        kernel_cache_stats((CMLKernelCache*)ctx->cache, hits, misses, &count, size);
    } else {
        if (hits)
            *hits = ctx->cache_hits;
        if (misses)
            *misses = ctx->cache_misses;
        if (size)
            *size = 0;
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

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
        case CML_BACKEND_STATUS_UNAVAILABLE:
            status = "Unavailable";
            break;
        case CML_BACKEND_STATUS_AVAILABLE:
            status = "Available";
            break;
        case CML_BACKEND_STATUS_INITIALIZED:
            status = "Initialized";
            break;
        case CML_BACKEND_STATUS_ERROR:
            status = "Error";
            break;
        default:
            status = "Unknown";
            break;
        }
        printf("  [%d] %-20s: %s", i, backend_names[i], status);
        if (ctx->backends[i].device_count > 0) {
            printf(" (%d devices)", ctx->backends[i].device_count);
        }
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

    // Synchronize all initialized backends
    if (ctx->backends[CML_BACKEND_CUDA].status == CML_BACKEND_STATUS_INITIALIZED &&
        g_cuda_backend) {
        cml_cuda_synchronize(g_cuda_backend);
    }
    if (ctx->backends[CML_BACKEND_ROCM].status == CML_BACKEND_STATUS_INITIALIZED &&
        g_rocm_backend) {
        cml_rocm_synchronize(g_rocm_backend);
    }
    if (ctx->backends[CML_BACKEND_METAL].status == CML_BACKEND_STATUS_INITIALIZED &&
        g_metal_backend) {
        cml_metal_synchronize(g_metal_backend);
    }
    if (ctx->backends[CML_BACKEND_VULKAN].status == CML_BACKEND_STATUS_INITIALIZED &&
        g_vulkan_backend) {
        cml_vulkan_synchronize(g_vulkan_backend);
    }
}

int cml_dispatch_set_from_env(CMLDispatchContext* ctx) {
    if (!ctx)
        return -1;

    const char* backend_env = getenv("CML_BACKEND");
    if (!backend_env)
        return 0;

    CMLBackendType backend = CML_BACKEND_CPU_FALLBACK;

    if (strcasecmp(backend_env, "cpu") == 0 || strcasecmp(backend_env, "llvm") == 0 ||
        strcasecmp(backend_env, "cpu_llvm") == 0) {
        backend = CML_BACKEND_CPU_LLVM;
    } else if (strcasecmp(backend_env, "fallback") == 0 ||
               strcasecmp(backend_env, "interpreter") == 0) {
        backend = CML_BACKEND_CPU_FALLBACK;
    } else if (strcasecmp(backend_env, "cuda") == 0 || strcasecmp(backend_env, "nvidia") == 0) {
        backend = CML_BACKEND_CUDA;
    } else if (strcasecmp(backend_env, "rocm") == 0 || strcasecmp(backend_env, "amd") == 0 ||
               strcasecmp(backend_env, "hip") == 0) {
        backend = CML_BACKEND_ROCM;
    } else if (strcasecmp(backend_env, "metal") == 0 || strcasecmp(backend_env, "apple") == 0) {
        backend = CML_BACKEND_METAL;
    } else if (strcasecmp(backend_env, "vulkan") == 0 || strcasecmp(backend_env, "vk") == 0) {
        backend = CML_BACKEND_VULKAN;
    } else {
        LOG_WARNING("Unknown CML_BACKEND value: %s", backend_env);
        return -1;
    }

    LOG_INFO("CML_BACKEND=%s -> %s", backend_env, backend_names[backend]);
    return cml_dispatch_set_preferred(ctx, backend);
}
