/*
 * Unified dispatch layer for multi-backend execution.
 * Handles backend detection, selection, fallback, and kernel cache integration.
 * Backends: CPU interpreter, LLVM JIT, CUDA (NVPTX), ROCm (AMDGPU).
 */

#ifndef CML_DISPATCH_H
#define CML_DISPATCH_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CMLGraph;
typedef struct CMLGraph* CMLGraph_t;
struct Tensor;
typedef struct Tensor Tensor;
struct CMLKernelCache;

typedef enum CMLBackendType {
    CML_BACKEND_CPU_FALLBACK = 0, // CPU interpreter (no JIT, always available)
    CML_BACKEND_CPU_LLVM,         // LLVM JIT (requires LLVM)
    CML_BACKEND_NV,               // NV userspace driver (direct ioctl, no libcuda)
    CML_BACKEND_CUDA,             // LLVM NVPTX -> cuLaunchKernel
    CML_BACKEND_AM,               // AM userspace driver (KFD ioctl, no libamdhip64)
    CML_BACKEND_ROCM,             // LLVM AMDGPU -> hipLaunchKernel
    CML_BACKEND_NIR,              // Mesa NIR -> SPIR-V -> Vulkan
    CML_BACKEND_METAL,            // Metal GPU (macOS)
    CML_BACKEND_VULKAN,           // Vulkan/SPIR-V compute
    CML_BACKEND_WEBGPU,           // WebGPU via wgpu-native
    CML_BACKEND_OPENCL,           // OpenCL compute
    CML_BACKEND_COUNT
} CMLBackendType;

typedef enum CMLBackendStatus {
    CML_BACKEND_STATUS_UNAVAILABLE = 0,
    CML_BACKEND_STATUS_AVAILABLE,
    CML_BACKEND_STATUS_INITIALIZED,
    CML_BACKEND_STATUS_ERROR
} CMLBackendStatus;

typedef struct CMLBackendInfo {
    CMLBackendType type;
    CMLBackendStatus status;
    const char* name;
    const char* description;
    int device_count;
    size_t total_memory;
    bool supports_async;
    bool supports_unified_mem;
} CMLBackendInfo;

typedef struct CMLDispatchContext {
    CMLBackendType preferred;
    CMLBackendType active;
    CMLBackendType fallback_chain[CML_BACKEND_COUNT];
    int fallback_count;

    CMLBackendInfo backends[CML_BACKEND_COUNT];

    struct CMLKernelCache* cache;

    size_t executions_total;
    size_t cache_hits;
    size_t cache_misses;
    size_t fallbacks_used;

    bool initialized;
    void* backend_contexts[CML_BACKEND_COUNT];
} CMLDispatchContext;

CMLDispatchContext* cml_dispatch_create(void);
int cml_dispatch_init(CMLDispatchContext* ctx);
void cml_dispatch_free(CMLDispatchContext* ctx);
CMLDispatchContext* cml_dispatch_get_global(void);

int cml_dispatch_detect_backends(CMLDispatchContext* ctx);
int cml_dispatch_set_preferred(CMLDispatchContext* ctx, CMLBackendType backend);
const CMLBackendInfo* cml_dispatch_get_backend_info(CMLDispatchContext* ctx,
                                                    CMLBackendType backend);
bool cml_dispatch_backend_available(CMLDispatchContext* ctx, CMLBackendType backend);
CMLBackendType cml_dispatch_get_best_backend(CMLDispatchContext* ctx);
const char* cml_dispatch_backend_name(CMLBackendType backend);

int cml_dispatch_execute(CMLDispatchContext* ctx, CMLGraph_t ir, Tensor** inputs, int num_inputs,
                         Tensor** outputs, int num_outputs);
int cml_dispatch_execute_on(CMLDispatchContext* ctx, CMLBackendType backend, CMLGraph_t ir,
                            Tensor** inputs, int num_inputs, Tensor** outputs, int num_outputs);
CMLBackendType cml_dispatch_select_backend(CMLDispatchContext* ctx, CMLGraph_t ir);

int cml_dispatch_enable_cache(CMLDispatchContext* ctx, size_t max_entries);
void cml_dispatch_disable_cache(CMLDispatchContext* ctx);
void cml_dispatch_clear_cache(CMLDispatchContext* ctx);
void cml_dispatch_cache_stats(CMLDispatchContext* ctx, size_t* hits, size_t* misses, size_t* size);

int cml_dispatch_execute_async(CMLDispatchContext* ctx, CMLGraph_t ir,
                               Tensor** inputs, int num_inputs,
                               Tensor** outputs, int num_outputs);

struct CMLCUDABackend;
struct CMLCUDABackend* cml_dispatch_get_cuda_backend(void);
struct CMLVulkanBackend;
struct CMLVulkanBackend* cml_dispatch_get_vulkan_backend(void);
struct CMLNVDriver;
struct CMLNVDriver* cml_dispatch_get_nv_driver(void);
struct CMLAMDriver;
struct CMLAMDriver* cml_dispatch_get_am_driver(void);

int cml_dispatch_execute_jit(CMLDispatchContext* ctx, CMLGraph_t ir,
                             Tensor** inputs, int num_inputs,
                             Tensor** outputs, int num_outputs);

void cml_dispatch_print_status(CMLDispatchContext* ctx);
void cml_dispatch_synchronize(CMLDispatchContext* ctx);
int cml_dispatch_set_from_env(CMLDispatchContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // CML_DISPATCH_H
