/**
 * @file mlir_dispatch.h
 * @brief Unified dispatch layer for multi-backend MLIR execution
 *
 * This module provides a single entry point for executing IR on any available
 * backend (CPU/LLVM, CUDA, ROCm, Metal, Vulkan). It handles:
 * - Backend detection and selection
 * - Automatic fallback when preferred backend unavailable
 * - Kernel cache integration
 * - Unified memory management
 */

#ifndef CML_MLIR_DISPATCH_H
#define CML_MLIR_DISPATCH_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct CMLIR;
typedef struct CMLIR* CMLIR_t;
struct Tensor;
typedef struct Tensor Tensor;
struct CMLKernelCache;

/**
 * @brief Execution backend types
 */
typedef enum CMLBackendType {
    CML_BACKEND_CPU_FALLBACK = 0, // CPU interpreter (no JIT, always available)
    CML_BACKEND_CPU_LLVM,         // MLIR -> LLVM JIT (requires MLIR)
    CML_BACKEND_CUDA,             // MLIR -> NVPTX -> cuLaunchKernel
    CML_BACKEND_ROCM,             // MLIR -> AMDGCN -> hipLaunchKernel
    CML_BACKEND_METAL,            // MLIR -> MSL -> MTLDispatch
    CML_BACKEND_VULKAN,           // MLIR -> SPIR-V -> vkCmdDispatch
    CML_BACKEND_COUNT             // Number of backends
} CMLBackendType;

/**
 * @brief Backend status
 */
typedef enum CMLBackendStatus {
    CML_BACKEND_STATUS_UNAVAILABLE = 0, // Backend not available on system
    CML_BACKEND_STATUS_AVAILABLE,       // Available but not initialized
    CML_BACKEND_STATUS_INITIALIZED,     // Initialized and ready
    CML_BACKEND_STATUS_ERROR            // Initialization failed
} CMLBackendStatus;

/**
 * @brief Backend information
 */
typedef struct CMLBackendInfo {
    CMLBackendType type;
    CMLBackendStatus status;
    const char* name;
    const char* description;
    int device_count;          // Number of devices (GPUs) available
    size_t total_memory;       // Total memory across all devices
    bool supports_async;       // Supports async execution
    bool supports_unified_mem; // Supports unified/managed memory
} CMLBackendInfo;

/**
 * @brief Dispatch context for managing backend execution
 */
typedef struct CMLDispatchContext {
    // Backend selection
    CMLBackendType preferred;                         // User-preferred backend
    CMLBackendType active;                            // Currently active backend
    CMLBackendType fallback_chain[CML_BACKEND_COUNT]; // Fallback order
    int fallback_count;

    // Backend status
    CMLBackendInfo backends[CML_BACKEND_COUNT];

    // Kernel cache (optional)
    struct CMLKernelCache* cache;

    // Statistics
    size_t executions_total;
    size_t cache_hits;
    size_t cache_misses;
    size_t fallbacks_used;

    // Internal state
    bool initialized;
    void* backend_contexts[CML_BACKEND_COUNT]; // Per-backend context pointers
} CMLDispatchContext;

// ============================================================================
// Initialization and Cleanup
// ============================================================================

/**
 * @brief Create a new dispatch context
 * @return New dispatch context or NULL on failure
 */
CMLDispatchContext* cml_dispatch_create(void);

/**
 * @brief Initialize the dispatch context with default settings
 * @param ctx Dispatch context
 * @return 0 on success, -1 on failure
 */
int cml_dispatch_init(CMLDispatchContext* ctx);

/**
 * @brief Free the dispatch context
 * @param ctx Dispatch context to free
 */
void cml_dispatch_free(CMLDispatchContext* ctx);

/**
 * @brief Get the global dispatch context (singleton)
 * @return Global dispatch context
 */
CMLDispatchContext* cml_dispatch_get_global(void);

// ============================================================================
// Backend Management
// ============================================================================

/**
 * @brief Detect available backends on the system
 * @param ctx Dispatch context
 * @return Number of available backends
 */
int cml_dispatch_detect_backends(CMLDispatchContext* ctx);

/**
 * @brief Set the preferred execution backend
 * @param ctx Dispatch context
 * @param backend Preferred backend type
 * @return 0 on success, -1 if backend unavailable
 */
int cml_dispatch_set_preferred(CMLDispatchContext* ctx, CMLBackendType backend);

/**
 * @brief Get backend information
 * @param ctx Dispatch context
 * @param backend Backend type
 * @return Backend info or NULL if invalid
 */
const CMLBackendInfo* cml_dispatch_get_backend_info(CMLDispatchContext* ctx,
                                                    CMLBackendType backend);

/**
 * @brief Check if a backend is available
 * @param ctx Dispatch context
 * @param backend Backend type
 * @return true if available, false otherwise
 */
bool cml_dispatch_backend_available(CMLDispatchContext* ctx, CMLBackendType backend);

/**
 * @brief Get the best available backend (auto-selection)
 * @param ctx Dispatch context
 * @return Best available backend type
 */
CMLBackendType cml_dispatch_get_best_backend(CMLDispatchContext* ctx);

/**
 * @brief Get backend name as string
 * @param backend Backend type
 * @return Backend name string
 */
const char* cml_dispatch_backend_name(CMLBackendType backend);

// ============================================================================
// Execution
// ============================================================================

/**
 * @brief Execute IR on the selected backend
 *
 * This is the main entry point for all IR execution. It:
 * 1. Checks the kernel cache for a compiled version
 * 2. If not cached, compiles the IR for the selected backend
 * 3. Executes the compiled kernel
 * 4. Falls back to next backend in chain if execution fails
 *
 * @param ctx Dispatch context
 * @param ir IR graph to execute
 * @param inputs Input tensors
 * @param num_inputs Number of input tensors
 * @param outputs Output tensors
 * @param num_outputs Number of output tensors
 * @return 0 on success, -1 on failure
 */
int cml_dispatch_execute(CMLDispatchContext* ctx, CMLIR_t ir, Tensor** inputs, int num_inputs,
                         Tensor** outputs, int num_outputs);

/**
 * @brief Execute IR on a specific backend (no fallback)
 * @param ctx Dispatch context
 * @param backend Target backend
 * @param ir IR graph to execute
 * @param inputs Input tensors
 * @param num_inputs Number of input tensors
 * @param outputs Output tensors
 * @param num_outputs Number of output tensors
 * @return 0 on success, -1 on failure
 */
int cml_dispatch_execute_on(CMLDispatchContext* ctx, CMLBackendType backend, CMLIR_t ir,
                            Tensor** inputs, int num_inputs, Tensor** outputs, int num_outputs);

/**
 * @brief Select the best backend for given IR
 *
 * Considers:
 * - Tensor device affinity
 * - Operation types in IR
 * - Backend availability
 * - Memory requirements
 *
 * @param ctx Dispatch context
 * @param ir IR graph to analyze
 * @return Recommended backend
 */
CMLBackendType cml_dispatch_select_backend(CMLDispatchContext* ctx, CMLIR_t ir);

// ============================================================================
// Cache Management
// ============================================================================

/**
 * @brief Enable kernel caching
 * @param ctx Dispatch context
 * @param max_entries Maximum cache entries (0 = unlimited)
 * @return 0 on success, -1 on failure
 */
int cml_dispatch_enable_cache(CMLDispatchContext* ctx, size_t max_entries);

/**
 * @brief Disable kernel caching
 * @param ctx Dispatch context
 */
void cml_dispatch_disable_cache(CMLDispatchContext* ctx);

/**
 * @brief Clear the kernel cache
 * @param ctx Dispatch context
 */
void cml_dispatch_clear_cache(CMLDispatchContext* ctx);

/**
 * @brief Get cache statistics
 * @param ctx Dispatch context
 * @param hits Output: number of cache hits
 * @param misses Output: number of cache misses
 * @param size Output: current cache size
 */
void cml_dispatch_cache_stats(CMLDispatchContext* ctx, size_t* hits, size_t* misses, size_t* size);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Print dispatch context status
 * @param ctx Dispatch context
 */
void cml_dispatch_print_status(CMLDispatchContext* ctx);

/**
 * @brief Synchronize all backends
 * @param ctx Dispatch context
 */
void cml_dispatch_synchronize(CMLDispatchContext* ctx);

/**
 * @brief Set backend from environment variable CML_BACKEND
 * @param ctx Dispatch context
 * @return 0 on success, -1 if env var invalid
 */
int cml_dispatch_set_from_env(CMLDispatchContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // CML_MLIR_DISPATCH_H
