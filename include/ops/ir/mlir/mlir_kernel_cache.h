/**
 * @file mlir_kernel_cache.h
 * @brief In-memory kernel cache for compiled MLIR/JIT kernels
 *
 * This module provides an LRU cache for compiled kernels to avoid
 * recompilation of identical IR graphs. The cache uses FNV-1a hashing
 * of IR structure and tensor shapes for lookup.
 */

#ifndef CML_MLIR_KERNEL_CACHE_H
#define CML_MLIR_KERNEL_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct CMLIR;
typedef struct CMLIR* CMLIR_t;
struct Tensor;
typedef struct Tensor Tensor;

/**
 * @brief Backend type (mirrors mlir_dispatch.h)
 */
typedef enum CMLKernelBackend {
    CML_KERNEL_CPU_FALLBACK = 0,
    CML_KERNEL_CPU_LLVM,
    CML_KERNEL_CUDA,
    CML_KERNEL_ROCM,
    CML_KERNEL_METAL,
    CML_KERNEL_VULKAN,
    CML_KERNEL_BACKEND_COUNT
} CMLKernelBackend;

/**
 * @brief Kernel entry in the cache
 */
typedef struct CMLKernelEntry {
    uint64_t hash;            // FNV-1a hash of IR + shapes
    CMLKernelBackend backend; // Backend this was compiled for
    void* compiled;           // Compiled kernel (backend-specific)
    uint64_t last_used;       // Timestamp for LRU eviction
    size_t memory_size;       // Estimated memory usage of compiled kernel

    // Metadata for debugging
    int num_ops;     // Number of operations in kernel
    int num_inputs;  // Number of input tensors
    int num_outputs; // Number of output tensors

    // Linked list for hash bucket
    struct CMLKernelEntry* next;
} CMLKernelEntry;

/**
 * @brief Kernel cache structure
 */
typedef struct CMLKernelCache {
    // Hash table
    CMLKernelEntry** buckets; // Hash table buckets
    size_t num_buckets;       // Number of buckets (power of 2)

    // Statistics
    size_t count;        // Current number of entries
    size_t max_entries;  // Maximum entries (0 = unlimited)
    size_t total_memory; // Total memory used by cached kernels
    size_t max_memory;   // Maximum memory limit (0 = unlimited)

    // Cache statistics
    size_t hits;      // Cache hit count
    size_t misses;    // Cache miss count
    size_t evictions; // Number of LRU evictions

    // Thread safety
    pthread_mutex_t lock;
    bool lock_initialized;

    // Global timestamp counter for LRU
    uint64_t timestamp;
} CMLKernelCache;

// ============================================================================
// Cache Lifecycle
// ============================================================================

/**
 * @brief Create a new kernel cache
 * @param max_entries Maximum number of cached kernels (0 = unlimited)
 * @return New kernel cache or NULL on failure
 */
CMLKernelCache* cml_kernel_cache_create(size_t max_entries);

/**
 * @brief Create cache with memory limit
 * @param max_entries Maximum entries
 * @param max_memory Maximum memory in bytes
 * @return New kernel cache or NULL on failure
 */
CMLKernelCache* cml_kernel_cache_create_with_limits(size_t max_entries, size_t max_memory);

/**
 * @brief Free kernel cache and all cached entries
 * @param cache Cache to free
 */
void cml_kernel_cache_free(CMLKernelCache* cache);

/**
 * @brief Clear all cached entries (internal, takes cache pointer)
 * @param cache Cache to clear
 */
void kernel_cache_clear(CMLKernelCache* cache);

// ============================================================================
// Cache Operations
// ============================================================================

/**
 * @brief Compute hash for IR + shapes
 *
 * Uses FNV-1a hashing algorithm to create a unique hash based on:
 * - IR operation types and order
 * - Input tensor shapes and dtypes
 * - Output tensor shapes
 *
 * @param ir IR graph to hash
 * @param inputs Input tensors
 * @param num_inputs Number of inputs
 * @param backend Target backend
 * @return 64-bit hash value
 */
uint64_t cml_kernel_cache_compute_hash(CMLIR_t ir, Tensor** inputs, int num_inputs,
                                       CMLKernelBackend backend);

/**
 * @brief Look up a cached kernel
 * @param cache Kernel cache
 * @param hash Pre-computed hash
 * @return Cached kernel entry or NULL if not found
 */
CMLKernelEntry* cml_kernel_cache_lookup(CMLKernelCache* cache, uint64_t hash);

/**
 * @brief Look up by IR and inputs (computes hash internally)
 * @param cache Kernel cache
 * @param ir IR graph
 * @param inputs Input tensors
 * @param num_inputs Number of inputs
 * @param backend Target backend
 * @return Cached kernel entry or NULL if not found
 */
CMLKernelEntry* cml_kernel_cache_lookup_ir(CMLKernelCache* cache, CMLIR_t ir, Tensor** inputs,
                                           int num_inputs, CMLKernelBackend backend);

/**
 * @brief Insert a compiled kernel into the cache
 * @param cache Kernel cache
 * @param hash Hash of the IR + shapes
 * @param backend Backend the kernel was compiled for
 * @param compiled Compiled kernel (ownership transferred to cache)
 * @param memory_size Estimated memory usage
 * @return 0 on success, -1 on failure
 */
int cml_kernel_cache_insert(CMLKernelCache* cache, uint64_t hash, CMLKernelBackend backend,
                            void* compiled, size_t memory_size);

/**
 * @brief Insert with full metadata
 * @param cache Kernel cache
 * @param ir IR that was compiled
 * @param inputs Input tensors
 * @param num_inputs Number of inputs
 * @param backend Backend
 * @param compiled Compiled kernel
 * @param memory_size Memory usage
 * @return 0 on success, -1 on failure
 */
int cml_kernel_cache_insert_ir(CMLKernelCache* cache, CMLIR_t ir, Tensor** inputs, int num_inputs,
                               CMLKernelBackend backend, void* compiled, size_t memory_size);

/**
 * @brief Remove a specific entry from the cache
 * @param cache Kernel cache
 * @param hash Hash of entry to remove
 * @return 0 on success, -1 if not found
 */
int cml_kernel_cache_remove(CMLKernelCache* cache, uint64_t hash);

// ============================================================================
// LRU Eviction
// ============================================================================

/**
 * @brief Evict the least recently used entry
 * @param cache Kernel cache
 * @return 0 on success, -1 if cache empty
 */
int cml_kernel_cache_evict_lru(CMLKernelCache* cache);

/**
 * @brief Evict entries until count is below max_entries
 * @param cache Kernel cache
 * @return Number of entries evicted
 */
int cml_kernel_cache_enforce_limits(CMLKernelCache* cache);

// ============================================================================
// Statistics (internal functions - use cml_ prefix versions for public API)
// ============================================================================

/**
 * @brief Get cache statistics (internal, takes cache pointer)
 * @param cache Kernel cache
 * @param hits Output: number of cache hits
 * @param misses Output: number of cache misses
 * @param count Output: current entry count
 * @param memory Output: total memory used
 */
void kernel_cache_stats(CMLKernelCache* cache, size_t* hits, size_t* misses, size_t* count,
                        size_t* memory);

/**
 * @brief Get cache hit rate (internal, takes cache pointer)
 * @param cache Kernel cache
 * @return Hit rate (0.0 to 1.0)
 */
double kernel_cache_hit_rate(CMLKernelCache* cache);

/**
 * @brief Print cache statistics (internal, takes cache pointer)
 * @param cache Kernel cache
 */
void kernel_cache_print_stats(CMLKernelCache* cache);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Set the kernel free function for a backend
 *
 * When a cached kernel is evicted, this function is called to free
 * backend-specific resources.
 *
 * @param backend Backend type
 * @param free_fn Function to free compiled kernels
 */
typedef void (*CMLKernelFreeFn)(void* compiled);
void cml_kernel_cache_set_free_fn(CMLKernelBackend backend, CMLKernelFreeFn free_fn);

/**
 * @brief Get the default cache (global singleton)
 * @return Global kernel cache
 */
CMLKernelCache* cml_kernel_cache_get_default(void);

#ifdef __cplusplus
}
#endif

#endif // CML_MLIR_KERNEL_CACHE_H
