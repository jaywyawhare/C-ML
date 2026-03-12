/**
 * @file kernel_cache.h
 * @brief In-memory kernel cache for compiled JIT kernels
 *
 * This module provides an LRU cache for compiled kernels to avoid
 * recompilation of identical IR graphs. The cache uses FNV-1a hashing
 * of IR structure and tensor shapes for lookup.
 */

#ifndef CML_KERNEL_CACHE_H
#define CML_KERNEL_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct CMLGraph;
typedef struct CMLGraph* CMLGraph_t;
struct Tensor;
typedef struct Tensor Tensor;

/**
 * @brief Backend type for cached kernels
 */
typedef enum CMLKernelBackend {
    CML_KERNEL_CPU_FALLBACK = 0,
    CML_KERNEL_CPU_LLVM,
    CML_KERNEL_CUDA,
    CML_KERNEL_ROCM,
    CML_KERNEL_METAL,
    CML_KERNEL_WEBGPU,
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

// Cache Lifecycle

CMLKernelCache* cml_kernel_cache_create(size_t max_entries);
CMLKernelCache* cml_kernel_cache_create_with_limits(size_t max_entries, size_t max_memory);
void cml_kernel_cache_free(CMLKernelCache* cache);
void kernel_cache_clear(CMLKernelCache* cache);

// Cache Operations

uint64_t cml_kernel_cache_compute_hash(CMLGraph_t ir, Tensor** inputs, int num_inputs,
                                       CMLKernelBackend backend);

CMLKernelEntry* cml_kernel_cache_lookup(CMLKernelCache* cache, uint64_t hash);
CMLKernelEntry* cml_kernel_cache_lookup_ir(CMLKernelCache* cache, CMLGraph_t ir, Tensor** inputs,
                                           int num_inputs, CMLKernelBackend backend);

int cml_kernel_cache_insert(CMLKernelCache* cache, uint64_t hash, CMLKernelBackend backend,
                            void* compiled, size_t memory_size);
int cml_kernel_cache_insert_ir(CMLKernelCache* cache, CMLGraph_t ir, Tensor** inputs, int num_inputs,
                               CMLKernelBackend backend, void* compiled, size_t memory_size);

int cml_kernel_cache_remove(CMLKernelCache* cache, uint64_t hash);

// LRU Eviction

int cml_kernel_cache_evict_lru(CMLKernelCache* cache);
int cml_kernel_cache_enforce_limits(CMLKernelCache* cache);

// Statistics

void kernel_cache_stats(CMLKernelCache* cache, size_t* hits, size_t* misses, size_t* count,
                        size_t* memory);
double kernel_cache_hit_rate(CMLKernelCache* cache);
void kernel_cache_print_stats(CMLKernelCache* cache);

// Utility Functions

typedef void (*CMLKernelFreeFn)(void* compiled);
void cml_kernel_cache_set_free_fn(CMLKernelBackend backend, CMLKernelFreeFn free_fn);
CMLKernelCache* cml_kernel_cache_get_default(void);

#ifdef __cplusplus
}
#endif

#endif // CML_KERNEL_CACHE_H
