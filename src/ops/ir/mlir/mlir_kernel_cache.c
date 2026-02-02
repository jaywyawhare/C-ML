/**
 * @file mlir_kernel_cache.c
 * @brief In-memory LRU kernel cache implementation
 */

#include "ops/ir/mlir/mlir_kernel_cache.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "tensor/tensor.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Constants
// ============================================================================

#define DEFAULT_NUM_BUCKETS 256
#define FNV_OFFSET_BASIS 0xcbf29ce484222325ULL
#define FNV_PRIME 0x100000001b3ULL

// ============================================================================
// Per-backend kernel free functions
// ============================================================================

static CMLKernelFreeFn g_kernel_free_fns[CML_KERNEL_BACKEND_COUNT] = {0};

// Global default cache
static CMLKernelCache* g_default_cache = NULL;

// ============================================================================
// FNV-1a Hash Implementation
// ============================================================================

static uint64_t fnv1a_hash_init(void) { return FNV_OFFSET_BASIS; }

static uint64_t fnv1a_hash_byte(uint64_t hash, uint8_t byte) {
    hash ^= byte;
    hash *= FNV_PRIME;
    return hash;
}

static uint64_t fnv1a_hash_bytes(uint64_t hash, const void* data, size_t len) {
    const uint8_t* bytes = (const uint8_t*)data;
    for (size_t i = 0; i < len; i++) {
        hash = fnv1a_hash_byte(hash, bytes[i]);
    }
    return hash;
}

static uint64_t fnv1a_hash_int(uint64_t hash, int value) {
    return fnv1a_hash_bytes(hash, &value, sizeof(value));
}

static uint64_t fnv1a_hash_size(uint64_t hash, size_t value) {
    return fnv1a_hash_bytes(hash, &value, sizeof(value));
}

// ============================================================================
// Cache Creation and Destruction
// ============================================================================

CMLKernelCache* cml_kernel_cache_create(size_t max_entries) {
    return cml_kernel_cache_create_with_limits(max_entries, 0);
}

CMLKernelCache* cml_kernel_cache_create_with_limits(size_t max_entries, size_t max_memory) {
    CMLKernelCache* cache = (CMLKernelCache*)calloc(1, sizeof(CMLKernelCache));
    if (!cache) {
        LOG_ERROR("Failed to allocate kernel cache");
        return NULL;
    }

    cache->num_buckets = DEFAULT_NUM_BUCKETS;
    cache->buckets     = (CMLKernelEntry**)calloc(cache->num_buckets, sizeof(CMLKernelEntry*));
    if (!cache->buckets) {
        LOG_ERROR("Failed to allocate kernel cache buckets");
        free(cache);
        return NULL;
    }

    cache->max_entries  = max_entries;
    cache->max_memory   = max_memory;
    cache->count        = 0;
    cache->total_memory = 0;
    cache->hits         = 0;
    cache->misses       = 0;
    cache->evictions    = 0;
    cache->timestamp    = 0;

    if (pthread_mutex_init(&cache->lock, NULL) != 0) {
        LOG_ERROR("Failed to initialize kernel cache mutex");
        free(cache->buckets);
        free(cache);
        return NULL;
    }
    cache->lock_initialized = true;

    LOG_DEBUG("Created kernel cache with max_entries=%zu, max_memory=%zu", max_entries, max_memory);

    return cache;
}

static void free_entry(CMLKernelEntry* entry) {
    if (!entry)
        return;

    // Call backend-specific free function
    if (entry->compiled && g_kernel_free_fns[entry->backend]) {
        g_kernel_free_fns[entry->backend](entry->compiled);
    }

    free(entry);
}

void cml_kernel_cache_free(CMLKernelCache* cache) {
    if (!cache)
        return;

    // Clear all entries
    kernel_cache_clear(cache);

    // Free mutex
    if (cache->lock_initialized) {
        pthread_mutex_destroy(&cache->lock);
    }

    // Free buckets and cache
    free(cache->buckets);
    free(cache);
}

void kernel_cache_clear(CMLKernelCache* cache) {
    if (!cache)
        return;

    pthread_mutex_lock(&cache->lock);

    for (size_t i = 0; i < cache->num_buckets; i++) {
        CMLKernelEntry* entry = cache->buckets[i];
        while (entry) {
            CMLKernelEntry* next = entry->next;
            free_entry(entry);
            entry = next;
        }
        cache->buckets[i] = NULL;
    }

    cache->count        = 0;
    cache->total_memory = 0;
    // Keep statistics

    pthread_mutex_unlock(&cache->lock);

    LOG_DEBUG("Cleared kernel cache");
}

// ============================================================================
// Hash Computation
// ============================================================================

uint64_t cml_kernel_cache_compute_hash(CMLIR_t ir, Tensor** inputs, int num_inputs,
                                       CMLKernelBackend backend) {
    uint64_t hash = fnv1a_hash_init();

    // Hash backend type
    hash = fnv1a_hash_int(hash, (int)backend);

    // Hash number of inputs
    hash = fnv1a_hash_int(hash, num_inputs);

    // Hash input tensor shapes and dtypes
    for (int i = 0; i < num_inputs; i++) {
        if (inputs && inputs[i]) {
            Tensor* t = inputs[i];
            hash      = fnv1a_hash_int(hash, t->ndim);
            for (int d = 0; d < t->ndim; d++) {
                hash = fnv1a_hash_int(hash, t->shape[d]);
            }
            hash = fnv1a_hash_int(hash, (int)t->dtype);
        }
    }

    // Hash IR structure
    if (ir) {
        // Hash number of nodes
        hash = fnv1a_hash_size(hash, ir->node_count);

        // Hash each node's operation type
        struct IRNode* node = ir->head;
        while (node) {
            hash = fnv1a_hash_int(hash, (int)node->type);
            hash = fnv1a_hash_int(hash, node->num_inputs);

            // Hash output shape if available
            if (node->output) {
                hash = fnv1a_hash_int(hash, node->output->ndim);
                for (int d = 0; d < node->output->ndim; d++) {
                    hash = fnv1a_hash_int(hash, node->output->shape[d]);
                }
            }

            node = node->next;
        }
    }

    return hash;
}

// ============================================================================
// Cache Lookup
// ============================================================================

CMLKernelEntry* cml_kernel_cache_lookup(CMLKernelCache* cache, uint64_t hash) {
    if (!cache)
        return NULL;

    pthread_mutex_lock(&cache->lock);

    size_t bucket_idx     = hash % cache->num_buckets;
    CMLKernelEntry* entry = cache->buckets[bucket_idx];

    while (entry) {
        if (entry->hash == hash) {
            // Cache hit - update timestamp
            entry->last_used = ++cache->timestamp;
            cache->hits++;
            pthread_mutex_unlock(&cache->lock);
            LOG_DEBUG("Kernel cache hit: hash=0x%016llx", (unsigned long long)hash);
            return entry;
        }
        entry = entry->next;
    }

    // Cache miss
    cache->misses++;
    pthread_mutex_unlock(&cache->lock);
    return NULL;
}

CMLKernelEntry* cml_kernel_cache_lookup_ir(CMLKernelCache* cache, CMLIR_t ir, Tensor** inputs,
                                           int num_inputs, CMLKernelBackend backend) {
    uint64_t hash = cml_kernel_cache_compute_hash(ir, inputs, num_inputs, backend);
    return cml_kernel_cache_lookup(cache, hash);
}

// ============================================================================
// Cache Insert
// ============================================================================

int cml_kernel_cache_insert(CMLKernelCache* cache, uint64_t hash, CMLKernelBackend backend,
                            void* compiled, size_t memory_size) {
    if (!cache || !compiled)
        return -1;

    pthread_mutex_lock(&cache->lock);

    // Check if we need to evict entries
    if (cache->max_entries > 0 && cache->count >= cache->max_entries) {
        pthread_mutex_unlock(&cache->lock);
        cml_kernel_cache_evict_lru(cache);
        pthread_mutex_lock(&cache->lock);
    }

    if (cache->max_memory > 0 && cache->total_memory + memory_size > cache->max_memory) {
        // Need to evict until we have space
        pthread_mutex_unlock(&cache->lock);
        while (cache->total_memory + memory_size > cache->max_memory && cache->count > 0) {
            cml_kernel_cache_evict_lru(cache);
        }
        pthread_mutex_lock(&cache->lock);
    }

    // Create new entry
    CMLKernelEntry* entry = (CMLKernelEntry*)calloc(1, sizeof(CMLKernelEntry));
    if (!entry) {
        pthread_mutex_unlock(&cache->lock);
        LOG_ERROR("Failed to allocate kernel cache entry");
        return -1;
    }

    entry->hash        = hash;
    entry->backend     = backend;
    entry->compiled    = compiled;
    entry->memory_size = memory_size;
    entry->last_used   = ++cache->timestamp;

    // Insert into hash table
    size_t bucket_idx          = hash % cache->num_buckets;
    entry->next                = cache->buckets[bucket_idx];
    cache->buckets[bucket_idx] = entry;

    cache->count++;
    cache->total_memory += memory_size;

    pthread_mutex_unlock(&cache->lock);

    LOG_DEBUG("Cached kernel: hash=0x%016llx, backend=%d, size=%zu bytes", (unsigned long long)hash,
              backend, memory_size);

    return 0;
}

int cml_kernel_cache_insert_ir(CMLKernelCache* cache, CMLIR_t ir, Tensor** inputs, int num_inputs,
                               CMLKernelBackend backend, void* compiled, size_t memory_size) {
    uint64_t hash = cml_kernel_cache_compute_hash(ir, inputs, num_inputs, backend);

    int result = cml_kernel_cache_insert(cache, hash, backend, compiled, memory_size);

    // Update metadata if successful
    if (result == 0 && ir) {
        pthread_mutex_lock(&cache->lock);

        size_t bucket_idx     = hash % cache->num_buckets;
        CMLKernelEntry* entry = cache->buckets[bucket_idx];
        while (entry) {
            if (entry->hash == hash) {
                entry->num_ops    = (int)ir->node_count;
                entry->num_inputs = num_inputs;
                // Count outputs from IR
                struct IRNode* node = ir->head;
                while (node && node->next)
                    node = node->next;
                entry->num_outputs = node ? 1 : 0;
                break;
            }
            entry = entry->next;
        }

        pthread_mutex_unlock(&cache->lock);
    }

    return result;
}

// ============================================================================
// Cache Removal
// ============================================================================

int cml_kernel_cache_remove(CMLKernelCache* cache, uint64_t hash) {
    if (!cache)
        return -1;

    pthread_mutex_lock(&cache->lock);

    size_t bucket_idx   = hash % cache->num_buckets;
    CMLKernelEntry** pp = &cache->buckets[bucket_idx];

    while (*pp) {
        if ((*pp)->hash == hash) {
            CMLKernelEntry* entry = *pp;
            *pp                   = entry->next;

            cache->count--;
            cache->total_memory -= entry->memory_size;

            free_entry(entry);

            pthread_mutex_unlock(&cache->lock);
            LOG_DEBUG("Removed kernel from cache: hash=0x%016llx", (unsigned long long)hash);
            return 0;
        }
        pp = &(*pp)->next;
    }

    pthread_mutex_unlock(&cache->lock);
    return -1; // Not found
}

// ============================================================================
// LRU Eviction
// ============================================================================

int cml_kernel_cache_evict_lru(CMLKernelCache* cache) {
    if (!cache || cache->count == 0)
        return -1;

    pthread_mutex_lock(&cache->lock);

    // Find LRU entry
    CMLKernelEntry* lru_entry = NULL;
    size_t lru_bucket         = 0;
    uint64_t oldest_time      = UINT64_MAX;

    for (size_t i = 0; i < cache->num_buckets; i++) {
        CMLKernelEntry* entry = cache->buckets[i];
        while (entry) {
            if (entry->last_used < oldest_time) {
                oldest_time = entry->last_used;
                lru_entry   = entry;
                lru_bucket  = i;
            }
            entry = entry->next;
        }
    }

    if (!lru_entry) {
        pthread_mutex_unlock(&cache->lock);
        return -1;
    }

    // Remove LRU entry from bucket
    CMLKernelEntry** pp = &cache->buckets[lru_bucket];
    while (*pp != lru_entry) {
        pp = &(*pp)->next;
    }
    *pp = lru_entry->next;

    cache->count--;
    cache->total_memory -= lru_entry->memory_size;
    cache->evictions++;

    uint64_t evicted_hash = lru_entry->hash;
    free_entry(lru_entry);

    pthread_mutex_unlock(&cache->lock);

    LOG_DEBUG("Evicted LRU kernel: hash=0x%016llx", (unsigned long long)evicted_hash);

    return 0;
}

int cml_kernel_cache_enforce_limits(CMLKernelCache* cache) {
    if (!cache)
        return 0;

    int evicted = 0;

    // Evict until under entry limit
    while (cache->max_entries > 0 && cache->count > cache->max_entries) {
        if (cml_kernel_cache_evict_lru(cache) != 0)
            break;
        evicted++;
    }

    // Evict until under memory limit
    while (cache->max_memory > 0 && cache->total_memory > cache->max_memory) {
        if (cml_kernel_cache_evict_lru(cache) != 0)
            break;
        evicted++;
    }

    return evicted;
}

// ============================================================================
// Statistics
// ============================================================================

void kernel_cache_stats(CMLKernelCache* cache, size_t* hits, size_t* misses, size_t* count,
                        size_t* memory) {
    if (!cache) {
        if (hits)
            *hits = 0;
        if (misses)
            *misses = 0;
        if (count)
            *count = 0;
        if (memory)
            *memory = 0;
        return;
    }

    pthread_mutex_lock(&cache->lock);

    if (hits)
        *hits = cache->hits;
    if (misses)
        *misses = cache->misses;
    if (count)
        *count = cache->count;
    if (memory)
        *memory = cache->total_memory;

    pthread_mutex_unlock(&cache->lock);
}

double kernel_cache_hit_rate(CMLKernelCache* cache) {
    if (!cache)
        return 0.0;

    pthread_mutex_lock(&cache->lock);

    size_t total = cache->hits + cache->misses;
    double rate  = (total > 0) ? (double)cache->hits / (double)total : 0.0;

    pthread_mutex_unlock(&cache->lock);

    return rate;
}

void kernel_cache_print_stats(CMLKernelCache* cache) {
    if (!cache) {
        printf("Kernel Cache: (null)\n");
        return;
    }

    pthread_mutex_lock(&cache->lock);

    size_t total    = cache->hits + cache->misses;
    double hit_rate = (total > 0) ? 100.0 * cache->hits / total : 0.0;

    printf("Kernel Cache Statistics:\n");
    printf("  Entries:    %zu / %zu (max)\n", cache->count,
           cache->max_entries ? cache->max_entries : SIZE_MAX);
    printf("  Memory:     %zu / %zu bytes\n", cache->total_memory,
           cache->max_memory ? cache->max_memory : SIZE_MAX);
    printf("  Hits:       %zu\n", cache->hits);
    printf("  Misses:     %zu\n", cache->misses);
    printf("  Hit Rate:   %.1f%%\n", hit_rate);
    printf("  Evictions:  %zu\n", cache->evictions);

    pthread_mutex_unlock(&cache->lock);
}

// ============================================================================
// Utility Functions
// ============================================================================

void cml_kernel_cache_set_free_fn(CMLKernelBackend backend, CMLKernelFreeFn free_fn) {
    if (backend >= 0 && backend < CML_KERNEL_BACKEND_COUNT) {
        g_kernel_free_fns[backend] = free_fn;
    }
}

CMLKernelCache* cml_kernel_cache_get_default(void) {
    if (!g_default_cache) {
        // Create default cache with 256 max entries
        g_default_cache = cml_kernel_cache_create(256);
    }
    return g_default_cache;
}

// ============================================================================
// Public API Wrappers (used by cml.c)
// These use _impl suffix to avoid name collision with public cml_ functions
// ============================================================================

void cml_kernel_cache_clear_impl(CMLKernelCache* cache) { kernel_cache_clear(cache); }

void cml_kernel_cache_stats_impl(CMLKernelCache* cache, size_t* hits, size_t* misses, size_t* count,
                                 size_t* memory) {
    kernel_cache_stats(cache, hits, misses, count, memory);
}

double cml_kernel_cache_hit_rate_impl(CMLKernelCache* cache) {
    return kernel_cache_hit_rate(cache);
}

void cml_kernel_cache_print_stats_impl(CMLKernelCache* cache) { kernel_cache_print_stats(cache); }
