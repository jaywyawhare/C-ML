/**
 * @file graph_cache.h
 * @brief Graph caching for fast repeated execution
 *
 * Caches compiled execution plans so the same graph structure
 * can be executed repeatedly without rebuilding.
 */

#ifndef CML_GRAPH_CACHE_H
#define CML_GRAPH_CACHE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Forward declarations
struct IRNode;
struct Tensor;
typedef struct CMLGraph* CMLGraph_t;

// Cached execution plan - pre-ordered nodes with pre-allocated buffers
typedef struct CMLExecutionPlan {
    struct IRNode** nodes; // Topologically sorted nodes
    size_t num_nodes;      // Number of nodes
    float** buffers;       // Pre-allocated output buffers
    size_t* buffer_sizes;  // Buffer sizes in floats
    uint64_t signature;    // Graph structure hash
    bool valid;            // Is this plan valid?
} CMLExecutionPlan;

// Graph cache entry
typedef struct CMLGraphCacheEntry {
    uint64_t signature;
    CMLExecutionPlan* plan;
    struct CMLGraphCacheEntry* next;
    size_t last_used; // For LRU eviction
} CMLGraphCacheEntry;

// Graph cache
typedef struct CMLGraphCache {
    CMLGraphCacheEntry** buckets;
    size_t num_buckets;
    size_t count;
    size_t max_entries;
    size_t timestamp;
    size_t hits;
    size_t misses;
} CMLGraphCache;

// Create/destroy cache
CMLGraphCache* cml_graph_cache_create(size_t max_entries);
void cml_graph_cache_destroy(CMLGraphCache* cache);

// Compute graph signature (hash of structure, not data)
uint64_t cml_graph_compute_signature(CMLGraph_t ir);

// Look up cached plan
CMLExecutionPlan* cml_graph_cache_lookup(CMLGraphCache* cache, uint64_t signature);

// Insert plan into cache
int cml_graph_cache_insert(CMLGraphCache* cache, uint64_t signature, CMLExecutionPlan* plan);

// Create execution plan from IR
CMLExecutionPlan* cml_create_execution_plan(CMLGraph_t ir);

// Execute using cached plan (fast path)
int cml_execute_plan(CMLExecutionPlan* plan, struct Tensor** inputs, size_t num_inputs);

// Free execution plan
void cml_free_execution_plan(CMLExecutionPlan* plan);

// Get global graph cache
CMLGraphCache* cml_get_graph_cache(void);

// Print cache stats
void cml_graph_cache_print_stats(CMLGraphCache* cache);

#endif // CML_GRAPH_CACHE_H
