/*
 * Graph caching for fast repeated execution.
 * Caches compiled execution plans so the same graph structure
 * can be executed repeatedly without rebuilding.
 */

#ifndef CML_GRAPH_CACHE_H
#define CML_GRAPH_CACHE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

struct IRNode;
struct Tensor;
typedef struct CMLGraph* CMLGraph_t;

typedef struct CMLExecutionPlan {
    struct IRNode** nodes; // Topologically sorted nodes
    size_t num_nodes;
    float** buffers;       // Pre-allocated output buffers
    size_t* buffer_sizes;  // Buffer sizes in floats
    uint64_t signature;    // Graph structure hash
    bool valid;
} CMLExecutionPlan;

typedef struct CMLGraphCacheEntry {
    uint64_t signature;
    CMLExecutionPlan* plan;
    struct CMLGraphCacheEntry* next;
    size_t last_used; // For LRU eviction
} CMLGraphCacheEntry;

typedef struct CMLGraphCache {
    CMLGraphCacheEntry** buckets;
    size_t num_buckets;
    size_t count;
    size_t max_entries;
    size_t timestamp;
    size_t hits;
    size_t misses;
} CMLGraphCache;

CMLGraphCache* cml_graph_cache_create(size_t max_entries);
void cml_graph_cache_destroy(CMLGraphCache* cache);
uint64_t cml_graph_compute_signature(CMLGraph_t ir);
CMLExecutionPlan* cml_graph_cache_lookup(CMLGraphCache* cache, uint64_t signature);
int cml_graph_cache_insert(CMLGraphCache* cache, uint64_t signature, CMLExecutionPlan* plan);
CMLExecutionPlan* cml_create_execution_plan(CMLGraph_t ir);
int cml_execute_plan(CMLExecutionPlan* plan, struct Tensor** inputs, size_t num_inputs);
void cml_free_execution_plan(CMLExecutionPlan* plan);
CMLGraphCache* cml_get_graph_cache(void);
void cml_graph_cache_print_stats(CMLGraphCache* cache);

int cml_execute_node_fast(struct IRNode* node, float* out_buf);

#endif // CML_GRAPH_CACHE_H
