#ifndef CML_NN_LAYERS_SEQUENTIAL_H
#define CML_NN_LAYERS_SEQUENTIAL_H

#include "nn.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CMLExecutionPlan;

/* Pre-resolved input source for a single (node, input) pair.
 * Eliminates per-pass O(n) scanning in resolve_input. */
typedef enum { SRC_MODEL_INPUT = 0, SRC_BUFFER, SRC_CONSTANT } InputSourceKind;
typedef struct {
    InputSourceKind kind;
    union {
        size_t buffer_idx;   // SRC_BUFFER: index into plan->buffers
        float* constant_ptr; // SRC_CONSTANT: weight/bias pointer
    };
} ResolvedInput;

typedef struct CachedModelGraph {
    struct CMLExecutionPlan* plan; // Compiled execution plan
    int* input_shape;              // Expected input shape
    int input_ndim;                // Input dimensions
    size_t input_numel;            // Input number of elements
    float* input_buffer;           // Pre-allocated input buffer (aligned)
    float* output_buffer;          // Pre-allocated output buffer (aligned)
    size_t output_numel;           // Output number of elements

    /* Snapshot of each node's input data pointers from the first pass.
       Weight/bias pointers are stable; model input is the only one that
       changes between passes.  original_inputs[i][j] = data ptr of
       node i, input j at cache-creation time. */
    float*** original_inputs;      // [num_nodes][num_inputs_per_node]
    int* node_num_inputs;          // num_inputs for each node

    /* Pre-resolved input table: resolved[i][j] for node i, input j */
    ResolvedInput** resolved;      // [num_nodes][num_inputs_per_node]
    float* orig_model_input;       // Cached original model-input data pointer

    /* Reusable output tensor — avoids malloc per forward pass */
    Tensor* reuse_output;          // Allocated once, data overwritten each pass

    bool valid;                    // Is cache valid?
    bool buffers_populated;        // Have plan buffers been filled at least once?
} CachedModelGraph;

typedef struct Sequential {
    Module base;
    Module** modules;
    int num_modules;
    int capacity;
    CachedModelGraph* cached_graph; // Optional cached execution graph
    bool enable_graph_cache;        // Whether to use graph caching
} Sequential;

Sequential* nn_sequential(void);

int sequential_add(Sequential* seq, Module* module);

Module* sequential_get(Sequential* seq, int index);

int sequential_get_length(Sequential* seq);

Sequential* sequential_add_chain(Sequential* seq, Module* module);

Sequential* nn_sequentialv(int num_layers, ...);

/* When enabled, first forward pass compiles the graph and caches it.
 * Subsequent passes with same input shapes reuse the cached graph. */
void sequential_enable_graph_cache(Sequential* seq, bool enable);

void sequential_invalidate_cache(Sequential* seq);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_SEQUENTIAL_H
