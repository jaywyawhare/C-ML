/**
 * @file sequential.h
 * @brief Sequential container
 *
 * Implements the nn.Sequential container with optional graph caching
 * for fast repeated execution.
 */

#ifndef CML_NN_LAYERS_SEQUENTIAL_H
#define CML_NN_LAYERS_SEQUENTIAL_H

#include "nn.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
struct CMLExecutionPlan;

/**
 * @brief Cached model graph for fast repeated execution
 *
 * Stores a compiled execution plan that can be reused across forward passes
 * when input shapes remain the same.
 */
typedef struct CachedModelGraph {
    struct CMLExecutionPlan* plan; // Compiled execution plan
    int* input_shape;              // Expected input shape
    int input_ndim;                // Input dimensions
    size_t input_numel;            // Input number of elements
    float* input_buffer;           // Pre-allocated input buffer
    float* output_buffer;          // Pre-allocated output buffer
    size_t output_numel;           // Output number of elements
    bool valid;                    // Is cache valid?
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

/**
 * @brief Add a layer to Sequential and return self for chaining
 *
 * @param seq Sequential container
 * @param module Module to add
 * @return Sequential container (for chaining)
 */
Sequential* sequential_add_chain(Sequential* seq, Module* module);

/**
 * @brief Create Sequential with varargs layers
 *
 * @param num_layers Number of layers
 * @param ... Module pointers (num_layers arguments)
 * @return Sequential container, or NULL on failure
 */
Sequential* nn_sequentialv(int num_layers, ...);

/**
 * @brief Enable or disable graph caching for fast repeated execution
 *
 * When enabled, the first forward pass compiles the graph and caches it.
 * Subsequent forward passes with same input shapes reuse the cached graph.
 *
 * @param seq Sequential container
 * @param enable true to enable, false to disable
 */
void sequential_enable_graph_cache(Sequential* seq, bool enable);

/**
 * @brief Invalidate the cached graph (call when model changes)
 *
 * @param seq Sequential container
 */
void sequential_invalidate_cache(Sequential* seq);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_SEQUENTIAL_H
