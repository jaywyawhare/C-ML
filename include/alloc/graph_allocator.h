/**
 * @file graph_allocator.h
 * @brief Graph-based memory allocator (inspired by ggml)
 *
 * Provides efficient memory allocation for computation graphs by:
 * - Pre-allocating buffers based on graph structure
 * - Reusing memory across graph executions
 * - Minimizing memory fragmentation
 */

#ifndef CML_CORE_GRAPH_ALLOCATOR_H
#define CML_CORE_GRAPH_ALLOCATOR_H

#include "backend/backend_buffer.h"
#include "tensor/tensor.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Graph allocator (opaque pointer)
 */
typedef struct CMLGraphAllocator* CMLGraphAllocator_t;

/**
 * @brief Create graph allocator with buffer type
 */
CMLGraphAllocator_t cml_graph_allocator_new(CMLBackendBufferType_t buft);

/**
 * @brief Create graph allocator with multiple buffer types
 */
CMLGraphAllocator_t cml_graph_allocator_new_n(CMLBackendBufferType_t* bufts, int n_bufs);

/**
 * @brief Free graph allocator
 */
void cml_graph_allocator_free(CMLGraphAllocator_t galloc);

/**
 * @brief Reserve buffers from a measurement graph
 *
 * Pre-allocates buffers from a worst-case graph to avoid reallocations.
 * Does not allocate or modify the graph itself.
 *
 * @param galloc Graph allocator
 * @param graph Computation graph (can be NULL for manual reservation)
 * @return true on success, false on failure
 */
bool cml_graph_allocator_reserve(CMLGraphAllocator_t galloc, void* graph);

/**
 * @brief Reserve buffers with explicit node/leaf buffer IDs
 */
bool cml_graph_allocator_reserve_n(CMLGraphAllocator_t galloc, void* graph,
                                   const int* node_buffer_ids, const int* leaf_buffer_ids);

/**
 * @brief Allocate graph
 *
 * Automatically reallocates if topology changes when using single buffer.
 * Returns false if using multiple buffers and re-allocation is needed
 * (call cml_graph_allocator_reserve_n first).
 *
 * @param galloc Graph allocator
 * @param graph Computation graph
 * @return true on success, false on failure
 */
bool cml_graph_allocator_alloc_graph(CMLGraphAllocator_t galloc, void* graph);

/**
 * @brief Get buffer size for specific buffer ID
 */
size_t cml_graph_allocator_get_buffer_size(CMLGraphAllocator_t galloc, int buffer_id);

/**
 * @brief Enable or disable memory pooling
 */
void cml_graph_allocator_enable_pooling(CMLGraphAllocator_t galloc, bool enable);

/**
 * @brief Dynamically reallocate buffer to new size (if larger)
 */
bool cml_graph_allocator_realloc_buffer(CMLGraphAllocator_t galloc, int buffer_id, size_t new_size);

/**
 * @brief Initialize memory pool for a buffer (for memory reuse)
 */
bool cml_graph_allocator_init_pool(CMLGraphAllocator_t galloc, int buffer_id, size_t block_size,
                                   int num_blocks, DType dtype);

/**
 * @brief Tensor allocator structure
 */
typedef struct {
    CMLBackendBuffer_t buffer;
    void* base;
    size_t alignment;
    size_t offset;
} CMLTensorAllocator;

/**
 * @brief Create tensor allocator from buffer
 */
void cml_tensor_allocator_new(CMLTensorAllocator* talloc, CMLBackendBuffer_t buffer);

/**
 * @brief Allocate tensor in allocator
 */
int cml_tensor_allocator_alloc(CMLTensorAllocator* talloc, Tensor* tensor);

/**
 * @brief Computation context (similar to ggml_context)
 */
typedef struct CMLContext* CMLContext_t;

/**
 * @brief Context initialization parameters
 */
typedef struct {
    size_t mem_size;  // Memory buffer size (0 = dynamic allocation)
    void* mem_buffer; // Pre-allocated buffer (NULL = allocate internally)
    bool no_alloc;    // Don't allocate, only measure
} CMLContextParams;

/**
 * @brief Create computation context
 */
CMLContext_t cml_context_new(CMLContextParams params);

/**
 * @brief Free computation context
 */
void cml_context_free(CMLContext_t ctx);

/**
 * @brief Get used memory in context
 */
size_t cml_context_used_mem(CMLContext_t ctx);

/**
 * @brief Get total memory in context
 */
size_t cml_context_total_mem(CMLContext_t ctx);

/**
 * @brief Allocate tensor in context
 */
Tensor* cml_context_alloc_tensor(CMLContext_t ctx, int* shape, int ndim, DType dtype,
                                 DeviceType device);

/**
 * @brief Mark tensor as parameter (for autograd/optimization)
 */
void cml_context_set_param(CMLContext_t ctx, Tensor* tensor);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_GRAPH_ALLOCATOR_H
