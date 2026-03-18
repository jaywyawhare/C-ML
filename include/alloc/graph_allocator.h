#ifndef CML_CORE_GRAPH_ALLOCATOR_H
#define CML_CORE_GRAPH_ALLOCATOR_H

#include "backend/backend_buffer.h"
#include "tensor/tensor.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLGraphAllocator* CMLGraphAllocator_t;

CMLGraphAllocator_t cml_graph_allocator_new(CMLBackendBufferType_t buft);

CMLGraphAllocator_t cml_graph_allocator_new_n(CMLBackendBufferType_t* bufts, int n_bufs);

void cml_graph_allocator_free(CMLGraphAllocator_t galloc);

/* Pre-allocates buffers from a worst-case graph to avoid reallocations. */
bool cml_graph_allocator_reserve(CMLGraphAllocator_t galloc, void* graph);

bool cml_graph_allocator_reserve_n(CMLGraphAllocator_t galloc, void* graph,
                                   const int* node_buffer_ids, const int* leaf_buffer_ids);

/* Auto-reallocates if topology changes with single buffer.
 * Returns false if using multiple buffers and re-allocation is needed. */
bool cml_graph_allocator_alloc_graph(CMLGraphAllocator_t galloc, void* graph);

size_t cml_graph_allocator_get_buffer_size(CMLGraphAllocator_t galloc, int buffer_id);

void cml_graph_allocator_enable_pooling(CMLGraphAllocator_t galloc, bool enable);

bool cml_graph_allocator_realloc_buffer(CMLGraphAllocator_t galloc, int buffer_id, size_t new_size);

bool cml_graph_allocator_init_pool(CMLGraphAllocator_t galloc, int buffer_id, size_t block_size,
                                   int num_blocks, DType dtype);

typedef struct {
    CMLBackendBuffer_t buffer;
    void* base;
    size_t alignment;
    size_t offset;
} CMLTensorAllocator;

void cml_tensor_allocator_new(CMLTensorAllocator* talloc, CMLBackendBuffer_t buffer);

int cml_tensor_allocator_alloc(CMLTensorAllocator* talloc, Tensor* tensor);

typedef struct CMLContext* CMLContext_t;

typedef struct {
    size_t mem_size;  // Memory buffer size (0 = dynamic allocation)
    void* mem_buffer; // Pre-allocated buffer (NULL = allocate internally)
    bool no_alloc;    // Don't allocate, only measure
} CMLContextParams;

CMLContext_t cml_context_new(CMLContextParams params);

void cml_context_free(CMLContext_t ctx);

size_t cml_context_used_mem(CMLContext_t ctx);

size_t cml_context_total_mem(CMLContext_t ctx);

Tensor* cml_context_alloc_tensor(CMLContext_t ctx, int* shape, int ndim, DType dtype,
                                 DeviceType device);

void cml_context_set_param(CMLContext_t ctx, Tensor* tensor);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_GRAPH_ALLOCATOR_H
