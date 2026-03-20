#include "alloc/graph_allocator.h"
#include "alloc/memory_pools.h"
#include "backend/backend_buffer.h"
#include "backend/device.h"
#include "core/logging.h"
#include "core/computation_graph.h"
#include "core/graph_context.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

struct CMLGraphAllocator {
    CMLBackendBufferType_t* buffer_types;
    CMLBackendBuffer_t* buffers;
    int num_buffers;
    size_t* buffer_sizes;
    bool* buffer_reserved;
    MemoryPool** memory_pools; // Memory pools for buffer reuse
    bool use_pooling;          // Enable memory pooling
};

CMLGraphAllocator_t cml_graph_allocator_new(CMLBackendBufferType_t buft) {
    if (!buft) {
        LOG_ERROR("Invalid buffer type");
        return NULL;
    }

    CMLGraphAllocator_t galloc = malloc(sizeof(struct CMLGraphAllocator));
    if (!galloc)
        return NULL;

    galloc->buffer_types    = malloc(sizeof(CMLBackendBufferType_t));
    galloc->buffers         = malloc(sizeof(CMLBackendBuffer_t));
    galloc->buffer_sizes    = malloc(sizeof(size_t));
    galloc->buffer_reserved = malloc(sizeof(bool));
    galloc->memory_pools    = malloc(sizeof(MemoryPool*));

    if (!galloc->buffer_types || !galloc->buffers || !galloc->buffer_sizes ||
        !galloc->buffer_reserved || !galloc->memory_pools) {
        if (galloc->buffer_types)
            free(galloc->buffer_types);
        if (galloc->buffers)
            free(galloc->buffers);
        if (galloc->buffer_sizes)
            free(galloc->buffer_sizes);
        if (galloc->buffer_reserved)
            free(galloc->buffer_reserved);
        if (galloc->memory_pools)
            free(galloc->memory_pools);
        free(galloc);
        return NULL;
    }

    galloc->buffer_types[0]    = buft;
    galloc->buffers[0]         = NULL;
    galloc->buffer_sizes[0]    = 0;
    galloc->buffer_reserved[0] = false;
    galloc->memory_pools[0]    = NULL;
    galloc->use_pooling        = false; // Disabled by default
    galloc->num_buffers        = 1;

    return galloc;
}

CMLGraphAllocator_t cml_graph_allocator_new_n(CMLBackendBufferType_t* bufts, int n_bufs) {
    if (!bufts || n_bufs <= 0) {
        LOG_ERROR("Invalid buffer types");
        return NULL;
    }

    CMLGraphAllocator_t galloc = malloc(sizeof(struct CMLGraphAllocator));
    if (!galloc)
        return NULL;

    galloc->buffer_types    = malloc((size_t)n_bufs * sizeof(CMLBackendBufferType_t));
    galloc->buffers         = malloc((size_t)n_bufs * sizeof(CMLBackendBuffer_t));
    galloc->buffer_sizes    = malloc((size_t)n_bufs * sizeof(size_t));
    galloc->buffer_reserved = malloc((size_t)n_bufs * sizeof(bool));
    galloc->memory_pools    = malloc((size_t)n_bufs * sizeof(MemoryPool*));

    if (!galloc->buffer_types || !galloc->buffers || !galloc->buffer_sizes ||
        !galloc->buffer_reserved || !galloc->memory_pools) {
        if (galloc->buffer_types)
            free(galloc->buffer_types);
        if (galloc->buffers)
            free(galloc->buffers);
        if (galloc->buffer_sizes)
            free(galloc->buffer_sizes);
        if (galloc->buffer_reserved)
            free(galloc->buffer_reserved);
        if (galloc->memory_pools)
            free(galloc->memory_pools);
        free(galloc);
        return NULL;
    }

    for (int i = 0; i < n_bufs; i++) {
        galloc->buffer_types[i]    = bufts[i];
        galloc->buffers[i]         = NULL;
        galloc->buffer_sizes[i]    = 0;
        galloc->buffer_reserved[i] = false;
        galloc->memory_pools[i]    = NULL;
    }

    galloc->use_pooling = false; // Disabled by default
    galloc->num_buffers = n_bufs;

    return galloc;
}

void cml_graph_allocator_free(CMLGraphAllocator_t galloc) {
    if (!galloc)
        return;

    if (galloc->buffers) {
        for (int i = 0; i < galloc->num_buffers; i++) {
            if (galloc->buffers[i]) {
                cml_backend_buffer_free(galloc->buffers[i]);
            }
        }
        free(galloc->buffers);
    }

    if (galloc->memory_pools) {
        for (int i = 0; i < galloc->num_buffers; i++) {
            if (galloc->memory_pools[i]) {
                memory_pool_free(galloc->memory_pools[i]);
            }
        }
        free(galloc->memory_pools);
    }

    if (galloc->buffer_types)
        free(galloc->buffer_types);
    if (galloc->buffer_sizes)
        free(galloc->buffer_sizes);
    if (galloc->buffer_reserved)
        free(galloc->buffer_reserved);

    free(galloc);
}

static size_t calculate_tensor_size(Tensor* tensor) {
    if (!tensor)
        return 0;
    size_t dtype_size = cml_dtype_size(tensor->dtype);
    return tensor->numel * dtype_size;
}

static size_t calculate_peak_memory_simple(CMLComputationGraph_t graph);

/*
 * Full liveness analysis: topo-sort the graph, track exact lifetime
 * intervals, and return peak(sum of alive tensor sizes) across all steps.
 */
static size_t calculate_peak_memory(CMLComputationGraph_t graph) {
    if (!graph)
        return 0;

    size_t node_count = cml_graph_get_node_count(graph);
    if (node_count == 0)
        return 0;

    size_t* tensor_sizes = calloc(node_count, sizeof(size_t));
    int* execution_order = calloc(node_count, sizeof(int));
    int* use_counts      = calloc(node_count, sizeof(int));
    bool* is_leaf        = calloc(node_count, sizeof(bool));

    if (!tensor_sizes || !execution_order || !use_counts || !is_leaf) {
        if (tensor_sizes)
            free(tensor_sizes);
        if (execution_order)
            free(execution_order);
        if (use_counts)
            free(use_counts);
        if (is_leaf)
            free(is_leaf);
        // Fallback to adaptive analysis if full analysis fails
        return calculate_peak_memory_simple(graph);
    }

    for (size_t i = 0; i < node_count; i++) {
        CMLGraphNode_t node = cml_graph_get_node_by_index(graph, i);
        if (!node)
            continue;

        is_leaf[i]     = cml_graph_node_is_leaf(node);
        Tensor* tensor = cml_graph_node_get_tensor(node);
        if (tensor) {
            tensor_sizes[i] = calculate_tensor_size(tensor);
        } else {
            // Fallback estimate
            tensor_sizes[i] = is_leaf[i] ? 8 * 1024 : 4 * 1024;
        }

        execution_order[i] = -1; // Not yet ordered
        use_counts[i]      = 0;

        // Count how many nodes use this node's output
        for (size_t j = 0; j < node_count; j++) {
            CMLGraphNode_t consumer = cml_graph_get_node_by_index(graph, j);
            if (!consumer || consumer == node)
                continue;

            int num_inputs = cml_graph_node_get_num_inputs(consumer);
            for (int k = 0; k < num_inputs; k++) {
                CMLGraphNode_t input = cml_graph_node_get_input(consumer, k);
                if (input == node) {
                    use_counts[i]++;
                    break;
                }
            }
        }
    }

    // Use Kahn's algorithm
    int* in_degree = calloc(node_count, sizeof(int));
    if (!in_degree) {
        free(tensor_sizes);
        free(execution_order);
        free(use_counts);
        free(is_leaf);
        return calculate_peak_memory_simple(graph);
    }

    for (size_t i = 0; i < node_count; i++) {
        CMLGraphNode_t node = cml_graph_get_node_by_index(graph, i);
        if (!node)
            continue;

        if (is_leaf[i]) {
            in_degree[i] = 0;
        } else {
            int num_inputs = cml_graph_node_get_num_inputs(node);
            in_degree[i]   = num_inputs;
        }
    }

    size_t* queue = malloc(node_count * sizeof(size_t));
    if (!queue) {
        free(tensor_sizes);
        free(execution_order);
        free(use_counts);
        free(is_leaf);
        free(in_degree);
        return calculate_peak_memory_simple(graph);
    }

    int queue_front = 0;
    int queue_back  = 0;

    for (size_t i = 0; i < node_count; i++) {
        if (in_degree[i] == 0) {
            queue[queue_back++] = i;
        }
    }

    int order = 0;
    while (queue_front < queue_back) {
        size_t current_idx           = queue[queue_front++];
        execution_order[current_idx] = order++;

        // Decrease in-degree of nodes that depend on current
        CMLGraphNode_t current_node = cml_graph_get_node_by_index(graph, current_idx);
        if (current_node) {
            for (size_t i = 0; i < node_count; i++) {
                if (in_degree[i] > 0) {
                    CMLGraphNode_t node = cml_graph_get_node_by_index(graph, i);
                    if (!node)
                        continue;

                    int num_inputs = cml_graph_node_get_num_inputs(node);
                    for (int j = 0; j < num_inputs; j++) {
                        CMLGraphNode_t input = cml_graph_node_get_input(node, j);
                        if (input == current_node) {
                            in_degree[i]--;
                            if (in_degree[i] == 0) {
                                queue[queue_back++] = i;
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    free(in_degree);
    free(queue);

    size_t peak_memory = 0;

    int* alive_until = calloc(node_count, sizeof(int));
    if (!alive_until) {
        free(tensor_sizes);
        free(execution_order);
        free(use_counts);
        free(is_leaf);
        return calculate_peak_memory_simple(graph);
    }

    // A tensor is alive until all its consumers have executed
    for (size_t i = 0; i < node_count; i++) {
        if (is_leaf[i]) {
            // Leaf nodes are always alive (never freed)
            alive_until[i] = order;
        } else {
            // Find the last consumer's execution order
            int last_consumer_order = execution_order[i];
            CMLGraphNode_t node     = cml_graph_get_node_by_index(graph, i);
            if (node) {
                for (size_t j = 0; j < node_count; j++) {
                    CMLGraphNode_t consumer = cml_graph_get_node_by_index(graph, j);
                    if (!consumer || consumer == node)
                        continue;

                    int num_inputs = cml_graph_node_get_num_inputs(consumer);
                    for (int k = 0; k < num_inputs; k++) {
                        CMLGraphNode_t input = cml_graph_node_get_input(consumer, k);
                        if (input == node) {
                            int consumer_order = execution_order[j];
                            if (consumer_order > last_consumer_order) {
                                last_consumer_order = consumer_order;
                            }
                            break;
                        }
                    }
                }
            }
            alive_until[i] = last_consumer_order + 1; // Freed after last consumer
        }
    }

    for (int step = 0; step < order; step++) {
        size_t current_memory = 0;

        for (size_t i = 0; i < node_count; i++) {
            int exec_order = execution_order[i];
            if (exec_order >= 0 && exec_order <= step && step < alive_until[i]) {
                current_memory += tensor_sizes[i];
            }
        }

        if (current_memory > peak_memory) {
            peak_memory = current_memory;
        }
    }

    free(tensor_sizes);
    free(execution_order);
    free(use_counts);
    free(is_leaf);
    free(alive_until);

    // Ensure minimum allocation (1MB) for small graphs
    if (peak_memory < 1024 * 1024) {
        peak_memory = 1024 * 1024;
    }

    return peak_memory;
}

/* Fallback: adaptive liveness factor (30-50%) based on graph complexity */
static size_t calculate_peak_memory_simple(CMLComputationGraph_t graph) {
    if (!graph)
        return 0;

    size_t node_count = cml_graph_get_node_count(graph);

    if (node_count == 0)
        return 0;

    size_t leaf_memory = 0;
    for (size_t i = 0; i < node_count; i++) {
        CMLGraphNode_t node = cml_graph_get_node_by_index(graph, i);
        if (node && cml_graph_node_is_leaf(node)) {
            Tensor* tensor = cml_graph_node_get_tensor(node);
            if (tensor) {
                leaf_memory += calculate_tensor_size(tensor);
            } else {
                leaf_memory += 8 * 1024;
            }
        }
    }

    // Factor depends on graph structure: deeper graphs have lower peak liveness
    size_t total_intermediate_memory = 0;
    size_t max_depth                 = 0;

    for (size_t i = 0; i < node_count; i++) {
        CMLGraphNode_t node = cml_graph_get_node_by_index(graph, i);
        if (node && !cml_graph_node_is_leaf(node)) {
            Tensor* tensor = cml_graph_node_get_tensor(node);
            if (tensor) {
                total_intermediate_memory += calculate_tensor_size(tensor);
            } else {
                total_intermediate_memory += 4 * 1024;
            }
            max_depth++;
        }
    }

    int liveness_percent            = (node_count < 10) ? 50 : (node_count < 50) ? 40 : 30;
    size_t peak_intermediate_memory = (total_intermediate_memory * (size_t)liveness_percent) / 100;
    size_t peak_memory              = leaf_memory + peak_intermediate_memory;

    if (peak_memory < 1024 * 1024) {
        peak_memory = 1024 * 1024;
    }

    return peak_memory;
}

bool cml_graph_allocator_reserve(CMLGraphAllocator_t galloc, void* graph) {
    if (!galloc || galloc->num_buffers == 0)
        return false;

    if (graph) {
        CMLComputationGraph_t cgraph = (CMLComputationGraph_t)graph;
        size_t peak_memory = calculate_peak_memory(cgraph);

        if (peak_memory == 0) {
            peak_memory = 1024 * 1024; // 1MB default
        }

        for (int i = 0; i < galloc->num_buffers; i++) {
            if (!galloc->buffer_reserved[i]) {
                galloc->buffer_reserved[i] = true;
                if (peak_memory > 0 && galloc->buffer_sizes[i] < peak_memory) {
                    galloc->buffer_sizes[i] = peak_memory;
                }
            }
        }
    } else {
        for (int i = 0; i < galloc->num_buffers; i++) {
            galloc->buffer_reserved[i] = true;
        }
    }

    return true;
}

bool cml_graph_allocator_reserve_n(CMLGraphAllocator_t galloc, void* graph,
                                   const int* node_buffer_ids, const int* leaf_buffer_ids) {
    if (!galloc || galloc->num_buffers == 0)
        return false;

    // This allows fine-grained control over which nodes use which buffers
    if (graph && node_buffer_ids && leaf_buffer_ids) {
        CMLComputationGraph_t cgraph = (CMLComputationGraph_t)graph;

        size_t* buffer_memory = calloc((size_t)galloc->num_buffers, sizeof(size_t));
        if (!buffer_memory) {
            // Fallback to basic reserve
            return cml_graph_allocator_reserve(galloc, graph);
        }

        size_t node_count = cml_graph_get_node_count(cgraph);
        size_t leaf_count = cml_graph_get_leaf_count(cgraph);

        for (size_t i = 0; i < node_count; i++) {
            CMLGraphNode_t node = cml_graph_get_node_by_index(cgraph, i);
            if (!node)
                continue;

            Tensor* tensor = cml_graph_node_get_tensor(node);
            size_t tensor_size =
                tensor ? calculate_tensor_size(tensor) : (4 * 1024); // 4KB fallback

            int buffer_id = (i < node_count) ? node_buffer_ids[i] : 0;
            if (buffer_id >= 0 && buffer_id < galloc->num_buffers) {
                buffer_memory[buffer_id] += tensor_size;
            }
        }

        for (size_t i = 0; i < leaf_count; i++) {
            for (size_t j = 0; j < node_count; j++) {
                CMLGraphNode_t node = cml_graph_get_node_by_index(cgraph, j);
                if (!node || !cml_graph_node_is_leaf(node))
                    continue;

                Tensor* tensor = cml_graph_node_get_tensor(node);
                size_t tensor_size =
                    tensor ? calculate_tensor_size(tensor) : (8 * 1024); // 8KB fallback for leaves

                int buffer_id = (i < leaf_count) ? leaf_buffer_ids[i] : 0;
                if (buffer_id >= 0 && buffer_id < galloc->num_buffers) {
                    buffer_memory[buffer_id] += tensor_size;
                }
                break; // Only count each leaf once
            }
        }

        for (int i = 0; i < galloc->num_buffers; i++) {
            if (!galloc->buffer_reserved[i]) {
                galloc->buffer_reserved[i] = true;
                if (buffer_memory[i] > 0 && galloc->buffer_sizes[i] < buffer_memory[i]) {
                    galloc->buffer_sizes[i] = buffer_memory[i];
                }
            }
        }

        free(buffer_memory);
    } else {
        // Fallback to basic reserve
        return cml_graph_allocator_reserve(galloc, graph);
    }

    return true;
}

bool cml_graph_allocator_alloc_graph(CMLGraphAllocator_t galloc, void* graph) {
    if (!galloc || galloc->num_buffers == 0)
        return false;

    for (int i = 0; i < galloc->num_buffers; i++) {
        if (!galloc->buffers[i]) {
            size_t buffer_size = galloc->buffer_sizes[i];

            if (buffer_size == 0) {
                buffer_size             = 1024 * 1024; // 1MB default
                galloc->buffer_sizes[i] = buffer_size;
            }

            if (graph) {
                CMLComputationGraph_t cgraph = (CMLComputationGraph_t)graph;
                size_t calculated_size       = calculate_peak_memory(cgraph);
                if (calculated_size > 0) {
                    if (calculated_size > buffer_size) {
                        buffer_size = calculated_size;
                    }
                    galloc->buffer_sizes[i] = buffer_size;
                }
            }

            galloc->buffers[i] =
                cml_backend_buffer_type_alloc_buffer(galloc->buffer_types[i], buffer_size);
            if (!galloc->buffers[i]) {
                LOG_ERROR("Failed to allocate graph buffer %d of size %zu", i, buffer_size);
                for (int j = 0; j < i; j++) {
                    if (galloc->buffers[j]) {
                        cml_backend_buffer_free(galloc->buffers[j]);
                        galloc->buffers[j] = NULL;
                    }
                }
                return false;
            }
        }
    }

    return true;
}

size_t cml_graph_allocator_get_buffer_size(CMLGraphAllocator_t galloc, int buffer_id) {
    if (!galloc || buffer_id < 0 || buffer_id >= galloc->num_buffers)
        return 0;
    return galloc->buffer_sizes[buffer_id];
}

void cml_graph_allocator_enable_pooling(CMLGraphAllocator_t galloc, bool enable) {
    if (!galloc)
        return;
    galloc->use_pooling = enable;
}

bool cml_graph_allocator_realloc_buffer(CMLGraphAllocator_t galloc, int buffer_id,
                                        size_t new_size) {
    if (!galloc || buffer_id < 0 || buffer_id >= galloc->num_buffers)
        return false;

    // If new size is smaller or equal, no need to reallocate
    if (new_size <= galloc->buffer_sizes[buffer_id]) {
        return true;
    }

    if (galloc->buffers[buffer_id]) {
        cml_backend_buffer_free(galloc->buffers[buffer_id]);
        galloc->buffers[buffer_id] = NULL;
    }

    galloc->buffer_sizes[buffer_id] = new_size;
    galloc->buffers[buffer_id] =
        cml_backend_buffer_type_alloc_buffer(galloc->buffer_types[buffer_id], new_size);

    if (!galloc->buffers[buffer_id]) {
        LOG_ERROR("Failed to reallocate buffer %d to size %zu", buffer_id, new_size);
        return false;
    }

    return true;
}

bool cml_graph_allocator_init_pool(CMLGraphAllocator_t galloc, int buffer_id, size_t block_size,
                                   int num_blocks, DType dtype) {
    if (!galloc || buffer_id < 0 || buffer_id >= galloc->num_buffers)
        return false;

    if (galloc->memory_pools[buffer_id]) {
        memory_pool_free(galloc->memory_pools[buffer_id]);
        galloc->memory_pools[buffer_id] = NULL;
    }

    galloc->memory_pools[buffer_id] = memory_pool_create(block_size, num_blocks, dtype);
    if (!galloc->memory_pools[buffer_id]) {
        LOG_ERROR("Failed to create memory pool for buffer %d", buffer_id);
        return false;
    }

    galloc->use_pooling = true;
    return true;
}

void cml_tensor_allocator_new(CMLTensorAllocator* talloc, CMLBackendBuffer_t buffer) {
    if (!talloc)
        return;
    *talloc = (CMLTensorAllocator){0};
    if (!buffer)
        return;

    talloc->buffer    = buffer;
    talloc->base      = cml_backend_buffer_get_base(buffer);
    talloc->alignment = cml_backend_buffer_get_alignment(buffer);
    talloc->offset    = 0;
}

int cml_tensor_allocator_alloc(CMLTensorAllocator* talloc, Tensor* tensor) {
    if (!talloc || !tensor || !talloc->buffer)
        return -1;

    size_t required_size = cml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size_t buffer_size   = cml_backend_buffer_get_size(talloc->buffer);

    size_t aligned_offset = (talloc->offset + talloc->alignment - 1) & ~(talloc->alignment - 1);

    if (aligned_offset + required_size > buffer_size) {
        LOG_ERROR("Tensor allocator out of memory");
        return -1;
    }

    tensor->data = (char*)talloc->base + aligned_offset;
    tensor->device =
        cml_backend_buffer_type_get_device(cml_backend_buffer_get_type(talloc->buffer));
    tensor->owns_data = false;

    talloc->offset = aligned_offset + required_size;

    return 0;
}

struct CMLContext {
    CMLBackendBuffer_t buffer; // Use backend buffer for unified device operations
    CMLTensorAllocator tensor_allocator;
    size_t mem_size;
    size_t used_mem;
    bool no_alloc;
    DeviceType device;
};

CMLContext_t cml_context_new(CMLContextParams params) {
    CMLContext_t ctx = malloc(sizeof(struct CMLContext));
    if (!ctx)
        return NULL;

    ctx->no_alloc         = params.no_alloc;
    ctx->device           = DEVICE_CPU;
    ctx->buffer           = NULL;
    ctx->tensor_allocator = (CMLTensorAllocator){0};
    ctx->used_mem         = 0;
    ctx->mem_size         = 0;

    if (params.mem_buffer) {
        CMLBackendBufferType_t buft = cml_backend_buffer_type_for_device(ctx->device);
        if (!buft) {
            free(ctx);
            return NULL;
        }

        ctx->buffer = cml_backend_buffer_type_alloc_buffer(buft, params.mem_size);
        if (!ctx->buffer) {
            free(ctx);
            return NULL;
        }

        void* base = cml_backend_buffer_get_base(ctx->buffer);
        if (base && params.mem_buffer) {
            memcpy(base, params.mem_buffer, params.mem_size);
        }

        ctx->mem_size = params.mem_size;
        cml_tensor_allocator_new(&ctx->tensor_allocator, ctx->buffer);
    } else if (params.mem_size > 0 && !params.no_alloc) {
        CMLBackendBufferType_t buft = cml_backend_buffer_type_for_device(ctx->device);
        if (!buft) {
            free(ctx);
            return NULL;
        }

        ctx->buffer = cml_backend_buffer_type_alloc_buffer(buft, params.mem_size);
        if (!ctx->buffer) {
            free(ctx);
            return NULL;
        }

        ctx->mem_size = params.mem_size;
        cml_tensor_allocator_new(&ctx->tensor_allocator, ctx->buffer);
    } else {
        // No pre-allocation - dynamic allocation mode
        ctx->buffer   = NULL;
        ctx->mem_size = 0;
    }

    return ctx;
}

void cml_context_free(CMLContext_t ctx) {
    if (!ctx)
        return;

    if (ctx->buffer) {
        cml_backend_buffer_free(ctx->buffer);
    }

    free(ctx);
}

size_t cml_context_used_mem(CMLContext_t ctx) {
    if (!ctx)
        return 0;
    return ctx->used_mem;
}

size_t cml_context_total_mem(CMLContext_t ctx) {
    if (!ctx)
        return 0;
    return ctx->mem_size;
}

Tensor* cml_context_alloc_tensor(CMLContext_t ctx, int* shape, int ndim, DType dtype,
                                 DeviceType device) {
    if (!ctx || !shape || ndim <= 0)
        return NULL;

    size_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        numel *= (size_t)shape[i];
    }
    size_t size = numel * cml_dtype_size(dtype);

    if (ctx->no_alloc) {
        // Just measure memory usage
        ctx->used_mem += size;
        return NULL; // Don't actually allocate
    }

    if (ctx->buffer && ctx->tensor_allocator.buffer) {
        TensorConfig config = {
            .dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
        Tensor* tensor = tensor_empty(shape, ndim, &config);
        if (!tensor)
            return NULL;

        // Release the default buffer allocated by tensor_empty; we'll use the context buffer
        if (tensor->buffer_handle) {
            cml_backend_buffer_free(tensor->buffer_handle);
            tensor->buffer_handle = NULL;
        } else if (tensor->data) {
            if (tensor->device == DEVICE_CPU || tensor->device == DEVICE_AUTO) {
                free(tensor->data);
            } else {
                device_free(tensor->data, tensor->device);
            }
        }
        tensor->data      = NULL;
        tensor->owns_data = false;

        if (cml_tensor_allocator_alloc(&ctx->tensor_allocator, tensor) != 0) {
            tensor_free(tensor);
            LOG_ERROR("Context out of memory: used %zu, total %zu, need %zu", ctx->used_mem,
                      ctx->mem_size, size);
            return NULL;
        }

        tensor->buffer_handle = ctx->buffer;
        ctx->used_mem += size;
        return tensor;
    }

    CMLBackendBufferType_t buft = cml_backend_buffer_type_for_device(device);
    if (!buft) {
        // Fallback to standard allocation
        TensorConfig config = {
            .dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
        return tensor_empty(shape, ndim, &config);
    }

    CMLBackendBuffer_t buffer = cml_backend_buffer_type_alloc_buffer(buft, size);
    if (!buffer) {
        LOG_ERROR("Failed to allocate buffer for tensor");
        return NULL;
    }

    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* tensor      = tensor_empty(shape, ndim, &config);
    if (!tensor) {
        cml_backend_buffer_free(buffer);
        return NULL;
    }

    if (cml_backend_buffer_init_tensor(buffer, tensor) != 0) {
        tensor_free(tensor);
        cml_backend_buffer_free(buffer);
        return NULL;
    }

    ctx->used_mem += size;
    return tensor;
}
