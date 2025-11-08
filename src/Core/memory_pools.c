/**
 * @file memory_pools.c
 * @brief Memory pool allocator implementation
 */

#include "Core/memory_pools.h"
#include "Core/memory_management.h"
#include "Core/logging.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>

MemoryPool* memory_pool_create(size_t block_size, int num_blocks, DType dtype) {
    if (block_size == 0 || num_blocks <= 0) {
        LOG_ERROR("Invalid parameters for memory_pool_create");
        return NULL;
    }

    MemoryPool* pool = CM_MALLOC(sizeof(MemoryPool));
    if (!pool)
        return NULL;

    pool->blocks      = CM_MALLOC(num_blocks * sizeof(void*));
    pool->block_sizes = CM_MALLOC(num_blocks * sizeof(size_t));
    pool->used        = CM_MALLOC(num_blocks * sizeof(size_t));

    if (!pool->blocks || !pool->block_sizes || !pool->used) {
        if (pool->blocks)
            CM_FREE(pool->blocks);
        if (pool->block_sizes)
            CM_FREE(pool->block_sizes);
        if (pool->used)
            CM_FREE(pool->used);
        CM_FREE(pool);
        return NULL;
    }

    // Allocate all blocks
    for (int i = 0; i < num_blocks; i++) {
        pool->blocks[i] = CM_MALLOC(block_size);
        if (!pool->blocks[i]) {
            // Free already allocated blocks
            for (int j = 0; j < i; j++) {
                CM_FREE(pool->blocks[j]);
            }
            CM_FREE(pool->blocks);
            CM_FREE(pool->block_sizes);
            CM_FREE(pool->used);
            CM_FREE(pool);
            return NULL;
        }
        pool->block_sizes[i] = block_size;
        pool->used[i]        = 0;
    }

    pool->num_blocks = num_blocks;
    pool->capacity   = num_blocks;
    pool->block_size = block_size;
    pool->dtype      = dtype;

    LOG_DEBUG("Created memory pool: %d blocks of %zu bytes", num_blocks, block_size);
    return pool;
}

void memory_pool_free(MemoryPool* pool) {
    if (!pool)
        return;

    if (pool->blocks) {
        for (int i = 0; i < pool->num_blocks; i++) {
            if (pool->blocks[i]) {
                CM_FREE(pool->blocks[i]);
            }
        }
        CM_FREE(pool->blocks);
    }

    if (pool->block_sizes)
        CM_FREE(pool->block_sizes);
    if (pool->used)
        CM_FREE(pool->used);
    CM_FREE(pool);
}

void* memory_pool_alloc(MemoryPool* pool) {
    if (!pool)
        return NULL;

    // Find first available block
    for (int i = 0; i < pool->num_blocks; i++) {
        if (!pool->used[i]) {
            pool->used[i] = 1;
            return pool->blocks[i];
        }
    }

    LOG_DEBUG("Memory pool exhausted");
    return NULL;
}

int memory_pool_free_block(MemoryPool* pool, void* block) {
    if (!pool || !block)
        return -1;

    // Find block and mark as unused
    for (int i = 0; i < pool->num_blocks; i++) {
        if (pool->blocks[i] == block) {
            pool->used[i] = 0;
            return 0;
        }
    }

    LOG_WARNING("Block not found in pool");
    return -1;
}

// Tensor Pool Implementation

TensorPool* tensor_pool_create(int* shape, int ndim, size_t num_tensors, DType dtype,
                               DeviceType device) {
    if (!shape || ndim <= 0 || num_tensors == 0) {
        LOG_ERROR("Invalid parameters for tensor_pool_create");
        return NULL;
    }

    TensorPool* pool = CM_MALLOC(sizeof(TensorPool));
    if (!pool)
        return NULL;

    pool->tensors = CM_MALLOC(num_tensors * sizeof(Tensor*));
    pool->in_use  = CM_MALLOC(num_tensors * sizeof(bool));
    pool->shape   = tensor_shape_copy(shape, ndim);

    if (!pool->tensors || !pool->in_use || !pool->shape) {
        if (pool->tensors)
            CM_FREE(pool->tensors);
        if (pool->in_use)
            CM_FREE(pool->in_use);
        if (pool->shape)
            CM_FREE(pool->shape);
        CM_FREE(pool);
        return NULL;
    }

    // Pre-allocate all tensors
    for (size_t i = 0; i < num_tensors; i++) {
        pool->tensors[i] = tensor_empty(shape, ndim, dtype, device);
        if (!pool->tensors[i]) {
            // Free already allocated tensors
            for (size_t j = 0; j < i; j++) {
                tensor_free(pool->tensors[j]);
            }
            CM_FREE(pool->tensors);
            CM_FREE(pool->in_use);
            CM_FREE(pool->shape);
            CM_FREE(pool);
            return NULL;
        }
        pool->in_use[i] = false;
    }

    pool->ndim        = ndim;
    pool->num_tensors = num_tensors;
    pool->capacity    = num_tensors;
    pool->dtype       = dtype;
    pool->device      = device;

    LOG_DEBUG("Created tensor pool: %zu tensors with shape", num_tensors);
    return pool;
}

void tensor_pool_free(TensorPool* pool) {
    if (!pool)
        return;

    if (pool->tensors) {
        for (size_t i = 0; i < pool->num_tensors; i++) {
            if (pool->tensors[i]) {
                tensor_free(pool->tensors[i]);
            }
        }
        CM_FREE(pool->tensors);
    }

    if (pool->in_use)
        CM_FREE(pool->in_use);
    if (pool->shape)
        CM_FREE(pool->shape);
    CM_FREE(pool);
}

Tensor* tensor_pool_get(TensorPool* pool) {
    if (!pool)
        return NULL;

    // Find first available tensor
    for (size_t i = 0; i < pool->num_tensors; i++) {
        if (!pool->in_use[i]) {
            pool->in_use[i] = true;
            return pool->tensors[i];
        }
    }

    LOG_DEBUG("Tensor pool exhausted");
    return NULL;
}

int tensor_pool_return(TensorPool* pool, Tensor* tensor) {
    if (!pool || !tensor)
        return -1;

    // Find tensor and mark as unused
    for (size_t i = 0; i < pool->num_tensors; i++) {
        if (pool->tensors[i] == tensor) {
            pool->in_use[i] = false;
            return 0;
        }
    }

    LOG_WARNING("Tensor not found in pool");
    return -1;
}

// Tensor Reuse Implementation

bool tensor_can_reuse(Tensor* tensor, int* new_shape, int new_ndim) {
    if (!tensor || !new_shape || new_ndim <= 0)
        return false;

    // Check if shapes match
    if (tensor->ndim != new_ndim)
        return false;

    // Check if total elements match (for reuse)
    size_t new_numel = 1;
    for (int i = 0; i < new_ndim; i++) {
        new_numel *= new_shape[i];
    }

    // Can reuse if numel matches and tensor is contiguous
    return tensor->numel == new_numel && tensor->is_contiguous;
}

Tensor* tensor_reuse(Tensor* tensor, int* new_shape, int new_ndim) {
    if (!tensor_can_reuse(tensor, new_shape, new_ndim)) {
        return NULL;
    }

    // Update shape and strides
    for (int i = 0; i < new_ndim; i++) {
        tensor->shape[i] = new_shape[i];
    }

    // Recompute strides
    if (tensor->strides) {
        CM_FREE(tensor->strides);
    }
    tensor->strides = compute_contiguous_strides(new_shape, new_ndim);
    tensor->ndim    = new_ndim;

    return tensor;
}
