#include "alloc/memory_pools.h"
#include "core/logging.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>
#include "alloc/cml_allocator.h"

/** Create a fixed-size block pool: preallocates `num_blocks` blocks of `block_size` bytes up
 *  front. Returns NULL on invalid args or OOM, rolling back any partial allocation. Destroy
 *  with memory_pool_free. */
MemoryPool* memory_pool_create(size_t block_size, int num_blocks, DType dtype) {
    if (block_size == 0 || num_blocks <= 0) {
        LOG_ERROR("Invalid parameters for memory_pool_create");
        return NULL;
    }

    MemoryPool* pool = cml_malloc(sizeof(MemoryPool));
    if (!pool)
        return NULL;

    pool->blocks      = cml_malloc((size_t)num_blocks * sizeof(void*));
    pool->block_sizes = cml_malloc((size_t)num_blocks * sizeof(size_t));
    pool->used        = cml_malloc((size_t)num_blocks * sizeof(size_t));

    if (!pool->blocks || !pool->block_sizes || !pool->used) {
        if (pool->blocks)
            cml_free(pool->blocks);
        if (pool->block_sizes)
            cml_free(pool->block_sizes);
        if (pool->used)
            cml_free(pool->used);
        cml_free(pool);
        return NULL;
    }

    for (int i = 0; i < num_blocks; i++) {
        pool->blocks[i] = cml_malloc(block_size);
        if (!pool->blocks[i]) {
            for (int j = 0; j < i; j++) {
                cml_free(pool->blocks[j]);
            }
            cml_free(pool->blocks);
            cml_free(pool->block_sizes);
            cml_free(pool->used);
            cml_free(pool);
            return NULL;
        }
        pool->block_sizes[i] = block_size;
        pool->used[i]        = 0;
    }

    pool->num_blocks = num_blocks;
    pool->capacity   = num_blocks;
    pool->block_size = block_size;
    pool->dtype      = dtype;
    pthread_mutex_init(&pool->lock, NULL);

    return pool;
}

/** Free a block pool: releases every preallocated block, the bookkeeping arrays, the mutex,
 *  and the pool itself. NULL-safe. Any block still handed out becomes dangling. */
void memory_pool_free(MemoryPool* pool) {
    if (!pool)
        return;

    if (pool->blocks) {
        for (int i = 0; i < pool->num_blocks; i++) {
            if (pool->blocks[i]) {
                cml_free(pool->blocks[i]);
            }
        }
        cml_free(pool->blocks);
    }

    if (pool->block_sizes)
        cml_free(pool->block_sizes);
    if (pool->used)
        cml_free(pool->used);
    pthread_mutex_destroy(&pool->lock);
    cml_free(pool);
}

/** Hand out the first free block under the pool lock, or NULL when the pool is exhausted or
 *  NULL. Thread-safe. The returned block stays owned by the pool; return it with
 *  memory_pool_free_block, never cml_free. */
void* memory_pool_alloc(MemoryPool* pool) {
    if (!pool)
        return NULL;

    pthread_mutex_lock(&pool->lock);
    for (int i = 0; i < pool->num_blocks; i++) {
        if (!pool->used[i]) {
            pool->used[i] = 1;
            pthread_mutex_unlock(&pool->lock);
            return pool->blocks[i];
        }
    }
    pthread_mutex_unlock(&pool->lock);

    return NULL;
}

/** Return a block to the pool, marking it reusable under the pool lock. Returns 0 on success,
 *  -1 (and logs) if the block does not belong to this pool. Thread-safe. */
int memory_pool_free_block(MemoryPool* pool, void* block) {
    if (!pool || !block)
        return -1;

    pthread_mutex_lock(&pool->lock);
    for (int i = 0; i < pool->num_blocks; i++) {
        if (pool->blocks[i] == block) {
            pool->used[i] = 0;
            pthread_mutex_unlock(&pool->lock);
            return 0;
        }
    }
    pthread_mutex_unlock(&pool->lock);

    LOG_WARNING("Block not found in pool");
    return -1;
}

/** Create a pool of `num_tensors` pre-built tensors sharing one shape/dtype/device, for reuse
 *  without repeated allocation. Returns NULL on invalid args or OOM, rolling back partial
 *  construction. Destroy with tensor_pool_free. */
TensorPool* tensor_pool_create(int* shape, int ndim, size_t num_tensors, DType dtype,
                               DeviceType device) {
    if (!shape || ndim <= 0 || num_tensors == 0) {
        LOG_ERROR("Invalid parameters for tensor_pool_create");
        return NULL;
    }

    TensorPool* pool = cml_malloc(sizeof(TensorPool));
    if (!pool)
        return NULL;

    pool->tensors = cml_malloc(num_tensors * sizeof(Tensor*));
    pool->in_use  = cml_malloc(num_tensors * sizeof(bool));
    pool->shape   = tensor_shape_copy(shape, ndim);

    if (!pool->tensors || !pool->in_use || !pool->shape) {
        if (pool->tensors)
            cml_free(pool->tensors);
        if (pool->in_use)
            cml_free(pool->in_use);
        if (pool->shape)
            cml_free(pool->shape);
        cml_free(pool);
        return NULL;
    }

    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    for (size_t i = 0; i < num_tensors; i++) {
        pool->tensors[i] = tensor_empty(shape, ndim, &config);
        if (!pool->tensors[i]) {
            for (size_t j = 0; j < i; j++) {
                tensor_free(pool->tensors[j]);
            }
            cml_free(pool->tensors);
            cml_free(pool->in_use);
            cml_free(pool->shape);
            cml_free(pool);
            return NULL;
        }
        pool->in_use[i] = false;
    }

    pool->ndim        = ndim;
    pool->num_tensors = num_tensors;
    pool->capacity    = num_tensors;
    pool->dtype       = dtype;
    pool->device      = device;

    return pool;
}

/** Free a tensor pool and every tensor it owns, plus its bookkeeping arrays. NULL-safe. */
void tensor_pool_free(TensorPool* pool) {
    if (!pool)
        return;

    if (pool->tensors) {
        for (size_t i = 0; i < pool->num_tensors; i++) {
            if (pool->tensors[i]) {
                tensor_free(pool->tensors[i]);
            }
        }
        cml_free(pool->tensors);
    }

    if (pool->in_use)
        cml_free(pool->in_use);
    if (pool->shape)
        cml_free(pool->shape);
    cml_free(pool);
}
