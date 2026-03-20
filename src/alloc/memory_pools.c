#include "alloc/memory_pools.h"
#include "core/logging.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>

MemoryPool* memory_pool_create(size_t block_size, int num_blocks, DType dtype) {
    if (block_size == 0 || num_blocks <= 0) {
        LOG_ERROR("Invalid parameters for memory_pool_create");
        return NULL;
    }

    MemoryPool* pool = malloc(sizeof(MemoryPool));
    if (!pool)
        return NULL;

    pool->blocks      = malloc((size_t)num_blocks * sizeof(void*));
    pool->block_sizes = malloc((size_t)num_blocks * sizeof(size_t));
    pool->used        = malloc((size_t)num_blocks * sizeof(size_t));

    if (!pool->blocks || !pool->block_sizes || !pool->used) {
        if (pool->blocks)
            free(pool->blocks);
        if (pool->block_sizes)
            free(pool->block_sizes);
        if (pool->used)
            free(pool->used);
        free(pool);
        return NULL;
    }

    for (int i = 0; i < num_blocks; i++) {
        pool->blocks[i] = malloc(block_size);
        if (!pool->blocks[i]) {
            for (int j = 0; j < i; j++) {
                free(pool->blocks[j]);
            }
            free(pool->blocks);
            free(pool->block_sizes);
            free(pool->used);
            free(pool);
            return NULL;
        }
        pool->block_sizes[i] = block_size;
        pool->used[i]        = 0;
    }

    pool->num_blocks = num_blocks;
    pool->capacity   = num_blocks;
    pool->block_size = block_size;
    pool->dtype      = dtype;

    return pool;
}

void memory_pool_free(MemoryPool* pool) {
    if (!pool)
        return;

    if (pool->blocks) {
        for (int i = 0; i < pool->num_blocks; i++) {
            if (pool->blocks[i]) {
                free(pool->blocks[i]);
            }
        }
        free(pool->blocks);
    }

    if (pool->block_sizes)
        free(pool->block_sizes);
    if (pool->used)
        free(pool->used);
    free(pool);
}

void* memory_pool_alloc(MemoryPool* pool) {
    if (!pool)
        return NULL;

    for (int i = 0; i < pool->num_blocks; i++) {
        if (!pool->used[i]) {
            pool->used[i] = 1;
            return pool->blocks[i];
        }
    }

    return NULL;
}

int memory_pool_free_block(MemoryPool* pool, void* block) {
    if (!pool || !block)
        return -1;

    for (int i = 0; i < pool->num_blocks; i++) {
        if (pool->blocks[i] == block) {
            pool->used[i] = 0;
            return 0;
        }
    }

    LOG_WARNING("Block not found in pool");
    return -1;
}

TensorPool* tensor_pool_create(int* shape, int ndim, size_t num_tensors, DType dtype,
                               DeviceType device) {
    if (!shape || ndim <= 0 || num_tensors == 0) {
        LOG_ERROR("Invalid parameters for tensor_pool_create");
        return NULL;
    }

    TensorPool* pool = malloc(sizeof(TensorPool));
    if (!pool)
        return NULL;

    pool->tensors = malloc(num_tensors * sizeof(Tensor*));
    pool->in_use  = malloc(num_tensors * sizeof(bool));
    pool->shape   = tensor_shape_copy(shape, ndim);

    if (!pool->tensors || !pool->in_use || !pool->shape) {
        if (pool->tensors)
            free(pool->tensors);
        if (pool->in_use)
            free(pool->in_use);
        if (pool->shape)
            free(pool->shape);
        free(pool);
        return NULL;
    }

    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    for (size_t i = 0; i < num_tensors; i++) {
        pool->tensors[i] = tensor_empty(shape, ndim, &config);
        if (!pool->tensors[i]) {
            for (size_t j = 0; j < i; j++) {
                tensor_free(pool->tensors[j]);
            }
            free(pool->tensors);
            free(pool->in_use);
            free(pool->shape);
            free(pool);
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

void tensor_pool_free(TensorPool* pool) {
    if (!pool)
        return;

    if (pool->tensors) {
        for (size_t i = 0; i < pool->num_tensors; i++) {
            if (pool->tensors[i]) {
                tensor_free(pool->tensors[i]);
            }
        }
        free(pool->tensors);
    }

    if (pool->in_use)
        free(pool->in_use);
    if (pool->shape)
        free(pool->shape);
    free(pool);
}
