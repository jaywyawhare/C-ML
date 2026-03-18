#ifndef CML_CORE_MEMORY_POOLS_H
#define CML_CORE_MEMORY_POOLS_H

#include "tensor/tensor.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MemoryPool {
    void** blocks;       // Array of memory blocks
    size_t* block_sizes; // Size of each block
    size_t* used;        // Usage flags
    int num_blocks;      // Number of blocks
    int capacity;        // Capacity of pool
    size_t block_size;   // Size of each block
    DType dtype;         // Data type for pool
} MemoryPool;

typedef struct TensorPool {
    Tensor** tensors;   // Array of pre-allocated tensors
    int* shape;         // Shape of tensors in pool
    int ndim;           // Number of dimensions
    size_t num_tensors; // Number of tensors in pool
    size_t capacity;    // Capacity of pool
    DType dtype;        // Data type
    DeviceType device;  // Device type
    bool* in_use;       // Usage flags
} TensorPool;

MemoryPool* memory_pool_create(size_t block_size, int num_blocks, DType dtype);

void memory_pool_free(MemoryPool* pool);

void* memory_pool_alloc(MemoryPool* pool);

int memory_pool_free_block(MemoryPool* pool, void* block);

TensorPool* tensor_pool_create(int* shape, int ndim, size_t num_tensors, DType dtype,
                               DeviceType device);

void tensor_pool_free(TensorPool* pool);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_MEMORY_POOLS_H
