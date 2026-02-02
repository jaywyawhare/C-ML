/**
 * @file memory_pools.h
 * @brief Memory pool allocator for tensor operations
 *
 * Provides pre-allocated memory pools for faster tensor operations
 * and reduced memory fragmentation.
 */

#ifndef CML_CORE_MEMORY_POOLS_H
#define CML_CORE_MEMORY_POOLS_H

#include "tensor/tensor.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Memory Pool Structure
typedef struct MemoryPool {
    void** blocks;       // Array of memory blocks
    size_t* block_sizes; // Size of each block
    size_t* used;        // Usage flags
    int num_blocks;      // Number of blocks
    int capacity;        // Capacity of pool
    size_t block_size;   // Size of each block
    DType dtype;         // Data type for pool
} MemoryPool;

// Tensor Pool Structure
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

// Memory Pool Management

/**
 * @brief Create a memory pool
 *
 * @param block_size Size of each block in bytes
 * @param num_blocks Number of blocks in pool
 * @param dtype Data type
 * @return New memory pool, or NULL on failure
 */
MemoryPool* memory_pool_create(size_t block_size, int num_blocks, DType dtype);

/**
 * @brief Free memory pool and all resources
 *
 * @param pool Memory pool to free
 */
void memory_pool_free(MemoryPool* pool);

/**
 * @brief Allocate block from pool
 *
 * @param pool Memory pool
 * @return Pointer to allocated block, or NULL if pool exhausted
 */
void* memory_pool_alloc(MemoryPool* pool);

/**
 * @brief Return block to pool
 *
 * @param pool Memory pool
 * @param block Block to return
 * @return 0 on success, negative value on failure
 */
int memory_pool_free_block(MemoryPool* pool, void* block);

// Tensor Pool Management

/**
 * @brief Create a tensor pool
 *
 * @param shape Shape of tensors in pool
 * @param ndim Number of dimensions
 * @param num_tensors Number of tensors in pool
 * @param dtype Data type
 * @param device Device type
 * @return New tensor pool, or NULL on failure
 */
TensorPool* tensor_pool_create(int* shape, int ndim, size_t num_tensors, DType dtype,
                               DeviceType device);

/**
 * @brief Free tensor pool and all resources
 *
 * @param pool Tensor pool to free
 */
void tensor_pool_free(TensorPool* pool);

/**
 * @brief Get tensor from pool
 *
 * @param pool Tensor pool
 * @return Tensor from pool, or NULL if pool exhausted
 */
Tensor* tensor_pool_get(TensorPool* pool);

/**
 * @brief Return tensor to pool
 *
 * @param pool Tensor pool
 * @param tensor Tensor to return
 * @return 0 on success, negative value on failure
 */
int tensor_pool_return(TensorPool* pool, Tensor* tensor);

// Tensor Reuse Utilities

/**
 * @brief Reuse tensor memory if possible
 *
 * @param tensor Tensor to reuse
 * @param new_shape New shape
 * @param new_ndim Number of dimensions
 * @return Reused tensor, or NULL if reuse not possible
 */
Tensor* tensor_reuse(Tensor* tensor, int* new_shape, int new_ndim);

/**
 * @brief Check if tensor can be reused for new shape
 *
 * @param tensor Tensor to check
 * @param new_shape New shape
 * @param new_ndim Number of dimensions
 * @return true if reuse possible, false otherwise
 */
bool tensor_can_reuse(Tensor* tensor, int* new_shape, int new_ndim);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_MEMORY_POOLS_H
