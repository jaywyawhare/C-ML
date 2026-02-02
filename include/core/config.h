/**
 * @file config.h
 * @brief Global configuration for C-ML library
 *
 * Provides sensible defaults for device, dtype, RNG, and allocator.
 * This removes noise from examples by eliminating the need to specify
 * DEVICE_CPU/DTYPE_FLOAT32 everywhere.
 */

#ifndef CML_CORE_CONFIG_H
#define CML_CORE_CONFIG_H

#include "backend/device.h"
#include "tensor/tensor.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Set the default device for tensor creation
 *
 * When tensors are created without specifying a device, this default
 * will be used. Default is DEVICE_CPU.
 *
 * @param device DeviceType to set as default
 */
void cml_set_default_device(DeviceType device);

/**
 * @brief Get the default device
 *
 * @return Default DeviceType
 */
DeviceType cml_get_default_device(void);

/**
 * @brief Set the default dtype for tensor creation
 *
 * When tensors are created without specifying a dtype, this default
 * will be used. Default is DTYPE_FLOAT32.
 *
 * @param dtype DType to set as default
 */
void cml_set_default_dtype(DType dtype);

/**
 * @brief Get the default dtype
 *
 * @return Default DType
 */
DType cml_get_default_dtype(void);

/**
 * @brief Seed the random number generator
 *
 * Sets the seed for all random number generation in the library.
 * This ensures reproducible results across runs.
 *
 * @param seed Random seed value
 */
void cml_seed(uint64_t seed);

/**
 * @brief Get a random seed based on current time
 *
 * @return Random seed value
 */
uint64_t cml_random_seed(void);

/**
 * @brief Allocator function pointer type
 *
 * @param size Size in bytes to allocate
 * @return Pointer to allocated memory, or NULL on failure
 */
typedef void* (*CMLAllocator)(size_t size);

/**
 * @brief Deallocator function pointer type
 *
 * @param ptr Pointer to memory to free
 */
typedef void (*CMLDeallocator)(void* ptr);

/**
 * @brief Reallocator function pointer type
 *
 * @param ptr Pointer to existing memory
 * @param size New size in bytes
 * @return Pointer to reallocated memory, or NULL on failure
 */
typedef void* (*CMLReallocator)(void* ptr, size_t size);

/**
 * @brief Set custom allocator functions
 *
 * Allows users to provide custom memory allocation functions.
 * This is useful for memory pools, arenas, or custom allocators.
 *
 * @param alloc Allocator function (NULL to use default)
 * @param dealloc Deallocator function (NULL to use default)
 * @param realloc Reallocator function (NULL to use default)
 */
void cml_set_allocator(CMLAllocator alloc, CMLDeallocator dealloc, CMLReallocator realloc);

/**
 * @brief Get current allocator functions
 *
 * @param alloc Output pointer for allocator function
 * @param dealloc Output pointer for deallocator function
 * @param realloc Output pointer for reallocator function
 */
void cml_get_allocator(CMLAllocator* alloc, CMLDeallocator* dealloc, CMLReallocator* realloc);

/**
 * @brief Reset allocator to default (malloc/free/realloc)
 */
void cml_reset_allocator(void);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_CONFIG_H
