#include <stdio.h>
#include <stdlib.h>
#include "../../include/Core/memory_management.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Safe memory management functions with logging and error handling.
 *
 * This module provides safe memory allocation and deallocation functions with:
 * - Error checking and logging
 * - File and line tracking for debugging
 * - NULL pointer protection
 * - Memory leak prevention
 */

/**
 * @brief Allocates memory safely and logs the file and line number in case of failure.
 *
 * This function attempts to allocate the requested memory size. If the allocation fails,
 * it logs an error message with the file name and line number, then exits the program.
 *
 * @param size The size of memory to allocate in bytes.
 * @param file The name of the file where the allocation is requested.
 * @param line The line number in the file where the allocation is requested.
 * @return A pointer to the allocated memory, or CM_MEMORY_ALLOCATION_ERROR on failure.
 */
void *cm_safe_malloc(size_t size, const char *file, int line)
{
    void *ptr = malloc(size);
    if (ptr == NULL)
    {
        LOG_ERROR("Memory allocation failed for %zu bytes in %s at line %d.", size, file, line);
        return (void *)CM_MEMORY_ALLOCATION_ERROR;
    }

    LOG_DEBUG("Allocated %zu bytes at %p in %s at line %d.", size, ptr, file, line);
    return ptr;
}

/**
 * @brief Allocates and zeros memory safely with logging and error handling.
 *
 * @param num Number of elements to allocate.
 * @param size Size of each element in bytes.
 * @param file The name of the file where the allocation is requested.
 * @param line The line number in the file where the allocation is requested.
 * @return A pointer to the allocated memory, or CM_MEMORY_ALLOCATION_ERROR on failure.
 */
void *cm_safe_calloc(size_t num, size_t size, const char *file, int line)
{
    void *ptr = calloc(num, size);
    if (ptr == NULL)
    {
        LOG_ERROR("Memory allocation failed for %zu elements of size %zu bytes in %s at line %d.",
                  num, size, file, line);
        return (void *)CM_MEMORY_ALLOCATION_ERROR;
    }

    LOG_DEBUG("Allocated and zeroed %zu bytes at %p in %s at line %d.",
              num * size, ptr, file, line);
    return ptr;
}

/**
 * @brief Frees allocated memory safely and sets the pointer to NULL.
 *
 * This function frees the memory pointed to by the given pointer. If the pointer is NULL,
 * the function does nothing. It also logs the memory address being freed if debugging is enabled.
 * After freeing the memory, it sets the pointer to NULL to avoid double-free issues.
 *
 * @param ptr A pointer to the memory to be freed.
 */
void cm_safe_free(void **ptr)
{
    if (ptr != NULL && *ptr != NULL)
    {
        LOG_DEBUG("Freeing memory at %p", *ptr);
        free(*ptr);
        *ptr = NULL;
    }
}

/**
 * @brief Reallocates memory safely with logging and error handling.
 *
 * @param ptr Pointer to the memory block to reallocate.
 * @param size New size in bytes.
 * @param file The name of the file where the reallocation is requested.
 * @param line The line number in the file where the reallocation is requested.
 * @return A pointer to the reallocated memory, or CM_MEMORY_ALLOCATION_ERROR on failure.
 */
void *cm_safe_realloc(void *ptr, size_t size, const char *file, int line)
{
    void *new_ptr = realloc(ptr, size);
    if (new_ptr == NULL)
    {
        LOG_ERROR("Memory reallocation failed for %zu bytes in %s at line %d.", size, file, line);
        return (void *)CM_MEMORY_ALLOCATION_ERROR;
    }

    LOG_DEBUG("Reallocated memory to %zu bytes at %p in %s at line %d.", size, new_ptr, file, line);
    return new_ptr;
}
