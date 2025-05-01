#ifndef MEMORY_MANAGEMENT_H
#define MEMORY_MANAGEMENT_H

#include <stddef.h>

/**
 * @brief Allocates memory safely and logs the file and line number in case of failure.
 *
 * @param size The size of memory to allocate in bytes.
 * @param file The name of the file where the allocation is requested.
 * @param line The line number in the file where the allocation is requested.
 * @return A pointer to the allocated memory, or CM_MEMORY_ALLOCATION_ERROR on failure.
 */
void *cm_safe_malloc(size_t size, const char *file, int line);

/**
 * @brief Allocates and zeros memory safely with logging and error handling.
 *
 * @param num Number of elements to allocate.
 * @param size Size of each element in bytes.
 * @param file The name of the file where the allocation is requested.
 * @param line The line number in the file where the allocation is requested.
 * @return A pointer to the allocated memory, or CM_MEMORY_ALLOCATION_ERROR on failure.
 */
void *cm_safe_calloc(size_t num, size_t size, const char *file, int line);

/**
 * @brief Reallocates memory safely with logging and error handling.
 *
 * @param ptr Pointer to the memory block to reallocate.
 * @param size New size in bytes.
 * @param file The name of the file where the reallocation is requested.
 * @param line The line number in the file where the reallocation is requested.
 * @return A pointer to the reallocated memory, or CM_MEMORY_ALLOCATION_ERROR on failure.
 */
void *cm_safe_realloc(void *ptr, size_t size, const char *file, int line);

/**
 * @brief Frees allocated memory safely and sets the pointer to NULL.
 *
 * @param ptr A double pointer to the memory to be freed.
 */
void cm_safe_free(void **ptr);

#endif