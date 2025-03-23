#ifndef MEMORY_MANAGEMENT_H
#define MEMORY_MANAGEMENT_H

#include <stddef.h>

/**
 * @brief Allocates memory safely and logs the file and line number in case of failure.
 *
 * @param size The size of memory to allocate in bytes.
 * @param file The name of the file where the allocation is requested.
 * @param line The line number in the file where the allocation is requested.
 * @return A pointer to the allocated memory, or exits the program on failure.
 */
void *cm_safe_malloc(size_t size, const char *file, int line);

/**
 * @brief Frees allocated memory safely and sets the pointer to NULL.
 *
 * @param ptr A pointer to the memory to be freed.
 */
void cm_safe_free(void *ptr);

#endif