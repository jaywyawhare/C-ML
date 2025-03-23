#include <stdio.h>
#include <stdlib.h>
#include "../../include/Core/memory_management.h"

// #define DEBUG

#ifdef DEBUG
#define DEBUG_LOG(fmt, ...) \
    fprintf(stderr, fmt, __VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...) \
    do                      \
    {                       \
    } while (0)
#endif

/**
 * @brief Allocates memory safely and logs the file and line number in case of failure.
 *
 * This function attempts to allocate the requested memory size. If the allocation fails,
 * it logs an error message with the file name and line number, then exits the program.
 *
 * @param size The size of memory to allocate in bytes.
 * @param file The name of the file where the allocation is requested.
 * @param line The line number in the file where the allocation is requested.
 * @return A pointer to the allocated memory, or exits the program on failure.
 */
void *cm_safe_malloc(size_t size, const char *file, int line)
{
    void *ptr = malloc(size);
    if (ptr == NULL)
    {
        fprintf(stderr, "Memory allocation failed for %zu bytes in %s at line %d\n", size, file, line);
        exit(EXIT_FAILURE);
    }

    DEBUG_LOG("Allocated %zu bytes at %p in %s at line %d\n", size, ptr, file, line);
    return ptr;
}

/**
 * @brief Frees allocated memory safely and sets the pointer to NULL.
 *
 * This function frees the memory pointed to by the given pointer. If the pointer is NULL,
 * the function does nothing. It also logs the memory address being freed if debugging is enabled.
 *
 * @param ptr A pointer to the memory to be freed.
 */
void cm_safe_free(void *ptr)
{
    if (ptr != NULL)
    {
        DEBUG_LOG("Freed memory at %p\n", ptr);
        free(ptr);
        ptr = NULL;
    }
}
