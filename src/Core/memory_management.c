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

void cm_safe_free(void *ptr)
{
    if (ptr != NULL)
    {
        DEBUG_LOG("Freed memory at %p\n", ptr); // Log before freeing
        free(ptr);
        ptr = NULL;
    }
}
