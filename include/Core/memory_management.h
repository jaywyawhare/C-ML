#ifndef CML_MEMORY_MANAGEMENT_H
#define CML_MEMORY_MANAGEMENT_H

#include <stddef.h>

// Safe memory allocation functions
void* cm_safe_malloc(size_t size, const char* file, int line);
void* cm_safe_calloc(size_t nmemb, size_t size, const char* file, int line);
void* cm_safe_realloc(void* ptr, size_t size, const char* file, int line);
void cm_safe_free(void** ptr);

// Convenience macros
#define CM_MALLOC(size) cm_safe_malloc(size, __FILE__, __LINE__)
#define CM_CALLOC(nmemb, size) cm_safe_calloc(nmemb, size, __FILE__, __LINE__)
#define CM_REALLOC(ptr, size) cm_safe_realloc(ptr, size, __FILE__, __LINE__)
#define CM_FREE(ptr) cm_safe_free((void**)&(ptr))

#endif // CML_MEMORY_MANAGEMENT_H
