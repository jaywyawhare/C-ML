#ifndef CML_MEMORY_MANAGEMENT_H
#define CML_MEMORY_MANAGEMENT_H

#include <stddef.h>
#include "backend/device.h"

/* For CPU-only allocations (structs, etc).
 * For tensor data, use device_alloc() / device_free() instead. */
void* cml_safe_malloc(size_t size, const char* file, int line);
void* cml_safe_calloc(size_t nmemb, size_t size, const char* file, int line);
void* cml_safe_realloc(void* ptr, size_t size, const char* file, int line);
void cml_safe_free(void** ptr);

void* cml_device_alloc(size_t size);
void cml_device_free(void* ptr, DeviceType device);

#endif // CML_MEMORY_MANAGEMENT_H
