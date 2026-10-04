#include <stdio.h>
#include <stdlib.h>
#include "alloc/memory_management.h"
#include "core/error_codes.h"
#include "core/logging.h"
#include "backend/device.h"
#include "alloc/cml_allocator.h"

/** cml_malloc that logs the failing file/line and returns the CM_MEMORY_ALLOCATION_ERROR
 *  sentinel (not NULL) on failure; usually reached via a macro that passes __FILE__/__LINE__. */
void* cml_safe_malloc(size_t size, const char* file, int line) {
    void* ptr = cml_malloc(size);
    if (ptr == NULL) {
        LOG_ERROR("Memory allocation failed for %zu bytes in %s at line %d.", size, file, line);
        return (void*)CM_MEMORY_ALLOCATION_ERROR;
    }

    return ptr;
}

/** cml_calloc variant that logs the failing site and returns the CM_MEMORY_ALLOCATION_ERROR
 *  sentinel on failure. */
void* cml_safe_calloc(size_t num, size_t size, const char* file, int line) {
    void* ptr = cml_calloc(num, size);
    if (ptr == NULL) {
        LOG_ERROR("Memory allocation failed for %zu elements of size %zu bytes in %s at line %d.",
                  num, size, file, line);
        return (void*)CM_MEMORY_ALLOCATION_ERROR;
    }

    return ptr;
}

/** Free through a pointer-to-pointer and set it to NULL to prevent dangling/double-free;
 *  ignores NULL or already-NULL targets. */
void cml_safe_free(void** ptr) {
    if (ptr != NULL && *ptr != NULL) {
        cml_free(*ptr);
        *ptr = NULL;
    }
}

/** cml_realloc variant that logs the failing site and returns the CM_MEMORY_ALLOCATION_ERROR
 *  sentinel on failure (the original block is left untouched by cml_realloc in that case). */
void* cml_safe_realloc(void* ptr, size_t size, const char* file, int line) {
    void* new_ptr = cml_realloc(ptr, size);
    if (new_ptr == NULL) {
        LOG_ERROR("Memory reallocation failed for %zu bytes in %s at line %d.", size, file, line);
        return (void*)CM_MEMORY_ALLOCATION_ERROR;
    }

    return new_ptr;
}

/** Allocate `size` bytes on the current default device; free with cml_device_free. */
void* cml_device_alloc(size_t size) {
    DeviceType device = device_get_default();
    return device_alloc(size, device);
}

/** Free memory from cml_device_alloc on the given device. */
void cml_device_free(void* ptr, DeviceType device) { device_free(ptr, device); }
