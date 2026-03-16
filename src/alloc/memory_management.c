#include <stdio.h>
#include <stdlib.h>
#include "alloc/memory_management.h"
#include "core/error_codes.h"
#include "core/logging.h"
#include "backend/device.h"

void* cml_safe_malloc(size_t size, const char* file, int line) {
    void* ptr = malloc(size);
    if (ptr == NULL) {
        LOG_ERROR("Memory allocation failed for %zu bytes in %s at line %d.", size, file, line);
        return (void*)CM_MEMORY_ALLOCATION_ERROR;
    }

    return ptr;
}

void* cml_safe_calloc(size_t num, size_t size, const char* file, int line) {
    void* ptr = calloc(num, size);
    if (ptr == NULL) {
        LOG_ERROR("Memory allocation failed for %zu elements of size %zu bytes in %s at line %d.",
                  num, size, file, line);
        return (void*)CM_MEMORY_ALLOCATION_ERROR;
    }

    return ptr;
}

void cml_safe_free(void** ptr) {
    if (ptr != NULL && *ptr != NULL) {
        free(*ptr);
        *ptr = NULL;
    }
}

void* cml_safe_realloc(void* ptr, size_t size, const char* file, int line) {
    void* new_ptr = realloc(ptr, size);
    if (new_ptr == NULL) {
        LOG_ERROR("Memory reallocation failed for %zu bytes in %s at line %d.", size, file, line);
        return (void*)CM_MEMORY_ALLOCATION_ERROR;
    }

    return new_ptr;
}

void* cml_device_alloc(size_t size) {
    DeviceType device = device_get_default();
    return device_alloc(size, device);
}

void cml_device_free(void* ptr, DeviceType device) { device_free(ptr, device); }
