/**
 * @file backend_buffer.c
 * @brief Backend buffer implementation
 */

#include "backend/backend_buffer.h"
#include "backend/device.h"
#include "core/logging.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

struct CMLBackendBufferType {
    const char* name;
    DeviceType device;
    size_t alignment;
    size_t max_size;
    bool is_host;

    // Allocation function
    CMLBackendBuffer_t (*alloc_buffer)(struct CMLBackendBufferType* buft, size_t size);
};

struct CMLBackendBuffer {
    CMLBackendBufferType_t type;
    void* base;
    size_t size;
    CMLBufferUsage usage;
    DeviceType device;
    bool owns_memory;
};

static CMLBackendBuffer_t cpu_buffer_alloc(struct CMLBackendBufferType* buft, size_t size) {
    (void)buft; // Unused
    void* ptr = malloc(size);
    if (!ptr) {
        LOG_ERROR("Failed to allocate CPU buffer of size %zu", size);
        return NULL;
    }

    CMLBackendBuffer_t buffer = malloc(sizeof(struct CMLBackendBuffer));
    if (!buffer) {
        free(ptr);
        return NULL;
    }

    buffer->type        = (CMLBackendBufferType_t)buft;
    buffer->base        = ptr;
    buffer->size        = size;
    buffer->usage       = CML_BUFFER_USAGE_ANY;
    buffer->device      = DEVICE_CPU;
    buffer->owns_memory = true;

    return buffer;
}

static CMLBackendBuffer_t cuda_buffer_alloc(struct CMLBackendBufferType* buft, size_t size) {
    (void)buft;

    if (!device_cuda_available()) {
        LOG_WARNING("CUDA not available, falling back to CPU");
        return cpu_buffer_alloc(buft, size);
    }

    void* ptr = device_alloc(size, DEVICE_CUDA);
    if (!ptr) {
        LOG_ERROR("Failed to allocate CUDA buffer of size %zu", size);
        return NULL;
    }

    CMLBackendBuffer_t buffer = malloc(sizeof(struct CMLBackendBuffer));
    if (!buffer) {
        device_free(ptr, DEVICE_CUDA);
        return NULL;
    }

    buffer->type        = (CMLBackendBufferType_t)buft;
    buffer->base        = ptr;
    buffer->size        = size;
    buffer->usage       = CML_BUFFER_USAGE_ANY;
    buffer->device      = DEVICE_CUDA;
    buffer->owns_memory = true;

    return buffer;
}

static CMLBackendBuffer_t metal_buffer_alloc(struct CMLBackendBufferType* buft, size_t size) {
    (void)buft;

    if (!device_metal_available()) {
        LOG_WARNING("Metal not available, falling back to CPU");
        return cpu_buffer_alloc(buft, size);
    }

    // Metal uses unified memory on Apple Silicon, but we still use device_alloc
    // which handles aligned allocation for Metal
    void* ptr = device_alloc(size, DEVICE_METAL);
    if (!ptr) {
        LOG_ERROR("Failed to allocate Metal buffer of size %zu", size);
        return NULL;
    }

    CMLBackendBuffer_t buffer = malloc(sizeof(struct CMLBackendBuffer));
    if (!buffer) {
        device_free(ptr, DEVICE_METAL);
        return NULL;
    }

    buffer->type        = (CMLBackendBufferType_t)buft;
    buffer->base        = ptr;
    buffer->size        = size;
    buffer->usage       = CML_BUFFER_USAGE_ANY;
    buffer->device      = DEVICE_METAL;
    buffer->owns_memory = true;

    return buffer;
}

static CMLBackendBuffer_t rocm_buffer_alloc(struct CMLBackendBufferType* buft, size_t size) {
    (void)buft;

    if (!device_rocm_available()) {
        LOG_WARNING("ROCm not available, falling back to CPU");
        return cpu_buffer_alloc(buft, size);
    }

    void* ptr = device_alloc(size, DEVICE_ROCM);
    if (!ptr) {
        LOG_ERROR("Failed to allocate ROCm buffer of size %zu", size);
        return NULL;
    }

    CMLBackendBuffer_t buffer = malloc(sizeof(struct CMLBackendBuffer));
    if (!buffer) {
        device_free(ptr, DEVICE_ROCM);
        return NULL;
    }

    buffer->type        = (CMLBackendBufferType_t)buft;
    buffer->base        = ptr;
    buffer->size        = size;
    buffer->usage       = CML_BUFFER_USAGE_ANY;
    buffer->device      = DEVICE_ROCM;
    buffer->owns_memory = true;

    return buffer;
}

static struct CMLBackendBufferType cpu_buffer_type = {.name      = "CPU",
                                                      .device    = DEVICE_CPU,
                                                      .alignment = 16, // 16-byte alignment for SIMD
                                                      .max_size  = SIZE_MAX,
                                                      .is_host   = true,
                                                      .alloc_buffer = cpu_buffer_alloc};

static struct CMLBackendBufferType cuda_buffer_type = {.name   = "CUDA",
                                                       .device = DEVICE_CUDA,
                                                       .alignment =
                                                           256, // CUDA prefers 256-byte alignment
                                                       .max_size     = SIZE_MAX,
                                                       .is_host      = false,
                                                       .alloc_buffer = cuda_buffer_alloc};

static struct CMLBackendBufferType metal_buffer_type = {
    .name         = "Metal",
    .device       = DEVICE_METAL,
    .alignment    = 256, // Metal prefers 256-byte alignment
    .max_size     = SIZE_MAX,
    .is_host      = true, // Unified memory on Apple Silicon
    .alloc_buffer = metal_buffer_alloc};

static struct CMLBackendBufferType rocm_buffer_type = {.name   = "ROCm",
                                                       .device = DEVICE_ROCM,
                                                       .alignment =
                                                           256, // ROCm prefers 256-byte alignment
                                                       .max_size     = SIZE_MAX,
                                                       .is_host      = false,
                                                       .alloc_buffer = rocm_buffer_alloc};

static CMLBackendBufferType_t get_buffer_type_for_device(DeviceType device) {
    switch (device) {
    case DEVICE_CPU:
        return (CMLBackendBufferType_t)&cpu_buffer_type;
    case DEVICE_CUDA:
        return (CMLBackendBufferType_t)&cuda_buffer_type;
    case DEVICE_METAL:
        return (CMLBackendBufferType_t)&metal_buffer_type;
    case DEVICE_ROCM:
        return (CMLBackendBufferType_t)&rocm_buffer_type;
    case DEVICE_AUTO:
        return (CMLBackendBufferType_t)&cpu_buffer_type;
    default:
        return (CMLBackendBufferType_t)&cpu_buffer_type;
    }
}

const char* cml_backend_buffer_type_name(CMLBackendBufferType_t buft) {
    if (!buft)
        return "Unknown";
    return ((struct CMLBackendBufferType*)buft)->name;
}

CMLBackendBuffer_t cml_backend_buffer_type_alloc_buffer(CMLBackendBufferType_t buft, size_t size) {
    if (!buft || size == 0)
        return NULL;
    return ((struct CMLBackendBufferType*)buft)
        ->alloc_buffer((struct CMLBackendBufferType*)buft, size);
}

size_t cml_backend_buffer_type_get_alignment(CMLBackendBufferType_t buft) {
    if (!buft)
        return 1;
    return ((struct CMLBackendBufferType*)buft)->alignment;
}

size_t cml_backend_buffer_type_get_max_size(CMLBackendBufferType_t buft) {
    if (!buft)
        return 0;
    return ((struct CMLBackendBufferType*)buft)->max_size;
}

size_t cml_backend_buffer_type_get_alloc_size(CMLBackendBufferType_t buft, const Tensor* tensor) {
    if (!buft || !tensor)
        return 0;
    size_t base_size = tensor->numel * cml_dtype_size(tensor->dtype);
    size_t alignment = ((struct CMLBackendBufferType*)buft)->alignment;
    // Align to buffer alignment
    return (base_size + alignment - 1) & ~(alignment - 1);
}

bool cml_backend_buffer_type_is_host(CMLBackendBufferType_t buft) {
    if (!buft)
        return false;
    return ((struct CMLBackendBufferType*)buft)->is_host;
}

DeviceType cml_backend_buffer_type_get_device(CMLBackendBufferType_t buft) {
    if (!buft)
        return DEVICE_CPU;
    return ((struct CMLBackendBufferType*)buft)->device;
}

const char* cml_backend_buffer_name(CMLBackendBuffer_t buffer) {
    if (!buffer)
        return "NULL";
    return cml_backend_buffer_type_name(buffer->type);
}

void cml_backend_buffer_free(CMLBackendBuffer_t buffer) {
    if (!buffer)
        return;

    if (buffer->owns_memory && buffer->base) {
        device_free(buffer->base, buffer->device);
    }

    free(buffer);
}

void* cml_backend_buffer_get_base(CMLBackendBuffer_t buffer) {
    if (!buffer)
        return NULL;
    return buffer->base;
}

size_t cml_backend_buffer_get_size(CMLBackendBuffer_t buffer) {
    if (!buffer)
        return 0;
    return buffer->size;
}

int cml_backend_buffer_init_tensor(CMLBackendBuffer_t buffer, Tensor* tensor) {
    if (!buffer || !tensor)
        return -1;

    size_t required_size = cml_backend_buffer_type_get_alloc_size(buffer->type, tensor);
    if (required_size > buffer->size) {
        LOG_ERROR("Buffer too small for tensor: need %zu, have %zu", required_size, buffer->size);
        return -1;
    }

    // Set tensor data pointer
    tensor->data      = buffer->base;
    tensor->device    = buffer->device;
    tensor->owns_data = false; // Buffer owns the memory

    return 0;
}

size_t cml_backend_buffer_get_alignment(CMLBackendBuffer_t buffer) {
    if (!buffer)
        return 1;
    return cml_backend_buffer_type_get_alignment(buffer->type);
}

size_t cml_backend_buffer_get_max_size(CMLBackendBuffer_t buffer) {
    if (!buffer)
        return 0;
    return cml_backend_buffer_type_get_max_size(buffer->type);
}

size_t cml_backend_buffer_get_alloc_size(CMLBackendBuffer_t buffer, const Tensor* tensor) {
    if (!buffer)
        return 0;
    return cml_backend_buffer_type_get_alloc_size(buffer->type, tensor);
}

void cml_backend_buffer_clear(CMLBackendBuffer_t buffer, uint8_t value) {
    if (!buffer || !buffer->base)
        return;
    memset(buffer->base, value, buffer->size);
}

bool cml_backend_buffer_is_host(CMLBackendBuffer_t buffer) {
    if (!buffer)
        return false;
    return cml_backend_buffer_type_is_host(buffer->type);
}

void cml_backend_buffer_set_usage(CMLBackendBuffer_t buffer, CMLBufferUsage usage) {
    if (!buffer)
        return;
    buffer->usage = usage;
}

CMLBufferUsage cml_backend_buffer_get_usage(CMLBackendBuffer_t buffer) {
    if (!buffer)
        return CML_BUFFER_USAGE_ANY;
    return buffer->usage;
}

CMLBackendBufferType_t cml_backend_buffer_get_type(CMLBackendBuffer_t buffer) {
    if (!buffer)
        return NULL;
    return buffer->type;
}

void cml_backend_buffer_reset(CMLBackendBuffer_t buffer) {
    // For now, just clear the buffer
    if (buffer) {
        cml_backend_buffer_clear(buffer, 0);
    }
}

void cml_backend_tensor_copy(Tensor* src, Tensor* dst) {
    if (!src || !dst)
        return;

    size_t size = src->numel * cml_dtype_size(src->dtype);

    int result = device_copy(dst->data, src->data, size, dst->device, src->device);
    if (result != 0) {
        LOG_ERROR("Failed to copy tensor between backends");
    }
}

CMLBackendBufferType_t cml_backend_buffer_type_for_device(DeviceType device) {
    return get_buffer_type_for_device(device);
}
