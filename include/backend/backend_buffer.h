/**
 * @file backend_buffer.h
 * @brief Backend buffer abstraction (inspired by ggml)
 *
 * Provides a unified interface for memory buffers across different backends.
 * This allows operations to work seamlessly across CPU, CUDA, Metal, etc.
 */

#ifndef CML_CORE_BACKEND_BUFFER_H
#define CML_CORE_BACKEND_BUFFER_H

#include "backend/device.h"
#include "tensor/tensor.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Backend buffer type (opaque pointer)
 */
typedef struct CMLBackendBufferType* CMLBackendBufferType_t;

/**
 * @brief Backend buffer (opaque pointer)
 */
typedef struct CMLBackendBuffer* CMLBackendBuffer_t;

/**
 * @brief Buffer usage type
 */
typedef enum {
    CML_BUFFER_USAGE_ANY     = 0, // Any usage
    CML_BUFFER_USAGE_WEIGHTS = 1, // Model weights (read-only during inference)
    CML_BUFFER_USAGE_COMPUTE = 2, // Computation buffers (temporary)
} CMLBufferUsage;

/**
 * @brief Get buffer type name
 */
const char* cml_backend_buffer_type_name(CMLBackendBufferType_t buft);

/**
 * @brief Allocate buffer of given size
 */
CMLBackendBuffer_t cml_backend_buffer_type_alloc_buffer(CMLBackendBufferType_t buft, size_t size);

/**
 * @brief Get alignment requirement for buffer type
 */
size_t cml_backend_buffer_type_get_alignment(CMLBackendBufferType_t buft);

/**
 * @brief Get maximum size for buffer type
 */
size_t cml_backend_buffer_type_get_max_size(CMLBackendBufferType_t buft);

/**
 * @brief Get allocation size for tensor
 */
size_t cml_backend_buffer_type_get_alloc_size(CMLBackendBufferType_t buft, const Tensor* tensor);

/**
 * @brief Check if buffer type is host-accessible
 */
bool cml_backend_buffer_type_is_host(CMLBackendBufferType_t buft);

/**
 * @brief Get device for buffer type
 */
DeviceType cml_backend_buffer_type_get_device(CMLBackendBufferType_t buft);

/**
 * @brief Get buffer name
 */
const char* cml_backend_buffer_name(CMLBackendBuffer_t buffer);

/**
 * @brief Free buffer
 */
void cml_backend_buffer_free(CMLBackendBuffer_t buffer);

/**
 * @brief Get base pointer of buffer
 */
void* cml_backend_buffer_get_base(CMLBackendBuffer_t buffer);

/**
 * @brief Get buffer size
 */
size_t cml_backend_buffer_get_size(CMLBackendBuffer_t buffer);

/**
 * @brief Initialize tensor in buffer
 */
int cml_backend_buffer_init_tensor(CMLBackendBuffer_t buffer, Tensor* tensor);

/**
 * @brief Get alignment requirement
 */
size_t cml_backend_buffer_get_alignment(CMLBackendBuffer_t buffer);

/**
 * @brief Get maximum size
 */
size_t cml_backend_buffer_get_max_size(CMLBackendBuffer_t buffer);

/**
 * @brief Get allocation size for tensor
 */
size_t cml_backend_buffer_get_alloc_size(CMLBackendBuffer_t buffer, const Tensor* tensor);

/**
 * @brief Clear buffer with value
 */
void cml_backend_buffer_clear(CMLBackendBuffer_t buffer, uint8_t value);

/**
 * @brief Check if buffer is host-accessible
 */
bool cml_backend_buffer_is_host(CMLBackendBuffer_t buffer);

/**
 * @brief Set buffer usage
 */
void cml_backend_buffer_set_usage(CMLBackendBuffer_t buffer, CMLBufferUsage usage);

/**
 * @brief Get buffer usage
 */
CMLBufferUsage cml_backend_buffer_get_usage(CMLBackendBuffer_t buffer);

/**
 * @brief Get buffer type
 */
CMLBackendBufferType_t cml_backend_buffer_get_type(CMLBackendBuffer_t buffer);

/**
 * @brief Reset buffer (free all allocations)
 */
void cml_backend_buffer_reset(CMLBackendBuffer_t buffer);

/**
 * @brief Copy tensor between different backends
 */
void cml_backend_tensor_copy(Tensor* src, Tensor* dst);

/**
 * @brief Get buffer type for device
 */
CMLBackendBufferType_t cml_backend_buffer_type_for_device(DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_BACKEND_BUFFER_H
