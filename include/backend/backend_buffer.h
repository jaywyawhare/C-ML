#ifndef CML_CORE_BACKEND_BUFFER_H
#define CML_CORE_BACKEND_BUFFER_H

#include "backend/device.h"
#include "tensor/tensor.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLBackendBufferType* CMLBackendBufferType_t;
typedef struct CMLBackendBuffer* CMLBackendBuffer_t;

typedef enum {
    CML_BUFFER_USAGE_ANY     = 0, // Any usage
    CML_BUFFER_USAGE_WEIGHTS = 1, // Model weights (read-only during inference)
    CML_BUFFER_USAGE_COMPUTE = 2, // Computation buffers (temporary)
} CMLBufferUsage;

const char* cml_backend_buffer_type_name(CMLBackendBufferType_t buft);
CMLBackendBuffer_t cml_backend_buffer_type_alloc_buffer(CMLBackendBufferType_t buft, size_t size);
size_t cml_backend_buffer_type_get_alignment(CMLBackendBufferType_t buft);
size_t cml_backend_buffer_type_get_max_size(CMLBackendBufferType_t buft);
size_t cml_backend_buffer_type_get_alloc_size(CMLBackendBufferType_t buft, const Tensor* tensor);
bool cml_backend_buffer_type_is_host(CMLBackendBufferType_t buft);
DeviceType cml_backend_buffer_type_get_device(CMLBackendBufferType_t buft);

const char* cml_backend_buffer_name(CMLBackendBuffer_t buffer);
void cml_backend_buffer_free(CMLBackendBuffer_t buffer);
void* cml_backend_buffer_get_base(CMLBackendBuffer_t buffer);
size_t cml_backend_buffer_get_size(CMLBackendBuffer_t buffer);
int cml_backend_buffer_init_tensor(CMLBackendBuffer_t buffer, Tensor* tensor);
size_t cml_backend_buffer_get_alignment(CMLBackendBuffer_t buffer);
size_t cml_backend_buffer_get_max_size(CMLBackendBuffer_t buffer);
size_t cml_backend_buffer_get_alloc_size(CMLBackendBuffer_t buffer, const Tensor* tensor);
void cml_backend_buffer_clear(CMLBackendBuffer_t buffer, uint8_t value);
bool cml_backend_buffer_is_host(CMLBackendBuffer_t buffer);
void cml_backend_buffer_set_usage(CMLBackendBuffer_t buffer, CMLBufferUsage usage);
CMLBufferUsage cml_backend_buffer_get_usage(CMLBackendBuffer_t buffer);
CMLBackendBufferType_t cml_backend_buffer_get_type(CMLBackendBuffer_t buffer);
void cml_backend_buffer_reset(CMLBackendBuffer_t buffer);
void cml_backend_tensor_copy(Tensor* src, Tensor* dst);
CMLBackendBufferType_t cml_backend_buffer_type_for_device(DeviceType device);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_BACKEND_BUFFER_H
