#ifndef CML_BACKEND_OPENCL_H
#define CML_BACKEND_OPENCL_H

#include "backend/backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Discovers OpenCL platforms/devices and creates context + command queue. */
int opencl_backend_init(void);
void opencl_backend_cleanup(void);
bool opencl_backend_is_available(void);
BackendOps opencl_backend_get_ops(void);
int opencl_backend_get_device_info(char* buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif // CML_BACKEND_OPENCL_H
