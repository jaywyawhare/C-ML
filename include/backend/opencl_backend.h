/**
 * @file opencl_backend.h
 * @brief OpenCL backend for cross-vendor GPU acceleration
 *
 * Provides OpenCL-based implementations of tensor operations.
 * Enable with -DENABLE_OPENCL=ON in CMake.
 */

#ifndef CML_BACKEND_OPENCL_H
#define CML_BACKEND_OPENCL_H

#include "backend/backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize OpenCL backend
 *
 * Discovers OpenCL platforms/devices and creates context + command queue.
 *
 * @return 0 on success, negative on failure
 */
int opencl_backend_init(void);

/**
 * @brief Cleanup OpenCL backend resources
 */
void opencl_backend_cleanup(void);

/**
 * @brief Check if OpenCL is available on this system
 * @return true if OpenCL is available
 */
bool opencl_backend_is_available(void);

/**
 * @brief Get OpenCL backend operations
 * @return BackendOps structure with OpenCL implementations
 */
BackendOps opencl_backend_get_ops(void);

/**
 * @brief Get OpenCL device info string
 * @param buffer Output buffer
 * @param buffer_size Size of output buffer
 * @return 0 on success
 */
int opencl_backend_get_device_info(char* buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif // CML_BACKEND_OPENCL_H
