#ifndef CML_CORE_CONFIG_H
#define CML_CORE_CONFIG_H

#include "backend/device.h"
#include "tensor/tensor.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Version macros
#define CML_VERSION_MAJOR 0
#define CML_VERSION_MINOR 0
#define CML_VERSION_PATCH 2
#define CML_VERSION_STRING "0.0.2"

/**
 * @brief Encode version components into a single integer for comparison
 *
 * Usage: #if CML_VERSION_ENCODE(CML_VERSION_MAJOR, CML_VERSION_MINOR, CML_VERSION_PATCH) >= CML_VERSION_ENCODE(0, 0, 2)
 */
#define CML_VERSION_ENCODE(major, minor, patch) \
    ((major) * 10000 + (minor) * 100 + (patch))

#define CML_VERSION CML_VERSION_ENCODE(CML_VERSION_MAJOR, CML_VERSION_MINOR, CML_VERSION_PATCH)

/**
 * @brief Get library version as an encoded integer
 * @return Encoded version (major*10000 + minor*100 + patch)
 */
int cml_version(void);

/**
 * @brief Get library version as a string
 * @return Version string (e.g., "0.0.2")
 */
const char* cml_version_string(void);

/**
 * @brief Set the default device for tensor creation
 *
 * When tensors are created without specifying a device, this default
 * will be used. Default is DEVICE_CPU.
 *
 * @param device DeviceType to set as default
 */
void cml_set_default_device(DeviceType device);

/**
 * @brief Get the default device
 *
 * @return Default DeviceType
 */
DeviceType cml_get_default_device(void);

/**
 * @brief Set the default dtype for tensor creation
 *
 * When tensors are created without specifying a dtype, this default
 * will be used. Default is DTYPE_FLOAT32.
 *
 * @param dtype DType to set as default
 */
void cml_set_default_dtype(DType dtype);

/**
 * @brief Get the default dtype
 *
 * @return Default DType
 */
DType cml_get_default_dtype(void);

/**
 * @brief Seed the random number generator
 *
 * Sets the seed for all random number generation in the library.
 * This ensures reproducible results across runs.
 *
 * @param seed Random seed value
 */
void cml_seed(uint64_t seed);

/**
 * @brief Get a random seed based on current time
 *
 * @return Random seed value
 */
uint64_t cml_random_seed(void);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_CONFIG_H
