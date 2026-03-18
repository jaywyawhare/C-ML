#ifndef CML_CORE_CONFIG_H
#define CML_CORE_CONFIG_H

#include "backend/device.h"
#include "tensor/tensor.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CML_VERSION_MAJOR 0
#define CML_VERSION_MINOR 0
#define CML_VERSION_PATCH 2
#define CML_VERSION_STRING "0.0.3"

#define CML_VERSION_ENCODE(major, minor, patch) \
    ((major) * 10000 + (minor) * 100 + (patch))

#define CML_VERSION CML_VERSION_ENCODE(CML_VERSION_MAJOR, CML_VERSION_MINOR, CML_VERSION_PATCH)

int cml_version(void);
const char* cml_version_string(void);
void cml_set_default_device(DeviceType device);
DeviceType cml_get_default_device(void);
void cml_set_default_dtype(DType dtype);
DType cml_get_default_dtype(void);
void cml_seed(uint64_t seed);
uint64_t cml_random_seed(void);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_CONFIG_H
