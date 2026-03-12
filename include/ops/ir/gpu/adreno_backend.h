/**
 * @file adreno_backend.h
 * @brief Qualcomm Adreno GPU backend
 *
 * Provides GPU acceleration via Qualcomm's Adreno GPU using OpenCL/Vulkan.
 * Targets Snapdragon SoCs for mobile inference.
 */

#ifndef CML_ADRENO_BACKEND_H
#define CML_ADRENO_BACKEND_H

#include "ops/ir/ir.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLAdrenoBackend {
    bool initialized;
    int gpu_version;          /* e.g., 730, 740, 750 */
    size_t global_mem_size;
    size_t max_alloc_size;
    int max_work_group_size;
    int compute_units;
    void* cl_context;         /* OpenCL context (optional) */
    void* cl_queue;           /* OpenCL command queue */
} CMLAdrenoBackend;

/** Check if Adreno GPU is available */
bool cml_adreno_available(void);

/** Create Adreno backend */
CMLAdrenoBackend* cml_adreno_backend_create(void);

/** Initialize Adreno backend */
int cml_adreno_backend_init(CMLAdrenoBackend* backend);

/** Free Adreno backend */
void cml_adreno_backend_free(CMLAdrenoBackend* backend);

/** Execute IR graph on Adreno GPU */
int cml_adreno_execute(CMLAdrenoBackend* backend, CMLGraph_t ir);

/** Get device info string */
const char* cml_adreno_device_info(const CMLAdrenoBackend* backend);

#ifdef __cplusplus
}
#endif

#endif /* CML_ADRENO_BACKEND_H */
