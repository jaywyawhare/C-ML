/*
 * Qualcomm Adreno GPU backend.
 * GPU acceleration via OpenCL/Vulkan on Snapdragon SoCs for mobile inference.
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

bool cml_adreno_available(void);
CMLAdrenoBackend* cml_adreno_backend_create(void);
int cml_adreno_backend_init(CMLAdrenoBackend* backend);
void cml_adreno_backend_free(CMLAdrenoBackend* backend);
int cml_adreno_execute(CMLAdrenoBackend* backend, CMLGraph_t ir);
const char* cml_adreno_device_info(const CMLAdrenoBackend* backend);

#ifdef __cplusplus
}
#endif

#endif /* CML_ADRENO_BACKEND_H */
