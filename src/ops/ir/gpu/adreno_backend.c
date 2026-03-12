/**
 * @file adreno_backend.c
 * @brief Qualcomm Adreno GPU backend (stub)
 */

#include "ops/ir/gpu/adreno_backend.h"
#include <stdlib.h>
#include <string.h>

bool cml_adreno_available(void) {
    /* Would check for Adreno GPU via OpenCL/Vulkan device enumeration */
    return false;
}

CMLAdrenoBackend* cml_adreno_backend_create(void) {
    CMLAdrenoBackend* b = (CMLAdrenoBackend*)calloc(1, sizeof(CMLAdrenoBackend));
    return b;
}

int cml_adreno_backend_init(CMLAdrenoBackend* backend) {
    if (!backend) return -1;
    if (!cml_adreno_available()) return -1;
    backend->initialized = true;
    return 0;
}

void cml_adreno_backend_free(CMLAdrenoBackend* backend) {
    free(backend);
}

int cml_adreno_execute(CMLAdrenoBackend* backend, CMLGraph_t ir) {
    (void)ir;
    if (!backend || !backend->initialized) return -1;
    return -1; /* Not yet implemented */
}

const char* cml_adreno_device_info(const CMLAdrenoBackend* backend) {
    (void)backend;
    return "Adreno GPU (not available)";
}
