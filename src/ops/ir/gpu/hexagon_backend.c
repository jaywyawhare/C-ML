/**
 * @file hexagon_backend.c
 * @brief Qualcomm Hexagon DSP backend (stub)
 */

#include "ops/ir/gpu/hexagon_backend.h"
#include <stdlib.h>

bool cml_hexagon_available(void) {
    return false;
}

CMLHexagonBackend* cml_hexagon_backend_create(void) {
    return (CMLHexagonBackend*)calloc(1, sizeof(CMLHexagonBackend));
}

int cml_hexagon_backend_init(CMLHexagonBackend* backend) {
    if (!backend) return -1;
    if (!cml_hexagon_available()) return -1;
    backend->initialized = true;
    return 0;
}

void cml_hexagon_backend_free(CMLHexagonBackend* backend) {
    free(backend);
}

int cml_hexagon_execute(CMLHexagonBackend* backend, CMLGraph_t ir) {
    (void)ir;
    if (!backend || !backend->initialized) return -1;
    return -1;
}
