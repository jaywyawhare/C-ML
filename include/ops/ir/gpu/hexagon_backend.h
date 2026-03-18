/*
 * Qualcomm Hexagon DSP backend.
 * DSP acceleration for efficient quantized inference on mobile devices.
 */

#ifndef CML_HEXAGON_BACKEND_H
#define CML_HEXAGON_BACKEND_H

#include "ops/ir/ir.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLHexagonBackend {
    bool initialized;
    int dsp_version;          /* e.g., V68, V69, V73 */
    int hvx_length;           /* HVX vector length (128 bytes) */
    bool has_hmx;             /* Has Hexagon Matrix eXtensions */
    void* handle;             /* DSP runtime handle */
} CMLHexagonBackend;

bool cml_hexagon_available(void);
CMLHexagonBackend* cml_hexagon_backend_create(void);
int cml_hexagon_backend_init(CMLHexagonBackend* backend);
void cml_hexagon_backend_free(CMLHexagonBackend* backend);
int cml_hexagon_execute(CMLHexagonBackend* backend, CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif /* CML_HEXAGON_BACKEND_H */
