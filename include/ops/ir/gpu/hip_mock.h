/* Mock HIP driver: validates the ROCm backend and its HCQ adapter without an
 * AMD GPU. See hip_mock.c for semantics. */
#ifndef CML_GPU_HIP_MOCK_H
#define CML_GPU_HIP_MOCK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "ops/ir/gpu/rocm_backend.h"

typedef enum {
    CML_HIP_MOCK_OP_INIT = 0,
    CML_HIP_MOCK_OP_STREAM_CREATE,
    CML_HIP_MOCK_OP_STREAM_SYNC,
    CML_HIP_MOCK_OP_MALLOC,
    CML_HIP_MOCK_OP_FREE,
    CML_HIP_MOCK_OP_MEMCPY_H2D,
    CML_HIP_MOCK_OP_MEMCPY_D2H,
    CML_HIP_MOCK_OP_MODULE_LOAD,
    CML_HIP_MOCK_OP_MODULE_GETFN,
    CML_HIP_MOCK_OP_LAUNCH,
    CML_HIP_MOCK_OP_EVENT_CREATE,
    CML_HIP_MOCK_OP_EVENT_RECORD,
    CML_HIP_MOCK_OP_EVENT_SYNC,
} CMLHIPMockOpKind;

typedef struct {
    CMLHIPMockOpKind kind;
    size_t bytes;             /* memcpy/malloc payload */
    uint32_t launches_grid_x; /* grid X of the last launch */
    char name[32];            /* kernel name where applicable */
} CMLHIPMockEntry;

/* Initialize `backend` fully against the mock driver (no dlopen). */
int cml_rocm_backend_init_mock(CMLROCmBackend* backend);

/* True when CML_HIP_MOCK=1 is set in the environment. */
bool cml_hip_mock_enabled_from_env(void);

/* Journal inspection for tests. */
void cml_hip_mock_reset(void);
int cml_hip_mock_journal_len(void);
const CMLHIPMockEntry* cml_hip_mock_journal_at(int i);
int cml_hip_mock_outstanding_allocs(void); /* device buffers not yet freed */
uint64_t cml_hip_mock_launches(void);

#endif /* CML_GPU_HIP_MOCK_H */
