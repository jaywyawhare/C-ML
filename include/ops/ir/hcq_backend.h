#ifndef CML_OPS_IR_HCQ_BACKEND_H
#define CML_OPS_IR_HCQ_BACKEND_H

#include "ops/ir/hcq.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLHCQBackendOps {
    const char* name;
    CMLHCQQueue* (*queue_create)(void);
    void (*queue_destroy)(CMLHCQQueue* queue);
    int (*submit_kernel)(CMLHCQQueue* queue, const CMLHCQKernelDesc* desc);
    int (*memcpy_h2d)(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes);
    int (*memcpy_d2h)(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes);
    int (*queue_synchronize)(CMLHCQQueue* queue);
    CMLHCQSignal* (*signal_create)(void);
    void (*signal_destroy)(CMLHCQSignal* signal);
    int (*signal_record)(CMLHCQQueue* queue, CMLHCQSignal* signal);
    int (*queue_wait)(CMLHCQQueue* queue, CMLHCQSignal* signal);
    int (*signal_wait_cpu)(CMLHCQSignal* signal, uint64_t timeout_ms);
} CMLHCQBackendOps;

const CMLHCQBackendOps* cml_hcq_backend_ops(CMLHCQBackendType backend);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_HCQ_BACKEND_H */
