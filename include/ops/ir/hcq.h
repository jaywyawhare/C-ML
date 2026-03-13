/**
 * @file hcq.h
 * @brief Hardware Command Queues — unified async execution abstraction
 *
 * Queue / signal / fence wrapping CUDA streams, OpenCL queues, and future
 * Metal / WebGPU queues.
 */

#ifndef CML_OPS_IR_HCQ_H
#define CML_OPS_IR_HCQ_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Backend types ── */

typedef enum {
    CML_HCQ_CPU = 0,
    CML_HCQ_CUDA,
    CML_HCQ_OPENCL,
    CML_HCQ_METAL,
    CML_HCQ_WEBGPU,
    CML_HCQ_VULKAN,
    CML_HCQ_NV,
    CML_HCQ_AM,
    CML_HCQ_BACKEND_COUNT
} CMLHCQBackendType;

/* ── Signal: timeline / event ── */

typedef struct CMLHCQSignal {
    CMLHCQBackendType backend;
    uint64_t timeline_value;
    void* native_handle;   /* cl_event, CUevent, etc. */
    bool signaled;
} CMLHCQSignal;

/* ── Queue ── */

#define CML_HCQ_MAX_WAIT_SIGNALS 8

typedef struct CMLHCQQueue {
    CMLHCQBackendType backend;
    void* native_handle;   /* CUstream, cl_command_queue, etc. */
    CMLHCQSignal* wait_signals[CML_HCQ_MAX_WAIT_SIGNALS];
    int num_wait_signals;
    bool active;
} CMLHCQQueue;

/* ── Kernel descriptor (for submit) ── */

typedef struct {
    void* compiled_kernel;  /* backend-specific compiled kernel */
    size_t grid[3];
    size_t block[3];
    void** args;
    int num_args;
} CMLHCQKernelDesc;

/* ── Pipeline: ordered sequence of queues ── */

#define CML_HCQ_MAX_STAGES 8

typedef struct {
    CMLHCQQueue* stages[CML_HCQ_MAX_STAGES];
    CMLHCQSignal* stage_signals[CML_HCQ_MAX_STAGES];
    int num_stages;
} CMLHCQPipeline;

/* ── Queue lifecycle ── */

CMLHCQQueue* cml_hcq_queue_create(CMLHCQBackendType backend);
void cml_hcq_queue_destroy(CMLHCQQueue* queue);

/* ── Kernel + memcpy ── */

int cml_hcq_submit_kernel(CMLHCQQueue* queue, const CMLHCQKernelDesc* desc);
int cml_hcq_memcpy_h2d(CMLHCQQueue* queue, void* dst_device, const void* src_host, size_t bytes);
int cml_hcq_memcpy_d2h(CMLHCQQueue* queue, void* dst_host, const void* src_device, size_t bytes);

/* ── Signals ── */

CMLHCQSignal* cml_hcq_signal_create(CMLHCQBackendType backend);
void cml_hcq_signal_destroy(CMLHCQSignal* signal);
int cml_hcq_signal_record(CMLHCQQueue* queue, CMLHCQSignal* signal);
int cml_hcq_queue_wait(CMLHCQQueue* queue, CMLHCQSignal* signal);
int cml_hcq_signal_wait_cpu(CMLHCQSignal* signal, uint64_t timeout_ms);

/* ── Sync ── */

int cml_hcq_queue_synchronize(CMLHCQQueue* queue);

/* ── Pipeline ── */

CMLHCQPipeline* cml_hcq_pipeline_create(void);
void cml_hcq_pipeline_destroy(CMLHCQPipeline* pipeline);
int cml_hcq_pipeline_add_stage(CMLHCQPipeline* pipeline, CMLHCQQueue* queue);
int cml_hcq_pipeline_execute(CMLHCQPipeline* pipeline);
int cml_hcq_pipeline_synchronize(CMLHCQPipeline* pipeline);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_HCQ_H */
