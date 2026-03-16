/**
 * @file hcq.c
 * @brief Hardware Command Queues -- CPU/generic fallback implementation
 *
 * This is the portable fallback that uses synchronous execution for the CPU
 * backend.  For GPU backends (CUDA, OpenCL, ...) the public API dispatches
 * to backend-specific translation units (hcq_cuda.c, hcq_opencl.c, ...).
 */

#include "ops/ir/hcq.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>

#ifdef CML_HAS_CUDA
extern CMLHCQQueue*  cml_hcq_cuda_queue_create(void);
extern void          cml_hcq_cuda_queue_destroy(CMLHCQQueue* queue);
extern int           cml_hcq_cuda_submit_kernel(CMLHCQQueue* queue,
                                                const CMLHCQKernelDesc* desc);
extern int           cml_hcq_cuda_memcpy_h2d(CMLHCQQueue* queue, void* dst,
                                             const void* src, size_t bytes);
extern int           cml_hcq_cuda_memcpy_d2h(CMLHCQQueue* queue, void* dst,
                                             const void* src, size_t bytes);
extern CMLHCQSignal* cml_hcq_cuda_signal_create(void);
extern void          cml_hcq_cuda_signal_destroy(CMLHCQSignal* signal);
extern int           cml_hcq_cuda_signal_record(CMLHCQQueue* queue,
                                                CMLHCQSignal* signal);
extern int           cml_hcq_cuda_queue_wait(CMLHCQQueue* queue,
                                             CMLHCQSignal* signal);
extern int           cml_hcq_cuda_signal_wait_cpu(CMLHCQSignal* signal,
                                                  uint64_t timeout_ms);
extern int           cml_hcq_cuda_queue_synchronize(CMLHCQQueue* queue);
#endif

#ifdef CML_HAS_OPENCL
extern CMLHCQQueue*  cml_hcq_opencl_queue_create(void);
extern void          cml_hcq_opencl_queue_destroy(CMLHCQQueue* queue);
extern int           cml_hcq_opencl_submit_kernel(CMLHCQQueue* queue,
                                                  const CMLHCQKernelDesc* desc);
extern int           cml_hcq_opencl_memcpy_h2d(CMLHCQQueue* queue, void* dst,
                                               const void* src, size_t bytes);
extern int           cml_hcq_opencl_memcpy_d2h(CMLHCQQueue* queue, void* dst,
                                               const void* src, size_t bytes);
extern CMLHCQSignal* cml_hcq_opencl_signal_create(void);
extern void          cml_hcq_opencl_signal_destroy(CMLHCQSignal* signal);
extern int           cml_hcq_opencl_signal_record(CMLHCQQueue* queue,
                                                  CMLHCQSignal* signal);
extern int           cml_hcq_opencl_queue_wait(CMLHCQQueue* queue,
                                               CMLHCQSignal* signal);
extern int           cml_hcq_opencl_signal_wait_cpu(CMLHCQSignal* signal,
                                                    uint64_t timeout_ms);
extern int           cml_hcq_opencl_queue_synchronize(CMLHCQQueue* queue);
#endif

/*
 * CPU kernel function pointer type.
 * On the CPU backend the compiled_kernel field of CMLHCQKernelDesc is cast
 * to this signature.  The function receives the args array and arg count
 * and is expected to execute synchronously.
 */
typedef void (*cml_cpu_kernel_fn)(void** args, int num_args);

CMLHCQQueue* cml_hcq_queue_create(CMLHCQBackendType backend) {
    /* Dispatch to backend-specific creation when available. */
    switch (backend) {
#ifdef CML_HAS_CUDA
    case CML_HCQ_CUDA:
        return cml_hcq_cuda_queue_create();
#endif
#ifdef CML_HAS_OPENCL
    case CML_HCQ_OPENCL:
        return cml_hcq_opencl_queue_create();
#endif
    case CML_HCQ_CPU:
        break; /* handled below */
    default:
        LOG_ERROR("Unsupported HCQ backend type: %d", (int)backend);
        return NULL;
    }

    /* CPU fallback */
    CMLHCQQueue* queue = (CMLHCQQueue*)calloc(1, sizeof(CMLHCQQueue));
    if (!queue) {
        LOG_ERROR("Failed to allocate CMLHCQQueue");
        return NULL;
    }
    queue->backend          = CML_HCQ_CPU;
    queue->native_handle    = NULL;
    queue->num_wait_signals = 0;
    queue->active           = true;
    return queue;
}

void cml_hcq_queue_destroy(CMLHCQQueue* queue) {
    if (!queue) return;

    switch (queue->backend) {
#ifdef CML_HAS_CUDA
    case CML_HCQ_CUDA:
        cml_hcq_cuda_queue_destroy(queue);
        return;
#endif
#ifdef CML_HAS_OPENCL
    case CML_HCQ_OPENCL:
        cml_hcq_opencl_queue_destroy(queue);
        return;
#endif
    default:
        break;
    }
    queue->active = false;
    free(queue);
}

int cml_hcq_submit_kernel(CMLHCQQueue* queue, const CMLHCQKernelDesc* desc) {
    if (!queue || !desc) {
        LOG_ERROR("NULL queue or descriptor in submit_kernel");
        return -1;
    }

    switch (queue->backend) {
#ifdef CML_HAS_CUDA
    case CML_HCQ_CUDA:
        return cml_hcq_cuda_submit_kernel(queue, desc);
#endif
#ifdef CML_HAS_OPENCL
    case CML_HCQ_OPENCL:
        return cml_hcq_opencl_submit_kernel(queue, desc);
#endif
    case CML_HCQ_CPU:
        break;
    default:
        LOG_ERROR("Unsupported backend %d for submit_kernel", (int)queue->backend);
        return -1;
    }

    /* CPU path: cast compiled_kernel to a function pointer and call it. */
    if (!desc->compiled_kernel) {
        LOG_ERROR("CPU kernel function pointer is NULL");
        return -1;
    }

    cml_cpu_kernel_fn fn = (cml_cpu_kernel_fn)desc->compiled_kernel;
    fn(desc->args, desc->num_args);
    return 0;
}

int cml_hcq_memcpy_h2d(CMLHCQQueue* queue, void* dst_device,
                        const void* src_host, size_t bytes) {
    if (!queue) {
        LOG_ERROR("NULL queue in memcpy_h2d");
        return -1;
    }

    switch (queue->backend) {
#ifdef CML_HAS_CUDA
    case CML_HCQ_CUDA:
        return cml_hcq_cuda_memcpy_h2d(queue, dst_device, src_host, bytes);
#endif
#ifdef CML_HAS_OPENCL
    case CML_HCQ_OPENCL:
        return cml_hcq_opencl_memcpy_h2d(queue, dst_device, src_host, bytes);
#endif
    case CML_HCQ_CPU:
        break;
    default:
        LOG_ERROR("Unsupported backend %d for memcpy_h2d", (int)queue->backend);
        return -1;
    }

    /* CPU: host and device memory are the same address space. */
    if (!dst_device || !src_host) {
        LOG_ERROR("NULL pointer in CPU memcpy_h2d");
        return -1;
    }
    memcpy(dst_device, src_host, bytes);
    return 0;
}

int cml_hcq_memcpy_d2h(CMLHCQQueue* queue, void* dst_host,
                        const void* src_device, size_t bytes) {
    if (!queue) {
        LOG_ERROR("NULL queue in memcpy_d2h");
        return -1;
    }

    switch (queue->backend) {
#ifdef CML_HAS_CUDA
    case CML_HCQ_CUDA:
        return cml_hcq_cuda_memcpy_d2h(queue, dst_host, src_device, bytes);
#endif
#ifdef CML_HAS_OPENCL
    case CML_HCQ_OPENCL:
        return cml_hcq_opencl_memcpy_d2h(queue, dst_host, src_device, bytes);
#endif
    case CML_HCQ_CPU:
        break;
    default:
        LOG_ERROR("Unsupported backend %d for memcpy_d2h", (int)queue->backend);
        return -1;
    }

    if (!dst_host || !src_device) {
        LOG_ERROR("NULL pointer in CPU memcpy_d2h");
        return -1;
    }
    memcpy(dst_host, src_device, bytes);
    return 0;
}

CMLHCQSignal* cml_hcq_signal_create(CMLHCQBackendType backend) {
    switch (backend) {
#ifdef CML_HAS_CUDA
    case CML_HCQ_CUDA:
        return cml_hcq_cuda_signal_create();
#endif
#ifdef CML_HAS_OPENCL
    case CML_HCQ_OPENCL:
        return cml_hcq_opencl_signal_create();
#endif
    case CML_HCQ_CPU:
        break;
    default:
        LOG_ERROR("Unsupported HCQ backend type %d for signal", (int)backend);
        return NULL;
    }

    CMLHCQSignal* signal = (CMLHCQSignal*)calloc(1, sizeof(CMLHCQSignal));
    if (!signal) {
        LOG_ERROR("Failed to allocate CMLHCQSignal");
        return NULL;
    }
    signal->backend        = CML_HCQ_CPU;
    signal->timeline_value = 0;
    signal->native_handle  = NULL;
    signal->signaled       = false;
    return signal;
}

void cml_hcq_signal_destroy(CMLHCQSignal* signal) {
    if (!signal) return;

    switch (signal->backend) {
#ifdef CML_HAS_CUDA
    case CML_HCQ_CUDA:
        cml_hcq_cuda_signal_destroy(signal);
        return;
#endif
#ifdef CML_HAS_OPENCL
    case CML_HCQ_OPENCL:
        cml_hcq_opencl_signal_destroy(signal);
        return;
#endif
    default:
        break;
    }
    free(signal);
}

int cml_hcq_signal_record(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    if (!queue || !signal) {
        LOG_ERROR("NULL queue or signal in signal_record");
        return -1;
    }

    switch (queue->backend) {
#ifdef CML_HAS_CUDA
    case CML_HCQ_CUDA:
        return cml_hcq_cuda_signal_record(queue, signal);
#endif
#ifdef CML_HAS_OPENCL
    case CML_HCQ_OPENCL:
        return cml_hcq_opencl_signal_record(queue, signal);
#endif
    case CML_HCQ_CPU:
        break;
    default:
        LOG_ERROR("Unsupported backend %d for signal_record", (int)queue->backend);
        return -1;
    }

    /* CPU: everything is synchronous, so the signal is immediately ready. */
    signal->signaled = true;
    signal->timeline_value++;
    return 0;
}

int cml_hcq_queue_wait(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    if (!queue || !signal) {
        LOG_ERROR("NULL queue or signal in queue_wait");
        return -1;
    }

    switch (queue->backend) {
#ifdef CML_HAS_CUDA
    case CML_HCQ_CUDA:
        return cml_hcq_cuda_queue_wait(queue, signal);
#endif
#ifdef CML_HAS_OPENCL
    case CML_HCQ_OPENCL:
        return cml_hcq_opencl_queue_wait(queue, signal);
#endif
    case CML_HCQ_CPU:
        break;
    default:
        LOG_ERROR("Unsupported backend %d for queue_wait", (int)queue->backend);
        return -1;
    }

    /*
     * CPU: execution is synchronous so if signal_record was called it is
     * already signaled.  We just verify and add it to the wait list for
     * bookkeeping.
     */
    if (!signal->signaled) {
        LOG_WARNING("CPU HCQ: queue_wait on unsignaled signal %p -- "
                    "this should not happen in synchronous mode",
                    (void*)signal);
    }

    if (queue->num_wait_signals < CML_HCQ_MAX_WAIT_SIGNALS) {
        queue->wait_signals[queue->num_wait_signals++] = signal;
    }
    return 0;
}

int cml_hcq_signal_wait_cpu(CMLHCQSignal* signal, uint64_t timeout_ms) {
    if (!signal) {
        LOG_ERROR("NULL signal in signal_wait_cpu");
        return -1;
    }

    switch (signal->backend) {
#ifdef CML_HAS_CUDA
    case CML_HCQ_CUDA:
        return cml_hcq_cuda_signal_wait_cpu(signal, timeout_ms);
#endif
#ifdef CML_HAS_OPENCL
    case CML_HCQ_OPENCL:
        return cml_hcq_opencl_signal_wait_cpu(signal, timeout_ms);
#endif
    case CML_HCQ_CPU:
        break;
    default:
        LOG_ERROR("Unsupported backend %d for signal_wait_cpu",
                  (int)signal->backend);
        return -1;
    }

    /*
     * CPU: everything completes synchronously so the signal is either
     * already set or something is wrong.  No busy-wait needed.
     */
    (void)timeout_ms;
    if (!signal->signaled) {
        LOG_WARNING("CPU HCQ: signal_wait_cpu on unsignaled signal %p",
                    (void*)signal);
        return -1;
    }
    return 0;
}

int cml_hcq_queue_synchronize(CMLHCQQueue* queue) {
    if (!queue) {
        LOG_ERROR("NULL queue in queue_synchronize");
        return -1;
    }

    switch (queue->backend) {
#ifdef CML_HAS_CUDA
    case CML_HCQ_CUDA:
        return cml_hcq_cuda_queue_synchronize(queue);
#endif
#ifdef CML_HAS_OPENCL
    case CML_HCQ_OPENCL:
        return cml_hcq_opencl_queue_synchronize(queue);
#endif
    case CML_HCQ_CPU:
        break;
    default:
        LOG_ERROR("Unsupported backend %d for queue_synchronize",
                  (int)queue->backend);
        return -1;
    }

    /* CPU: synchronous -- nothing to wait for. */
    queue->num_wait_signals = 0;
    return 0;
}

CMLHCQPipeline* cml_hcq_pipeline_create(void) {
    CMLHCQPipeline* pipeline =
        (CMLHCQPipeline*)calloc(1, sizeof(CMLHCQPipeline));
    if (!pipeline) {
        LOG_ERROR("Failed to allocate CMLHCQPipeline");
        return NULL;
    }
    pipeline->num_stages = 0;
    LOG_DEBUG("Created HCQ pipeline %p", (void*)pipeline);
    return pipeline;
}

void cml_hcq_pipeline_destroy(CMLHCQPipeline* pipeline) {
    if (!pipeline) return;

    /* Destroy the inter-stage signals that were created during add_stage. */
    for (int i = 0; i < pipeline->num_stages; i++) {
        if (pipeline->stage_signals[i]) {
            cml_hcq_signal_destroy(pipeline->stage_signals[i]);
            pipeline->stage_signals[i] = NULL;
        }
    }

    LOG_DEBUG("Destroying HCQ pipeline %p", (void*)pipeline);
    free(pipeline);
}

int cml_hcq_pipeline_add_stage(CMLHCQPipeline* pipeline, CMLHCQQueue* queue) {
    if (!pipeline || !queue) {
        LOG_ERROR("NULL pipeline or queue in pipeline_add_stage");
        return -1;
    }
    if (pipeline->num_stages >= CML_HCQ_MAX_STAGES) {
        LOG_ERROR("Pipeline has reached maximum number of stages (%d)",
                  CML_HCQ_MAX_STAGES);
        return -1;
    }

    int idx = pipeline->num_stages;
    pipeline->stages[idx] = queue;

    /* Create an inter-stage signal so the next stage can wait on this one. */
    CMLHCQSignal* sig = cml_hcq_signal_create(queue->backend);
    if (!sig) {
        LOG_ERROR("Failed to create inter-stage signal for pipeline stage %d",
                  idx);
        return -1;
    }
    pipeline->stage_signals[idx] = sig;
    pipeline->num_stages++;

    LOG_DEBUG("Pipeline %p: added stage %d (queue %p, signal %p)",
              (void*)pipeline, idx, (void*)queue, (void*)sig);
    return 0;
}

int cml_hcq_pipeline_execute(CMLHCQPipeline* pipeline) {
    if (!pipeline) {
        LOG_ERROR("NULL pipeline in pipeline_execute");
        return -1;
    }
    if (pipeline->num_stages == 0) {
        LOG_WARNING("Pipeline has no stages to execute");
        return 0;
    }
    for (int i = 0; i < pipeline->num_stages; i++) {
        CMLHCQQueue*  stage  = pipeline->stages[i];
        CMLHCQSignal* signal = pipeline->stage_signals[i];

        /* If not the first stage, wait on the previous stage's signal. */
        if (i > 0) {
            CMLHCQSignal* prev_signal = pipeline->stage_signals[i - 1];
            int ret = cml_hcq_queue_wait(stage, prev_signal);
            if (ret != 0) {
                LOG_ERROR("Pipeline stage %d failed to wait on stage %d signal",
                          i, i - 1);
                return -1;
            }
        }

        /* Record signal so the next stage (or synchronize) can wait on it. */
        int ret = cml_hcq_signal_record(stage, signal);
        if (ret != 0) {
            LOG_ERROR("Pipeline stage %d failed to record signal", i);
            return -1;
        }
    }

    LOG_DEBUG("Pipeline %p: all stages executed", (void*)pipeline);
    return 0;
}

int cml_hcq_pipeline_synchronize(CMLHCQPipeline* pipeline) {
    if (!pipeline) {
        LOG_ERROR("NULL pipeline in pipeline_synchronize");
        return -1;
    }
    if (pipeline->num_stages == 0) {
        return 0;
    }

    /* Wait on the last stage's signal to ensure everything has completed. */
    CMLHCQSignal* last = pipeline->stage_signals[pipeline->num_stages - 1];
    int ret = cml_hcq_signal_wait_cpu(last, 0);
    if (ret != 0) {
        LOG_ERROR("Pipeline synchronize failed on last stage signal");
        return ret;
    }

    /* Also synchronize each queue for good measure. */
    for (int i = 0; i < pipeline->num_stages; i++) {
        ret = cml_hcq_queue_synchronize(pipeline->stages[i]);
        if (ret != 0) {
            LOG_ERROR("Pipeline synchronize failed on stage %d queue", i);
            return ret;
        }
    }
    return 0;
}
