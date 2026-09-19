#include "ops/ir/hcq.h"
#include "ops/ir/hcq_backend.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include "alloc/cml_allocator.h"

/*
 * Every backend — including CPU — is a first-class entry in the
 * CMLHCQBackendOps table (see hcq_backend.c). These thin dispatchers just look
 * up the ops for a queue/signal's backend and forward; there are no per-backend
 * special-cases here anymore.
 */

CMLHCQQueue* cml_hcq_queue_create(CMLHCQBackendType backend) {
    const CMLHCQBackendOps* ops = cml_hcq_backend_ops(backend);
    if (ops && ops->queue_create)
        return ops->queue_create();
    LOG_ERROR("Unsupported HCQ backend type: %d", (int)backend);
    return NULL;
}

void cml_hcq_queue_destroy(CMLHCQQueue* queue) {
    if (!queue)
        return;

    /* Read backend before destroy — the ops may free the queue wrapper. */
    CMLHCQBackendType backend   = queue->backend;
    const CMLHCQBackendOps* ops = cml_hcq_backend_ops(backend);
    if (ops && ops->queue_destroy) {
        ops->queue_destroy(queue);
        /* Vulkan/AM tear down the native handle but leave the wrapper to us. */
        if (backend == CML_HCQ_VULKAN || backend == CML_HCQ_AM)
            free(queue);
        return;
    }

    queue->active = false;
    cml_free(queue);
}

int cml_hcq_submit_kernel(CMLHCQQueue* queue, const CMLHCQKernelDesc* desc) {
    if (!queue || !desc) {
        LOG_ERROR("NULL queue or descriptor in submit_kernel");
        return -1;
    }
    const CMLHCQBackendOps* ops = cml_hcq_backend_ops(queue->backend);
    if (ops && ops->submit_kernel)
        return ops->submit_kernel(queue, desc);
    LOG_ERROR("Unsupported backend %d for submit_kernel", (int)queue->backend);
    return -1;
}

int cml_hcq_memcpy_h2d(CMLHCQQueue* queue, void* dst_device, const void* src_host, size_t bytes) {
    if (!queue) {
        LOG_ERROR("NULL queue in memcpy_h2d");
        return -1;
    }
    const CMLHCQBackendOps* ops = cml_hcq_backend_ops(queue->backend);
    if (ops && ops->memcpy_h2d)
        return ops->memcpy_h2d(queue, dst_device, src_host, bytes);
    LOG_ERROR("Unsupported backend %d for memcpy_h2d", (int)queue->backend);
    return -1;
}

int cml_hcq_memcpy_d2h(CMLHCQQueue* queue, void* dst_host, const void* src_device, size_t bytes) {
    if (!queue) {
        LOG_ERROR("NULL queue in memcpy_d2h");
        return -1;
    }
    const CMLHCQBackendOps* ops = cml_hcq_backend_ops(queue->backend);
    if (ops && ops->memcpy_d2h)
        return ops->memcpy_d2h(queue, dst_host, src_device, bytes);
    LOG_ERROR("Unsupported backend %d for memcpy_d2h", (int)queue->backend);
    return -1;
}

CMLHCQSignal* cml_hcq_signal_create(CMLHCQBackendType backend) {
    const CMLHCQBackendOps* ops = cml_hcq_backend_ops(backend);
    if (ops && ops->signal_create)
        return ops->signal_create();
    LOG_ERROR("Unsupported HCQ backend type %d for signal", (int)backend);
    return NULL;
}

void cml_hcq_signal_destroy(CMLHCQSignal* signal) {
    if (!signal)
        return;

    CMLHCQBackendType backend   = signal->backend;
    const CMLHCQBackendOps* ops = cml_hcq_backend_ops(backend);
    if (ops && ops->signal_destroy) {
        ops->signal_destroy(signal);
        if (backend == CML_HCQ_VULKAN || backend == CML_HCQ_AM)
            cml_free(signal);
        return;
    }

    cml_free(signal);
}

int cml_hcq_signal_record(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    if (!queue || !signal) {
        LOG_ERROR("NULL queue or signal in signal_record");
        return -1;
    }
    const CMLHCQBackendOps* ops = cml_hcq_backend_ops(queue->backend);
    if (ops && ops->signal_record)
        return ops->signal_record(queue, signal);
    LOG_ERROR("Unsupported backend %d for signal_record", (int)queue->backend);
    return -1;
}

int cml_hcq_queue_wait(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    if (!queue || !signal) {
        LOG_ERROR("NULL queue or signal in queue_wait");
        return -1;
    }
    const CMLHCQBackendOps* ops = cml_hcq_backend_ops(queue->backend);
    if (ops && ops->queue_wait)
        return ops->queue_wait(queue, signal);
    LOG_ERROR("Unsupported backend %d for queue_wait", (int)queue->backend);
    return -1;
}

int cml_hcq_signal_wait_cpu(CMLHCQSignal* signal, uint64_t timeout_ms) {
    if (!signal) {
        LOG_ERROR("NULL signal in signal_wait_cpu");
        return -1;
    }
    const CMLHCQBackendOps* ops = cml_hcq_backend_ops(signal->backend);
    if (ops && ops->signal_wait_cpu)
        return ops->signal_wait_cpu(signal, timeout_ms);
    LOG_ERROR("Unsupported backend %d for signal_wait_cpu", (int)signal->backend);
    return -1;
}

int cml_hcq_queue_synchronize(CMLHCQQueue* queue) {
    if (!queue) {
        LOG_ERROR("NULL queue in queue_synchronize");
        return -1;
    }
    const CMLHCQBackendOps* ops = cml_hcq_backend_ops(queue->backend);
    if (ops && ops->queue_synchronize)
        return ops->queue_synchronize(queue);
    LOG_ERROR("Unsupported backend %d for queue_synchronize", (int)queue->backend);
    return -1;
}

CMLHCQPipeline* cml_hcq_pipeline_create(void) {
    CMLHCQPipeline* pipeline = (CMLHCQPipeline*)cml_calloc(1, sizeof(CMLHCQPipeline));
    if (!pipeline) {
        LOG_ERROR("Failed to allocate CMLHCQPipeline");
        return NULL;
    }
    pipeline->num_stages = 0;
    LOG_DEBUG("Created HCQ pipeline %p", (void*)pipeline);
    return pipeline;
}

void cml_hcq_pipeline_destroy(CMLHCQPipeline* pipeline) {
    if (!pipeline)
        return;

    /* Destroy the inter-stage signals that were created during add_stage. */
    for (int i = 0; i < pipeline->num_stages; i++) {
        if (pipeline->stage_signals[i]) {
            cml_hcq_signal_destroy(pipeline->stage_signals[i]);
            pipeline->stage_signals[i] = NULL;
        }
    }

    LOG_DEBUG("Destroying HCQ pipeline %p", (void*)pipeline);
    cml_free(pipeline);
}

int cml_hcq_pipeline_add_stage(CMLHCQPipeline* pipeline, CMLHCQQueue* queue) {
    if (!pipeline || !queue) {
        LOG_ERROR("NULL pipeline or queue in pipeline_add_stage");
        return -1;
    }
    if (pipeline->num_stages >= CML_HCQ_MAX_STAGES) {
        LOG_ERROR("Pipeline has reached maximum number of stages (%d)", CML_HCQ_MAX_STAGES);
        return -1;
    }

    int idx               = pipeline->num_stages;
    pipeline->stages[idx] = queue;

    /* Create an inter-stage signal so the next stage can wait on this one. */
    CMLHCQSignal* sig = cml_hcq_signal_create(queue->backend);
    if (!sig) {
        LOG_ERROR("Failed to create inter-stage signal for pipeline stage %d", idx);
        return -1;
    }
    pipeline->stage_signals[idx] = sig;
    pipeline->num_stages++;

    LOG_DEBUG("Pipeline %p: added stage %d (queue %p, signal %p)", (void*)pipeline, idx,
              (void*)queue, (void*)sig);
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
        CMLHCQQueue* stage   = pipeline->stages[i];
        CMLHCQSignal* signal = pipeline->stage_signals[i];

        /* If not the first stage, wait on the previous stage's signal. */
        if (i > 0) {
            CMLHCQSignal* prev_signal = pipeline->stage_signals[i - 1];
            int ret                   = cml_hcq_queue_wait(stage, prev_signal);
            if (ret != 0) {
                LOG_ERROR("Pipeline stage %d failed to wait on stage %d signal", i, i - 1);
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
    int ret            = cml_hcq_signal_wait_cpu(last, 0);
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
