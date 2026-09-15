/* HCQ adapter for the ROCm (HIP) backend.
 *
 * Wraps the cml_rocm_* API (gpu/rocm_backend.c) in the CMLHCQBackendOps
 * contract so ROCm dispatches uniformly through cml_hcq_backend_ops(), like
 * CPU/CUDA/OpenCL/Vulkan/NV/AM. The HIP library is dlopen'd at runtime, so
 * this file compiles everywhere; without libamdhip64 the queue/signal
 * constructors fail gracefully and every operation reports -1.
 *
 * Signals use HIP events when the runtime provides them; otherwise they
 * degrade to stream-synchronize semantics (record == wait-for-submitted-work).
 */
#include "ops/ir/hcq.h"
#include "ops/ir/gpu/rocm_backend.h"
#include "core/logging.h"
#include "alloc/cml_allocator.h"

#ifdef __linux__

typedef struct {
    CMLROCmBackend* backend;
    uint64_t submit_count;
} ROCmQueueData;

typedef struct {
    CMLROCmBackend* backend; /* set at record time; needed for wait/destroy */
    void* event;             /* HIP event, or NULL under sync semantics */
} ROCmSignalData;

static CMLROCmBackend* rocm_backend_of(CMLHCQQueue* queue) {
    ROCmQueueData* qd = (ROCmQueueData*)(queue ? queue->native_handle : NULL);
    return qd ? qd->backend : NULL;
}

/* Public accessor so tests can reach the backend behind a queue (e.g. the
 * HIP mock driver's journal checks). */
CMLROCmBackend* cml_hcq_rocm_queue_backend(CMLHCQQueue* queue) {
    return rocm_backend_of(queue);
}

CMLHCQQueue* cml_hcq_rocm_queue_create(void) {
    CMLROCmBackend* backend = cml_rocm_backend_create();
    if (!backend)
        return NULL;
    if (cml_rocm_backend_init(backend, 0) != 0) {
        cml_rocm_backend_free(backend);
        return NULL;
    }

    CMLHCQQueue* queue = (CMLHCQQueue*)cml_calloc(1, sizeof(CMLHCQQueue));
    ROCmQueueData* qd  = (ROCmQueueData*)cml_calloc(1, sizeof(ROCmQueueData));
    if (!queue || !qd) {
        cml_free(queue);
        cml_free(qd);
        cml_rocm_backend_free(backend);
        return NULL;
    }

    qd->backend = backend;
    queue->backend       = CML_HCQ_ROCM;
    queue->native_handle = qd;
    queue->active        = true;
    return queue;
}

void cml_hcq_rocm_queue_destroy(CMLHCQQueue* queue) {
    if (!queue)
        return;
    ROCmQueueData* qd = (ROCmQueueData*)queue->native_handle;
    if (qd) {
        cml_rocm_backend_free(qd->backend);
        cml_free(qd);
    }
    queue->native_handle = NULL;
    queue->active        = false;
    cml_free(queue);
}

int cml_hcq_rocm_submit_kernel(CMLHCQQueue* queue, const CMLHCQKernelDesc* desc) {
    if (!queue || !desc || !desc->compiled_kernel)
        return -1;
    CMLROCmBackend* backend = rocm_backend_of(queue);
    if (!backend)
        return -1;

    /* The desc carries the launch geometry; the rocm kernel struct keeps its
     * own copy that cml_rocm_launch_kernel reads. */
    CMLROCmKernel* kernel = (CMLROCmKernel*)desc->compiled_kernel;
    for (int i = 0; i < 3; i++) {
        kernel->grid_dim[i]  = (int)desc->grid[i];
        kernel->block_dim[i] = (int)desc->block[i];
    }

    int rc = cml_rocm_launch_kernel(backend, kernel, desc->args, desc->num_args);
    if (rc == 0)
        ((ROCmQueueData*)queue->native_handle)->submit_count++;
    return rc;
}

int cml_hcq_rocm_memcpy_h2d(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes) {
    if (!queue || !dst || !src || bytes == 0)
        return -1;
    return cml_rocm_memcpy_h2d(rocm_backend_of(queue), dst, src, bytes);
}

int cml_hcq_rocm_memcpy_d2h(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes) {
    if (!queue || !dst || !src || bytes == 0)
        return -1;
    /* rocm_backend's d2h predates the const qualifier on the ops contract. */
    return cml_rocm_memcpy_d2h(rocm_backend_of(queue), dst, (void*)(uintptr_t)src, bytes);
}

int cml_hcq_rocm_queue_synchronize(CMLHCQQueue* queue) {
    return cml_rocm_synchronize(rocm_backend_of(queue));
}

CMLHCQSignal* cml_hcq_rocm_signal_create(void) {
    /* Must use the CML allocator: hcq.c frees signal wrappers with cml_free,
     * so system calloc here would corrupt the heap (see the Vulkan adapter). */
    CMLHCQSignal* signal = (CMLHCQSignal*)cml_calloc(1, sizeof(CMLHCQSignal));
    ROCmSignalData* sd   = (ROCmSignalData*)cml_calloc(1, sizeof(ROCmSignalData));
    if (!signal || !sd) {
        cml_free(signal);
        cml_free(sd);
        return NULL;
    }
    signal->backend       = CML_HCQ_ROCM;
    signal->native_handle = sd;
    signal->signaled      = false;
    return signal;
}

void cml_hcq_rocm_signal_destroy(CMLHCQSignal* signal) {
    if (!signal)
        return;
    ROCmSignalData* sd = (ROCmSignalData*)signal->native_handle;
    if (sd) {
        if (sd->event && sd->backend && sd->backend->hipEventDestroy)
            sd->backend->hipEventDestroy(sd->event);
        cml_free(sd);
    }
    signal->native_handle = NULL;
    cml_free(signal);
}

int cml_hcq_rocm_signal_record(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    if (!queue || !signal)
        return -1;
    CMLROCmBackend* backend = rocm_backend_of(queue);
    ROCmSignalData* sd      = (ROCmSignalData*)signal->native_handle;
    if (!backend || !sd)
        return -1;

    if (backend->hipEventCreate && backend->hipEventRecord && !sd->event) {
        void* ev = NULL;
        if (backend->hipEventCreate(&ev) == 0)
            sd->event = ev;
        else
            LOG_WARNING("ROCm HCQ: hipEventCreate failed; using sync semantics");
    }
    sd->backend = backend;

    if (sd->event) {
        if (backend->hipEventRecord(sd->event, backend->stream) != 0)
            return -1;
        signal->signaled = false;
    } else {
        /* No event support: record == wait for all submitted work. */
        if (cml_rocm_synchronize(backend) != 0)
            return -1;
        signal->signaled = true;
    }
    signal->timeline_value++;
    return 0;
}

int cml_hcq_rocm_queue_wait(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    /* A single in-order stream observes prior submissions by construction, so
     * a cross-queue wait has nothing to enqueue; the dependency is already
     * ordered. */
    (void)queue;
    (void)signal;
    return 0;
}

int cml_hcq_rocm_signal_wait_cpu(CMLHCQSignal* signal, uint64_t timeout_ms) {
    (void)timeout_ms; /* hipEventSynchronize blocks without a timeout knob */
    if (!signal)
        return -1;
    ROCmSignalData* sd = (ROCmSignalData*)signal->native_handle;
    if (!sd)
        return -1;
    if (signal->signaled)
        return 0;
    if (!sd->event || !sd->backend || !sd->backend->hipEventSynchronize)
        return -1; /* nothing recorded, or no way to block on it */

    if (sd->backend->hipEventSynchronize(sd->event) != 0)
        return -1;
    signal->signaled = true;
    return 0;
}

#else /* !__linux__ */

CMLHCQQueue* cml_hcq_rocm_queue_create(void) { return NULL; }
void cml_hcq_rocm_queue_destroy(CMLHCQQueue* q) { (void)q; }

int cml_hcq_rocm_submit_kernel(CMLHCQQueue* q, const CMLHCQKernelDesc* d)
    { (void)q; (void)d; return -1; }

int cml_hcq_rocm_memcpy_h2d(CMLHCQQueue* q, void* d, const void* s, size_t n)
    { (void)q; (void)d; (void)s; (void)n; return -1; }

int cml_hcq_rocm_memcpy_d2h(CMLHCQQueue* q, void* d, const void* s, size_t n)
    { (void)q; (void)d; (void)s; (void)n; return -1; }

CMLHCQSignal* cml_hcq_rocm_signal_create(void) { return NULL; }
void          cml_hcq_rocm_signal_destroy(CMLHCQSignal* s) { (void)s; }

int cml_hcq_rocm_signal_record(CMLHCQQueue* q, CMLHCQSignal* s)
    { (void)q; (void)s; return -1; }

int cml_hcq_rocm_queue_wait(CMLHCQQueue* q, CMLHCQSignal* s)
    { (void)q; (void)s; return -1; }

int cml_hcq_rocm_signal_wait_cpu(CMLHCQSignal* s, uint64_t t)
    { (void)s; (void)t; return -1; }

int cml_hcq_rocm_queue_synchronize(CMLHCQQueue* q) { (void)q; return -1; }

#endif /* __linux__ */
