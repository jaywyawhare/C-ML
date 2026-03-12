/**
 * @file hcq_cuda.c
 * @brief Hardware Command Queues -- CUDA backend stub
 *
 * Guarded by CML_HAS_CUDA.  When the flag is not defined this translation
 * unit compiles to nothing.  When it *is* defined the functions below are
 * linked in and called from the dispatcher in hcq.c.
 *
 * The actual CUDA stream/event management requires the CMLCUDABackend
 * infrastructure from ops/ir/gpu/cuda_backend.h.  For now every function
 * logs a "not implemented" message and returns -1 (or NULL) so the build
 * succeeds and callers get a clear diagnostic.
 */

#ifdef CML_HAS_CUDA

#include "ops/ir/hcq.h"
#include "ops/ir/gpu/cuda_backend.h"
#include "core/logging.h"

#include <stdlib.h>

/* ── Stub helpers ────────────────────────────────────────────────────────── */

#define CUDA_STUB_WARN(name)                                                \
    LOG_WARNING("CUDA HCQ: %s not implemented", (name))

/* ══════════════════════════════════════════════════════════════════════════
 * Queue lifecycle
 * ══════════════════════════════════════════════════════════════════════════ */

CMLHCQQueue* cml_hcq_cuda_queue_create(void) {
    CUDA_STUB_WARN("queue_create");
    /*
     * TODO: allocate a CMLHCQQueue, set backend = CML_HCQ_CUDA,
     * and wrap a CUstream obtained from cuStreamCreate() inside
     * native_handle.
     */
    return NULL;
}

void cml_hcq_cuda_queue_destroy(CMLHCQQueue* queue) {
    CUDA_STUB_WARN("queue_destroy");
    if (queue) {
        /*
         * TODO: destroy the CUstream stored in native_handle via
         * cuStreamDestroy() before freeing the wrapper.
         */
        free(queue);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * Kernel submission
 * ══════════════════════════════════════════════════════════════════════════ */

int cml_hcq_cuda_submit_kernel(CMLHCQQueue* queue,
                               const CMLHCQKernelDesc* desc) {
    (void)queue;
    (void)desc;
    CUDA_STUB_WARN("submit_kernel");
    /*
     * TODO: extract CUstream from queue->native_handle, cast
     * desc->compiled_kernel to CUfunction and call cuLaunchKernel().
     */
    return -1;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Memcpy
 * ══════════════════════════════════════════════════════════════════════════ */

int cml_hcq_cuda_memcpy_h2d(CMLHCQQueue* queue, void* dst,
                             const void* src, size_t bytes) {
    (void)queue;
    (void)dst;
    (void)src;
    (void)bytes;
    CUDA_STUB_WARN("memcpy_h2d");
    /*
     * TODO: use cuMemcpyHtoDAsync() on the CUstream inside
     * queue->native_handle.
     */
    return -1;
}

int cml_hcq_cuda_memcpy_d2h(CMLHCQQueue* queue, void* dst,
                             const void* src, size_t bytes) {
    (void)queue;
    (void)dst;
    (void)src;
    (void)bytes;
    CUDA_STUB_WARN("memcpy_d2h");
    /*
     * TODO: use cuMemcpyDtoHAsync() on the CUstream inside
     * queue->native_handle.
     */
    return -1;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Signals  (CUevent wrappers)
 * ══════════════════════════════════════════════════════════════════════════ */

CMLHCQSignal* cml_hcq_cuda_signal_create(void) {
    CUDA_STUB_WARN("signal_create");
    /*
     * TODO: allocate CMLHCQSignal, set backend = CML_HCQ_CUDA, create a
     * CUevent via cuEventCreate() and store it in native_handle.
     */
    return NULL;
}

void cml_hcq_cuda_signal_destroy(CMLHCQSignal* signal) {
    CUDA_STUB_WARN("signal_destroy");
    if (signal) {
        /*
         * TODO: destroy the CUevent stored in native_handle via
         * cuEventDestroy() before freeing the wrapper.
         */
        free(signal);
    }
}

int cml_hcq_cuda_signal_record(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    (void)queue;
    (void)signal;
    CUDA_STUB_WARN("signal_record");
    /*
     * TODO: call cuEventRecord(signal->native_handle,
     *                          queue->native_handle).
     */
    return -1;
}

int cml_hcq_cuda_queue_wait(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    (void)queue;
    (void)signal;
    CUDA_STUB_WARN("queue_wait");
    /*
     * TODO: call cuStreamWaitEvent(queue->native_handle,
     *                              signal->native_handle, 0).
     */
    return -1;
}

int cml_hcq_cuda_signal_wait_cpu(CMLHCQSignal* signal, uint64_t timeout_ms) {
    (void)signal;
    (void)timeout_ms;
    CUDA_STUB_WARN("signal_wait_cpu");
    /*
     * TODO: poll cuEventQuery() in a loop or call
     * cuEventSynchronize() with a timeout mechanism.
     */
    return -1;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Synchronize
 * ══════════════════════════════════════════════════════════════════════════ */

int cml_hcq_cuda_queue_synchronize(CMLHCQQueue* queue) {
    (void)queue;
    CUDA_STUB_WARN("queue_synchronize");
    /*
     * TODO: call cuStreamSynchronize(queue->native_handle).
     */
    return -1;
}

#endif /* CML_HAS_CUDA */
