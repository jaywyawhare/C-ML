/**
 * @file hcq_opencl.c
 * @brief Hardware Command Queues -- OpenCL backend stub
 *
 * Guarded by CML_HAS_OPENCL.  When the flag is not defined this translation
 * unit compiles to nothing.  When it *is* defined the functions below are
 * linked in and called from the dispatcher in hcq.c.
 *
 * Queue wraps cl_command_queue, Signal wraps cl_event.  For now every
 * function logs a "not implemented" message and returns -1 (or NULL) so
 * the build succeeds and callers get a clear diagnostic.
 */

#ifdef CML_HAS_OPENCL

#include "ops/ir/hcq.h"
#include "core/logging.h"

#include <stdlib.h>

/*
 * We only need the OpenCL types for documentation / future implementation.
 * Including the full header here so that types like cl_command_queue and
 * cl_event are available when we flesh out the stubs.
 */
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/* ── Stub helpers ────────────────────────────────────────────────────────── */

#define OCL_STUB_WARN(name)                                                 \
    LOG_WARNING("OpenCL HCQ: %s not implemented", (name))

/* ══════════════════════════════════════════════════════════════════════════
 * Queue lifecycle
 * ══════════════════════════════════════════════════════════════════════════ */

CMLHCQQueue* cml_hcq_opencl_queue_create(void) {
    OCL_STUB_WARN("queue_create");
    /*
     * TODO: select a cl_platform_id and cl_device_id, create a
     * cl_command_queue with clCreateCommandQueueWithProperties(),
     * store it in queue->native_handle, and set
     * backend = CML_HCQ_OPENCL.
     */
    return NULL;
}

void cml_hcq_opencl_queue_destroy(CMLHCQQueue* queue) {
    OCL_STUB_WARN("queue_destroy");
    if (queue) {
        /*
         * TODO: call clReleaseCommandQueue() on the cl_command_queue
         * stored in native_handle before freeing the wrapper.
         */
        free(queue);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * Kernel submission
 * ══════════════════════════════════════════════════════════════════════════ */

int cml_hcq_opencl_submit_kernel(CMLHCQQueue* queue,
                                 const CMLHCQKernelDesc* desc) {
    (void)queue;
    (void)desc;
    OCL_STUB_WARN("submit_kernel");
    /*
     * TODO: extract cl_command_queue from queue->native_handle, cast
     * desc->compiled_kernel to cl_kernel and call
     * clEnqueueNDRangeKernel() with desc->grid and desc->block.
     */
    return -1;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Memcpy
 * ══════════════════════════════════════════════════════════════════════════ */

int cml_hcq_opencl_memcpy_h2d(CMLHCQQueue* queue, void* dst,
                               const void* src, size_t bytes) {
    (void)queue;
    (void)dst;
    (void)src;
    (void)bytes;
    OCL_STUB_WARN("memcpy_h2d");
    /*
     * TODO: use clEnqueueWriteBuffer() on the cl_command_queue
     * inside queue->native_handle to copy src to the cl_mem in dst.
     */
    return -1;
}

int cml_hcq_opencl_memcpy_d2h(CMLHCQQueue* queue, void* dst,
                               const void* src, size_t bytes) {
    (void)queue;
    (void)dst;
    (void)src;
    (void)bytes;
    OCL_STUB_WARN("memcpy_d2h");
    /*
     * TODO: use clEnqueueReadBuffer() on the cl_command_queue
     * inside queue->native_handle to copy the cl_mem in src to dst.
     */
    return -1;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Signals  (cl_event wrappers)
 * ══════════════════════════════════════════════════════════════════════════ */

CMLHCQSignal* cml_hcq_opencl_signal_create(void) {
    OCL_STUB_WARN("signal_create");
    /*
     * TODO: allocate CMLHCQSignal, set backend = CML_HCQ_OPENCL.
     * cl_event objects in OpenCL are typically created by enqueue
     * commands rather than up front, so native_handle may stay NULL
     * until signal_record is called.
     */
    return NULL;
}

void cml_hcq_opencl_signal_destroy(CMLHCQSignal* signal) {
    OCL_STUB_WARN("signal_destroy");
    if (signal) {
        /*
         * TODO: if native_handle is a valid cl_event call
         * clReleaseEvent() before freeing the wrapper.
         */
        free(signal);
    }
}

int cml_hcq_opencl_signal_record(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    (void)queue;
    (void)signal;
    OCL_STUB_WARN("signal_record");
    /*
     * TODO: enqueue a marker with clEnqueueMarkerWithWaitList()
     * on queue->native_handle and store the resulting cl_event in
     * signal->native_handle.
     */
    return -1;
}

int cml_hcq_opencl_queue_wait(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    (void)queue;
    (void)signal;
    OCL_STUB_WARN("queue_wait");
    /*
     * TODO: enqueue a barrier with clEnqueueBarrierWithWaitList()
     * passing the cl_event in signal->native_handle as a dependency.
     */
    return -1;
}

int cml_hcq_opencl_signal_wait_cpu(CMLHCQSignal* signal, uint64_t timeout_ms) {
    (void)signal;
    (void)timeout_ms;
    OCL_STUB_WARN("signal_wait_cpu");
    /*
     * TODO: call clWaitForEvents(1, &signal->native_handle).
     * For timeout support one can poll clGetEventInfo() in a loop.
     */
    return -1;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Synchronize
 * ══════════════════════════════════════════════════════════════════════════ */

int cml_hcq_opencl_queue_synchronize(CMLHCQQueue* queue) {
    (void)queue;
    OCL_STUB_WARN("queue_synchronize");
    /*
     * TODO: call clFinish(queue->native_handle).
     */
    return -1;
}

#endif /* CML_HAS_OPENCL */
