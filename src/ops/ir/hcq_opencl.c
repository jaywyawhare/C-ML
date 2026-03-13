/**
 * @file hcq_opencl.c
 * @brief Hardware Command Queues -- OpenCL backend
 *
 * Guarded by CML_HAS_OPENCL.  When the flag is not defined this translation
 * unit compiles to nothing.  When it *is* defined the functions below are
 * linked in and called from the dispatcher in hcq.c.
 *
 * Queue wraps cl_command_queue, Signal wraps cl_event.
 */

#ifdef CML_HAS_OPENCL

#include "ops/ir/hcq.h"
#include "core/logging.h"

#include <stdlib.h>
#include <stdbool.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/* ── File-local OpenCL context ──────────────────────────────────────────── */

static struct {
    cl_platform_id platform;
    cl_device_id   device;
    cl_context     context;
    bool           initialized;
} g_ocl_ctx = {0};

/**
 * @brief Lazily initialise the shared OpenCL platform / device / context.
 * @return 0 on success, -1 on failure.
 */
static int ensure_ocl_init(void) {
    if (g_ocl_ctx.initialized) return 0;

    cl_int err;

    err = clGetPlatformIDs(1, &g_ocl_ctx.platform, NULL);
    if (err != CL_SUCCESS) {
        LOG_ERROR("OpenCL HCQ: clGetPlatformIDs failed (err=%d)", err);
        return -1;
    }

    err = clGetDeviceIDs(g_ocl_ctx.platform, CL_DEVICE_TYPE_GPU, 1,
                         &g_ocl_ctx.device, NULL);
    if (err != CL_SUCCESS) {
        /* Fall back to the default device type */
        err = clGetDeviceIDs(g_ocl_ctx.platform, CL_DEVICE_TYPE_DEFAULT, 1,
                             &g_ocl_ctx.device, NULL);
        if (err != CL_SUCCESS) {
            LOG_ERROR("OpenCL HCQ: clGetDeviceIDs failed (err=%d)", err);
            return -1;
        }
    }

    g_ocl_ctx.context = clCreateContext(NULL, 1, &g_ocl_ctx.device,
                                        NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        LOG_ERROR("OpenCL HCQ: clCreateContext failed (err=%d)", err);
        return -1;
    }

    g_ocl_ctx.initialized = true;
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Queue lifecycle
 * ══════════════════════════════════════════════════════════════════════════ */

CMLHCQQueue* cml_hcq_opencl_queue_create(void) {
    if (ensure_ocl_init() != 0) {
        return NULL;
    }

    CMLHCQQueue* queue = (CMLHCQQueue*)calloc(1, sizeof(CMLHCQQueue));
    if (!queue) {
        LOG_ERROR("OpenCL HCQ: failed to allocate queue");
        return NULL;
    }

    queue->backend = CML_HCQ_OPENCL;

    cl_int err;
    cl_command_queue cq = clCreateCommandQueueWithProperties(
        g_ocl_ctx.context, g_ocl_ctx.device, NULL, &err);
    if (err != CL_SUCCESS) {
        LOG_ERROR("OpenCL HCQ: clCreateCommandQueueWithProperties failed (err=%d)", err);
        free(queue);
        return NULL;
    }

    queue->native_handle = (void*)cq;
    queue->active = true;
    return queue;
}

void cml_hcq_opencl_queue_destroy(CMLHCQQueue* queue) {
    if (!queue) return;

    if (queue->native_handle) {
        cl_int err = clReleaseCommandQueue((cl_command_queue)queue->native_handle);
        if (err != CL_SUCCESS) {
            LOG_WARNING("OpenCL HCQ: clReleaseCommandQueue failed (err=%d)", err);
        }
    }

    free(queue);
}

/* ══════════════════════════════════════════════════════════════════════════
 * Kernel submission
 * ══════════════════════════════════════════════════════════════════════════ */

int cml_hcq_opencl_submit_kernel(CMLHCQQueue* queue,
                                 const CMLHCQKernelDesc* desc) {
    if (!queue || !desc) return -1;

    cl_command_queue cq = (cl_command_queue)queue->native_handle;
    cl_kernel kernel = (cl_kernel)desc->compiled_kernel;

    cl_int err = clEnqueueNDRangeKernel(
        cq,
        kernel,
        3,             /* work_dim */
        NULL,          /* global_work_offset */
        desc->grid,    /* global_work_size */
        desc->block,   /* local_work_size */
        0,             /* num_events_in_wait_list */
        NULL,          /* event_wait_list */
        NULL           /* event */
    );
    if (err != CL_SUCCESS) {
        LOG_ERROR("OpenCL HCQ: clEnqueueNDRangeKernel failed (err=%d)", err);
        return -1;
    }

    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Memcpy
 * ══════════════════════════════════════════════════════════════════════════ */

int cml_hcq_opencl_memcpy_h2d(CMLHCQQueue* queue, void* dst,
                               const void* src, size_t bytes) {
    if (!queue || !dst || !src) return -1;

    cl_command_queue cq = (cl_command_queue)queue->native_handle;

    cl_int err = clEnqueueWriteBuffer(
        cq,
        (cl_mem)dst,
        CL_FALSE,   /* non-blocking */
        0,           /* offset */
        bytes,
        src,
        0, NULL, NULL
    );
    if (err != CL_SUCCESS) {
        LOG_ERROR("OpenCL HCQ: clEnqueueWriteBuffer failed (err=%d)", err);
        return -1;
    }

    return 0;
}

int cml_hcq_opencl_memcpy_d2h(CMLHCQQueue* queue, void* dst,
                               const void* src, size_t bytes) {
    if (!queue || !dst || !src) return -1;

    cl_command_queue cq = (cl_command_queue)queue->native_handle;

    cl_int err = clEnqueueReadBuffer(
        cq,
        (cl_mem)src,
        CL_FALSE,   /* non-blocking */
        0,           /* offset */
        bytes,
        dst,
        0, NULL, NULL
    );
    if (err != CL_SUCCESS) {
        LOG_ERROR("OpenCL HCQ: clEnqueueReadBuffer failed (err=%d)", err);
        return -1;
    }

    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Signals  (cl_event wrappers)
 * ══════════════════════════════════════════════════════════════════════════ */

CMLHCQSignal* cml_hcq_opencl_signal_create(void) {
    CMLHCQSignal* signal = (CMLHCQSignal*)calloc(1, sizeof(CMLHCQSignal));
    if (!signal) {
        LOG_ERROR("OpenCL HCQ: failed to allocate signal");
        return NULL;
    }

    signal->backend = CML_HCQ_OPENCL;
    signal->native_handle = NULL;  /* created lazily on signal_record */
    signal->signaled = false;
    return signal;
}

void cml_hcq_opencl_signal_destroy(CMLHCQSignal* signal) {
    if (!signal) return;

    if (signal->native_handle) {
        cl_int err = clReleaseEvent((cl_event)signal->native_handle);
        if (err != CL_SUCCESS) {
            LOG_WARNING("OpenCL HCQ: clReleaseEvent failed (err=%d)", err);
        }
    }

    free(signal);
}

int cml_hcq_opencl_signal_record(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    if (!queue || !signal) return -1;

    /* Release any previously recorded event */
    if (signal->native_handle) {
        clReleaseEvent((cl_event)signal->native_handle);
        signal->native_handle = NULL;
    }

    cl_event evt = NULL;
    cl_int err = clEnqueueMarkerWithWaitList(
        (cl_command_queue)queue->native_handle,
        0, NULL, &evt);
    if (err != CL_SUCCESS) {
        LOG_ERROR("OpenCL HCQ: clEnqueueMarkerWithWaitList failed (err=%d)", err);
        return -1;
    }

    signal->native_handle = (void*)evt;
    signal->signaled = true;
    return 0;
}

int cml_hcq_opencl_queue_wait(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    if (!queue || !signal) return -1;

    cl_event evt = (cl_event)signal->native_handle;
    if (!evt) {
        LOG_ERROR("OpenCL HCQ: queue_wait called on signal with no event");
        return -1;
    }

    cl_int err = clEnqueueBarrierWithWaitList(
        (cl_command_queue)queue->native_handle,
        1, &evt, NULL);
    if (err != CL_SUCCESS) {
        LOG_ERROR("OpenCL HCQ: clEnqueueBarrierWithWaitList failed (err=%d)", err);
        return -1;
    }

    return 0;
}

int cml_hcq_opencl_signal_wait_cpu(CMLHCQSignal* signal, uint64_t timeout_ms) {
    if (!signal) return -1;
    (void)timeout_ms; /* clWaitForEvents does not support timeout */

    cl_event evt = (cl_event)signal->native_handle;
    if (!evt) {
        LOG_ERROR("OpenCL HCQ: signal_wait_cpu called on signal with no event");
        return -1;
    }

    cl_int err = clWaitForEvents(1, &evt);
    if (err != CL_SUCCESS) {
        LOG_ERROR("OpenCL HCQ: clWaitForEvents failed (err=%d)", err);
        return -1;
    }

    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Synchronize
 * ══════════════════════════════════════════════════════════════════════════ */

int cml_hcq_opencl_queue_synchronize(CMLHCQQueue* queue) {
    if (!queue) return -1;

    cl_int err = clFinish((cl_command_queue)queue->native_handle);
    if (err != CL_SUCCESS) {
        LOG_ERROR("OpenCL HCQ: clFinish failed (err=%d)", err);
        return -1;
    }

    return 0;
}

#endif /* CML_HAS_OPENCL */
