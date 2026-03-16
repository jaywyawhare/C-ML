/**
 * @file hcq_cuda.c
 * @brief Hardware Command Queues -- CUDA backend
 *
 * Guarded by CML_HAS_CUDA.  When the flag is not defined this translation
 * unit compiles to nothing.  When it *is* defined the functions below are
 * linked in and called from the dispatcher in hcq.c.
 *
 * Uses the CMLCUDABackend infrastructure from ops/ir/gpu/cuda_backend.h
 * which provides dynamically loaded CUDA driver API function pointers.
 */

#ifdef CML_HAS_CUDA

#include "ops/ir/hcq.h"
#include "ops/ir/gpu/cuda_backend.h"
#include "ops/ir/dispatch.h"
#include "core/logging.h"

#include <stdlib.h>
#include <stdint.h>

CMLHCQQueue* cml_hcq_cuda_queue_create(void) {
    CMLCUDABackend* cuda = cml_dispatch_get_cuda_backend();
    if (!cuda) {
        LOG_ERROR("CUDA HCQ: no CUDA backend available");
        return NULL;
    }

    CMLHCQQueue* queue = (CMLHCQQueue*)calloc(1, sizeof(CMLHCQQueue));
    if (!queue) {
        LOG_ERROR("CUDA HCQ: failed to allocate queue");
        return NULL;
    }

    queue->backend = CML_HCQ_CUDA;

    CUstream stream = NULL;
    CUresult err = cuda->cuStreamCreate(&stream, 0);
    if (err != 0) {
        LOG_ERROR("CUDA HCQ: cuStreamCreate failed (CUresult=%d)", err);
        free(queue);
        return NULL;
    }

    queue->native_handle = stream;
    queue->active = true;
    return queue;
}

void cml_hcq_cuda_queue_destroy(CMLHCQQueue* queue) {
    if (!queue) return;

    CMLCUDABackend* cuda = cml_dispatch_get_cuda_backend();
    if (cuda && queue->native_handle) {
        CUresult err = cuda->cuStreamDestroy(queue->native_handle);
        if (err != 0) {
            LOG_WARNING("CUDA HCQ: cuStreamDestroy failed (CUresult=%d)", err);
        }
    }

    free(queue);
}

int cml_hcq_cuda_submit_kernel(CMLHCQQueue* queue,
                               const CMLHCQKernelDesc* desc) {
    if (!queue || !desc) return -1;

    CMLCUDABackend* cuda = cml_dispatch_get_cuda_backend();
    if (!cuda) {
        LOG_ERROR("CUDA HCQ: no CUDA backend available");
        return -1;
    }

    CUstream stream = (CUstream)queue->native_handle;
    CUfunction fn = (CUfunction)desc->compiled_kernel;

    CUresult err = cuda->cuLaunchKernel(
        fn,
        (unsigned int)desc->grid[0],
        (unsigned int)desc->grid[1],
        (unsigned int)desc->grid[2],
        (unsigned int)desc->block[0],
        (unsigned int)desc->block[1],
        (unsigned int)desc->block[2],
        0,       /* sharedMemBytes */
        stream,
        desc->args,
        NULL     /* extra */
    );
    if (err != 0) {
        LOG_ERROR("CUDA HCQ: cuLaunchKernel failed (CUresult=%d)", err);
        return -1;
    }

    return 0;
}

int cml_hcq_cuda_memcpy_h2d(CMLHCQQueue* queue, void* dst,
                             const void* src, size_t bytes) {
    if (!queue || !dst || !src) return -1;

    CMLCUDABackend* cuda = cml_dispatch_get_cuda_backend();
    if (!cuda) {
        LOG_ERROR("CUDA HCQ: no CUDA backend available");
        return -1;
    }

    CUresult err = cuda->cuMemcpyHtoDAsync(
        (CUdeviceptr)(uintptr_t)dst,
        src,
        bytes,
        (CUstream)queue->native_handle
    );
    if (err != 0) {
        LOG_ERROR("CUDA HCQ: cuMemcpyHtoDAsync failed (CUresult=%d)", err);
        return -1;
    }

    return 0;
}

int cml_hcq_cuda_memcpy_d2h(CMLHCQQueue* queue, void* dst,
                             const void* src, size_t bytes) {
    if (!queue || !dst || !src) return -1;

    CMLCUDABackend* cuda = cml_dispatch_get_cuda_backend();
    if (!cuda) {
        LOG_ERROR("CUDA HCQ: no CUDA backend available");
        return -1;
    }

    CUresult err = cuda->cuMemcpyDtoHAsync(
        dst,
        (CUdeviceptr)(uintptr_t)src,
        bytes,
        (CUstream)queue->native_handle
    );
    if (err != 0) {
        LOG_ERROR("CUDA HCQ: cuMemcpyDtoHAsync failed (CUresult=%d)", err);
        return -1;
    }

    return 0;
}

CMLHCQSignal* cml_hcq_cuda_signal_create(void) {
    CMLCUDABackend* cuda = cml_dispatch_get_cuda_backend();
    if (!cuda) {
        LOG_ERROR("CUDA HCQ: no CUDA backend available");
        return NULL;
    }

    CMLHCQSignal* signal = (CMLHCQSignal*)calloc(1, sizeof(CMLHCQSignal));
    if (!signal) {
        LOG_ERROR("CUDA HCQ: failed to allocate signal");
        return NULL;
    }

    signal->backend = CML_HCQ_CUDA;

    CUevent event = NULL;
    CUresult err = cuda->cuEventCreate(&event, 0);
    if (err != 0) {
        LOG_ERROR("CUDA HCQ: cuEventCreate failed (CUresult=%d)", err);
        free(signal);
        return NULL;
    }

    signal->native_handle = event;
    signal->signaled = false;
    return signal;
}

void cml_hcq_cuda_signal_destroy(CMLHCQSignal* signal) {
    if (!signal) return;

    CMLCUDABackend* cuda = cml_dispatch_get_cuda_backend();
    if (cuda && signal->native_handle) {
        CUresult err = cuda->cuEventDestroy(signal->native_handle);
        if (err != 0) {
            LOG_WARNING("CUDA HCQ: cuEventDestroy failed (CUresult=%d)", err);
        }
    }

    free(signal);
}

int cml_hcq_cuda_signal_record(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    if (!queue || !signal) return -1;

    CMLCUDABackend* cuda = cml_dispatch_get_cuda_backend();
    if (!cuda) {
        LOG_ERROR("CUDA HCQ: no CUDA backend available");
        return -1;
    }

    CUresult err = cuda->cuEventRecord(
        signal->native_handle,
        (CUstream)queue->native_handle
    );
    if (err != 0) {
        LOG_ERROR("CUDA HCQ: cuEventRecord failed (CUresult=%d)", err);
        return -1;
    }

    signal->signaled = true;
    return 0;
}

int cml_hcq_cuda_queue_wait(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    if (!queue || !signal) return -1;

    CMLCUDABackend* cuda = cml_dispatch_get_cuda_backend();
    if (!cuda) {
        LOG_ERROR("CUDA HCQ: no CUDA backend available");
        return -1;
    }

    CUresult err = cuda->cuStreamWaitEvent(
        (CUstream)queue->native_handle,
        signal->native_handle,
        0
    );
    if (err != 0) {
        LOG_ERROR("CUDA HCQ: cuStreamWaitEvent failed (CUresult=%d)", err);
        return -1;
    }

    return 0;
}

int cml_hcq_cuda_signal_wait_cpu(CMLHCQSignal* signal, uint64_t timeout_ms) {
    if (!signal) return -1;
    (void)timeout_ms; /* cuEventSynchronize does not support timeout */

    CMLCUDABackend* cuda = cml_dispatch_get_cuda_backend();
    if (!cuda) {
        LOG_ERROR("CUDA HCQ: no CUDA backend available");
        return -1;
    }

    CUresult err = cuda->cuEventSynchronize(signal->native_handle);
    if (err != 0) {
        LOG_ERROR("CUDA HCQ: cuEventSynchronize failed (CUresult=%d)", err);
        return -1;
    }

    return 0;
}

int cml_hcq_cuda_queue_synchronize(CMLHCQQueue* queue) {
    if (!queue) return -1;

    CMLCUDABackend* cuda = cml_dispatch_get_cuda_backend();
    if (!cuda) {
        LOG_ERROR("CUDA HCQ: no CUDA backend available");
        return -1;
    }

    CUresult err = cuda->cuStreamSynchronize((CUstream)queue->native_handle);
    if (err != 0) {
        LOG_ERROR("CUDA HCQ: cuStreamSynchronize failed (CUresult=%d)", err);
        return -1;
    }

    return 0;
}

#endif /* CML_HAS_CUDA */
