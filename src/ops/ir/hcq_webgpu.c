/* HCQ adapter for the WebGPU (wgpu-native) backend.
 *
 * Wraps the cml_webgpu_* API (gpu/webgpu_backend.c) in the CMLHCQBackendOps
 * contract so WebGPU dispatches uniformly through cml_hcq_backend_ops(). The
 * wgpu-native library is dlopen'd at runtime and the whole backend compiles
 * to graceful stubs without CML_HAS_WEBGPU, so this adapter follows the
 * hcq_nv.c pattern: real implementation under the define, -1/NULL stubs
 * otherwise.
 *
 * WGPU has no host-visible timeline semaphore in this binding, so signals are
 * synchronous: record drains the queue (device poll + queue submit is already
 * ordered) and marks the signal ready. memcpy_d2h maps through the backend's
 * synchronous download wrapper.
 */
#include "ops/ir/hcq.h"
#include "ops/ir/gpu/webgpu_backend.h"
#include "core/logging.h"
#include "alloc/cml_allocator.h"

#ifdef CML_HAS_WEBGPU

typedef struct {
    CMLWebGPUBackend* backend;
    uint64_t submit_count;
} WebGPUQueueData;

static CMLWebGPUBackend* webgpu_backend_of(CMLHCQQueue* queue) {
    WebGPUQueueData* qd = (WebGPUQueueData*)(queue ? queue->native_handle : NULL);
    return qd ? qd->backend : NULL;
}

CMLHCQQueue* cml_hcq_webgpu_queue_create(void) {
    if (!cml_webgpu_available())
        return NULL;

    CMLWebGPUBackend* backend = cml_webgpu_backend_create();
    if (!backend)
        return NULL;
    if (cml_webgpu_backend_init(backend) != 0) {
        cml_webgpu_backend_free(backend);
        return NULL;
    }

    CMLHCQQueue* queue  = (CMLHCQQueue*)cml_calloc(1, sizeof(CMLHCQQueue));
    WebGPUQueueData* qd = (WebGPUQueueData*)cml_calloc(1, sizeof(WebGPUQueueData));
    if (!queue || !qd) {
        cml_free(queue);
        cml_free(qd);
        cml_webgpu_backend_free(backend);
        return NULL;
    }

    qd->backend          = backend;
    queue->backend       = CML_HCQ_WEBGPU;
    queue->native_handle = qd;
    queue->active        = true;
    return queue;
}

void cml_hcq_webgpu_queue_destroy(CMLHCQQueue* queue) {
    if (!queue)
        return;
    WebGPUQueueData* qd = (WebGPUQueueData*)queue->native_handle;
    if (qd) {
        cml_webgpu_backend_free(qd->backend);
        cml_free(qd);
    }
    queue->native_handle = NULL;
    queue->active        = false;
    cml_free(queue);
}

int cml_hcq_webgpu_submit_kernel(CMLHCQQueue* queue, const CMLHCQKernelDesc* desc) {
    if (!queue || !desc || !desc->compiled_kernel || !desc->args)
        return -1;
    CMLWebGPUBackend* backend = webgpu_backend_of(queue);
    if (!backend)
        return -1;

    size_t wg[3] = {desc->grid[0], desc->grid[1], desc->grid[2]};
    /* Buffer sizes live in the WGPUBuffer objects; the backend tolerates a
     * NULL sizes array. */
    int rc = cml_webgpu_launch_kernel(backend, (CMLWebGPUKernel*)desc->compiled_kernel, wg,
                                      desc->args, NULL, desc->num_args);
    if (rc == 0)
        ((WebGPUQueueData*)queue->native_handle)->submit_count++;
    return rc;
}

int cml_hcq_webgpu_memcpy_h2d(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes) {
    if (!queue || !dst || !src || bytes == 0)
        return -1;
    return cml_webgpu_upload(webgpu_backend_of(queue), dst, src, bytes);
}

int cml_hcq_webgpu_memcpy_d2h(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes) {
    if (!queue || !dst || !src || bytes == 0)
        return -1;
    return cml_webgpu_download(webgpu_backend_of(queue), dst, (void*)(uintptr_t)src, bytes);
}

int cml_hcq_webgpu_queue_synchronize(CMLHCQQueue* queue) {
    CMLWebGPUBackend* backend = webgpu_backend_of(queue);
    if (!backend || !backend->initialized)
        return -1;
    /* Drain the queue: an empty submit is fenced by device poll. */
    if (backend->fn_queue_submit && backend->fn_device_poll) {
        void (*queue_submit)(void*, void*) = (void (*)(void*, void*))backend->fn_queue_submit;
        /* wgpuDevicePoll(device, wait, options) */
        unsigned int (*device_poll)(void*, int, const void*) =
            (unsigned int (*)(void*, int, const void*))backend->fn_device_poll;
        void* device = backend->device;
        void* queue  = backend->queue;
        if (device && queue) {
            queue_submit(queue, NULL);         /* WGPUQueueSubmit(q, 0, NULL) — fence */
            if (!device_poll(device, 1, NULL)) /* wait for the fence */
                return -1;
            return 0;
        }
    }
    return -1;
}

CMLHCQSignal* cml_hcq_webgpu_signal_create(void) {
    /* Must use the CML allocator: hcq.c frees signal wrappers with cml_free,
     * so system calloc here would corrupt the heap (see the Vulkan adapter). */
    CMLHCQSignal* signal = (CMLHCQSignal*)cml_calloc(1, sizeof(CMLHCQSignal));
    if (!signal)
        return NULL;
    signal->backend  = CML_HCQ_WEBGPU;
    signal->signaled = false;
    return signal;
}

void cml_hcq_webgpu_signal_destroy(CMLHCQSignal* signal) {
    if (!signal)
        return;
    cml_free(signal);
}

int cml_hcq_webgpu_signal_record(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    if (!queue || !signal)
        return -1;
    /* Synchronous semantics: everything submitted so far is complete once the
     * queue drains. */
    if (cml_hcq_webgpu_queue_synchronize(queue) != 0)
        return -1;
    signal->signaled = true;
    signal->timeline_value++;
    return 0;
}

int cml_hcq_webgpu_queue_wait(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    (void)queue;
    (void)signal; /* work is already ordered through the single WGPUQueue */
    return 0;
}

int cml_hcq_webgpu_signal_wait_cpu(CMLHCQSignal* signal, uint64_t timeout_ms) {
    (void)timeout_ms;
    if (!signal)
        return -1;
    return signal->signaled ? 0 : -1;
}

#else /* !CML_HAS_WEBGPU */

CMLHCQQueue* cml_hcq_webgpu_queue_create(void) { return NULL; }
void cml_hcq_webgpu_queue_destroy(CMLHCQQueue* q) { (void)q; }

int cml_hcq_webgpu_submit_kernel(CMLHCQQueue* q, const CMLHCQKernelDesc* d) {
    (void)q;
    (void)d;
    return -1;
}

int cml_hcq_webgpu_memcpy_h2d(CMLHCQQueue* q, void* d, const void* s, size_t n) {
    (void)q;
    (void)d;
    (void)s;
    (void)n;
    return -1;
}

int cml_hcq_webgpu_memcpy_d2h(CMLHCQQueue* q, void* d, const void* s, size_t n) {
    (void)q;
    (void)d;
    (void)s;
    (void)n;
    return -1;
}

CMLHCQSignal* cml_hcq_webgpu_signal_create(void) { return NULL; }
void cml_hcq_webgpu_signal_destroy(CMLHCQSignal* s) { (void)s; }

int cml_hcq_webgpu_signal_record(CMLHCQQueue* q, CMLHCQSignal* s) {
    (void)q;
    (void)s;
    return -1;
}

int cml_hcq_webgpu_queue_wait(CMLHCQQueue* q, CMLHCQSignal* s) {
    (void)q;
    (void)s;
    return -1;
}

int cml_hcq_webgpu_signal_wait_cpu(CMLHCQSignal* s, uint64_t t) {
    (void)s;
    (void)t;
    return -1;
}

int cml_hcq_webgpu_queue_synchronize(CMLHCQQueue* q) {
    (void)q;
    return -1;
}

#endif /* CML_HAS_WEBGPU */
