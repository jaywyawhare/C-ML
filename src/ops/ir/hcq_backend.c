#include "ops/ir/hcq_backend.h"
#include "core/logging.h"
#include "alloc/cml_allocator.h"
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * CPU HCQ backend — synchronous, host==device address space.
 *
 * Previously the CPU path was open-coded as special-cases scattered through
 * hcq.c (~8 `if (backend != CML_HCQ_CPU)` branches). It is now a first-class
 * ops entry so every backend — CPU included — dispatches uniformly through
 * cml_hcq_backend_ops(). This is the reference implementation the GPU backends
 * converge onto, and it makes the queue/signal/pipeline machinery testable
 * without any GPU hardware.
 * ---------------------------------------------------------------------- */

/* On the CPU backend, CMLHCQKernelDesc.compiled_kernel is cast to this and run
 * synchronously. */
typedef void (*cml_cpu_kernel_fn)(void** args, int num_args);

static CMLHCQQueue* hcq_cpu_queue_create(void) {
    CMLHCQQueue* q = (CMLHCQQueue*)cml_calloc(1, sizeof(CMLHCQQueue));
    if (!q) { LOG_ERROR("Failed to allocate CPU HCQ queue"); return NULL; }
    q->backend = CML_HCQ_CPU;
    q->active  = true;
    return q;
}

static void hcq_cpu_queue_destroy(CMLHCQQueue* q) {
    if (!q) return;
    q->active = false;
    cml_free(q);
}

static int hcq_cpu_submit_kernel(CMLHCQQueue* q, const CMLHCQKernelDesc* desc) {
    (void)q;
    if (!desc->compiled_kernel) { LOG_ERROR("CPU kernel function pointer is NULL"); return -1; }
    ((cml_cpu_kernel_fn)desc->compiled_kernel)(desc->args, desc->num_args);
    return 0;
}

static int hcq_cpu_memcpy_h2d(CMLHCQQueue* q, void* dst, const void* src, size_t bytes) {
    (void)q;
    if (!dst || !src) { LOG_ERROR("NULL pointer in CPU memcpy_h2d"); return -1; }
    memcpy(dst, src, bytes);
    return 0;
}

static int hcq_cpu_memcpy_d2h(CMLHCQQueue* q, void* dst, const void* src, size_t bytes) {
    (void)q;
    if (!dst || !src) { LOG_ERROR("NULL pointer in CPU memcpy_d2h"); return -1; }
    memcpy(dst, src, bytes);
    return 0;
}

static int hcq_cpu_queue_synchronize(CMLHCQQueue* q) {
    if (q) q->num_wait_signals = 0; /* synchronous — nothing to wait for */
    return 0;
}

static CMLHCQSignal* hcq_cpu_signal_create(void) {
    CMLHCQSignal* s = (CMLHCQSignal*)cml_calloc(1, sizeof(CMLHCQSignal));
    if (!s) { LOG_ERROR("Failed to allocate CPU HCQ signal"); return NULL; }
    s->backend = CML_HCQ_CPU;
    return s;
}

static void hcq_cpu_signal_destroy(CMLHCQSignal* s) {
    if (s) cml_free(s);
}

static int hcq_cpu_signal_record(CMLHCQQueue* q, CMLHCQSignal* s) {
    (void)q;
    s->signaled = true; /* synchronous — immediately ready */
    s->timeline_value++;
    return 0;
}

static int hcq_cpu_queue_wait(CMLHCQQueue* q, CMLHCQSignal* s) {
    if (!s->signaled)
        LOG_WARNING("CPU HCQ: queue_wait on unsignaled signal %p (synchronous mode)", (void*)s);
    if (q->num_wait_signals < CML_HCQ_MAX_WAIT_SIGNALS)
        q->wait_signals[q->num_wait_signals++] = s;
    return 0;
}

static int hcq_cpu_signal_wait_cpu(CMLHCQSignal* s, uint64_t timeout_ms) {
    (void)timeout_ms;
    if (!s->signaled) {
        LOG_WARNING("CPU HCQ: signal_wait_cpu on unsignaled signal %p", (void*)s);
        return -1;
    }
    return 0;
}

static const CMLHCQBackendOps g_hcq_cpu_ops = {
    .name              = "CPU",
    .queue_create      = hcq_cpu_queue_create,
    .queue_destroy     = hcq_cpu_queue_destroy,
    .submit_kernel     = hcq_cpu_submit_kernel,
    .memcpy_h2d        = hcq_cpu_memcpy_h2d,
    .memcpy_d2h        = hcq_cpu_memcpy_d2h,
    .queue_synchronize = hcq_cpu_queue_synchronize,
    .signal_create     = hcq_cpu_signal_create,
    .signal_destroy    = hcq_cpu_signal_destroy,
    .signal_record     = hcq_cpu_signal_record,
    .queue_wait        = hcq_cpu_queue_wait,
    .signal_wait_cpu   = hcq_cpu_signal_wait_cpu,
};

#ifdef CML_HAS_CUDA
extern CMLHCQQueue* cml_hcq_cuda_queue_create(void);
extern void cml_hcq_cuda_queue_destroy(CMLHCQQueue* queue);
extern int cml_hcq_cuda_submit_kernel(CMLHCQQueue* queue, const CMLHCQKernelDesc* desc);
extern int cml_hcq_cuda_memcpy_h2d(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes);
extern int cml_hcq_cuda_memcpy_d2h(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes);
extern int cml_hcq_cuda_queue_synchronize(CMLHCQQueue* queue);
extern CMLHCQSignal* cml_hcq_cuda_signal_create(void);
extern void cml_hcq_cuda_signal_destroy(CMLHCQSignal* signal);
extern int cml_hcq_cuda_signal_record(CMLHCQQueue* queue, CMLHCQSignal* signal);
extern int cml_hcq_cuda_queue_wait(CMLHCQQueue* queue, CMLHCQSignal* signal);
extern int cml_hcq_cuda_signal_wait_cpu(CMLHCQSignal* signal, uint64_t timeout_ms);
#endif

#ifdef CML_HAS_OPENCL
extern CMLHCQQueue* cml_hcq_opencl_queue_create(void);
extern void cml_hcq_opencl_queue_destroy(CMLHCQQueue* queue);
extern int cml_hcq_opencl_submit_kernel(CMLHCQQueue* queue, const CMLHCQKernelDesc* desc);
extern int cml_hcq_opencl_memcpy_h2d(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes);
extern int cml_hcq_opencl_memcpy_d2h(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes);
extern int cml_hcq_opencl_queue_synchronize(CMLHCQQueue* queue);
extern CMLHCQSignal* cml_hcq_opencl_signal_create(void);
extern void cml_hcq_opencl_signal_destroy(CMLHCQSignal* signal);
extern int cml_hcq_opencl_signal_record(CMLHCQQueue* queue, CMLHCQSignal* signal);
extern int cml_hcq_opencl_queue_wait(CMLHCQQueue* queue, CMLHCQSignal* signal);
extern int cml_hcq_opencl_signal_wait_cpu(CMLHCQSignal* signal, uint64_t timeout_ms);
#endif

extern int cml_hcq_vulkan_queue_init(CMLHCQQueue* queue);
extern void cml_hcq_vulkan_queue_destroy(CMLHCQQueue* queue);
extern int cml_hcq_vulkan_submit_kernel(CMLHCQQueue* queue, const CMLHCQKernelDesc* desc);
extern int cml_hcq_vulkan_memcpy_h2d(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes);
extern int cml_hcq_vulkan_memcpy_d2h(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes);
extern int cml_hcq_vulkan_synchronize(CMLHCQQueue* queue);
extern int cml_hcq_vulkan_signal_create(CMLHCQSignal* signal);
extern void cml_hcq_vulkan_signal_destroy(CMLHCQSignal* signal);
extern int cml_hcq_vulkan_signal_wait(CMLHCQSignal* signal, uint64_t timeout_ms);

extern CMLHCQQueue* cml_hcq_nv_queue_create(void);
extern void cml_hcq_nv_queue_destroy(CMLHCQQueue* queue);
extern int cml_hcq_nv_submit_kernel(CMLHCQQueue* queue, const CMLHCQKernelDesc* desc);
extern int cml_hcq_nv_memcpy_h2d(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes);
extern int cml_hcq_nv_memcpy_d2h(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes);
extern int cml_hcq_nv_queue_synchronize(CMLHCQQueue* queue);
extern CMLHCQSignal* cml_hcq_nv_signal_create(void);
extern void cml_hcq_nv_signal_destroy(CMLHCQSignal* signal);
extern int cml_hcq_nv_signal_record(CMLHCQQueue* queue, CMLHCQSignal* signal);
extern int cml_hcq_nv_queue_wait(CMLHCQQueue* queue, CMLHCQSignal* signal);
extern int cml_hcq_nv_signal_wait_cpu(CMLHCQSignal* signal, uint64_t timeout_ms);

extern int cml_hcq_am_queue_init(CMLHCQQueue* queue);
extern void cml_hcq_am_queue_destroy(CMLHCQQueue* queue);
extern int cml_hcq_am_submit_kernel(CMLHCQQueue* queue, const CMLHCQKernelDesc* desc);
extern int cml_hcq_am_memcpy_h2d(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes);
extern int cml_hcq_am_memcpy_d2h(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes);
extern int cml_hcq_am_synchronize(CMLHCQQueue* queue);
extern int cml_hcq_am_signal_create(CMLHCQSignal* signal);
extern void cml_hcq_am_signal_destroy(CMLHCQSignal* signal);
extern int cml_hcq_am_signal_record(CMLHCQQueue* queue, CMLHCQSignal* signal);
extern int cml_hcq_am_queue_wait(CMLHCQQueue* queue, CMLHCQSignal* signal);
extern int cml_hcq_am_signal_wait(CMLHCQSignal* signal, uint64_t timeout_ms);

static CMLHCQQueue* hcq_vulkan_queue_create(void) {
    CMLHCQQueue* q = calloc(1, sizeof(CMLHCQQueue));
    if (!q)
        return NULL;
    q->backend = CML_HCQ_VULKAN;
    if (cml_hcq_vulkan_queue_init(q) != 0) {
        free(q);
        return NULL;
    }
    return q;
}

static CMLHCQQueue* hcq_am_queue_create(void) {
    CMLHCQQueue* q = calloc(1, sizeof(CMLHCQQueue));
    if (!q)
        return NULL;
    q->backend = CML_HCQ_AM;
    if (cml_hcq_am_queue_init(q) != 0) {
        free(q);
        return NULL;
    }
    return q;
}

static int hcq_vulkan_signal_record(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    (void)queue;
    (void)signal;
    return 0;
}

static int hcq_vulkan_queue_wait(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    (void)queue;
    (void)signal;
    return 0;
}

static CMLHCQSignal* hcq_vulkan_signal_create(void) {
    CMLHCQSignal* s = calloc(1, sizeof(CMLHCQSignal));
    if (!s)
        return NULL;
    s->backend = CML_HCQ_VULKAN;
    if (cml_hcq_vulkan_signal_create(s) != 0) {
        free(s);
        return NULL;
    }
    return s;
}

static CMLHCQSignal* hcq_am_signal_create(void) {
    CMLHCQSignal* s = calloc(1, sizeof(CMLHCQSignal));
    if (!s)
        return NULL;
    s->backend = CML_HCQ_AM;
    if (cml_hcq_am_signal_create(s) != 0) {
        free(s);
        return NULL;
    }
    return s;
}

#ifdef CML_HAS_CUDA
static const CMLHCQBackendOps g_hcq_cuda_ops = {
    .name = "CUDA",
    .queue_create = cml_hcq_cuda_queue_create,
    .queue_destroy = cml_hcq_cuda_queue_destroy,
    .submit_kernel = cml_hcq_cuda_submit_kernel,
    .memcpy_h2d = cml_hcq_cuda_memcpy_h2d,
    .memcpy_d2h = cml_hcq_cuda_memcpy_d2h,
    .queue_synchronize = cml_hcq_cuda_queue_synchronize,
    .signal_create = cml_hcq_cuda_signal_create,
    .signal_destroy = cml_hcq_cuda_signal_destroy,
    .signal_record = cml_hcq_cuda_signal_record,
    .queue_wait = cml_hcq_cuda_queue_wait,
    .signal_wait_cpu = cml_hcq_cuda_signal_wait_cpu,
};
#endif

#ifdef CML_HAS_OPENCL
static const CMLHCQBackendOps g_hcq_opencl_ops = {
    .name = "OpenCL",
    .queue_create = cml_hcq_opencl_queue_create,
    .queue_destroy = cml_hcq_opencl_queue_destroy,
    .submit_kernel = cml_hcq_opencl_submit_kernel,
    .memcpy_h2d = cml_hcq_opencl_memcpy_h2d,
    .memcpy_d2h = cml_hcq_opencl_memcpy_d2h,
    .queue_synchronize = cml_hcq_opencl_queue_synchronize,
    .signal_create = cml_hcq_opencl_signal_create,
    .signal_destroy = cml_hcq_opencl_signal_destroy,
    .signal_record = cml_hcq_opencl_signal_record,
    .queue_wait = cml_hcq_opencl_queue_wait,
    .signal_wait_cpu = cml_hcq_opencl_signal_wait_cpu,
};
#endif

static const CMLHCQBackendOps g_hcq_vulkan_ops = {
    .name = "Vulkan",
    .queue_create = hcq_vulkan_queue_create,
    .queue_destroy = cml_hcq_vulkan_queue_destroy,
    .submit_kernel = cml_hcq_vulkan_submit_kernel,
    .memcpy_h2d = cml_hcq_vulkan_memcpy_h2d,
    .memcpy_d2h = cml_hcq_vulkan_memcpy_d2h,
    .queue_synchronize = cml_hcq_vulkan_synchronize,
    .signal_create = hcq_vulkan_signal_create,
    .signal_destroy = cml_hcq_vulkan_signal_destroy,
    .signal_record = hcq_vulkan_signal_record,
    .queue_wait = hcq_vulkan_queue_wait,
    .signal_wait_cpu = cml_hcq_vulkan_signal_wait,
};

static const CMLHCQBackendOps g_hcq_nv_ops = {
    .name = "NV",
    .queue_create = cml_hcq_nv_queue_create,
    .queue_destroy = cml_hcq_nv_queue_destroy,
    .submit_kernel = cml_hcq_nv_submit_kernel,
    .memcpy_h2d = cml_hcq_nv_memcpy_h2d,
    .memcpy_d2h = cml_hcq_nv_memcpy_d2h,
    .queue_synchronize = cml_hcq_nv_queue_synchronize,
    .signal_create = cml_hcq_nv_signal_create,
    .signal_destroy = cml_hcq_nv_signal_destroy,
    .signal_record = cml_hcq_nv_signal_record,
    .queue_wait = cml_hcq_nv_queue_wait,
    .signal_wait_cpu = cml_hcq_nv_signal_wait_cpu,
};

static const CMLHCQBackendOps g_hcq_am_ops = {
    .name = "AM",
    .queue_create = hcq_am_queue_create,
    .queue_destroy = cml_hcq_am_queue_destroy,
    .submit_kernel = cml_hcq_am_submit_kernel,
    .memcpy_h2d = cml_hcq_am_memcpy_h2d,
    .memcpy_d2h = cml_hcq_am_memcpy_d2h,
    .queue_synchronize = cml_hcq_am_synchronize,
    .signal_create = hcq_am_signal_create,
    .signal_destroy = cml_hcq_am_signal_destroy,
    .signal_record = cml_hcq_am_signal_record,
    .queue_wait = cml_hcq_am_queue_wait,
    .signal_wait_cpu = cml_hcq_am_signal_wait,
};

const CMLHCQBackendOps* cml_hcq_backend_ops(CMLHCQBackendType backend) {
    switch (backend) {
    case CML_HCQ_CPU:
        return &g_hcq_cpu_ops;
#ifdef CML_HAS_CUDA
    case CML_HCQ_CUDA:
        return &g_hcq_cuda_ops;
#endif
#ifdef CML_HAS_OPENCL
    case CML_HCQ_OPENCL:
        return &g_hcq_opencl_ops;
#endif
    case CML_HCQ_VULKAN:
        return &g_hcq_vulkan_ops;
    case CML_HCQ_NV:
        return &g_hcq_nv_ops;
    case CML_HCQ_AM:
        return &g_hcq_am_ops;
    default:
        return NULL;
    }
}
