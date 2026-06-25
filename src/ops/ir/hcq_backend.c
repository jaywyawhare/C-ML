#include "ops/ir/hcq_backend.h"
#include "core/logging.h"
#include <stdlib.h>

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
