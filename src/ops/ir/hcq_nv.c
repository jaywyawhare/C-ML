#include "ops/ir/hcq.h"
#include "core/logging.h"

#ifdef CML_HAS_NV_DRIVER

#include "ops/ir/gpu/nv_driver.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern CMLNVDriver* cml_dispatch_get_nv_driver(void);

typedef struct {
    uint64_t submit_count;
    uint64_t last_sem_value;
} NVQueueData;

typedef struct {
    uint64_t target_value;
    uint64_t sem_gpu_va;
} NVSignalData;

CMLHCQQueue* cml_hcq_nv_queue_create(void) {
    CMLNVDriver *nv = cml_dispatch_get_nv_driver();
    if (!nv || !nv->initialized) {
        LOG_ERROR("NV HCQ: no NV driver available");
        return NULL;
    }

    CMLHCQQueue *queue = (CMLHCQQueue *)calloc(1, sizeof(CMLHCQQueue));
    if (!queue) return NULL;

    NVQueueData *qd = (NVQueueData *)calloc(1, sizeof(NVQueueData));
    if (!qd) {
        free(queue);
        return NULL;
    }

    queue->backend       = CML_HCQ_NV;
    queue->native_handle = (void *)qd;
    queue->active        = true;
    return queue;
}

void cml_hcq_nv_queue_destroy(CMLHCQQueue *queue) {
    if (!queue) return;
    free(queue->native_handle);
    queue->native_handle = NULL;
    queue->active = false;
    free(queue);
}

int cml_hcq_nv_submit_kernel(CMLHCQQueue *queue,
                              const CMLHCQKernelDesc *desc) {
    if (!queue || !desc) return -1;

    CMLNVDriver *nv = cml_dispatch_get_nv_driver();
    if (!nv || !nv->initialized) return -1;

    NVQueueData *qd = (NVQueueData *)queue->native_handle;
    if (!qd) return -1;

    uint32_t grid[3]  = { (uint32_t)desc->grid[0],
                           (uint32_t)desc->grid[1],
                           (uint32_t)desc->grid[2] };
    uint32_t block[3] = { (uint32_t)desc->block[0],
                           (uint32_t)desc->block[1],
                           (uint32_t)desc->block[2] };

    int ret = cml_nv_kernel_launch(nv, (CMLNVKernel *)desc->compiled_kernel,
                                   grid, block, desc->args, desc->num_args);
    if (ret == 0) {
        qd->submit_count++;
        qd->last_sem_value = nv->semaphore_value;
    }

    return ret;
}

int cml_hcq_nv_memcpy_h2d(CMLHCQQueue *queue, void *dst,
                            const void *src, size_t bytes) {
    if (!queue || !dst || !src || bytes == 0) return -1;

    CMLNVDriver *nv = cml_dispatch_get_nv_driver();
    if (!nv || !nv->initialized) return -1;

    NVQueueData *qd = (NVQueueData *)queue->native_handle;
    CMLNVBuffer *dst_buf = (CMLNVBuffer *)dst;

    int ret = cml_nv_buffer_upload(nv, dst_buf, src, bytes);
    if (ret == 0 && qd)
        qd->submit_count++;

    return ret;
}

int cml_hcq_nv_memcpy_d2h(CMLHCQQueue *queue, void *dst,
                            const void *src, size_t bytes) {
    if (!queue || !dst || !src || bytes == 0) return -1;

    CMLNVDriver *nv = cml_dispatch_get_nv_driver();
    if (!nv || !nv->initialized) return -1;

    NVQueueData *qd = (NVQueueData *)queue->native_handle;
    CMLNVBuffer *src_buf = (CMLNVBuffer *)src;

    int ret = cml_nv_buffer_download(nv, src_buf, dst, bytes);
    if (ret == 0 && qd)
        qd->submit_count++;

    return ret;
}

CMLHCQSignal* cml_hcq_nv_signal_create(void) {
    CMLNVDriver *nv = cml_dispatch_get_nv_driver();
    if (!nv || !nv->initialized) return NULL;

    CMLHCQSignal *signal = (CMLHCQSignal *)calloc(1, sizeof(CMLHCQSignal));
    if (!signal) return NULL;

    NVSignalData *sd = (NVSignalData *)calloc(1, sizeof(NVSignalData));
    if (!sd) {
        free(signal);
        return NULL;
    }

    sd->sem_gpu_va = nv->semaphore_gpu_va;

    signal->backend       = CML_HCQ_NV;
    signal->native_handle = (void *)sd;
    signal->signaled      = false;
    return signal;
}

void cml_hcq_nv_signal_destroy(CMLHCQSignal *signal) {
    if (!signal) return;
    free(signal->native_handle);
    signal->native_handle = NULL;
    free(signal);
}

int cml_hcq_nv_signal_record(CMLHCQQueue *queue, CMLHCQSignal *signal) {
    if (!queue || !signal) return -1;

    NVQueueData  *qd = (NVQueueData *)queue->native_handle;
    NVSignalData *sd = (NVSignalData *)signal->native_handle;
    if (!qd || !sd) return -1;

    CMLNVDriver *nv = cml_dispatch_get_nv_driver();
    if (!nv || !nv->initialized) return -1;

    sd->target_value = nv->semaphore_value;
    sd->sem_gpu_va = nv->semaphore_gpu_va;
    signal->signaled = false;
    signal->timeline_value = sd->target_value;

    return 0;
}

int cml_hcq_nv_queue_wait(CMLHCQQueue *queue, CMLHCQSignal *signal) {
    if (!queue || !signal) return -1;

    NVSignalData *sd = (NVSignalData *)signal->native_handle;
    if (!sd) return -1;

    CMLNVDriver *nv = cml_dispatch_get_nv_driver();
    if (!nv || !nv->initialized) return -1;

    return cml_nv_gpu_wait_semaphore(nv, sd->sem_gpu_va, (uint32_t)sd->target_value);
}

int cml_hcq_nv_signal_wait_cpu(CMLHCQSignal *signal, uint64_t timeout_ms) {
    if (!signal) return -1;

    NVSignalData *sd = (NVSignalData *)signal->native_handle;
    if (!sd) return -1;

    CMLNVDriver *nv = cml_dispatch_get_nv_driver();
    if (!nv || !nv->initialized) return -1;

    if (!nv->semaphore) return -1;

    uint64_t deadline = 0;
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        deadline = (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
        deadline += timeout_ms;
    }

    uint32_t target = (uint32_t)sd->target_value;

    while (*nv->semaphore < target) {
        __sync_synchronize();

        struct timespec now_ts;
        clock_gettime(CLOCK_MONOTONIC, &now_ts);
        uint64_t now = (uint64_t)now_ts.tv_sec * 1000ULL + (uint64_t)now_ts.tv_nsec / 1000000ULL;

        if (now >= deadline) {
            LOG_ERROR("NV HCQ: signal wait timed out (current=%u, target=%u)",
                      *nv->semaphore, target);
            return -1;
        }

        struct timespec sleep_ts = {0, 100000};
        nanosleep(&sleep_ts, NULL);
    }

    signal->signaled = true;
    return 0;
}

int cml_hcq_nv_queue_synchronize(CMLHCQQueue *queue) {
    if (!queue) return -1;

    CMLNVDriver *nv = cml_dispatch_get_nv_driver();
    if (!nv || !nv->initialized) return -1;

    return cml_nv_synchronize(nv);
}

#else /* !CML_HAS_NV_DRIVER */

CMLHCQQueue* cml_hcq_nv_queue_create(void) { return NULL; }
void         cml_hcq_nv_queue_destroy(CMLHCQQueue* q) { (void)q; }

int cml_hcq_nv_submit_kernel(CMLHCQQueue* q, const CMLHCQKernelDesc* d)
    { (void)q; (void)d; return -1; }

int cml_hcq_nv_memcpy_h2d(CMLHCQQueue* q, void* d, const void* s, size_t n)
    { (void)q; (void)d; (void)s; (void)n; return -1; }

int cml_hcq_nv_memcpy_d2h(CMLHCQQueue* q, void* d, const void* s, size_t n)
    { (void)q; (void)d; (void)s; (void)n; return -1; }

CMLHCQSignal* cml_hcq_nv_signal_create(void) { return NULL; }
void          cml_hcq_nv_signal_destroy(CMLHCQSignal* s) { (void)s; }

int cml_hcq_nv_signal_record(CMLHCQQueue* q, CMLHCQSignal* s)
    { (void)q; (void)s; return -1; }

int cml_hcq_nv_queue_wait(CMLHCQQueue* q, CMLHCQSignal* s)
    { (void)q; (void)s; return -1; }

int cml_hcq_nv_signal_wait_cpu(CMLHCQSignal* s, uint64_t t)
    { (void)s; (void)t; return -1; }

int cml_hcq_nv_queue_synchronize(CMLHCQQueue* q) { (void)q; return -1; }

#endif /* CML_HAS_NV_DRIVER */
