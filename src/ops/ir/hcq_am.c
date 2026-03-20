#define _GNU_SOURCE

#include "ops/ir/hcq.h"
#include "core/logging.h"

#ifdef CML_HAS_AM_DRIVER

#include "ops/ir/gpu/am_driver.h"

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

extern CMLAMDriver* cml_dispatch_get_am_driver(void);

static uint16_t am_barrier_header(void) {
    uint16_t header = (2 << 0)
                    | (1 << 8)
                    | (3 << 9)
                    | (3 << 11);
    return header;
}

int cml_hcq_am_queue_init(CMLHCQQueue* queue) {
    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) {
        LOG_ERROR("AM HCQ: no AM driver available");
        return -1;
    }

    queue->native_handle = &drv->aql_queue;
    queue->active = true;
    return 0;
}

void cml_hcq_am_queue_destroy(CMLHCQQueue* queue) {
    if (!queue) return;
    queue->native_handle = NULL;
    queue->active = false;
}

int cml_hcq_am_submit_kernel(CMLHCQQueue* queue, const CMLHCQKernelDesc* desc) {
    if (!queue || !desc) return -1;

    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) return -1;

    CMLAMKernel* kernel = (CMLAMKernel*)desc->compiled_kernel;
    if (!kernel) return -1;

    uint32_t grid[3] = {
        (uint32_t)desc->grid[0],
        (uint32_t)desc->grid[1],
        (uint32_t)desc->grid[2]
    };
    uint32_t block[3] = {
        (uint32_t)desc->block[0],
        (uint32_t)desc->block[1],
        (uint32_t)desc->block[2]
    };

    return cml_am_kernel_launch(drv, kernel, grid, block,
                                desc->args, (uint32_t)(desc->num_args * sizeof(void*)));
}

int cml_hcq_am_memcpy_h2d(CMLHCQQueue* queue, void* dst,
                           const void* src, size_t bytes) {
    (void)queue;

    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) {
        LOG_ERROR("AM HCQ: no AM driver for memcpy H2D");
        return -1;
    }

    if (drv->has_sdma) {
        /* dst is a GPU VA; src is host memory.
         * Allocate a GTT staging buffer, copy host data in, then SDMA to dst. */
        CMLAMBuffer* staging = cml_am_buffer_create(drv, bytes, false);
        if (!staging) return -1;

        memcpy(staging->cpu_addr, src, bytes);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);

        uint64_t dst_va = (uint64_t)(uintptr_t)dst;
        int ret = cml_am_sdma_copy(drv, dst_va, staging->gpu_va, bytes);
        if (ret == 0)
            ret = cml_am_sdma_synchronize(drv);

        cml_am_buffer_free(drv, staging);
        return ret;
    }

    return -1;
}

int cml_hcq_am_memcpy_d2h(CMLHCQQueue* queue, void* dst,
                           const void* src, size_t bytes) {
    (void)queue;

    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) {
        LOG_ERROR("AM HCQ: no AM driver for memcpy D2H");
        return -1;
    }

    if (drv->has_sdma) {
        CMLAMBuffer* staging = cml_am_buffer_create(drv, bytes, false);
        if (!staging) return -1;

        uint64_t src_va = (uint64_t)(uintptr_t)src;
        int ret = cml_am_sdma_copy(drv, staging->gpu_va, src_va, bytes);
        if (ret == 0)
            ret = cml_am_sdma_synchronize(drv);

        if (ret == 0) {
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            memcpy(dst, staging->cpu_addr, bytes);
        }

        cml_am_buffer_free(drv, staging);
        return ret;
    }

    return -1;
}

int cml_hcq_am_signal_create(CMLHCQSignal* signal) {
    if (!signal) return -1;

    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) {
        LOG_ERROR("AM HCQ: no AM driver for signal_create");
        return -1;
    }

    CMLAMSignal* am_sig = cml_am_signal_create(drv, 0);
    if (!am_sig) {
        /* Fallback to shared driver signal */
        signal->native_handle = (void*)drv->signal;
        signal->timeline_value = drv->signal_value;
        signal->signaled = false;
        return 0;
    }

    signal->native_handle = am_sig;
    signal->timeline_value = 0;
    signal->signaled = false;
    return 0;
}

void cml_hcq_am_signal_destroy(CMLHCQSignal* signal) {
    if (!signal) return;

    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv) {
        signal->native_handle = NULL;
        return;
    }

    CMLAMSignal* am_sig = (CMLAMSignal*)signal->native_handle;
    if (am_sig && am_sig != (CMLAMSignal*)(void*)drv->signal) {
        cml_am_signal_free(drv, am_sig);
    }
    signal->native_handle = NULL;
}

int cml_hcq_am_signal_record(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    if (!queue || !signal) return -1;

    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) return -1;

    CMLAMQueue* q = (CMLAMQueue*)queue->native_handle;
    if (!q || !q->ring || !q->write_dispatch_id || !q->doorbell) return -1;

    /* Determine signal GPU VA */
    uint64_t comp_signal;
    CMLAMSignal* am_sig = (CMLAMSignal*)signal->native_handle;
    if (am_sig && am_sig != (CMLAMSignal*)(void*)drv->signal) {
        am_sig->target++;
        signal->timeline_value = am_sig->target;
        comp_signal = am_sig->gpu_va;
    } else {
        drv->signal_value++;
        signal->timeline_value = drv->signal_value;
        comp_signal = drv->signal_gpu_va;
    }

    uint64_t write_idx = *q->write_dispatch_id;
    uint32_t slot = (uint32_t)(write_idx % q->ring_size);

    uint8_t* pkt = (uint8_t*)&q->ring[slot];
    memset(pkt, 0, 64);

    memcpy(pkt + 56, &comp_signal, sizeof(uint64_t));

    __atomic_thread_fence(__ATOMIC_SEQ_CST);

    uint16_t header = am_barrier_header();
    memcpy(pkt + 0, &header, sizeof(uint16_t));

    __atomic_thread_fence(__ATOMIC_SEQ_CST);

    *q->write_dispatch_id = write_idx + 1;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    *q->doorbell = (uint32_t)(write_idx + 1);

    signal->signaled = true;
    return 0;
}

int cml_hcq_am_queue_wait(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    if (!queue || !signal) return -1;

    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) return -1;

    CMLAMQueue* q = (CMLAMQueue*)queue->native_handle;
    if (!q || !q->ring || !q->write_dispatch_id || !q->doorbell) return -1;

    uint64_t dep_va;
    CMLAMSignal* am_sig = (CMLAMSignal*)signal->native_handle;
    if (am_sig && am_sig != (CMLAMSignal*)(void*)drv->signal) {
        dep_va = am_sig->gpu_va;
    } else {
        dep_va = drv->signal_gpu_va;
    }

    uint64_t write_idx = *q->write_dispatch_id;
    uint32_t slot = (uint32_t)(write_idx % q->ring_size);

    uint8_t* pkt = (uint8_t*)&q->ring[slot];
    memset(pkt, 0, 64);

    memcpy(pkt + 8, &dep_va, sizeof(uint64_t));

    __atomic_thread_fence(__ATOMIC_SEQ_CST);

    uint16_t header = am_barrier_header();
    memcpy(pkt + 0, &header, sizeof(uint16_t));

    __atomic_thread_fence(__ATOMIC_SEQ_CST);

    *q->write_dispatch_id = write_idx + 1;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    *q->doorbell = (uint32_t)(write_idx + 1);
    return 0;
}

int cml_hcq_am_signal_wait(CMLHCQSignal* signal, uint64_t timeout_ms) {
    if (!signal) return -1;

    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) return -1;

    CMLAMSignal* am_sig = (CMLAMSignal*)signal->native_handle;
    if (am_sig && am_sig != (CMLAMSignal*)(void*)drv->signal) {
        uint64_t timeout_ns = timeout_ms * 1000000ULL;
        int ret = cml_am_signal_wait(am_sig, signal->timeline_value, timeout_ns);
        if (ret == 0) signal->signaled = true;
        return ret;
    }

    /* Fallback: poll shared driver signal */
    if (!drv->signal) return -1;

    uint64_t expected = signal->timeline_value;
    uint64_t timeout_us = timeout_ms * 1000ULL;
    uint64_t elapsed = 0;
    uint64_t poll_us = 10;

    while (elapsed < timeout_us) {
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        uint64_t current = *drv->signal;
        if (current >= expected) {
            signal->signaled = true;
            return 0;
        }

        usleep((useconds_t)poll_us);
        elapsed += poll_us;
        if (poll_us < 1000) poll_us *= 2;
    }

    LOG_ERROR("AM HCQ: signal wait timed out after %lu ms", (unsigned long)timeout_ms);
    return -1;
}

int cml_hcq_am_synchronize(CMLHCQQueue* queue) {
    (void)queue;

    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) return -1;

    return cml_am_synchronize(drv);
}

#else /* !CML_HAS_AM_DRIVER */

#include <stddef.h>
#include <stdint.h>

int  cml_hcq_am_queue_init(CMLHCQQueue* q)    { (void)q; return -1; }
void cml_hcq_am_queue_destroy(CMLHCQQueue* q)  { (void)q; }
int  cml_hcq_am_submit_kernel(CMLHCQQueue* q, const CMLHCQKernelDesc* d)
    { (void)q; (void)d; return -1; }
int  cml_hcq_am_memcpy_h2d(CMLHCQQueue* q, void* d, const void* s, size_t n)
    { (void)q; (void)d; (void)s; (void)n; return -1; }
int  cml_hcq_am_memcpy_d2h(CMLHCQQueue* q, void* d, const void* s, size_t n)
    { (void)q; (void)d; (void)s; (void)n; return -1; }
int  cml_hcq_am_signal_create(CMLHCQSignal* s)  { (void)s; return -1; }
void cml_hcq_am_signal_destroy(CMLHCQSignal* s) { (void)s; }
int  cml_hcq_am_signal_record(CMLHCQQueue* q, CMLHCQSignal* s)
    { (void)q; (void)s; return -1; }
int  cml_hcq_am_queue_wait(CMLHCQQueue* q, CMLHCQSignal* s)
    { (void)q; (void)s; return -1; }
int  cml_hcq_am_signal_wait(CMLHCQSignal* s, uint64_t t)
    { (void)s; (void)t; return -1; }
int  cml_hcq_am_synchronize(CMLHCQQueue* q)     { (void)q; return -1; }

#endif /* CML_HAS_AM_DRIVER */
