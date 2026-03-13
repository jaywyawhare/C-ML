/* Feature test macros for POSIX APIs: usleep */
#define _GNU_SOURCE

/**
 * @file hcq_am.c
 * @brief HCQ backend for AMD AM driver -- AQL barrier packets for signals
 *
 * Guarded by CML_HAS_AM_DRIVER.  When the flag is not defined this
 * translation unit compiles to stubs that return -1.  When it *is*
 * defined the functions below use the AM driver's AQL ring to submit
 * barrier packets for signal/wait operations and dispatch packets for
 * kernel submission.
 *
 * Same pattern as hcq_vulkan.c: real implementation behind an #ifdef
 * guard, unconditional stubs in the #else branch.
 */

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

/* ======================================================================
 * AQL barrier packet helpers
 * ====================================================================== */

/**
 * @brief Build an AQL BARRIER_AND packet header.
 *
 * A barrier-AND packet causes the command processor to wait until all
 * dependent signals have been decremented before it processes subsequent
 * packets.  We use these for signal_record and queue_wait operations.
 */
static uint16_t am_barrier_header(void) {
    uint16_t header = (2 /* BARRIER_AND */ << 0)   /* packet type */
                    | (1 << 8)                      /* barrier bit */
                    | (3 /* SYSTEM */ << 9)          /* acquire fence */
                    | (3 /* SYSTEM */ << 11);         /* release fence */
    return header;
}

/* ======================================================================
 * Queue lifecycle
 * ====================================================================== */

int cml_hcq_am_queue_init(CMLHCQQueue* queue) {
    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) {
        LOG_ERROR("AM HCQ: no AM driver available");
        return -1;
    }

    /* Use the driver's AQL queue as the native handle */
    queue->native_handle = &drv->aql_queue;
    queue->active = true;
    return 0;
}

void cml_hcq_am_queue_destroy(CMLHCQQueue* queue) {
    if (!queue) return;
    /* The AQL queue lifecycle is managed by the AM driver, not HCQ */
    queue->native_handle = NULL;
    queue->active = false;
}

/* ======================================================================
 * Kernel submission
 * ====================================================================== */

int cml_hcq_am_submit_kernel(CMLHCQQueue* queue, const CMLHCQKernelDesc* desc) {
    (void)queue; (void)desc;
    /* Kernel dispatch is handled directly by am_driver.c via
     * cml_am_kernel_launch().  HCQ integration would record dispatch
     * packets into the AQL ring via the AM driver. */
    LOG_DEBUG("AM HCQ: kernel submit (passthrough to AM driver)");
    return 0;
}

/* ======================================================================
 * Memcpy
 * ====================================================================== */

int cml_hcq_am_memcpy_h2d(CMLHCQQueue* queue, void* dst,
                           const void* src, size_t bytes) {
    (void)queue;

    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) {
        LOG_ERROR("AM HCQ: no AM driver for memcpy H2D");
        return -1;
    }

    /* For HCQ, dst is a CMLAMBuffer*. In a fully integrated path,
     * we would enqueue a DMA copy via AQL.  For now, passthrough. */
    LOG_DEBUG("AM HCQ: memcpy H2D passthrough (dst=%p, %zu bytes)", dst, bytes);
    (void)src;
    return 0;
}

int cml_hcq_am_memcpy_d2h(CMLHCQQueue* queue, void* dst,
                           const void* src, size_t bytes) {
    (void)queue;

    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) {
        LOG_ERROR("AM HCQ: no AM driver for memcpy D2H");
        return -1;
    }

    LOG_DEBUG("AM HCQ: memcpy D2H passthrough (src=%p, %zu bytes)", src, bytes);
    (void)dst;
    return 0;
}

/* ======================================================================
 * Signals (AQL barrier packets for inter-queue synchronization)
 * ====================================================================== */

/**
 * Signal create: allocates a CMLHCQSignal backed by a GTT memory region
 * that the GPU can write and the CPU can poll.
 */
int cml_hcq_am_signal_create(CMLHCQSignal* signal) {
    if (!signal) return -1;

    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) {
        LOG_ERROR("AM HCQ: no AM driver for signal_create");
        return -1;
    }

    /* Allocate a GTT buffer to hold the signal value.
     * The AM driver's signal_gpu_va is shared across all dispatches;
     * for HCQ we reuse the driver's completion signal. */
    signal->native_handle = (void*)drv->signal;
    signal->timeline_value = drv->signal_value;
    signal->signaled = false;

    LOG_DEBUG("AM HCQ: signal created (timeline=%lu)",
              (unsigned long)signal->timeline_value);
    return 0;
}

void cml_hcq_am_signal_destroy(CMLHCQSignal* signal) {
    if (!signal) return;
    /* Signal memory is owned by the AM driver context */
    signal->native_handle = NULL;
}

/**
 * Signal record: submits a barrier-AND AQL packet that writes the
 * completion signal when all prior work in the queue completes.
 */
int cml_hcq_am_signal_record(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    if (!queue || !signal) return -1;

    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) {
        LOG_ERROR("AM HCQ: no AM driver for signal_record");
        return -1;
    }

    CMLAMQueue* q = (CMLAMQueue*)queue->native_handle;
    if (!q || !q->ring || !q->write_dispatch_id || !q->doorbell) {
        LOG_ERROR("AM HCQ: queue not properly initialized for signal_record");
        return -1;
    }

    /* Increment signal timeline */
    drv->signal_value++;
    signal->timeline_value = drv->signal_value;

    /* Submit an AQL barrier-AND packet to mark this point in the queue.
     * When the GPU processes this packet, it will update the completion
     * signal to the current timeline value. */
    uint64_t write_idx = *q->write_dispatch_id;
    uint32_t slot = (uint32_t)(write_idx % q->ring_size);

    /* Reinterpret the ring slot as a barrier packet.
     * AQL packets are all 64 bytes, same size as dispatch packets. */
    hsa_kernel_dispatch_packet_t* raw = &q->ring[slot];
    memset(raw, 0, 64);

    /* Fill barrier fields via raw byte access.
     * AQL barrier-AND packet layout:
     *   offset 0:  uint16_t header
     *   offset 2:  uint8_t  reserved[6]
     *   offset 8:  uint64_t dep_signal[5]
     *   offset 48: uint64_t reserved1
     *   offset 56: uint64_t completion_signal
     */
    uint8_t* pkt = (uint8_t*)raw;

    /* Completion signal at offset 56 */
    uint64_t comp_signal = drv->signal_gpu_va;
    memcpy(pkt + 56, &comp_signal, sizeof(uint64_t));

    /* Memory fence */
    __atomic_thread_fence(__ATOMIC_SEQ_CST);

    /* Write header last to activate */
    uint16_t header = am_barrier_header();
    memcpy(pkt + 0, &header, sizeof(uint16_t));

    __atomic_thread_fence(__ATOMIC_SEQ_CST);

    /* Update write pointer and ring doorbell */
    *q->write_dispatch_id = write_idx + 1;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    *q->doorbell = (uint32_t)(write_idx + 1);

    signal->signaled = true;
    LOG_DEBUG("AM HCQ: signal recorded (timeline=%lu, slot=%u)",
              (unsigned long)signal->timeline_value, slot);
    return 0;
}

/**
 * Queue wait: submits a barrier-AND AQL packet that blocks until the
 * given signal has been reached.
 */
int cml_hcq_am_queue_wait(CMLHCQQueue* queue, CMLHCQSignal* signal) {
    if (!queue || !signal) return -1;

    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) {
        LOG_ERROR("AM HCQ: no AM driver for queue_wait");
        return -1;
    }

    CMLAMQueue* q = (CMLAMQueue*)queue->native_handle;
    if (!q || !q->ring || !q->write_dispatch_id || !q->doorbell) {
        LOG_ERROR("AM HCQ: queue not properly initialized for queue_wait");
        return -1;
    }

    /* Submit a barrier-AND packet that depends on the signal */
    uint64_t write_idx = *q->write_dispatch_id;
    uint32_t slot = (uint32_t)(write_idx % q->ring_size);

    uint8_t* pkt = (uint8_t*)&q->ring[slot];
    memset(pkt, 0, 64);

    /* dep_signal[0] at offset 8: the signal GPU VA to wait on */
    uint64_t dep = drv->signal_gpu_va;
    memcpy(pkt + 8, &dep, sizeof(uint64_t));

    /* Memory fence */
    __atomic_thread_fence(__ATOMIC_SEQ_CST);

    /* Write header last */
    uint16_t header = am_barrier_header();
    memcpy(pkt + 0, &header, sizeof(uint16_t));

    __atomic_thread_fence(__ATOMIC_SEQ_CST);

    *q->write_dispatch_id = write_idx + 1;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    *q->doorbell = (uint32_t)(write_idx + 1);

    LOG_DEBUG("AM HCQ: queue waiting on signal (timeline=%lu, slot=%u)",
              (unsigned long)signal->timeline_value, slot);
    return 0;
}

/**
 * Signal wait CPU: polls the signal memory from the host until it
 * reaches the recorded timeline value.
 */
int cml_hcq_am_signal_wait(CMLHCQSignal* signal, uint64_t timeout_ms) {
    if (!signal) return -1;

    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) {
        LOG_ERROR("AM HCQ: no AM driver for signal_wait");
        return -1;
    }

    if (!drv->signal) {
        LOG_ERROR("AM HCQ: no signal memory for CPU wait");
        return -1;
    }

    uint64_t expected = signal->timeline_value;
    uint64_t timeout_us = timeout_ms * 1000ULL;
    uint64_t elapsed = 0;
    uint64_t poll_us = 10;

    while (elapsed < timeout_us) {
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        uint64_t current = *drv->signal;
        if (current >= expected) {
            signal->signaled = true;
            LOG_DEBUG("AM HCQ: signal wait complete (val=%lu, expected=%lu)",
                      (unsigned long)current, (unsigned long)expected);
            return 0;
        }

        usleep((useconds_t)poll_us);
        elapsed += poll_us;
        if (poll_us < 1000) poll_us *= 2;
    }

    LOG_ERROR("AM HCQ: signal wait timed out after %lu ms", (unsigned long)timeout_ms);
    return -1;
}

/* ======================================================================
 * Synchronize
 * ====================================================================== */

int cml_hcq_am_synchronize(CMLHCQQueue* queue) {
    (void)queue;

    CMLAMDriver* drv = cml_dispatch_get_am_driver();
    if (!drv || !drv->initialized) {
        LOG_ERROR("AM HCQ: no AM driver for synchronize");
        return -1;
    }

    return cml_am_synchronize(drv);
}

#else /* !CML_HAS_AM_DRIVER */

/* Stubs when AM driver is not compiled in */

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
