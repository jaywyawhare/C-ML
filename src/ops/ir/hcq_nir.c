/**
 * @file hcq_nir.c
 * @brief Hardware Command Queues -- NIR/Mesa backend
 *
 * Thin wrapper that forwards HCQ operations to the Vulkan HCQ backend.
 * NIR compiles to SPIR-V which is consumed by the Vulkan pipeline, so
 * queue management and kernel dispatch reuse the existing hcq_vulkan
 * infrastructure.
 *
 * Guarded by CML_HAS_NIR.  When the flag is not defined this translation
 * unit compiles to stubs that return -1.
 */

#include "ops/ir/hcq.h"
#include "core/logging.h"

#include <stdlib.h>
#include <stdint.h>

#ifdef CML_HAS_NIR

/* ── Forward declarations for hcq_vulkan functions ───────────────────── */

extern int  cml_hcq_vulkan_queue_init(CMLHCQQueue* queue);
extern void cml_hcq_vulkan_queue_destroy(CMLHCQQueue* queue);
extern int  cml_hcq_vulkan_submit_kernel(CMLHCQQueue* queue,
                                          const CMLHCQKernelDesc* desc);
extern int  cml_hcq_vulkan_memcpy_h2d(CMLHCQQueue* queue, void* dst,
                                       const void* src, size_t bytes);
extern int  cml_hcq_vulkan_memcpy_d2h(CMLHCQQueue* queue, void* dst,
                                       const void* src, size_t bytes);
extern int  cml_hcq_vulkan_signal_create(CMLHCQSignal* signal);
extern void cml_hcq_vulkan_signal_destroy(CMLHCQSignal* signal);
extern int  cml_hcq_vulkan_signal_wait(CMLHCQSignal* signal, uint64_t timeout_ms);
extern int  cml_hcq_vulkan_synchronize(CMLHCQQueue* queue);

/* ══════════════════════════════════════════════════════════════════════════
 * NIR HCQ API -- forwards to Vulkan HCQ
 * ══════════════════════════════════════════════════════════════════════════ */

int cml_hcq_nir_queue_init(CMLHCQQueue* queue) {
    LOG_DEBUG("HCQ NIR: queue init (forwarding to Vulkan HCQ)");
    return cml_hcq_vulkan_queue_init(queue);
}

void cml_hcq_nir_queue_destroy(CMLHCQQueue* queue) {
    LOG_DEBUG("HCQ NIR: queue destroy (forwarding to Vulkan HCQ)");
    cml_hcq_vulkan_queue_destroy(queue);
}

int cml_hcq_nir_submit_kernel(CMLHCQQueue* queue,
                               const CMLHCQKernelDesc* desc) {
    LOG_DEBUG("HCQ NIR: submit kernel (forwarding to Vulkan HCQ)");
    return cml_hcq_vulkan_submit_kernel(queue, desc);
}

int cml_hcq_nir_memcpy_h2d(CMLHCQQueue* queue, void* dst,
                            const void* src, size_t bytes) {
    return cml_hcq_vulkan_memcpy_h2d(queue, dst, src, bytes);
}

int cml_hcq_nir_memcpy_d2h(CMLHCQQueue* queue, void* dst,
                            const void* src, size_t bytes) {
    return cml_hcq_vulkan_memcpy_d2h(queue, dst, src, bytes);
}

int cml_hcq_nir_signal_create(CMLHCQSignal* signal) {
    return cml_hcq_vulkan_signal_create(signal);
}

void cml_hcq_nir_signal_destroy(CMLHCQSignal* signal) {
    cml_hcq_vulkan_signal_destroy(signal);
}

int cml_hcq_nir_signal_wait(CMLHCQSignal* signal, uint64_t timeout_ms) {
    return cml_hcq_vulkan_signal_wait(signal, timeout_ms);
}

int cml_hcq_nir_synchronize(CMLHCQQueue* queue) {
    return cml_hcq_vulkan_synchronize(queue);
}

#else /* !CML_HAS_NIR */

/* ── Stubs when NIR is not compiled in ─────────────────────────────────── */

int  cml_hcq_nir_queue_init(CMLHCQQueue* q)    { (void)q; return -1; }
void cml_hcq_nir_queue_destroy(CMLHCQQueue* q)  { (void)q; }
int  cml_hcq_nir_submit_kernel(CMLHCQQueue* q, const CMLHCQKernelDesc* d)
    { (void)q; (void)d; return -1; }
int  cml_hcq_nir_memcpy_h2d(CMLHCQQueue* q, void* d, const void* s, size_t n)
    { (void)q; (void)d; (void)s; (void)n; return -1; }
int  cml_hcq_nir_memcpy_d2h(CMLHCQQueue* q, void* d, const void* s, size_t n)
    { (void)q; (void)d; (void)s; (void)n; return -1; }
int  cml_hcq_nir_signal_create(CMLHCQSignal* s)  { (void)s; return -1; }
void cml_hcq_nir_signal_destroy(CMLHCQSignal* s) { (void)s; }
int  cml_hcq_nir_signal_wait(CMLHCQSignal* s, uint64_t t)
    { (void)s; (void)t; return -1; }
int  cml_hcq_nir_synchronize(CMLHCQQueue* q)     { (void)q; return -1; }

#endif /* CML_HAS_NIR */
