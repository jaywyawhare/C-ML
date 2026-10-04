/* Forwards HCQ operations to the Vulkan HCQ backend since NIR
 * compiles to SPIR-V consumed by the Vulkan pipeline. */

#include "ops/ir/hcq.h"
#include "core/logging.h"

#include <stdlib.h>
#include <stdint.h>

#ifdef CML_HAS_NIR

extern int cml_hcq_vulkan_queue_init(CMLHCQQueue* queue);
extern void cml_hcq_vulkan_queue_destroy(CMLHCQQueue* queue);
extern int cml_hcq_vulkan_submit_kernel(CMLHCQQueue* queue, const CMLHCQKernelDesc* desc);
extern int cml_hcq_vulkan_memcpy_h2d(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes);
extern int cml_hcq_vulkan_memcpy_d2h(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes);
extern int cml_hcq_vulkan_signal_create(CMLHCQSignal* signal);
extern void cml_hcq_vulkan_signal_destroy(CMLHCQSignal* signal);
extern int cml_hcq_vulkan_signal_wait(CMLHCQSignal* signal, uint64_t timeout_ms);
extern int cml_hcq_vulkan_synchronize(CMLHCQQueue* queue);

/** Init a NIR queue by forwarding to the Vulkan HCQ backend. */
int cml_hcq_nir_queue_init(CMLHCQQueue* queue) { return cml_hcq_vulkan_queue_init(queue); }

/** Destroy a NIR queue via the Vulkan HCQ backend. */
void cml_hcq_nir_queue_destroy(CMLHCQQueue* queue) { cml_hcq_vulkan_queue_destroy(queue); }

/** Submit a kernel via the Vulkan HCQ backend (NIR compiles to SPIR-V). */
int cml_hcq_nir_submit_kernel(CMLHCQQueue* queue, const CMLHCQKernelDesc* desc) {
    LOG_DEBUG("HCQ NIR: submit kernel (forwarding to Vulkan HCQ)");
    return cml_hcq_vulkan_submit_kernel(queue, desc);
}

/** H2D copy via the Vulkan HCQ backend. */
int cml_hcq_nir_memcpy_h2d(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes) {
    return cml_hcq_vulkan_memcpy_h2d(queue, dst, src, bytes);
}

/** D2H copy via the Vulkan HCQ backend. */
int cml_hcq_nir_memcpy_d2h(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes) {
    return cml_hcq_vulkan_memcpy_d2h(queue, dst, src, bytes);
}

/** Create a NIR signal via the Vulkan HCQ backend. */
int cml_hcq_nir_signal_create(CMLHCQSignal* signal) { return cml_hcq_vulkan_signal_create(signal); }

/** Destroy a NIR signal via the Vulkan HCQ backend. */
void cml_hcq_nir_signal_destroy(CMLHCQSignal* signal) { cml_hcq_vulkan_signal_destroy(signal); }

/** Block the host on a NIR signal via the Vulkan HCQ backend. */
int cml_hcq_nir_signal_wait(CMLHCQSignal* signal, uint64_t timeout_ms) {
    return cml_hcq_vulkan_signal_wait(signal, timeout_ms);
}

/** Synchronize a NIR queue via the Vulkan HCQ backend. */
int cml_hcq_nir_synchronize(CMLHCQQueue* queue) { return cml_hcq_vulkan_synchronize(queue); }

#else /* !CML_HAS_NIR */

/** Stub: NIR not compiled in, so the queue cannot init. */
int cml_hcq_nir_queue_init(CMLHCQQueue* q) {
    (void)q;
    return -1;
}
/** Stub: nothing to tear down without NIR. */
void cml_hcq_nir_queue_destroy(CMLHCQQueue* q) { (void)q; }
/** Stub: kernel submission unavailable without NIR. */
int cml_hcq_nir_submit_kernel(CMLHCQQueue* q, const CMLHCQKernelDesc* d) {
    (void)q;
    (void)d;
    return -1;
}
/** Stub: H2D copy unavailable without NIR. */
int cml_hcq_nir_memcpy_h2d(CMLHCQQueue* q, void* d, const void* s, size_t n) {
    (void)q;
    (void)d;
    (void)s;
    (void)n;
    return -1;
}
/** Stub: D2H copy unavailable without NIR. */
int cml_hcq_nir_memcpy_d2h(CMLHCQQueue* q, void* d, const void* s, size_t n) {
    (void)q;
    (void)d;
    (void)s;
    (void)n;
    return -1;
}
/** Stub: no NIR, so no signal can be created. */
int cml_hcq_nir_signal_create(CMLHCQSignal* s) {
    (void)s;
    return -1;
}
/** Stub: nothing to free without NIR. */
void cml_hcq_nir_signal_destroy(CMLHCQSignal* s) { (void)s; }
/** Stub: host wait unavailable without NIR. */
int cml_hcq_nir_signal_wait(CMLHCQSignal* s, uint64_t t) {
    (void)s;
    (void)t;
    return -1;
}
/** Stub: nothing to synchronize without NIR. */
int cml_hcq_nir_synchronize(CMLHCQQueue* q) {
    (void)q;
    return -1;
}

#endif /* CML_HAS_NIR */
