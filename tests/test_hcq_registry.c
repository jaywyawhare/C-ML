/* HCQ backend registry: every backend must resolve through
 * cml_hcq_backend_ops() and fail gracefully where no hardware exists.
 *
 * AUDIT #10 follow-up: ROCm and WebGPU previously had no HCQ adapter at all,
 * so cml_hcq_backend_ops() returned NULL for them and their memory/dispatch
 * code sat outside the uniform queue/signal machinery. This pins the full
 * registry: each entry either provides working ops (exercised when hardware
 * is present) or fails cleanly with -1/NULL and no crashes.
 */
#include <stdio.h>
#include <string.h>

#include "ops/ir/hcq_backend.h"
#include "alloc/cml_allocator.h"
#include "test_harness.h"

static int check(const char* name, int ok) {
    tests_run++;
    if (ok) { tests_passed++; printf("  PASS: %s\n", name); }
    else    { printf("  FAIL: %s\n", name); }
    return ok;
}

/* Every enum value resolves to ops or NULL — never a crash — and any ops
 * table it does return has a name and the mandatory entry points. */
static int test_registry_complete(void) {
    for (int b = 0; b < CML_HCQ_BACKEND_COUNT; b++) {
        const CMLHCQBackendOps* ops = cml_hcq_backend_ops((CMLHCQBackendType)b);
        if (!ops) continue;
        if (!ops->name || !ops->queue_create || !ops->queue_destroy ||
            !ops->submit_kernel || !ops->memcpy_h2d || !ops->memcpy_d2h ||
            !ops->queue_synchronize || !ops->signal_create ||
            !ops->signal_destroy || !ops->signal_record || !ops->queue_wait ||
            !ops->signal_wait_cpu)
            return 0;
    }
    return cml_hcq_backend_ops((CMLHCQBackendType)-1) == NULL &&
           cml_hcq_backend_ops(CML_HCQ_BACKEND_COUNT) == NULL;
}

/* Exercise one backend end-to-end when its hardware/library is present;
 * otherwise verify the graceful failure contract. */
static void exercise_backend(CMLHCQBackendType type, int* ok) {
    CMLHCQQueue* q = cml_hcq_queue_create(type);
    if (!q) {
        /* No hardware: submits/memcpys on a NULL queue must fail cleanly. */
        CMLHCQKernelDesc desc;
        memset(&desc, 0, sizeof(desc));
        *ok &= cml_hcq_submit_kernel(NULL, &desc) == -1;
        *ok &= cml_hcq_memcpy_h2d(NULL, NULL, NULL, 0) == -1;
        *ok &= cml_hcq_memcpy_d2h(NULL, NULL, NULL, 0) == -1;
        return;
    }

    *ok &= q->backend == type;

    /* Host<->device round-trip through the uniform API. */
    unsigned char host[64];
    unsigned char back[64];
    for (int i = 0; i < 64; i++) host[i] = (unsigned char)i;
    memset(back, 0, sizeof(back));
    void* dev = cml_malloc(64);
    *ok &= dev != NULL;
    if (dev) {
        *ok &= cml_hcq_memcpy_h2d(q, dev, host, sizeof(host)) == 0;
        *ok &= cml_hcq_memcpy_d2h(q, back, dev, sizeof(back)) == 0;
        *ok &= memcmp(host, back, sizeof(host)) == 0;
        cml_free(dev);
    }

    /* Signal lifecycle through the uniform API. */
    CMLHCQSignal* s = cml_hcq_signal_create(type);
    *ok &= s != NULL;
    if (s) {
        *ok &= cml_hcq_signal_record(q, s) == 0;
        *ok &= cml_hcq_signal_wait_cpu(s, 1000) == 0;
        cml_hcq_signal_destroy(s);
    }

    *ok &= cml_hcq_queue_synchronize(q) == 0;
    cml_hcq_queue_destroy(q);
}

int main(void) {
    printf("=== HCQ backend registry ===\n");
    int ok = 1;

    check("registry_complete", test_registry_complete());

    /* CPU is always present and synchronous. */
    exercise_backend(CML_HCQ_CPU, &ok);
    check("cpu_roundtrip", ok);

    /* The dlopen'd GPU adapters: exercised when the library/hardware exists,
     * held to the graceful-failure contract when it does not. */
    int rocm_ok = 1, webgpu_ok = 1;
    exercise_backend(CML_HCQ_ROCM, &rocm_ok);
    exercise_backend(CML_HCQ_WEBGPU, &webgpu_ok);
    check("rocm_contract", rocm_ok);
    check("webgpu_contract", webgpu_ok);

    return TEST_SUMMARY();
}
