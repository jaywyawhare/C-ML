/* Validates the ROCm backend + HCQ adapter against the mock HIP driver:
 * buffer mapping (H2D/launch/D2H ordering and payload), event-based signal
 * lifecycle, and leak-free teardown. Run without AMD hardware. */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "ops/ir/hcq.h"
#include "ops/ir/gpu/rocm_backend.h"
#include "ops/ir/gpu/hip_mock.h"
#include "test_harness.h"

/* Backend accessor for the queue's private handle (hcq_rocm.c). */
extern CMLROCmBackend* cml_hcq_rocm_queue_backend(CMLHCQQueue* queue);
extern void cml_hcq_rocm_signal_destroy(CMLHCQSignal* signal);

static int test_roundtrip_and_order(void) {
    setenv("CML_HIP_MOCK", "1", 1);
    cml_hip_mock_reset();

    CMLHCQQueue* q = cml_hcq_rocm_queue_create();
    if (!q) return 0;

    CMLROCmBackend* backend = cml_hcq_rocm_queue_backend(q);
    if (!backend) return 0;

    float host[64];
    for (int i = 0; i < 64; i++) host[i] = (float)i;

    /* device buffer via the backend's malloc path */
    void* dev = NULL;
    if (backend->hipMalloc(&dev, sizeof(host)) != 0) return 0;

    if (cml_hcq_rocm_memcpy_h2d(q, dev, host, sizeof(host)) != 0) return 0;

    /* launch a fake kernel through the HCQ contract */
    CMLROCmKernel k;
    memset(&k, 0, sizeof(k));
    static const char* kn = "mock_vec_kernel";
    if (backend->hipModuleGetFunction(&k.function, NULL, kn) != 0) return 0;
    /* HIP kernelParams is a NULL-terminated array of pointers to arguments. */
    void* args[3] = {dev, host, NULL};
    CMLHCQKernelDesc desc = {
        .compiled_kernel = &k,
        .grid = {4, 1, 1}, .block = {16, 1, 1},
        .args = args, .num_args = 2,
    };
    if (cml_hcq_rocm_submit_kernel(q, &desc) != 0) return 0;

    float back[64] = {0};
    if (cml_hcq_rocm_memcpy_d2h(q, back, dev, sizeof(host)) != 0) return 0;
    for (int i = 0; i < 64; i++)
        if (back[i] != host[i]) return 0; /* payload survived the roundtrip */

    /* signal lifecycle: record then queue wait */
    CMLHCQSignal* sig = cml_hcq_rocm_signal_create();
    if (!sig) return 0;
    if (cml_hcq_rocm_signal_record(q, sig) != 0) return 0;
    if (cml_hcq_rocm_queue_wait(q, sig) != 0) return 0;
    cml_hcq_rocm_signal_destroy(sig);

    if (cml_hcq_rocm_queue_synchronize(q) != 0) return 0;

    backend->hipFree(dev);
    cml_hcq_rocm_queue_destroy(q);

    /* journal order: H2D ... LAUNCH ... D2H, single launch, no leaks */
    int n = cml_hip_mock_journal_len();
    int i_h2d = -1, i_launch = -1, i_d2h = -1;
    for (int i = 0; i < n; i++) {
        const CMLHIPMockEntry* e = cml_hip_mock_journal_at(i);
        if (!e) break;
        if (e->kind == CML_HIP_MOCK_OP_MEMCPY_H2D && i_h2d < 0) i_h2d = i;
        if (e->kind == CML_HIP_MOCK_OP_LAUNCH && i_launch < 0) i_launch = i;
        if (e->kind == CML_HIP_MOCK_OP_MEMCPY_D2H && i_d2h < 0) i_d2h = i;
    }
    if (!(i_h2d >= 0 && i_launch > i_h2d && i_d2h > i_launch)) return 0;
    if (cml_hip_mock_launches() != 1) return 0;
    if (cml_hip_mock_outstanding_allocs() != 0) return 0; /* all device buffers freed */
    return 1;
}

static int test_event_lifecycle(void) {
    cml_hip_mock_reset();
    CMLROCmBackend b;
    memset(&b, 0, sizeof(b));
    if (cml_rocm_backend_init_mock(&b) != 0) return 0;
    if (strcmp(b.device_name, "Mock-RDNA3") != 0) return 0;
    void* ev = NULL;
    if (b.hipEventCreate(&ev) != 0) return 0;
    if (b.hipEventRecord(ev, b.stream) != 0) return 0;
    if (b.hipEventSynchronize(ev) != 0) return 0;
    b.hipEventDestroy(ev);
    return 1;
}

static int test_double_init_is_noop(void) {
    CMLROCmBackend b;
    memset(&b, 0, sizeof(b));
    if (cml_rocm_backend_init_mock(&b) != 0) return 0;
    int len_before = cml_hip_mock_journal_len();
    if (cml_rocm_backend_init_mock(&b) != 0) return 0; /* already initialized */
    return cml_hip_mock_journal_len() == len_before;   /* no re-init journaling */
}

#ifndef RUN_TEST
#define RUN_TEST(t) TEST(t)
#endif

int main(void) {
    printf("=== hip mock driver ===\n");
    TEST(roundtrip_and_order);
    TEST(event_lifecycle);
    TEST(double_init_is_noop);
    return TEST_SUMMARY();
}
