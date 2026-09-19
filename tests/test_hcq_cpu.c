/*
 * AUDIT #10: the CPU is now a first-class HCQ backend (g_hcq_cpu_ops) rather
 * than open-coded special-cases scattered through hcq.c. This exercises the
 * full ops surface via the CPU backend — queue, memcpy, a real submitted
 * kernel, signals, synchronize, and a pipeline — proving the unified
 * abstraction runs end-to-end with no GPU hardware.
 */
#include <stdio.h>
#include <string.h>

#include "ops/ir/hcq.h"
#include "test_harness.h"
#include "ops/ir/hcq_backend.h"

static int check(const char* name, int ok) {
    tests_run++;
    if (ok) {
        tests_passed++;
        printf("  PASS: %s\n", name);
    } else {
        printf("  FAIL: %s\n", name);
    }
    return ok;
}

/* A real CPU kernel: out[i] = a[i] + b[i]. */
static void vadd_kernel(void** args, int num_args) {
    if (num_args < 4)
        return;
    const float* a = (const float*)args[0];
    const float* b = (const float*)args[1];
    float* out     = (float*)args[2];
    int n          = *(int*)args[3];
    for (int i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}

int main(void) {
    printf("=== AUDIT #10: CPU as first-class HCQ backend ===\n");

    /* CPU ops are registered and named. */
    const CMLHCQBackendOps* ops = cml_hcq_backend_ops(CML_HCQ_CPU);
    check("cpu_ops_registered", ops != NULL && ops->name && strcmp(ops->name, "CPU") == 0);

    CMLHCQQueue* q = cml_hcq_queue_create(CML_HCQ_CPU);
    check("queue_create", q != NULL && q->backend == CML_HCQ_CPU && q->active);

    /* memcpy round-trip through the queue. */
    float src[4] = {1, 2, 3, 4};
    float dev[4] = {0}, host[4] = {0};
    int ok = cml_hcq_memcpy_h2d(q, dev, src, sizeof(src)) == 0 &&
             cml_hcq_memcpy_d2h(q, host, dev, sizeof(dev)) == 0 &&
             memcmp(host, src, sizeof(src)) == 0;
    check("memcpy_h2d_d2h", ok);

    /* submit a real kernel and check the result. */
    float a[4] = {1, 2, 3, 4}, b[4] = {10, 20, 30, 40}, out[4] = {0};
    int n                 = 4;
    void* args[4]         = {a, b, out, &n};
    CMLHCQKernelDesc desc = {
        .compiled_kernel = (void*)vadd_kernel,
        .grid            = {1, 1, 1},
        .block           = {4, 1, 1},
        .args            = args,
        .num_args        = 4,
    };
    int rc          = cml_hcq_submit_kernel(q, &desc);
    float expect[4] = {11, 22, 33, 44};
    check("submit_kernel_runs", rc == 0 && memcmp(out, expect, sizeof(out)) == 0);

    /* signal record / wait / synchronize. */
    CMLHCQSignal* sig = cml_hcq_signal_create(CML_HCQ_CPU);
    int sok           = sig != NULL && cml_hcq_signal_record(q, sig) == 0 && sig->signaled &&
              cml_hcq_queue_wait(q, sig) == 0 && cml_hcq_signal_wait_cpu(sig, 0) == 0 &&
              cml_hcq_queue_synchronize(q) == 0;
    check("signal_and_sync", sok);
    cml_hcq_signal_destroy(sig);

    /* pipeline of two CPU stages. */
    CMLHCQPipeline* pipe = cml_hcq_pipeline_create();
    CMLHCQQueue* s1      = cml_hcq_queue_create(CML_HCQ_CPU);
    CMLHCQQueue* s2      = cml_hcq_queue_create(CML_HCQ_CPU);
    int pok              = pipe && s1 && s2 && cml_hcq_pipeline_add_stage(pipe, s1) == 0 &&
              cml_hcq_pipeline_add_stage(pipe, s2) == 0 && cml_hcq_pipeline_execute(pipe) == 0 &&
              cml_hcq_pipeline_synchronize(pipe) == 0;
    check("pipeline_execute", pok);
    cml_hcq_pipeline_destroy(pipe);
    cml_hcq_queue_destroy(s1);
    cml_hcq_queue_destroy(s2);
    cml_hcq_queue_destroy(q);

    return TEST_SUMMARY();
}
