/**
 * @file test_hcq.c
 * @brief Tests for Hardware Command Queues (HCQ)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ops/ir/hcq.h"

static void test_queue_create_destroy(void) {
    printf("  test_queue_create_destroy...");

    CMLHCQQueue* queue = cml_hcq_queue_create(CML_HCQ_CPU);
    assert(queue != NULL);
    assert(queue->backend == CML_HCQ_CPU);
    assert(queue->active == true);
    assert(queue->num_wait_signals == 0);

    cml_hcq_queue_destroy(queue);
    printf(" PASS\n");
}

static void test_signal_create_destroy(void) {
    printf("  test_signal_create_destroy...");

    CMLHCQSignal* signal = cml_hcq_signal_create(CML_HCQ_CPU);
    assert(signal != NULL);
    assert(signal->backend == CML_HCQ_CPU);
    assert(signal->timeline_value == 0);
    assert(signal->signaled == false);

    cml_hcq_signal_destroy(signal);
    printf(" PASS\n");
}

static void test_signal_record_and_wait(void) {
    printf("  test_signal_record_and_wait...");

    CMLHCQQueue* queue = cml_hcq_queue_create(CML_HCQ_CPU);
    assert(queue != NULL);

    CMLHCQSignal* signal = cml_hcq_signal_create(CML_HCQ_CPU);
    assert(signal != NULL);

    /* Record signal on queue */
    int ret = cml_hcq_signal_record(queue, signal);
    assert(ret == 0);

    /* For CPU backend, signal should be immediately signaled */
    assert(signal->signaled == true);

    /* Wait for signal on CPU (should return immediately) */
    ret = cml_hcq_signal_wait_cpu(signal, 1000);
    assert(ret == 0);

    cml_hcq_signal_destroy(signal);
    cml_hcq_queue_destroy(queue);
    printf(" PASS\n");
}

static void test_queue_synchronize(void) {
    printf("  test_queue_synchronize...");

    CMLHCQQueue* queue = cml_hcq_queue_create(CML_HCQ_CPU);
    assert(queue != NULL);

    /* Synchronize should be a no-op for CPU and return 0 */
    int ret = cml_hcq_queue_synchronize(queue);
    assert(ret == 0);

    cml_hcq_queue_destroy(queue);
    printf(" PASS\n");
}

static void test_queue_wait_signal(void) {
    printf("  test_queue_wait_signal...");

    CMLHCQQueue* queue = cml_hcq_queue_create(CML_HCQ_CPU);
    assert(queue != NULL);

    CMLHCQSignal* signal = cml_hcq_signal_create(CML_HCQ_CPU);
    assert(signal != NULL);

    /* Queue wait on signal */
    int ret = cml_hcq_queue_wait(queue, signal);
    assert(ret == 0);

    cml_hcq_signal_destroy(signal);
    cml_hcq_queue_destroy(queue);
    printf(" PASS\n");
}

static void test_pipeline_create_destroy(void) {
    printf("  test_pipeline_create_destroy...");

    CMLHCQPipeline* pipeline = cml_hcq_pipeline_create();
    assert(pipeline != NULL);
    assert(pipeline->num_stages == 0);

    cml_hcq_pipeline_destroy(pipeline);
    printf(" PASS\n");
}

static void test_pipeline_add_stages(void) {
    printf("  test_pipeline_add_stages...");

    CMLHCQPipeline* pipeline = cml_hcq_pipeline_create();
    assert(pipeline != NULL);

    CMLHCQQueue* q1 = cml_hcq_queue_create(CML_HCQ_CPU);
    CMLHCQQueue* q2 = cml_hcq_queue_create(CML_HCQ_CPU);
    assert(q1 != NULL);
    assert(q2 != NULL);

    int ret;
    ret = cml_hcq_pipeline_add_stage(pipeline, q1);
    assert(ret == 0);
    assert(pipeline->num_stages == 1);

    ret = cml_hcq_pipeline_add_stage(pipeline, q2);
    assert(ret == 0);
    assert(pipeline->num_stages == 2);

    printf(" PASS\n");

    /* Test execute and synchronize */
    printf("  test_pipeline_execute_sync...");
    ret = cml_hcq_pipeline_execute(pipeline);
    assert(ret == 0);

    ret = cml_hcq_pipeline_synchronize(pipeline);
    assert(ret == 0);

    printf(" PASS\n");

    /* Cleanup: pipeline destroy should handle its internal state,
       but we still need to free the queues we created */
    cml_hcq_pipeline_destroy(pipeline);
    cml_hcq_queue_destroy(q1);
    cml_hcq_queue_destroy(q2);
}

int main(void) {
    printf("=== Hardware Command Queue (HCQ) Tests ===\n");

    test_queue_create_destroy();
    test_signal_create_destroy();
    test_signal_record_and_wait();
    test_queue_synchronize();
    test_queue_wait_signal();
    test_pipeline_create_destroy();
    test_pipeline_add_stages();

    printf("All HCQ tests passed.\n");
    return 0;
}
