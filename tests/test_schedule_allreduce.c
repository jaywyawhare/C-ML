/* Tests for the multi-device all-reduce PLANNER (single-process).
 *
 * Covers the two bugs the audit flagged:
 *   1. schedule_allreduce_run reduced the buffer against itself (x += x),
 *      doubling it regardless of device count — data corruption.
 *   2. schedule_allreduce_inject was a stub that returned success without
 *      touching the schedule.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "cml.h"
#include "ops/ir/schedule.h"
#include "ops/ir/schedule_allreduce.h"
#include "test_harness.h"

#define EPS 1e-4f

static Tensor* mk1d(const float* d, int n) {
    int shape[1]     = {n};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    return tensor_from_data(d, shape, 1, &cfg);
}

/* run(SUM) over N identical replicas → each element scaled by N (not doubled). */
static int test_run_sum_scales_by_device_count(void) {
    float d[8]  = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor* t   = mk1d(d, 8);
    int devs[4] = {0, 1, 2, 3};

    ScheduleAllReduce* ar = schedule_allreduce_build(t, AR_OP_SUM, AR_ALGO_RING, devs, 4);
    if (!ar) {
        tensor_free(t);
        return 0;
    }

    int ok = (schedule_allreduce_run(ar) == 0);
    tensor_ensure_executed(t);
    const float* out = (const float*)tensor_data_ptr(t);
    for (int i = 0; i < 8 && ok; i++)
        if (fabsf(out[i] - d[i] * 4.0f) > EPS) {
            printf("(run[%d]=%f want %f) ", i, out[i], d[i] * 4.0f);
            ok = 0;
        }

    schedule_allreduce_free(ar);
    tensor_free(t);
    return ok;
}

/* inject appends one SCHED_COPY item per all-reduce step; free stays consistent. */
static int test_inject_appends_copy_items(void) {
    CMLGraph_t g   = cml_ir_new(IR_TARGET_C);
    CMLSchedule* s = cml_schedule_create(g, NULL); /* empty → num_items == 0 */
    if (!s) {
        cml_ir_free(g);
        return 0;
    }
    int before = s->num_items;

    float d[16];
    for (int i = 0; i < 16; i++)
        d[i] = (float)i;
    Tensor* t             = mk1d(d, 16);
    int devs[4]           = {0, 1, 2, 3};
    ScheduleAllReduce* ar = schedule_allreduce_build(t, AR_OP_SUM, AR_ALGO_RING, devs, 4);
    if (!ar) {
        cml_schedule_free(s);
        cml_ir_free(g);
        tensor_free(t);
        return 0;
    }

    int rc = schedule_allreduce_inject(s, ar);
    int ok = (rc == 0) && (s->num_items == before + ar->num_steps) && (ar->num_steps > 0);
    for (int i = before; i < s->num_items && ok; i++) {
        const CMLScheduleItem* it = cml_schedule_get_item(s, i);
        if (!it || it->type != SCHED_COPY) {
            printf("(item %d not COPY) ", i);
            ok = 0;
        }
    }

    schedule_allreduce_free(ar);
    cml_schedule_free(s); /* must not crash: dependency arrays grew in lockstep */
    cml_ir_free(g);
    tensor_free(t);
    return ok;
}

#undef TEST
#define TEST(fn)                                                                                   \
    do {                                                                                           \
        tests_run++;                                                                               \
        printf("  %-50s ", #fn);                                                                   \
        fflush(stdout);                                                                            \
        if (fn()) {                                                                                \
            tests_passed++;                                                                        \
            printf("[PASS]\n");                                                                    \
        } else                                                                                     \
            printf("[FAIL]\n");                                                                    \
    } while (0)

int main(void) {
    printf("Schedule all-reduce planner tests:\n");
    TEST(test_run_sum_scales_by_device_count);
    TEST(test_inject_appends_copy_items);
    return TEST_SUMMARY();
}
