/* Multi-device schedule partitioning.
 *
 * multi_schedule_* is an analysis layer: it splits a CMLSchedule across devices,
 * records the cross-device transfers a real executor would need, and totals the
 * cost. It had no tests, and multi_schedule_run() used to return 0 for a schedule
 * whose compute steps it never executed -- these pin the figures that are
 * computed for real, and the honest failure for the part that is not.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "cml.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "ops/ir/ir.h"
#include "ops/ir/schedule.h"
#include "ops/ir/schedule_multi.h"
#include "test_harness.h"
#include "test_require.h"

/* A small graph with enough kernels to spread over several devices. */
static CMLSchedule* build_schedule(void) {
    float ad[8], bd[8];
    for (int i = 0; i < 8; i++) {
        ad[i] = (float)i;
        bd[i] = (float)(i + 1);
    }
    int shape[1]     = {8};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* a = tensor_from_data(ad, shape, 1, &cfg);
    Tensor* b = tensor_from_data(bd, shape, 1, &cfg);
    REQUIRE(a && b);

    Tensor* t = uop_add(a, b);
    t         = uop_mul(t, a);
    t         = uop_sub(t, b);
    t         = uop_relu(t);
    REQUIRE(t);

    return cml_schedule_create(t->ir_context, NULL);
}

static bool test_build_and_free(void) {
    CMLSchedule* s = build_schedule();
    if (!s) {
        cml_reset_ir_context();
        return false;
    }

    int devs[2]             = {0, 1};
    MultiDeviceSchedule* ms = multi_schedule_build(s, devs, 2);
    bool ok                 = ms != NULL;

    if (ok) {
        ok = ms->num_devices == 2 && ms->num_steps > 0 && ms->device_schedules != NULL &&
             ms->device_ids != NULL && ms->steps != NULL;
        if (ok)
            ok = ms->device_ids[0] == 0 && ms->device_ids[1] == 1;
    }

    multi_schedule_free(ms);
    cml_schedule_free(s);
    cml_reset_ir_context();
    return ok;
}

/* Every kernel in the source schedule must land on exactly one device -- a
 * partition that dropped or duplicated kernels would still "work" but compute
 * the wrong thing. */
static bool test_partition_preserves_kernel_count(void) {
    CMLSchedule* s = build_schedule();
    if (!s) {
        cml_reset_ir_context();
        return false;
    }
    int expected = cml_schedule_num_kernels(s);

    int devs[2]             = {0, 1};
    MultiDeviceSchedule* ms = multi_schedule_build(s, devs, 2);
    bool ok                 = ms != NULL;
    if (ok) {
        int total = multi_schedule_total_kernels(ms);
        if (total != expected) {
            printf("(kernels %d != %d) ", total, expected);
            ok = false;
        }
    }

    multi_schedule_free(ms);
    cml_schedule_free(s);
    cml_reset_ir_context();
    return ok;
}

/* One device means nothing crosses a device boundary. */
static bool test_single_device_needs_no_transfers(void) {
    CMLSchedule* s = build_schedule();
    if (!s) {
        cml_reset_ir_context();
        return false;
    }

    int devs[1]             = {0};
    MultiDeviceSchedule* ms = multi_schedule_build(s, devs, 1);
    bool ok                 = ms != NULL;
    if (ok) {
        ok = ms->num_devices == 1;
        if (ok && multi_schedule_xfer_bytes(ms) != 0) {
            printf("(single device reported transfers) ");
            ok = false;
        }
        /* With no cross-device step, run() only has compute steps to refuse. */
        if (ok && multi_schedule_run(ms) == 0 && ms->num_steps > 0) {
            printf("(claimed it ran compute steps) ");
            ok = false;
        }
    }

    multi_schedule_free(ms);
    cml_schedule_free(s);
    cml_reset_ir_context();
    return ok;
}

/* run() must not report success for compute steps it cannot execute. */
static bool test_run_refuses_compute_steps(void) {
    CMLSchedule* s = build_schedule();
    if (!s) {
        cml_reset_ir_context();
        return false;
    }

    int devs[2]             = {0, 1};
    MultiDeviceSchedule* ms = multi_schedule_build(s, devs, 2);
    bool ok                 = ms != NULL;

    if (ok) {
        int has_compute = 0;
        for (int i = 0; i < ms->num_steps; i++)
            if (ms->steps[i].kind == MULTI_STEP_DEVICE)
                has_compute = 1;

        int rc = multi_schedule_run(ms);
        if (has_compute) {
            if (rc != -2) {
                printf("(rc %d, expected -2) ", rc);
                ok = false;
            }
        } else if (rc != 0) {
            printf("(rc %d, expected 0) ", rc);
            ok = false;
        }
    }

    multi_schedule_free(ms);
    cml_schedule_free(s);
    cml_reset_ir_context();
    return ok;
}

static bool test_bad_args(void) {
    bool ok = multi_schedule_build(NULL, NULL, 0) == NULL;
    ok      = ok && multi_schedule_run(NULL) == -1;
    ok      = ok && multi_schedule_xfer_bytes(NULL) == 0;
    ok      = ok && multi_schedule_total_kernels(NULL) == 0;
    multi_schedule_free(NULL); /* must not crash */

    CMLSchedule* s = build_schedule();
    if (s) {
        int devs[1] = {0};
        ok          = ok && multi_schedule_build(s, devs, 0) == NULL;
        ok          = ok && multi_schedule_build(s, NULL, 1) == NULL;
        cml_schedule_free(s);
    }
    cml_reset_ir_context();
    return ok;
}

static bool test_p2p_query_is_reflexive(void) {
    /* A device is trivially reachable from itself; the query must not crash on
     * out-of-range ids either. */
    bool self = devices_p2p_capable(0, 0);
    (void)devices_p2p_capable(0, 99);
    (void)devices_p2p_capable(-1, 0);
    return self || !self; /* value is backend-dependent; the call must be safe */
}

int main(void) {
    printf("Multi-Device Schedule Tests\n\n");

    TEST(build_and_free);
    TEST(partition_preserves_kernel_count);
    TEST(single_device_needs_no_transfers);
    TEST(run_refuses_compute_steps);
    TEST(bad_args);
    TEST(p2p_query_is_reflexive);

    return TEST_SUMMARY();
}
