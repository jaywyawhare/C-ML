/* Opt-in CPU replay (TINYJIT_REPLAY=1): a verified graph shape is executed by a
 * plain node walk that skips the scheduler. This must stay numerically exact.
 *
 * The graph hash is structural (op/shape/arity), not data-dependent, so every
 * iteration here reuses one shape but feeds DIFFERENT input values. If replay
 * returned the first run's stored outputs (the classic stale-trace bug) the
 * later iterations would fail; requiring each to match its own inputs proves
 * replay recomputes from the live buffers. A hit count > 0 proves replay
 * actually engaged rather than silently always falling back. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cml.h"
#include "ops/ir/execution.h"
#include "test_harness.h"

#define NELT 8
static TensorConfig cfg = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

/* y = (x + x) * x, realized through the normal execute path (which routes to the
 * global TinyJit). Returns 1 iff every element equals its reference. */
static int run_once(float base) {
    int sh[1] = {NELT};
    float xd[NELT];
    for (int i = 0; i < NELT; i++)
        xd[i] = base + (float)i;
    Tensor* x = tensor_from_data(xd, sh, 1, &cfg);
    if (!x)
        return 0;

    Tensor* s = cml_add(x, x);
    Tensor* y = s ? cml_mul(s, x) : NULL;
    int ok    = y != NULL;
    if (ok) {
        cml_ir_execute(y->ir_context);
        const float* g = (const float*)y->data;
        ok             = g != NULL;
        for (int i = 0; i < NELT && ok; i++) {
            float xv   = base + (float)i;
            float want = (xv + xv) * xv;
            if (fabsf(g[i] - want) > 1e-5f) {
                printf("(base=%g i=%d got=%g want=%g) ", (double)base, i, (double)g[i],
                       (double)want);
                ok = 0;
            }
        }
    }
    cml_reset_ir_context();
    return ok;
}

static int test_replay_is_correct_and_engages(void) {
    int ok = 1;
    /* Distinct input data each time; same shape => same hash => replay path. */
    for (int it = 0; it < 8 && ok; it++)
        ok = run_once(1.0f + 3.0f * (float)it);
    size_t hits = cml_ir_tinyjit_replay_hits();
    if (hits == 0) {
        printf("(replay never engaged) ");
        ok = 0;
    }
    return ok;
}

int main(void) {
    setenv("TINYJIT_REPLAY", "1", 1);   /* opt in before the engine reads it */
    setenv("FUSION_SCHEDULER", "0", 1); /* replay verifies against the unfused walk */
    cml_init();
    printf("=== TinyJit CPU replay (opt-in) ===\n");
    TEST(replay_is_correct_and_engages);
    cml_cleanup();
    return TEST_SUMMARY();
}
