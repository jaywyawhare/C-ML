/* Trace-and-replay JIT: the fallback has to stay honest.
 *
 * `cml_tinyjit_execute` records a trace on a miss and replays it on a hit. On the
 * CPU path no trace entries are ever recorded, so it always falls through to real
 * execution -- and that is deliberate, not an oversight waiting to be tidied up.
 * The cheap-looking "fix" (cache the trace anyway and replay it) returns the
 * recording run's values on every later call, which is silently wrong.
 *
 * Attempting the CPU recording is what showed why it cannot work as a trace of
 * IR nodes. In short: after a
 * graph executes its nodes are marked executed and a re-run is a no-op, so there is
 * nothing to replay; and getting a fresh run means `cml_reset_ir_context()`, which
 * frees the very nodes a trace would point at. A CPU trace therefore has to record
 * something that outlives the graph -- ops plus buffers, with its own executor --
 * which is the second replay-shaped engine the design notes call out.
 *
 * So these tests pin the fallback: results are correct, and an empty or truncated
 * trace is never cached or replayed.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cml.h"
#include "ops/ir/ir.h"
#include "ops/ir/tiny_jit.h"
#include "ops/ir/trace.h"
#include "ops/uops.h"
#include "test_harness.h"
#include "test_require.h"

static TensorConfig cfg = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

#define NELT 8

/* y = (x + x) * x */
static Tensor* build_graph(Tensor** x_out, int n, float base) {
    int sh[1] = {n};
    Tensor* x = tensor_zeros(sh, 1, &cfg);
    if (!x)
        return NULL;
    float* d = (float*)tensor_data_ptr(x);
    for (int i = 0; i < n; i++)
        d[i] = base + (float)i;

    Tensor* s = uop_add(x, x);
    Tensor* y = s ? uop_mul(s, x) : NULL;
    *x_out    = x;
    return y;
}

static int values_ok(Tensor* y, int n, float base) {
    if (!y || !y->data)
        return 0;
    const float* got = (const float*)y->data;
    for (int i = 0; i < n; i++) {
        float x    = base + (float)i;
        float want = (x + x) * x;
        if (fabsf(got[i] - want) > 1e-5f) {
            printf("(i=%d got=%g want=%g) ", i, (double)got[i], (double)want);
            return 0;
        }
    }
    return 1;
}

/* Going through the JIT must produce the same numbers as plain execution. */
static int test_execute_is_correct(void) {
    cml_reset_ir_context();
    Tensor* x = NULL;
    Tensor* y = build_graph(&x, NELT, 1.0f);
    if (!y) {
        cml_reset_ir_context();
        return 0;
    }

    CMLTinyJit* jit = cml_tinyjit_create();
    REQUIRE(jit);

    int ok = cml_tinyjit_execute(jit, y->ir_context) == 0 && values_ok(y, NELT, 1.0f);

    cml_tinyjit_free(jit);
    cml_reset_ir_context();
    return ok;
}

/* No CPU trace entries are recorded, so every call is a miss and nothing is ever
 * replayed. A hit here would mean a trace got cached that cannot faithfully
 * reproduce the graph -- the stale-output bug this fallback exists to avoid. */
static int test_never_replays_an_empty_trace(void) {
    cml_reset_ir_context();
    Tensor* x = NULL;
    Tensor* y = build_graph(&x, NELT, 2.0f);
    if (!y) {
        cml_reset_ir_context();
        return 0;
    }

    CMLTinyJit* jit = cml_tinyjit_create();
    REQUIRE(jit);

    int ok = cml_tinyjit_execute(jit, y->ir_context) == 0;
    ok     = ok && cml_tinyjit_execute(jit, y->ir_context) == 0;

    size_t hits = 0, misses = 0, inval = 0;
    cml_tinyjit_stats(jit, &hits, &misses, &inval);
    if (ok && hits != 0) {
        printf("(replayed a trace it should not have: hits=%zu) ", hits);
        ok = 0;
    }
    if (ok && misses == 0) {
        printf("(no misses recorded at all) ");
        ok = 0;
    }

    cml_tinyjit_free(jit);
    cml_reset_ir_context();
    return ok;
}

/* Results must stay correct across repeated executions, whichever path is taken. */
static int test_repeated_execution_stays_correct(void) {
    cml_reset_ir_context();
    Tensor* x = NULL;
    Tensor* y = build_graph(&x, NELT, 3.0f);
    if (!y) {
        cml_reset_ir_context();
        return 0;
    }

    CMLTinyJit* jit = cml_tinyjit_create();
    REQUIRE(jit);

    int ok = 1;
    for (int r = 0; r < 3 && ok; r++) {
        ok = cml_tinyjit_execute(jit, y->ir_context) == 0;
        ok = ok && values_ok(y, NELT, 3.0f);
    }

    cml_tinyjit_free(jit);
    cml_reset_ir_context();
    return ok;
}

/* A trace that hit the entry cap describes only part of the graph, so it must not
 * be marked complete -- replaying it would skip the tail and leave stale outputs.
 * Driven directly, since the CPU path records nothing. */
static int test_truncated_trace_is_not_complete(void) {
    CMLTrace* t = cml_trace_create();
    REQUIRE(t);

    cml_trace_begin(t, 0x1234);
    size_t grid[3] = {1, 1, 1}, block[3] = {1, 1, 1};
    int args[1] = {0};

    /* One past the cap: the final record must be refused, not silently dropped. */
    int refused = 0;
    for (int i = 0; i < CML_TRACE_MAX_ENTRIES + 1; i++) {
        int rc =
            cml_trace_record_kernel(t, (uint64_t)i, (void*)(uintptr_t)0x1, grid, block, args, 1);
        if (rc != 0)
            refused++;
    }
    cml_trace_end(t);

    int ok =
        refused == 1 && t->truncated && !t->is_complete && t->num_entries == CML_TRACE_MAX_ENTRIES;
    if (!ok)
        printf("(refused=%d truncated=%d complete=%d entries=%d) ", refused, (int)t->truncated,
               (int)t->is_complete, t->num_entries);

    cml_trace_free(t);
    return ok;
}

/* A trace within the cap is complete and replayable. */
static int test_untruncated_trace_is_complete(void) {
    CMLTrace* t = cml_trace_create();
    REQUIRE(t);
    cml_trace_begin(t, 0x5678);
    cml_trace_end(t);
    int ok = !t->truncated && t->is_complete && t->num_entries == 0;
    cml_trace_free(t);
    return ok;
}

/* A graph shape that records nothing is remembered as such, so later executes
 * skip the record attempt instead of re-allocating a trace and re-hashing to
 * rediscover it every time. The saving is real -- that probe cost ~37% of a small
 * step -- but the shortcut must not change results, which is what this checks:
 * values stay correct across repeats, and it still never reports a replay. */
static int test_empty_trace_is_negatively_cached(void) {
    cml_reset_ir_context();
    Tensor* x = NULL;
    Tensor* y = build_graph(&x, NELT, 4.0f);
    if (!y) {
        cml_reset_ir_context();
        return 0;
    }

    CMLTinyJit* jit = cml_tinyjit_create();
    REQUIRE(jit);

    int ok = 1;
    for (int r = 0; r < 8 && ok; r++) {
        ok = cml_tinyjit_execute(jit, y->ir_context) == 0;
        ok = ok && values_ok(y, NELT, 4.0f);
    }

    size_t hits = 0, misses = 0, inval = 0;
    cml_tinyjit_stats(jit, &hits, &misses, &inval);

    /* Still no replay: the negative entry is a shortcut past recording, never a
     * licence to hand back cached values. */
    if (ok && hits != 0) {
        printf("(reported a replay: hits=%zu) ", hits);
        ok = 0;
    }
    /* The point of the negative cache is that the record attempt is paid a BOUNDED
     * number of times, not once per execute. Two is expected here, not one: the
     * first execute runs the fusion pass, which rewrites nodes and so changes the
     * graph hash, giving one shape before fusion and one after. It is stable from
     * then on, so eight executes must still cost only those two. */
    if (ok && misses > 2) {
        printf("(record attempted per execute: misses=%zu of 8 calls, inval=%zu) ", misses, inval);
        ok = 0;
    }

    cml_tinyjit_free(jit);
    cml_reset_ir_context();
    return ok;
}

static int test_bad_args(void) {
    CMLTinyJit* jit = cml_tinyjit_create();
    REQUIRE(jit);
    int ok = cml_tinyjit_execute(jit, NULL) == -1 && cml_tinyjit_execute(NULL, NULL) == -1;
    cml_tinyjit_stats(NULL, NULL, NULL, NULL); /* must not crash */
    cml_tinyjit_free(jit);
    cml_tinyjit_free(NULL);
    return ok;
}

int main(void) {
    printf("TinyJit Trace/Replay Tests\n\n");

    printf("Execution:\n");
    TEST(bad_args);
    TEST(execute_is_correct);
    TEST(repeated_execution_stays_correct);

    printf("\nThe fallback stays honest:\n");
    TEST(never_replays_an_empty_trace);
    TEST(empty_trace_is_negatively_cached);
    TEST(truncated_trace_is_not_complete);
    TEST(untruncated_trace_is_complete);

    return TEST_SUMMARY();
}
