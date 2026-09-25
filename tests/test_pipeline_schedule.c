/* Schedules produced by cml_pipeline_build_schedule, and the single-process
 * pipeline running under each of them.
 *
 * The two schedules must be interchangeable numerically -- weight gradients
 * accumulate over micro-batches, so the order they arrive in cannot change the
 * sum -- and differ only in how early a backward retires. These tests pin both
 * halves of that claim: the orders are valid and distinct, and training through
 * them lands on the same gradients.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "nn/layers/linear.h"
#include "distributed/distributed.h"
#include "distributed/pipeline_parallel.h"
#include "alloc/cml_allocator.h"
#include "test_harness.h"
#include "test_require.h"

/* Walk a schedule and check it is a valid execution order: every unit appears
 * exactly once, and each runs only after the units it depends on
 * (F(s,m) after F(s-1,m); B(s,m) after F(s,m) and after B(s+1,m)). */
static bool schedule_is_valid(const PipeUnit* sched, int n, int P, int M) {
    if (!sched || n != 2 * P * M)
        return false;

    bool* f_done = calloc((size_t)P * (size_t)M, sizeof(bool));
    bool* b_done = calloc((size_t)P * (size_t)M, sizeof(bool));
    REQUIRE(f_done && b_done);

    bool ok = true;
    for (int u = 0; u < n && ok; u++) {
        int s = sched[u].stage, m = sched[u].micro_batch;
        if (s < 0 || s >= P || m < 0 || m >= M) {
            ok = false;
            break;
        }
        size_t idx = (size_t)s * (size_t)M + (size_t)m;

        if (sched[u].kind == PIPE_UNIT_FORWARD) {
            if (f_done[idx]) /* duplicate */
                ok = false;
            else if (s > 0 && !f_done[idx - (size_t)M]) /* upstream forward */
                ok = false;
            else
                f_done[idx] = true;
        } else {
            if (b_done[idx])
                ok = false;
            else if (!f_done[idx]) /* own forward */
                ok = false;
            else if (s < P - 1 && !b_done[idx + (size_t)M]) /* downstream backward */
                ok = false;
            else
                b_done[idx] = true;
        }
    }

    for (size_t i = 0; ok && i < (size_t)P * (size_t)M; i++)
        if (!f_done[i] || !b_done[i])
            ok = false;

    free(f_done);
    free(b_done);
    return ok;
}

/* Peak number of micro-batches whose activations are live at once: a forward
 * makes (s,m) live, the backward at stage 0 retires m everywhere. This is the
 * property 1F1B exists to bound. */
static int schedule_peak_live(const PipeUnit* sched, int n, int P, int M) {
    bool* live = calloc((size_t)M, sizeof(bool));
    REQUIRE(live);

    int peak = 0, cur = 0;
    for (int u = 0; u < n; u++) {
        int m = sched[u].micro_batch;
        if (sched[u].kind == PIPE_UNIT_FORWARD) {
            if (!live[m]) {
                live[m] = true;
                if (++cur > peak)
                    peak = cur;
            }
        } else if (sched[u].stage == 0) {
            if (live[m]) {
                live[m] = false;
                cur--;
            }
        }
    }
    (void)P;
    free(live);
    return peak;
}

static bool test_gpipe_schedule_valid(void) {
    for (int P = 1; P <= 5; P++) {
        for (int M = 1; M <= 6; M++) {
            int n           = 0;
            PipeUnit* sched = cml_pipeline_build_schedule(P, M, false, &n);
            if (!schedule_is_valid(sched, n, P, M)) {
                cml_free(sched);
                printf("(GPipe invalid at P=%d M=%d) ", P, M);
                return false;
            }
            cml_free(sched);
        }
    }
    return true;
}

static bool test_interleaved_schedule_valid(void) {
    for (int P = 1; P <= 6; P++) {
        for (int M = 1; M <= 8; M++) {
            int n           = 0;
            PipeUnit* sched = cml_pipeline_build_schedule(P, M, true, &n);
            if (!schedule_is_valid(sched, n, P, M)) {
                cml_free(sched);
                printf("(1F1B invalid at P=%d M=%d) ", P, M);
                return false;
            }
            cml_free(sched);
        }
    }
    return true;
}

/* GPipe holds every micro-batch at once; 1F1B must hold strictly fewer as soon
 * as there are more micro-batches than stages -- that gap is the whole reason
 * the schedule exists. */
static bool test_interleaved_bounds_live_activations(void) {
    const int P = 4, M = 8;

    int ng = 0, ni = 0;
    PipeUnit* g = cml_pipeline_build_schedule(P, M, false, &ng);
    PipeUnit* i = cml_pipeline_build_schedule(P, M, true, &ni);
    REQUIRE(g && i);

    int peak_g = schedule_peak_live(g, ng, P, M);
    int peak_i = schedule_peak_live(i, ni, P, M);
    cml_free(g);
    cml_free(i);

    /* GPipe runs every forward before any backward. */
    if (peak_g != M) {
        printf("(gpipe peak %d != %d) ", peak_g, M);
        return false;
    }
    /* 1F1B keeps at most one in flight per stage. */
    if (peak_i > P) {
        printf("(1f1b peak %d > %d) ", peak_i, P);
        return false;
    }
    return peak_i < peak_g;
}

/* With one micro-batch there is no room to interleave, so both schedules must
 * agree -- a guard against the 1F1B generator inventing an order in the
 * degenerate case. */
static bool test_single_micro_batch_orders_match(void) {
    const int P = 3;
    int ng = 0, ni = 0;
    PipeUnit* g = cml_pipeline_build_schedule(P, 1, false, &ng);
    PipeUnit* i = cml_pipeline_build_schedule(P, 1, true, &ni);
    REQUIRE(g && i);

    bool ok = (ng == ni) && memcmp(g, i, (size_t)ng * sizeof(PipeUnit)) == 0;
    cml_free(g);
    cml_free(i);
    return ok;
}

static bool test_schedule_rejects_bad_input(void) {
    int n   = 0;
    bool ok = cml_pipeline_build_schedule(0, 4, false, &n) == NULL;
    ok      = ok && cml_pipeline_build_schedule(4, 0, false, &n) == NULL;
    ok      = ok && cml_pipeline_build_schedule(-1, 4, true, &n) == NULL;
    ok      = ok && cml_pipeline_build_schedule(4, 4, false, NULL) == NULL;
    return ok;
}

/* The schedules differ in order for a realistic shape. */
static bool test_schedules_differ(void) {
    const int P = 3, M = 4;
    int ng = 0, ni = 0;
    PipeUnit* g = cml_pipeline_build_schedule(P, M, false, &ng);
    PipeUnit* i = cml_pipeline_build_schedule(P, M, true, &ni);
    REQUIRE(g && i);
    bool differ = (ng == ni) && memcmp(g, i, (size_t)ng * sizeof(PipeUnit)) != 0;
    cml_free(g);
    cml_free(i);
    return differ;
}

/* ---- end-to-end: both schedules must train to the same gradients ---------- */

static Tensor* make_input(int rows, int cols, float bias) {
    float* d = malloc((size_t)rows * (size_t)cols * sizeof(float));
    REQUIRE(d);
    for (int i = 0; i < rows * cols; i++)
        d[i] = bias + 0.05f * (float)((i % 7) - 3);

    int shape[2]     = {rows, cols};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* t = tensor_from_data(d, shape, 2, &cfg);
    free(d);
    return t;
}

/* Run one forward+backward of a 2-stage pipeline under `interleaved` and return
 * stage 0's accumulated weight gradient. The two stages' weights are seeded
 * identically across calls so the two runs are comparable. */
static bool run_pipeline_grads(bool interleaved, int M, float* out, int out_len) {
    Linear* l0 = nn_linear(4, 4, DTYPE_FLOAT32, DEVICE_CPU, false);
    Linear* l1 = nn_linear(4, 2, DTYPE_FLOAT32, DEVICE_CPU, false);
    REQUIRE(l0 && l1);

    /* Deterministic weights: nn_linear randomises, which would make the two
     * runs incomparable. */
    Tensor* w0 = l0->weight->tensor;
    Tensor* w1 = l1->weight->tensor;
    REQUIRE(tensor_ensure_executed(w0) == 0 && tensor_ensure_executed(w1) == 0);
    float* w0d = (float*)tensor_data_ptr(w0);
    float* w1d = (float*)tensor_data_ptr(w1);
    REQUIRE(w0d && w1d);
    for (size_t i = 0; i < w0->numel; i++)
        w0d[i] = 0.1f + 0.01f * (float)(i % 5);
    for (size_t i = 0; i < w1->numel; i++)
        w1d[i] = -0.05f + 0.02f * (float)(i % 3);

    PipelineStage stages[2] = {{.module = &l0->base, .stage_id = 0, .device = DEVICE_CPU},
                               {.module = &l1->base, .stage_id = 1, .device = DEVICE_CPU}};
    PipelineConfig cfg      = {.num_micro_batches = M, .num_stages = 2, .interleaved = interleaved};

    CMLPipelineParallel* p = cml_pipeline_create(stages, 2, &cfg);
    REQUIRE(p);

    const int B   = 4;
    Tensor* input = make_input(B, 4, 0.5f);
    REQUIRE(input);
    tensor_set_requires_grad(input, false);

    Tensor* output = cml_pipeline_forward(p, input);
    bool ok        = output != NULL && output->shape[0] == B && output->shape[1] == 2;

    if (ok) {
        Tensor* grad = make_input(B, 2, 1.0f);
        REQUIRE(grad);
        ok = cml_pipeline_backward(p, grad) == 0;
        tensor_free(grad);
    }

    if (ok) {
        Tensor* g = w0->grad;
        ok        = g != NULL && (int)g->numel == out_len;
        if (ok) {
            REQUIRE(tensor_ensure_executed(g) == 0);
            const float* gd = (const float*)tensor_data_ptr(g);
            ok              = gd != NULL;
            if (ok)
                memcpy(out, gd, (size_t)out_len * sizeof(float));
        }
    }

    if (output)
        tensor_free(output);
    tensor_free(input);
    cml_pipeline_free(p);
    module_free(&l0->base);
    module_free(&l1->base);
    return ok;
}

static bool test_interleaved_matches_gpipe_grads(void) {
    const int M = 4, N = 16; /* stage 0 is 4x4 */
    float g_gpipe[16], g_1f1b[16];

    if (!run_pipeline_grads(false, M, g_gpipe, N))
        return false;
    if (!run_pipeline_grads(true, M, g_1f1b, N))
        return false;

    for (int i = 0; i < N; i++) {
        /* Same additions, same order per micro-batch chain -- expect bitwise
         * agreement, not just closeness. */
        if (g_gpipe[i] != g_1f1b[i]) {
            printf("(grad[%d] %g vs %g) ", i, (double)g_gpipe[i], (double)g_1f1b[i]);
            return false;
        }
    }

    /* A pipeline that produced no gradient at all would pass the loop above. */
    bool nonzero = false;
    for (int i = 0; i < N; i++)
        if (fabsf(g_gpipe[i]) > 1e-9f)
            nonzero = true;
    return nonzero;
}

int main(void) {
    printf("Pipeline Schedule Tests\n\n");

    printf("Schedule construction:\n");
    TEST(gpipe_schedule_valid);
    TEST(interleaved_schedule_valid);
    TEST(schedules_differ);
    TEST(single_micro_batch_orders_match);
    TEST(schedule_rejects_bad_input);

    printf("\n1F1B activation bound:\n");
    TEST(interleaved_bounds_live_activations);

    printf("\nEnd-to-end equivalence:\n");
    TEST(interleaved_matches_gpipe_grads);

    return TEST_SUMMARY();
}
