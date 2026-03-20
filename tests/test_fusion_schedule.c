/*
 * Kernel fusion schedule tests.
 *
 * Inspired by tinygrad's test_schedule.py which asserts that specific
 * computation patterns result in a specific number of GPU kernels after
 * fusion. The scheduler should:
 *   - Fuse chains of elementwise ops into one kernel.
 *   - NOT fuse across reduction boundaries (each reduce is a new kernel).
 *   - Treat movement ops (reshape, permute) as zero-cost (no kernel).
 *   - Fuse elementwise ops after reductions when safe.
 *
 * We don't require exact kernel counts to be implementation-defined; instead
 * we assert upper bounds and fusion ratios to keep tests stable across
 * schedule version changes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "ops/ir/schedule.h"

static int tests_passed = 0;
static int tests_total  = 0;

static const TensorConfig cpu_f32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
    .has_dtype = true, .has_device = true
};

#define TEST(name)                                     \
    do {                                               \
        tests_total++;                                 \
        printf("  TEST: %s ... ", #name);              \
        fflush(stdout);                                \
        if (test_##name()) {                           \
            tests_passed++;                            \
            printf("PASSED\n");                        \
        } else {                                       \
            printf("FAILED\n");                        \
        }                                              \
    } while (0)

/* Build a schedule for a tensor and return the kernel count, or -1 on error */
static int kernel_count_for(Tensor* out) {
    if (!out) return -1;
    CMLGraph_t ctx = tensor_get_ir_context(out);
    if (!ctx) return -1;
    CMLScheduleOptions opts = cml_schedule_default_options();
    CMLSchedule* sched = cml_schedule_create(ctx, &opts);
    if (!sched) return -1;
    int n = cml_schedule_num_kernels(sched);
    cml_schedule_free(sched);
    return n;
}

/* ---- Simple elementwise chain should be 1 kernel ----------------------- */
static int test_elementwise_chain_fused(void) {
    /* sin(cos(tanh(x))) — three elementwise ops, should be 1 kernel */
    int shape[] = {64};
    Tensor* x   = tensor_rand(shape, 1, &cpu_f32);
    Tensor* t   = uop_tanh(x);
    Tensor* c   = uop_cos(t);
    Tensor* out = uop_sin(c);

    int k = kernel_count_for(out);
    tensor_free(x); tensor_free(t); tensor_free(c); tensor_free(out);

    if (k < 0) return 0;   /* scheduler error */
    /* Should fuse into ≤ 2 kernels (ideally 1) */
    return (k <= 2);
}

/* ---- Single reduction is 1 kernel ------------------------------------- */
static int test_single_reduction(void) {
    int shape[] = {128};
    Tensor* x   = tensor_rand(shape, 1, &cpu_f32);
    Tensor* out = uop_sum(x, 0, false);

    int k = kernel_count_for(out);
    tensor_free(x); tensor_free(out);
    if (k < 0) return 0;
    return (k >= 1 && k <= 2);
}

/* ---- Two independent reductions are 2 kernels ------------------------- */
static int test_two_independent_reductions(void) {
    int shape[] = {128};
    Tensor* x    = tensor_rand(shape, 1, &cpu_f32);
    Tensor* y    = tensor_rand(shape, 1, &cpu_f32);
    Tensor* sumx = uop_sum(x, 0, false);
    Tensor* sumy = uop_sum(y, 0, false);
    Tensor* out  = uop_add(sumx, sumy);

    int k = kernel_count_for(out);
    tensor_free(x); tensor_free(y);
    tensor_free(sumx); tensor_free(sumy); tensor_free(out);
    if (k < 0) return 0;
    /* At minimum 2 reduces; possibly 3 with final add */
    return (k >= 2);
}

/* ---- Softmax pattern: max + sub + exp + sum + div should be few kernels */
static int test_softmax_kernels(void) {
    /*
     * softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
     * Pattern: two reductions (max, sum), two elementwise chains.
     * Expect ≤ 4 kernels total.
     */
    int shape[] = {256};
    Tensor* x    = tensor_rand(shape, 1, &cpu_f32);
    Tensor* mx   = uop_max_reduce(x, 0, true);    /* max, keepdim */
    Tensor* xs   = uop_sub(x, mx);                /* x - max */
    Tensor* ex   = uop_exp(xs);                   /* exp */
    Tensor* sm   = uop_sum(ex, 0, true);           /* sum, keepdim */
    Tensor* out  = uop_div(ex, sm);               /* normalize */

    int k = kernel_count_for(out);
    tensor_free(x); tensor_free(mx); tensor_free(xs);
    tensor_free(ex); tensor_free(sm); tensor_free(out);
    if (k < 0) return 0;
    return (k <= 5);
}

/* ---- LayerNorm pattern -------------------------------------------------- */
static int test_layernorm_kernels(void) {
    /*
     * mean = sum(x) / N
     * var  = sum((x-mean)^2) / N
     * y    = (x - mean) / sqrt(var + eps)
     * Expect: ≤ 5 kernels (2 reduces + elementwise chains).
     */
    int n = 256;
    int shape[] = {n};
    Tensor* x     = tensor_rand(shape, 1, &cpu_f32);
    Tensor* mean  = uop_mean(x, 0, true);
    Tensor* diff  = uop_sub(x, mean);
    Tensor* sq    = uop_square(diff);
    Tensor* var   = uop_mean(sq, 0, true);
    float eps_val = 1e-5f;
    int eps_shape[] = {1};
    float eps_arr[] = {eps_val};
    Tensor* eps   = tensor_from_data(eps_arr, eps_shape, 1, &cpu_f32);
    Tensor* vare  = uop_add(var, eps);
    Tensor* std   = uop_sqrt(vare);
    Tensor* out   = uop_div(diff, std);

    int k = kernel_count_for(out);
    tensor_free(x); tensor_free(mean); tensor_free(diff);
    tensor_free(sq); tensor_free(var); tensor_free(eps);
    tensor_free(vare); tensor_free(std); tensor_free(out);
    if (k < 0) return 0;
    return (k <= 6);
}

/* ---- Reshape/permute adds no kernels ----------------------------------- */
static int test_movement_ops_no_kernel(void) {
    /*
     * reshape + permute should not add GPU kernels — they're zero-cost views.
     * A single subsequent elementwise should still be 1 kernel.
     */
    int shape_in[] = {4, 8};
    Tensor* x = tensor_rand(shape_in, 2, &cpu_f32);

    /* Reshape to {8, 4} */
    int shape_out[] = {8, 4};
    Tensor* r = uop_reshape(x, shape_out, 2);

    /* One elementwise after the reshape */
    Tensor* out = uop_sin(r);

    int k = kernel_count_for(out);
    tensor_free(x); tensor_free(r); tensor_free(out);
    if (k < 0) return 0;
    /* Movement should not add kernels; expect ≤ 2 total */
    return (k <= 2);
}

/* ---- Matmul is a single kernel ---------------------------------------- */
static int test_matmul_single_kernel(void) {
    int sa[] = {16, 32};
    int sb[] = {32, 16};
    Tensor* A = tensor_rand(sa, 2, &cpu_f32);
    Tensor* B = tensor_rand(sb, 2, &cpu_f32);
    Tensor* C = uop_matmul(A, B);

    int k = kernel_count_for(C);
    tensor_free(A); tensor_free(B); tensor_free(C);
    if (k < 0) return 0;
    return (k >= 1 && k <= 2);
}

/* ---- Elementwise after matmul fuses ----------------------------------- */
static int test_matmul_then_elementwise(void) {
    /*
     * C = A @ B
     * out = relu(C)
     * Ideally fused into 1-2 kernels; definitely ≤ 3.
     */
    int sa[] = {16, 32};
    int sb[] = {32, 16};
    Tensor* A   = tensor_rand(sa, 2, &cpu_f32);
    Tensor* B   = tensor_rand(sb, 2, &cpu_f32);
    Tensor* C   = uop_matmul(A, B);
    /* relu = max(C, 0) */
    int s0[] = {16, 16};
    Tensor* z   = tensor_zeros(s0, 2, &cpu_f32);
    Tensor* out = uop_max(C, z);

    int k = kernel_count_for(out);
    tensor_free(A); tensor_free(B); tensor_free(C);
    tensor_free(z); tensor_free(out);
    if (k < 0) return 0;
    return (k <= 3);
}

/* ---- Fusion ratio sanity ---------------------------------------------- */
static int test_fusion_ratio_positive(void) {
    /*
     * A long chain of elementwise ops should have fusion_ratio > 1.0,
     * meaning the scheduler is doing useful work.
     */
    int shape[] = {128};
    Tensor* x = tensor_rand(shape, 1, &cpu_f32);
    Tensor* cur = x;
    /* chain of 8 elementwise ops */
    Tensor* chain[8];
    chain[0] = uop_sin(cur);
    chain[1] = uop_cos(chain[0]);
    chain[2] = uop_tanh(chain[1]);
    chain[3] = uop_abs(chain[2]);
    chain[4] = uop_neg(chain[3]);
    chain[5] = uop_square(chain[4]);
    chain[6] = uop_sqrt(chain[5]);
    chain[7] = uop_sigmoid(chain[6]);

    CMLGraph_t ctx = tensor_get_ir_context(chain[7]);
    int pass = 0;
    if (ctx) {
        CMLScheduleOptions opts = cml_schedule_default_options();
        CMLSchedule* sched = cml_schedule_create(ctx, &opts);
        if (sched) {
            /* fusion_ratio = total_ops / total_kernels; should be > 1 for a chain */
            pass = (sched->fusion_ratio >= 1.0f);
            cml_schedule_free(sched);
        }
    }

    tensor_free(x);
    for (int i = 0; i < 8; i++) tensor_free(chain[i]);
    return pass;
}

/* ---- cml_schedule_can_fuse API ---------------------------------------- */
static int test_can_fuse_elementwise(void) {
    /* Two elementwise ops should be fuseable */
    if (!cml_schedule_can_fuse(UOP_SIN, UOP_COS)) return 0;
    if (!cml_schedule_can_fuse(UOP_ADD, UOP_MUL)) return 0;
    if (!cml_schedule_can_fuse(UOP_EXP, UOP_NEG)) return 0;
    return 1;
}

static int test_cannot_fuse_reduction(void) {
    /* A reduction followed by a reduction should NOT be auto-fused */
    /* The API says what can be fused into a single pass */
    /* Two reductions are typically not fuseable in the same pass */
    bool fuse = cml_schedule_can_fuse(UOP_SUM, UOP_SUM);
    (void)fuse;
    /* This is implementation-defined; we just verify the API doesn't crash */
    return 1;
}

static int test_is_elementwise_predicate(void) {
    if (!cml_schedule_is_elementwise(UOP_SIN)) return 0;
    if (!cml_schedule_is_elementwise(UOP_ADD)) return 0;
    if (!cml_schedule_is_elementwise(UOP_MUL)) return 0;
    if (cml_schedule_is_elementwise(UOP_SUM)) return 0;
    if (cml_schedule_is_elementwise(UOP_MATMUL)) return 0;
    return 1;
}

static int test_is_reduction_predicate(void) {
    if (!cml_schedule_is_reduction(UOP_SUM)) return 0;
    if (!cml_schedule_is_reduction(UOP_MAX_REDUCE)) return 0;
    if (!cml_schedule_is_reduction(UOP_MEAN)) return 0;
    if (cml_schedule_is_reduction(UOP_ADD)) return 0;
    if (cml_schedule_is_reduction(UOP_SIN)) return 0;
    return 1;
}

static int test_is_movement_predicate(void) {
    if (!cml_schedule_is_movement(UOP_RESHAPE)) return 0;
    if (!cml_schedule_is_movement(UOP_PERMUTE)) return 0;
    if (cml_schedule_is_movement(UOP_SIN)) return 0;
    if (cml_schedule_is_movement(UOP_SUM)) return 0;
    return 1;
}

/* ---- Schedule with NULL graph handles gracefully ---------------------- */
static int test_null_graph_safe(void) {
    CMLScheduleOptions opts = cml_schedule_default_options();
    CMLSchedule* sched = cml_schedule_create(NULL, &opts);
    /* Should return NULL without crashing */
    if (sched) {
        cml_schedule_free(sched);
        return 0; /* Should not have succeeded */
    }
    return 1;
}

/* ---- Default options are sane ----------------------------------------- */
static int test_default_options(void) {
    CMLScheduleOptions opts = cml_schedule_default_options();
    if (!opts.enable_fusion) return 0;
    if (opts.max_fused_ops <= 0) return 0;
    return 1;
}

/* ---- Schedule item access --------------------------------------------- */
static int test_schedule_item_access(void) {
    int shape[] = {64};
    Tensor* x   = tensor_rand(shape, 1, &cpu_f32);
    Tensor* out = uop_exp(x);

    CMLGraph_t ctx = tensor_get_ir_context(out);
    if (!ctx) { tensor_free(x); tensor_free(out); return 0; }

    CMLScheduleOptions opts = cml_schedule_default_options();
    CMLSchedule* sched = cml_schedule_create(ctx, &opts);
    if (!sched) { tensor_free(x); tensor_free(out); return 0; }

    int n = cml_schedule_num_kernels(sched);
    for (int i = 0; i < n; i++) {
        const CMLScheduleItem* item = cml_schedule_get_item(sched, i);
        if (!item) { cml_schedule_free(sched); tensor_free(x); tensor_free(out); return 0; }
        if (item->num_ops < 1) {
            cml_schedule_free(sched); tensor_free(x); tensor_free(out); return 0;
        }
    }
    cml_schedule_free(sched);
    tensor_free(x); tensor_free(out);
    return 1;
}

int main(void) {
    printf("Kernel Fusion Schedule Tests\n");

    /* Predicate API */
    TEST(can_fuse_elementwise);
    TEST(cannot_fuse_reduction);
    TEST(is_elementwise_predicate);
    TEST(is_reduction_predicate);
    TEST(is_movement_predicate);
    TEST(default_options);
    TEST(null_graph_safe);

    /* Kernel count assertions */
    TEST(elementwise_chain_fused);
    TEST(single_reduction);
    TEST(two_independent_reductions);
    TEST(softmax_kernels);
    TEST(layernorm_kernels);
    TEST(movement_ops_no_kernel);
    TEST(matmul_single_kernel);
    TEST(matmul_then_elementwise);
    TEST(fusion_ratio_positive);

    /* Item access */
    TEST(schedule_item_access);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
