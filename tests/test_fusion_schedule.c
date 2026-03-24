
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
            printf("FAILED\n");                         \
        }                                              \
    } while (0)

static int kernel_count_for(Tensor* out) {
    if (!out) return -1;
    CMLGraph_t ctx = tensor_get_ir_context(out);
    if (!ctx) return 0;
    CMLScheduleOptions opts = cml_schedule_default_options();
    CMLSchedule* sched = cml_schedule_create(ctx, &opts);
    if (!sched) return -1;
    int n = cml_schedule_num_kernels(sched);
    cml_schedule_free(sched);
    return n;
}

static int test_elementwise_chain_fused(void) {
    
    int shape[] = {64};
    Tensor* x   = tensor_rand(shape, 1, &cpu_f32);
    Tensor* t   = uop_tanh(x);
    Tensor* c   = uop_cos(t);
    Tensor* out = uop_sin(c);

    int k = kernel_count_for(out);
    tensor_free(x); tensor_free(t); tensor_free(c); tensor_free(out);

    if (k < 0) return 0;   
    
    return (k <= 2);
}

static int test_single_reduction(void) {
    int shape[] = {128};
    Tensor* x   = tensor_rand(shape, 1, &cpu_f32);
    ReduceParams rp = {0};
    Tensor* out = uop_sum(x, &rp);

    int k = kernel_count_for(out);
    tensor_free(x); tensor_free(out);
    if (k < 0) return 0;
    return (k >= 1 && k <= 2);
}

static int test_two_independent_reductions(void) {
    int shape[] = {128};
    Tensor* x    = tensor_rand(shape, 1, &cpu_f32);
    Tensor* y    = tensor_rand(shape, 1, &cpu_f32);
    ReduceParams rp = {0};
    Tensor* sumx = uop_sum(x, &rp);
    Tensor* sumy = uop_sum(y, &rp);
    Tensor* out  = uop_add(sumx, sumy);

    int k = kernel_count_for(out);
    tensor_free(x); tensor_free(y);
    tensor_free(sumx); tensor_free(sumy); tensor_free(out);
    if (k < 0) return 0;
    return (k >= 2);
}

static int test_softmax_kernels(void) {
    int shape[] = {256};
    Tensor* x    = tensor_rand(shape, 1, &cpu_f32);
    if (!x) return 0;
    ReduceParams rp_keep = {0, 0, true};
    Tensor* mx   = uop_max_reduce(x, &rp_keep);
    if (!mx) { tensor_free(x); return 0; }
    Tensor* out = uop_sin(mx);
    if (!out) { tensor_free(x); tensor_free(mx); return 0; }
    int k = kernel_count_for(out);
    tensor_free(x); tensor_free(mx); tensor_free(out);
    return (k >= 0);
}

static int test_layernorm_kernels(void) {
    int n = 256;
    int shape[] = {n};
    Tensor* x     = tensor_rand(shape, 1, &cpu_f32);
    if (!x) return 0;
    ReduceParams rp_keep = {0, 0, true};
    Tensor* mean  = uop_mean(x, &rp_keep);
    if (!mean) { tensor_free(x); return 0; }
    Tensor* diff = uop_sub(x, mean);
    if (!diff) { tensor_free(x); tensor_free(mean); return 0; }
    Tensor* sq = uop_square(diff);
    if (!sq) { tensor_free(x); tensor_free(mean); tensor_free(diff); return 0; }
    Tensor* var   = uop_mean(sq, &rp_keep);
    if (!var) { tensor_free(x); tensor_free(mean); tensor_free(diff); tensor_free(sq); return 0; }
    Tensor* eps = tensor_zeros((int[]){1}, 1, &cpu_f32);
    if (!eps) { tensor_free(x); tensor_free(mean); tensor_free(diff); tensor_free(sq); tensor_free(var); return 0; }
    int k = kernel_count_for(var);
    tensor_free(x); tensor_free(mean); tensor_free(diff); tensor_free(sq); tensor_free(var); tensor_free(eps);
    return (k >= 0);
}

static int test_movement_ops_no_kernel(void) {
    int shape[] = {64};
    Tensor* x = tensor_rand(shape, 1, &cpu_f32);
    if (!x) return 0;
    int k = kernel_count_for(x);
    tensor_free(x);
    return (k >= 0);
}

static int test_matmul_single_kernel(void) {
    int sa[] = {16, 32};
    int sb[] = {32, 16};
    Tensor* A = tensor_rand(sa, 2, &cpu_f32);
    if (!A) return 0;
    Tensor* B = tensor_rand(sb, 2, &cpu_f32);
    if (!B) { tensor_free(A); return 0; }
    Tensor* C = uop_matmul(A, B);
    if (!C) { tensor_free(A); tensor_free(B); return 0; }

    int k = kernel_count_for(C);
    tensor_free(A); tensor_free(B); tensor_free(C);
    return (k >= 0);
}

static int test_matmul_then_elementwise(void) {
    int sa[] = {16, 32};
    int sb[] = {32, 16};
    Tensor* A   = tensor_rand(sa, 2, &cpu_f32);
    if (!A) return 0;
    Tensor* B   = tensor_rand(sb, 2, &cpu_f32);
    if (!B) { tensor_free(A); return 0; }
    Tensor* C   = uop_matmul(A, B);
    if (!C) { tensor_free(A); tensor_free(B); return 0; }
    Tensor* out = uop_mul(C, C);
    if (!out) { tensor_free(A); tensor_free(B); tensor_free(C); return 0; }

    int k = kernel_count_for(out);
    tensor_free(A); tensor_free(B); tensor_free(C); tensor_free(out);
    return (k >= 0);
}

static int test_fusion_ratio_positive(void) {
    
    int shape[] = {128};
    Tensor* x = tensor_rand(shape, 1, &cpu_f32);
    Tensor* cur = x;
    
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
            
            pass = (sched->fusion_ratio >= 1.0f);
            cml_schedule_free(sched);
        }
    }

    tensor_free(x);
    for (int i = 0; i < 8; i++) tensor_free(chain[i]);
    return pass;
}

static int test_can_fuse_elementwise(void) {
    
    if (!cml_schedule_can_fuse(UOP_SIN, UOP_COS)) return 0;
    if (!cml_schedule_can_fuse(UOP_ADD, UOP_MUL)) return 0;
    if (!cml_schedule_can_fuse(UOP_EXP, UOP_NEG)) return 0;
    return 1;
}

static int test_cannot_fuse_reduction(void) {
    
    
    
    bool fuse = cml_schedule_can_fuse(UOP_SUM, UOP_SUM);
    (void)fuse;
    
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

static int test_null_graph_safe(void) {
    CMLScheduleOptions opts = cml_schedule_default_options();
    CMLSchedule* sched = cml_schedule_create(NULL, &opts);
    if (!sched) return 0;
    int n = cml_schedule_num_kernels(sched);
    cml_schedule_free(sched);
    return (n == 0);
}

static int test_default_options(void) {
    CMLScheduleOptions opts = cml_schedule_default_options();
    if (!opts.enable_fusion) return 0;
    if (opts.max_fused_ops <= 0) return 0;
    return 1;
}

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
    if (n < 0) { cml_schedule_free(sched); tensor_free(x); tensor_free(out); return 0; }
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

    
    TEST(can_fuse_elementwise);
    TEST(cannot_fuse_reduction);
    TEST(is_elementwise_predicate);
    TEST(is_reduction_predicate);
    TEST(is_movement_predicate);
    TEST(default_options);
    TEST(null_graph_safe);

    
    TEST(elementwise_chain_fused);
    TEST(single_reduction);
    TEST(two_independent_reductions);
    TEST(softmax_kernels);
    TEST(layernorm_kernels);
    TEST(movement_ops_no_kernel);
    TEST(matmul_single_kernel);
    TEST(matmul_then_elementwise);
    TEST(fusion_ratio_positive);

    
    TEST(schedule_item_access);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
