#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cml.h"
#include "ops/ir/schedule.h"
#include "ops/ir/ir.h"
#include "ops/uops.h"
#include "tensor/tensor.h"

static int tests_run    = 0;
static int tests_passed = 0;

#define RUN_TEST(test) do { \
    tests_run++; \
    printf("  [%d] %-55s ", tests_run, #test); \
    if (test()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)


static Tensor* make_tensor(int d0, int d1) {
    int shape[2] = { d0, d1 };
    TensorConfig cfg = {0};
    return tensor_empty(shape, 2, &cfg);
}


static int test_v2_create_empty(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    CMLScheduleV2* s = cml_schedule_v2_create(g, NULL);
    int ok = (s != NULL
              && s->num_groups == 0
              && s->total_ops_before == 0
              && s->total_groups_after == 0);
    cml_schedule_v2_free(s);
    cml_ir_free(g);
    return ok;
}

static int test_v2_create_null(void) {
    CMLScheduleV2* s = cml_schedule_v2_create(NULL, NULL);
    int ok = (s != NULL && s->num_groups == 0);
    cml_schedule_v2_free(s);
    return ok;
}

static int test_v2_free_null(void) {
    /* Must not crash */
    cml_schedule_v2_free(NULL);
    return 1;
}

static int test_v2_single_op(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);
    Tensor* ins[2] = { a, b };
    cml_ir_add_uop(g, UOP_ADD, ins, 2, NULL);

    CMLScheduleV2* s = cml_schedule_v2_create(g, NULL);
    int ok = (s != NULL
              && s->num_groups == 1
              && s->total_ops_before == 1);
    if (ok) {
        ok = (s->groups[0] != NULL
              && s->groups[0]->num_nodes == 1
              && s->groups[0]->type == SCHED_ELEMENTWISE);
    }
    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}


static int test_analyze_fusion_elem_elem(void) {
    /* Two elementwise ops should be fuseable */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);
    Tensor* ins[2] = { a, b };

    cml_ir_add_uop(g, UOP_ADD, ins, 2, NULL);
    struct IRNode* node_add = cml_ir_get_tail(g);

    cml_ir_add_uop(g, UOP_MUL, ins, 2, NULL);
    struct IRNode* node_mul = cml_ir_get_tail(g);

    CMLFusionAnalysis analysis = cml_schedule_analyze_fusion(node_add, node_mul);
    int ok = (analysis.can_fuse == true);

    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_analyze_fusion_reduce_elem_no(void) {
    /* reduce -> elem should NOT fuse */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);

    Tensor* ins_sum[1] = { a };
    cml_ir_add_uop(g, UOP_SUM, ins_sum, 1, NULL);
    struct IRNode* node_sum = cml_ir_get_tail(g);

    Tensor* ins_add[2] = { a, b };
    cml_ir_add_uop(g, UOP_ADD, ins_add, 2, NULL);
    struct IRNode* node_add = cml_ir_get_tail(g);

    CMLFusionAnalysis analysis = cml_schedule_analyze_fusion(node_sum, node_add);
    int ok = (analysis.can_fuse == false);

    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_analyze_fusion_null(void) {
    CMLFusionAnalysis analysis = cml_schedule_analyze_fusion(NULL, NULL);
    return (analysis.can_fuse == false && analysis.benefit == 0.0f);
}


static int test_v2_elementwise_chain_fused(void) {
    /* add -> mul -> neg should fuse into 1 group */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);

    Tensor* ins2[2] = { a, b };
    Tensor* ins1[1] = { a };

    cml_ir_add_uop(g, UOP_ADD, ins2, 2, NULL);
    cml_ir_add_uop(g, UOP_MUL, ins2, 2, NULL);
    cml_ir_add_uop(g, UOP_NEG, ins1, 1, NULL);

    CMLScheduleV2* s = cml_schedule_v2_create(g, NULL);
    int ok = (s != NULL
              && s->num_groups == 1
              && s->total_ops_before == 3);
    if (ok) {
        ok = (s->groups[0]->num_nodes == 3
              && s->groups[0]->type == SCHED_ELEMENTWISE);
    }
    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}


static int test_v2_movement_no_kernel(void) {
    /* reshape between two elem ops: absorbed, no extra kernel */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);
    Tensor* ins2[2] = { a, b };
    Tensor* ins1[1] = { a };

    cml_ir_add_uop(g, UOP_ADD, ins2, 2, NULL);
    cml_ir_add_uop(g, UOP_RESHAPE, ins1, 1, NULL);
    cml_ir_add_uop(g, UOP_MUL, ins2, 2, NULL);

    CMLScheduleV2* s = cml_schedule_v2_create(g, NULL);
    int ok = (s != NULL && s->num_groups == 1 && s->total_ops_before == 3);
    if (ok) {
        /* All 3 ops (add, reshape, mul) should be in a single group */
        ok = (s->groups[0]->num_nodes == 3);
    }
    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_v2_movement_only(void) {
    /* A graph with only movements should produce groups but no real kernels */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* ins[1] = { a };

    cml_ir_add_uop(g, UOP_RESHAPE, ins, 1, NULL);
    cml_ir_add_uop(g, UOP_PERMUTE, ins, 1, NULL);

    CMLScheduleV2* s = cml_schedule_v2_create(g, NULL);
    int ok = (s != NULL && s->total_ops_before == 2);
    /* Movements absorbed into 1 group */
    ok = ok && (s->num_groups == 1);
    if (ok) {
        ok = (s->groups[0]->type == SCHED_MOVEMENT);
    }
    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    return ok;
}


static int test_v2_reduce_breaks_fusion(void) {
    /* add -> sum -> mul: add+sum fuse, mul starts a new group */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);

    Tensor* ins2[2] = { a, b };
    Tensor* ins1[1] = { a };

    cml_ir_add_uop(g, UOP_ADD, ins2, 2, NULL);
    cml_ir_add_uop(g, UOP_SUM, ins1, 1, NULL);
    cml_ir_add_uop(g, UOP_MUL, ins2, 2, NULL);

    CMLScheduleV2* s = cml_schedule_v2_create(g, NULL);
    int ok = (s != NULL
              && s->total_ops_before == 3
              && s->num_groups == 2);
    if (ok) {
        /* First group: add + sum (reduce), second: mul (elem) */
        ok = (s->groups[0]->type == SCHED_REDUCE
              && s->groups[0]->num_nodes == 2);
        ok = ok && (s->groups[1]->type == SCHED_ELEMENTWISE
                    && s->groups[1]->num_nodes == 1);
    }
    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}


static int test_v2_matmul_bias_fused(void) {
    /* matmul -> add (bias) should fuse */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 8);
    Tensor* b = make_tensor(8, 4);
    Tensor* bias = make_tensor(1, 4);

    Tensor* ins_mm[2] = { a, b };
    cml_ir_add_uop(g, UOP_MATMUL, ins_mm, 2, NULL);

    Tensor* ins_add[2] = { a, bias };
    cml_ir_add_uop(g, UOP_ADD, ins_add, 2, NULL);

    CMLScheduleV2* s = cml_schedule_v2_create(g, NULL);
    int ok = (s != NULL && s->num_groups == 1);
    if (ok) {
        ok = (s->groups[0]->type == SCHED_MATMUL
              && s->groups[0]->num_nodes == 2);
    }
    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    tensor_free(bias);
    return ok;
}


static int test_v2_fusion_ratio_gt_1(void) {
    /* 6 elem ops -> 1 group => ratio = 6.0 (well above 1.0) */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);
    Tensor* ins2[2] = { a, b };
    Tensor* ins1[1] = { a };

    cml_ir_add_uop(g, UOP_ADD, ins2, 2, NULL);
    cml_ir_add_uop(g, UOP_MUL, ins2, 2, NULL);
    cml_ir_add_uop(g, UOP_SUB, ins2, 2, NULL);
    cml_ir_add_uop(g, UOP_NEG, ins1, 1, NULL);
    cml_ir_add_uop(g, UOP_EXP, ins1, 1, NULL);
    cml_ir_add_uop(g, UOP_SIGMOID, ins1, 1, NULL);

    CMLScheduleV2* s = cml_schedule_v2_create(g, NULL);
    int ok = (s != NULL
              && s->total_ops_before == 6
              && s->total_groups_after == 1
              && s->fusion_ratio > 1.0f);
    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_v2_fusion_ratio_value(void) {
    /* 6 elem ops in 1 group => ratio should be ~6.0 */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);
    Tensor* ins2[2] = { a, b };
    Tensor* ins1[1] = { a };

    cml_ir_add_uop(g, UOP_ADD, ins2, 2, NULL);
    cml_ir_add_uop(g, UOP_MUL, ins2, 2, NULL);
    cml_ir_add_uop(g, UOP_SUB, ins2, 2, NULL);
    cml_ir_add_uop(g, UOP_NEG, ins1, 1, NULL);
    cml_ir_add_uop(g, UOP_EXP, ins1, 1, NULL);
    cml_ir_add_uop(g, UOP_SIGMOID, ins1, 1, NULL);

    CMLScheduleV2* s = cml_schedule_v2_create(g, NULL);
    float ratio = s ? s->fusion_ratio : 0.0f;
    int ok = (ratio > 5.5f && ratio < 6.5f);
    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}


static int test_v2_execution_order(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);
    Tensor* ins2[2] = { a, b };
    Tensor* ins1[1] = { a };

    cml_ir_add_uop(g, UOP_ADD, ins2, 2, NULL);
    cml_ir_add_uop(g, UOP_SUM, ins1, 1, NULL);
    cml_ir_add_uop(g, UOP_MUL, ins2, 2, NULL);

    CMLScheduleV2* s = cml_schedule_v2_create(g, NULL);
    int ok = (s != NULL
              && s->execution_order != NULL
              && s->num_ordered == s->num_groups);
    /* Order should be sequential: 0, 1, ... */
    if (ok) {
        for (int i = 0; i < s->num_ordered; i++) {
            if (s->execution_order[i] != i) { ok = 0; break; }
        }
    }
    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}


static int test_v2_print(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(2, 2);
    Tensor* ins[1] = { a };
    cml_ir_add_uop(g, UOP_NEG, ins, 1, NULL);

    CMLScheduleV2* s = cml_schedule_v2_create(g, NULL);
    cml_schedule_v2_print(s);
    cml_schedule_v2_print(NULL);  /* should not crash */
    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    return 1;
}


static int test_v2_group_colors(void) {
    /* Two separate groups should have distinct colors */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);

    Tensor* ins2[2] = { a, b };
    Tensor* ins1[1] = { a };

    cml_ir_add_uop(g, UOP_ADD, ins2, 2, NULL);
    cml_ir_add_uop(g, UOP_SUM, ins1, 1, NULL);  /* fuses with add */
    cml_ir_add_uop(g, UOP_MUL, ins2, 2, NULL);  /* new group */

    CMLScheduleV2* s = cml_schedule_v2_create(g, NULL);
    int ok = (s != NULL && s->num_groups == 2);
    if (ok) {
        ok = (s->groups[0]->color != s->groups[1]->color);
    }
    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}


int main(void) {
    printf("\n=== test_schedule_v2 ===\n\n");

    RUN_TEST(test_v2_create_empty);
    RUN_TEST(test_v2_create_null);
    RUN_TEST(test_v2_free_null);
    RUN_TEST(test_v2_single_op);
    RUN_TEST(test_analyze_fusion_elem_elem);
    RUN_TEST(test_analyze_fusion_reduce_elem_no);
    RUN_TEST(test_analyze_fusion_null);
    RUN_TEST(test_v2_elementwise_chain_fused);
    RUN_TEST(test_v2_movement_no_kernel);
    RUN_TEST(test_v2_movement_only);
    RUN_TEST(test_v2_reduce_breaks_fusion);
    RUN_TEST(test_v2_matmul_bias_fused);
    RUN_TEST(test_v2_fusion_ratio_gt_1);
    RUN_TEST(test_v2_fusion_ratio_value);
    RUN_TEST(test_v2_execution_order);
    RUN_TEST(test_v2_print);
    RUN_TEST(test_v2_group_colors);

    printf("\n  Results: %d/%d passed\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
