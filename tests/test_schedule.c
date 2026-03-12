/**
 * @file test_schedule.c
 * @brief Tests for automatic kernel scheduling and fusion
 */

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
    printf("  [%d] %-50s ", tests_run, #test); \
    if (test()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

/* ---------- helpers ---------- */

static Tensor* make_tensor(int d0, int d1) {
    int shape[2] = { d0, d1 };
    TensorConfig cfg = {0};
    return tensor_empty(shape, 2, &cfg);
}

static Tensor* make_tensor_4d(int n, int c, int h, int w) {
    int shape[4] = { n, c, h, w };
    TensorConfig cfg = {0};
    return tensor_empty(shape, 4, &cfg);
}

/* ---------- classification tests ---------- */

static int test_is_elementwise(void) {
    return cml_schedule_is_elementwise(UOP_ADD)
        && cml_schedule_is_elementwise(UOP_MUL)
        && cml_schedule_is_elementwise(UOP_SIGMOID)
        && cml_schedule_is_elementwise(UOP_NEG)
        && cml_schedule_is_elementwise(UOP_TANH)
        && cml_schedule_is_elementwise(UOP_ELU)
        && cml_schedule_is_elementwise(UOP_SILU)
        && !cml_schedule_is_elementwise(UOP_SUM)
        && !cml_schedule_is_elementwise(UOP_MATMUL);
}

static int test_is_reduction(void) {
    return cml_schedule_is_reduction(UOP_SUM)
        && cml_schedule_is_reduction(UOP_MEAN)
        && cml_schedule_is_reduction(UOP_MAX_REDUCE)
        && cml_schedule_is_reduction(UOP_PROD)
        && cml_schedule_is_reduction(UOP_ARGMAX)
        && cml_schedule_is_reduction(UOP_LOGSUMEXP)
        && !cml_schedule_is_reduction(UOP_ADD)
        && !cml_schedule_is_reduction(UOP_MATMUL);
}

static int test_is_movement(void) {
    return cml_schedule_is_movement(UOP_RESHAPE)
        && cml_schedule_is_movement(UOP_PERMUTE)
        && cml_schedule_is_movement(UOP_EXPAND)
        && cml_schedule_is_movement(UOP_FLATTEN)
        && cml_schedule_is_movement(UOP_CAT)
        && !cml_schedule_is_movement(UOP_ADD)
        && !cml_schedule_is_movement(UOP_SUM);
}

/* ---------- fusion tests ---------- */

static int test_can_fuse_elementwise(void) {
    return cml_schedule_can_fuse(UOP_ADD, UOP_MUL)
        && cml_schedule_can_fuse(UOP_MUL, UOP_SIGMOID);
}

static int test_cannot_fuse_after_reduce(void) {
    /* reduction -> elementwise must NOT fuse */
    return !cml_schedule_can_fuse(UOP_SUM, UOP_ADD)
        && !cml_schedule_can_fuse(UOP_MEAN, UOP_MUL);
}

static int test_can_fuse_before_reduce(void) {
    /* elementwise -> reduction CAN fuse */
    return cml_schedule_can_fuse(UOP_ADD, UOP_SUM)
        && cml_schedule_can_fuse(UOP_MUL, UOP_MEAN);
}

static int test_can_fuse_matmul_elementwise(void) {
    /* matmul + bias add should fuse */
    return cml_schedule_can_fuse(UOP_MATMUL, UOP_ADD);
}

static int test_movement_fuses_with_anything(void) {
    return cml_schedule_can_fuse(UOP_RESHAPE, UOP_ADD)
        && cml_schedule_can_fuse(UOP_ADD, UOP_RESHAPE)
        && cml_schedule_can_fuse(UOP_RESHAPE, UOP_SUM)
        && cml_schedule_can_fuse(UOP_PERMUTE, UOP_MATMUL);
}

/* ---------- options tests ---------- */

static int test_default_options(void) {
    CMLScheduleOptions opts = cml_schedule_default_options();
    return opts.enable_fusion == true
        && opts.enable_movement_fold == true
        && opts.max_fused_ops == CML_SCHEDULE_MAX_FUSED_OPS
        && opts.estimate_costs == true
        && opts.topological_sort == true;
}

/* ---------- schedule creation tests ---------- */

static int test_schedule_empty_graph(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    CMLSchedule* s = cml_schedule_create(g, NULL);
    int ok = (s != NULL && s->num_items == 0 && s->total_ops == 0);
    cml_schedule_free(s);
    cml_ir_free(g);
    return ok;
}

static int test_schedule_null_graph(void) {
    CMLSchedule* s = cml_schedule_create(NULL, NULL);
    int ok = (s != NULL && s->num_items == 0);
    cml_schedule_free(s);
    return ok;
}

static int test_schedule_single_op(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);
    Tensor* ins[2] = { a, b };
    cml_ir_add_uop(g, UOP_ADD, ins, 2, NULL);

    CMLSchedule* s = cml_schedule_create(g, NULL);
    int ok = (s != NULL
              && cml_schedule_num_kernels(s) == 1
              && s->total_ops == 1);
    if (ok) {
        const CMLScheduleItem* it = cml_schedule_get_item(s, 0);
        ok = (it != NULL && it->type == SCHED_ELEMENTWISE && it->num_ops == 1);
    }
    cml_schedule_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_schedule_elementwise_chain(void) {
    /* add -> mul -> neg should fuse into 1 kernel */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);

    Tensor* ins_add[2] = { a, b };
    cml_ir_add_uop(g, UOP_ADD, ins_add, 2, NULL);

    Tensor* ins_mul[2] = { a, b };
    cml_ir_add_uop(g, UOP_MUL, ins_mul, 2, NULL);

    Tensor* ins_neg[1] = { a };
    cml_ir_add_uop(g, UOP_NEG, ins_neg, 1, NULL);

    CMLSchedule* s = cml_schedule_create(g, NULL);
    int ok = (s != NULL
              && cml_schedule_num_kernels(s) == 1
              && s->total_ops == 3);
    if (ok) {
        const CMLScheduleItem* it = cml_schedule_get_item(s, 0);
        ok = (it && it->type == SCHED_ELEMENTWISE && it->num_ops == 3);
    }
    cml_schedule_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_schedule_reduce_breaks_fusion(void) {
    /*
     * add -> sum -> mul
     * With fusion enabled, add feeds into sum (fused into one REDUCE item),
     * then mul starts a new item.
     * Expected: 2 items (REDUCE containing add+sum, ELEMENTWISE containing mul)
     */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);

    Tensor* ins_add[2] = { a, b };
    cml_ir_add_uop(g, UOP_ADD, ins_add, 2, NULL);

    Tensor* ins_sum[1] = { a };
    cml_ir_add_uop(g, UOP_SUM, ins_sum, 1, NULL);

    Tensor* ins_mul[2] = { a, b };
    cml_ir_add_uop(g, UOP_MUL, ins_mul, 2, NULL);

    CMLSchedule* s = cml_schedule_create(g, NULL);
    int ok = (s != NULL && s->total_ops == 3 && cml_schedule_num_kernels(s) == 2);
    if (ok) {
        const CMLScheduleItem* it0 = cml_schedule_get_item(s, 0);
        const CMLScheduleItem* it1 = cml_schedule_get_item(s, 1);
        ok = (it0 && it0->type == SCHED_REDUCE && it0->num_ops == 2)
          && (it1 && it1->type == SCHED_ELEMENTWISE && it1->num_ops == 1);
    }
    cml_schedule_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_schedule_matmul_item(void) {
    /* matmul gets its own item type */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 8);
    Tensor* b = make_tensor(8, 4);

    Tensor* ins[2] = { a, b };
    cml_ir_add_uop(g, UOP_MATMUL, ins, 2, NULL);

    CMLSchedule* s = cml_schedule_create(g, NULL);
    int ok = (s != NULL && cml_schedule_num_kernels(s) == 1);
    if (ok) {
        const CMLScheduleItem* it = cml_schedule_get_item(s, 0);
        ok = (it && it->type == SCHED_MATMUL && it->num_ops == 1);
    }
    cml_schedule_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_schedule_matmul_bias_fused(void) {
    /* matmul -> add (bias) should fuse into single matmul item */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 8);
    Tensor* b = make_tensor(8, 4);
    Tensor* bias = make_tensor(1, 4);

    Tensor* ins_mm[2] = { a, b };
    cml_ir_add_uop(g, UOP_MATMUL, ins_mm, 2, NULL);

    Tensor* ins_add[2] = { a, bias };
    cml_ir_add_uop(g, UOP_ADD, ins_add, 2, NULL);

    CMLSchedule* s = cml_schedule_create(g, NULL);
    int ok = (s != NULL && cml_schedule_num_kernels(s) == 1);
    if (ok) {
        const CMLScheduleItem* it = cml_schedule_get_item(s, 0);
        /* matmul item absorbs the following elementwise add */
        ok = (it && it->type == SCHED_MATMUL && it->num_ops == 2);
    }
    cml_schedule_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    tensor_free(bias);
    return ok;
}

static int test_schedule_fusion_ratio(void) {
    /* 6 elementwise ops should fuse into 1 kernel => ratio = 6.0 */
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

    CMLSchedule* s = cml_schedule_create(g, NULL);
    int ok = (s != NULL
              && s->total_ops == 6
              && s->total_kernels == 1);
    if (ok) {
        float ratio = s->fusion_ratio;
        ok = (ratio > 5.5f && ratio < 6.5f);
    }
    cml_schedule_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_schedule_no_fusion(void) {
    /* With fusion disabled, each op gets its own item */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);
    Tensor* ins[2] = { a, b };

    cml_ir_add_uop(g, UOP_ADD, ins, 2, NULL);
    cml_ir_add_uop(g, UOP_MUL, ins, 2, NULL);
    cml_ir_add_uop(g, UOP_SUB, ins, 2, NULL);

    CMLScheduleOptions opts = cml_schedule_default_options();
    opts.enable_fusion = false;

    CMLSchedule* s = cml_schedule_create(g, &opts);
    int ok = (s != NULL && s->total_ops == 3 && cml_schedule_num_kernels(s) == 3);
    cml_schedule_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_schedule_print(void) {
    /* Just verify print does not crash */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(2, 2);
    Tensor* ins[1] = { a };
    cml_ir_add_uop(g, UOP_NEG, ins, 1, NULL);

    CMLSchedule* s = cml_schedule_create(g, NULL);
    cml_schedule_print(s);
    cml_schedule_print(NULL);  /* should not crash */
    cml_schedule_free(s);
    cml_ir_free(g);
    tensor_free(a);
    return 1;
}

static int test_schedule_to_string(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(2, 2);
    Tensor* ins[1] = { a };
    cml_ir_add_uop(g, UOP_NEG, ins, 1, NULL);

    CMLSchedule* s = cml_schedule_create(g, NULL);
    char* str = cml_schedule_to_string(s);
    int ok = (str != NULL && strlen(str) > 0);
    free(str);

    /* NULL schedule returns NULL */
    char* str2 = cml_schedule_to_string(NULL);
    ok = ok && (str2 == NULL);

    cml_schedule_free(s);
    cml_ir_free(g);
    tensor_free(a);
    return ok;
}

static int test_schedule_free_null(void) {
    /* free(NULL) must not crash */
    cml_schedule_free(NULL);
    return 1;
}

static int test_schedule_get_item_bounds(void) {
    /* Out-of-bounds returns NULL */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(2, 2);
    Tensor* ins[1] = { a };
    cml_ir_add_uop(g, UOP_NEG, ins, 1, NULL);

    CMLSchedule* s = cml_schedule_create(g, NULL);
    int ok = (cml_schedule_get_item(s, -1) == NULL)
          && (cml_schedule_get_item(s, 999) == NULL)
          && (cml_schedule_get_item(NULL, 0) == NULL)
          && (cml_schedule_get_item(s, 0) != NULL);
    cml_schedule_free(s);
    cml_ir_free(g);
    tensor_free(a);
    return ok;
}

static int test_schedule_num_kernels_null(void) {
    return cml_schedule_num_kernels(NULL) == 0;
}

static int test_schedule_movement_folded(void) {
    /*
     * With movement folding enabled (default), a reshape between two
     * elementwise ops gets absorbed into the elementwise item.
     * add -> reshape -> mul => 1 item
     */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);
    Tensor* ins2[2] = { a, b };
    Tensor* ins1[1] = { a };

    cml_ir_add_uop(g, UOP_ADD, ins2, 2, NULL);
    cml_ir_add_uop(g, UOP_RESHAPE, ins1, 1, NULL);
    cml_ir_add_uop(g, UOP_MUL, ins2, 2, NULL);

    CMLSchedule* s = cml_schedule_create(g, NULL);
    int ok = (s != NULL && cml_schedule_num_kernels(s) == 1 && s->total_ops == 3);
    cml_schedule_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_schedule_custom_op(void) {
    /* An unknown/custom op (like UOP_GATHER) should produce a CUSTOM item */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* idx = make_tensor(4, 1);
    Tensor* ins[2] = { a, idx };
    cml_ir_add_uop(g, UOP_GATHER, ins, 2, NULL);

    CMLSchedule* s = cml_schedule_create(g, NULL);
    int ok = (s != NULL && cml_schedule_num_kernels(s) == 1);
    if (ok) {
        const CMLScheduleItem* it = cml_schedule_get_item(s, 0);
        ok = (it && it->type == SCHED_CUSTOM);
    }
    cml_schedule_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(idx);
    return ok;
}

static int test_schedule_conv_item(void) {
    /* Conv2D gets its own SCHED_CONV item */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* input  = make_tensor_4d(1, 3, 8, 8);
    Tensor* weight = make_tensor_4d(16, 3, 3, 3);
    Tensor* ins[2] = { input, weight };
    cml_ir_add_uop(g, UOP_CONV2D, ins, 2, NULL);

    CMLSchedule* s = cml_schedule_create(g, NULL);
    int ok = (s != NULL && cml_schedule_num_kernels(s) == 1);
    if (ok) {
        const CMLScheduleItem* it = cml_schedule_get_item(s, 0);
        ok = (it && it->type == SCHED_CONV);
    }
    cml_schedule_free(s);
    cml_ir_free(g);
    tensor_free(input);
    tensor_free(weight);
    return ok;
}

static int test_schedule_standalone_reduce(void) {
    /* A lone reduction with no preceding elementwise => 1 REDUCE item */
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* ins[1] = { a };
    cml_ir_add_uop(g, UOP_SUM, ins, 1, NULL);

    CMLSchedule* s = cml_schedule_create(g, NULL);
    int ok = (s != NULL && cml_schedule_num_kernels(s) == 1);
    if (ok) {
        const CMLScheduleItem* it = cml_schedule_get_item(s, 0);
        ok = (it && it->type == SCHED_REDUCE && it->num_ops == 1);
    }
    cml_schedule_free(s);
    cml_ir_free(g);
    tensor_free(a);
    return ok;
}

/* ---------- main ---------- */

int main(void) {
    printf("\n=== test_schedule ===\n\n");

    RUN_TEST(test_is_elementwise);
    RUN_TEST(test_is_reduction);
    RUN_TEST(test_is_movement);
    RUN_TEST(test_can_fuse_elementwise);
    RUN_TEST(test_cannot_fuse_after_reduce);
    RUN_TEST(test_can_fuse_before_reduce);
    RUN_TEST(test_can_fuse_matmul_elementwise);
    RUN_TEST(test_movement_fuses_with_anything);
    RUN_TEST(test_default_options);
    RUN_TEST(test_schedule_empty_graph);
    RUN_TEST(test_schedule_null_graph);
    RUN_TEST(test_schedule_single_op);
    RUN_TEST(test_schedule_elementwise_chain);
    RUN_TEST(test_schedule_reduce_breaks_fusion);
    RUN_TEST(test_schedule_matmul_item);
    RUN_TEST(test_schedule_matmul_bias_fused);
    RUN_TEST(test_schedule_fusion_ratio);
    RUN_TEST(test_schedule_no_fusion);
    RUN_TEST(test_schedule_print);
    RUN_TEST(test_schedule_to_string);
    RUN_TEST(test_schedule_free_null);
    RUN_TEST(test_schedule_get_item_bounds);
    RUN_TEST(test_schedule_num_kernels_null);
    RUN_TEST(test_schedule_movement_folded);
    RUN_TEST(test_schedule_custom_op);
    RUN_TEST(test_schedule_conv_item);
    RUN_TEST(test_schedule_standalone_reduce);

    printf("\n  Results: %d/%d passed\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
