#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cml.h"
#include "ops/ir/schedule.h"
#include "ops/ir/multi_output.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
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

/* ── Multi-output tests ── */

static int test_multi_output_analyze_no_merge(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);
    Tensor* c = make_tensor(4, 4);
    Tensor* ins_ab[2] = { a, b };
    Tensor* ins_ac[2] = { a, c };

    cml_ir_add_uop(g, UOP_ADD, ins_ab, 2, NULL);
    cml_ir_add_uop(g, UOP_SUM, (Tensor*[]){ a }, 1, NULL);
    cml_ir_add_uop(g, UOP_MUL, ins_ac, 2, NULL);

    CMLScheduleV2* s = cml_schedule_v2_create(g, NULL);
    int* merge_groups = NULL;
    int num_merges = 0;
    cml_multi_output_analyze(s, &merge_groups, &num_merges);

    int ok = (num_merges == 0);

    free(merge_groups);
    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    return ok;
}

static int test_multi_output_analyze_null(void) {
    int* mg = NULL;
    int nm = 0;
    int rc = cml_multi_output_analyze(NULL, &mg, &nm);
    return (rc == -1 && nm == 0);
}

static int test_multi_output_fuse_null(void) {
    int rc = cml_multi_output_fuse(NULL, NULL, 0);
    return (rc == -1);
}

static int test_multi_output_same_inputs(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);
    Tensor* ins[2] = { a, b };

    cml_ir_add_uop(g, UOP_ADD, ins, 2, NULL);
    cml_ir_add_uop(g, UOP_SUM, (Tensor*[]){ a }, 1, NULL);
    cml_ir_add_uop(g, UOP_MUL, ins, 2, NULL);

    CMLScheduleOptions opts = cml_schedule_default_options();
    opts.allow_reduce_elem_fusion = false;
    CMLScheduleV2* s = cml_schedule_v2_create(g, &opts);

    int groups_before = s->num_groups;

    int* merge_groups = NULL;
    int num_merges = 0;
    int found = cml_multi_output_analyze(s, &merge_groups, &num_merges);

    int ok = 1;
    if (found > 0) {
        cml_multi_output_fuse(s, merge_groups, num_merges);
        ok = (s->num_groups < groups_before);
    }

    free(merge_groups);
    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

/* ── Reduce->elem fusion tests ── */

static int test_reduce_elem_fusion_enabled(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* ins1[1] = { a };

    cml_ir_add_uop(g, UOP_SUM, ins1, 1, NULL);
    struct IRNode* sum_node = cml_ir_get_tail(g);
    sum_node->use_count = 1;

    Tensor* sum_out = sum_node->output;
    Tensor* ins_div[2] = { a, sum_out ? sum_out : a };
    cml_ir_add_uop(g, UOP_DIV, ins_div, 2, NULL);

    CMLScheduleOptions opts = cml_schedule_default_options();
    opts.allow_reduce_elem_fusion = true;
    CMLScheduleV2* s = cml_schedule_v2_create(g, &opts);

    int ok = (s != NULL && s->num_groups == 1);
    if (ok) {
        ok = (s->groups[0]->num_nodes == 2);
    }

    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    return ok;
}

static int test_reduce_elem_fusion_disabled(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* ins1[1] = { a };

    cml_ir_add_uop(g, UOP_SUM, ins1, 1, NULL);
    struct IRNode* sum_node = cml_ir_get_tail(g);
    sum_node->use_count = 1;

    Tensor* sum_out = sum_node->output;
    Tensor* ins_div[2] = { a, sum_out ? sum_out : a };
    cml_ir_add_uop(g, UOP_DIV, ins_div, 2, NULL);

    CMLScheduleOptions opts = cml_schedule_default_options();
    opts.allow_reduce_elem_fusion = false;
    CMLScheduleV2* s = cml_schedule_v2_create(g, &opts);

    int ok = (s != NULL && s->num_groups == 2);

    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    return ok;
}

static int test_reduce_elem_multi_consumer_blocked(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* ins1[1] = { a };

    cml_ir_add_uop(g, UOP_SUM, ins1, 1, NULL);
    struct IRNode* sum_node = cml_ir_get_tail(g);
    sum_node->use_count = 3;

    Tensor* ins2[2] = { a, a };
    cml_ir_add_uop(g, UOP_DIV, ins2, 2, NULL);

    CMLScheduleOptions opts = cml_schedule_default_options();
    opts.allow_reduce_elem_fusion = true;
    CMLScheduleV2* s = cml_schedule_v2_create(g, &opts);

    int ok = (s != NULL && s->num_groups == 2);

    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    return ok;
}

static int test_reduce_elem_softmax_pattern(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* x = make_tensor(4, 8);
    Tensor* ins1[1] = { x };

    cml_ir_add_uop(g, UOP_EXP, ins1, 1, NULL);
    struct IRNode* exp_node = cml_ir_get_tail(g);

    Tensor* exp_out = exp_node->output ? exp_node->output : x;
    Tensor* ins_sum[1] = { exp_out };
    cml_ir_add_uop(g, UOP_SUM, ins_sum, 1, NULL);
    struct IRNode* sum_node = cml_ir_get_tail(g);
    sum_node->use_count = 1;

    Tensor* sum_out = sum_node->output ? sum_node->output : x;
    Tensor* ins_div[2] = { exp_out, sum_out };
    cml_ir_add_uop(g, UOP_DIV, ins_div, 2, NULL);

    CMLScheduleOptions opts = cml_schedule_default_options();
    opts.allow_reduce_elem_fusion = true;
    CMLScheduleV2* s = cml_schedule_v2_create(g, &opts);

    int ok = (s != NULL && s->num_groups == 1 && s->groups[0]->num_nodes == 3);

    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(x);
    return ok;
}

/* ── BFS scheduling tests ── */

static int test_bfs_single_group(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);
    Tensor* ins[2] = { a, b };
    cml_ir_add_uop(g, UOP_ADD, ins, 2, NULL);

    CMLScheduleOptions opts = cml_schedule_default_options();
    opts.schedule_order = CML_SCHEDULE_ORDER_BFS;
    CMLScheduleV2* s = cml_schedule_v2_create(g, &opts);

    int ok = (s != NULL && s->num_groups == 1
              && s->num_ordered == 1
              && s->execution_order[0] == 0);

    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_bfs_linear_chain(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);
    Tensor* ins2[2] = { a, b };
    Tensor* ins1[1] = { a };

    cml_ir_add_uop(g, UOP_ADD, ins2, 2, NULL);
    cml_ir_add_uop(g, UOP_SUM, ins1, 1, NULL);
    cml_ir_add_uop(g, UOP_MUL, ins2, 2, NULL);

    CMLScheduleOptions opts = cml_schedule_default_options();
    opts.schedule_order = CML_SCHEDULE_ORDER_BFS;
    opts.allow_reduce_elem_fusion = false;
    CMLScheduleV2* s = cml_schedule_v2_create(g, &opts);

    int ok = (s != NULL && s->num_ordered == s->num_groups);
    if (ok) {
        for (int i = 0; i < s->num_ordered; i++) {
            if (s->execution_order[i] < 0 || s->execution_order[i] >= s->num_groups) {
                ok = 0;
                break;
            }
        }
    }

    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_bfs_all_groups_visited(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* b = make_tensor(4, 4);
    Tensor* ins2[2] = { a, b };
    Tensor* ins1[1] = { a };

    cml_ir_add_uop(g, UOP_ADD, ins2, 2, NULL);
    cml_ir_add_uop(g, UOP_SUM, ins1, 1, NULL);
    cml_ir_add_uop(g, UOP_MUL, ins2, 2, NULL);
    cml_ir_add_uop(g, UOP_SUM, ins1, 1, NULL);
    cml_ir_add_uop(g, UOP_NEG, ins1, 1, NULL);

    CMLScheduleOptions opts = cml_schedule_default_options();
    opts.schedule_order = CML_SCHEDULE_ORDER_BFS;
    opts.allow_reduce_elem_fusion = false;
    CMLScheduleV2* s = cml_schedule_v2_create(g, &opts);

    int ok = (s != NULL && s->num_ordered == s->num_groups);

    int* visited = calloc((size_t)s->num_groups, sizeof(int));
    if (ok && visited) {
        for (int i = 0; i < s->num_ordered; i++) {
            visited[s->execution_order[i]] = 1;
        }
        for (int i = 0; i < s->num_groups; i++) {
            if (!visited[i]) { ok = 0; break; }
        }
    }
    free(visited);

    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_bfs_topo_default(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make_tensor(4, 4);
    Tensor* ins[1] = { a };

    cml_ir_add_uop(g, UOP_NEG, ins, 1, NULL);
    cml_ir_add_uop(g, UOP_SUM, ins, 1, NULL);
    cml_ir_add_uop(g, UOP_EXP, ins, 1, NULL);

    CMLScheduleOptions opts = cml_schedule_default_options();
    opts.allow_reduce_elem_fusion = false;
    CMLScheduleV2* s = cml_schedule_v2_create(g, &opts);

    int ok = (s != NULL && s->num_ordered == s->num_groups);
    if (ok) {
        for (int i = 0; i < s->num_ordered; i++) {
            if (s->execution_order[i] != i) { ok = 0; break; }
        }
    }

    cml_schedule_v2_free(s);
    cml_ir_free(g);
    tensor_free(a);
    return ok;
}

static int test_bfs_empty_graph(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);

    CMLScheduleOptions opts = cml_schedule_default_options();
    opts.schedule_order = CML_SCHEDULE_ORDER_BFS;
    CMLScheduleV2* s = cml_schedule_v2_create(g, &opts);

    int ok = (s != NULL && s->num_groups == 0 && s->num_ordered == 0);

    cml_schedule_v2_free(s);
    cml_ir_free(g);
    return ok;
}

static int test_schedule_options_defaults(void) {
    CMLScheduleOptions opts = cml_schedule_default_options();
    int ok = (opts.allow_reduce_elem_fusion == true
              && opts.schedule_order == CML_SCHEDULE_ORDER_TOPO);
    return ok;
}

int main(void) {
    printf("\ntest_compiler_features\n\n");

    printf("  -- Multi-output kernel codegen --\n");
    RUN_TEST(test_multi_output_analyze_null);
    RUN_TEST(test_multi_output_fuse_null);
    RUN_TEST(test_multi_output_analyze_no_merge);
    RUN_TEST(test_multi_output_same_inputs);

    printf("  -- Reduce->elem fusion --\n");
    RUN_TEST(test_reduce_elem_fusion_enabled);
    RUN_TEST(test_reduce_elem_fusion_disabled);
    RUN_TEST(test_reduce_elem_multi_consumer_blocked);
    RUN_TEST(test_reduce_elem_softmax_pattern);

    printf("  -- BFS kernel scheduling --\n");
    RUN_TEST(test_bfs_single_group);
    RUN_TEST(test_bfs_linear_chain);
    RUN_TEST(test_bfs_all_groups_visited);
    RUN_TEST(test_bfs_topo_default);
    RUN_TEST(test_bfs_empty_graph);
    RUN_TEST(test_schedule_options_defaults);

    printf("\n  Results: %d/%d passed\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
