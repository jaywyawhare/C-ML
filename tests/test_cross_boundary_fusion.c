#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ops/ir/cross_boundary_fusion.h"
#include "ops/ir/schedule.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/ir/fusion_patterns.h"
#include "tensor/tensor.h"

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(test) do { \
    tests_run++; \
    printf("  [%d] %-50s ", tests_run, #test); \
    if (test()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

static int test_analyze_null(void) {
    CMLCrossBoundaryFusion* out = NULL;
    int count = 0;
    int ret = cml_cross_boundary_analyze(NULL, &out, &count);
    return ret == -1;
}

static int test_analyze_empty_schedule(void) {
    CMLScheduleV2 sched = {0};
    CMLCrossBoundaryFusion* out = NULL;
    int count = 0;
    int ret = cml_cross_boundary_analyze(&sched, &out, &count);
    return ret == 0 && count == 0 && out == NULL;
}

static int test_fuse_null(void) {
    int ret = cml_cross_boundary_fuse(NULL, NULL, 0);
    return ret == -1;
}

static int test_fuse_empty(void) {
    CMLScheduleV2 sched = {0};
    CMLCrossBoundaryFusion f = {0};
    int ret = cml_cross_boundary_fuse(&sched, &f, 0);
    return ret == -1;
}

static int test_free_null_fusions(void) {
    cml_cross_boundary_fusions_free(NULL);
    return 1;
}

static int test_stats_empty(void) {
    CMLCrossBoundaryStats s = cml_cross_boundary_stats(NULL, 0);
    return s.patterns_found == 0 && s.memory_saved == 0;
}

static int test_stats_softmax_ce(void) {
    CMLCrossBoundaryFusion f = {
        .forward_node_idx = 0,
        .backward_node_idx = 1,
        .pattern_type = CBF_SOFTMAX_CE
    };
    CMLCrossBoundaryStats s = cml_cross_boundary_stats(&f, 1);
    return s.patterns_found == 1 && s.memory_saved > 0 && s.flops_saved > 0;
}

static int test_stats_layernorm(void) {
    CMLCrossBoundaryFusion f = {
        .forward_node_idx = 0,
        .backward_node_idx = 1,
        .pattern_type = CBF_LAYERNORM_BWD
    };
    CMLCrossBoundaryStats s = cml_cross_boundary_stats(&f, 1);
    return s.patterns_found == 1 && s.memory_saved > 0;
}

static int test_stats_gelu(void) {
    CMLCrossBoundaryFusion f = {
        .forward_node_idx = 0,
        .backward_node_idx = 1,
        .pattern_type = CBF_GELU_BWD
    };
    CMLCrossBoundaryStats s = cml_cross_boundary_stats(&f, 1);
    return s.patterns_found == 1 && s.memory_saved > 0;
}

static int test_stats_multiple(void) {
    CMLCrossBoundaryFusion fusions[3] = {
        { .pattern_type = CBF_SOFTMAX_CE },
        { .pattern_type = CBF_LAYERNORM_BWD },
        { .pattern_type = CBF_GELU_BWD },
    };
    CMLCrossBoundaryStats s = cml_cross_boundary_stats(fusions, 3);
    return s.patterns_found == 3;
}

static int test_fusion_pattern_registration(void) {
    FusionPatternRegistry* reg = cml_fusion_registry_create();
    if (!reg) return 0;

    int has_softmax = 0, has_layernorm = 0, has_gelu = 0;
    for (int t = 0; t < FUSION_TARGET_COUNT; t++) {
        FusionPattern* p = reg->patterns[t];
        while (p) {
            if (p->kind == FUSION_PATTERN_SOFTMAX_CE_BWD) has_softmax = 1;
            if (p->kind == FUSION_PATTERN_LAYERNORM_BWD) has_layernorm = 1;
            if (p->kind == FUSION_PATTERN_GELU_BWD) has_gelu = 1;
            p = p->next;
        }
    }

    cml_fusion_registry_free(reg);
    return has_softmax && has_layernorm && has_gelu;
}

static int test_analyze_with_schedule(void) {
    /* Build a minimal schedule with nodes that don't match any pattern */
    CMLFusionGroup group = {0};
    struct IRNode node = {0};
    node.type = UOP_ADD;
    struct IRNode* nodes[1] = { &node };
    group.nodes = nodes;
    group.num_nodes = 1;

    CMLFusionGroup* groups[1] = { &group };
    CMLScheduleV2 sched = {
        .groups = groups,
        .num_groups = 1,
    };

    CMLCrossBoundaryFusion* out = NULL;
    int count = 0;
    int ret = cml_cross_boundary_analyze(&sched, &out, &count);

    if (out) free(out);
    return ret == 0 && count == 0;
}

int main(void) {
    printf("Cross-Boundary Fusion Tests\n");

    RUN_TEST(test_analyze_null);
    RUN_TEST(test_analyze_empty_schedule);
    RUN_TEST(test_fuse_null);
    RUN_TEST(test_fuse_empty);
    RUN_TEST(test_free_null_fusions);
    RUN_TEST(test_stats_empty);
    RUN_TEST(test_stats_softmax_ce);
    RUN_TEST(test_stats_layernorm);
    RUN_TEST(test_stats_gelu);
    RUN_TEST(test_stats_multiple);
    RUN_TEST(test_fusion_pattern_registration);
    RUN_TEST(test_analyze_with_schedule);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
