#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ops/ir/tc_opt.h"
#include "ops/ir/pattern_matcher.h"
#include "ops/ir/internal.h"
#include "ops/ir/gpu/wmma.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-50s ", #name); \
    fflush(stdout); \
    if (test_##name()) { \
        tests_passed++; \
        printf("[PASS]\n"); \
    } else { \
        printf("[FAIL]\n"); \
    } \
} while(0)

static struct CMLGraph* make_empty_graph(void) {
    struct CMLGraph* g = calloc(1, sizeof(struct CMLGraph));
    if (!g) return NULL;
    g->target = IR_TARGET_CUDA;
    return g;
}

static struct IRNode* make_matmul_node(const char* name, int m, int n, int k) {
    struct IRNode* node = calloc(1, sizeof(struct IRNode));
    if (!node) return NULL;

    node->type = UOP_MATMUL;
    node->num_inputs = 2;
    node->input_names = malloc(2 * sizeof(char*));
    node->input_names[0] = strdup("input_a");
    node->input_names[1] = strdup("input_b");
    node->output_name = strdup(name);

    node->input_ndims = malloc(2 * sizeof(int));
    node->input_ndims[0] = 2;
    node->input_ndims[1] = 2;

    node->input_shapes = malloc(2 * sizeof(int*));
    node->input_shapes[0] = malloc(2 * sizeof(int));
    node->input_shapes[0][0] = m;
    node->input_shapes[0][1] = k;
    node->input_shapes[1] = malloc(2 * sizeof(int));
    node->input_shapes[1][0] = k;
    node->input_shapes[1][1] = n;

    node->output_ndim = 2;
    node->output_shape = malloc(2 * sizeof(int));
    node->output_shape[0] = m;
    node->output_shape[1] = n;

    return node;
}

static struct IRNode* make_simple_node(const char* name, UOpType type,
                                       int num_inputs, const char** input_names) {
    struct IRNode* node = calloc(1, sizeof(struct IRNode));
    if (!node) return NULL;

    node->type = type;
    node->num_inputs = num_inputs;
    if (num_inputs > 0) {
        node->input_names = malloc((size_t)num_inputs * sizeof(char*));
        for (int i = 0; i < num_inputs; i++)
            node->input_names[i] = strdup(input_names[i]);
    }
    node->output_name = strdup(name);
    return node;
}

static void append_node(struct CMLGraph* g, struct IRNode* node) {
    node->next = NULL;
    if (g->tail) {
        g->tail->next = node;
    } else {
        g->head = node;
    }
    g->tail = node;
    g->node_count++;
}

static void free_graph(struct CMLGraph* g) {
    if (!g) return;
    struct IRNode* n = g->head;
    while (n) {
        struct IRNode* next = n->next;
        if (n->input_names) {
            for (int i = 0; i < n->num_inputs; i++)
                free(n->input_names[i]);
            free(n->input_names);
        }
        if (n->input_shapes) {
            for (int i = 0; i < n->num_inputs; i++)
                free(n->input_shapes[i]);
            free(n->input_shapes);
        }
        free(n->input_ndims);
        free(n->output_name);
        free(n->output_shape);
        free(n->params);
        free(n->users);
        free(n);
        n = next;
    }
    free(g);
}

static int test_config_defaults(void) {
    CMLTCConfig cfg = cml_tc_get_config();
    if (cfg.min_m != CML_TC_DEFAULT_MIN_DIM) return 0;
    if (cfg.min_n != CML_TC_DEFAULT_MIN_DIM) return 0;
    if (cfg.min_k != CML_TC_DEFAULT_MIN_DIM) return 0;
    if (!cfg.allow_padding) return 0;
    if (cfg.prefer_fp16) return 0;
    return 1;
}

static int test_config_set_get(void) {
    CMLTCConfig original = cml_tc_get_config();

    CMLTCConfig cfg = {
        .min_m = 32, .min_n = 32, .min_k = 32,
        .allow_padding = false, .prefer_fp16 = true,
    };
    cml_tc_set_config(&cfg);

    CMLTCConfig got = cml_tc_get_config();
    if (got.min_m != 32 || got.min_n != 32 || got.min_k != 32) {
        cml_tc_set_config(&original);
        return 0;
    }
    if (got.allow_padding || !got.prefer_fp16) {
        cml_tc_set_config(&original);
        return 0;
    }

    cml_tc_set_config(&original);
    return 1;
}

static int test_config_null_safe(void) {
    cml_tc_set_config(NULL);
    return 1;
}

static int test_optimize_null_graph(void) {
    return cml_tc_optimize(NULL) == 0;
}

static int test_optimize_empty_graph(void) {
    struct CMLGraph* g = make_empty_graph();
    if (!g) return 0;
    int ret = cml_tc_optimize(g);
    free(g);
    return ret == 0;
}

static int test_optimize_aligned_matmul(void) {
    if (!cml_tc_available()) {
        printf("(WMMA unavail, skip) ");
        return 1;
    }

    struct CMLGraph* g = make_empty_graph();
    if (!g) return 0;

    const char* dummy_inputs[] = {"dummy_a", "dummy_b"};
    struct IRNode* a = make_simple_node("input_a", UOP_FILL, 0, NULL);
    struct IRNode* b = make_simple_node("input_b", UOP_FILL, 0, NULL);
    struct IRNode* mm = make_matmul_node("mm_out", 256, 256, 256);

    append_node(g, a);
    append_node(g, b);
    append_node(g, mm);

    int count_before = g->node_count;
    int rewrites = cml_tc_optimize(g);

    if (rewrites <= 0) {
        printf("(expected rewrite) ");
        free_graph(g);
        return 0;
    }

    int found_fused = 0;
    struct IRNode* n = g->head;
    while (n) {
        if (n->is_fused && n->fusion_type == FUSION_FMA)
            found_fused++;
        n = n->next;
    }
    if (found_fused == 0) {
        printf("(no fused WMMA node) ");
        free_graph(g);
        return 0;
    }

    (void)count_before;
    (void)dummy_inputs;

    free_graph(g);
    return 1;
}

static int test_optimize_small_matmul_skipped(void) {
    struct CMLGraph* g = make_empty_graph();
    if (!g) return 0;

    struct IRNode* a = make_simple_node("input_a", UOP_FILL, 0, NULL);
    struct IRNode* b = make_simple_node("input_b", UOP_FILL, 0, NULL);
    struct IRNode* mm = make_matmul_node("mm_out", 4, 4, 4);

    append_node(g, a);
    append_node(g, b);
    append_node(g, mm);

    int rewrites = cml_tc_optimize(g);
    if (rewrites != 0) {
        printf("(expected 0 rewrites for small matmul) ");
        free_graph(g);
        return 0;
    }

    free_graph(g);
    return 1;
}

static int test_optimize_non_matmul_untouched(void) {
    struct CMLGraph* g = make_empty_graph();
    if (!g) return 0;

    const char* inputs[] = {"x"};
    struct IRNode* x = make_simple_node("x", UOP_FILL, 0, NULL);
    struct IRNode* e = make_simple_node("exp_out", UOP_EXP, 1, inputs);

    append_node(g, x);
    append_node(g, e);

    int rewrites = cml_tc_optimize(g);
    if (rewrites != 0) {
        printf("(expected 0 rewrites) ");
        free_graph(g);
        return 0;
    }

    free_graph(g);
    return 1;
}

static int test_tc_available_returns_bool(void) {
    bool avail = cml_tc_available();
    (void)avail;
    return 1;
}

static int test_already_fused_skipped(void) {
    if (!cml_tc_available()) {
        printf("(WMMA unavail, skip) ");
        return 1;
    }

    struct CMLGraph* g = make_empty_graph();
    if (!g) return 0;

    struct IRNode* a = make_simple_node("input_a", UOP_FILL, 0, NULL);
    struct IRNode* b = make_simple_node("input_b", UOP_FILL, 0, NULL);
    struct IRNode* mm = make_matmul_node("mm_out", 64, 64, 64);
    mm->is_fused = true;

    append_node(g, a);
    append_node(g, b);
    append_node(g, mm);

    int rewrites = cml_tc_optimize(g);
    if (rewrites != 0) {
        printf("(expected 0 for already-fused) ");
        free_graph(g);
        return 0;
    }

    free_graph(g);
    return 1;
}

static int test_padding_disabled(void) {
    if (!cml_tc_available()) {
        printf("(WMMA unavail, skip) ");
        return 1;
    }

    CMLTCConfig original = cml_tc_get_config();
    CMLTCConfig cfg = original;
    cfg.allow_padding = false;
    cml_tc_set_config(&cfg);

    struct CMLGraph* g = make_empty_graph();
    if (!g) { cml_tc_set_config(&original); return 0; }

    struct IRNode* a = make_simple_node("input_a", UOP_FILL, 0, NULL);
    struct IRNode* b = make_simple_node("input_b", UOP_FILL, 0, NULL);
    struct IRNode* mm = make_matmul_node("mm_out", 20, 20, 20);

    append_node(g, a);
    append_node(g, b);
    append_node(g, mm);

    int rewrites = cml_tc_optimize(g);

    cml_tc_set_config(&original);
    free_graph(g);
    return rewrites == 0;
}

int main(void) {
    printf("test_tc_opt\n\n");

    TEST(config_defaults);
    TEST(config_set_get);
    TEST(config_null_safe);
    TEST(optimize_null_graph);
    TEST(optimize_empty_graph);
    TEST(optimize_aligned_matmul);
    TEST(optimize_small_matmul_skipped);
    TEST(optimize_non_matmul_untouched);
    TEST(tc_available_returns_bool);
    TEST(already_fused_skipped);
    TEST(padding_disabled);

    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
