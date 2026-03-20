#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "ops/ir/tree_automaton.h"
#include "ops/ir/pattern_matcher.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"

static struct IRNode* make_node(UOpType type, const char* output,
                                const char** inputs, int num_inputs) {
    struct IRNode* node = calloc(1, sizeof(struct IRNode));
    node->type = type;
    node->output_name = strdup(output);
    node->num_inputs = num_inputs;
    if (num_inputs > 0) {
        node->input_names = malloc((size_t)num_inputs * sizeof(char*));
        for (int i = 0; i < num_inputs; i++)
            node->input_names[i] = strdup(inputs[i]);
    }
    return node;
}

static struct IRNode* make_fill_node(const char* output, float value) {
    struct IRNode* node = calloc(1, sizeof(struct IRNode));
    node->type = UOP_FILL;
    node->output_name = strdup(output);
    node->num_inputs = 0;
    node->output_ndim = 1;
    node->output_shape = malloc(sizeof(int));
    node->output_shape[0] = 1;

    FillParams* fp = malloc(sizeof(FillParams));
    fp->value = value;
    fp->ndim = 1;
    fp->shape = malloc(sizeof(int));
    fp->shape[0] = 1;
    node->params = fp;
    return node;
}

static void graph_append(struct CMLGraph* g, struct IRNode* node) {
    node->next = NULL;
    if (!g->head) {
        g->head = node;
        g->tail = node;
    } else {
        g->tail->next = node;
        g->tail = node;
    }
    g->node_count++;
}

static void free_graph_nodes(struct CMLGraph* g) {
    struct IRNode* n = g->head;
    while (n) {
        struct IRNode* next = n->next;
        if (n->input_names) {
            for (int i = 0; i < n->num_inputs; i++)
                free(n->input_names[i]);
            free(n->input_names);
        }
        free(n->output_name);
        free(n->output_shape);
        if (n->params) {
            if (n->type == UOP_FILL) {
                FillParams* fp = (FillParams*)n->params;
                free(fp->shape);
            }
            free(n->params);
        }
        free(n->users);
        free(n);
        n = next;
    }
    g->head = g->tail = NULL;
    g->node_count = 0;
}

static void test_compile_empty(void) {
    printf("  test_compile_empty...");

    CMLRewriteRegistry* reg = cml_rewrite_registry_create();
    CMLAutomaton* aut = cml_automaton_compile(reg);
    assert(aut == NULL);
    cml_rewrite_registry_free(reg);

    printf(" PASS\n");
}

static void test_compile_builtin(void) {
    printf("  test_compile_builtin...");

    CMLRewriteRegistry* reg = cml_rewrite_builtin_rules();
    assert(reg != NULL);
    assert(reg->num_rules > 0);

    CMLAutomaton* aut = cml_automaton_compile(reg);
    assert(aut != NULL);
    assert(cml_automaton_num_states(aut) > 2);
    assert(cml_automaton_num_transitions(aut) > 0);

    printf(" states=%d transitions=%d",
           cml_automaton_num_states(aut),
           cml_automaton_num_transitions(aut));

    cml_automaton_free(aut);
    cml_rewrite_registry_free(reg);

    printf(" PASS\n");
}

static void test_mul_by_one(void) {
    printf("  test_mul_by_one...");

    CMLRewriteRegistry* reg = cml_rewrite_builtin_rules();
    CMLAutomaton* aut = cml_automaton_compile(reg);
    assert(aut != NULL);

    struct CMLGraph graph;
    memset(&graph, 0, sizeof(graph));

    struct IRNode* x = make_fill_node("x", 42.0f);
    struct IRNode* one = make_fill_node("one", 1.0f);
    const char* mul_inputs[] = {"x", "one"};
    struct IRNode* mul = make_node(UOP_MUL, "result", mul_inputs, 2);

    graph_append(&graph, x);
    graph_append(&graph, one);
    graph_append(&graph, mul);

    int rewrites = cml_automaton_rewrite(aut, &graph);
    assert(rewrites > 0);

    free_graph_nodes(&graph);
    cml_automaton_free(aut);
    cml_rewrite_registry_free(reg);

    printf(" PASS\n");
}

static void test_add_zero(void) {
    printf("  test_add_zero...");

    CMLRewriteRegistry* reg = cml_rewrite_builtin_rules();
    CMLAutomaton* aut = cml_automaton_compile(reg);
    assert(aut != NULL);

    struct CMLGraph graph;
    memset(&graph, 0, sizeof(graph));

    struct IRNode* x = make_fill_node("x", 7.0f);
    struct IRNode* zero = make_fill_node("zero", 0.0f);
    const char* add_inputs[] = {"x", "zero"};
    struct IRNode* add = make_node(UOP_ADD, "result", add_inputs, 2);

    graph_append(&graph, x);
    graph_append(&graph, zero);
    graph_append(&graph, add);

    int rewrites = cml_automaton_rewrite(aut, &graph);
    assert(rewrites > 0);

    free_graph_nodes(&graph);
    cml_automaton_free(aut);
    cml_rewrite_registry_free(reg);

    printf(" PASS\n");
}

static void test_constant_fold(void) {
    printf("  test_constant_fold...");

    CMLRewriteRegistry* reg = cml_rewrite_builtin_rules();
    CMLAutomaton* aut = cml_automaton_compile(reg);
    assert(aut != NULL);

    struct CMLGraph graph;
    memset(&graph, 0, sizeof(graph));

    struct IRNode* a = make_fill_node("a", 3.0f);
    struct IRNode* b = make_fill_node("b", 4.0f);
    const char* add_inputs[] = {"a", "b"};
    struct IRNode* add = make_node(UOP_ADD, "result", add_inputs, 2);

    graph_append(&graph, a);
    graph_append(&graph, b);
    graph_append(&graph, add);

    int rewrites = cml_automaton_rewrite(aut, &graph);
    assert(rewrites > 0);

    /* After constant folding, the ADD should be replaced by FILL(7) */
    struct IRNode* n = graph.head;
    bool found_folded = false;
    while (n) {
        if (n->type == UOP_FILL && n->params) {
            FillParams* fp = (FillParams*)n->params;
            float diff = fp->value - 7.0f;
            if (diff < 0) diff = -diff;
            if (diff < 1e-5f)
                found_folded = true;
        }
        n = n->next;
    }
    assert(found_folded);

    free_graph_nodes(&graph);
    cml_automaton_free(aut);
    cml_rewrite_registry_free(reg);

    printf(" PASS\n");
}

static void test_neg_neg(void) {
    printf("  test_neg_neg...");

    CMLRewriteRegistry* reg = cml_rewrite_builtin_rules();
    CMLAutomaton* aut = cml_automaton_compile(reg);
    assert(aut != NULL);

    struct CMLGraph graph;
    memset(&graph, 0, sizeof(graph));

    struct IRNode* x = make_fill_node("x", 5.0f);
    const char* neg1_inputs[] = {"x"};
    struct IRNode* neg1 = make_node(UOP_NEG, "neg1", neg1_inputs, 1);
    const char* neg2_inputs[] = {"neg1"};
    struct IRNode* neg2 = make_node(UOP_NEG, "neg2", neg2_inputs, 1);

    graph_append(&graph, x);
    graph_append(&graph, neg1);
    graph_append(&graph, neg2);

    int rewrites = cml_automaton_rewrite(aut, &graph);
    assert(rewrites > 0);

    free_graph_nodes(&graph);
    cml_automaton_free(aut);
    cml_rewrite_registry_free(reg);

    printf(" PASS\n");
}

static void test_fixpoint_convergence(void) {
    printf("  test_fixpoint_convergence...");

    CMLRewriteRegistry* reg = cml_rewrite_builtin_rules();
    CMLAutomaton* aut = cml_automaton_compile(reg);
    assert(aut != NULL);

    struct CMLGraph graph;
    memset(&graph, 0, sizeof(graph));

    /* neg(neg(x)) + 0 should reduce to x in multiple passes */
    struct IRNode* x = make_fill_node("x", 9.0f);
    const char* neg1_inputs[] = {"x"};
    struct IRNode* neg1 = make_node(UOP_NEG, "neg1", neg1_inputs, 1);
    const char* neg2_inputs[] = {"neg1"};
    struct IRNode* neg2 = make_node(UOP_NEG, "neg2", neg2_inputs, 1);
    struct IRNode* zero = make_fill_node("zero", 0.0f);
    const char* add_inputs[] = {"neg2", "zero"};
    struct IRNode* add = make_node(UOP_ADD, "result", add_inputs, 2);

    graph_append(&graph, x);
    graph_append(&graph, neg1);
    graph_append(&graph, neg2);
    graph_append(&graph, zero);
    graph_append(&graph, add);

    int rewrites = cml_automaton_rewrite(aut, &graph);
    assert(rewrites >= 2);

    free_graph_nodes(&graph);
    cml_automaton_free(aut);
    cml_rewrite_registry_free(reg);

    printf(" PASS\n");
}

static void test_stats(void) {
    printf("  test_stats...");

    assert(cml_automaton_num_states(NULL) == 0);
    assert(cml_automaton_num_transitions(NULL) == 0);

    CMLRewriteRegistry* reg = cml_rewrite_builtin_rules();
    CMLAutomaton* aut = cml_automaton_compile(reg);
    assert(cml_automaton_num_states(aut) >= 2);
    assert(cml_automaton_num_transitions(aut) >= 1);

    cml_automaton_free(aut);
    cml_rewrite_registry_free(reg);

    printf(" PASS\n");
}

static void bench_automaton_vs_linear(void) {
    printf("  bench_automaton_vs_linear...");

    CMLRewriteRegistry* reg = cml_rewrite_builtin_rules();
    CMLAutomaton* aut = cml_automaton_compile(reg);
    assert(aut != NULL);

    /* Build a graph with many nodes that won't match (worst case for linear) */
    int graph_size = 500;

    /* Benchmark automaton */
    {
        struct CMLGraph graph;
        memset(&graph, 0, sizeof(graph));

        for (int i = 0; i < graph_size; i++) {
            char name[32];
            snprintf(name, sizeof(name), "n%d", i);
            struct IRNode* node = make_fill_node(name, (float)i);
            graph_append(&graph, node);
        }

        /* Add some matchable patterns */
        struct IRNode* one = make_fill_node("bench_one", 1.0f);
        graph_append(&graph, one);
        const char* mul_inputs[] = {"n0", "bench_one"};
        struct IRNode* mul = make_node(UOP_MUL, "bench_mul", mul_inputs, 2);
        graph_append(&graph, mul);

        clock_t start = clock();
        int rewrites = cml_automaton_rewrite(aut, &graph);
        clock_t end = clock();
        double automaton_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
        printf(" automaton=%d rewrites %.2fms", rewrites, automaton_ms);

        free_graph_nodes(&graph);
    }

    /* Benchmark linear */
    {
        struct CMLGraph graph;
        memset(&graph, 0, sizeof(graph));

        for (int i = 0; i < graph_size; i++) {
            char name[32];
            snprintf(name, sizeof(name), "m%d", i);
            struct IRNode* node = make_fill_node(name, (float)i);
            graph_append(&graph, node);
        }

        struct IRNode* one = make_fill_node("lin_one", 1.0f);
        graph_append(&graph, one);
        const char* mul_inputs[] = {"m0", "lin_one"};
        struct IRNode* mul = make_node(UOP_MUL, "lin_mul", mul_inputs, 2);
        graph_append(&graph, mul);

        clock_t start = clock();
        int rewrites = cml_rewrite_apply(reg, &graph, 0);
        clock_t end = clock();
        double linear_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
        printf(" linear=%d rewrites %.2fms", rewrites, linear_ms);

        free_graph_nodes(&graph);
    }

    cml_automaton_free(aut);
    cml_rewrite_registry_free(reg);

    printf(" PASS\n");
}

int main(void) {
    printf("test_tree_automaton:\n");

    test_compile_empty();
    test_compile_builtin();
    test_mul_by_one();
    test_add_zero();
    test_constant_fold();
    test_neg_neg();
    test_fixpoint_convergence();
    test_stats();
    bench_automaton_vs_linear();

    printf("All tree automaton tests passed.\n");
    return 0;
}
