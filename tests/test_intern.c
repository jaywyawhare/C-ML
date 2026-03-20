#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cml.h"
#include "ops/ir/internal.h"
#include "ops/ir/intern.h"

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

static int test_create_free(void) {
    CMLInternTable* table = cml_intern_table_create();
    if (!table) return 0;
    if (table->capacity < 1) { cml_intern_table_free(table); return 0; }
    if (table->count != 0) { cml_intern_table_free(table); return 0; }
    cml_intern_table_free(table);
    return 1;
}

static int test_free_null(void) {
    cml_intern_table_free(NULL);
    return 1;
}

static int test_hash_deterministic(void) {
    uint64_t h1 = cml_intern_hash_node(0, 0, NULL, 0, NULL, 0);
    uint64_t h2 = cml_intern_hash_node(0, 0, NULL, 0, NULL, 0);
    return h1 == h2;
}

static int test_hash_varies_by_op(void) {
    uint64_t h1 = cml_intern_hash_node(0, 0, NULL, 0, NULL, 0);
    uint64_t h2 = cml_intern_hash_node(1, 0, NULL, 0, NULL, 0);
    return h1 != h2;
}

static int test_hash_varies_by_dtype(void) {
    uint64_t h1 = cml_intern_hash_node(0, 0, NULL, 0, NULL, 0);
    uint64_t h2 = cml_intern_hash_node(0, 1, NULL, 0, NULL, 0);
    return h1 != h2;
}

static int test_insert_lookup(void) {
    CMLInternTable* table = cml_intern_table_create();
    if (!table) return 0;

    struct IRNode node;
    memset(&node, 0, sizeof(node));
    node.type      = UOP_ADD;
    node.hash      = cml_intern_hash_node(UOP_ADD, 0, NULL, 0, NULL, 0);
    node.ref_count = 1;

    if (cml_intern_insert(table, &node) != 0) {
        cml_intern_table_free(table);
        return 0;
    }
    if (table->count != 1) {
        cml_intern_table_free(table);
        return 0;
    }

    struct IRNode* found = cml_intern_lookup(table, node.hash, UOP_ADD, 0,
                                             NULL, 0, NULL, 0);
    cml_intern_table_free(table);
    return found == &node;
}

static int test_lookup_miss(void) {
    CMLInternTable* table = cml_intern_table_create();
    if (!table) return 0;

    uint64_t hash = cml_intern_hash_node(UOP_MUL, 0, NULL, 0, NULL, 0);
    struct IRNode* found = cml_intern_lookup(table, hash, UOP_MUL, 0,
                                             NULL, 0, NULL, 0);
    cml_intern_table_free(table);
    return found == NULL;
}

static int test_remove(void) {
    CMLInternTable* table = cml_intern_table_create();
    if (!table) return 0;

    struct IRNode node;
    memset(&node, 0, sizeof(node));
    node.type      = UOP_SUB;
    node.hash      = cml_intern_hash_node(UOP_SUB, 0, NULL, 0, NULL, 0);
    node.ref_count = 1;

    cml_intern_insert(table, &node);
    cml_intern_remove(table, &node);

    if (table->count != 0) {
        cml_intern_table_free(table);
        return 0;
    }

    struct IRNode* found = cml_intern_lookup(table, node.hash, UOP_SUB, 0,
                                             NULL, 0, NULL, 0);
    cml_intern_table_free(table);
    return found == NULL;
}

static int test_remove_null(void) {
    CMLInternTable* table = cml_intern_table_create();
    cml_intern_remove(table, NULL);
    cml_intern_remove(NULL, NULL);
    cml_intern_table_free(table);
    return 1;
}

static int test_many_inserts_trigger_resize(void) {
    CMLInternTable* table = cml_intern_table_create();
    if (!table) return 0;

    size_t initial_cap = table->capacity;

    struct IRNode nodes[100];
    memset(nodes, 0, sizeof(nodes));

    for (int i = 0; i < 100; i++) {
        nodes[i].type      = (UOpType)(i % UOP_COUNT);
        nodes[i].hash      = cml_intern_hash_node(i, i, NULL, 0, NULL, 0);
        nodes[i].ref_count = 1;
        if (cml_intern_insert(table, &nodes[i]) != 0) {
            cml_intern_table_free(table);
            return 0;
        }
    }

    if (table->count != 100) {
        cml_intern_table_free(table);
        return 0;
    }
    if (table->capacity <= initial_cap) {
        cml_intern_table_free(table);
        return 0;
    }

    for (int i = 0; i < 100; i++) {
        struct IRNode* found = cml_intern_lookup(table, nodes[i].hash,
                                                 nodes[i].type, i,
                                                 NULL, 0, NULL, 0);
        if (found != &nodes[i]) {
            cml_intern_table_free(table);
            return 0;
        }
    }

    cml_intern_table_free(table);
    return 1;
}

static int test_graph_has_intern_table(void) {
    CMLGraph_t graph = cml_ir_new(IR_TARGET_C);
    if (!graph) return 0;

    int ok = (graph->intern_table != NULL);
    cml_ir_free(graph);
    return ok;
}

static int test_node_has_hash(void) {
    CMLGraph_t graph = cml_ir_new(IR_TARGET_C);
    if (!graph) return 0;

    int shape[] = {2, 3};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .has_dtype = true,
        .has_device = true
    };
    Tensor* a = tensor_zeros(shape, 2, &cfg);
    Tensor* b = tensor_zeros(shape, 2, &cfg);
    if (!a || !b) {
        if (a) tensor_free(a);
        if (b) tensor_free(b);
        cml_ir_free(graph);
        return 0;
    }

    Tensor* inputs[] = {a, b};
    cml_ir_add_uop(graph, UOP_ADD, inputs, 2, NULL);

    struct IRNode* tail = cml_ir_get_tail(graph);
    int ok = (tail != NULL && tail->hash != 0 && tail->ref_count == 1);

    tensor_free(a);
    tensor_free(b);
    cml_ir_free(graph);
    return ok;
}

static int test_ref_count_on_graph_free(void) {
    CMLGraph_t graph = cml_ir_new(IR_TARGET_C);
    if (!graph) return 0;

    int shape[] = {4};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .has_dtype = true,
        .has_device = true
    };
    Tensor* a = tensor_zeros(shape, 1, &cfg);
    if (!a) { cml_ir_free(graph); return 0; }

    Tensor* inputs[] = {a};
    cml_ir_add_uop(graph, UOP_NEG, inputs, 1, NULL);

    tensor_free(a);
    cml_ir_free(graph);
    return 1;
}

int main(void) {
    printf("test_intern\n\n");

    TEST(create_free);
    TEST(free_null);
    TEST(hash_deterministic);
    TEST(hash_varies_by_op);
    TEST(hash_varies_by_dtype);
    TEST(insert_lookup);
    TEST(lookup_miss);
    TEST(remove);
    TEST(remove_null);
    TEST(many_inserts_trigger_resize);
    TEST(graph_has_intern_table);
    TEST(node_has_hash);
    TEST(ref_count_on_graph_free);

    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
