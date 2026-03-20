#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cml.h"
#include "ops/ir/rangeify.h"
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

static int test_range_program_create(void) {
    RangeProgram* prog = range_program_create();
    if (!prog) return 0;
    int ok = (prog->num_nodes == 0 && prog->head == NULL && prog->tail == NULL);
    range_program_free(prog);
    return ok;
}

static int test_add_range(void) {
    RangeProgram* prog = range_program_create();
    if (!prog) return 0;

    int r0 = range_program_add_range(prog, 0, 4, 1, 0);
    int r1 = range_program_add_range(prog, 0, 8, 1, 1);
    int ok = (r0 == 0 && r1 == 1 && prog->num_nodes == 2);

    RangeNode* n = prog->head;
    ok = ok && (n->type == UOP_RANGE && n->range.start == 0 && n->range.end == 4);
    n = n->next;
    ok = ok && (n->type == UOP_RANGE && n->range.start == 0 && n->range.end == 8);

    range_program_free(prog);
    return ok;
}

static int test_add_index(void) {
    RangeProgram* prog = range_program_create();
    if (!prog) return 0;

    int r0 = range_program_add_range(prog, 0, 3, 1, 0);
    int r1 = range_program_add_range(prog, 0, 5, 1, 1);

    int range_ids[2] = {r0, r1};
    size_t strides[2] = {5, 1};
    int idx = range_program_add_index(prog, range_ids, strides, 2);

    int ok = (idx == 2 && prog->num_nodes == 3);
    RangeNode* n = prog->tail;
    ok = ok && (n->type == UOP_INDEX && n->index.num_ranges == 2);
    ok = ok && (n->index.strides[0] == 5 && n->index.strides[1] == 1);

    range_program_free(prog);
    return ok;
}

static int test_invalid_range(void) {
    RangeProgram* prog = range_program_create();
    if (!prog) return 0;

    int r = range_program_add_range(prog, 5, 3, 1, 0);
    int ok = (r == -1);

    r = range_program_add_range(prog, 0, 5, 0, 0);
    ok = ok && (r == -1);

    r = range_program_add_range(NULL, 0, 5, 1, 0);
    ok = ok && (r == -1);

    range_program_free(prog);
    return ok;
}

static int test_rangeify_graph(void) {
    cml_init();
    CMLGraph_t graph = cml_ir_new(IR_TARGET_C);
    if (!graph) { cml_cleanup(); return 0; }

    int shape_a[] = {2, 3};
    int shape_b[] = {2, 3};
    TensorConfig cfg = {0};
    Tensor* a = tensor_ones(shape_a, 2, &cfg);
    Tensor* b = tensor_ones(shape_b, 2, &cfg);
    if (!a || !b) { cml_ir_free(graph); cml_cleanup(); return 0; }

    Tensor* inputs[] = {a, b};
    cml_ir_add_uop(graph, UOP_ADD, inputs, 2, NULL);

    int converted = cml_rangeify(graph);
    int ok = (converted >= 0);

    cml_ir_free(graph);
    cml_cleanup();
    return ok;
}

static int test_rangeify_null(void) {
    return cml_rangeify(NULL) == -1;
}

static int test_range_print(void) {
    RangeProgram* prog = range_program_create();
    if (!prog) return 0;

    range_program_add_range(prog, 0, 10, 2, 0);
    range_program_print(prog);
    range_program_print(NULL);

    range_program_free(prog);
    return 1;
}

static int test_broadcast_index(void) {
    RangeProgram* prog = range_program_create();
    if (!prog) return 0;

    int r0 = range_program_add_range(prog, 0, 4, 1, 0);
    int r1 = range_program_add_range(prog, 0, 8, 1, 1);

    int range_ids[2] = {r0, r1};
    size_t strides_broadcast[2] = {0, 1};
    int idx = range_program_add_index(prog, range_ids, strides_broadcast, 2);
    int ok = (idx >= 0);

    RangeNode* n = prog->tail;
    ok = ok && (n->index.strides[0] == 0 && n->index.strides[1] == 1);

    range_program_free(prog);
    return ok;
}

int main(void) {
    printf("=== Rangeify Tests ===\n");

    RUN_TEST(test_range_program_create);
    RUN_TEST(test_add_range);
    RUN_TEST(test_add_index);
    RUN_TEST(test_invalid_range);
    RUN_TEST(test_rangeify_graph);
    RUN_TEST(test_rangeify_null);
    RUN_TEST(test_range_print);
    RUN_TEST(test_broadcast_index);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
