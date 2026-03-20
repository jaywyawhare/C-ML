#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cml.h"
#include "ops/ir/spec.h"
#include "ops/ir/ir.h"
#include "ops/uops.h"
#include "tensor/tensor.h"

static int tests_run    = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-55s ", #name); \
    fflush(stdout); \
    if (test_##name()) { tests_passed++; printf("[PASS]\n"); } \
    else { printf("[FAIL]\n"); } \
} while(0)

static Tensor* make(int d0, int d1) {
    int shape[2] = { d0, d1 };
    TensorConfig cfg = {0};
    return tensor_empty(shape, 2, &cfg);
}

static int test_null_graph(void) {
    CMLSpecResult* r = cml_spec_validate(NULL, CML_SPEC_TENSOR);
    if (!r) return 0;
    int ok = r->valid;
    cml_spec_result_free(r);
    return ok;
}

static int test_empty_graph(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    CMLSpecResult* r = cml_spec_validate(g, CML_SPEC_TENSOR);
    int ok = r && r->valid && r->num_errors == 0;
    cml_spec_result_free(r);
    cml_ir_free(g);
    return ok;
}

static int test_valid_binary(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make(4, 4);
    Tensor* b = make(4, 4);
    Tensor* ins[2] = { a, b };
    cml_ir_add_uop(g, UOP_ADD, ins, 2, NULL);

    CMLSpecResult* r = cml_spec_validate(g, CML_SPEC_TENSOR);
    int ok = r && r->valid;
    cml_spec_result_free(r);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_binary_wrong_arity(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make(4, 4);
    Tensor* ins[1] = { a };
    cml_ir_add_uop(g, UOP_ADD, ins, 1, NULL);

    CMLSpecResult* r = cml_spec_validate(g, CML_SPEC_TENSOR);
    int ok = r && !r->valid && r->num_errors > 0;
    cml_spec_result_free(r);
    cml_ir_free(g);
    tensor_free(a);
    return ok;
}

static int test_unary_wrong_arity(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make(4, 4);
    Tensor* b = make(4, 4);
    Tensor* ins[2] = { a, b };
    cml_ir_add_uop(g, UOP_NEG, ins, 2, NULL);

    CMLSpecResult* r = cml_spec_validate(g, CML_SPEC_TENSOR);
    int ok = r && !r->valid;
    cml_spec_result_free(r);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_broadcast_incompatible(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make(3, 5);
    Tensor* b = make(4, 5);
    Tensor* ins[2] = { a, b };
    cml_ir_add_uop(g, UOP_ADD, ins, 2, NULL);

    CMLSpecResult* r = cml_spec_validate(g, CML_SPEC_TENSOR);
    int ok = r && !r->valid;
    cml_spec_result_free(r);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_broadcast_valid(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make(3, 1);
    Tensor* b = make(1, 5);
    Tensor* ins[2] = { a, b };
    cml_ir_add_uop(g, UOP_ADD, ins, 2, NULL);

    CMLSpecResult* r = cml_spec_validate(g, CML_SPEC_TENSOR);
    int ok = r && r->valid;
    cml_spec_result_free(r);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_kernel_level(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make(4, 4);
    Tensor* b = make(4, 4);
    Tensor* ins[2] = { a, b };
    cml_ir_add_uop(g, UOP_ADD, ins, 2, NULL);

    CMLSpecResult* r = cml_spec_validate(g, CML_SPEC_KERNEL);
    int ok = r && r->valid;
    cml_spec_result_free(r);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_program_level(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make(4, 4);
    Tensor* b = make(4, 4);
    Tensor* ins[2] = { a, b };
    cml_ir_add_uop(g, UOP_ADD, ins, 2, NULL);

    CMLSpecResult* r = cml_spec_validate(g, CML_SPEC_PROGRAM);
    int ok = (r != NULL);
    if (r) cml_spec_result_free(r);
    cml_ir_free(g);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_result_free_null(void) {
    cml_spec_result_free(NULL);
    return 1;
}

static int test_result_print(void) {
    CMLGraph_t g = cml_ir_new(IR_TARGET_C);
    Tensor* a = make(4, 4);
    Tensor* ins[1] = { a };
    cml_ir_add_uop(g, UOP_ADD, ins, 1, NULL);

    CMLSpecResult* r = cml_spec_validate(g, CML_SPEC_TENSOR);
    cml_spec_result_print(r);
    cml_spec_result_print(NULL);
    int ok = r && !r->valid;
    cml_spec_result_free(r);
    cml_ir_free(g);
    tensor_free(a);
    return ok;
}

int main(void) {
    printf("test_spec\n");

    TEST(null_graph);
    TEST(empty_graph);
    TEST(valid_binary);
    TEST(binary_wrong_arity);
    TEST(unary_wrong_arity);
    TEST(broadcast_incompatible);
    TEST(broadcast_valid);
    TEST(kernel_level);
    TEST(program_level);
    TEST(result_free_null);
    TEST(result_print);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
