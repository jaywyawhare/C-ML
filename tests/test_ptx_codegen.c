/**
 * @file test_ptx_codegen.c
 * @brief Tests for standalone PTX code generation (string-based, no GPU needed)
 */

#include "ops/ir/gpu/ptx_codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    do { printf("  TEST: %-50s ", #name); } while(0)

#define PASS() \
    do { printf("[PASS]\n"); tests_passed++; } while(0)

#define FAIL(msg) \
    do { printf("[FAIL] %s\n", msg); tests_failed++; } while(0)

#define ASSERT_NOT_NULL(ptr) \
    do { if ((ptr) == NULL) { FAIL(#ptr " is NULL"); return; } } while(0)

#define ASSERT_CONTAINS(haystack, needle) \
    do { if (strstr((haystack), (needle)) == NULL) { \
        char _msg[512]; \
        snprintf(_msg, sizeof(_msg), "Missing '%s' in PTX output", (needle)); \
        FAIL(_msg); free(ptx); return; \
    }} while(0)

// ============================================================================
// Common checks
// ============================================================================

static void assert_ptx_common(const char* ptx, const char* kernel_name) {
    // Every PTX kernel must have these
    if (!strstr(ptx, ".version")) { FAIL("Missing .version"); return; }
    if (!strstr(ptx, ".target")) { FAIL("Missing .target"); return; }
    if (!strstr(ptx, ".entry")) { FAIL("Missing .entry"); return; }
    if (!strstr(ptx, "%tid.x")) { FAIL("Missing %tid.x"); return; }
    if (!strstr(ptx, kernel_name)) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Missing kernel name '%s'", kernel_name);
        FAIL(msg);
        return;
    }
}

// ============================================================================
// Tests
// ============================================================================

static void test_create_destroy(void) {
    TEST(create_destroy);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(75, NULL);
    ASSERT_NOT_NULL(cg);
    if (cg->sm_version != 75) { FAIL("Wrong sm_version"); cml_ptx_codegen_destroy(cg); return; }
    if (!cg->initialized) { FAIL("Not initialized"); cml_ptx_codegen_destroy(cg); return; }
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_unary_neg(void) {
    TEST(unary_neg);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_unary(cg, UOP_NEG, "kernel_neg");
    ASSERT_NOT_NULL(ptx);
    assert_ptx_common(ptx, "kernel_neg");
    ASSERT_CONTAINS(ptx, "neg.f32");
    ASSERT_CONTAINS(ptx, "param_in");
    ASSERT_CONTAINS(ptx, "param_out");
    ASSERT_CONTAINS(ptx, "sm_50");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_unary_exp(void) {
    TEST(unary_exp);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_unary(cg, UOP_EXP, "kernel_exp");
    ASSERT_NOT_NULL(ptx);
    assert_ptx_common(ptx, "kernel_exp");
    ASSERT_CONTAINS(ptx, "ex2.approx.f32");
    ASSERT_CONTAINS(ptx, "mul.f32");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_unary_log(void) {
    TEST(unary_log);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_unary(cg, UOP_LOG, "kernel_log");
    ASSERT_NOT_NULL(ptx);
    assert_ptx_common(ptx, "kernel_log");
    ASSERT_CONTAINS(ptx, "lg2.approx.f32");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_unary_sqrt(void) {
    TEST(unary_sqrt);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_unary(cg, UOP_SQRT, "kernel_sqrt");
    ASSERT_NOT_NULL(ptx);
    ASSERT_CONTAINS(ptx, "sqrt.approx.f32");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_unary_abs(void) {
    TEST(unary_abs);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_unary(cg, UOP_ABS, "kernel_abs");
    ASSERT_NOT_NULL(ptx);
    ASSERT_CONTAINS(ptx, "abs.f32");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_unary_sin(void) {
    TEST(unary_sin);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_unary(cg, UOP_SIN, "kernel_sin");
    ASSERT_NOT_NULL(ptx);
    ASSERT_CONTAINS(ptx, "sin.approx.f32");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_unary_cos(void) {
    TEST(unary_cos);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_unary(cg, UOP_COS, "kernel_cos");
    ASSERT_NOT_NULL(ptx);
    ASSERT_CONTAINS(ptx, "cos.approx.f32");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_unary_sigmoid(void) {
    TEST(unary_sigmoid);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_unary(cg, UOP_SIGMOID, "kernel_sigmoid");
    ASSERT_NOT_NULL(ptx);
    assert_ptx_common(ptx, "kernel_sigmoid");
    ASSERT_CONTAINS(ptx, "neg.f32");
    ASSERT_CONTAINS(ptx, "ex2.approx.f32");
    ASSERT_CONTAINS(ptx, "rcp.approx.f32");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_unary_tanh(void) {
    TEST(unary_tanh);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_unary(cg, UOP_TANH, "kernel_tanh");
    ASSERT_NOT_NULL(ptx);
    ASSERT_CONTAINS(ptx, "ex2.approx.f32");
    ASSERT_CONTAINS(ptx, "rcp.approx.f32");
    ASSERT_CONTAINS(ptx, "sub.f32");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_binary_add(void) {
    TEST(binary_add);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_binary(cg, UOP_ADD, "kernel_add");
    ASSERT_NOT_NULL(ptx);
    assert_ptx_common(ptx, "kernel_add");
    ASSERT_CONTAINS(ptx, "add.f32");
    ASSERT_CONTAINS(ptx, "param_a");
    ASSERT_CONTAINS(ptx, "param_b");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_binary_mul(void) {
    TEST(binary_mul);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_binary(cg, UOP_MUL, "kernel_mul");
    ASSERT_NOT_NULL(ptx);
    ASSERT_CONTAINS(ptx, "mul.f32");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_binary_max(void) {
    TEST(binary_max);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_binary(cg, UOP_MAX, "kernel_max");
    ASSERT_NOT_NULL(ptx);
    ASSERT_CONTAINS(ptx, "max.f32");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_binary_cmplt(void) {
    TEST(binary_cmplt);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_binary(cg, UOP_CMPLT, "kernel_cmplt");
    ASSERT_NOT_NULL(ptx);
    ASSERT_CONTAINS(ptx, "setp.lt.f32");
    ASSERT_CONTAINS(ptx, "selp.f32");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_binary_pow(void) {
    TEST(binary_pow);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_binary(cg, UOP_POW, "kernel_pow");
    ASSERT_NOT_NULL(ptx);
    ASSERT_CONTAINS(ptx, "lg2.approx.f32");
    ASSERT_CONTAINS(ptx, "ex2.approx.f32");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_fill(void) {
    TEST(fill);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_fill(cg, 3.14f, "kernel_fill");
    ASSERT_NOT_NULL(ptx);
    assert_ptx_common(ptx, "kernel_fill");
    ASSERT_CONTAINS(ptx, "mov.f32");
    ASSERT_CONTAINS(ptx, "st.global.f32");
    ASSERT_CONTAINS(ptx, "0f");  // IEEE hex float
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_where(void) {
    TEST(where);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_where(cg, "kernel_where");
    ASSERT_NOT_NULL(ptx);
    assert_ptx_common(ptx, "kernel_where");
    ASSERT_CONTAINS(ptx, "param_cond");
    ASSERT_CONTAINS(ptx, "selp.f32");
    ASSERT_CONTAINS(ptx, "setp.ne.f32");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_reduction_sum(void) {
    TEST(reduction_sum);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_reduction(cg, UOP_SUM, "kernel_sum");
    ASSERT_NOT_NULL(ptx);
    assert_ptx_common(ptx, "kernel_sum");
    ASSERT_CONTAINS(ptx, "atom.global.add.f32");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_reduction_mean(void) {
    TEST(reduction_mean);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_reduction(cg, UOP_MEAN, "kernel_mean");
    ASSERT_NOT_NULL(ptx);
    ASSERT_CONTAINS(ptx, "atom.global.add.f32");
    ASSERT_CONTAINS(ptx, "div.approx.f32");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_matmul(void) {
    TEST(matmul);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_matmul(cg, "kernel_matmul");
    ASSERT_NOT_NULL(ptx);
    assert_ptx_common(ptx, "kernel_matmul");
    ASSERT_CONTAINS(ptx, "fma.rn.f32");
    ASSERT_CONTAINS(ptx, "param_M");
    ASSERT_CONTAINS(ptx, "param_N");
    ASSERT_CONTAINS(ptx, "param_K");
    ASSERT_CONTAINS(ptx, "bra");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_sm_version(void) {
    TEST(sm_version_custom);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(86, NULL);
    char* ptx = cml_ptx_gen_unary(cg, UOP_NEG, "kernel_neg86");
    ASSERT_NOT_NULL(ptx);
    ASSERT_CONTAINS(ptx, "sm_86");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_kernel_count(void) {
    TEST(kernel_count);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    if (cg->kernel_count != 0) { FAIL("Initial count not 0"); cml_ptx_codegen_destroy(cg); return; }
    char* p1 = cml_ptx_gen_unary(cg, UOP_NEG, "k1");
    char* p2 = cml_ptx_gen_binary(cg, UOP_ADD, "k2");
    char* p3 = cml_ptx_gen_fill(cg, 0.0f, "k3");
    if (cg->kernel_count != 3) { FAIL("Expected count 3"); }
    else { PASS(); }
    free(p1); free(p2); free(p3);
    cml_ptx_codegen_destroy(cg);
}

static void test_invalid_unary_op(void) {
    TEST(invalid_unary_op);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_unary(cg, UOP_ADD, "invalid"); // ADD is binary, not unary
    if (ptx != NULL) { FAIL("Should return NULL for invalid op"); free(ptx); }
    else { PASS(); }
    cml_ptx_codegen_destroy(cg);
}

static void test_register_declarations(void) {
    TEST(register_declarations);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_unary(cg, UOP_NEG, "kernel_test");
    ASSERT_NOT_NULL(ptx);
    ASSERT_CONTAINS(ptx, ".reg .pred");
    ASSERT_CONTAINS(ptx, ".reg .b32");
    ASSERT_CONTAINS(ptx, ".reg .b64");
    ASSERT_CONTAINS(ptx, ".reg .f32");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

static void test_bounds_check(void) {
    TEST(bounds_check);
    CMLPTXCodegen* cg = cml_ptx_codegen_create(50, NULL);
    char* ptx = cml_ptx_gen_unary(cg, UOP_NEG, "kernel_bc");
    ASSERT_NOT_NULL(ptx);
    ASSERT_CONTAINS(ptx, "setp.ge.u32");
    ASSERT_CONTAINS(ptx, "ret");
    free(ptx);
    cml_ptx_codegen_destroy(cg);
    PASS();
}

// ============================================================================
// Main
// ============================================================================

int main(void) {
    printf("\n=== PTX Codegen Tests ===\n\n");

    test_create_destroy();

    // Unary ops
    test_unary_neg();
    test_unary_exp();
    test_unary_log();
    test_unary_sqrt();
    test_unary_abs();
    test_unary_sin();
    test_unary_cos();
    test_unary_sigmoid();
    test_unary_tanh();

    // Binary ops
    test_binary_add();
    test_binary_mul();
    test_binary_max();
    test_binary_cmplt();
    test_binary_pow();

    // Special kernels
    test_fill();
    test_where();
    test_reduction_sum();
    test_reduction_mean();
    test_matmul();

    // Meta / edge cases
    test_sm_version();
    test_kernel_count();
    test_invalid_unary_op();
    test_register_declarations();
    test_bounds_check();

    printf("\n=== Results: %d passed, %d failed ===\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
