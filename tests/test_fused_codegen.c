/**
 * @file test_fused_codegen.c
 * @brief Tests for fused kernel code generation
 */

#include "ops/ir/fused_codegen.h"
#include "ops/ir/schedule.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

/* Helper: create a minimal LinearProgram manually */
static CMLLinearProgram* make_test_program(void) {
    CMLLinearProgram* prog = calloc(1, sizeof(CMLLinearProgram));
    if (!prog) return NULL;
    prog->capacity = 16;
    prog->ops = calloc(16, sizeof(CMLLinearOp));
    if (!prog->ops) { free(prog); return NULL; }

    /* LOAD v0 from buf0 */
    prog->ops[0] = (CMLLinearOp){.kind = LINOP_LOAD, .dest_reg = 0};
    prog->num_ops++;

    /* LOAD v1 from buf1 */
    prog->ops[1] = (CMLLinearOp){.kind = LINOP_LOAD, .dest_reg = 1};
    prog->num_ops++;

    /* COMPUTE v2 = v0 + v1 */
    prog->ops[2] = (CMLLinearOp){
        .kind = LINOP_COMPUTE, .uop = UOP_ADD, .dest_reg = 2,
        .src_regs = {0, 1}, .num_srcs = 2, .is_eliminated = true
    };
    prog->num_ops++;

    /* COMPUTE v3 = exp(v2) */
    prog->ops[3] = (CMLLinearOp){
        .kind = LINOP_COMPUTE, .uop = UOP_EXP, .dest_reg = 3,
        .src_regs = {2}, .num_srcs = 1, .is_eliminated = false
    };
    prog->num_ops++;

    /* STORE v3 to buf2 */
    prog->ops[4] = (CMLLinearOp){.kind = LINOP_STORE, .dest_reg = 3};
    prog->num_ops++;

    prog->next_vreg = 4;
    return prog;
}

static void test_c_codegen(void) {
    printf("test_c_codegen...\n");

    CMLLinearProgram* prog = make_test_program();
    ASSERT(prog != NULL, "create program");

    CMLFusedKernel* kernel = cml_fused_codegen(prog, CML_FUSED_BACKEND_C, 1024);
    ASSERT(kernel != NULL, "codegen");
    ASSERT(kernel->source != NULL, "source generated");
    ASSERT(kernel->num_inputs == 2, "2 inputs");
    ASSERT(kernel->num_outputs == 1, "1 output");
    ASSERT(kernel->num_vregs == 4, "4 vregs");

    /* Check source contains expected patterns */
    ASSERT(strstr(kernel->source, "fused_kernel") != NULL, "kernel name");
    ASSERT(strstr(kernel->source, "expf") != NULL, "exp function");
    ASSERT(strstr(kernel->source, "+") != NULL, "add op");

    printf("  Generated C source:\n%s\n", kernel->source);

    cml_fused_kernel_free(kernel);
    cml_linear_program_free(prog);
    tests_passed++;
    printf("  PASS\n");
}

static void test_ptx_codegen(void) {
    printf("test_ptx_codegen...\n");

    CMLLinearProgram* prog = make_test_program();
    ASSERT(prog != NULL, "create program");

    CMLFusedKernel* kernel = cml_fused_codegen(prog, CML_FUSED_BACKEND_PTX, 1024);
    ASSERT(kernel != NULL, "codegen");
    ASSERT(kernel->source != NULL, "PTX source");

    /* Check PTX patterns */
    ASSERT(strstr(kernel->source, ".version") != NULL, "PTX version");
    ASSERT(strstr(kernel->source, ".entry fused_kernel") != NULL, "entry point");
    ASSERT(strstr(kernel->source, "add.f32") != NULL, "PTX add");
    ASSERT(strstr(kernel->source, "ld.global.f32") != NULL, "PTX load");
    ASSERT(strstr(kernel->source, "st.global.f32") != NULL, "PTX store");

    printf("  Generated PTX (%zu bytes)\n", strlen(kernel->source));

    cml_fused_kernel_free(kernel);
    cml_linear_program_free(prog);
    tests_passed++;
    printf("  PASS\n");
}

static void test_spirv_codegen(void) {
    printf("test_spirv_codegen...\n");

    CMLLinearProgram* prog = make_test_program();
    ASSERT(prog != NULL, "create program");

    CMLFusedKernel* kernel = cml_fused_codegen(prog, CML_FUSED_BACKEND_SPIRV, 1024);
    ASSERT(kernel != NULL, "codegen");
    ASSERT(kernel->spirv_words != NULL, "SPIR-V words");
    ASSERT(kernel->spirv_num_words > 0, "non-empty");
    ASSERT(kernel->spirv_words[0] == 0x07230203, "SPIR-V magic");

    printf("  Generated SPIR-V: %d words\n", kernel->spirv_num_words);

    cml_fused_kernel_free(kernel);
    cml_linear_program_free(prog);
    tests_passed++;
    printf("  PASS\n");
}

static void test_linear_program_print(void) {
    printf("test_linear_program_print...\n");

    CMLLinearProgram* prog = make_test_program();
    ASSERT(prog != NULL, "create program");

    cml_linear_program_print(prog);
    cml_linear_program_free(prog);
    tests_passed++;
    printf("  PASS\n");
}

static void test_mul_chain_codegen(void) {
    printf("test_mul_chain_codegen...\n");

    CMLLinearProgram* prog = calloc(1, sizeof(CMLLinearProgram));
    ASSERT(prog != NULL, "alloc");
    prog->capacity = 16;
    prog->ops = calloc(16, sizeof(CMLLinearOp));
    ASSERT(prog->ops != NULL, "alloc ops");

    /* LOAD v0 */
    prog->ops[prog->num_ops++] = (CMLLinearOp){.kind = LINOP_LOAD, .dest_reg = 0};
    /* LOAD v1 */
    prog->ops[prog->num_ops++] = (CMLLinearOp){.kind = LINOP_LOAD, .dest_reg = 1};
    /* v2 = v0 * v1 (eliminated) */
    prog->ops[prog->num_ops++] = (CMLLinearOp){
        .kind = LINOP_COMPUTE, .uop = UOP_MUL, .dest_reg = 2,
        .src_regs = {0, 1}, .num_srcs = 2, .is_eliminated = true
    };
    /* v3 = v2 * v0 */
    prog->ops[prog->num_ops++] = (CMLLinearOp){
        .kind = LINOP_COMPUTE, .uop = UOP_MUL, .dest_reg = 3,
        .src_regs = {2, 0}, .num_srcs = 2, .is_eliminated = false
    };
    /* STORE v3 */
    prog->ops[prog->num_ops++] = (CMLLinearOp){.kind = LINOP_STORE, .dest_reg = 3};
    prog->next_vreg = 4;

    CMLFusedKernel* kernel = cml_fused_codegen(prog, CML_FUSED_BACKEND_C, 256);
    ASSERT(kernel != NULL, "codegen");
    ASSERT(kernel->source != NULL, "source");

    /* Should have two muls chained */
    char* first_mul = strstr(kernel->source, "*");
    ASSERT(first_mul != NULL, "first mul");
    char* second_mul = strstr(first_mul + 1, "*");
    ASSERT(second_mul != NULL, "second mul");

    printf("  Chain codegen OK\n");
    cml_fused_kernel_free(kernel);
    cml_linear_program_free(prog);
    tests_passed++;
    printf("  PASS\n");
}

int main(void) {
    printf("=== Fused Codegen Tests ===\n\n");

    test_c_codegen();
    test_ptx_codegen();
    test_spirv_codegen();
    test_linear_program_print();
    test_mul_chain_codegen();

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
