#include "ops/ir/runtime_compiler.h"
#include "ops/ir/fused_codegen.h"
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

/* Helper: create a minimal LinearProgram */
static CMLLinearProgram* make_test_program(int variant) {
    CMLLinearProgram* prog = calloc(1, sizeof(CMLLinearProgram));
    if (!prog) return NULL;
    prog->capacity = 16;
    prog->ops = calloc(16, sizeof(CMLLinearOp));
    if (!prog->ops) { free(prog); return NULL; }

    /* LOAD v0, LOAD v1 */
    prog->ops[prog->num_ops++] = (CMLLinearOp){.kind = LINOP_LOAD, .dest_reg = 0};
    prog->ops[prog->num_ops++] = (CMLLinearOp){.kind = LINOP_LOAD, .dest_reg = 1};

    /* Variant 0: ADD, Variant 1: MUL */
    UOpType uop = (variant == 0) ? UOP_ADD : UOP_MUL;
    prog->ops[prog->num_ops++] = (CMLLinearOp){
        .kind = LINOP_COMPUTE, .uop = uop, .dest_reg = 2,
        .src_regs = {0, 1}, .num_srcs = 2, .is_eliminated = false
    };

    /* STORE v2 */
    prog->ops[prog->num_ops++] = (CMLLinearOp){.kind = LINOP_STORE, .dest_reg = 2};
    prog->next_vreg = 3;
    return prog;
}

static void test_compiler_create_free(void) {
    printf("test_compiler_create_free...\n");

    CMLRuntimeCompiler* rc = cml_runtime_compiler_create();
    ASSERT(rc != NULL, "create");
    ASSERT(rc->enable_caching == true, "caching enabled by default");
    ASSERT(rc->num_cached == 0, "empty cache");

    cml_runtime_compiler_free(rc);
    tests_passed++;
    printf("  PASS\n");
}

static void test_compile_program(void) {
    printf("test_compile_program...\n");

    CMLRuntimeCompiler* rc = cml_runtime_compiler_create();
    ASSERT(rc != NULL, "create");

    CMLLinearProgram* prog = make_test_program(0);
    ASSERT(prog != NULL, "create program");

    const CMLCompiledKernel* k = cml_runtime_compile_program(rc, prog, 1024);
    ASSERT(k != NULL, "compile");
    ASSERT(k->valid, "valid");
    ASSERT(k->num_inputs == 2, "2 inputs");
    ASSERT(k->num_outputs == 1, "1 output");
    ASSERT(k->source != NULL, "has source");

    size_t hits, misses, compilations;
    cml_runtime_compiler_stats(rc, &hits, &misses, &compilations);
    ASSERT(misses == 1, "1 miss");
    ASSERT(compilations == 1, "1 compilation");

    cml_linear_program_free(prog);
    cml_runtime_compiler_free(rc);
    tests_passed++;
    printf("  PASS\n");
}

static void test_cache_hit(void) {
    printf("test_cache_hit...\n");

    CMLRuntimeCompiler* rc = cml_runtime_compiler_create();
    ASSERT(rc != NULL, "create");

    CMLLinearProgram* prog = make_test_program(0);
    ASSERT(prog != NULL, "create program");

    /* First compilation: miss */
    const CMLCompiledKernel* k1 = cml_runtime_compile_program(rc, prog, 1024);
    ASSERT(k1 != NULL, "first compile");

    /* Second compilation: hit */
    const CMLCompiledKernel* k2 = cml_runtime_compile_program(rc, prog, 1024);
    ASSERT(k2 != NULL, "second compile");
    ASSERT(k1 == k2, "same cached entry");

    size_t hits, misses, compilations;
    cml_runtime_compiler_stats(rc, &hits, &misses, &compilations);
    ASSERT(hits == 1, "1 hit");
    ASSERT(misses == 1, "1 miss");
    ASSERT(compilations == 1, "still 1 compilation");

    cml_linear_program_free(prog);
    cml_runtime_compiler_free(rc);
    tests_passed++;
    printf("  PASS\n");
}

static void test_different_programs_cached_separately(void) {
    printf("test_different_programs_cached_separately...\n");

    CMLRuntimeCompiler* rc = cml_runtime_compiler_create();
    ASSERT(rc != NULL, "create");

    CMLLinearProgram* prog_add = make_test_program(0);
    CMLLinearProgram* prog_mul = make_test_program(1);
    ASSERT(prog_add && prog_mul, "create programs");

    const CMLCompiledKernel* k1 = cml_runtime_compile_program(rc, prog_add, 1024);
    const CMLCompiledKernel* k2 = cml_runtime_compile_program(rc, prog_mul, 1024);
    ASSERT(k1 != NULL && k2 != NULL, "both compile");
    ASSERT(k1 != k2, "different entries");

    size_t hits, misses, compilations;
    cml_runtime_compiler_stats(rc, &hits, &misses, &compilations);
    ASSERT(misses == 2, "2 misses");
    ASSERT(compilations == 2, "2 compilations");

    cml_linear_program_free(prog_add);
    cml_linear_program_free(prog_mul);
    cml_runtime_compiler_free(rc);
    tests_passed++;
    printf("  PASS\n");
}

static void test_clear_cache(void) {
    printf("test_clear_cache...\n");

    CMLRuntimeCompiler* rc = cml_runtime_compiler_create();
    ASSERT(rc != NULL, "create");

    CMLLinearProgram* prog = make_test_program(0);
    cml_runtime_compile_program(rc, prog, 1024);
    ASSERT(rc->num_cached == 1, "1 cached");

    cml_runtime_compiler_clear_cache(rc);
    ASSERT(rc->num_cached == 0, "cache cleared");

    /* Compiling again should miss */
    cml_runtime_compile_program(rc, prog, 1024);

    size_t hits, misses, compilations;
    cml_runtime_compiler_stats(rc, &hits, &misses, &compilations);
    ASSERT(misses == 2, "2 misses after clear");

    cml_linear_program_free(prog);
    cml_runtime_compiler_free(rc);
    tests_passed++;
    printf("  PASS\n");
}

static void test_set_backend(void) {
    printf("test_set_backend...\n");

    CMLRuntimeCompiler* rc = cml_runtime_compiler_create();
    ASSERT(rc != NULL, "create");

    cml_runtime_compiler_set_backend(rc, CML_FUSED_BACKEND_PTX);
    ASSERT(rc->preferred_backend == CML_FUSED_BACKEND_PTX, "PTX set");

    CMLLinearProgram* prog = make_test_program(0);
    const CMLCompiledKernel* k = cml_runtime_compile_program(rc, prog, 1024);
    ASSERT(k != NULL, "compile with PTX");
    ASSERT(k->backend == CML_FUSED_BACKEND_PTX, "PTX backend");
    ASSERT(k->source != NULL, "has PTX source");

    cml_linear_program_free(prog);
    cml_runtime_compiler_free(rc);
    tests_passed++;
    printf("  PASS\n");
}

int main(void) {
    printf("=== Runtime Compiler Tests ===\n\n");

    test_compiler_create_free();
    test_compile_program();
    test_cache_hit();
    test_different_programs_cached_separately();
    test_clear_cache();
    test_set_backend();

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
