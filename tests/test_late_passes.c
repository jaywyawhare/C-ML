#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "ops/ir/late_passes.h"
#include "ops/ir/linearize.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-55s ", #name); \
    fflush(stdout); \
    if (test_##name()) { tests_passed++; printf("[PASS]\n"); } \
    else { printf("[FAIL]\n"); } \
} while(0)

static LinearProgram* make_vec_prog(int vec_width) {
    LinearProgram* prog = linear_program_create();
    if (!prog) return NULL;

    int r0 = alloc_vreg(prog);
    LinearOp load = {0};
    load.kind = LINOP_LOAD;
    load.dest_reg = r0;
    load.vec_width = vec_width;
    linear_program_emit(prog, load);

    int r1 = alloc_vreg(prog);
    LinearOp compute = {0};
    compute.kind = LINOP_COMPUTE;
    compute.dest_reg = r1;
    compute.src_regs[0] = r0;
    compute.num_srcs = 1;
    compute.uop = UOP_NEG;
    compute.vec_width = vec_width;
    linear_program_emit(prog, compute);

    LinearOp store = {0};
    store.kind = LINOP_STORE;
    store.dest_reg = r1;
    store.vec_width = vec_width;
    linear_program_emit(prog, store);

    return prog;
}

static LinearProgram* make_loop_prog(int extent) {
    LinearProgram* prog = linear_program_create();
    if (!prog) return NULL;

    LinearOp loop = {0};
    loop.kind = LINOP_LOOP;
    loop.loop_axis = 0;
    loop.loop_extent = extent;
    loop.loop_stride = 1;
    linear_program_emit(prog, loop);

    int r0 = alloc_vreg(prog);
    LinearOp load = {0};
    load.kind = LINOP_LOAD;
    load.dest_reg = r0;
    linear_program_emit(prog, load);

    int r1 = alloc_vreg(prog);
    LinearOp compute = {0};
    compute.kind = LINOP_COMPUTE;
    compute.dest_reg = r1;
    compute.src_regs[0] = r0;
    compute.num_srcs = 1;
    compute.uop = UOP_NEG;
    linear_program_emit(prog, compute);

    LinearOp store = {0};
    store.kind = LINOP_STORE;
    store.dest_reg = r1;
    linear_program_emit(prog, store);

    LinearOp endloop = {0};
    endloop.kind = LINOP_ENDLOOP;
    endloop.loop_axis = 0;
    linear_program_emit(prog, endloop);

    return prog;
}

static int test_devec_null(void) {
    return cml_devectorize(NULL) == -1;
}

static int test_devec_scalar_noop(void) {
    LinearProgram* prog = make_vec_prog(1);
    if (!prog) return 0;
    int before = prog->num_ops;
    int rc = cml_devectorize(prog);
    int ok = (rc == 0 && prog->num_ops == before);
    linear_program_free(prog);
    return ok;
}

static int test_devec_vec4(void) {
    LinearProgram* prog = make_vec_prog(4);
    if (!prog) return 0;
    int rc = cml_devectorize(prog);
    if (rc != 0) { linear_program_free(prog); return 0; }

    int loads = 0, computes = 0, stores = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].kind == LINOP_LOAD) loads++;
        if (prog->ops[i].kind == LINOP_COMPUTE) computes++;
        if (prog->ops[i].kind == LINOP_STORE) stores++;
    }

    int ok = (loads == 4 && computes == 4 && stores == 4);
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].vec_width > 1) ok = 0;
    }
    linear_program_free(prog);
    return ok;
}

static int test_devec_vec2(void) {
    LinearProgram* prog = make_vec_prog(2);
    if (!prog) return 0;
    int rc = cml_devectorize(prog);
    if (rc != 0) { linear_program_free(prog); return 0; }

    int loads = 0, computes = 0, stores = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].kind == LINOP_LOAD) loads++;
        if (prog->ops[i].kind == LINOP_COMPUTE) computes++;
        if (prog->ops[i].kind == LINOP_STORE) stores++;
    }

    int ok = (loads == 2 && computes == 2 && stores == 2);
    linear_program_free(prog);
    return ok;
}

static int test_expand_null(void) {
    return cml_expand_groups(NULL) == -1;
}

static int test_expand_small_loop(void) {
    LinearProgram* prog = make_loop_prog(64);
    if (!prog) return 0;
    int before = prog->num_ops;
    int rc = cml_expand_groups(prog);
    int ok = (rc == 0 && prog->num_ops == before);
    linear_program_free(prog);
    return ok;
}

static int test_expand_large_loop(void) {
    LinearProgram* prog = make_loop_prog(1024);
    if (!prog) return 0;
    int rc = cml_expand_groups(prog);
    if (rc != 0) { linear_program_free(prog); return 0; }

    int loop_count = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].kind == LINOP_LOOP) loop_count++;
    }

    int ok = (loop_count >= 4);

    int total_extent = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].kind == LINOP_LOOP)
            total_extent += prog->ops[i].loop_extent;
    }
    if (total_extent != 1024) ok = 0;

    linear_program_free(prog);
    return ok;
}

static int test_expand_exact_boundary(void) {
    LinearProgram* prog = make_loop_prog(512);
    if (!prog) return 0;
    int rc = cml_expand_groups(prog);
    if (rc != 0) { linear_program_free(prog); return 0; }

    int loop_count = 0;
    int total_extent = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].kind == LINOP_LOOP) {
            loop_count++;
            total_extent += prog->ops[i].loop_extent;
        }
    }

    int ok = (loop_count == 2 && total_extent == 512);
    linear_program_free(prog);
    return ok;
}

static int test_late_lower_null(void) {
    return cml_late_lower(NULL) == -1;
}

static int test_late_lower_combined(void) {
    LinearProgram* prog = linear_program_create();
    if (!prog) return 0;

    LinearOp loop = {0};
    loop.kind = LINOP_LOOP;
    loop.loop_axis = 0;
    loop.loop_extent = 512;
    loop.loop_stride = 1;
    linear_program_emit(prog, loop);

    int r0 = alloc_vreg(prog);
    LinearOp load = {0};
    load.kind = LINOP_LOAD;
    load.dest_reg = r0;
    load.vec_width = 2;
    linear_program_emit(prog, load);

    int r1 = alloc_vreg(prog);
    LinearOp compute = {0};
    compute.kind = LINOP_COMPUTE;
    compute.dest_reg = r1;
    compute.src_regs[0] = r0;
    compute.num_srcs = 1;
    compute.uop = UOP_EXP;
    compute.vec_width = 2;
    linear_program_emit(prog, compute);

    LinearOp store = {0};
    store.kind = LINOP_STORE;
    store.dest_reg = r1;
    store.vec_width = 2;
    linear_program_emit(prog, store);

    LinearOp endloop = {0};
    endloop.kind = LINOP_ENDLOOP;
    endloop.loop_axis = 0;
    linear_program_emit(prog, endloop);

    int rc = cml_late_lower(prog);
    if (rc != 0) { linear_program_free(prog); return 0; }

    int has_vec = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].vec_width > 1) has_vec = 1;
    }

    int loop_count = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].kind == LINOP_LOOP) loop_count++;
    }

    int ok = (rc == 0 && !has_vec && loop_count >= 2);
    linear_program_free(prog);
    return ok;
}

int main(void) {
    printf("test_late_passes\n");

    TEST(devec_null);
    TEST(devec_scalar_noop);
    TEST(devec_vec4);
    TEST(devec_vec2);
    TEST(expand_null);
    TEST(expand_small_loop);
    TEST(expand_large_loop);
    TEST(expand_exact_boundary);
    TEST(late_lower_null);
    TEST(late_lower_combined);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
