#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "ops/ir/opt_transforms.h"
#include "ops/ir/linearize.h"

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

/* ── Helpers ── */

static LinearProgram* make_test_prog(int num_axes, const int* extents) {
    LinearProgram* prog = linear_program_create();
    if (!prog) return NULL;

    for (int i = 0; i < num_axes; i++) {
        if (prog->num_axes >= prog->axes_capacity) {
            int nc = prog->axes_capacity * 2;
            int* tmp = realloc(prog->loop_axes, (size_t)nc * sizeof(int));
            if (!tmp) { linear_program_free(prog); return NULL; }
            prog->loop_axes = tmp;
            prog->axes_capacity = nc;
        }
        prog->loop_axes[prog->num_axes++] = extents[i];
    }

    for (int i = 0; i < num_axes; i++) {
        LinearOp loop;
        memset(&loop, 0, sizeof(loop));
        loop.kind = LINOP_LOOP;
        loop.loop_axis = i;
        loop.loop_extent = extents[i];
        loop.loop_stride = 1;
        linear_program_emit(prog, loop);
    }

    LinearOp load;
    memset(&load, 0, sizeof(load));
    load.kind = LINOP_LOAD;
    load.dest_reg = alloc_vreg(prog);
    linear_program_emit(prog, load);

    LinearOp compute;
    memset(&compute, 0, sizeof(compute));
    compute.kind = LINOP_COMPUTE;
    compute.dest_reg = alloc_vreg(prog);
    compute.src_regs[0] = load.dest_reg;
    compute.num_srcs = 1;
    linear_program_emit(prog, compute);

    LinearOp store;
    memset(&store, 0, sizeof(store));
    store.kind = LINOP_STORE;
    store.dest_reg = compute.dest_reg;
    linear_program_emit(prog, store);

    for (int i = num_axes - 1; i >= 0; i--) {
        LinearOp endloop;
        memset(&endloop, 0, sizeof(endloop));
        endloop.kind = LINOP_ENDLOOP;
        endloop.loop_axis = i;
        linear_program_emit(prog, endloop);
    }

    return prog;
}

/* ── Create / free tests ── */

static int test_create_free(void) {
    CMLOptList* list = cml_opt_list_create();
    if (!list) return 0;
    if (list->num_opts != 0) { cml_opt_list_free(list); return 0; }
    cml_opt_list_free(list);
    return 1;
}

static int test_free_null(void) {
    cml_opt_list_free(NULL);
    return 1;
}

static int test_add_opts(void) {
    CMLOptList* list = cml_opt_list_create();
    if (!list) return 0;
    cml_opt_list_add(list, OPT_UNROLL, 0, 4);
    cml_opt_list_add(list, OPT_UPCAST, 1, 2);
    if (list->num_opts != 2) { cml_opt_list_free(list); return 0; }
    if (list->opts[0].type != OPT_UNROLL) { cml_opt_list_free(list); return 0; }
    if (list->opts[0].axis != 0) { cml_opt_list_free(list); return 0; }
    if (list->opts[0].amount != 4) { cml_opt_list_free(list); return 0; }
    if (list->opts[1].type != OPT_UPCAST) { cml_opt_list_free(list); return 0; }
    cml_opt_list_free(list);
    return 1;
}

static int test_add_many(void) {
    CMLOptList* list = cml_opt_list_create();
    if (!list) return 0;
    for (int i = 0; i < 100; i++)
        cml_opt_list_add(list, OPT_UNROLL, i % 4, 2);
    if (list->num_opts != 100) { cml_opt_list_free(list); return 0; }
    cml_opt_list_free(list);
    return 1;
}

/* ── Type name test ── */

static int test_type_names(void) {
    if (strcmp(cml_opt_type_name(OPT_LOCAL), "LOCAL") != 0) return 0;
    if (strcmp(cml_opt_type_name(OPT_GROUP), "GROUP") != 0) return 0;
    if (strcmp(cml_opt_type_name(OPT_UNROLL), "UNROLL") != 0) return 0;
    if (strcmp(cml_opt_type_name(OPT_UPCAST), "UPCAST") != 0) return 0;
    if (strcmp(cml_opt_type_name(OPT_PADTO), "PADTO") != 0) return 0;
    if (strcmp(cml_opt_type_name(OPT_NOLOCALS), "NOLOCALS") != 0) return 0;
    return 1;
}

/* ── Apply null args ── */

static int test_apply_null(void) {
    if (cml_opt_apply(NULL, NULL) != -1) return 0;
    CMLOptList* list = cml_opt_list_create();
    if (cml_opt_apply(list, NULL) != -1) { cml_opt_list_free(list); return 0; }
    cml_opt_list_free(list);
    return 1;
}

/* ── UNROLL transform ── */

static int test_unroll_basic(void) {
    int extents[] = {32, 16};
    LinearProgram* prog = make_test_prog(2, extents);
    if (!prog) return 0;

    CMLOptList* opts = cml_opt_list_create();
    cml_opt_list_add(opts, OPT_UNROLL, 0, 4);

    int rc = cml_opt_apply(opts, prog);
    if (rc != 0) { cml_opt_list_free(opts); linear_program_free(prog); return 0; }

    if (prog->loop_axes[0] != 8) {
        printf("(expected axis 0 = 8, got %d) ", prog->loop_axes[0]);
        cml_opt_list_free(opts); linear_program_free(prog); return 0;
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return 1;
}

static int test_unroll_bad_factor(void) {
    int extents[] = {32};
    LinearProgram* prog = make_test_prog(1, extents);
    if (!prog) return 0;

    CMLOptList* opts = cml_opt_list_create();
    cml_opt_list_add(opts, OPT_UNROLL, 0, 5); /* 32 % 5 != 0 */

    int rc = cml_opt_apply(opts, prog);
    if (rc == 0) { cml_opt_list_free(opts); linear_program_free(prog); return 0; }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return 1;
}

/* ── UPCAST transform ── */

static int test_upcast_basic(void) {
    int extents[] = {64};
    LinearProgram* prog = make_test_prog(1, extents);
    if (!prog) return 0;

    CMLOptList* opts = cml_opt_list_create();
    cml_opt_list_add(opts, OPT_UPCAST, 0, 4);

    int rc = cml_opt_apply(opts, prog);
    if (rc != 0) { cml_opt_list_free(opts); linear_program_free(prog); return 0; }

    if (prog->loop_axes[0] != 16) {
        printf("(expected 16, got %d) ", prog->loop_axes[0]);
        cml_opt_list_free(opts); linear_program_free(prog); return 0;
    }

    bool found_vec = false;
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].vec_width == 4) { found_vec = true; break; }
    }
    if (!found_vec) {
        printf("(no vec_width=4 found) ");
        cml_opt_list_free(opts); linear_program_free(prog); return 0;
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return 1;
}

static int test_upcast_non_power2(void) {
    int extents[] = {64};
    LinearProgram* prog = make_test_prog(1, extents);
    if (!prog) return 0;

    CMLOptList* opts = cml_opt_list_create();
    cml_opt_list_add(opts, OPT_UPCAST, 0, 3); /* not power of 2 */

    int rc = cml_opt_apply(opts, prog);
    if (rc == 0) { cml_opt_list_free(opts); linear_program_free(prog); return 0; }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return 1;
}

/* ── GROUP transform ── */

static int test_group_basic(void) {
    int extents[] = {256};
    LinearProgram* prog = make_test_prog(1, extents);
    if (!prog) return 0;

    CMLOptList* opts = cml_opt_list_create();
    cml_opt_list_add(opts, OPT_GROUP, 0, 32);

    int rc = cml_opt_apply(opts, prog);
    if (rc != 0) { cml_opt_list_free(opts); linear_program_free(prog); return 0; }

    if (prog->loop_axes[0] != 8) {
        printf("(expected 8, got %d) ", prog->loop_axes[0]);
        cml_opt_list_free(opts); linear_program_free(prog); return 0;
    }
    if (prog->group_dims[0] != 32) {
        printf("(expected group_dims[0]=32, got %d) ", prog->group_dims[0]);
        cml_opt_list_free(opts); linear_program_free(prog); return 0;
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return 1;
}

/* ── LOCAL transform ── */

static int test_local_basic(void) {
    int extents[] = {128};
    LinearProgram* prog = make_test_prog(1, extents);
    if (!prog) return 0;

    CMLOptList* opts = cml_opt_list_create();
    cml_opt_list_add(opts, OPT_LOCAL, 0, 16);

    int rc = cml_opt_apply(opts, prog);
    if (rc != 0) { cml_opt_list_free(opts); linear_program_free(prog); return 0; }

    if (prog->loop_axes[0] != 8) {
        printf("(expected 8, got %d) ", prog->loop_axes[0]);
        cml_opt_list_free(opts); linear_program_free(prog); return 0;
    }
    if (!prog->has_local_memory) {
        printf("(expected has_local_memory) ");
        cml_opt_list_free(opts); linear_program_free(prog); return 0;
    }
    if (prog->local_mem_used != 16 * sizeof(float)) {
        printf("(bad local_mem_used) ");
        cml_opt_list_free(opts); linear_program_free(prog); return 0;
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return 1;
}

/* ── PADTO transform ── */

static int test_padto_basic(void) {
    int extents[] = {50};
    LinearProgram* prog = make_test_prog(1, extents);
    if (!prog) return 0;

    CMLOptList* opts = cml_opt_list_create();
    cml_opt_list_add(opts, OPT_PADTO, 0, 16);

    int rc = cml_opt_apply(opts, prog);
    if (rc != 0) { cml_opt_list_free(opts); linear_program_free(prog); return 0; }

    if (prog->loop_axes[0] != 64) { /* ceil(50/16)*16 = 64 */
        printf("(expected 64, got %d) ", prog->loop_axes[0]);
        cml_opt_list_free(opts); linear_program_free(prog); return 0;
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return 1;
}

static int test_padto_already_aligned(void) {
    int extents[] = {64};
    LinearProgram* prog = make_test_prog(1, extents);
    if (!prog) return 0;

    CMLOptList* opts = cml_opt_list_create();
    cml_opt_list_add(opts, OPT_PADTO, 0, 16);

    int rc = cml_opt_apply(opts, prog);
    if (rc != 0) { cml_opt_list_free(opts); linear_program_free(prog); return 0; }

    if (prog->loop_axes[0] != 64) {
        printf("(expected 64, got %d) ", prog->loop_axes[0]);
        cml_opt_list_free(opts); linear_program_free(prog); return 0;
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return 1;
}

/* ── NOLOCALS transform ── */

static int test_nolocals(void) {
    int extents[] = {128};
    LinearProgram* prog = make_test_prog(1, extents);
    if (!prog) return 0;

    /* First apply LOCAL to add shared memory ops. */
    CMLOptList* opts1 = cml_opt_list_create();
    cml_opt_list_add(opts1, OPT_LOCAL, 0, 16);
    cml_opt_apply(opts1, prog);
    cml_opt_list_free(opts1);

    if (!prog->has_local_memory) {
        linear_program_free(prog);
        return 0;
    }

    int ops_before = prog->num_ops;

    CMLOptList* opts2 = cml_opt_list_create();
    cml_opt_list_add(opts2, OPT_NOLOCALS, 0, 0);
    int rc = cml_opt_apply(opts2, prog);
    cml_opt_list_free(opts2);

    if (rc != 0) { linear_program_free(prog); return 0; }
    if (prog->has_local_memory) {
        printf("(local memory not cleared) ");
        linear_program_free(prog);
        return 0;
    }
    if (prog->num_ops >= ops_before) {
        printf("(ops not reduced: %d >= %d) ", prog->num_ops, ops_before);
        linear_program_free(prog);
        return 0;
    }

    linear_program_free(prog);
    return 1;
}

/* ── Combination tests ── */

static int test_group_then_upcast(void) {
    int extents[] = {256, 64};
    LinearProgram* prog = make_test_prog(2, extents);
    if (!prog) return 0;

    CMLOptList* opts = cml_opt_list_create();
    cml_opt_list_add(opts, OPT_GROUP, 0, 32);
    cml_opt_list_add(opts, OPT_UPCAST, 1, 4);

    int rc = cml_opt_apply(opts, prog);
    if (rc != 0) { cml_opt_list_free(opts); linear_program_free(prog); return 0; }

    if (prog->loop_axes[0] != 8) {
        printf("(axis 0 expected 8, got %d) ", prog->loop_axes[0]);
        cml_opt_list_free(opts); linear_program_free(prog); return 0;
    }
    if (prog->loop_axes[1] != 16) {
        printf("(axis 1 expected 16, got %d) ", prog->loop_axes[1]);
        cml_opt_list_free(opts); linear_program_free(prog); return 0;
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return 1;
}

static int test_unroll_then_upcast(void) {
    int extents[] = {64, 32};
    LinearProgram* prog = make_test_prog(2, extents);
    if (!prog) return 0;

    CMLOptList* opts = cml_opt_list_create();
    cml_opt_list_add(opts, OPT_UNROLL, 0, 4);
    cml_opt_list_add(opts, OPT_UPCAST, 1, 2);

    int rc = cml_opt_apply(opts, prog);
    if (rc != 0) { cml_opt_list_free(opts); linear_program_free(prog); return 0; }

    if (prog->loop_axes[0] != 16) {
        cml_opt_list_free(opts); linear_program_free(prog); return 0;
    }
    if (prog->loop_axes[1] != 16) {
        cml_opt_list_free(opts); linear_program_free(prog); return 0;
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return 1;
}

/* ── Enumeration tests ── */

static int test_enumerate_basic(void) {
    int extents[] = {32, 16};
    LinearProgram* prog = make_test_prog(2, extents);
    if (!prog) return 0;

    CMLOptList** lists = NULL;
    int count = 0;

    int rc = cml_opt_enumerate(prog, &lists, &count, 256);
    if (rc != 0) { linear_program_free(prog); return 0; }

    if (count < 2) {
        printf("(too few combinations: %d) ", count);
        for (int i = 0; i < count; i++) cml_opt_list_free(lists[i]);
        free(lists);
        linear_program_free(prog);
        return 0;
    }

    /* First should be the empty baseline. */
    if (lists[0]->num_opts != 0) {
        printf("(first list not empty) ");
        for (int i = 0; i < count; i++) cml_opt_list_free(lists[i]);
        free(lists);
        linear_program_free(prog);
        return 0;
    }

    for (int i = 0; i < count; i++) cml_opt_list_free(lists[i]);
    free(lists);
    linear_program_free(prog);
    return 1;
}

static int test_enumerate_respects_max(void) {
    int extents[] = {64, 32, 16};
    LinearProgram* prog = make_test_prog(3, extents);
    if (!prog) return 0;

    CMLOptList** lists = NULL;
    int count = 0;

    int rc = cml_opt_enumerate(prog, &lists, &count, 10);
    if (rc != 0) { linear_program_free(prog); return 0; }

    if (count > 10) {
        printf("(count %d exceeds max 10) ", count);
        for (int i = 0; i < count; i++) cml_opt_list_free(lists[i]);
        free(lists);
        linear_program_free(prog);
        return 0;
    }

    for (int i = 0; i < count; i++) cml_opt_list_free(lists[i]);
    free(lists);
    linear_program_free(prog);
    return 1;
}

static int test_enumerate_null_args(void) {
    if (cml_opt_enumerate(NULL, NULL, NULL, 10) != -1) return 0;
    return 1;
}

static int test_enumerate_all_valid(void) {
    int extents[] = {64, 32};
    LinearProgram* prog = make_test_prog(2, extents);
    if (!prog) return 0;

    CMLOptList** lists = NULL;
    int count = 0;

    int rc = cml_opt_enumerate(prog, &lists, &count, 512);
    if (rc != 0) { linear_program_free(prog); return 0; }

    int valid = 0;
    for (int i = 0; i < count; i++) {
        /* Clone the program and try applying. */
        LinearProgram* clone = make_test_prog(2, extents);
        if (!clone) continue;
        if (cml_opt_apply(lists[i], clone) == 0) valid++;
        linear_program_free(clone);
    }

    for (int i = 0; i < count; i++) cml_opt_list_free(lists[i]);
    free(lists);
    linear_program_free(prog);

    /* At least the baseline should always be valid. */
    if (valid < 1) {
        printf("(no valid configs) ");
        return 0;
    }
    return 1;
}

/* ── Invalid axis tests ── */

static int test_bad_axis(void) {
    int extents[] = {32};
    LinearProgram* prog = make_test_prog(1, extents);
    if (!prog) return 0;

    CMLOptList* opts = cml_opt_list_create();
    cml_opt_list_add(opts, OPT_UNROLL, 5, 4); /* axis 5 doesn't exist */

    int rc = cml_opt_apply(opts, prog);
    if (rc == 0) { cml_opt_list_free(opts); linear_program_free(prog); return 0; }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return 1;
}

int main(void) {
    printf("test_opt_transforms\n\n");

    TEST(create_free);
    TEST(free_null);
    TEST(add_opts);
    TEST(add_many);
    TEST(type_names);
    TEST(apply_null);

    TEST(unroll_basic);
    TEST(unroll_bad_factor);
    TEST(upcast_basic);
    TEST(upcast_non_power2);
    TEST(group_basic);
    TEST(local_basic);
    TEST(padto_basic);
    TEST(padto_already_aligned);
    TEST(nolocals);

    TEST(group_then_upcast);
    TEST(unroll_then_upcast);

    TEST(enumerate_basic);
    TEST(enumerate_respects_max);
    TEST(enumerate_null_args);
    TEST(enumerate_all_valid);

    TEST(bad_axis);

    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
