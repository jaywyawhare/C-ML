#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ops/ir/heuristic_opt.h"
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

static LinearProgram* make_prog(int num_axes, const int* extents,
                                UOpType uop) {
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
    compute.uop = uop;
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

static int test_config_defaults(void) {
    CMLHeuristicConfig cfg = cml_heuristic_get_config();
    if (cfg.max_local_size != CML_HEURISTIC_DEFAULT_LOCAL_SIZE) return 0;
    if (cfg.preferred_vec_width != CML_HEURISTIC_DEFAULT_VEC_WIDTH) return 0;
    if (!cfg.use_local_memory) return 0;
    return 1;
}

static int test_config_set_get(void) {
    CMLHeuristicConfig original = cml_heuristic_get_config();

    CMLHeuristicConfig cfg = {
        .max_local_size = 512,
        .preferred_vec_width = 8,
        .use_local_memory = false,
    };
    cml_heuristic_set_config(&cfg);

    CMLHeuristicConfig got = cml_heuristic_get_config();
    cml_heuristic_set_config(&original);

    if (got.max_local_size != 512) return 0;
    if (got.preferred_vec_width != 8) return 0;
    if (got.use_local_memory) return 0;
    return 1;
}

static int test_null_prog(void) {
    CMLOptList* opts = cml_heuristic_optimize(NULL);
    if (!opts) return 0;
    if (opts->num_opts != 0) { cml_opt_list_free(opts); return 0; }
    cml_opt_list_free(opts);
    return 1;
}

static int test_elementwise_generates_upcast(void) {
    int extents[] = {256, 64};
    LinearProgram* prog = make_prog(2, extents, UOP_ADD);
    if (!prog) return 0;

    CMLOptList* opts = cml_heuristic_optimize(prog);
    if (!opts) { linear_program_free(prog); return 0; }

    bool has_upcast = false;
    for (int i = 0; i < opts->num_opts; i++) {
        if (opts->opts[i].type == OPT_UPCAST)
            has_upcast = true;
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return has_upcast;
}

static int test_elementwise_generates_group(void) {
    int extents[] = {1024};
    LinearProgram* prog = make_prog(1, extents, UOP_MUL);
    if (!prog) return 0;

    CMLOptList* opts = cml_heuristic_optimize(prog);
    if (!opts) { linear_program_free(prog); return 0; }

    bool has_group = false;
    for (int i = 0; i < opts->num_opts; i++) {
        if (opts->opts[i].type == OPT_GROUP)
            has_group = true;
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return has_group;
}

static int test_reduce_generates_local(void) {
    int extents[] = {64, 256};
    LinearProgram* prog = make_prog(2, extents, UOP_SUM);
    if (!prog) return 0;

    CMLOptList* opts = cml_heuristic_optimize(prog);
    if (!opts) { linear_program_free(prog); return 0; }

    bool has_local = false;
    for (int i = 0; i < opts->num_opts; i++) {
        if (opts->opts[i].type == OPT_LOCAL)
            has_local = true;
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return has_local;
}

static int test_matmul_generates_local(void) {
    int extents[] = {128, 128, 64};
    LinearProgram* prog = make_prog(3, extents, UOP_MATMUL);
    if (!prog) return 0;

    CMLOptList* opts = cml_heuristic_optimize(prog);
    if (!opts) { linear_program_free(prog); return 0; }

    int local_count = 0;
    bool has_unroll = false;
    for (int i = 0; i < opts->num_opts; i++) {
        if (opts->opts[i].type == OPT_LOCAL)
            local_count++;
        if (opts->opts[i].type == OPT_UNROLL)
            has_unroll = true;
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return local_count >= 2 && has_unroll;
}

static int test_conv_generates_spatial_local(void) {
    int extents[] = {32, 32, 64};
    LinearProgram* prog = make_prog(3, extents, UOP_CONV2D);
    if (!prog) return 0;

    CMLOptList* opts = cml_heuristic_optimize(prog);
    if (!opts) { linear_program_free(prog); return 0; }

    int local_count = 0;
    bool has_unroll = false;
    for (int i = 0; i < opts->num_opts; i++) {
        if (opts->opts[i].type == OPT_LOCAL)
            local_count++;
        if (opts->opts[i].type == OPT_UNROLL)
            has_unroll = true;
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return local_count >= 1 && has_unroll;
}

static int test_opts_are_valid(void) {
    int extents[] = {256, 64};
    LinearProgram* prog = make_prog(2, extents, UOP_ADD);
    if (!prog) return 0;

    CMLOptList* opts = cml_heuristic_optimize(prog);
    if (!opts) { linear_program_free(prog); return 0; }

    for (int i = 0; i < opts->num_opts; i++) {
        if (opts->opts[i].amount <= 0) {
            printf("(invalid amount %d) ", opts->opts[i].amount);
            cml_opt_list_free(opts);
            linear_program_free(prog);
            return 0;
        }
        if (opts->opts[i].axis < 0) {
            printf("(invalid axis %d) ", opts->opts[i].axis);
            cml_opt_list_free(opts);
            linear_program_free(prog);
            return 0;
        }
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return 1;
}

static int test_opts_can_be_applied(void) {
    int extents[] = {256, 64};
    LinearProgram* prog = make_prog(2, extents, UOP_ADD);
    if (!prog) return 0;

    CMLOptList* opts = cml_heuristic_optimize(prog);
    if (!opts) { linear_program_free(prog); return 0; }

    LinearProgram* fresh = make_prog(2, extents, UOP_ADD);
    if (!fresh) { cml_opt_list_free(opts); linear_program_free(prog); return 0; }

    int rc = cml_opt_apply(opts, fresh);

    cml_opt_list_free(opts);
    linear_program_free(prog);
    linear_program_free(fresh);
    return rc == 0;
}

static int test_no_local_memory_config(void) {
    CMLHeuristicConfig original = cml_heuristic_get_config();
    CMLHeuristicConfig cfg = original;
    cfg.use_local_memory = false;
    cml_heuristic_set_config(&cfg);

    int extents[] = {64, 256};
    LinearProgram* prog = make_prog(2, extents, UOP_SUM);
    if (!prog) { cml_heuristic_set_config(&original); return 0; }

    CMLOptList* opts = cml_heuristic_optimize(prog);
    cml_heuristic_set_config(&original);

    if (!opts) { linear_program_free(prog); return 0; }

    for (int i = 0; i < opts->num_opts; i++) {
        if (opts->opts[i].type == OPT_LOCAL) {
            printf("(LOCAL not expected when disabled) ");
            cml_opt_list_free(opts);
            linear_program_free(prog);
            return 0;
        }
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return 1;
}

static int test_single_element_axis(void) {
    int extents[] = {1};
    LinearProgram* prog = make_prog(1, extents, UOP_ADD);
    if (!prog) return 0;

    CMLOptList* opts = cml_heuristic_optimize(prog);
    if (!opts) { linear_program_free(prog); return 0; }

    if (opts->num_opts != 0) {
        printf("(expected 0 opts for trivial axis, got %d) ", opts->num_opts);
        cml_opt_list_free(opts);
        linear_program_free(prog);
        return 0;
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return 1;
}

static int test_power_of_two_amounts(void) {
    int extents[] = {256, 128};
    LinearProgram* prog = make_prog(2, extents, UOP_ADD);
    if (!prog) return 0;

    CMLOptList* opts = cml_heuristic_optimize(prog);
    if (!opts) { linear_program_free(prog); return 0; }

    for (int i = 0; i < opts->num_opts; i++) {
        int a = opts->opts[i].amount;
        if (a > 1 && (a & (a - 1)) != 0) {
            printf("(amount %d not power-of-2) ", a);
            cml_opt_list_free(opts);
            linear_program_free(prog);
            return 0;
        }
    }

    cml_opt_list_free(opts);
    linear_program_free(prog);
    return 1;
}

int main(void) {
    printf("test_heuristic_opt\n\n");

    TEST(config_defaults);
    TEST(config_set_get);
    TEST(null_prog);
    TEST(elementwise_generates_upcast);
    TEST(elementwise_generates_group);
    TEST(reduce_generates_local);
    TEST(matmul_generates_local);
    TEST(conv_generates_spatial_local);
    TEST(opts_are_valid);
    TEST(opts_can_be_applied);
    TEST(no_local_memory_config);
    TEST(single_element_axis);
    TEST(power_of_two_amounts);

    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
