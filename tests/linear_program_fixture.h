/**
 * A synthetic LinearProgram for the linearizer/optimizer suites.
 *
 * test_heuristic_opt and test_opt_transforms both need a well-formed program to
 * transform -- nested loops around a load/compute/store -- and both must build
 * it the same way, or a transform that passes in one suite proves nothing about
 * the other.
 */
#ifndef CML_LINEAR_PROGRAM_FIXTURE_H
#define CML_LINEAR_PROGRAM_FIXTURE_H

#include <string.h>

#include "ops/ir/linearize.h"
#include "alloc/cml_allocator.h"

/* `num_axes` nested loops over `extents`, wrapping one load -> compute(`uop`)
 * -> store. Returns NULL on allocation failure. */
static inline LinearProgram* make_prog(int num_axes, const int* extents, UOpType uop) {
    LinearProgram* prog = linear_program_create();
    if (!prog)
        return NULL;

    for (int i = 0; i < num_axes; i++) {
        if (prog->num_axes >= prog->axes_capacity) {
            int nc   = prog->axes_capacity * 2;
            int* tmp = cml_realloc(prog->loop_axes, (size_t)nc * sizeof(int));
            if (!tmp) {
                linear_program_free(prog);
                return NULL;
            }
            prog->loop_axes     = tmp;
            prog->axes_capacity = nc;
        }
        prog->loop_axes[prog->num_axes++] = extents[i];
    }

    for (int i = 0; i < num_axes; i++) {
        LinearOp loop;
        memset(&loop, 0, sizeof(loop));
        loop.kind        = LINOP_LOOP;
        loop.loop_axis   = i;
        loop.loop_extent = extents[i];
        loop.loop_stride = 1;
        linear_program_emit(prog, loop);
    }

    LinearOp load;
    memset(&load, 0, sizeof(load));
    load.kind     = LINOP_LOAD;
    load.dest_reg = alloc_vreg(prog);
    linear_program_emit(prog, load);

    LinearOp compute;
    memset(&compute, 0, sizeof(compute));
    compute.kind        = LINOP_COMPUTE;
    compute.uop         = uop;
    compute.dest_reg    = alloc_vreg(prog);
    compute.src_regs[0] = load.dest_reg;
    compute.num_srcs    = 1;
    linear_program_emit(prog, compute);

    LinearOp store;
    memset(&store, 0, sizeof(store));
    store.kind     = LINOP_STORE;
    store.dest_reg = compute.dest_reg;
    linear_program_emit(prog, store);

    for (int i = num_axes - 1; i >= 0; i--) {
        LinearOp endloop;
        memset(&endloop, 0, sizeof(endloop));
        endloop.kind      = LINOP_ENDLOOP;
        endloop.loop_axis = i;
        linear_program_emit(prog, endloop);
    }

    return prog;
}

#endif /* CML_LINEAR_PROGRAM_FIXTURE_H */
