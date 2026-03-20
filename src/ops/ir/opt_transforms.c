#include "ops/ir/opt_transforms.h"
#include "ops/ir/linearize.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define OPT_MAX_LOCAL_MEM    49152  /* 48 KiB typical GPU shared memory */
#define OPT_MAX_WORKGROUP    1024
#define OPT_MAX_VEC_WIDTH    16

static const int TILE_FACTORS[]   = {2, 4, 8, 16, 32};
static const int UNROLL_FACTORS[] = {2, 4, 8};
static const int VEC_WIDTHS[]     = {2, 4, 8};
static const int PAD_MULTIPLES[]  = {16, 32, 64};

#define NUM_TILE_FACTORS   (int)(sizeof(TILE_FACTORS)   / sizeof(TILE_FACTORS[0]))
#define NUM_UNROLL_FACTORS (int)(sizeof(UNROLL_FACTORS) / sizeof(UNROLL_FACTORS[0]))
#define NUM_VEC_WIDTHS     (int)(sizeof(VEC_WIDTHS)     / sizeof(VEC_WIDTHS[0]))
#define NUM_PAD_MULTIPLES  (int)(sizeof(PAD_MULTIPLES)  / sizeof(PAD_MULTIPLES[0]))

const char* cml_opt_type_name(CMLOptType type) {
    switch (type) {
        case OPT_LOCAL:    return "LOCAL";
        case OPT_GROUP:    return "GROUP";
        case OPT_UNROLL:   return "UNROLL";
        case OPT_UPCAST:   return "UPCAST";
        case OPT_PADTO:    return "PADTO";
        case OPT_NOLOCALS: return "NOLOCALS";
    }
    return "???";
}

CMLOptList* cml_opt_list_create(void) {
    CMLOptList* list = calloc(1, sizeof(CMLOptList));
    if (!list) return NULL;
    list->capacity = 8;
    list->opts = calloc((size_t)list->capacity, sizeof(CMLOpt));
    if (!list->opts) { free(list); return NULL; }
    return list;
}

void cml_opt_list_free(CMLOptList* list) {
    if (!list) return;
    free(list->opts);
    free(list);
}

void cml_opt_list_add(CMLOptList* list, CMLOptType type, int axis, int amount) {
    if (!list) return;
    if (list->num_opts >= list->capacity) {
        int nc = list->capacity * 2;
        CMLOpt* tmp = realloc(list->opts, (size_t)nc * sizeof(CMLOpt));
        if (!tmp) return;
        list->opts = tmp;
        list->capacity = nc;
    }
    CMLOpt opt = { .type = type, .axis = axis, .amount = amount };
    list->opts[list->num_opts++] = opt;
}

static CMLOptList* opt_list_clone(const CMLOptList* src) {
    if (!src) return NULL;
    CMLOptList* dst = cml_opt_list_create();
    if (!dst) return NULL;
    for (int i = 0; i < src->num_opts; i++)
        cml_opt_list_add(dst, src->opts[i].type, src->opts[i].axis,
                         src->opts[i].amount);
    return dst;
}

/* ── Axis helpers ── */

static int prog_add_axis(LinearProgram* prog, int extent) {
    if (!prog) return -1;
    if (prog->num_axes >= prog->axes_capacity) {
        int nc = prog->axes_capacity * 2;
        int* tmp = realloc(prog->loop_axes, (size_t)nc * sizeof(int));
        if (!tmp) return -1;
        prog->loop_axes = tmp;
        prog->axes_capacity = nc;
    }
    int idx = prog->num_axes;
    prog->loop_axes[prog->num_axes++] = extent;
    return idx;
}

static int prog_get_axis_extent(const LinearProgram* prog, int axis) {
    if (!prog || axis < 0 || axis >= prog->num_axes) return -1;
    return prog->loop_axes[axis];
}

/* ── Individual transform implementations ── */

static int apply_local(int axis, int amount, LinearProgram* prog) {
    int extent = prog_get_axis_extent(prog, axis);
    if (extent <= 0) {
        LOG_WARNING("OPT_LOCAL: invalid axis %d", axis);
        return -1;
    }
    if (amount <= 0 || extent % amount != 0) {
        LOG_WARNING("OPT_LOCAL: amount %d does not divide extent %d", amount, extent);
        return -1;
    }
    size_t local_bytes = (size_t)amount * sizeof(float);
    if (prog->local_mem_used + local_bytes > OPT_MAX_LOCAL_MEM) {
        LOG_WARNING("OPT_LOCAL: exceeds local memory limit (%zu + %zu > %d)",
                    prog->local_mem_used, local_bytes, OPT_MAX_LOCAL_MEM);
        return -1;
    }

    int outer_extent = extent / amount;
    prog->loop_axes[axis] = outer_extent;
    int inner_axis = prog_add_axis(prog, amount);
    if (inner_axis < 0) return -1;

    LinearOp alloc_op;
    memset(&alloc_op, 0, sizeof(alloc_op));
    alloc_op.kind = LINOP_LOCAL_ALLOC;
    alloc_op.local_size = local_bytes;
    alloc_op.loop_axis = inner_axis;
    linear_program_emit(prog, alloc_op);

    LinearOp barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.kind = LINOP_BARRIER;
    linear_program_emit(prog, barrier);

    prog->has_local_memory = true;
    prog->local_mem_used += local_bytes;

    LOG_DEBUG("OPT_LOCAL: axis %d split %d -> %d * %d, local_mem=%zu",
              axis, extent, outer_extent, amount, prog->local_mem_used);
    return 0;
}

static int apply_group(int axis, int amount, LinearProgram* prog) {
    int extent = prog_get_axis_extent(prog, axis);
    if (extent <= 0) {
        LOG_WARNING("OPT_GROUP: invalid axis %d", axis);
        return -1;
    }
    if (amount <= 0 || extent % amount != 0) {
        LOG_WARNING("OPT_GROUP: amount %d does not divide extent %d", amount, extent);
        return -1;
    }

    int group_extent = extent / amount;
    int local_extent = amount;

    if (local_extent > OPT_MAX_WORKGROUP) {
        LOG_WARNING("OPT_GROUP: local extent %d exceeds max workgroup size %d",
                    local_extent, OPT_MAX_WORKGROUP);
        return -1;
    }

    prog->loop_axes[axis] = group_extent;
    int local_axis = prog_add_axis(prog, local_extent);
    if (local_axis < 0) return -1;

    int dim = -1;
    for (int d = 0; d < 3; d++) {
        if (prog->group_dims[d] == 1) { dim = d; break; }
    }
    if (dim < 0) {
        LOG_WARNING("OPT_GROUP: all 3 workgroup dimensions already used");
        return -1;
    }
    prog->group_dims[dim] = local_extent;

    LOG_DEBUG("OPT_GROUP: axis %d split %d -> group(%d) * local(%d), dim=%d",
              axis, extent, group_extent, local_extent, dim);
    return 0;
}

static int apply_unroll(int axis, int amount, LinearProgram* prog) {
    int extent = prog_get_axis_extent(prog, axis);
    if (extent <= 0) {
        LOG_WARNING("OPT_UNROLL: invalid axis %d", axis);
        return -1;
    }
    if (amount <= 0 || extent % amount != 0) {
        LOG_WARNING("OPT_UNROLL: amount %d does not divide extent %d", amount, extent);
        return -1;
    }

    int new_extent = extent / amount;
    prog->loop_axes[axis] = new_extent;

    /* Find all LOOPs on this axis and replicate the body. */
    int orig_num = prog->num_ops;
    for (int i = 0; i < orig_num; i++) {
        LinearOp* op = &prog->ops[i];
        if (op->kind == LINOP_LOOP && op->loop_axis == axis) {
            op->loop_extent = new_extent;

            /* Find the matching ENDLOOP. */
            int end = -1;
            int depth = 1;
            for (int j = i + 1; j < orig_num; j++) {
                if (prog->ops[j].kind == LINOP_LOOP) depth++;
                if (prog->ops[j].kind == LINOP_ENDLOOP) {
                    depth--;
                    if (depth == 0) { end = j; break; }
                }
            }

            if (end < 0) break;

            int body_len = end - i - 1;
            if (body_len <= 0) break;

            /* Emit (amount - 1) copies of the body after the original body. */
            for (int u = 1; u < amount; u++) {
                for (int b = 0; b < body_len; b++) {
                    LinearOp copy = prog->ops[i + 1 + b];
                    if (copy.kind == LINOP_COMPUTE || copy.kind == LINOP_LOAD) {
                        int new_reg = alloc_vreg(prog);
                        if (new_reg >= 0) copy.dest_reg = new_reg;
                    }
                    linear_program_emit(prog, copy);
                }
            }
            break;
        }
    }

    LOG_DEBUG("OPT_UNROLL: axis %d factor %d, extent %d -> %d",
              axis, amount, extent, new_extent);
    return 0;
}

static int apply_upcast(int axis, int amount, LinearProgram* prog) {
    int extent = prog_get_axis_extent(prog, axis);
    if (extent <= 0) {
        LOG_WARNING("OPT_UPCAST: invalid axis %d", axis);
        return -1;
    }
    if (amount <= 0 || amount > OPT_MAX_VEC_WIDTH || extent % amount != 0) {
        LOG_WARNING("OPT_UPCAST: invalid amount %d for extent %d", amount, extent);
        return -1;
    }
    if ((amount & (amount - 1)) != 0) {
        LOG_WARNING("OPT_UPCAST: amount %d is not a power of 2", amount);
        return -1;
    }

    prog->loop_axes[axis] = extent / amount;

    for (int i = 0; i < prog->num_ops; i++) {
        LinearOp* op = &prog->ops[i];
        if (op->kind == LINOP_LOAD || op->kind == LINOP_STORE ||
            op->kind == LINOP_COMPUTE) {
            if (op->vec_width <= 1)
                op->vec_width = amount;
        }
    }

    LOG_DEBUG("OPT_UPCAST: axis %d vec_width=%d, extent %d -> %d",
              axis, amount, extent, extent / amount);
    return 0;
}

static int apply_padto(int axis, int amount, LinearProgram* prog) {
    int extent = prog_get_axis_extent(prog, axis);
    if (extent <= 0) {
        LOG_WARNING("OPT_PADTO: invalid axis %d", axis);
        return -1;
    }
    if (amount <= 0) {
        LOG_WARNING("OPT_PADTO: invalid pad multiple %d", amount);
        return -1;
    }

    int padded = ((extent + amount - 1) / amount) * amount;
    if (padded == extent) return 0; /* already aligned */

    prog->loop_axes[axis] = padded;

    LOG_DEBUG("OPT_PADTO: axis %d padded %d -> %d (multiple of %d)",
              axis, extent, padded, amount);
    return 0;
}

static int apply_nolocals(LinearProgram* prog) {
    int dst = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        LinearOpKind k = prog->ops[i].kind;
        if (k == LINOP_LOCAL_ALLOC || k == LINOP_LOCAL_LOAD ||
            k == LINOP_LOCAL_STORE || k == LINOP_BARRIER) {
            continue;
        }
        if (dst != i) prog->ops[dst] = prog->ops[i];
        dst++;
    }
    prog->num_ops = dst;
    prog->has_local_memory = false;
    prog->local_mem_used = 0;

    LOG_DEBUG("OPT_NOLOCALS: stripped local memory ops, %d ops remain", dst);
    return 0;
}

/* ── Public apply ── */

int cml_opt_apply(CMLOptList* opts, struct LinearProgram* prog) {
    if (!opts || !prog) return -1;

    for (int i = 0; i < opts->num_opts; i++) {
        CMLOpt* o = &opts->opts[i];
        int rc = 0;
        switch (o->type) {
            case OPT_LOCAL:    rc = apply_local(o->axis, o->amount, prog);   break;
            case OPT_GROUP:    rc = apply_group(o->axis, o->amount, prog);   break;
            case OPT_UNROLL:   rc = apply_unroll(o->axis, o->amount, prog);  break;
            case OPT_UPCAST:   rc = apply_upcast(o->axis, o->amount, prog);  break;
            case OPT_PADTO:    rc = apply_padto(o->axis, o->amount, prog);   break;
            case OPT_NOLOCALS: rc = apply_nolocals(prog);                    break;
        }
        if (rc != 0) {
            LOG_WARNING("Opt %s(axis=%d, amount=%d) failed at step %d",
                        cml_opt_type_name(o->type), o->axis, o->amount, i);
            return rc;
        }
    }
    return 0;
}

/* ── Enumeration ── */

int cml_opt_enumerate(struct LinearProgram* prog, CMLOptList*** out_lists,
                      int* out_count, int max_combinations) {
    if (!prog || !out_lists || !out_count || max_combinations <= 0) return -1;

    int cap = max_combinations;
    CMLOptList** lists = calloc((size_t)cap, sizeof(CMLOptList*));
    if (!lists) return -1;
    int count = 0;

    /* Empty opt list (baseline). */
    lists[count++] = cml_opt_list_create();

    /* Single-opt transforms for each axis. */
    for (int ax = 0; ax < prog->num_axes && count < cap; ax++) {
        int extent = prog->loop_axes[ax];
        if (extent <= 1) continue;

        for (int ti = 0; ti < NUM_UNROLL_FACTORS && count < cap; ti++) {
            if (extent % UNROLL_FACTORS[ti] != 0) continue;
            CMLOptList* l = cml_opt_list_create();
            cml_opt_list_add(l, OPT_UNROLL, ax, UNROLL_FACTORS[ti]);
            lists[count++] = l;
        }

        for (int vi = 0; vi < NUM_VEC_WIDTHS && count < cap; vi++) {
            if (extent % VEC_WIDTHS[vi] != 0) continue;
            CMLOptList* l = cml_opt_list_create();
            cml_opt_list_add(l, OPT_UPCAST, ax, VEC_WIDTHS[vi]);
            lists[count++] = l;
        }

        for (int ti = 0; ti < NUM_TILE_FACTORS && count < cap; ti++) {
            if (extent % TILE_FACTORS[ti] != 0) continue;
            if (TILE_FACTORS[ti] > OPT_MAX_WORKGROUP) continue;
            CMLOptList* l = cml_opt_list_create();
            cml_opt_list_add(l, OPT_GROUP, ax, TILE_FACTORS[ti]);
            lists[count++] = l;
        }

        for (int ti = 0; ti < NUM_TILE_FACTORS && count < cap; ti++) {
            size_t local_bytes = (size_t)TILE_FACTORS[ti] * sizeof(float);
            if (extent % TILE_FACTORS[ti] != 0) continue;
            if (local_bytes > OPT_MAX_LOCAL_MEM) continue;
            CMLOptList* l = cml_opt_list_create();
            cml_opt_list_add(l, OPT_LOCAL, ax, TILE_FACTORS[ti]);
            lists[count++] = l;
        }

        for (int pi = 0; pi < NUM_PAD_MULTIPLES && count < cap; pi++) {
            if (extent % PAD_MULTIPLES[pi] == 0) continue; /* already aligned */
            CMLOptList* l = cml_opt_list_create();
            cml_opt_list_add(l, OPT_PADTO, ax, PAD_MULTIPLES[pi]);
            lists[count++] = l;
        }
    }

    /* Two-opt combinations: UPCAST + UNROLL, GROUP + UPCAST. */
    int single_end = count;
    for (int a = 1; a < single_end && count < cap; a++) {
        for (int b = a + 1; b < single_end && count < cap; b++) {
            CMLOpt* oa = &lists[a]->opts[0];
            CMLOpt* ob = &lists[b]->opts[0];

            if (oa->axis == ob->axis) continue; /* same axis conflicts */

            bool good_pair =
                (oa->type == OPT_UPCAST  && ob->type == OPT_UNROLL) ||
                (oa->type == OPT_UNROLL  && ob->type == OPT_UPCAST) ||
                (oa->type == OPT_GROUP   && ob->type == OPT_UPCAST) ||
                (oa->type == OPT_UPCAST  && ob->type == OPT_GROUP)  ||
                (oa->type == OPT_GROUP   && ob->type == OPT_UNROLL) ||
                (oa->type == OPT_UNROLL  && ob->type == OPT_GROUP)  ||
                (oa->type == OPT_LOCAL   && ob->type == OPT_UPCAST) ||
                (oa->type == OPT_UPCAST  && ob->type == OPT_LOCAL);

            if (!good_pair) continue;

            CMLOptList* l = opt_list_clone(lists[a]);
            if (!l) continue;
            cml_opt_list_add(l, ob->type, ob->axis, ob->amount);
            lists[count++] = l;
        }
    }

    /* NOLOCALS variant. */
    if (count < cap) {
        CMLOptList* l = cml_opt_list_create();
        cml_opt_list_add(l, OPT_NOLOCALS, 0, 0);
        lists[count++] = l;
    }

    *out_lists = lists;
    *out_count = count;

    LOG_INFO("OPT enumerate: generated %d configurations for %d axes",
             count, prog->num_axes);
    return 0;
}
