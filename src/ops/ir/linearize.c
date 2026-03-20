#include "ops/ir/linearize.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static bool is_eliminated(const CMLFusionGroup* g, int node_idx) {
    if (!g) return false;
    for (int i = 0; i < g->num_eliminated; i++) {
        if (g->eliminated_buffers[i] == node_idx)
            return true;
    }
    return false;
}

LinearProgram* linear_program_create(void) {
    LinearProgram* prog = calloc(1, sizeof(LinearProgram));
    if (!prog) return NULL;
    prog->capacity = 32;
    prog->ops = calloc((size_t)prog->capacity, sizeof(LinearOp));
    if (!prog->ops) { free(prog); return NULL; }
    prog->next_vreg = 0;
    prog->axes_capacity = 8;
    prog->loop_axes = calloc((size_t)prog->axes_capacity, sizeof(int));
    if (!prog->loop_axes) { free(prog->ops); free(prog); return NULL; }
    prog->group_dims[0] = 1;
    prog->group_dims[1] = 1;
    prog->group_dims[2] = 1;
    return prog;
}

void linear_program_free(LinearProgram* prog) {
    if (!prog) return;
    free(prog->loop_axes);
    free(prog->ops);
    free(prog);
}

int linear_program_emit(LinearProgram* prog, LinearOp op) {
    if (!prog) return -1;
    if (prog->num_ops >= prog->capacity) {
        int nc = prog->capacity * 2;
        LinearOp* tmp = realloc(prog->ops, (size_t)nc * sizeof(LinearOp));
        if (!tmp) return -1;
        prog->ops = tmp;
        prog->capacity = nc;
    }
    prog->ops[prog->num_ops++] = op;
    return 0;
}

int alloc_vreg(LinearProgram* prog) {
    if (!prog) return -1;
    if (prog->next_vreg >= MAX_VIRTUAL_REGS) {
        LOG_WARNING("Virtual register file exhausted (%d regs)", MAX_VIRTUAL_REGS);
        return -1;
    }
    return prog->next_vreg++;
}

LinearProgram* linearize_group(const CMLFusionGroup* g) {
    if (!g || g->num_nodes == 0) return NULL;

    LinearProgram* prog = linear_program_create();
    if (!prog) return NULL;

    int* node_vreg = calloc((size_t)g->num_nodes, sizeof(int));
    if (!node_vreg) { linear_program_free(prog); return NULL; }
    for (int i = 0; i < g->num_nodes; i++) node_vreg[i] = -1;

    Tensor* loaded_tensors[MAX_VIRTUAL_REGS];
    int     loaded_vregs[MAX_VIRTUAL_REGS];
    int     num_loaded = 0;

    for (int i = 0; i < g->num_nodes; i++) {
        struct IRNode* node = g->nodes[i];
        if (!node) continue;

        LinearOp compute_op;
        memset(&compute_op, 0, sizeof(compute_op));
        compute_op.kind = LINOP_COMPUTE;
        compute_op.uop  = node->type;
        compute_op.num_srcs = 0;

        for (int j = 0; j < node->num_inputs && j < 8; j++) {
            Tensor* inp = (node->inputs) ? node->inputs[j] : NULL;
            if (!inp) continue;

            int src_reg = -1;

            for (int k = 0; k < i; k++) {
                if (g->nodes[k] && g->nodes[k]->output == inp) {
                    src_reg = node_vreg[k];
                    break;
                }
            }

            if (src_reg < 0) {
                for (int k = 0; k < num_loaded; k++) {
                    if (loaded_tensors[k] == inp) {
                        src_reg = loaded_vregs[k];
                        break;
                    }
                }
            }

            if (src_reg < 0) {
                src_reg = alloc_vreg(prog);
                if (src_reg < 0) break;

                LinearOp load_op;
                memset(&load_op, 0, sizeof(load_op));
                load_op.kind     = LINOP_LOAD;
                load_op.dest_reg = src_reg;
                load_op.tensor   = inp;
                linear_program_emit(prog, load_op);

                if (num_loaded < MAX_VIRTUAL_REGS) {
                    loaded_tensors[num_loaded] = inp;
                    loaded_vregs[num_loaded]   = src_reg;
                    num_loaded++;
                }
            }

            compute_op.src_regs[compute_op.num_srcs++] = src_reg;
        }

        int dest = alloc_vreg(prog);
        if (dest < 0) break;
        compute_op.dest_reg = dest;
        compute_op.is_eliminated = is_eliminated(g, i);
        node_vreg[i] = dest;

        linear_program_emit(prog, compute_op);

        if (!compute_op.is_eliminated) {
            LinearOp store_op;
            memset(&store_op, 0, sizeof(store_op));
            store_op.kind     = LINOP_STORE;
            store_op.dest_reg = dest;
            store_op.tensor   = node->output;
            linear_program_emit(prog, store_op);
        }
    }

    free(node_vreg);
    return prog;
}

const char* linop_name(LinearOpKind k) {
    switch (k) {
        case LINOP_LOAD:        return "LOAD";
        case LINOP_COMPUTE:     return "COMPUTE";
        case LINOP_STORE:       return "STORE";
        case LINOP_LOOP:        return "LOOP";
        case LINOP_ENDLOOP:     return "ENDLOOP";
        case LINOP_BARRIER:     return "BARRIER";
        case LINOP_LOCAL_ALLOC: return "LOCAL_ALLOC";
        case LINOP_LOCAL_LOAD:  return "LOCAL_LOAD";
        case LINOP_LOCAL_STORE: return "LOCAL_STORE";
        default:                return "???";
    }
}

void linear_program_print(const LinearProgram* prog) {
    if (!prog) { printf("LinearProgram: (null)\n"); return; }

    printf("Linear Program (%d ops, %d vregs)\n",
           prog->num_ops, prog->next_vreg);
    for (int i = 0; i < prog->num_ops; i++) {
        const LinearOp* op = &prog->ops[i];
        printf("  [%d] %-12s  dest=v%d", i, linop_name(op->kind), op->dest_reg);
        if (op->kind == LINOP_COMPUTE) {
            printf("  uop=%s  srcs=[", uop_type_to_string(op->uop));
            for (int j = 0; j < op->num_srcs; j++) {
                if (j > 0) printf(", ");
                printf("v%d", op->src_regs[j]);
            }
            printf("]");
            if (op->is_eliminated) printf("  [ELIM]");
            if (op->vec_width > 1) printf("  [VEC%d]", op->vec_width);
        }
        if (op->kind == LINOP_LOOP)
            printf("  axis=%d extent=%d stride=%d",
                   op->loop_axis, op->loop_extent, op->loop_stride);
        printf("\n");
    }
}

void cml_linearize_group_print(const CMLFusionGroup* g) {
    if (!g) return;
    LinearProgram* prog = linearize_group(g);
    if (prog) {
        linear_program_print(prog);
        linear_program_free(prog);
    }
}

int cml_linearize_group_count(const CMLFusionGroup* g) {
    if (!g) return -1;
    LinearProgram* prog = linearize_group(g);
    if (!prog) return -1;
    int count = prog->num_ops;
    linear_program_free(prog);
    return count;
}
