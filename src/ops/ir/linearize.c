/**
 * @file linearize.c
 * @brief Convert CMLFusionGroup to linear instruction sequence
 *
 * Maps eliminated buffers to virtual registers and emits a
 * load -> compute -> store instruction sequence for each fusion group.
 * Simple register allocation for eliminated intermediates.
 */

#include "ops/ir/schedule.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_VIRTUAL_REGS 64

typedef enum {
    LINOP_LOAD,     /* Load tensor from memory into vreg */
    LINOP_COMPUTE,  /* Execute UOp, result in vreg */
    LINOP_STORE,    /* Store vreg to output tensor */
} LinearOpKind;

typedef struct {
    LinearOpKind kind;
    UOpType uop;             /* The original operation (for COMPUTE) */
    int dest_reg;            /* Destination virtual register */
    int src_regs[8];         /* Source virtual registers */
    int num_srcs;
    Tensor* tensor;          /* Associated tensor (for LOAD/STORE) */
    bool is_eliminated;      /* True if this intermediate stays in reg */
} LinearOp;

typedef struct {
    LinearOp* ops;
    int num_ops;
    int capacity;
    int next_vreg;           /* Next free virtual register */
} LinearProgram;

static LinearProgram* linear_program_create(void) {
    LinearProgram* prog = calloc(1, sizeof(LinearProgram));
    if (!prog) return NULL;
    prog->capacity = 32;
    prog->ops = calloc((size_t)prog->capacity, sizeof(LinearOp));
    if (!prog->ops) { free(prog); return NULL; }
    prog->next_vreg = 0;
    return prog;
}

static void linear_program_free(LinearProgram* prog) {
    if (!prog) return;
    free(prog->ops);
    free(prog);
}

static int linear_program_emit(LinearProgram* prog, LinearOp op) {
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

static int alloc_vreg(LinearProgram* prog) {
    if (!prog) return -1;
    if (prog->next_vreg >= MAX_VIRTUAL_REGS) {
        LOG_WARNING("Virtual register file exhausted (%d regs)", MAX_VIRTUAL_REGS);
        return -1;
    }
    return prog->next_vreg++;
}

static bool is_eliminated(const CMLFusionGroup* g, int node_idx) {
    if (!g) return false;
    for (int i = 0; i < g->num_eliminated; i++) {
        if (g->eliminated_buffers[i] == node_idx)
            return true;
    }
    return false;
}

/**
 * Convert a CMLFusionGroup into a linear instruction sequence.
 *
 * Strategy:
 *   1. For each node, emit LOAD for inputs not already in a vreg.
 *   2. Emit COMPUTE with source vregs, allocating a dest vreg.
 *   3. If the node's output is *not* eliminated, emit STORE.
 *      Otherwise keep the result in the vreg for downstream use.
 *
 * Returns a freshly allocated LinearProgram (caller frees).
 */
static LinearProgram* linearize_group(const CMLFusionGroup* g) {
    if (!g || g->num_nodes == 0) return NULL;

    LinearProgram* prog = linear_program_create();
    if (!prog) return NULL;

    /* Map: node index -> vreg holding its output (-1 = not yet allocated) */
    int* node_vreg = calloc((size_t)g->num_nodes, sizeof(int));
    if (!node_vreg) { linear_program_free(prog); return NULL; }
    for (int i = 0; i < g->num_nodes; i++) node_vreg[i] = -1;

    /* Map: Tensor pointer -> vreg (for external inputs loaded once) */
    /* Simple linear scan is fine for small groups */
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

        /* Resolve source operands */
        for (int j = 0; j < node->num_inputs && j < 8; j++) {
            Tensor* inp = (node->inputs) ? node->inputs[j] : NULL;
            if (!inp) continue;

            int src_reg = -1;

            /* Check if this tensor is the output of a previous node in group */
            for (int k = 0; k < i; k++) {
                if (g->nodes[k] && g->nodes[k]->output == inp) {
                    src_reg = node_vreg[k];
                    break;
                }
            }

            /* Check if already loaded from external */
            if (src_reg < 0) {
                for (int k = 0; k < num_loaded; k++) {
                    if (loaded_tensors[k] == inp) {
                        src_reg = loaded_vregs[k];
                        break;
                    }
                }
            }

            /* Not found -- emit a LOAD */
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

        /* Allocate dest vreg */
        int dest = alloc_vreg(prog);
        if (dest < 0) break;
        compute_op.dest_reg = dest;
        compute_op.is_eliminated = is_eliminated(g, i);
        node_vreg[i] = dest;

        linear_program_emit(prog, compute_op);

        /* If not eliminated, emit STORE */
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

static const char* linop_name(LinearOpKind k) {
    switch (k) {
        case LINOP_LOAD:    return "LOAD";
        case LINOP_COMPUTE: return "COMPUTE";
        case LINOP_STORE:   return "STORE";
        default:            return "???";
    }
}

static void linear_program_print(const LinearProgram* prog) {
    if (!prog) { printf("LinearProgram: (null)\n"); return; }

    printf("--- Linear Program (%d ops, %d vregs) ---\n",
           prog->num_ops, prog->next_vreg);
    for (int i = 0; i < prog->num_ops; i++) {
        const LinearOp* op = &prog->ops[i];
        printf("  [%d] %-8s  dest=v%d", i, linop_name(op->kind), op->dest_reg);
        if (op->kind == LINOP_COMPUTE) {
            printf("  uop=%s  srcs=[", uop_type_to_string(op->uop));
            for (int j = 0; j < op->num_srcs; j++) {
                if (j > 0) printf(", ");
                printf("v%d", op->src_regs[j]);
            }
            printf("]");
            if (op->is_eliminated) printf("  [ELIM]");
        }
        printf("\n");
    }
    printf("---\n");
}

void cml_linearize_group_print(const CMLFusionGroup* g) {
    if (!g) return;
    LinearProgram* prog = linearize_group(g);
    if (prog) {
        linear_program_print(prog);
        linear_program_free(prog);
    }
}

/**
 * Linearize a fusion group and return the number of instructions generated.
 * Returns -1 on error.
 */
int cml_linearize_group_count(const CMLFusionGroup* g) {
    if (!g) return -1;
    LinearProgram* prog = linearize_group(g);
    if (!prog) return -1;
    int count = prog->num_ops;
    linear_program_free(prog);
    return count;
}
