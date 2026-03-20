#include "ops/ir/late_passes.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_WORKGROUP_EXTENT 256

static int ensure_capacity(LinearProgram* prog, int needed) {
    if (prog->num_ops + needed <= prog->capacity) return 0;
    int nc = prog->capacity;
    while (nc < prog->num_ops + needed) nc *= 2;
    LinearOp* tmp = realloc(prog->ops, (size_t)nc * sizeof(LinearOp));
    if (!tmp) return -1;
    prog->ops = tmp;
    prog->capacity = nc;
    return 0;
}

static void remap_srcs(LinearOp* op, int old_reg, int new_reg) {
    for (int s = 0; s < op->num_srcs; s++) {
        if (op->src_regs[s] == old_reg)
            op->src_regs[s] = new_reg;
    }
}

int cml_devectorize(struct LinearProgram* prog) {
    if (!prog) return -1;

    int i = 0;
    while (i < prog->num_ops) {
        LinearOp* op = &prog->ops[i];
        int vw = op->vec_width;
        if (vw <= 1) { i++; continue; }

        int expanded = vw;

        if (op->kind == LINOP_LOAD) {
            if (ensure_capacity(prog, expanded - 1) != 0) return -1;

            int orig_reg = op->dest_reg;
            int first_reg = alloc_vreg(prog);
            if (first_reg < 0) return -1;

            LinearOp scalar0 = *op;
            scalar0.vec_width = 1;
            scalar0.dest_reg = first_reg;
            prog->ops[i] = scalar0;

            for (int lane = 1; lane < expanded; lane++) {
                memmove(&prog->ops[i + lane + 1],
                        &prog->ops[i + lane],
                        (size_t)(prog->num_ops - i - lane) * sizeof(LinearOp));
                prog->num_ops++;

                int r = alloc_vreg(prog);
                if (r < 0) return -1;

                LinearOp scalar_load = scalar0;
                scalar_load.dest_reg = r;
                prog->ops[i + lane] = scalar_load;
            }

            for (int j = i + expanded; j < prog->num_ops; j++) {
                remap_srcs(&prog->ops[j], orig_reg, first_reg);
            }

            i += expanded;

        } else if (op->kind == LINOP_COMPUTE) {
            if (ensure_capacity(prog, expanded - 1) != 0) return -1;

            int orig_reg = op->dest_reg;
            int first_reg = alloc_vreg(prog);
            if (first_reg < 0) return -1;

            LinearOp scalar0 = *op;
            scalar0.vec_width = 1;
            scalar0.dest_reg = first_reg;
            prog->ops[i] = scalar0;

            for (int lane = 1; lane < expanded; lane++) {
                memmove(&prog->ops[i + lane + 1],
                        &prog->ops[i + lane],
                        (size_t)(prog->num_ops - i - lane) * sizeof(LinearOp));
                prog->num_ops++;

                int r = alloc_vreg(prog);
                if (r < 0) return -1;

                LinearOp scalar_comp = scalar0;
                scalar_comp.dest_reg = r;
                prog->ops[i + lane] = scalar_comp;
            }

            for (int j = i + expanded; j < prog->num_ops; j++) {
                remap_srcs(&prog->ops[j], orig_reg, first_reg);
                if (prog->ops[j].kind == LINOP_STORE &&
                    prog->ops[j].dest_reg == orig_reg) {
                    prog->ops[j].dest_reg = first_reg;
                }
            }

            i += expanded;

        } else if (op->kind == LINOP_STORE) {
            if (ensure_capacity(prog, expanded - 1) != 0) return -1;

            LinearOp scalar0 = *op;
            scalar0.vec_width = 1;
            prog->ops[i] = scalar0;

            for (int lane = 1; lane < expanded; lane++) {
                memmove(&prog->ops[i + lane + 1],
                        &prog->ops[i + lane],
                        (size_t)(prog->num_ops - i - lane) * sizeof(LinearOp));
                prog->num_ops++;

                LinearOp scalar_store = scalar0;
                prog->ops[i + lane] = scalar_store;
            }

            i += expanded;
        } else {
            i++;
        }
    }

    return 0;
}

int cml_expand_groups(struct LinearProgram* prog) {
    if (!prog) return -1;

    int i = 0;
    while (i < prog->num_ops) {
        LinearOp* op = &prog->ops[i];

        if (op->kind == LINOP_LOOP && op->loop_extent > MAX_WORKGROUP_EXTENT) {
            int extent = op->loop_extent;
            int chunk = MAX_WORKGROUP_EXTENT;
            int num_chunks = (extent + chunk - 1) / chunk;

            if (num_chunks <= 1) { i++; continue; }
            if (ensure_capacity(prog, num_chunks - 1) != 0) return -1;

            int endloop_idx = -1;
            int depth = 0;
            for (int j = i + 1; j < prog->num_ops; j++) {
                if (prog->ops[j].kind == LINOP_LOOP) depth++;
                else if (prog->ops[j].kind == LINOP_ENDLOOP) {
                    if (depth == 0) { endloop_idx = j; break; }
                    depth--;
                }
            }

            if (endloop_idx < 0) { i++; continue; }

            int body_len = endloop_idx - i - 1;
            LinearOp* body = NULL;
            if (body_len > 0) {
                body = malloc((size_t)body_len * sizeof(LinearOp));
                if (!body) return -1;
                memcpy(body, &prog->ops[i + 1], (size_t)body_len * sizeof(LinearOp));
            }

            int total_new = num_chunks * (body_len + 2);
            int old_block = endloop_idx - i + 1;
            int delta = total_new - old_block;

            if (delta > 0) {
                if (ensure_capacity(prog, delta) != 0) { free(body); return -1; }
                memmove(&prog->ops[i + total_new],
                        &prog->ops[endloop_idx + 1],
                        (size_t)(prog->num_ops - endloop_idx - 1) * sizeof(LinearOp));
            } else if (delta < 0) {
                memmove(&prog->ops[i + total_new],
                        &prog->ops[endloop_idx + 1],
                        (size_t)(prog->num_ops - endloop_idx - 1) * sizeof(LinearOp));
            }
            prog->num_ops += delta;

            int pos = i;
            int remaining = extent;
            for (int c = 0; c < num_chunks; c++) {
                int this_extent = remaining < chunk ? remaining : chunk;
                remaining -= this_extent;

                LinearOp loop_hdr = *op;
                loop_hdr.loop_extent = this_extent;
                prog->ops[pos++] = loop_hdr;

                if (body_len > 0) {
                    memcpy(&prog->ops[pos], body, (size_t)body_len * sizeof(LinearOp));
                    pos += body_len;
                }

                LinearOp endloop;
                memset(&endloop, 0, sizeof(endloop));
                endloop.kind = LINOP_ENDLOOP;
                endloop.loop_axis = op->loop_axis;
                prog->ops[pos++] = endloop;
            }

            free(body);
            i = pos;
        } else {
            i++;
        }
    }

    return 0;
}

int cml_late_lower(struct LinearProgram* prog) {
    if (!prog) return -1;
    int rc = cml_devectorize(prog);
    if (rc != 0) return rc;
    return cml_expand_groups(prog);
}
