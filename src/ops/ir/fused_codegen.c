/**
 * @file fused_codegen.c
 * @brief Fused kernel code generation from LinearProgram
 *
 * Walks a LinearProgram and emits fused kernel source for multiple backends.
 * LOAD -> global memory read, COMPUTE -> register arithmetic, STORE -> global write.
 * Eliminated buffers stay in registers (no memory traffic).
 */

#include "ops/ir/fused_codegen.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "core/logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define FUSED_BUF_SIZE 16384

/* ========================================================================
 * Public LinearProgram API (mirrors internal linearize.c but public)
 * ======================================================================== */

CMLLinearProgram* cml_linearize_group(const CMLFusionGroup* g) {
    if (!g || g->num_nodes == 0) return NULL;

    CMLLinearProgram* prog = calloc(1, sizeof(CMLLinearProgram));
    if (!prog) return NULL;
    prog->capacity = 32;
    prog->ops = calloc((size_t)prog->capacity, sizeof(CMLLinearOp));
    if (!prog->ops) { free(prog); return NULL; }

    /* Map node index -> vreg */
    int* node_vreg = calloc((size_t)g->num_nodes, sizeof(int));
    if (!node_vreg) { free(prog->ops); free(prog); return NULL; }
    for (int i = 0; i < g->num_nodes; i++) node_vreg[i] = -1;

    /* Track loaded external tensors */
    Tensor* loaded_tensors[64];
    int loaded_vregs[64];
    int num_loaded = 0;

    for (int i = 0; i < g->num_nodes; i++) {
        struct IRNode* node = g->nodes[i];
        if (!node) continue;

        CMLLinearOp compute_op;
        memset(&compute_op, 0, sizeof(compute_op));
        compute_op.kind = LINOP_COMPUTE;
        compute_op.uop = node->type;
        compute_op.num_srcs = 0;

        /* Resolve source operands */
        for (int j = 0; j < node->num_inputs && j < 8; j++) {
            Tensor* inp = (node->inputs) ? node->inputs[j] : NULL;
            if (!inp) continue;

            int src_reg = -1;

            /* Check previous group nodes */
            for (int k = 0; k < i; k++) {
                if (g->nodes[k] && g->nodes[k]->output == inp) {
                    src_reg = node_vreg[k];
                    break;
                }
            }

            /* Check already loaded */
            if (src_reg < 0) {
                for (int k = 0; k < num_loaded; k++) {
                    if (loaded_tensors[k] == inp) {
                        src_reg = loaded_vregs[k];
                        break;
                    }
                }
            }

            /* Emit LOAD */
            if (src_reg < 0) {
                src_reg = prog->next_vreg++;

                CMLLinearOp load_op;
                memset(&load_op, 0, sizeof(load_op));
                load_op.kind = LINOP_LOAD;
                load_op.dest_reg = src_reg;
                load_op.tensor = inp;

                /* Emit */
                if (prog->num_ops >= prog->capacity) {
                    int nc = prog->capacity * 2;
                    CMLLinearOp* tmp = realloc(prog->ops, (size_t)nc * sizeof(CMLLinearOp));
                    if (!tmp) break;
                    prog->ops = tmp;
                    prog->capacity = nc;
                }
                prog->ops[prog->num_ops++] = load_op;

                if (num_loaded < 64) {
                    loaded_tensors[num_loaded] = inp;
                    loaded_vregs[num_loaded] = src_reg;
                    num_loaded++;
                }
            }

            compute_op.src_regs[compute_op.num_srcs++] = src_reg;
        }

        /* Dest vreg */
        int dest = prog->next_vreg++;
        compute_op.dest_reg = dest;

        /* Check eliminated */
        bool eliminated = false;
        for (int e = 0; e < g->num_eliminated; e++) {
            if (g->eliminated_buffers[e] == i) { eliminated = true; break; }
        }
        compute_op.is_eliminated = eliminated;
        node_vreg[i] = dest;

        /* Emit compute */
        if (prog->num_ops >= prog->capacity) {
            int nc = prog->capacity * 2;
            CMLLinearOp* tmp = realloc(prog->ops, (size_t)nc * sizeof(CMLLinearOp));
            if (!tmp) break;
            prog->ops = tmp;
            prog->capacity = nc;
        }
        prog->ops[prog->num_ops++] = compute_op;

        /* Emit store if not eliminated */
        if (!eliminated) {
            CMLLinearOp store_op;
            memset(&store_op, 0, sizeof(store_op));
            store_op.kind = LINOP_STORE;
            store_op.dest_reg = dest;
            store_op.tensor = node->output;

            if (prog->num_ops >= prog->capacity) {
                int nc = prog->capacity * 2;
                CMLLinearOp* tmp = realloc(prog->ops, (size_t)nc * sizeof(CMLLinearOp));
                if (!tmp) break;
                prog->ops = tmp;
                prog->capacity = nc;
            }
            prog->ops[prog->num_ops++] = store_op;
        }
    }

    free(node_vreg);
    return prog;
}

void cml_linear_program_free(CMLLinearProgram* prog) {
    if (!prog) return;
    free(prog->ops);
    free(prog);
}

static const char* linop_kind_str(CMLLinearOpKind k) {
    switch (k) {
        case LINOP_LOAD:    return "LOAD";
        case LINOP_COMPUTE: return "COMPUTE";
        case LINOP_STORE:   return "STORE";
        default:            return "???";
    }
}

void cml_linear_program_print(const CMLLinearProgram* prog) {
    if (!prog) { printf("LinearProgram: (null)\n"); return; }
    printf("--- CMLLinearProgram (%d ops, %d vregs) ---\n",
           prog->num_ops, prog->next_vreg);
    for (int i = 0; i < prog->num_ops; i++) {
        const CMLLinearOp* op = &prog->ops[i];
        printf("  [%d] %-8s dest=v%d", i, linop_kind_str(op->kind), op->dest_reg);
        if (op->kind == LINOP_COMPUTE) {
            printf("  uop=%d  srcs=[", (int)op->uop);
            for (int j = 0; j < op->num_srcs; j++) {
                if (j > 0) printf(",");
                printf("v%d", op->src_regs[j]);
            }
            printf("]");
            if (op->is_eliminated) printf(" [ELIM]");
        }
        printf("\n");
    }
    printf("---\n");
}

/* ========================================================================
 * UOp -> C expression mapping
 * ======================================================================== */

static const char* uop_to_c_binary(UOpType uop) {
    switch (uop) {
        case UOP_ADD: return "+";
        case UOP_SUB: return "-";
        case UOP_MUL: return "*";
        case UOP_DIV: return "/";
        default: return "+";
    }
}

static bool uop_is_binary(UOpType uop) {
    return uop == UOP_ADD || uop == UOP_SUB || uop == UOP_MUL || uop == UOP_DIV ||
           uop == UOP_MAX || uop == UOP_POW;
}

static bool uop_is_unary(UOpType uop) {
    return uop == UOP_NEG || uop == UOP_EXP || uop == UOP_LOG || uop == UOP_SQRT ||
           uop == UOP_ABS || uop == UOP_SIN || uop == UOP_COS || uop == UOP_TANH ||
           uop == UOP_SIGMOID || uop == UOP_RECIP || uop == UOP_SILU;
}

/* ========================================================================
 * C Backend codegen
 * ======================================================================== */

static char* fused_codegen_c(const CMLLinearProgram* prog, size_t work_size) {
    char* buf = malloc(FUSED_BUF_SIZE);
    if (!buf) return NULL;
    int pos = 0;

    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
        "/* Fused kernel: %d ops, %d vregs */\n"
        "#include <math.h>\n"
        "void fused_kernel(", prog->num_ops, prog->next_vreg);

    /* Count input/output buffers */
    int buf_idx = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].kind == LINOP_LOAD || prog->ops[i].kind == LINOP_STORE) {
            if (buf_idx > 0) pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos, ", ");
            pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                "float* buf%d", buf_idx);
            buf_idx++;
        }
    }
    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
        ", int n) {\n"
        "    for (int i = 0; i < n; i++) {\n");

    /* Emit operations */
    buf_idx = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        const CMLLinearOp* op = &prog->ops[i];
        switch (op->kind) {
        case LINOP_LOAD:
            pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                "        float v%d = buf%d[i];\n", op->dest_reg, buf_idx);
            buf_idx++;
            break;

        case LINOP_COMPUTE:
            if (uop_is_binary(op->uop) && op->num_srcs >= 2) {
                if (op->uop == UOP_MAX) {
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "        float v%d = fmaxf(v%d, v%d);\n",
                        op->dest_reg, op->src_regs[0], op->src_regs[1]);
                } else if (op->uop == UOP_POW) {
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "        float v%d = powf(v%d, v%d);\n",
                        op->dest_reg, op->src_regs[0], op->src_regs[1]);
                } else {
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "        float v%d = v%d %s v%d;\n",
                        op->dest_reg, op->src_regs[0],
                        uop_to_c_binary(op->uop), op->src_regs[1]);
                }
            } else if (uop_is_unary(op->uop) && op->num_srcs >= 1) {
                int s = op->src_regs[0];
                switch (op->uop) {
                case UOP_NEG:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "        float v%d = -v%d;\n", op->dest_reg, s);
                    break;
                case UOP_EXP:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "        float v%d = expf(v%d);\n", op->dest_reg, s);
                    break;
                case UOP_LOG:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "        float v%d = logf(v%d);\n", op->dest_reg, s);
                    break;
                case UOP_SQRT:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "        float v%d = sqrtf(v%d);\n", op->dest_reg, s);
                    break;
                case UOP_ABS:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "        float v%d = fabsf(v%d);\n", op->dest_reg, s);
                    break;
                case UOP_SIN:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "        float v%d = sinf(v%d);\n", op->dest_reg, s);
                    break;
                case UOP_COS:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "        float v%d = cosf(v%d);\n", op->dest_reg, s);
                    break;
                case UOP_TANH:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "        float v%d = tanhf(v%d);\n", op->dest_reg, s);
                    break;
                case UOP_SIGMOID:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "        float v%d = 1.0f / (1.0f + expf(-v%d));\n",
                        op->dest_reg, s);
                    break;
                case UOP_RECIP:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "        float v%d = 1.0f / v%d;\n", op->dest_reg, s);
                    break;
                case UOP_SILU:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "        float v%d = v%d / (1.0f + expf(-v%d));\n",
                        op->dest_reg, s, s);
                    break;
                default:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "        float v%d = v%d; /* unknown unary */\n",
                        op->dest_reg, s);
                    break;
                }
            } else {
                pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                    "        float v%d = 0.0f; /* unsupported op */\n",
                    op->dest_reg);
            }
            break;

        case LINOP_STORE:
            pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                "        buf%d[i] = v%d;\n", buf_idx, op->dest_reg);
            buf_idx++;
            break;
        }
    }

    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
        "    }\n}\n");

    (void)work_size;
    return buf;
}

/* ========================================================================
 * PTX Backend codegen
 * ======================================================================== */

char* cml_ptx_gen_fused_kernel(const CMLLinearProgram* prog, size_t work_size) {
    char* buf = malloc(FUSED_BUF_SIZE);
    if (!buf) return NULL;
    int pos = 0;

    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
        ".version 7.0\n"
        ".target sm_50\n"
        ".address_size 64\n\n");

    /* Count unique buffers */
    int num_loads = 0, num_stores = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].kind == LINOP_LOAD) num_loads++;
        if (prog->ops[i].kind == LINOP_STORE) num_stores++;
    }

    /* Entry point with parameters */
    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
        ".visible .entry fused_kernel(\n");

    int param_idx = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].kind == LINOP_LOAD || prog->ops[i].kind == LINOP_STORE) {
            pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                "    .param .u64 param_buf%d,\n", param_idx++);
        }
    }
    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
        "    .param .u32 param_n\n"
        ") {\n");

    /* Registers */
    int num_vregs = prog->next_vreg;
    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
        "    .reg .pred %%p<4>;\n"
        "    .reg .b32 %%r<16>;\n"
        "    .reg .b64 %%rd<%d>;\n"
        "    .reg .f32 %%f<%d>;\n\n",
        param_idx + 4, num_vregs + 4);

    /* Thread ID + bounds check */
    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
        "    mov.u32 %%r0, %%tid.x;\n"
        "    mov.u32 %%r1, %%ctaid.x;\n"
        "    mov.u32 %%r2, %%ntid.x;\n"
        "    mad.lo.u32 %%r3, %%r1, %%r2, %%r0;\n"
        "    ld.param.u32 %%r4, [param_n];\n"
        "    setp.ge.u32 %%p0, %%r3, %%r4;\n"
        "    @%%p0 ret;\n\n");

    /* Load param pointers and compute addresses */
    param_idx = 0;
    int rd_idx = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].kind == LINOP_LOAD || prog->ops[i].kind == LINOP_STORE) {
            pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                "    ld.param.u64 %%rd%d, [param_buf%d];\n",
                rd_idx, param_idx);
            /* Compute byte offset: gid * 4 */
            pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                "    cvt.u64.u32 %%rd%d, %%r3;\n"
                "    shl.b64 %%rd%d, %%rd%d, 2;\n"
                "    add.u64 %%rd%d, %%rd%d, %%rd%d;\n",
                rd_idx + 1, rd_idx + 1, rd_idx + 1,
                rd_idx, rd_idx, rd_idx + 1);
            param_idx++;
            rd_idx += 2;
        }
    }
    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos, "\n");

    /* Emit operations */
    param_idx = 0;
    rd_idx = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        const CMLLinearOp* op = &prog->ops[i];
        switch (op->kind) {
        case LINOP_LOAD:
            pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                "    ld.global.f32 %%f%d, [%%rd%d];\n",
                op->dest_reg, rd_idx);
            rd_idx += 2;
            param_idx++;
            break;

        case LINOP_COMPUTE:
            if (uop_is_binary(op->uop) && op->num_srcs >= 2) {
                switch (op->uop) {
                case UOP_ADD:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    add.f32 %%f%d, %%f%d, %%f%d;\n",
                        op->dest_reg, op->src_regs[0], op->src_regs[1]);
                    break;
                case UOP_SUB:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    sub.f32 %%f%d, %%f%d, %%f%d;\n",
                        op->dest_reg, op->src_regs[0], op->src_regs[1]);
                    break;
                case UOP_MUL:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    mul.f32 %%f%d, %%f%d, %%f%d;\n",
                        op->dest_reg, op->src_regs[0], op->src_regs[1]);
                    break;
                case UOP_DIV:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    div.approx.f32 %%f%d, %%f%d, %%f%d;\n",
                        op->dest_reg, op->src_regs[0], op->src_regs[1]);
                    break;
                default:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    mov.f32 %%f%d, %%f%d; // unsupported binary\n",
                        op->dest_reg, op->src_regs[0]);
                    break;
                }
            } else if (uop_is_unary(op->uop) && op->num_srcs >= 1) {
                int s = op->src_regs[0];
                switch (op->uop) {
                case UOP_NEG:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    neg.f32 %%f%d, %%f%d;\n", op->dest_reg, s);
                    break;
                case UOP_EXP:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    ex2.approx.f32 %%f%d, %%f%d;\n", op->dest_reg, s);
                    break;
                case UOP_SQRT:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    sqrt.approx.f32 %%f%d, %%f%d;\n", op->dest_reg, s);
                    break;
                case UOP_ABS:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    abs.f32 %%f%d, %%f%d;\n", op->dest_reg, s);
                    break;
                case UOP_RECIP:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    rcp.approx.f32 %%f%d, %%f%d;\n", op->dest_reg, s);
                    break;
                default:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    mov.f32 %%f%d, %%f%d; // unsupported unary\n",
                        op->dest_reg, s);
                    break;
                }
            } else {
                pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                    "    mov.f32 %%f%d, 0f00000000; // nop\n", op->dest_reg);
            }
            break;

        case LINOP_STORE:
            pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                "    st.global.f32 [%%rd%d], %%f%d;\n",
                rd_idx, op->dest_reg);
            rd_idx += 2;
            param_idx++;
            break;
        }
    }

    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
        "    ret;\n}\n");

    (void)work_size;
    return buf;
}

/* ========================================================================
 * SPIR-V Backend codegen (stub - produces placeholder)
 * ======================================================================== */

uint32_t* cml_spirv_gen_fused_kernel(const CMLLinearProgram* prog,
                                      size_t work_size, int* out_num_words) {
    /* SPIR-V binary generation is complex; provide a minimal valid stub */
    if (!prog || !out_num_words) return NULL;

    /* Minimal SPIR-V module that does nothing - real implementation would
     * walk the LinearProgram like the PTX backend does */
    int num_words = 5;  /* Just the header */
    uint32_t* words = calloc((size_t)num_words, sizeof(uint32_t));
    if (!words) return NULL;

    words[0] = 0x07230203;  /* Magic */
    words[1] = 0x00010300;  /* Version 1.3 */
    words[2] = 0;           /* Generator */
    words[3] = 1;           /* Bound */
    words[4] = 0;           /* Schema */

    *out_num_words = num_words;
    (void)work_size;
    LOG_DEBUG("SPIR-V fused codegen: %d ops -> %d words (stub)", prog->num_ops, num_words);
    return words;
}

/* ========================================================================
 * High-level API
 * ======================================================================== */

CMLFusedKernel* cml_fused_codegen(const CMLLinearProgram* prog,
                                    CMLFusedBackend backend,
                                    size_t work_size) {
    if (!prog || prog->num_ops == 0) return NULL;

    CMLFusedKernel* kernel = calloc(1, sizeof(CMLFusedKernel));
    if (!kernel) return NULL;

    kernel->backend = backend;
    kernel->num_vregs = prog->next_vreg;
    kernel->work_size = work_size;

    /* Count inputs and outputs */
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].kind == LINOP_LOAD) kernel->num_inputs++;
        if (prog->ops[i].kind == LINOP_STORE) kernel->num_outputs++;
    }

    switch (backend) {
    case CML_FUSED_BACKEND_C:
        kernel->source = fused_codegen_c(prog, work_size);
        if (!kernel->source) { free(kernel); return NULL; }
        break;

    case CML_FUSED_BACKEND_PTX:
        kernel->source = cml_ptx_gen_fused_kernel(prog, work_size);
        if (!kernel->source) { free(kernel); return NULL; }
        break;

    case CML_FUSED_BACKEND_SPIRV:
        kernel->spirv_words = cml_spirv_gen_fused_kernel(prog, work_size,
                                                          &kernel->spirv_num_words);
        if (!kernel->spirv_words) { free(kernel); return NULL; }
        break;

    default:
        LOG_WARNING("Fused codegen: unsupported backend %d, falling back to C", backend);
        kernel->backend = CML_FUSED_BACKEND_C;
        kernel->source = fused_codegen_c(prog, work_size);
        if (!kernel->source) { free(kernel); return NULL; }
        break;
    }

    return kernel;
}

CMLFusedKernel* cml_fused_codegen_group(const CMLFusionGroup* group,
                                          CMLFusedBackend backend) {
    if (!group) return NULL;

    CMLLinearProgram* prog = cml_linearize_group(group);
    if (!prog) return NULL;

    /* Estimate work size from first node's output */
    size_t work_size = 0;
    if (group->num_nodes > 0 && group->nodes[0] && group->nodes[0]->output) {
        Tensor* t = group->nodes[0]->output;
        work_size = 1;
        for (int i = 0; i < t->ndim; i++) {
            work_size *= (size_t)t->shape[i];
        }
    }

    CMLFusedKernel* kernel = cml_fused_codegen(prog, backend, work_size);
    cml_linear_program_free(prog);
    return kernel;
}

void cml_fused_kernel_free(CMLFusedKernel* kernel) {
    if (!kernel) return;
    free(kernel->source);
    free(kernel->spirv_words);
    free(kernel);
}

void cml_fused_kernel_print(const CMLFusedKernel* kernel) {
    if (!kernel) { printf("FusedKernel: (null)\n"); return; }

    const char* backend_names[] = {"C", "PTX", "SPIR-V", "WGSL", "Metal"};
    printf("=== Fused Kernel (%s) ===\n", backend_names[kernel->backend]);
    printf("  inputs: %d, outputs: %d, vregs: %d, work_size: %zu\n",
           kernel->num_inputs, kernel->num_outputs,
           kernel->num_vregs, kernel->work_size);

    if (kernel->source) {
        printf("--- Source ---\n%s\n", kernel->source);
    }
    if (kernel->spirv_words) {
        printf("  SPIR-V: %d words\n", kernel->spirv_num_words);
    }
    printf("===\n");
}
