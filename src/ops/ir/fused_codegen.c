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

/*
 * SPIR-V Backend codegen
 *
 * Generates a valid SPIR-V 1.3 compute shader that implements the fused
 * kernel.  Each LOAD reads from a storage buffer, each COMPUTE performs
 * arithmetic, and each STORE writes to a storage buffer.
 */

/* SPIR-V opcode helpers */
#define SPIRV_OP(opcode, word_count) (((uint32_t)(word_count) << 16) | (uint32_t)(opcode))

/* Common opcodes */
#define SpvOpCapability         17
#define SpvOpExtInstImport      11
#define SpvOpMemoryModel        14
#define SpvOpEntryPoint         15
#define SpvOpExecutionMode      16
#define SpvOpDecorate           71
#define SpvOpMemberDecorate     72
#define SpvOpTypeVoid            19
#define SpvOpTypeBool            20
#define SpvOpTypeInt             21
#define SpvOpTypeFloat           22
#define SpvOpTypeVector          23
#define SpvOpTypeArray           28
#define SpvOpTypeRuntimeArray    29
#define SpvOpTypeStruct          30
#define SpvOpTypePointer         32
#define SpvOpTypeFunction        33
#define SpvOpConstant            43
#define SpvOpFunction            54
#define SpvOpFunctionEnd         56
#define SpvOpLabel               248
#define SpvOpReturn              253
#define SpvOpVariable            59
#define SpvOpLoad                61
#define SpvOpStore               62
#define SpvOpAccessChain         65
#define SpvOpFAdd               129
#define SpvOpFSub               131
#define SpvOpFMul               133
#define SpvOpFDiv               136
#define SpvOpFNegate             127
#define SpvOpCompositeExtract    81

/* Decoration values */
#define SpvDecorationBinding     33
#define SpvDecorationDescriptorSet 34
#define SpvDecorationOffset      35
#define SpvDecorationArrayStride 6
#define SpvDecorationBlock       2
#define SpvDecorationBuiltIn     11

/* BuiltIn values */
#define SpvBuiltInGlobalInvocationId 28

/* Storage classes */
#define SpvStorageClassUniform          2
#define SpvStorageClassInput            1
#define SpvStorageClassFunction         7
#define SpvStorageClassStorageBuffer    12

/* Capability values */
#define SpvCapabilityShader      1

/* Execution model / mode */
#define SpvExecutionModelGLCompute 5
#define SpvExecutionModeLocalSize  17

/* Addressing / Memory model */
#define SpvAddressingModelLogical 0
#define SpvMemoryModelGLSL450    1

/* ExtInst GLSL.std.450 opcodes */
#define GLSL_STD_450_Exp    27
#define GLSL_STD_450_Log    28
#define GLSL_STD_450_Sqrt   31
#define GLSL_STD_450_FAbs   4
#define GLSL_STD_450_Sin    13
#define GLSL_STD_450_Cos    14
#define GLSL_STD_450_Tanh   21

#define SpvOpExtInst 12

static void emit(uint32_t** ptr, uint32_t word) {
    **ptr = word;
    (*ptr)++;
}

uint32_t* cml_spirv_gen_fused_kernel(const CMLLinearProgram* prog,
                                      size_t work_size, int* out_num_words) {
    if (!prog || !out_num_words) return NULL;

    /* Count buffers (loads + stores) */
    int num_buffers = 0;
    int num_loads = 0, num_stores = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].kind == LINOP_LOAD) { num_buffers++; num_loads++; }
        if (prog->ops[i].kind == LINOP_STORE) { num_buffers++; num_stores++; }
    }

    /* Allocate generous buffer for SPIR-V words */
    int max_words = 1024 + prog->num_ops * 32 + num_buffers * 64;
    uint32_t* words = calloc((size_t)max_words, sizeof(uint32_t));
    if (!words) return NULL;

    uint32_t* w = words;

    /* Reserve header (5 words) - filled at end */
    uint32_t* header = w;
    w += 5;

    /* ID allocation: start from 1 */
    uint32_t next_id = 1;

    /* Pre-allocate key type/variable IDs */
    uint32_t id_void_type    = next_id++;  /* 1 */
    uint32_t id_float_type   = next_id++;  /* 2 */
    uint32_t id_uint_type    = next_id++;  /* 3 */
    uint32_t id_uint3_type   = next_id++;  /* 4 */
    uint32_t id_func_type    = next_id++;  /* 5 */
    uint32_t id_rt_array     = next_id++;  /* 6 - RuntimeArray of float */
    uint32_t id_struct_type  = next_id++;  /* 7 - struct { RuntimeArray } */
    uint32_t id_ptr_sb       = next_id++;  /* 8 - pointer to struct (StorageBuffer) */
    uint32_t id_ptr_sb_float = next_id++;  /* 9 - pointer to float (StorageBuffer) */
    uint32_t id_ptr_input_uint3 = next_id++; /* 10 */
    uint32_t id_gl_invoc     = next_id++;  /* 11 - gl_GlobalInvocationID */
    uint32_t id_uint_zero    = next_id++;  /* 12 */
    uint32_t id_main         = next_id++;  /* 13 */
    uint32_t id_label        = next_id++;  /* 14 */
    uint32_t id_ext_glsl     = next_id++;  /* 15 */

    /* Allocate IDs for buffer variables */
    uint32_t buf_var_ids[64];
    for (int i = 0; i < num_buffers && i < 64; i++) {
        buf_var_ids[i] = next_id++;
    }

    /* Allocate IDs for computation results */
    uint32_t load_gid_id = next_id++;
    uint32_t gid_x_id    = next_id++;
    /* Virtual register IDs for each linear op */
    uint32_t vreg_base = next_id;
    next_id += (uint32_t)(prog->next_vreg + prog->num_ops * 4 + 16);

    emit(&w, SPIRV_OP(SpvOpCapability, 2));
    emit(&w, SpvCapabilityShader);

    /* "GLSL.std.450" = 12 chars + null = 13 bytes = 4 words */
    emit(&w, SPIRV_OP(SpvOpExtInstImport, 6));
    emit(&w, id_ext_glsl);
    emit(&w, 0x534C4C47); /* GLSL */
    emit(&w, 0x6474732E); /* .std */
    emit(&w, 0x3035342E); /* .450 */
    emit(&w, 0x00000000); /* null terminator */

    emit(&w, SPIRV_OP(SpvOpMemoryModel, 3));
    emit(&w, SpvAddressingModelLogical);
    emit(&w, SpvMemoryModelGLSL450);

    /* "main" = 4 chars + null = 5 bytes = 2 words */
    emit(&w, SPIRV_OP(SpvOpEntryPoint, 6));
    emit(&w, SpvExecutionModelGLCompute);
    emit(&w, id_main);
    emit(&w, 0x6E69616D); /* main */
    emit(&w, 0x00000000);
    emit(&w, id_gl_invoc);

    emit(&w, SPIRV_OP(SpvOpExecutionMode, 6));
    emit(&w, id_main);
    emit(&w, SpvExecutionModeLocalSize);
    emit(&w, 256);
    emit(&w, 1);
    emit(&w, 1);

    /* RuntimeArray stride */
    emit(&w, SPIRV_OP(SpvOpDecorate, 4));
    emit(&w, id_rt_array);
    emit(&w, SpvDecorationArrayStride);
    emit(&w, 4); /* sizeof(float) */

    /* Struct Block decoration */
    emit(&w, SPIRV_OP(SpvOpDecorate, 3));
    emit(&w, id_struct_type);
    emit(&w, SpvDecorationBlock);

    /* MemberDecorate offset 0 */
    emit(&w, SPIRV_OP(SpvOpMemberDecorate, 5));
    emit(&w, id_struct_type);
    emit(&w, 0);
    emit(&w, SpvDecorationOffset);
    emit(&w, 0);

    /* Buffer variable bindings */
    for (int i = 0; i < num_buffers && i < 64; i++) {
        emit(&w, SPIRV_OP(SpvOpDecorate, 4));
        emit(&w, buf_var_ids[i]);
        emit(&w, SpvDecorationDescriptorSet);
        emit(&w, 0);

        emit(&w, SPIRV_OP(SpvOpDecorate, 4));
        emit(&w, buf_var_ids[i]);
        emit(&w, SpvDecorationBinding);
        emit(&w, (uint32_t)i);
    }

    /* BuiltIn decoration for gl_GlobalInvocationID */
    emit(&w, SPIRV_OP(SpvOpDecorate, 4));
    emit(&w, id_gl_invoc);
    emit(&w, SpvDecorationBuiltIn);
    emit(&w, SpvBuiltInGlobalInvocationId);

    emit(&w, SPIRV_OP(SpvOpTypeVoid, 2));
    emit(&w, id_void_type);

    emit(&w, SPIRV_OP(SpvOpTypeFloat, 3));
    emit(&w, id_float_type);
    emit(&w, 32);

    emit(&w, SPIRV_OP(SpvOpTypeInt, 4));
    emit(&w, id_uint_type);
    emit(&w, 32);
    emit(&w, 0); /* unsigned */

    emit(&w, SPIRV_OP(SpvOpTypeVector, 4));
    emit(&w, id_uint3_type);
    emit(&w, id_uint_type);
    emit(&w, 3);

    emit(&w, SPIRV_OP(SpvOpTypeFunction, 3));
    emit(&w, id_func_type);
    emit(&w, id_void_type);

    emit(&w, SPIRV_OP(SpvOpTypeRuntimeArray, 3));
    emit(&w, id_rt_array);
    emit(&w, id_float_type);

    emit(&w, SPIRV_OP(SpvOpTypeStruct, 3));
    emit(&w, id_struct_type);
    emit(&w, id_rt_array);

    emit(&w, SPIRV_OP(SpvOpTypePointer, 4));
    emit(&w, id_ptr_sb);
    emit(&w, SpvStorageClassStorageBuffer);
    emit(&w, id_struct_type);

    emit(&w, SPIRV_OP(SpvOpTypePointer, 4));
    emit(&w, id_ptr_sb_float);
    emit(&w, SpvStorageClassStorageBuffer);
    emit(&w, id_float_type);

    emit(&w, SPIRV_OP(SpvOpTypePointer, 4));
    emit(&w, id_ptr_input_uint3);
    emit(&w, SpvStorageClassInput);
    emit(&w, id_uint3_type);

    emit(&w, SPIRV_OP(SpvOpConstant, 4));
    emit(&w, id_uint_type);
    emit(&w, id_uint_zero);
    emit(&w, 0);

    for (int i = 0; i < num_buffers && i < 64; i++) {
        emit(&w, SPIRV_OP(SpvOpVariable, 4));
        emit(&w, id_ptr_sb);
        emit(&w, buf_var_ids[i]);
        emit(&w, SpvStorageClassStorageBuffer);
    }

    emit(&w, SPIRV_OP(SpvOpVariable, 4));
    emit(&w, id_ptr_input_uint3);
    emit(&w, id_gl_invoc);
    emit(&w, SpvStorageClassInput);

    emit(&w, SPIRV_OP(SpvOpFunction, 5));
    emit(&w, id_void_type);
    emit(&w, id_main);
    emit(&w, 0); /* None */
    emit(&w, id_func_type);

    emit(&w, SPIRV_OP(SpvOpLabel, 2));
    emit(&w, id_label);

    /* Load GlobalInvocationID.x */
    emit(&w, SPIRV_OP(SpvOpLoad, 4));
    emit(&w, id_uint3_type);
    emit(&w, load_gid_id);
    emit(&w, id_gl_invoc);

    emit(&w, SPIRV_OP(SpvOpCompositeExtract, 5));
    emit(&w, id_uint_type);
    emit(&w, gid_x_id);
    emit(&w, load_gid_id);
    emit(&w, 0); /* component 0 = x */

    /* Emit load/compute/store operations */
    int buf_idx = 0;
    uint32_t vid = vreg_base;

    for (int i = 0; i < prog->num_ops; i++) {
        const CMLLinearOp* op = &prog->ops[i];

        switch (op->kind) {
        case LINOP_LOAD: {
            /* AccessChain + Load from buffer */
            uint32_t chain_id = vid++;
            uint32_t load_id = vid++;

            emit(&w, SPIRV_OP(SpvOpAccessChain, 6));
            emit(&w, id_ptr_sb_float);
            emit(&w, chain_id);
            emit(&w, buf_var_ids[buf_idx]);
            emit(&w, id_uint_zero);
            emit(&w, gid_x_id);

            emit(&w, SPIRV_OP(SpvOpLoad, 4));
            emit(&w, id_float_type);
            emit(&w, load_id);
            emit(&w, chain_id);

            /* Map dest_reg to this load_id */
            /* We use a simple offset: vreg_base + dest_reg maps to load_id */
            buf_idx++;
            break;
        }

        case LINOP_COMPUTE: {
            uint32_t result_id = vid++;
            if (uop_is_binary(op->uop) && op->num_srcs >= 2) {
                uint32_t src0 = vreg_base + (uint32_t)op->src_regs[0];
                uint32_t src1 = vreg_base + (uint32_t)op->src_regs[1];
                uint32_t spirv_op;
                switch (op->uop) {
                    case UOP_ADD: spirv_op = SpvOpFAdd; break;
                    case UOP_SUB: spirv_op = SpvOpFSub; break;
                    case UOP_MUL: spirv_op = SpvOpFMul; break;
                    case UOP_DIV: spirv_op = SpvOpFDiv; break;
                    default:      spirv_op = SpvOpFAdd; break;
                }
                emit(&w, SPIRV_OP(spirv_op, 5));
                emit(&w, id_float_type);
                emit(&w, result_id);
                emit(&w, src0);
                emit(&w, src1);
            } else if (uop_is_unary(op->uop) && op->num_srcs >= 1) {
                uint32_t src = vreg_base + (uint32_t)op->src_regs[0];
                if (op->uop == UOP_NEG) {
                    emit(&w, SPIRV_OP(SpvOpFNegate, 4));
                    emit(&w, id_float_type);
                    emit(&w, result_id);
                    emit(&w, src);
                } else {
                    /* Use GLSL.std.450 extended instructions */
                    uint32_t glsl_op;
                    switch (op->uop) {
                        case UOP_EXP:  glsl_op = GLSL_STD_450_Exp; break;
                        case UOP_LOG:  glsl_op = GLSL_STD_450_Log; break;
                        case UOP_SQRT: glsl_op = GLSL_STD_450_Sqrt; break;
                        case UOP_ABS:  glsl_op = GLSL_STD_450_FAbs; break;
                        case UOP_SIN:  glsl_op = GLSL_STD_450_Sin; break;
                        case UOP_COS:  glsl_op = GLSL_STD_450_Cos; break;
                        case UOP_TANH: glsl_op = GLSL_STD_450_Tanh; break;
                        default:       glsl_op = GLSL_STD_450_FAbs; break;
                    }
                    emit(&w, SPIRV_OP(SpvOpExtInst, 6));
                    emit(&w, id_float_type);
                    emit(&w, result_id);
                    emit(&w, id_ext_glsl);
                    emit(&w, glsl_op);
                    emit(&w, src);
                }
            } else {
                /* No-op: just copy input */
                uint32_t src = (op->num_srcs > 0) ? vreg_base + (uint32_t)op->src_regs[0] : id_uint_zero;
                emit(&w, SPIRV_OP(SpvOpFAdd, 5));
                emit(&w, id_float_type);
                emit(&w, result_id);
                emit(&w, src);
                emit(&w, src);
            }
            break;
        }

        case LINOP_STORE: {
            uint32_t chain_id = vid++;
            uint32_t src = vreg_base + (uint32_t)op->dest_reg;

            emit(&w, SPIRV_OP(SpvOpAccessChain, 6));
            emit(&w, id_ptr_sb_float);
            emit(&w, chain_id);
            emit(&w, buf_var_ids[buf_idx]);
            emit(&w, id_uint_zero);
            emit(&w, gid_x_id);

            emit(&w, SPIRV_OP(SpvOpStore, 3));
            emit(&w, chain_id);
            emit(&w, src);

            buf_idx++;
            break;
        }
        }
    }

    /* Return + FunctionEnd */
    emit(&w, SPIRV_OP(SpvOpReturn, 1));
    emit(&w, SPIRV_OP(SpvOpFunctionEnd, 1));

    /* Fill in header */
    int total_words = (int)(w - words);
    header[0] = 0x07230203;  /* Magic number */
    header[1] = 0x00010300;  /* Version 1.3 */
    header[2] = 0x00434D4C;  /* Generator: "CML" */
    header[3] = next_id;     /* Bound */
    header[4] = 0;           /* Schema */

    *out_num_words = total_words;
    (void)work_size;
    LOG_DEBUG("SPIR-V fused codegen: %d ops -> %d words", prog->num_ops, total_words);
    return words;
}

static char* fused_codegen_wgsl(const CMLLinearProgram* prog, size_t work_size) {
    char* buf = malloc(FUSED_BUF_SIZE);
    if (!buf) return NULL;
    int pos = 0;

    /* Emit storage buffer bindings */
    int buf_idx = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].kind == LINOP_LOAD || prog->ops[i].kind == LINOP_STORE) {
            pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                "@group(0) @binding(%d) var<storage, read_write> buf%d: array<f32>;\n",
                buf_idx, buf_idx);
            buf_idx++;
        }
    }

    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
        "\n@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    let i = gid.x;\n");

    /* Emit operations */
    buf_idx = 0;
    for (int i = 0; i < prog->num_ops; i++) {
        const CMLLinearOp* op = &prog->ops[i];
        switch (op->kind) {
        case LINOP_LOAD:
            pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                "    let v%d = buf%d[i];\n", op->dest_reg, buf_idx);
            buf_idx++;
            break;

        case LINOP_COMPUTE:
            if (uop_is_binary(op->uop) && op->num_srcs >= 2) {
                const char* wgsl_op;
                switch (op->uop) {
                    case UOP_ADD: wgsl_op = "+"; break;
                    case UOP_SUB: wgsl_op = "-"; break;
                    case UOP_MUL: wgsl_op = "*"; break;
                    case UOP_DIV: wgsl_op = "/"; break;
                    default:      wgsl_op = "+"; break;
                }
                if (op->uop == UOP_MAX) {
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    let v%d = max(v%d, v%d);\n",
                        op->dest_reg, op->src_regs[0], op->src_regs[1]);
                } else if (op->uop == UOP_POW) {
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    let v%d = pow(v%d, v%d);\n",
                        op->dest_reg, op->src_regs[0], op->src_regs[1]);
                } else {
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    let v%d = v%d %s v%d;\n",
                        op->dest_reg, op->src_regs[0], wgsl_op, op->src_regs[1]);
                }
            } else if (uop_is_unary(op->uop) && op->num_srcs >= 1) {
                int s = op->src_regs[0];
                switch (op->uop) {
                case UOP_NEG:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    let v%d = -v%d;\n", op->dest_reg, s);
                    break;
                case UOP_EXP:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    let v%d = exp(v%d);\n", op->dest_reg, s);
                    break;
                case UOP_LOG:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    let v%d = log(v%d);\n", op->dest_reg, s);
                    break;
                case UOP_SQRT:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    let v%d = sqrt(v%d);\n", op->dest_reg, s);
                    break;
                case UOP_ABS:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    let v%d = abs(v%d);\n", op->dest_reg, s);
                    break;
                case UOP_SIN:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    let v%d = sin(v%d);\n", op->dest_reg, s);
                    break;
                case UOP_COS:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    let v%d = cos(v%d);\n", op->dest_reg, s);
                    break;
                case UOP_TANH:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    let v%d = tanh(v%d);\n", op->dest_reg, s);
                    break;
                case UOP_SIGMOID:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    let v%d = 1.0 / (1.0 + exp(-v%d));\n",
                        op->dest_reg, s);
                    break;
                case UOP_RECIP:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    let v%d = 1.0 / v%d;\n", op->dest_reg, s);
                    break;
                case UOP_SILU:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    let v%d = v%d / (1.0 + exp(-v%d));\n",
                        op->dest_reg, s, s);
                    break;
                default:
                    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                        "    let v%d = v%d;\n", op->dest_reg, s);
                    break;
                }
            } else {
                pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                    "    let v%d = 0.0;\n", op->dest_reg);
            }
            break;

        case LINOP_STORE:
            pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos,
                "    buf%d[i] = v%d;\n", buf_idx, op->dest_reg);
            buf_idx++;
            break;
        }
    }

    pos += snprintf(buf + pos, FUSED_BUF_SIZE - pos, "}\n");

    (void)work_size;
    return buf;
}

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

    case CML_FUSED_BACKEND_WGSL:
        kernel->source = fused_codegen_wgsl(prog, work_size);
        if (!kernel->source) { free(kernel); return NULL; }
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
