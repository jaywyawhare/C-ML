/**
 * @file ptx_codegen.c
 * @brief Standalone PTX assembly code generation (no LLVM dependency)
 *
 * Direct PTX assembly text generation via snprintf into dynamic buffers.
 * Uses IEEE 754 hex float format (0fHHHHHHHH) for float literals.
 */

#include "ops/ir/gpu/ptx_codegen.h"
#include "core/logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Maximum PTX buffer size per kernel
#define PTX_BUF_SIZE 8192

// ============================================================================
// Internal helpers
// ============================================================================

/** Convert float to PTX hex literal (0fHHHHHHHH) */
static void float_to_ptx_hex(float val, char* buf, int buf_size) {
    uint32_t bits;
    memcpy(&bits, &val, sizeof(bits));
    snprintf(buf, (size_t)buf_size, "0f%08X", bits);
}

/** Write PTX header: version, target, address_size */
static int ptx_write_header(char* buf, int buf_size, int sm_version) {
    return snprintf(buf, (size_t)buf_size,
        ".version 7.0\n"
        ".target sm_%d\n"
        ".address_size 64\n\n",
        sm_version);
}

/**
 * Write standard kernel prologue: entry point, register decls, thread ID calculation,
 * bounds check. Uses these registers:
 *   %%r0 = tid.x, %%r1 = ctaid.x, %%r2 = ntid.x, %%r3 = gid
 *   %%r4 = n (loaded from param), %%p0 = bounds predicate
 */
static int ptx_write_prologue_unary(char* buf, int buf_size, const char* kernel_name) {
    return snprintf(buf, (size_t)buf_size,
        ".visible .entry %s(\n"
        "    .param .u64 param_in,\n"
        "    .param .u64 param_out,\n"
        "    .param .u32 param_n\n"
        ") {\n"
        "    .reg .pred %%p<4>;\n"
        "    .reg .b32 %%r<16>;\n"
        "    .reg .b64 %%rd<8>;\n"
        "    .reg .f32 %%f<8>;\n\n"
        "    mov.u32 %%r0, %%tid.x;\n"
        "    mov.u32 %%r1, %%ctaid.x;\n"
        "    mov.u32 %%r2, %%ntid.x;\n"
        "    mad.lo.u32 %%r3, %%r1, %%r2, %%r0;\n"
        "    ld.param.u32 %%r4, [param_n];\n"
        "    setp.ge.u32 %%p0, %%r3, %%r4;\n"
        "    @%%p0 ret;\n\n",
        kernel_name);
}

/** Prologue for binary kernels (two input pointers) */
static int ptx_write_prologue_binary(char* buf, int buf_size, const char* kernel_name) {
    return snprintf(buf, (size_t)buf_size,
        ".visible .entry %s(\n"
        "    .param .u64 param_a,\n"
        "    .param .u64 param_b,\n"
        "    .param .u64 param_out,\n"
        "    .param .u32 param_n\n"
        ") {\n"
        "    .reg .pred %%p<4>;\n"
        "    .reg .b32 %%r<16>;\n"
        "    .reg .b64 %%rd<12>;\n"
        "    .reg .f32 %%f<12>;\n\n"
        "    mov.u32 %%r0, %%tid.x;\n"
        "    mov.u32 %%r1, %%ctaid.x;\n"
        "    mov.u32 %%r2, %%ntid.x;\n"
        "    mad.lo.u32 %%r3, %%r1, %%r2, %%r0;\n"
        "    ld.param.u32 %%r4, [param_n];\n"
        "    setp.ge.u32 %%p0, %%r3, %%r4;\n"
        "    @%%p0 ret;\n\n",
        kernel_name);
}

/** Load element from pointer at gid offset (unary pattern) */
static int ptx_write_load_unary(char* buf, int buf_size) {
    return snprintf(buf, (size_t)buf_size,
        "    // Load input\n"
        "    ld.param.u64 %%rd0, [param_in];\n"
        "    cvt.u64.u32 %%rd1, %%r3;\n"
        "    shl.b64 %%rd2, %%rd1, 2;\n"
        "    add.u64 %%rd3, %%rd0, %%rd2;\n"
        "    ld.global.f32 %%f0, [%%rd3];\n\n");
}

/** Store result to output pointer at gid offset (unary pattern) */
static int ptx_write_store_unary(char* buf, int buf_size) {
    return snprintf(buf, (size_t)buf_size,
        "    // Store output\n"
        "    ld.param.u64 %%rd4, [param_out];\n"
        "    add.u64 %%rd5, %%rd4, %%rd2;\n"
        "    st.global.f32 [%%rd5], %%f1;\n"
        "    ret;\n"
        "}\n");
}

/** Load two elements for binary ops */
static int ptx_write_load_binary(char* buf, int buf_size) {
    return snprintf(buf, (size_t)buf_size,
        "    // Load inputs\n"
        "    ld.param.u64 %%rd0, [param_a];\n"
        "    ld.param.u64 %%rd1, [param_b];\n"
        "    cvt.u64.u32 %%rd2, %%r3;\n"
        "    shl.b64 %%rd3, %%rd2, 2;\n"
        "    add.u64 %%rd4, %%rd0, %%rd3;\n"
        "    add.u64 %%rd5, %%rd1, %%rd3;\n"
        "    ld.global.f32 %%f0, [%%rd4];\n"
        "    ld.global.f32 %%f1, [%%rd5];\n\n");
}

/** Store result for binary ops */
static int ptx_write_store_binary(char* buf, int buf_size) {
    return snprintf(buf, (size_t)buf_size,
        "    // Store output\n"
        "    ld.param.u64 %%rd6, [param_out];\n"
        "    add.u64 %%rd7, %%rd6, %%rd3;\n"
        "    st.global.f32 [%%rd7], %%f2;\n"
        "    ret;\n"
        "}\n");
}

// ============================================================================
// Public API
// ============================================================================

CMLPTXCodegen* cml_ptx_codegen_create(int sm_version) {
    CMLPTXCodegen* cg = (CMLPTXCodegen*)calloc(1, sizeof(CMLPTXCodegen));
    if (!cg) return NULL;
    cg->sm_version = sm_version > 0 ? sm_version : 50;
    cg->kernel_count = 0;
    cg->initialized = true;
    return cg;
}

void cml_ptx_codegen_destroy(CMLPTXCodegen* cg) {
    free(cg);
}

char* cml_ptx_gen_unary(CMLPTXCodegen* cg, UOpType op, const char* kernel_name) {
    if (!cg || !kernel_name) return NULL;

    char* ptx = (char*)malloc(PTX_BUF_SIZE);
    if (!ptx) return NULL;

    int pos = 0;
    pos += ptx_write_header(ptx + pos, PTX_BUF_SIZE - pos, cg->sm_version);
    pos += ptx_write_prologue_unary(ptx + pos, PTX_BUF_SIZE - pos, kernel_name);
    pos += ptx_write_load_unary(ptx + pos, PTX_BUF_SIZE - pos);

    // Emit the operation: input is %f0, output is %f1
    pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos), "    // Compute\n");

    switch (op) {
    case UOP_NEG:
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    neg.f32 %%f1, %%f0;\n\n");
        break;
    case UOP_EXP:
        // exp(x) = exp2(x * log2(e))
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    mul.f32 %%f1, %%f0, 0f3FB8AA3B;\n"  // log2(e) = 1.4426950f
            "    ex2.approx.f32 %%f1, %%f1;\n\n");
        break;
    case UOP_LOG:
        // log(x) = log2(x) * ln(2)
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    lg2.approx.f32 %%f1, %%f0;\n"
            "    mul.f32 %%f1, %%f1, 0f3F317218;\n\n");  // ln(2) = 0.6931472f
        break;
    case UOP_SQRT:
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    sqrt.approx.f32 %%f1, %%f0;\n\n");
        break;
    case UOP_ABS:
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    abs.f32 %%f1, %%f0;\n\n");
        break;
    case UOP_SIN:
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    sin.approx.f32 %%f1, %%f0;\n\n");
        break;
    case UOP_COS:
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    cos.approx.f32 %%f1, %%f0;\n\n");
        break;
    case UOP_SIGMOID:
        // sigmoid(x) = 1/(1+exp(-x)) = 1/(1+exp2(-x*log2e))
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    neg.f32 %%f1, %%f0;\n"
            "    mul.f32 %%f1, %%f1, 0f3FB8AA3B;\n"  // log2(e)
            "    ex2.approx.f32 %%f1, %%f1;\n"
            "    add.f32 %%f1, %%f1, 0f3F800000;\n"   // 1.0f
            "    rcp.approx.f32 %%f1, %%f1;\n\n");
        break;
    case UOP_TANH:
        // tanh(x) = 2*sigmoid(2x) - 1
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    add.f32 %%f1, %%f0, %%f0;\n"         // 2x
            "    neg.f32 %%f1, %%f1;\n"                // -2x
            "    mul.f32 %%f1, %%f1, 0f3FB8AA3B;\n"   // -2x * log2(e)
            "    ex2.approx.f32 %%f1, %%f1;\n"         // exp(-2x)
            "    add.f32 %%f1, %%f1, 0f3F800000;\n"   // 1 + exp(-2x)
            "    rcp.approx.f32 %%f1, %%f1;\n"         // sigmoid(2x)
            "    add.f32 %%f1, %%f1, %%f1;\n"          // 2*sigmoid(2x)
            "    sub.f32 %%f1, %%f1, 0f3F800000;\n\n"); // - 1
        break;
    case UOP_RECIP:
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    rcp.approx.f32 %%f1, %%f0;\n\n");
        break;
    default:
        free(ptx);
        return NULL;
    }

    pos += ptx_write_store_unary(ptx + pos, PTX_BUF_SIZE - pos);
    cg->kernel_count++;
    return ptx;
}

char* cml_ptx_gen_binary(CMLPTXCodegen* cg, UOpType op, const char* kernel_name) {
    if (!cg || !kernel_name) return NULL;

    char* ptx = (char*)malloc(PTX_BUF_SIZE);
    if (!ptx) return NULL;

    int pos = 0;
    pos += ptx_write_header(ptx + pos, PTX_BUF_SIZE - pos, cg->sm_version);
    pos += ptx_write_prologue_binary(ptx + pos, PTX_BUF_SIZE - pos, kernel_name);
    pos += ptx_write_load_binary(ptx + pos, PTX_BUF_SIZE - pos);

    // Emit the operation: inputs are %f0, %f1; output is %f2
    pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos), "    // Compute\n");

    switch (op) {
    case UOP_ADD:
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    add.f32 %%f2, %%f0, %%f1;\n\n");
        break;
    case UOP_SUB:
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    sub.f32 %%f2, %%f0, %%f1;\n\n");
        break;
    case UOP_MUL:
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    mul.f32 %%f2, %%f0, %%f1;\n\n");
        break;
    case UOP_DIV:
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    div.approx.f32 %%f2, %%f0, %%f1;\n\n");
        break;
    case UOP_MAX:
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    max.f32 %%f2, %%f0, %%f1;\n\n");
        break;
    case UOP_POW:
        // pow(a,b) = exp2(b * log2(a))
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    lg2.approx.f32 %%f2, %%f0;\n"    // log2(a)
            "    mul.f32 %%f2, %%f1, %%f2;\n"      // b * log2(a)
            "    ex2.approx.f32 %%f2, %%f2;\n\n"); // 2^(b*log2(a))
        break;
    case UOP_CMPLT:
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    setp.lt.f32 %%p1, %%f0, %%f1;\n"
            "    selp.f32 %%f2, 0f3F800000, 0f00000000, %%p1;\n\n"); // 1.0 : 0.0
        break;
    default:
        free(ptx);
        return NULL;
    }

    pos += ptx_write_store_binary(ptx + pos, PTX_BUF_SIZE - pos);
    cg->kernel_count++;
    return ptx;
}

char* cml_ptx_gen_fill(CMLPTXCodegen* cg, float value, const char* kernel_name) {
    if (!cg || !kernel_name) return NULL;

    char hex[16];
    float_to_ptx_hex(value, hex, sizeof(hex));

    char* ptx = (char*)malloc(PTX_BUF_SIZE);
    if (!ptx) return NULL;

    int pos = 0;
    pos += ptx_write_header(ptx + pos, PTX_BUF_SIZE - pos, cg->sm_version);

    pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
        ".visible .entry %s(\n"
        "    .param .u64 param_out,\n"
        "    .param .u32 param_n\n"
        ") {\n"
        "    .reg .pred %%p<2>;\n"
        "    .reg .b32 %%r<8>;\n"
        "    .reg .b64 %%rd<4>;\n"
        "    .reg .f32 %%f<2>;\n\n"
        "    mov.u32 %%r0, %%tid.x;\n"
        "    mov.u32 %%r1, %%ctaid.x;\n"
        "    mov.u32 %%r2, %%ntid.x;\n"
        "    mad.lo.u32 %%r3, %%r1, %%r2, %%r0;\n"
        "    ld.param.u32 %%r4, [param_n];\n"
        "    setp.ge.u32 %%p0, %%r3, %%r4;\n"
        "    @%%p0 ret;\n\n"
        "    // Fill with constant\n"
        "    mov.f32 %%f0, %s;\n"
        "    ld.param.u64 %%rd0, [param_out];\n"
        "    cvt.u64.u32 %%rd1, %%r3;\n"
        "    shl.b64 %%rd2, %%rd1, 2;\n"
        "    add.u64 %%rd3, %%rd0, %%rd2;\n"
        "    st.global.f32 [%%rd3], %%f0;\n"
        "    ret;\n"
        "}\n",
        kernel_name, hex);

    cg->kernel_count++;
    return ptx;
}

char* cml_ptx_gen_where(CMLPTXCodegen* cg, const char* kernel_name) {
    if (!cg || !kernel_name) return NULL;

    char* ptx = (char*)malloc(PTX_BUF_SIZE);
    if (!ptx) return NULL;

    int pos = 0;
    pos += ptx_write_header(ptx + pos, PTX_BUF_SIZE - pos, cg->sm_version);

    pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
        ".visible .entry %s(\n"
        "    .param .u64 param_cond,\n"
        "    .param .u64 param_a,\n"
        "    .param .u64 param_b,\n"
        "    .param .u64 param_out,\n"
        "    .param .u32 param_n\n"
        ") {\n"
        "    .reg .pred %%p<4>;\n"
        "    .reg .b32 %%r<8>;\n"
        "    .reg .b64 %%rd<12>;\n"
        "    .reg .f32 %%f<8>;\n\n"
        "    mov.u32 %%r0, %%tid.x;\n"
        "    mov.u32 %%r1, %%ctaid.x;\n"
        "    mov.u32 %%r2, %%ntid.x;\n"
        "    mad.lo.u32 %%r3, %%r1, %%r2, %%r0;\n"
        "    ld.param.u32 %%r4, [param_n];\n"
        "    setp.ge.u32 %%p0, %%r3, %%r4;\n"
        "    @%%p0 ret;\n\n"
        "    // Load cond, a, b\n"
        "    cvt.u64.u32 %%rd0, %%r3;\n"
        "    shl.b64 %%rd1, %%rd0, 2;\n"
        "    ld.param.u64 %%rd2, [param_cond];\n"
        "    add.u64 %%rd3, %%rd2, %%rd1;\n"
        "    ld.global.f32 %%f0, [%%rd3];\n"
        "    ld.param.u64 %%rd4, [param_a];\n"
        "    add.u64 %%rd5, %%rd4, %%rd1;\n"
        "    ld.global.f32 %%f1, [%%rd5];\n"
        "    ld.param.u64 %%rd6, [param_b];\n"
        "    add.u64 %%rd7, %%rd6, %%rd1;\n"
        "    ld.global.f32 %%f2, [%%rd7];\n\n"
        "    // where: out = cond != 0 ? a : b\n"
        "    setp.ne.f32 %%p1, %%f0, 0f00000000;\n"
        "    selp.f32 %%f3, %%f1, %%f2, %%p1;\n\n"
        "    // Store\n"
        "    ld.param.u64 %%rd8, [param_out];\n"
        "    add.u64 %%rd9, %%rd8, %%rd1;\n"
        "    st.global.f32 [%%rd9], %%f3;\n"
        "    ret;\n"
        "}\n",
        kernel_name);

    cg->kernel_count++;
    return ptx;
}

char* cml_ptx_gen_reduction(CMLPTXCodegen* cg, UOpType op, const char* kernel_name) {
    if (!cg || !kernel_name) return NULL;
    if (op != UOP_SUM && op != UOP_MEAN) return NULL;

    char* ptx = (char*)malloc(PTX_BUF_SIZE);
    if (!ptx) return NULL;

    int pos = 0;
    pos += ptx_write_header(ptx + pos, PTX_BUF_SIZE - pos, cg->sm_version);

    // Atomic reduction: each thread processes one element and atomically adds to output
    pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
        ".visible .entry %s(\n"
        "    .param .u64 param_in,\n"
        "    .param .u64 param_out,\n"
        "    .param .u32 param_n\n"
        ") {\n"
        "    .reg .pred %%p<4>;\n"
        "    .reg .b32 %%r<16>;\n"
        "    .reg .b64 %%rd<8>;\n"
        "    .reg .f32 %%f<8>;\n\n"
        "    mov.u32 %%r0, %%tid.x;\n"
        "    mov.u32 %%r1, %%ctaid.x;\n"
        "    mov.u32 %%r2, %%ntid.x;\n"
        "    mad.lo.u32 %%r3, %%r1, %%r2, %%r0;\n"
        "    ld.param.u32 %%r4, [param_n];\n"
        "    setp.ge.u32 %%p0, %%r3, %%r4;\n"
        "    @%%p0 ret;\n\n"
        "    // Load element\n"
        "    ld.param.u64 %%rd0, [param_in];\n"
        "    cvt.u64.u32 %%rd1, %%r3;\n"
        "    shl.b64 %%rd2, %%rd1, 2;\n"
        "    add.u64 %%rd3, %%rd0, %%rd2;\n"
        "    ld.global.f32 %%f0, [%%rd3];\n\n",
        kernel_name);

    if (op == UOP_MEAN) {
        // Divide by n before atomic add
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    // Divide by n for mean\n"
            "    cvt.rn.f32.u32 %%f1, %%r4;\n"
            "    div.approx.f32 %%f0, %%f0, %%f1;\n\n");
    }

    pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
        "    // Atomic add to output\n"
        "    ld.param.u64 %%rd4, [param_out];\n"
        "    atom.global.add.f32 %%f2, [%%rd4], %%f0;\n"
        "    ret;\n"
        "}\n");

    cg->kernel_count++;
    return ptx;
}

char* cml_ptx_gen_matmul(CMLPTXCodegen* cg, const char* kernel_name) {
    if (!cg || !kernel_name) return NULL;

    char* ptx = (char*)malloc(PTX_BUF_SIZE);
    if (!ptx) return NULL;

    int pos = 0;
    pos += ptx_write_header(ptx + pos, PTX_BUF_SIZE - pos, cg->sm_version);

    // Simple matmul: C[row][col] = sum_k A[row][k] * B[k][col]
    // Grid: (M, N), each thread computes one output element
    pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
        ".visible .entry %s(\n"
        "    .param .u64 param_a,\n"
        "    .param .u64 param_b,\n"
        "    .param .u64 param_c,\n"
        "    .param .u32 param_M,\n"
        "    .param .u32 param_N,\n"
        "    .param .u32 param_K\n"
        ") {\n"
        "    .reg .pred %%p<4>;\n"
        "    .reg .b32 %%r<32>;\n"
        "    .reg .b64 %%rd<16>;\n"
        "    .reg .f32 %%f<8>;\n\n"
        "    // Thread indices: row = blockIdx.x * blockDim.x + threadIdx.x\n"
        "    //                 col = blockIdx.y * blockDim.y + threadIdx.y\n"
        "    mov.u32 %%r0, %%tid.x;\n"
        "    mov.u32 %%r1, %%ctaid.x;\n"
        "    mov.u32 %%r2, %%ntid.x;\n"
        "    mad.lo.u32 %%r3, %%r1, %%r2, %%r0;\n"  // row
        "    mov.u32 %%r4, %%tid.y;\n"
        "    mov.u32 %%r5, %%ctaid.y;\n"
        "    mov.u32 %%r6, %%ntid.y;\n"
        "    mad.lo.u32 %%r7, %%r5, %%r6, %%r4;\n"  // col
        "\n"
        "    // Bounds check\n"
        "    ld.param.u32 %%r8, [param_M];\n"
        "    ld.param.u32 %%r9, [param_N];\n"
        "    ld.param.u32 %%r10, [param_K];\n"
        "    setp.ge.u32 %%p0, %%r3, %%r8;\n"
        "    @%%p0 ret;\n"
        "    setp.ge.u32 %%p1, %%r7, %%r9;\n"
        "    @%%p1 ret;\n\n"
        "    // Accumulator\n"
        "    mov.f32 %%f0, 0f00000000;\n"
        "    mov.u32 %%r11, 0;\n\n"
        "LOOP_%s:\n"
        "    setp.ge.u32 %%p2, %%r11, %%r10;\n"
        "    @%%p2 bra DONE_%s;\n\n"
        "    // A[row][k]: offset = (row * K + k) * 4\n"
        "    ld.param.u64 %%rd0, [param_a];\n"
        "    mad.lo.u32 %%r12, %%r3, %%r10, %%r11;\n"
        "    cvt.u64.u32 %%rd1, %%r12;\n"
        "    shl.b64 %%rd2, %%rd1, 2;\n"
        "    add.u64 %%rd3, %%rd0, %%rd2;\n"
        "    ld.global.f32 %%f1, [%%rd3];\n\n"
        "    // B[k][col]: offset = (k * N + col) * 4\n"
        "    ld.param.u64 %%rd4, [param_b];\n"
        "    mad.lo.u32 %%r13, %%r11, %%r9, %%r7;\n"
        "    cvt.u64.u32 %%rd5, %%r13;\n"
        "    shl.b64 %%rd6, %%rd5, 2;\n"
        "    add.u64 %%rd7, %%rd4, %%rd6;\n"
        "    ld.global.f32 %%f2, [%%rd7];\n\n"
        "    // Accumulate\n"
        "    fma.rn.f32 %%f0, %%f1, %%f2, %%f0;\n"
        "    add.u32 %%r11, %%r11, 1;\n"
        "    bra LOOP_%s;\n\n"
        "DONE_%s:\n"
        "    // Store C[row][col]\n"
        "    ld.param.u64 %%rd8, [param_c];\n"
        "    mad.lo.u32 %%r14, %%r3, %%r9, %%r7;\n"
        "    cvt.u64.u32 %%rd9, %%r14;\n"
        "    shl.b64 %%rd10, %%rd9, 2;\n"
        "    add.u64 %%rd11, %%rd8, %%rd10;\n"
        "    st.global.f32 [%%rd11], %%f0;\n"
        "    ret;\n"
        "}\n",
        kernel_name, kernel_name, kernel_name, kernel_name, kernel_name);

    cg->kernel_count++;
    return ptx;
}

int cml_ptx_execute_graph(CMLPTXCodegen* cg, CMLGraph_t ir) {
    if (!cg || !ir) return -1;
    // Full graph execution would iterate IR nodes, generate PTX per-op,
    // and use cml_cuda_compile_ptx + cml_cuda_launch_kernel.
    // This is a placeholder for non-LLVM CUDA execution path.
    LOG_DEBUG("PTX codegen: graph execution not yet implemented (use per-kernel API)");
    return -1;
}
