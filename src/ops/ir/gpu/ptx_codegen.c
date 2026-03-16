/**
 * @file ptx_codegen.c
 * @brief Standalone PTX assembly code generation (no LLVM dependency)
 *
 * Direct PTX assembly text generation via snprintf into dynamic buffers.
 * Uses IEEE 754 hex float format (0fHHHHHHHH) for float literals.
 */

#include "ops/ir/gpu/ptx_codegen.h"
#include "ops/ir/gpu/cuda_backend.h"
#include "ops/ir/internal.h"
#include "ops/ir/execution.h"
#include "tensor/tensor.h"
#include "core/logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

// Maximum PTX buffer size per kernel
#define PTX_BUF_SIZE 8192

// Larger buffer for CUDA C source (tiled matmul, conv2d)
#define CUDA_SRC_BUF_SIZE 16384

// Default block size for kernel launches
#define PTX_BLOCK_SIZE 256

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

/** Classify op as unary for PTX codegen dispatch */
static bool ptx_is_unary_op(UOpType t) {
    switch (t) {
    case UOP_NEG: case UOP_EXP: case UOP_LOG: case UOP_SQRT:
    case UOP_ABS: case UOP_SIN: case UOP_COS: case UOP_SIGMOID:
    case UOP_TANH: case UOP_RECIP: case UOP_FLOOR: case UOP_CEIL:
    case UOP_ELU: case UOP_SELU: case UOP_SILU: case UOP_MISH:
    case UOP_HARDSWISH:
        return true;
    default:
        return false;
    }
}

/** Classify op as binary for PTX codegen dispatch */
static bool ptx_is_binary_op(UOpType t) {
    switch (t) {
    case UOP_ADD: case UOP_SUB: case UOP_MUL: case UOP_DIV:
    case UOP_MAX: case UOP_POW: case UOP_CMPLT: case UOP_MOD:
    case UOP_IDIV:
        return true;
    default:
        return false;
    }
}

/** Classify op as reduction for PTX codegen dispatch */
static bool ptx_is_reduction(UOpType t) {
    switch (t) {
    case UOP_SUM: case UOP_MEAN: case UOP_MAX_REDUCE:
    case UOP_MIN_REDUCE: case UOP_PROD:
        return true;
    default:
        return false;
    }
}

CMLPTXCodegen* cml_ptx_codegen_create(int sm_version, struct CMLCUDABackend* cuda) {
    CMLPTXCodegen* cg = (CMLPTXCodegen*)calloc(1, sizeof(CMLPTXCodegen));
    if (!cg) return NULL;
    cg->sm_version = sm_version > 0 ? sm_version : 50;
    cg->kernel_count = 0;
    cg->initialized = true;
    cg->cuda = cuda;
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
    case UOP_FLOOR:
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    cvt.rmi.f32.f32 %%f1, %%f0;\n\n");
        break;
    case UOP_CEIL:
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    cvt.rpi.f32.f32 %%f1, %%f0;\n\n");
        break;
    case UOP_ELU:
        // ELU: x > 0 ? x : exp(x) - 1  (alpha=1.0)
        // exp(x) = ex2(x * log2e)
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    setp.gt.f32 %%p1, %%f0, 0f00000000;\n"     // x > 0?
            "    mul.f32 %%f2, %%f0, 0f3FB8AA3B;\n"          // x * log2(e)
            "    ex2.approx.f32 %%f2, %%f2;\n"                // exp(x)
            "    sub.f32 %%f2, %%f2, 0f3F800000;\n"           // exp(x) - 1
            "    selp.f32 %%f1, %%f0, %%f2, %%p1;\n\n");     // x > 0 ? x : exp(x)-1
        break;
    case UOP_SELU:
        // SELU: scale * (x > 0 ? x : alpha*(exp(x)-1))
        // scale = 1.0507 = 0f3F86592A, alpha = 1.67326 = 0f3FD62E30
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    setp.gt.f32 %%p1, %%f0, 0f00000000;\n"     // x > 0?
            "    mul.f32 %%f2, %%f0, 0f3FB8AA3B;\n"          // x * log2(e)
            "    ex2.approx.f32 %%f2, %%f2;\n"                // exp(x)
            "    sub.f32 %%f2, %%f2, 0f3F800000;\n"           // exp(x) - 1
            "    mul.f32 %%f2, %%f2, 0f3FD62E30;\n"           // alpha*(exp(x)-1)
            "    selp.f32 %%f1, %%f0, %%f2, %%p1;\n"         // x > 0 ? x : alpha*(exp(x)-1)
            "    mul.f32 %%f1, %%f1, 0f3F86592A;\n\n");      // * scale
        break;
    case UOP_SILU:
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    neg.f32 %%f2, %%f0;\n"                      // -x
            "    mul.f32 %%f2, %%f2, 0f3FB8AA3B;\n"          // -x * log2(e)
            "    ex2.approx.f32 %%f2, %%f2;\n"                // exp(-x)
            "    add.f32 %%f2, %%f2, 0f3F800000;\n"           // 1 + exp(-x)
            "    rcp.approx.f32 %%f2, %%f2;\n"                // sigmoid(x)
            "    mul.f32 %%f1, %%f0, %%f2;\n\n");             // x * sigmoid(x)
        break;
    case UOP_MISH:
        // Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
        // softplus(x) = log(1+exp(x)) = log2(1+exp2(x*log2e)) * ln2
        // tanh(sp) = 2*sigmoid(2*sp) - 1
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    // softplus(x) = log(1 + exp(x))\n"
            "    mul.f32 %%f2, %%f0, 0f3FB8AA3B;\n"          // x * log2(e)
            "    ex2.approx.f32 %%f2, %%f2;\n"                // exp(x)
            "    add.f32 %%f2, %%f2, 0f3F800000;\n"           // 1 + exp(x)
            "    lg2.approx.f32 %%f2, %%f2;\n"                // log2(1+exp(x))
            "    mul.f32 %%f2, %%f2, 0f3F317218;\n"           // * ln(2) = softplus(x)
            "    // tanh(softplus) = 2*sigmoid(2*sp) - 1\n"
            "    add.f32 %%f3, %%f2, %%f2;\n"                 // 2*sp
            "    neg.f32 %%f3, %%f3;\n"                       // -2*sp
            "    mul.f32 %%f3, %%f3, 0f3FB8AA3B;\n"           // -2*sp * log2(e)
            "    ex2.approx.f32 %%f3, %%f3;\n"                // exp(-2*sp)
            "    add.f32 %%f3, %%f3, 0f3F800000;\n"           // 1 + exp(-2*sp)
            "    rcp.approx.f32 %%f3, %%f3;\n"                // sigmoid(2*sp)
            "    add.f32 %%f3, %%f3, %%f3;\n"                 // 2*sigmoid(2*sp)
            "    sub.f32 %%f3, %%f3, 0f3F800000;\n"           // tanh(sp)
            "    mul.f32 %%f1, %%f0, %%f3;\n\n");             // x * tanh(sp)
        break;
    case UOP_HARDSWISH:
        // HardSwish: x > 3 ? x : x < -3 ? 0 : x*(x+3)/6
        // 3.0 = 0f40400000, -3.0 = 0fC0400000, 1/6 = 0f3E2AAAAB
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    setp.gt.f32 %%p1, %%f0, 0f40400000;\n"     // x > 3?
            "    setp.lt.f32 %%p2, %%f0, 0fC0400000;\n"     // x < -3?
            "    add.f32 %%f2, %%f0, 0f40400000;\n"          // x + 3
            "    mul.f32 %%f2, %%f0, %%f2;\n"                // x * (x + 3)
            "    mul.f32 %%f2, %%f2, 0f3E2AAAAB;\n"          // x*(x+3) / 6
            "    selp.f32 %%f1, 0f00000000, %%f2, %%p2;\n"  // x < -3 ? 0 : mid
            "    selp.f32 %%f1, %%f0, %%f1, %%p1;\n\n");    // x > 3 ? x : result
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
    case UOP_MOD:
        // mod(a, b) = a - b * floor(a / b)
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    div.approx.f32 %%f2, %%f0, %%f1;\n"    // a / b
            "    cvt.rmi.f32.f32 %%f2, %%f2;\n"          // floor(a / b)
            "    mul.f32 %%f2, %%f2, %%f1;\n"             // b * floor(a/b)
            "    sub.f32 %%f2, %%f0, %%f2;\n\n");         // a - b*floor(a/b)
        break;
    case UOP_IDIV:
        // idiv(a, b) = floor(a / b)
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    div.approx.f32 %%f2, %%f0, %%f1;\n"    // a / b
            "    cvt.rmi.f32.f32 %%f2, %%f2;\n\n");      // floor(a / b)
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
    if (op != UOP_SUM && op != UOP_MEAN && op != UOP_MAX_REDUCE &&
        op != UOP_MIN_REDUCE && op != UOP_PROD) {
        return NULL;
    }

    char* ptx = (char*)malloc(PTX_BUF_SIZE);
    if (!ptx) return NULL;

    int pos = 0;
    pos += ptx_write_header(ptx + pos, PTX_BUF_SIZE - pos, cg->sm_version);

    // Atomic reduction: each thread processes one element and atomically updates output
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

    if (op == UOP_SUM || op == UOP_MEAN) {
        // atom.global.add.f32 is natively supported
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    // Atomic add to output\n"
            "    ld.param.u64 %%rd4, [param_out];\n"
            "    atom.global.add.f32 %%f2, [%%rd4], %%f0;\n"
            "    ret;\n"
            "}\n");
    } else if (op == UOP_MAX_REDUCE) {
        // CAS loop for atomic max: load old, compute max, atomicCAS until success
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    // CAS loop for atomic max\n"
            "    ld.param.u64 %%rd4, [param_out];\n"
            "    ld.global.f32 %%f1, [%%rd4];\n"           // old = *out
            "    mov.b32 %%r5, %%f1;\n"                    // old_bits
            "CAS_MAX_%s:\n"
            "    max.f32 %%f2, %%f1, %%f0;\n"              // desired = max(old, val)
            "    mov.b32 %%r6, %%f2;\n"                    // desired_bits
            "    atom.global.cas.b32 %%r7, [%%rd4], %%r5, %%r6;\n"
            "    setp.ne.u32 %%p1, %%r7, %%r5;\n"         // swapped?
            "    mov.b32 %%f1, %%r7;\n"                    // old = returned
            "    mov.u32 %%r5, %%r7;\n"                    // old_bits = returned
            "    @%%p1 bra CAS_MAX_%s;\n"
            "    ret;\n"
            "}\n",
            kernel_name, kernel_name);
    } else if (op == UOP_MIN_REDUCE) {
        // CAS loop for atomic min
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    // CAS loop for atomic min\n"
            "    ld.param.u64 %%rd4, [param_out];\n"
            "    ld.global.f32 %%f1, [%%rd4];\n"           // old = *out
            "    mov.b32 %%r5, %%f1;\n"                    // old_bits
            "CAS_MIN_%s:\n"
            "    min.f32 %%f2, %%f1, %%f0;\n"              // desired = min(old, val)
            "    mov.b32 %%r6, %%f2;\n"                    // desired_bits
            "    atom.global.cas.b32 %%r7, [%%rd4], %%r5, %%r6;\n"
            "    setp.ne.u32 %%p1, %%r7, %%r5;\n"         // swapped?
            "    mov.b32 %%f1, %%r7;\n"                    // old = returned
            "    mov.u32 %%r5, %%r7;\n"                    // old_bits = returned
            "    @%%p1 bra CAS_MIN_%s;\n"
            "    ret;\n"
            "}\n",
            kernel_name, kernel_name);
    } else if (op == UOP_PROD) {
        // CAS loop for atomic mul
        pos += snprintf(ptx + pos, (size_t)(PTX_BUF_SIZE - pos),
            "    // CAS loop for atomic mul (product)\n"
            "    ld.param.u64 %%rd4, [param_out];\n"
            "    ld.global.f32 %%f1, [%%rd4];\n"           // old = *out
            "    mov.b32 %%r5, %%f1;\n"                    // old_bits
            "CAS_MUL_%s:\n"
            "    mul.f32 %%f2, %%f1, %%f0;\n"              // desired = old * val
            "    mov.b32 %%r6, %%f2;\n"                    // desired_bits
            "    atom.global.cas.b32 %%r7, [%%rd4], %%r5, %%r6;\n"
            "    setp.ne.u32 %%p1, %%r7, %%r5;\n"         // swapped?
            "    mov.b32 %%f1, %%r7;\n"                    // old = returned
            "    mov.u32 %%r5, %%r7;\n"                    // old_bits = returned
            "    @%%p1 bra CAS_MUL_%s;\n"
            "    ret;\n"
            "}\n",
            kernel_name, kernel_name);
    }

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

char* cml_ptx_gen_tiled_matmul(CMLPTXCodegen* cg, const char* kernel_name) {
    if (!cg || !kernel_name) return NULL;

    char* src = (char*)malloc(CUDA_SRC_BUF_SIZE);
    if (!src) return NULL;

    // Generate CUDA C source for 16x16 tiled matmul using shared memory.
    // This will be compiled via NVRTC (cml_cuda_compile_source).
    int pos = 0;
    pos += snprintf(src + pos, (size_t)(CUDA_SRC_BUF_SIZE - pos),
        "#define TILE_SIZE 16\n\n"
        "extern \"C\" __global__ void %s(\n"
        "    const float* __restrict__ A,\n"
        "    const float* __restrict__ B,\n"
        "    float* __restrict__ C,\n"
        "    int M, int N, int K)\n"
        "{\n"
        "    __shared__ float As[TILE_SIZE][TILE_SIZE];\n"
        "    __shared__ float Bs[TILE_SIZE][TILE_SIZE];\n"
        "\n"
        "    int row = blockIdx.y * TILE_SIZE + threadIdx.y;\n"
        "    int col = blockIdx.x * TILE_SIZE + threadIdx.x;\n"
        "\n"
        "    float acc = 0.0f;\n"
        "    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;\n"
        "\n"
        "    for (int t = 0; t < numTiles; t++) {\n"
        "        int aCol = t * TILE_SIZE + threadIdx.x;\n"
        "        int bRow = t * TILE_SIZE + threadIdx.y;\n"
        "\n"
        "        // Load tile from A into shared memory\n"
        "        if (row < M && aCol < K)\n"
        "            As[threadIdx.y][threadIdx.x] = A[row * K + aCol];\n"
        "        else\n"
        "            As[threadIdx.y][threadIdx.x] = 0.0f;\n"
        "\n"
        "        // Load tile from B into shared memory\n"
        "        if (bRow < K && col < N)\n"
        "            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];\n"
        "        else\n"
        "            Bs[threadIdx.y][threadIdx.x] = 0.0f;\n"
        "\n"
        "        __syncthreads();\n"
        "\n"
        "        // Compute partial dot product for this tile\n"
        "        for (int k = 0; k < TILE_SIZE; k++) {\n"
        "            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];\n"
        "        }\n"
        "\n"
        "        __syncthreads();\n"
        "    }\n"
        "\n"
        "    // Write result\n"
        "    if (row < M && col < N) {\n"
        "        C[row * N + col] = acc;\n"
        "    }\n"
        "}\n",
        kernel_name);

    cg->kernel_count++;
    return src;
}

char* cml_ptx_gen_conv2d(CMLPTXCodegen* cg, const char* kernel_name) {
    if (!cg || !kernel_name) return NULL;

    char* src = (char*)malloc(CUDA_SRC_BUF_SIZE);
    if (!src) return NULL;

    // Generate CUDA C source for direct conv2d kernel.
    // Layout: NCHW for input, OIHW for weight.
    // Each thread computes one output element.
    int pos = 0;
    pos += snprintf(src + pos, (size_t)(CUDA_SRC_BUF_SIZE - pos),
        "extern \"C\" __global__ void %s(\n"
        "    const float* __restrict__ input,\n"
        "    const float* __restrict__ weight,\n"
        "    const float* __restrict__ bias,\n"
        "    float* __restrict__ output,\n"
        "    int batch, int in_channels, int out_channels,\n"
        "    int in_h, int in_w,\n"
        "    int out_h, int out_w,\n"
        "    int kernel_h, int kernel_w,\n"
        "    int stride_h, int stride_w,\n"
        "    int pad_h, int pad_w,\n"
        "    int dilation_h, int dilation_w,\n"
        "    int groups)\n"
        "{\n"
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    int total = batch * out_channels * out_h * out_w;\n"
        "    if (idx >= total) return;\n"
        "\n"
        "    // Decompose linear index into (n, oc, oh, ow)\n"
        "    int ow = idx %% out_w;\n"
        "    int tmp = idx / out_w;\n"
        "    int oh = tmp %% out_h;\n"
        "    tmp = tmp / out_h;\n"
        "    int oc = tmp %% out_channels;\n"
        "    int n = tmp / out_channels;\n"
        "\n"
        "    // Determine group and channel range\n"
        "    int channels_per_group = in_channels / groups;\n"
        "    int group = oc / (out_channels / groups);\n"
        "    int ic_start = group * channels_per_group;\n"
        "\n"
        "    float acc = 0.0f;\n"
        "\n"
        "    for (int ic = 0; ic < channels_per_group; ic++) {\n"
        "        for (int kh = 0; kh < kernel_h; kh++) {\n"
        "            for (int kw = 0; kw < kernel_w; kw++) {\n"
        "                int ih = oh * stride_h - pad_h + kh * dilation_h;\n"
        "                int iw = ow * stride_w - pad_w + kw * dilation_w;\n"
        "                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {\n"
        "                    int in_idx = ((n * in_channels + (ic_start + ic)) * in_h + ih) * in_w + iw;\n"
        "                    int w_idx = ((oc * channels_per_group + ic) * kernel_h + kh) * kernel_w + kw;\n"
        "                    acc += input[in_idx] * weight[w_idx];\n"
        "                }\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "\n"
        "    // Add bias if present (bias pointer is not NULL)\n"
        "    if (bias) {\n"
        "        acc += bias[oc];\n"
        "    }\n"
        "\n"
        "    output[idx] = acc;\n"
        "}\n",
        kernel_name);

    cg->kernel_count++;
    return src;
}

/**
 * Execute a single IR node using PTX codegen.
 * Returns 0 on success, -1 on failure (caller should use CPU fallback).
 */
static int ptx_execute_node(CMLPTXCodegen* cg, struct IRNode* node) {
    if (!node || !node->output || !cg->cuda || !cg->cuda->initialized) return -1;

    Tensor* out = node->output;
    UOpType type = node->type;

    // Allocate output host memory if needed
    if (!out->data && out->numel > 0) {
        size_t sz = out->numel * sizeof(float);
        out->data = cml_buffer_cache_alloc(sz);
        if (!out->data) {
            LOG_ERROR("PTX codegen: Failed to allocate output tensor");
            return -1;
        }
        out->owns_data = true;
    }

    // Build unique kernel name
    char fn_name[64];
    snprintf(fn_name, sizeof(fn_name), "ptx_k%d", cg->kernel_count);

    CMLCUDABackend* cuda = cg->cuda;
    char* ptx_code = NULL;
    int result = -1;

    // Unary ops
    if (ptx_is_unary_op(type)) {
        if (node->num_inputs < 1 || !node->inputs[0] || !node->inputs[0]->data)
            return -1;

        ptx_code = cml_ptx_gen_unary(cg, type, fn_name);
        if (!ptx_code) return -1;

        CMLCUDAKernel* kernel = cml_cuda_compile_ptx(cuda, ptx_code, fn_name);
        free(ptx_code);
        if (!kernel) return -1;

        size_t in_bytes = node->inputs[0]->numel * sizeof(float);
        size_t out_bytes = out->numel * sizeof(float);
        CUdeviceptr d_in = cml_cuda_malloc(cuda, in_bytes);
        CUdeviceptr d_out = cml_cuda_malloc(cuda, out_bytes);
        if (!d_in || !d_out) goto unary_cleanup;

        cml_cuda_memcpy_h2d(cuda, d_in, node->inputs[0]->data, in_bytes);

        int32_t n = (int32_t)out->numel;
        void* args[] = { &d_in, &d_out, &n };
        int grid = ((int)out->numel + PTX_BLOCK_SIZE - 1) / PTX_BLOCK_SIZE;
        cml_cuda_kernel_set_launch_config(kernel, grid, 1, 1, PTX_BLOCK_SIZE, 1, 1);
        if (cml_cuda_launch_kernel(cuda, kernel, args, 3) == 0) {
            cml_cuda_synchronize(cuda);
            cml_cuda_memcpy_d2h(cuda, out->data, d_out, out_bytes);
            result = 0;
        }

    unary_cleanup:
        if (d_in) cml_cuda_free(cuda, d_in);
        if (d_out) cml_cuda_free(cuda, d_out);
        cml_cuda_kernel_free(cuda, kernel);
        if (result == 0) { node->is_executed = true; out->is_executed = true; }
        return result;
    }

    // Binary ops
    if (ptx_is_binary_op(type)) {
        if (node->num_inputs < 2 || !node->inputs[0] || !node->inputs[1] ||
            !node->inputs[0]->data || !node->inputs[1]->data)
            return -1;

        ptx_code = cml_ptx_gen_binary(cg, type, fn_name);
        if (!ptx_code) return -1;

        CMLCUDAKernel* kernel = cml_cuda_compile_ptx(cuda, ptx_code, fn_name);
        free(ptx_code);
        if (!kernel) return -1;

        size_t a_bytes = node->inputs[0]->numel * sizeof(float);
        size_t b_bytes = node->inputs[1]->numel * sizeof(float);
        size_t out_bytes = out->numel * sizeof(float);
        CUdeviceptr d_a = cml_cuda_malloc(cuda, a_bytes);
        CUdeviceptr d_b = cml_cuda_malloc(cuda, b_bytes);
        CUdeviceptr d_out = cml_cuda_malloc(cuda, out_bytes);
        if (!d_a || !d_b || !d_out) goto binary_cleanup;

        cml_cuda_memcpy_h2d(cuda, d_a, node->inputs[0]->data, a_bytes);
        cml_cuda_memcpy_h2d(cuda, d_b, node->inputs[1]->data, b_bytes);

        int32_t n = (int32_t)out->numel;
        void* args[] = { &d_a, &d_b, &d_out, &n };
        int grid = ((int)out->numel + PTX_BLOCK_SIZE - 1) / PTX_BLOCK_SIZE;
        cml_cuda_kernel_set_launch_config(kernel, grid, 1, 1, PTX_BLOCK_SIZE, 1, 1);
        if (cml_cuda_launch_kernel(cuda, kernel, args, 4) == 0) {
            cml_cuda_synchronize(cuda);
            cml_cuda_memcpy_d2h(cuda, out->data, d_out, out_bytes);
            result = 0;
        }

    binary_cleanup:
        if (d_a) cml_cuda_free(cuda, d_a);
        if (d_b) cml_cuda_free(cuda, d_b);
        if (d_out) cml_cuda_free(cuda, d_out);
        cml_cuda_kernel_free(cuda, kernel);
        if (result == 0) { node->is_executed = true; out->is_executed = true; }
        return result;
    }

    // Reduction ops
    if (ptx_is_reduction(type)) {
        if (node->num_inputs < 1 || !node->inputs[0] || !node->inputs[0]->data)
            return -1;

        ptx_code = cml_ptx_gen_reduction(cg, type, fn_name);
        if (!ptx_code) return -1;

        CMLCUDAKernel* kernel = cml_cuda_compile_ptx(cuda, ptx_code, fn_name);
        free(ptx_code);
        if (!kernel) return -1;

        size_t in_bytes = node->inputs[0]->numel * sizeof(float);
        size_t out_bytes = out->numel * sizeof(float);
        CUdeviceptr d_in = cml_cuda_malloc(cuda, in_bytes);
        CUdeviceptr d_out = cml_cuda_malloc(cuda, out_bytes);
        if (!d_in || !d_out) goto reduce_cleanup;

        cml_cuda_memcpy_h2d(cuda, d_in, node->inputs[0]->data, in_bytes);

        // Zero-init output for atomics (SUM/MEAN/PROD need identity init)
        if (type == UOP_PROD) {
            float one = 1.0f;
            cml_cuda_memcpy_h2d(cuda, d_out, &one, sizeof(float));
        } else if (type == UOP_MAX_REDUCE) {
            float neg_inf = -INFINITY;
            cml_cuda_memcpy_h2d(cuda, d_out, &neg_inf, sizeof(float));
        } else if (type == UOP_MIN_REDUCE) {
            float pos_inf = INFINITY;
            cml_cuda_memcpy_h2d(cuda, d_out, &pos_inf, sizeof(float));
        } else {
            float zero = 0.0f;
            cml_cuda_memcpy_h2d(cuda, d_out, &zero, sizeof(float));
        }

        int32_t n = (int32_t)node->inputs[0]->numel;
        void* args[] = { &d_in, &d_out, &n };
        int grid = ((int)node->inputs[0]->numel + PTX_BLOCK_SIZE - 1) / PTX_BLOCK_SIZE;
        cml_cuda_kernel_set_launch_config(kernel, grid, 1, 1, PTX_BLOCK_SIZE, 1, 1);
        if (cml_cuda_launch_kernel(cuda, kernel, args, 3) == 0) {
            cml_cuda_synchronize(cuda);
            cml_cuda_memcpy_d2h(cuda, out->data, d_out, out_bytes);
            result = 0;
        }

    reduce_cleanup:
        if (d_in) cml_cuda_free(cuda, d_in);
        if (d_out) cml_cuda_free(cuda, d_out);
        cml_cuda_kernel_free(cuda, kernel);
        if (result == 0) { node->is_executed = true; out->is_executed = true; }
        return result;
    }

    // Fill
    if (type == UOP_FILL) {
        FillParams* p = (FillParams*)node->params;
        float fill_val = p ? p->value : 0.0f;

        ptx_code = cml_ptx_gen_fill(cg, fill_val, fn_name);
        if (!ptx_code) return -1;

        CMLCUDAKernel* kernel = cml_cuda_compile_ptx(cuda, ptx_code, fn_name);
        free(ptx_code);
        if (!kernel) return -1;

        size_t out_bytes = out->numel * sizeof(float);
        CUdeviceptr d_out = cml_cuda_malloc(cuda, out_bytes);
        if (!d_out) goto fill_cleanup;

        int32_t n = (int32_t)out->numel;
        void* args[] = { &d_out, &n };
        int grid = ((int)out->numel + PTX_BLOCK_SIZE - 1) / PTX_BLOCK_SIZE;
        cml_cuda_kernel_set_launch_config(kernel, grid, 1, 1, PTX_BLOCK_SIZE, 1, 1);
        if (cml_cuda_launch_kernel(cuda, kernel, args, 2) == 0) {
            cml_cuda_synchronize(cuda);
            cml_cuda_memcpy_d2h(cuda, out->data, d_out, out_bytes);
            result = 0;
        }

    fill_cleanup:
        if (d_out) cml_cuda_free(cuda, d_out);
        cml_cuda_kernel_free(cuda, kernel);
        if (result == 0) { node->is_executed = true; out->is_executed = true; }
        return result;
    }

    // Where
    if (type == UOP_WHERE) {
        if (node->num_inputs < 3 || !node->inputs[0] || !node->inputs[1] ||
            !node->inputs[2] || !node->inputs[0]->data ||
            !node->inputs[1]->data || !node->inputs[2]->data)
            return -1;

        ptx_code = cml_ptx_gen_where(cg, fn_name);
        if (!ptx_code) return -1;

        CMLCUDAKernel* kernel = cml_cuda_compile_ptx(cuda, ptx_code, fn_name);
        free(ptx_code);
        if (!kernel) return -1;

        size_t c_bytes = node->inputs[0]->numel * sizeof(float);
        size_t a_bytes = node->inputs[1]->numel * sizeof(float);
        size_t b_bytes = node->inputs[2]->numel * sizeof(float);
        size_t out_bytes = out->numel * sizeof(float);
        CUdeviceptr d_c = cml_cuda_malloc(cuda, c_bytes);
        CUdeviceptr d_a = cml_cuda_malloc(cuda, a_bytes);
        CUdeviceptr d_b = cml_cuda_malloc(cuda, b_bytes);
        CUdeviceptr d_out = cml_cuda_malloc(cuda, out_bytes);
        if (!d_c || !d_a || !d_b || !d_out) goto where_cleanup;

        cml_cuda_memcpy_h2d(cuda, d_c, node->inputs[0]->data, c_bytes);
        cml_cuda_memcpy_h2d(cuda, d_a, node->inputs[1]->data, a_bytes);
        cml_cuda_memcpy_h2d(cuda, d_b, node->inputs[2]->data, b_bytes);

        int32_t n = (int32_t)out->numel;
        void* args[] = { &d_c, &d_a, &d_b, &d_out, &n };
        int grid = ((int)out->numel + PTX_BLOCK_SIZE - 1) / PTX_BLOCK_SIZE;
        cml_cuda_kernel_set_launch_config(kernel, grid, 1, 1, PTX_BLOCK_SIZE, 1, 1);
        if (cml_cuda_launch_kernel(cuda, kernel, args, 5) == 0) {
            cml_cuda_synchronize(cuda);
            cml_cuda_memcpy_d2h(cuda, out->data, d_out, out_bytes);
            result = 0;
        }

    where_cleanup:
        if (d_c) cml_cuda_free(cuda, d_c);
        if (d_a) cml_cuda_free(cuda, d_a);
        if (d_b) cml_cuda_free(cuda, d_b);
        if (d_out) cml_cuda_free(cuda, d_out);
        cml_cuda_kernel_free(cuda, kernel);
        if (result == 0) { node->is_executed = true; out->is_executed = true; }
        return result;
    }

    // Matmul
    if (type == UOP_MATMUL) {
        if (node->num_inputs < 2 || !node->inputs[0] || !node->inputs[1] ||
            !node->inputs[0]->data || !node->inputs[1]->data)
            return -1;
        Tensor* a = node->inputs[0];
        Tensor* b = node->inputs[1];
        if (a->ndim < 2 || b->ndim < 2) return -1;

        ptx_code = cml_ptx_gen_matmul(cg, fn_name);
        if (!ptx_code) return -1;

        CMLCUDAKernel* kernel = cml_cuda_compile_ptx(cuda, ptx_code, fn_name);
        free(ptx_code);
        if (!kernel) return -1;

        size_t a_bytes = a->numel * sizeof(float);
        size_t b_bytes = b->numel * sizeof(float);
        size_t out_bytes = out->numel * sizeof(float);
        CUdeviceptr d_A = cml_cuda_malloc(cuda, a_bytes);
        CUdeviceptr d_B = cml_cuda_malloc(cuda, b_bytes);
        CUdeviceptr d_C = cml_cuda_malloc(cuda, out_bytes);
        if (!d_A || !d_B || !d_C) goto matmul_cleanup;

        cml_cuda_memcpy_h2d(cuda, d_A, a->data, a_bytes);
        cml_cuda_memcpy_h2d(cuda, d_B, b->data, b_bytes);

        int32_t M = (int32_t)a->shape[a->ndim - 2];
        int32_t K = (int32_t)a->shape[a->ndim - 1];
        int32_t N = (int32_t)b->shape[b->ndim - 1];
        void* args[] = { &d_A, &d_B, &d_C, &M, &N, &K };

        // 2D grid: blocks cover (M, N) with 16x16 thread blocks
        int grid_x = (M + 15) / 16;
        int grid_y = (N + 15) / 16;
        cml_cuda_kernel_set_launch_config(kernel, grid_x, grid_y, 1, 16, 16, 1);
        if (cml_cuda_launch_kernel(cuda, kernel, args, 6) == 0) {
            cml_cuda_synchronize(cuda);
            cml_cuda_memcpy_d2h(cuda, out->data, d_C, out_bytes);
            result = 0;
        }

    matmul_cleanup:
        if (d_A) cml_cuda_free(cuda, d_A);
        if (d_B) cml_cuda_free(cuda, d_B);
        if (d_C) cml_cuda_free(cuda, d_C);
        cml_cuda_kernel_free(cuda, kernel);
        if (result == 0) { node->is_executed = true; out->is_executed = true; }
        return result;
    }

    // Unsupported op
    return -1;
}

int cml_ptx_execute_graph(CMLPTXCodegen* cg, CMLGraph_t ir) {
    if (!cg || !ir || !cg->initialized) return -1;

    LOG_DEBUG("PTX codegen: Executing IR graph (%d nodes)", ir->node_count);

    struct IRNode* node = ir->head;
    while (node) {
        if (!node->is_executed) {
            int rc = ptx_execute_node(cg, node);
            if (rc != 0) {
                LOG_WARNING("PTX codegen: Node op %d failed on GPU, using CPU fallback",
                            (int)node->type);
                cpu_execute_node(node);
                node->is_executed = true;
                if (node->output) node->output->is_executed = true;
            }
        }
        node = node->next;
    }

    ir->is_executed = true;
    return 0;
}
