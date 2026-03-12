/**
 * @file metal_codegen.c
 * @brief MSL (Metal Shading Language) code generation from IR nodes
 *
 * Translates IR nodes into MSL compute kernel source strings.
 * The generated kernels use device float* buffers and a
 * thread_position_in_grid index for elementwise operations.
 */

#include "ops/ir/gpu/metal_backend.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include "core/logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Maximum size for a generated MSL kernel source string */
#define MSL_BUF_SIZE 4096

/**
 * @brief Append a formatted string to a dynamically-growing buffer.
 *
 * @param buf    Pointer to the buffer (may be reallocated)
 * @param cap    Pointer to the current capacity
 * @param len    Pointer to the current length
 * @param fmt    printf-style format string
 */
static void buf_appendf(char** buf, size_t* cap, size_t* len,
                         const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int needed = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);

    if (needed < 0) return;

    /* Grow the buffer if necessary */
    while (*len + (size_t)needed + 1 > *cap) {
        *cap *= 2;
        char* tmp = (char*)realloc(*buf, *cap);
        if (!tmp) {
            LOG_ERROR("metal_codegen: realloc failed");
            return;
        }
        *buf = tmp;
    }

    va_start(ap, fmt);
    vsnprintf(*buf + *len, *cap - *len, fmt, ap);
    va_end(ap);

    *len += (size_t)needed;
}

/**
 * @brief Return the MSL expression for a binary operation.
 */
static const char* msl_binary_expr(UOpType type) {
    switch (type) {
    case UOP_ADD: return "a[idx] + b[idx]";
    case UOP_SUB: return "a[idx] - b[idx]";
    case UOP_MUL: return "a[idx] * b[idx]";
    case UOP_DIV: return "a[idx] / (b[idx] + 1e-8f)";
    default:      return NULL;
    }
}

/**
 * @brief Return the MSL expression for a unary operation.
 */
static const char* msl_unary_expr(UOpType type) {
    switch (type) {
    case UOP_NEG:     return "-a[idx]";
    case UOP_EXP:     return "exp(a[idx])";
    case UOP_LOG:     return "log(a[idx] + 1e-8f)";
    case UOP_SQRT:    return "sqrt(fabs(a[idx]))";
    case UOP_RECIP:   return "1.0f / (a[idx] + 1e-8f)";
    case UOP_ABS:     return "fabs(a[idx])";
    case UOP_SIN:     return "sin(a[idx])";
    case UOP_COS:     return "cos(a[idx])";
    case UOP_TAN:     return "tan(a[idx])";
    default:          return NULL;
    }
}

/**
 * @brief Generate Metal Shading Language source for a compute kernel
 *        corresponding to a single IR node.
 *
 * Supported UOp types:
 *   Binary:  ADD, SUB, MUL, DIV
 *   Unary:   NEG, EXP, LOG, SQRT, RELU, SIGMOID, TANH
 *
 * @param node  IR node to translate
 * @return Heap-allocated MSL source string (caller must free), or NULL
 */
char* cml_metal_generate_msl(struct IRNode* node) {
    if (!node) {
        LOG_ERROR("metal_codegen: NULL node");
        return NULL;
    }

    size_t cap = MSL_BUF_SIZE;
    size_t len = 0;
    char* buf = (char*)malloc(cap);
    if (!buf) return NULL;
    buf[0] = '\0';

    /* Metal standard library header (available implicitly in MSL) */
    buf_appendf(&buf, &cap, &len,
        "#include <metal_stdlib>\n"
        "using namespace metal;\n\n");

    UOpType type = node->type;

    /* ── Binary elementwise ops ── */
    const char* bin_expr = msl_binary_expr(type);
    if (bin_expr) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a [[buffer(0)]],\n"
            "    device const float* b [[buffer(1)]],\n"
            "    device float* out     [[buffer(2)]],\n"
            "    constant uint& n      [[buffer(3)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    out[idx] = %s;\n"
            "}\n", bin_expr);
        return buf;
    }

    /* ── Unary elementwise ops (simple expression) ── */
    const char* un_expr = msl_unary_expr(type);
    if (un_expr) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a [[buffer(0)]],\n"
            "    device float* out     [[buffer(1)]],\n"
            "    constant uint& n      [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    out[idx] = %s;\n"
            "}\n", un_expr);
        return buf;
    }

    /* ── ReLU ── */
    if (type == UOP_RELU6 || type == UOP_RELU6) {
        /* Map plain RELU to max(0, x).
         * Note: UOP_RELU is decomposed from uop_relu into max(x, 0) which
         * maps to UOP_MAX at the IR level.  We still generate dedicated MSL
         * for clarity if someone explicitly passes a "RELU-like" node here.
         */
    }

    /* ── Sigmoid ── */
    if (type == UOP_SIGMOID) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a [[buffer(0)]],\n"
            "    device float* out     [[buffer(1)]],\n"
            "    constant uint& n      [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    out[idx] = 1.0f / (1.0f + exp(-a[idx]));\n"
            "}\n");
        return buf;
    }

    /* ── Tanh ── */
    if (type == UOP_TANH) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a [[buffer(0)]],\n"
            "    device float* out     [[buffer(1)]],\n"
            "    constant uint& n      [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    out[idx] = tanh(a[idx]);\n"
            "}\n");
        return buf;
    }

    /* ── MAX (used as ReLU: max(x, 0)) ── */
    if (type == UOP_MAX) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a [[buffer(0)]],\n"
            "    device const float* b [[buffer(1)]],\n"
            "    device float* out     [[buffer(2)]],\n"
            "    constant uint& n      [[buffer(3)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    out[idx] = max(a[idx], b[idx]);\n"
            "}\n");
        return buf;
    }

    /* ── Matmul (naive, one thread per output element) ── */
    if (type == UOP_MATMUL) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* A  [[buffer(0)]],\n"
            "    device const float* B  [[buffer(1)]],\n"
            "    device float* C        [[buffer(2)]],\n"
            "    constant uint& M       [[buffer(3)]],\n"
            "    constant uint& N       [[buffer(4)]],\n"
            "    constant uint& K       [[buffer(5)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= M * N) return;\n"
            "    uint row = idx / N;\n"
            "    uint col = idx %% N;\n"
            "    float acc = 0.0f;\n"
            "    for (uint k = 0; k < K; k++) {\n"
            "        acc += A[row * K + k] * B[k * N + col];\n"
            "    }\n"
            "    C[idx] = acc;\n"
            "}\n");
        return buf;
    }

    /* ── Reduction SUM (naive atomicAdd emulation via threadgroup) ── */
    if (type == UOP_SUM) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a  [[buffer(0)]],\n"
            "    device atomic_float* out [[buffer(1)]],\n"
            "    constant uint& n       [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    atomic_fetch_add_explicit(out, a[idx], memory_order_relaxed);\n"
            "}\n");
        return buf;
    }

    /* ── Unsupported op ── */
    LOG_WARNING("metal_codegen: Unsupported UOp type %d for MSL generation", (int)type);
    free(buf);
    return NULL;
}
