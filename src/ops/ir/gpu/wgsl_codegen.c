/**
 * @file wgsl_codegen.c
 * @brief WGSL (WebGPU Shading Language) code generation from IR nodes
 *
 * Translates IR nodes into WGSL compute shader source strings.
 * Each generated shader uses:
 *   @group(0) @binding(N) for storage buffers
 *   @compute @workgroup_size(256) fn main(...)
 *   @builtin(global_invocation_id) for thread indexing
 */

#include "ops/ir/gpu/webgpu_backend.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include "core/logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

/* Initial allocation size for the WGSL source buffer */
#define WGSL_BUF_INIT_SIZE 4096

/**
 * @brief Append a formatted string to a dynamically-growing buffer.
 */
static void wgsl_appendf(char** buf, size_t* cap, size_t* len,
                          const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int needed = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);

    if (needed < 0) return;

    while (*len + (size_t)needed + 1 > *cap) {
        *cap *= 2;
        char* tmp = (char*)realloc(*buf, *cap);
        if (!tmp) {
            LOG_ERROR("wgsl_codegen: realloc failed");
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
 * @brief Generate a WGSL binary elementwise compute shader.
 *
 * Layout:
 *   @group(0) @binding(0) var<storage, read>       a : array<f32>;
 *   @group(0) @binding(1) var<storage, read>       b : array<f32>;
 *   @group(0) @binding(2) var<storage, read_write>  out : array<f32>;
 *   @group(0) @binding(3) var<uniform>              n : u32;
 */
static char* wgsl_binary_kernel(UOpType type) {
    const char* expr = NULL;
    switch (type) {
    case UOP_ADD: expr = "a[idx] + b[idx]";               break;
    case UOP_SUB: expr = "a[idx] - b[idx]";               break;
    case UOP_MUL: expr = "a[idx] * b[idx]";               break;
    case UOP_DIV: expr = "a[idx] / (b[idx] + 1e-8)";     break;
    default:      return NULL;
    }

    size_t cap = WGSL_BUF_INIT_SIZE;
    size_t len = 0;
    char* buf = (char*)malloc(cap);
    if (!buf) return NULL;
    buf[0] = '\0';

    wgsl_appendf(&buf, &cap, &len,
        "@group(0) @binding(0) var<storage, read> a : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read> b : array<f32>;\n"
        "@group(0) @binding(2) var<storage, read_write> result : array<f32>;\n"
        "\n"
        "struct Params {\n"
        "    n : u32,\n"
        "};\n"
        "@group(0) @binding(3) var<uniform> params : Params;\n"
        "\n"
        "@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "    let idx = gid.x;\n"
        "    if (idx >= params.n) {\n"
        "        return;\n"
        "    }\n"
        "    result[idx] = %s;\n"
        "}\n", expr);

    return buf;
}

/**
 * @brief Generate a WGSL unary elementwise compute shader.
 *
 * Layout:
 *   @group(0) @binding(0) var<storage, read>       a : array<f32>;
 *   @group(0) @binding(1) var<storage, read_write>  out : array<f32>;
 *   @group(0) @binding(2) var<uniform>              n : u32;
 */
static char* wgsl_unary_kernel(UOpType type) {
    const char* expr = NULL;
    switch (type) {
    case UOP_NEG:  expr = "-a[idx]";                      break;
    case UOP_EXP:  expr = "exp(a[idx])";                  break;
    case UOP_LOG:  expr = "log(a[idx] + 1e-8)";          break;
    case UOP_SQRT: expr = "sqrt(abs(a[idx]))";            break;
    case UOP_ABS:  expr = "abs(a[idx])";                  break;
    case UOP_SIN:  expr = "sin(a[idx])";                  break;
    case UOP_COS:  expr = "cos(a[idx])";                  break;
    case UOP_TAN:  expr = "tan(a[idx])";                  break;
    default:       return NULL;
    }

    size_t cap = WGSL_BUF_INIT_SIZE;
    size_t len = 0;
    char* buf = (char*)malloc(cap);
    if (!buf) return NULL;
    buf[0] = '\0';

    wgsl_appendf(&buf, &cap, &len,
        "@group(0) @binding(0) var<storage, read> a : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> result : array<f32>;\n"
        "\n"
        "struct Params {\n"
        "    n : u32,\n"
        "};\n"
        "@group(0) @binding(2) var<uniform> params : Params;\n"
        "\n"
        "@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "    let idx = gid.x;\n"
        "    if (idx >= params.n) {\n"
        "        return;\n"
        "    }\n"
        "    result[idx] = %s;\n"
        "}\n", expr);

    return buf;
}

/**
 * @brief Generate a WGSL sigmoid compute shader.
 */
static char* wgsl_sigmoid_kernel(void) {
    size_t cap = WGSL_BUF_INIT_SIZE;
    size_t len = 0;
    char* buf = (char*)malloc(cap);
    if (!buf) return NULL;
    buf[0] = '\0';

    wgsl_appendf(&buf, &cap, &len,
        "@group(0) @binding(0) var<storage, read> a : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> result : array<f32>;\n"
        "\n"
        "struct Params {\n"
        "    n : u32,\n"
        "};\n"
        "@group(0) @binding(2) var<uniform> params : Params;\n"
        "\n"
        "@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "    let idx = gid.x;\n"
        "    if (idx >= params.n) {\n"
        "        return;\n"
        "    }\n"
        "    result[idx] = 1.0 / (1.0 + exp(-a[idx]));\n"
        "}\n");

    return buf;
}

/**
 * @brief Generate a WGSL tanh compute shader.
 */
static char* wgsl_tanh_kernel(void) {
    size_t cap = WGSL_BUF_INIT_SIZE;
    size_t len = 0;
    char* buf = (char*)malloc(cap);
    if (!buf) return NULL;
    buf[0] = '\0';

    wgsl_appendf(&buf, &cap, &len,
        "@group(0) @binding(0) var<storage, read> a : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> result : array<f32>;\n"
        "\n"
        "struct Params {\n"
        "    n : u32,\n"
        "};\n"
        "@group(0) @binding(2) var<uniform> params : Params;\n"
        "\n"
        "@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "    let idx = gid.x;\n"
        "    if (idx >= params.n) {\n"
        "        return;\n"
        "    }\n"
        "    result[idx] = tanh(a[idx]);\n"
        "}\n");

    return buf;
}

/**
 * @brief Generate a WGSL ReLU compute shader (max(0, x)).
 */
static char* wgsl_relu_kernel(void) {
    size_t cap = WGSL_BUF_INIT_SIZE;
    size_t len = 0;
    char* buf = (char*)malloc(cap);
    if (!buf) return NULL;
    buf[0] = '\0';

    wgsl_appendf(&buf, &cap, &len,
        "@group(0) @binding(0) var<storage, read> a : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> result : array<f32>;\n"
        "\n"
        "struct Params {\n"
        "    n : u32,\n"
        "};\n"
        "@group(0) @binding(2) var<uniform> params : Params;\n"
        "\n"
        "@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "    let idx = gid.x;\n"
        "    if (idx >= params.n) {\n"
        "        return;\n"
        "    }\n"
        "    result[idx] = max(a[idx], 0.0);\n"
        "}\n");

    return buf;
}

/**
 * @brief Generate a WGSL matmul compute shader (naive).
 */
static char* wgsl_matmul_kernel(void) {
    size_t cap = WGSL_BUF_INIT_SIZE;
    size_t len = 0;
    char* buf = (char*)malloc(cap);
    if (!buf) return NULL;
    buf[0] = '\0';

    wgsl_appendf(&buf, &cap, &len,
        "@group(0) @binding(0) var<storage, read> A : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read> B : array<f32>;\n"
        "@group(0) @binding(2) var<storage, read_write> C : array<f32>;\n"
        "\n"
        "struct MatParams {\n"
        "    M : u32,\n"
        "    N : u32,\n"
        "    K : u32,\n"
        "};\n"
        "@group(0) @binding(3) var<uniform> params : MatParams;\n"
        "\n"
        "@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "    let idx = gid.x;\n"
        "    let total = params.M * params.N;\n"
        "    if (idx >= total) {\n"
        "        return;\n"
        "    }\n"
        "    let row = idx / params.N;\n"
        "    let col = idx %% params.N;\n"
        "    var acc : f32 = 0.0;\n"
        "    for (var k : u32 = 0u; k < params.K; k = k + 1u) {\n"
        "        acc = acc + A[row * params.K + k] * B[k * params.N + col];\n"
        "    }\n"
        "    C[idx] = acc;\n"
        "}\n");

    return buf;
}

/**
 * @brief Generate a WGSL sum reduction compute shader (naive atomic).
 */
static char* wgsl_sum_kernel(void) {
    size_t cap = WGSL_BUF_INIT_SIZE;
    size_t len = 0;
    char* buf = (char*)malloc(cap);
    if (!buf) return NULL;
    buf[0] = '\0';

    /* Note: WGSL does not have a native atomicAdd for f32 in storage buffers.
     * We use an atomic<u32> with bitcast for a simple lock-free accumulator.
     * A production implementation would use a two-pass tree reduction. */
    wgsl_appendf(&buf, &cap, &len,
        "@group(0) @binding(0) var<storage, read> a : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> result : array<atomic<u32>>;\n"
        "\n"
        "struct Params {\n"
        "    n : u32,\n"
        "};\n"
        "@group(0) @binding(2) var<uniform> params : Params;\n"
        "\n"
        "// Atomic float add via CAS loop\n"
        "fn atomic_add_f32(addr : ptr<storage, atomic<u32>, read_write>, val : f32) {\n"
        "    var old = atomicLoad(addr);\n"
        "    loop {\n"
        "        let new_val = bitcast<u32>(bitcast<f32>(old) + val);\n"
        "        let result = atomicCompareExchangeWeak(addr, old, new_val);\n"
        "        if (result.exchanged) {\n"
        "            break;\n"
        "        }\n"
        "        old = result.old_value;\n"
        "    }\n"
        "}\n"
        "\n"
        "@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "    let idx = gid.x;\n"
        "    if (idx >= params.n) {\n"
        "        return;\n"
        "    }\n"
        "    atomic_add_f32(&result[0], a[idx]);\n"
        "}\n");

    return buf;
}

/**
 * @brief Generate WGSL compute shader source for a single IR node.
 *
 * Supported UOp types:
 *   Binary:  ADD, SUB, MUL, DIV
 *   Unary:   NEG, EXP, LOG, SQRT
 *   Special: SIGMOID, TANH, MAX (as ReLU), MATMUL, SUM
 *
 * @param node  IR node to translate
 * @return Heap-allocated WGSL source string (caller must free), or NULL
 */
char* cml_wgsl_generate(struct IRNode* node) {
    if (!node) {
        LOG_ERROR("wgsl_codegen: NULL node");
        return NULL;
    }

    UOpType type = node->type;

    /* Binary ops */
    switch (type) {
    case UOP_ADD:
    case UOP_SUB:
    case UOP_MUL:
    case UOP_DIV:
        return wgsl_binary_kernel(type);
    default:
        break;
    }

    /* Unary ops (simple expression) */
    switch (type) {
    case UOP_NEG:
    case UOP_EXP:
    case UOP_LOG:
    case UOP_SQRT:
    case UOP_ABS:
    case UOP_SIN:
    case UOP_COS:
    case UOP_TAN:
        return wgsl_unary_kernel(type);
    default:
        break;
    }

    /* Activations and special ops */
    if (type == UOP_SIGMOID) return wgsl_sigmoid_kernel();
    if (type == UOP_TANH)    return wgsl_tanh_kernel();
    if (type == UOP_MAX)     return wgsl_relu_kernel();
    if (type == UOP_MATMUL)  return wgsl_matmul_kernel();
    if (type == UOP_SUM)     return wgsl_sum_kernel();

    LOG_WARNING("wgsl_codegen: Unsupported UOp type %d", (int)type);
    return NULL;
}
