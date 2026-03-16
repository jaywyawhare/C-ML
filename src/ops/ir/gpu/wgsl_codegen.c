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

/* Tile size for tiled matmul */
#define TILE_SIZE 16

/** Append a formatted string to a dynamically-growing buffer. */
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

static char* wgsl_buf_new(size_t* cap, size_t* len) {
    *cap = WGSL_BUF_INIT_SIZE;
    *len = 0;
    char* buf = (char*)malloc(*cap);
    if (buf) buf[0] = '\0';
    return buf;
}


static char* wgsl_binary_kernel(UOpType type) {
    const char* expr = NULL;
    switch (type) {
    case UOP_ADD:   expr = "a[idx] + b[idx]";                                         break;
    case UOP_SUB:   expr = "a[idx] - b[idx]";                                         break;
    case UOP_MUL:   expr = "a[idx] * b[idx]";                                         break;
    case UOP_DIV:   expr = "a[idx] / (b[idx] + 1e-8)";                               break;
    case UOP_MAX:   expr = "max(a[idx], b[idx])";                                     break;
    case UOP_POW:   expr = "pow(a[idx], b[idx])";                                     break;
    case UOP_CMPLT: expr = "select(0.0, 1.0, a[idx] < b[idx])";                      break;
    case UOP_MOD:   expr = "a[idx] - floor(a[idx] / (b[idx] + 1e-8)) * b[idx]";      break;
    case UOP_IDIV:  expr = "floor(a[idx] / (b[idx] + 1e-8))";                        break;
    default:        return NULL;
    }

    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

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


static char* wgsl_unary_kernel(UOpType type) {
    const char* expr = NULL;
    switch (type) {
    case UOP_NEG:   expr = "-a[idx]";               break;
    case UOP_EXP:   expr = "exp(a[idx])";           break;
    case UOP_LOG:   expr = "log(a[idx] + 1e-8)";   break;
    case UOP_SQRT:  expr = "sqrt(abs(a[idx]))";     break;
    case UOP_ABS:   expr = "abs(a[idx])";           break;
    case UOP_SIN:   expr = "sin(a[idx])";           break;
    case UOP_COS:   expr = "cos(a[idx])";           break;
    case UOP_TAN:   expr = "tan(a[idx])";           break;
    case UOP_RECIP: expr = "1.0 / (a[idx] + 1e-8)"; break;
    case UOP_FLOOR: expr = "floor(a[idx])";         break;
    case UOP_CEIL:  expr = "ceil(a[idx])";          break;
    default:        return NULL;
    }

    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

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


static char* wgsl_sigmoid_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

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

static char* wgsl_tanh_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

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

static char* wgsl_elu_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

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
        "    result[idx] = select(exp(a[idx]) - 1.0, a[idx], a[idx] > 0.0);\n"
        "}\n");

    return buf;
}

static char* wgsl_selu_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

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
        "    let scale = 1.0507;\n"
        "    let alpha = 1.67326;\n"
        "    let x = a[idx];\n"
        "    result[idx] = scale * select(alpha * (exp(x) - 1.0), x, x > 0.0);\n"
        "}\n");

    return buf;
}

static char* wgsl_silu_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

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
        "    let x = a[idx];\n"
        "    result[idx] = x / (1.0 + exp(-x));\n"
        "}\n");

    return buf;
}

static char* wgsl_mish_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

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
        "    let x = a[idx];\n"
        "    result[idx] = x * tanh(log(1.0 + exp(x)));\n"
        "}\n");

    return buf;
}

static char* wgsl_hardswish_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

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
        "    let x = a[idx];\n"
        "    result[idx] = select(select(0.0, x * (x + 3.0) / 6.0, x >= -3.0), x, x >= 3.0);\n"
        "}\n");

    return buf;
}


static char* wgsl_fill_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

    wgsl_appendf(&buf, &cap, &len,
        "@group(0) @binding(0) var<storage, read_write> result : array<f32>;\n"
        "\n"
        "struct FillParams {\n"
        "    n : u32,\n"
        "    value : f32,\n"
        "};\n"
        "@group(0) @binding(1) var<uniform> params : FillParams;\n"
        "\n"
        "@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "    let idx = gid.x;\n"
        "    if (idx >= params.n) {\n"
        "        return;\n"
        "    }\n"
        "    result[idx] = params.value;\n"
        "}\n");

    return buf;
}


static char* wgsl_where_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

    wgsl_appendf(&buf, &cap, &len,
        "@group(0) @binding(0) var<storage, read> cond : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read> a : array<f32>;\n"
        "@group(0) @binding(2) var<storage, read> b : array<f32>;\n"
        "@group(0) @binding(3) var<storage, read_write> result : array<f32>;\n"
        "\n"
        "struct Params {\n"
        "    n : u32,\n"
        "};\n"
        "@group(0) @binding(4) var<uniform> params : Params;\n"
        "\n"
        "@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "    let idx = gid.x;\n"
        "    if (idx >= params.n) {\n"
        "        return;\n"
        "    }\n"
        "    result[idx] = select(b[idx], a[idx], cond[idx] != 0.0);\n"
        "}\n");

    return buf;
}


static char* wgsl_gather_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

    wgsl_appendf(&buf, &cap, &len,
        "@group(0) @binding(0) var<storage, read> input : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read> indices : array<f32>;\n"
        "@group(0) @binding(2) var<storage, read_write> result : array<f32>;\n"
        "\n"
        "struct GatherParams {\n"
        "    n : u32,\n"
        "    C : u32,\n"
        "};\n"
        "@group(0) @binding(3) var<uniform> params : GatherParams;\n"
        "\n"
        "@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "    let idx = gid.x;\n"
        "    if (idx >= params.n) {\n"
        "        return;\n"
        "    }\n"
        "    result[idx] = input[u32(indices[idx]) * params.C + idx %% params.C];\n"
        "}\n");

    return buf;
}


static char* wgsl_max_reduce_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

    wgsl_appendf(&buf, &cap, &len,
        "@group(0) @binding(0) var<storage, read> a : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> result : array<atomic<u32>>;\n"
        "\n"
        "struct Params {\n"
        "    n : u32,\n"
        "};\n"
        "@group(0) @binding(2) var<uniform> params : Params;\n"
        "\n"
        "// Atomic float max via CAS loop\n"
        "fn atomic_max_f32(addr : ptr<storage, atomic<u32>, read_write>, val : f32) {\n"
        "    var old = atomicLoad(addr);\n"
        "    loop {\n"
        "        let old_f = bitcast<f32>(old);\n"
        "        if (old_f >= val) {\n"
        "            break;\n"
        "        }\n"
        "        let new_val = bitcast<u32>(val);\n"
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
        "    atomic_max_f32(&result[0], a[idx]);\n"
        "}\n");

    return buf;
}

static char* wgsl_min_reduce_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

    wgsl_appendf(&buf, &cap, &len,
        "@group(0) @binding(0) var<storage, read> a : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> result : array<atomic<u32>>;\n"
        "\n"
        "struct Params {\n"
        "    n : u32,\n"
        "};\n"
        "@group(0) @binding(2) var<uniform> params : Params;\n"
        "\n"
        "// Atomic float min via CAS loop\n"
        "fn atomic_min_f32(addr : ptr<storage, atomic<u32>, read_write>, val : f32) {\n"
        "    var old = atomicLoad(addr);\n"
        "    loop {\n"
        "        let old_f = bitcast<f32>(old);\n"
        "        if (old_f <= val) {\n"
        "            break;\n"
        "        }\n"
        "        let new_val = bitcast<u32>(val);\n"
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
        "    atomic_min_f32(&result[0], a[idx]);\n"
        "}\n");

    return buf;
}

static char* wgsl_mean_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

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
        "    atomic_add_f32(&result[0], a[idx] / f32(params.n));\n"
        "}\n");

    return buf;
}

static char* wgsl_prod_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

    wgsl_appendf(&buf, &cap, &len,
        "@group(0) @binding(0) var<storage, read> a : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> result : array<atomic<u32>>;\n"
        "\n"
        "struct Params {\n"
        "    n : u32,\n"
        "};\n"
        "@group(0) @binding(2) var<uniform> params : Params;\n"
        "\n"
        "// Atomic float mul via CAS loop\n"
        "fn atomic_mul_f32(addr : ptr<storage, atomic<u32>, read_write>, val : f32) {\n"
        "    var old = atomicLoad(addr);\n"
        "    loop {\n"
        "        let new_val = bitcast<u32>(bitcast<f32>(old) * val);\n"
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
        "    atomic_mul_f32(&result[0], a[idx]);\n"
        "}\n");

    return buf;
}

static char* wgsl_sum_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

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


static char* wgsl_matmul_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

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
        "var<workgroup> As : array<array<f32, 16>, 16>;\n"
        "var<workgroup> Bs : array<array<f32, 16>, 16>;\n"
        "\n"
        "@compute @workgroup_size(16, 16)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>,\n"
        "        @builtin(local_invocation_id) lid : vec3<u32>) {\n"
        "    let row = gid.y;\n"
        "    let col = gid.x;\n"
        "    let lr  = lid.y;\n"
        "    let lc  = lid.x;\n"
        "\n"
        "    var acc : f32 = 0.0;\n"
        "    let num_tiles = (params.K + 15u) / 16u;\n"
        "\n"
        "    for (var t : u32 = 0u; t < num_tiles; t = t + 1u) {\n"
        "        let a_col = t * 16u + lc;\n"
        "        let b_row = t * 16u + lr;\n"
        "\n"
        "        if (row < params.M && a_col < params.K) {\n"
        "            As[lr][lc] = A[row * params.K + a_col];\n"
        "        } else {\n"
        "            As[lr][lc] = 0.0;\n"
        "        }\n"
        "\n"
        "        if (b_row < params.K && col < params.N) {\n"
        "            Bs[lr][lc] = B[b_row * params.N + col];\n"
        "        } else {\n"
        "            Bs[lr][lc] = 0.0;\n"
        "        }\n"
        "\n"
        "        workgroupBarrier();\n"
        "\n"
        "        for (var k : u32 = 0u; k < 16u; k = k + 1u) {\n"
        "            acc = acc + As[lr][k] * Bs[k][lc];\n"
        "        }\n"
        "\n"
        "        workgroupBarrier();\n"
        "    }\n"
        "\n"
        "    if (row < params.M && col < params.N) {\n"
        "        C[row * params.N + col] = acc;\n"
        "    }\n"
        "}\n");

    return buf;
}


static char* wgsl_conv2d_kernel(void) {
    size_t cap, len;
    char* buf = wgsl_buf_new(&cap, &len);
    if (!buf) return NULL;

    wgsl_appendf(&buf, &cap, &len,
        "@group(0) @binding(0) var<storage, read> input   : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read> weight  : array<f32>;\n"
        "@group(0) @binding(2) var<storage, read_write> result : array<f32>;\n"
        "\n"
        "struct ConvParams {\n"
        "    N      : u32,\n"
        "    Cin    : u32,\n"
        "    Cout   : u32,\n"
        "    H      : u32,\n"
        "    W      : u32,\n"
        "    KH     : u32,\n"
        "    KW     : u32,\n"
        "    stride : u32,\n"
        "    pad    : u32,\n"
        "    OH     : u32,\n"
        "    OW     : u32,\n"
        "};\n"
        "@group(0) @binding(3) var<uniform> params : ConvParams;\n"
        "\n"
        "@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "    let idx = gid.x;\n"
        "    let total = params.N * params.Cout * params.OH * params.OW;\n"
        "    if (idx >= total) {\n"
        "        return;\n"
        "    }\n"
        "\n"
        "    let ow = idx %% params.OW;\n"
        "    let tmp1 = idx / params.OW;\n"
        "    let oh = tmp1 %% params.OH;\n"
        "    let tmp2 = tmp1 / params.OH;\n"
        "    let co = tmp2 %% params.Cout;\n"
        "    let n  = tmp2 / params.Cout;\n"
        "\n"
        "    var acc : f32 = 0.0;\n"
        "    for (var ci : u32 = 0u; ci < params.Cin; ci = ci + 1u) {\n"
        "        for (var kh : u32 = 0u; kh < params.KH; kh = kh + 1u) {\n"
        "            for (var kw : u32 = 0u; kw < params.KW; kw = kw + 1u) {\n"
        "                let ih = oh * params.stride + kh - params.pad;\n"
        "                let iw = ow * params.stride + kw - params.pad;\n"
        "                if (ih < params.H && iw < params.W) {\n"
        "                    let in_idx  = ((n * params.Cin + ci) * params.H + ih) * params.W + iw;\n"
        "                    let w_idx   = ((co * params.Cin + ci) * params.KH + kh) * params.KW + kw;\n"
        "                    acc = acc + input[in_idx] * weight[w_idx];\n"
        "                }\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "    result[idx] = acc;\n"
        "}\n");

    return buf;
}


/**
 * @brief Generate WGSL compute shader source for a single IR node.
 *
 * Supported UOp types:
 *   Binary:      ADD, SUB, MUL, DIV, MAX, POW, CMPLT, MOD, IDIV
 *   Unary:       NEG, EXP, LOG, SQRT, ABS, SIN, COS, TAN, RECIP, FLOOR, CEIL
 *   Activations: SIGMOID, TANH, ELU, SELU, SILU, MISH, HARDSWISH
 *   Reductions:  SUM, MAX_REDUCE, MIN_REDUCE, MEAN, PROD
 *   Special:     MATMUL, CONV2D, FILL, WHERE, GATHER
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
    case UOP_MAX:
    case UOP_POW:
    case UOP_CMPLT:
    case UOP_MOD:
    case UOP_IDIV:
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
    case UOP_RECIP:
    case UOP_FLOOR:
    case UOP_CEIL:
        return wgsl_unary_kernel(type);
    default:
        break;
    }

    /* Activations */
    if (type == UOP_SIGMOID)   return wgsl_sigmoid_kernel();
    if (type == UOP_TANH)      return wgsl_tanh_kernel();
    if (type == UOP_ELU)       return wgsl_elu_kernel();
    if (type == UOP_SELU)      return wgsl_selu_kernel();
    if (type == UOP_SILU)      return wgsl_silu_kernel();
    if (type == UOP_MISH)      return wgsl_mish_kernel();
    if (type == UOP_HARDSWISH) return wgsl_hardswish_kernel();

    /* Reductions */
    if (type == UOP_SUM)        return wgsl_sum_kernel();
    if (type == UOP_MAX_REDUCE) return wgsl_max_reduce_kernel();
    if (type == UOP_MIN_REDUCE) return wgsl_min_reduce_kernel();
    if (type == UOP_MEAN)       return wgsl_mean_kernel();
    if (type == UOP_PROD)       return wgsl_prod_kernel();

    /* Special ops */
    if (type == UOP_MATMUL) return wgsl_matmul_kernel();
    if (type == UOP_CONV2D) return wgsl_conv2d_kernel();
    if (type == UOP_FILL)   return wgsl_fill_kernel();
    if (type == UOP_WHERE)  return wgsl_where_kernel();
    if (type == UOP_GATHER) return wgsl_gather_kernel();

    LOG_WARNING("wgsl_codegen: Unsupported UOp type %d", (int)type);
    return NULL;
}


/**
 * @brief Execute a full IR graph on the WebGPU backend.
 *
 * Walks each node in the graph, generates a WGSL shader via
 * cml_wgsl_generate(), compiles it, uploads input data, launches
 * the kernel, and downloads the result.
 *
 * @param backend  Initialised WebGPU backend context
 * @param graph    IR graph to execute
 * @return 0 on success, negative on failure
 */
int cml_webgpu_execute_graph(CMLWebGPUBackend* backend, CMLGraph_t graph) {
    if (!backend || !backend->initialized) {
        LOG_ERROR("webgpu_execute_graph: backend not initialised");
        return -1;
    }
    if (!graph) {
        LOG_ERROR("webgpu_execute_graph: NULL graph");
        return -1;
    }

    struct IRNode* node = graph->head;
    while (node) {
        /* 1. Generate WGSL source for this node */
        char* wgsl_src = cml_wgsl_generate(node);
        if (!wgsl_src) {
            LOG_ERROR("webgpu_execute_graph: codegen failed for UOp %d",
                      (int)node->type);
            return -1;
        }

        /* 2. Compile the WGSL source into a kernel */
        CMLWebGPUKernel* kernel =
            cml_webgpu_compile_wgsl(backend, wgsl_src, "main");
        free(wgsl_src);
        if (!kernel) {
            LOG_ERROR("webgpu_execute_graph: compile failed for UOp %d",
                      (int)node->type);
            return -1;
        }

        /* 3. Upload input tensors to GPU buffers */
        int num_inputs = node->num_inputs;
        void** gpu_buffers = NULL;
        size_t* buffer_sizes = NULL;
        size_t out_size = 0;
        int total_buffers = num_inputs + 1; /* inputs + output */

        if (node->output && node->output->data) {
            out_size = 1;
            for (int d = 0; d < node->output->ndim; d++) {
                out_size *= (size_t)node->output->shape[d];
            }
            out_size *= sizeof(float);
        }

        gpu_buffers  = (void**)calloc((size_t)total_buffers, sizeof(void*));
        buffer_sizes = (size_t*)calloc((size_t)total_buffers, sizeof(size_t));
        if (!gpu_buffers || !buffer_sizes) {
            free(gpu_buffers);
            free(buffer_sizes);
            cml_webgpu_kernel_free(kernel);
            LOG_ERROR("webgpu_execute_graph: alloc failed");
            return -1;
        }

        int ok = 0;
        for (int i = 0; i < num_inputs; i++) {
            Tensor* t = node->inputs[i];
            if (!t || !t->data) { ok = -1; break; }
            size_t sz = 1;
            for (int d = 0; d < t->ndim; d++) {
                sz *= (size_t)t->shape[d];
            }
            sz *= sizeof(float);
            buffer_sizes[i] = sz;
            gpu_buffers[i] = cml_webgpu_alloc(backend, sz);
            if (!gpu_buffers[i]) { ok = -1; break; }
            if (cml_webgpu_upload(backend, gpu_buffers[i], t->data, sz) != 0) {
                ok = -1;
                break;
            }
        }

        if (ok == 0 && out_size > 0) {
            buffer_sizes[num_inputs] = out_size;
            gpu_buffers[num_inputs] = cml_webgpu_alloc(backend, out_size);
            if (!gpu_buffers[num_inputs]) ok = -1;
        }

        /* 4. Launch kernel */
        if (ok == 0) {
            size_t n_elements = out_size / sizeof(float);
            size_t workgroups[3] = { (n_elements + 255) / 256, 1, 1 };
            ok = cml_webgpu_launch_kernel(backend, kernel, workgroups,
                                          gpu_buffers, buffer_sizes,
                                          total_buffers);
        }

        /* 5. Download result */
        if (ok == 0 && out_size > 0 && node->output && node->output->data) {
            ok = cml_webgpu_download(backend, node->output->data,
                                     gpu_buffers[num_inputs], out_size);
        }

        /* Cleanup GPU buffers */
        for (int i = 0; i < total_buffers; i++) {
            if (gpu_buffers[i]) {
                cml_webgpu_free(backend, gpu_buffers[i]);
            }
        }
        free(gpu_buffers);
        free(buffer_sizes);
        cml_webgpu_kernel_free(kernel);

        if (ok != 0) {
            LOG_ERROR("webgpu_execute_graph: execution failed for UOp %d",
                      (int)node->type);
            return -1;
        }

        node->is_executed = true;
        node = node->next;
    }

    return 0;
}
