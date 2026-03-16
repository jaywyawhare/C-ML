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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Maximum size for a generated MSL kernel source string */
#define MSL_BUF_SIZE 4096

/* Tile size for tiled matmul */
#define TILE_SIZE 16

/** Append a formatted string to a dynamically-growing buffer. */
static void buf_appendf(char** buf, size_t* cap, size_t* len,
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

static const char* msl_binary_expr(UOpType type) {
    switch (type) {
    case UOP_ADD:   return "a[idx] + b[idx]";
    case UOP_SUB:   return "a[idx] - b[idx]";
    case UOP_MUL:   return "a[idx] * b[idx]";
    case UOP_DIV:   return "a[idx] / (b[idx] + 1e-8f)";
    case UOP_POW:   return "pow(a[idx], b[idx])";
    case UOP_CMPLT: return "float(a[idx] < b[idx])";
    case UOP_MOD:   return "fmod(a[idx], b[idx])";
    case UOP_IDIV:  return "floor(a[idx] / (b[idx] + 1e-8f))";
    default:        return NULL;
    }
}

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
    case UOP_FLOOR:   return "floor(a[idx])";
    case UOP_CEIL:    return "ceil(a[idx])";
    default:          return NULL;
    }
}

/**
 * @brief Generate Metal Shading Language source for a compute kernel
 *        corresponding to a single IR node.
 *
 * Supported UOp types:
 *   Binary:  ADD, SUB, MUL, DIV, POW, CMPLT, MOD, IDIV
 *   Unary:   NEG, EXP, LOG, SQRT, RECIP, ABS, SIN, COS, TAN, FLOOR, CEIL
 *   Activations: RELU6, SIGMOID, TANH, ELU, SELU, SILU, MISH, HARDSWISH
 *   Reductions: SUM, MAX_REDUCE, MIN_REDUCE, MEAN, PROD
 *   Special: MATMUL (tiled), CONV2D, FILL, WHERE, GATHER, MAX
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

    buf_appendf(&buf, &cap, &len,
        "#include <metal_stdlib>\n"
        "using namespace metal;\n\n");

    UOpType type = node->type;

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

    if (type == UOP_RELU6) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a [[buffer(0)]],\n"
            "    device float* out     [[buffer(1)]],\n"
            "    constant uint& n      [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    float x = a[idx];\n"
            "    out[idx] = min(max(x, 0.0f), 6.0f);\n"
            "}\n");
        return buf;
    }

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

    if (type == UOP_ELU) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a [[buffer(0)]],\n"
            "    device float* out     [[buffer(1)]],\n"
            "    constant uint& n      [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    float x = a[idx];\n"
            "    out[idx] = x > 0.0f ? x : (exp(x) - 1.0f);\n"
            "}\n");
        return buf;
    }

    if (type == UOP_SELU) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a [[buffer(0)]],\n"
            "    device float* out     [[buffer(1)]],\n"
            "    constant uint& n      [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    float scale = 1.0507f;\n"
            "    float alpha = 1.67326f;\n"
            "    float x = a[idx];\n"
            "    out[idx] = scale * (x > 0.0f ? x : alpha * (exp(x) - 1.0f));\n"
            "}\n");
        return buf;
    }

    if (type == UOP_SILU) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a [[buffer(0)]],\n"
            "    device float* out     [[buffer(1)]],\n"
            "    constant uint& n      [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    float x = a[idx];\n"
            "    out[idx] = x / (1.0f + exp(-x));\n"
            "}\n");
        return buf;
    }

    if (type == UOP_MISH) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a [[buffer(0)]],\n"
            "    device float* out     [[buffer(1)]],\n"
            "    constant uint& n      [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    float x = a[idx];\n"
            "    out[idx] = x * tanh(log(1.0f + exp(x)));\n"
            "}\n");
        return buf;
    }

    if (type == UOP_HARDSWISH) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a [[buffer(0)]],\n"
            "    device float* out     [[buffer(1)]],\n"
            "    constant uint& n      [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    float x = a[idx];\n"
            "    out[idx] = x > 3.0f ? x : (x < -3.0f ? 0.0f : x * (x + 3.0f) / 6.0f);\n"
            "}\n");
        return buf;
    }

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

    if (type == UOP_FILL) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    constant float& fill_val [[buffer(0)]],\n"
            "    device float* out        [[buffer(1)]],\n"
            "    constant uint& n         [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    out[idx] = fill_val;\n"
            "}\n");
        return buf;
    }

    if (type == UOP_WHERE) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* cond [[buffer(0)]],\n"
            "    device const float* a    [[buffer(1)]],\n"
            "    device const float* b    [[buffer(2)]],\n"
            "    device float* out        [[buffer(3)]],\n"
            "    constant uint& n         [[buffer(4)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    out[idx] = cond[idx] != 0.0f ? a[idx] : b[idx];\n"
            "}\n");
        return buf;
    }

    if (type == UOP_GATHER) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* input   [[buffer(0)]],\n"
            "    device const float* indices [[buffer(1)]],\n"
            "    device float* out           [[buffer(2)]],\n"
            "    constant uint& n            [[buffer(3)]],\n"
            "    constant uint& C            [[buffer(4)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    out[idx] = input[(int)indices[idx] * C + idx %% C];\n"
            "}\n");
        return buf;
    }

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

    if (type == UOP_MAX_REDUCE) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a    [[buffer(0)]],\n"
            "    device atomic_uint* out  [[buffer(1)]],\n"
            "    constant uint& n         [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    float val = a[idx];\n"
            "    uint val_bits = as_type<uint>(val);\n"
            "    uint old_bits = atomic_load_explicit(out, memory_order_relaxed);\n"
            "    while (val > as_type<float>(old_bits)) {\n"
            "        if (atomic_compare_exchange_weak_explicit(out, &old_bits, val_bits,\n"
            "                memory_order_relaxed, memory_order_relaxed)) {\n"
            "            break;\n"
            "        }\n"
            "    }\n"
            "}\n");
        return buf;
    }

    if (type == UOP_MIN_REDUCE) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a    [[buffer(0)]],\n"
            "    device atomic_uint* out  [[buffer(1)]],\n"
            "    constant uint& n         [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    float val = a[idx];\n"
            "    uint val_bits = as_type<uint>(val);\n"
            "    uint old_bits = atomic_load_explicit(out, memory_order_relaxed);\n"
            "    while (val < as_type<float>(old_bits)) {\n"
            "        if (atomic_compare_exchange_weak_explicit(out, &old_bits, val_bits,\n"
            "                memory_order_relaxed, memory_order_relaxed)) {\n"
            "            break;\n"
            "        }\n"
            "    }\n"
            "}\n");
        return buf;
    }

    if (type == UOP_MEAN) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a  [[buffer(0)]],\n"
            "    device atomic_float* out [[buffer(1)]],\n"
            "    constant uint& n       [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    atomic_fetch_add_explicit(out, a[idx] / float(n), memory_order_relaxed);\n"
            "}\n");
        return buf;
    }

    if (type == UOP_PROD) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a    [[buffer(0)]],\n"
            "    device atomic_uint* out  [[buffer(1)]],\n"
            "    constant uint& n         [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    float val = a[idx];\n"
            "    uint old_bits = atomic_load_explicit(out, memory_order_relaxed);\n"
            "    while (true) {\n"
            "        float old_val = as_type<float>(old_bits);\n"
            "        float new_val = old_val * val;\n"
            "        uint new_bits = as_type<uint>(new_val);\n"
            "        if (atomic_compare_exchange_weak_explicit(out, &old_bits, new_bits,\n"
            "                memory_order_relaxed, memory_order_relaxed)) {\n"
            "            break;\n"
            "        }\n"
            "    }\n"
            "}\n");
        return buf;
    }

    if (type == UOP_MATMUL) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* A  [[buffer(0)]],\n"
            "    device const float* B  [[buffer(1)]],\n"
            "    device float* C        [[buffer(2)]],\n"
            "    constant uint& M       [[buffer(3)]],\n"
            "    constant uint& N       [[buffer(4)]],\n"
            "    constant uint& K       [[buffer(5)]],\n"
            "    uint2 tid [[thread_position_in_threadgroup]],\n"
            "    uint2 gid [[threadgroup_position_in_grid]]\n"
            ") {\n"
            "    threadgroup float As[16][16];\n"
            "    threadgroup float Bs[16][16];\n"
            "\n"
            "    uint row = gid.y * 16 + tid.y;\n"
            "    uint col = gid.x * 16 + tid.x;\n"
            "\n"
            "    float acc = 0.0f;\n"
            "    uint num_tiles = (K + 15) / 16;\n"
            "\n"
            "    for (uint t = 0; t < num_tiles; t++) {\n"
            "        uint a_col = t * 16 + tid.x;\n"
            "        uint b_row = t * 16 + tid.y;\n"
            "\n"
            "        As[tid.y][tid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;\n"
            "        Bs[tid.y][tid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;\n"
            "\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "\n"
            "        for (uint k = 0; k < 16; k++) {\n"
            "            acc += As[tid.y][k] * Bs[k][tid.x];\n"
            "        }\n"
            "\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    }\n"
            "\n"
            "    if (row < M && col < N) {\n"
            "        C[row * N + col] = acc;\n"
            "    }\n"
            "}\n");
        return buf;
    }

    if (type == UOP_CONV2D) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* input  [[buffer(0)]],\n"
            "    device const float* weight [[buffer(1)]],\n"
            "    device const float* bias   [[buffer(2)]],\n"
            "    device float* out          [[buffer(3)]],\n"
            "    constant uint& batch       [[buffer(4)]],\n"
            "    constant uint& in_c        [[buffer(5)]],\n"
            "    constant uint& in_h        [[buffer(6)]],\n"
            "    constant uint& in_w        [[buffer(7)]],\n"
            "    constant uint& out_c       [[buffer(8)]],\n"
            "    constant uint& out_h       [[buffer(9)]],\n"
            "    constant uint& out_w       [[buffer(10)]],\n"
            "    constant uint& kH          [[buffer(11)]],\n"
            "    constant uint& kW          [[buffer(12)]],\n"
            "    constant uint& stride_h    [[buffer(13)]],\n"
            "    constant uint& stride_w    [[buffer(14)]],\n"
            "    constant uint& pad_h       [[buffer(15)]],\n"
            "    constant uint& pad_w       [[buffer(16)]],\n"
            "    constant uint& has_bias    [[buffer(17)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    uint total = batch * out_c * out_h * out_w;\n"
            "    if (idx >= total) return;\n"
            "\n"
            "    uint ow = idx %% out_w;\n"
            "    uint tmp = idx / out_w;\n"
            "    uint oh = tmp %% out_h;\n"
            "    tmp = tmp / out_h;\n"
            "    uint oc = tmp %% out_c;\n"
            "    uint b = tmp / out_c;\n"
            "\n"
            "    float acc = has_bias ? bias[oc] : 0.0f;\n"
            "\n"
            "    for (uint ic = 0; ic < in_c; ic++) {\n"
            "        for (uint fh = 0; fh < kH; fh++) {\n"
            "            for (uint fw = 0; fw < kW; fw++) {\n"
            "                int ih = (int)(oh * stride_h + fh) - (int)pad_h;\n"
            "                int iw = (int)(ow * stride_w + fw) - (int)pad_w;\n"
            "                if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {\n"
            "                    uint in_idx = b * (in_c * in_h * in_w) +\n"
            "                                 ic * (in_h * in_w) +\n"
            "                                 (uint)ih * in_w + (uint)iw;\n"
            "                    uint w_idx = oc * (in_c * kH * kW) +\n"
            "                                ic * (kH * kW) +\n"
            "                                fh * kW + fw;\n"
            "                    acc += input[in_idx] * weight[w_idx];\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "\n"
            "    out[idx] = acc;\n"
            "}\n");
        return buf;
    }

    LOG_WARNING("metal_codegen: Unsupported UOp type %d for MSL generation", (int)type);
    free(buf);
    return NULL;
}

/**
 * @brief Execute an entire IR graph on the Metal backend.
 *
 * Walks every node in the graph, generates MSL, compiles it, copies tensor
 * data into Metal buffers, launches the kernel, and copies results back.
 * Metal uses unified memory so copies are cheap but still explicit through
 * the cml_metal_upload / cml_metal_download API.
 *
 * @param backend  Initialized Metal backend context
 * @param graph    IR graph to execute
 * @return 0 on success, negative on failure
 */
#ifdef CML_HAS_METAL
int cml_metal_execute_graph(CMLMetalBackend* backend, CMLGraph_t graph) {
    if (!backend || !backend->initialized) {
        LOG_ERROR("metal_execute_graph: backend not initialized");
        return -1;
    }
    if (!graph) {
        LOG_ERROR("metal_execute_graph: NULL graph");
        return -1;
    }

    struct IRNode* node = graph->head;

    while (node) {
        /* Skip already-executed or movement-only nodes */
        if (node->is_executed) {
            node = node->next;
            continue;
        }

            char* msl_source = cml_metal_generate_msl(node);
        if (!msl_source) {
            LOG_WARNING("metal_execute_graph: skipping unsupported node type %d",
                        (int)node->type);
            node = node->next;
            continue;
        }

            CMLMetalKernel* kernel = cml_metal_compile_msl(backend, msl_source, "cml_kernel");
        free(msl_source);
        if (!kernel) {
            LOG_ERROR("metal_execute_graph: failed to compile kernel for node type %d",
                      (int)node->type);
            return -1;
        }

            int max_bufs = node->num_inputs + 8; /* inputs + output + scalar params */
        void** metal_buffers = (void**)calloc((size_t)max_bufs, sizeof(void*));
        if (!metal_buffers) {
            cml_metal_kernel_free(kernel);
            return -1;
        }
        int buf_count = 0;

        for (int i = 0; i < node->num_inputs; i++) {
            Tensor* t = node->inputs[i];
            if (!t || !t->data) continue;
            size_t bytes = t->numel * sizeof(float);
            void* mbuf = cml_metal_alloc(backend, bytes);
            if (!mbuf) {
                LOG_ERROR("metal_execute_graph: alloc failed for input %d", i);
                /* Cleanup previously allocated buffers */
                for (int j = 0; j < buf_count; j++) {
                    cml_metal_free(backend, metal_buffers[j]);
                }
                free(metal_buffers);
                cml_metal_kernel_free(kernel);
                return -1;
            }
            cml_metal_upload(backend, mbuf, t->data, bytes);
            metal_buffers[buf_count++] = mbuf;
        }

        Tensor* out = node->output;
        size_t out_bytes = out ? out->numel * sizeof(float) : 0;
        void* out_buf = NULL;
        if (out && out_bytes > 0) {
            out_buf = cml_metal_alloc(backend, out_bytes);
            if (!out_buf) {
                LOG_ERROR("metal_execute_graph: alloc failed for output");
                for (int j = 0; j < buf_count; j++) {
                    cml_metal_free(backend, metal_buffers[j]);
                }
                free(metal_buffers);
                cml_metal_kernel_free(kernel);
                return -1;
            }
            metal_buffers[buf_count++] = out_buf;
        }

            size_t total_threads = out ? out->numel : 1;
        size_t block_size = 256;
        if (node->type == UOP_MATMUL) {
            /* Tiled matmul uses 2D threadgroups of TILE_SIZE x TILE_SIZE */
            size_t M_dim = (node->num_inputs >= 1 && node->inputs[0])
                            ? (size_t)node->inputs[0]->shape[0] : 1;
            size_t N_dim = (node->num_inputs >= 2 && node->inputs[1])
                            ? (size_t)node->inputs[1]->shape[1] : 1;
            size_t grid[3]  = { (N_dim + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
                                (M_dim + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
                                1 };
            size_t block[3] = { TILE_SIZE, TILE_SIZE, 1 };
            cml_metal_launch_kernel(backend, kernel, grid, block,
                                    metal_buffers, buf_count);
        } else {
            size_t grid_size = (total_threads + block_size - 1) / block_size * block_size;
            size_t grid[3]  = { grid_size, 1, 1 };
            size_t block[3] = { block_size, 1, 1 };
            cml_metal_launch_kernel(backend, kernel, grid, block,
                                    metal_buffers, buf_count);
        }

            if (out && out_buf && out_bytes > 0) {
            if (!out->data) {
                out->data = malloc(out_bytes);
                out->owns_data = true;
            }
            if (out->data) {
                cml_metal_download(backend, out->data, out_buf, out_bytes);
            }
            out->is_executed = true;
        }
        node->is_executed = true;

            for (int j = 0; j < buf_count; j++) {
            cml_metal_free(backend, metal_buffers[j]);
        }
        free(metal_buffers);
        cml_metal_kernel_free(kernel);

        node = node->next;
    }

    return 0;
}
#else /* !CML_HAS_METAL */
int cml_metal_execute_graph(CMLMetalBackend* backend, CMLGraph_t graph) {
    (void)backend;
    (void)graph;
    return -1;
}
#endif /* CML_HAS_METAL */
