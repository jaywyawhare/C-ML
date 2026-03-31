#include "ops/ir/gpu/metal_backend.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include "core/logging.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MSL_BUF_SIZE 4096

#define TILE_SIZE     16
#define OPT_TILE      64
#define OPT_REG       8
#define OPT_TSK       16
#define REDUCE_WG    256

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
        if (!tmp) { LOG_ERROR("metal_codegen: realloc failed"); return; }
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


char* cml_metal_generate_msl(struct IRNode* node) {
    if (!node) { LOG_ERROR("metal_codegen: NULL node"); return NULL; }

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

    if (type == UOP_RELU) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* a [[buffer(0)]],\n"
            "    device float* out     [[buffer(1)]],\n"
            "    constant uint& n      [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    float x = a[idx]; out[idx] = x > 0.0f ? x : 0.0f;\n"
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
            "    float scale = 1.0507f, alpha = 1.67326f;\n"
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
            "    device float* out      [[buffer(0)]],\n"
            "    constant float& val    [[buffer(1)]],\n"
            "    constant uint& n       [[buffer(2)]],\n"
            "    uint idx [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (idx >= n) return;\n"
            "    out[idx] = val;\n"
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
            "    device const float* in     [[buffer(0)]],\n"
            "    device float* out          [[buffer(1)]],\n"
            "    constant uint& n           [[buffer(2)]],\n"
            "    threadgroup float* shmem   [[threadgroup(0)]],\n"
            "    uint lid  [[thread_position_in_threadgroup]],\n"
            "    uint gid  [[threadgroup_position_in_grid]],\n"
            "    uint lsize [[threads_per_threadgroup]]) {\n"
            "    uint i = gid * lsize + lid;\n"
            "    shmem[lid] = (i < n) ? in[i] : 0.0f;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    for (uint s = lsize / 2; s > 0; s >>= 1) {\n"
            "        if (lid < s) shmem[lid] += shmem[lid + s];\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    }\n"
            "    if (lid == 0) out[gid] = shmem[0];\n"
            "}\n");
        return buf;
    }

    if (type == UOP_MAX_REDUCE) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* in     [[buffer(0)]],\n"
            "    device float* out          [[buffer(1)]],\n"
            "    constant uint& n           [[buffer(2)]],\n"
            "    threadgroup float* shmem   [[threadgroup(0)]],\n"
            "    uint lid  [[thread_position_in_threadgroup]],\n"
            "    uint gid  [[threadgroup_position_in_grid]],\n"
            "    uint lsize [[threads_per_threadgroup]]) {\n"
            "    uint i = gid * lsize + lid;\n"
            "    shmem[lid] = (i < n) ? in[i] : -INFINITY;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    for (uint s = lsize / 2; s > 0; s >>= 1) {\n"
            "        if (lid < s) shmem[lid] = max(shmem[lid], shmem[lid + s]);\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    }\n"
            "    if (lid == 0) out[gid] = shmem[0];\n"
            "}\n");
        return buf;
    }

    if (type == UOP_MEAN) {
        /* mean = sum then divide; generate same kernel as SUM, division done in executor */
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* in     [[buffer(0)]],\n"
            "    device float* out          [[buffer(1)]],\n"
            "    constant uint& n           [[buffer(2)]],\n"
            "    threadgroup float* shmem   [[threadgroup(0)]],\n"
            "    uint lid  [[thread_position_in_threadgroup]],\n"
            "    uint gid  [[threadgroup_position_in_grid]],\n"
            "    uint lsize [[threads_per_threadgroup]]) {\n"
            "    uint i = gid * lsize + lid;\n"
            "    shmem[lid] = (i < n) ? in[i] : 0.0f;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    for (uint s = lsize / 2; s > 0; s >>= 1) {\n"
            "        if (lid < s) shmem[lid] += shmem[lid + s];\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    }\n"
            "    if (lid == 0) out[gid] = shmem[0];\n"
            "}\n");
        return buf;
    }

    if (type == UOP_MIN_REDUCE) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* in     [[buffer(0)]],\n"
            "    device float* out          [[buffer(1)]],\n"
            "    constant uint& n           [[buffer(2)]],\n"
            "    threadgroup float* shmem   [[threadgroup(0)]],\n"
            "    uint lid  [[thread_position_in_threadgroup]],\n"
            "    uint gid  [[threadgroup_position_in_grid]],\n"
            "    uint lsize [[threads_per_threadgroup]]) {\n"
            "    uint i = gid * lsize + lid;\n"
            "    shmem[lid] = (i < n) ? in[i] : INFINITY;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    for (uint s = lsize / 2; s > 0; s >>= 1) {\n"
            "        if (lid < s) shmem[lid] = min(shmem[lid], shmem[lid + s]);\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    }\n"
            "    if (lid == 0) out[gid] = shmem[0];\n"
            "}\n");
        return buf;
    }

    if (type == UOP_PROD) {
        buf_appendf(&buf, &cap, &len,
            "kernel void cml_kernel(\n"
            "    device const float* in     [[buffer(0)]],\n"
            "    device float* out          [[buffer(1)]],\n"
            "    constant uint& n           [[buffer(2)]],\n"
            "    threadgroup float* shmem   [[threadgroup(0)]],\n"
            "    uint lid  [[thread_position_in_threadgroup]],\n"
            "    uint gid  [[threadgroup_position_in_grid]],\n"
            "    uint lsize [[threads_per_threadgroup]]) {\n"
            "    uint i = gid * lsize + lid;\n"
            "    shmem[lid] = (i < n) ? in[i] : 1.0f;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    for (uint s = lsize / 2; s > 0; s >>= 1) {\n"
            "        if (lid < s) shmem[lid] *= shmem[lid + s];\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    }\n"
            "    if (lid == 0) out[gid] = shmem[0];\n"
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
            "    uint row = gid.y * 16 + tid.y;\n"
            "    uint col = gid.x * 16 + tid.x;\n"
            "    float acc = 0.0f;\n"
            "    uint num_tiles = (K + 15) / 16;\n"
            "    for (uint t = 0; t < num_tiles; t++) {\n"
            "        uint a_col = t * 16 + tid.x;\n"
            "        uint b_row = t * 16 + tid.y;\n"
            "        As[tid.y][tid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;\n"
            "        Bs[tid.y][tid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        for (uint k = 0; k < 16; k++) acc += As[tid.y][k] * Bs[k][tid.x];\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    }\n"
            "    if (row < M && col < N) C[row * N + col] = acc;\n"
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
            "    uint ow = idx %% out_w; uint tmp = idx / out_w;\n"
            "    uint oh = tmp %% out_h; tmp = tmp / out_h;\n"
            "    uint oc = tmp %% out_c; uint b = tmp / out_c;\n"
            "    float acc = has_bias ? bias[oc] : 0.0f;\n"
            "    for (uint ic = 0; ic < in_c; ic++) {\n"
            "        for (uint fh = 0; fh < kH; fh++) {\n"
            "            for (uint fw = 0; fw < kW; fw++) {\n"
            "                int ih = (int)(oh * stride_h + fh) - (int)pad_h;\n"
            "                int iw = (int)(ow * stride_w + fw) - (int)pad_w;\n"
            "                if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {\n"
            "                    uint in_idx = b*(in_c*in_h*in_w)+ic*(in_h*in_w)+(uint)ih*in_w+(uint)iw;\n"
            "                    uint w_idx  = oc*(in_c*kH*kW)+ic*(kH*kW)+fh*kW+fw;\n"
            "                    acc += input[in_idx] * weight[w_idx];\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    out[idx] = acc;\n"
            "}\n");
        return buf;
    }

    LOG_WARNING("metal_codegen: Unsupported UOp type %d for MSL generation", (int)type);
    free(buf);
    return NULL;
}


static void* mtl_dyn_cache_lookup(CMLMetalBackend* b, int uop_type) {
    for (int i = 0; i < b->dyn_cache_count; i++) {
        if (b->dyn_cache[i].uop_type == uop_type)
            return b->dyn_cache[i].pso;
    }
    return NULL;
}

static void mtl_dyn_cache_insert(CMLMetalBackend* b, int uop_type, void* pso) {
    if (!pso || b->dyn_cache_count >= CML_MTL_DYN_CACHE_SIZE) return;
    CFRetain((CFTypeRef)pso);
    b->dyn_cache[b->dyn_cache_count].uop_type = uop_type;
    b->dyn_cache[b->dyn_cache_count].pso      = pso;
    b->dyn_cache_count++;
}


#ifdef CML_HAS_METAL


typedef struct {
    Tensor* tensor;
    void*   gpu_buf;  /* borrowed from backend buffer cache */
} NodeBufEntry;


static void* ensure_gpu(CMLMetalBackend* b,
                         Tensor* t,
                         NodeBufEntry* live, int* n_live) {
    if (!t || !t->data) return NULL;
    size_t bytes = t->numel * sizeof(float); /* only float32 supported on Metal */

    for (int i = 0; i < *n_live; i++) {
        if (live[i].tensor == t) return live[i].gpu_buf;
    }

    bool is_input = (t->ir_node == NULL);
    void* gpu_buf = cml_metal_get_or_upload_buffer(b, t->data, bytes, is_input);

    if (gpu_buf && *n_live < 512) {
        live[*n_live].tensor  = t;
        live[*n_live].gpu_buf = gpu_buf;
        (*n_live)++;
    }
    return gpu_buf;
}

static void* alloc_output(CMLMetalBackend* b,
                           Tensor* t,
                           NodeBufEntry* live, int* n_live) {
    if (!t) return NULL;
    size_t bytes = t->numel * sizeof(float);

    void* gpu_buf = cml_metal_alloc_output_buffer(b, (void*)t, bytes);
    if (gpu_buf && *n_live < 512) {
        live[*n_live].tensor  = t;
        live[*n_live].gpu_buf = gpu_buf;
        (*n_live)++;
    }
    return gpu_buf;
}

static void* find_gpu(NodeBufEntry* live, int n_live, Tensor* t) {
    for (int i = 0; i < n_live; i++) {
        if (live[i].tensor == t) return live[i].gpu_buf;
    }
    return NULL;
}

static int download_tensor(CMLMetalBackend* b,
                            Tensor* t,
                            NodeBufEntry* live, int n_live) {
    void* gpu_buf = find_gpu(live, n_live, t);
    if (!gpu_buf) return -1;
    size_t bytes = t->numel * sizeof(float);
    if (!t->data) {
        t->data = malloc(bytes);
        if (!t->data) return -1;
        t->owns_data = true;
    }
    return cml_metal_download_buffer(b, gpu_buf, t->data, bytes);
}

static int exec_reduce(CMLMetalBackend* b,
                        void* cmd_queue,   /* id<MTLCommandQueue> */
                        void* pso,         /* id<MTLComputePipelineState> */
                        void* buf_in,
                        void* buf_out,
                        size_t n,
                        bool is_mean) {
    if (!pso) return -1;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"

    @autoreleasepool {
        id<MTLCommandQueue>           queue    = (__bridge id<MTLCommandQueue>)cmd_queue;
        id<MTLComputePipelineState>   pipeline = (__bridge id<MTLComputePipelineState>)pso;
        id<MTLBuffer>                 mbuf_in  = (__bridge id<MTLBuffer>)buf_in;
        id<MTLBuffer>                 mbuf_out = (__bridge id<MTLBuffer>)buf_out;

        NSUInteger lsize = REDUCE_WG;
        while (lsize > n) lsize >>= 1;
        if (lsize < 1) lsize = 1;

        NSUInteger num_groups = (n + lsize - 1) / lsize;

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pipeline];

        if (num_groups == 1) {
            uint32_t un = (uint32_t)n;
            [enc setBuffer:mbuf_in  offset:0 atIndex:0];
            [enc setBuffer:mbuf_out offset:0 atIndex:1];
            [enc setBytes:&un length:sizeof(un) atIndex:2];
            [enc setThreadgroupMemoryLength:lsize * sizeof(float) atIndex:0];
            MTLSize gs = MTLSizeMake(num_groups, 1, 1);
            MTLSize ls = MTLSizeMake(lsize, 1, 1);
            [enc dispatchThreadgroups:gs threadsPerThreadgroup:ls];
        } else {
            id<MTLDevice> device = [queue device];
            id<MTLBuffer> tmp = [device newBufferWithLength:num_groups * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
            uint32_t un = (uint32_t)n;
            [enc setBuffer:mbuf_in offset:0 atIndex:0];
            [enc setBuffer:tmp     offset:0 atIndex:1];
            [enc setBytes:&un length:sizeof(un) atIndex:2];
            [enc setThreadgroupMemoryLength:lsize * sizeof(float) atIndex:0];
            MTLSize gs1 = MTLSizeMake(num_groups, 1, 1);
            MTLSize ls1 = MTLSizeMake(lsize, 1, 1);
            [enc dispatchThreadgroups:gs1 threadsPerThreadgroup:ls1];

            NSUInteger lsize2 = lsize;
            while (lsize2 > num_groups) lsize2 >>= 1;
            if (lsize2 < 1) lsize2 = 1;
            uint32_t n2 = (uint32_t)num_groups;
            [enc setBuffer:tmp      offset:0 atIndex:0];
            [enc setBuffer:mbuf_out offset:0 atIndex:1];
            [enc setBytes:&n2 length:sizeof(n2) atIndex:2];
            [enc setThreadgroupMemoryLength:lsize2 * sizeof(float) atIndex:0];
            MTLSize gs2 = MTLSizeMake(1, 1, 1);
            MTLSize ls2 = MTLSizeMake(lsize2, 1, 1);
            [enc dispatchThreadgroups:gs2 threadsPerThreadgroup:ls2];
        }

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if ([cmdBuf error]) return -1;

        if (is_mean && n > 0) {
            float result;
            memcpy(&result, [mbuf_out contents], sizeof(float));
            result /= (float)n;
            memcpy([mbuf_out contents], &result, sizeof(float));
        }
    }

#pragma clang diagnostic pop
    return 0;
}


static bool is_view_op(UOpType type) {
    return type == UOP_RESHAPE || type == UOP_EXPAND || type == UOP_PERMUTE ||
           type == UOP_STRIDE  || type == UOP_SLICE;
}

static bool is_gpu_supported(UOpType type) {
    switch (type) {
    case UOP_MATMUL:
    case UOP_ADD: case UOP_SUB: case UOP_MUL: case UOP_DIV:
    case UOP_NEG: case UOP_EXP: case UOP_LOG: case UOP_SQRT:
    case UOP_RELU: case UOP_SIGMOID: case UOP_TANH:
    case UOP_RELU6: case UOP_ELU: case UOP_SELU: case UOP_SILU:
    case UOP_MISH: case UOP_HARDSWISH:
    case UOP_SUM: case UOP_MAX_REDUCE: case UOP_MIN_REDUCE: case UOP_MEAN:
    case UOP_FILL: case UOP_MAX: case UOP_WHERE: case UOP_GATHER:
    case UOP_CONV2D: case UOP_PROD:
        return true;
    default:
        return false;
    }
}


int cml_metal_execute_graph(CMLMetalBackend* backend, CMLGraph_t graph) {
    if (!backend || !backend->initialized) {
        LOG_ERROR("metal_execute_graph: backend not initialized");
        return -1;
    }
    if (!graph) {
        LOG_ERROR("metal_execute_graph: NULL graph");
        return -1;
    }

    int    n_live = 0;
    int    live_cap = 512;
    NodeBufEntry* live = (NodeBufEntry*)calloc((size_t)live_cap, sizeof(NodeBufEntry));
    if (!live) return -1;

#define LIVE_GROW() \
    do { if (n_live >= live_cap) { \
        live_cap *= 2; \
        NodeBufEntry* tmp2 = (NodeBufEntry*)realloc(live, (size_t)live_cap * sizeof(NodeBufEntry)); \
        if (!tmp2) { free(live); return -1; } \
        live = tmp2; \
    } } while (0)

    struct IRNode* node = graph->head;

    while (node) {
        if (node->is_executed) {
            node = node->next;
            continue;
        }
        if (!node->output) {
            node = node->next;
            continue;
        }

        Tensor* out = node->output;

        if (is_view_op(node->type)) {
            for (int i = 0; i < node->num_inputs; i++) {
                Tensor* inp = node->inputs[i];
                if (inp && !inp->data) {
                    void* gbuf = find_gpu(live, n_live, inp);
                    if (gbuf) download_tensor(backend, inp, live, n_live);
                }
            }
            cpu_execute_node(node);
            node->is_executed = true;
            out->is_executed  = true;
            node = node->next;
            continue;
        }

        if (!is_gpu_supported(node->type)) {
            LOG_WARNING("Metal: unsupported op %d, falling back to CPU", (int)node->type);
            for (int i = 0; i < node->num_inputs; i++) {
                Tensor* inp = node->inputs[i];
                if (inp && !inp->data) {
                    void* gbuf = find_gpu(live, n_live, inp);
                    if (gbuf) download_tensor(backend, inp, live, n_live);
                }
            }
            cpu_execute_node(node);
            node->is_executed = true;
            out->is_executed  = true;
            if (out->data && out->numel > 0) {
                size_t bytes = out->numel * sizeof(float);
                void* gbuf = cml_metal_get_or_upload_buffer(backend, out->data, bytes, false);
                if (gbuf) {
                    LIVE_GROW();
                    live[n_live].tensor  = out;
                    live[n_live].gpu_buf = gbuf;
                    n_live++;
                }
            }
            node = node->next;
            continue;
        }

        if (node->type == UOP_MATMUL && backend->k_matmul_fused_bias_relu &&
            node->num_inputs >= 2 && node->inputs[0] && node->inputs[1]) {
            struct IRNode* add_n  = node->next;
            struct IRNode* relu_n = add_n ? add_n->next : NULL;

            if (add_n && relu_n &&
                add_n->type  == UOP_ADD  && !add_n->is_executed &&
                relu_n->type == UOP_RELU && !relu_n->is_executed &&
                add_n->num_inputs == 2 && relu_n->num_inputs == 1 &&
                relu_n->output) {

                Tensor* mm_out    = out;
                Tensor* bias_t    = (add_n->inputs[0] == mm_out) ? add_n->inputs[1]
                                                                  : add_n->inputs[0];
                Tensor* final_out = relu_n->output;
                Tensor* ta = node->inputs[0], *tb = node->inputs[1];

                if (mm_out && mm_out->ndim == 2 && bias_t && ta && ta->ndim >= 2) {
                    int fM = mm_out->shape[0], fN = mm_out->shape[1];
                    int fK = ta->shape[ta->ndim - 1];

                    if ((fM % OPT_TILE) == 0 && (fN % OPT_TILE) == 0 &&
                        (fK % OPT_TSK)  == 0 && (int)bias_t->numel == fN) {

                        void* ba    = ensure_gpu(backend, ta,     live, &n_live);
                        void* bb2   = ensure_gpu(backend, tb,     live, &n_live);
                        void* bbias = ensure_gpu(backend, bias_t, live, &n_live);
                        size_t final_bytes = final_out->numel * sizeof(float);
                        void* bfinal = cml_metal_alloc_output_buffer(backend,
                                                                      (void*)final_out,
                                                                      final_bytes);
                        if (ba && bb2 && bbias && bfinal) {
                            @autoreleasepool {
                                id<MTLCommandQueue> queue =
                                    (__bridge id<MTLCommandQueue>)backend->command_queue;
                                id<MTLComputePipelineState> pso =
                                    (__bridge id<MTLComputePipelineState>)backend->k_matmul_fused_bias_relu;

                                id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                                [enc setComputePipelineState:pso];
                                [enc setBuffer:(__bridge id<MTLBuffer>)ba    offset:0 atIndex:0];
                                [enc setBuffer:(__bridge id<MTLBuffer>)bb2   offset:0 atIndex:1];
                                [enc setBuffer:(__bridge id<MTLBuffer>)bbias offset:0 atIndex:2];
                                [enc setBuffer:(__bridge id<MTLBuffer>)bfinal offset:0 atIndex:3];
                                uint32_t uM = (uint32_t)fM, uN = (uint32_t)fN, uK = (uint32_t)fK;
                                [enc setBytes:&uM length:4 atIndex:4];
                                [enc setBytes:&uN length:4 atIndex:5];
                                [enc setBytes:&uK length:4 atIndex:6];
                                [enc setThreadgroupMemoryLength:16*65*sizeof(float) atIndex:0];
                                [enc setThreadgroupMemoryLength:16*65*sizeof(float) atIndex:1];
                                MTLSize gs = MTLSizeMake((NSUInteger)(fN/OPT_TILE),
                                                         (NSUInteger)(fM/OPT_TILE), 1);
                                MTLSize ls = MTLSizeMake(OPT_REG, OPT_REG, 1);
                                [enc dispatchThreadgroups:gs threadsPerThreadgroup:ls];
                                [enc endEncoding];
                                [cmdBuf commit];
                                [cmdBuf waitUntilCompleted];

                                if (![cmdBuf error]) {
                                    LIVE_GROW();
                                    live[n_live].tensor  = final_out;
                                    live[n_live].gpu_buf = bfinal;
                                    n_live++;

                                    node->is_executed = out->is_executed        = true;
                                    add_n->is_executed = true;
                                    if (add_n->output) add_n->output->is_executed = true;
                                    relu_n->is_executed = final_out->is_executed = true;
                                    node = relu_n->next;
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
        }

        void* bufs_in[8] = {0};
        bool input_ok = true;
        for (int i = 0; i < node->num_inputs && i < 8; i++) {
            Tensor* inp = node->inputs[i];
            if (!inp) continue;
            LIVE_GROW();
            bufs_in[i] = ensure_gpu(backend, inp, live, &n_live);
            if (!bufs_in[i]) { input_ok = false; break; }
        }
        if (!input_ok) goto cpu_fallback;

        {
            size_t out_bytes = out->numel * sizeof(float);
            void* gpu_out = cml_metal_alloc_output_buffer(backend, (void*)out, out_bytes);
            if (!gpu_out) goto cpu_fallback;
            LIVE_GROW();
            live[n_live].tensor  = out;
            live[n_live].gpu_buf = gpu_out;
            n_live++;

            int rc = 0;

            if (node->type == UOP_MATMUL) {
                Tensor* a_t  = node->inputs[0];
                Tensor* b_t  = node->inputs[1];
                int M = (a_t->ndim >= 2) ? a_t->shape[0] : 1;
                int K = (a_t->ndim >= 2) ? a_t->shape[a_t->ndim - 1] : (int)a_t->numel;
                int N = (b_t->ndim >= 2) ? b_t->shape[b_t->ndim - 1] : 1;

                if (backend->k_matmul_opt &&
                    (M % OPT_TILE) == 0 && (N % OPT_TILE) == 0 && (K % OPT_TSK) == 0) {
                    @autoreleasepool {
                        id<MTLCommandQueue> queue =
                            (__bridge id<MTLCommandQueue>)backend->command_queue;
                        id<MTLComputePipelineState> pso =
                            (__bridge id<MTLComputePipelineState>)backend->k_matmul_opt;

                        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                        [enc setComputePipelineState:pso];
                        [enc setBuffer:(__bridge id<MTLBuffer>)bufs_in[0] offset:0 atIndex:0];
                        [enc setBuffer:(__bridge id<MTLBuffer>)bufs_in[1] offset:0 atIndex:1];
                        [enc setBuffer:(__bridge id<MTLBuffer>)gpu_out     offset:0 atIndex:2];
                        uint32_t uM=(uint32_t)M, uN=(uint32_t)N, uK=(uint32_t)K;
                        [enc setBytes:&uM length:4 atIndex:3];
                        [enc setBytes:&uN length:4 atIndex:4];
                        [enc setBytes:&uK length:4 atIndex:5];
                        [enc setThreadgroupMemoryLength:OPT_TSK*65*sizeof(float) atIndex:0];
                        [enc setThreadgroupMemoryLength:OPT_TSK*65*sizeof(float) atIndex:1];
                        MTLSize gs = MTLSizeMake((NSUInteger)(N/OPT_TILE),
                                                 (NSUInteger)(M/OPT_TILE), 1);
                        MTLSize ls = MTLSizeMake(OPT_REG, OPT_REG, 1);
                        [enc dispatchThreadgroups:gs threadsPerThreadgroup:ls];
                        [enc endEncoding];
                        [cmdBuf commit];
                        [cmdBuf waitUntilCompleted];
                        rc = [cmdBuf error] ? -1 : 0;
                    }
                } else {
                    char* msl = cml_metal_generate_msl(node);
                    if (!msl) { rc = -1; goto check_rc; }
                    CMLMetalKernel* k = cml_metal_compile_msl(backend, msl, "cml_kernel");
                    free(msl);
                    if (!k) { rc = -1; goto check_rc; }

                    @autoreleasepool {
                        id<MTLCommandQueue> queue =
                            (__bridge id<MTLCommandQueue>)backend->command_queue;
                        id<MTLComputePipelineState> pso =
                            (__bridge id<MTLComputePipelineState>)k->pipeline;

                        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                        [enc setComputePipelineState:pso];
                        [enc setBuffer:(__bridge id<MTLBuffer>)bufs_in[0] offset:0 atIndex:0];
                        [enc setBuffer:(__bridge id<MTLBuffer>)bufs_in[1] offset:0 atIndex:1];
                        [enc setBuffer:(__bridge id<MTLBuffer>)gpu_out     offset:0 atIndex:2];
                        uint32_t uM=(uint32_t)M, uN=(uint32_t)N, uK=(uint32_t)K;
                        [enc setBytes:&uM length:4 atIndex:3];
                        [enc setBytes:&uN length:4 atIndex:4];
                        [enc setBytes:&uK length:4 atIndex:5];
                        size_t ng_x = ((size_t)N + 15) / 16;
                        size_t ng_y = ((size_t)M + 15) / 16;
                        MTLSize gs = MTLSizeMake(ng_x, ng_y, 1);
                        MTLSize ls = MTLSizeMake(16, 16, 1);
                        [enc dispatchThreadgroups:gs threadsPerThreadgroup:ls];
                        [enc endEncoding];
                        [cmdBuf commit];
                        [cmdBuf waitUntilCompleted];
                        rc = [cmdBuf error] ? -1 : 0;
                    }
                    cml_metal_kernel_free(k);
                }
            } else if (node->type == UOP_SUM || node->type == UOP_MEAN) {
                void* pso = (node->type == UOP_SUM) ? backend->k_sum_reduce
                                                     : backend->k_sum_reduce;
                if (!pso) { rc = -1; goto check_rc; }
                int n_elem = (node->num_inputs >= 1 && node->inputs[0])
                             ? (int)node->inputs[0]->numel : 0;
                rc = exec_reduce(backend, backend->command_queue,
                                 pso, bufs_in[0], gpu_out,
                                 (size_t)n_elem, (node->type == UOP_MEAN));
            } else if (node->type == UOP_MAX_REDUCE) {
                if (!backend->k_max_reduce) { rc = -1; goto check_rc; }
                int n_elem = (node->num_inputs >= 1 && node->inputs[0])
                             ? (int)node->inputs[0]->numel : 0;
                rc = exec_reduce(backend, backend->command_queue,
                                 backend->k_max_reduce, bufs_in[0], gpu_out,
                                 (size_t)n_elem, false);
            } else {
                void* static_pso = NULL;
                switch (node->type) {
                case UOP_FILL:    static_pso = backend->k_fill;    break;
                case UOP_RELU:    static_pso = backend->k_relu;    break;
                case UOP_SIGMOID: static_pso = backend->k_sigmoid; break;
                case UOP_TANH:    static_pso = backend->k_tanh_k;  break;
                default: break;
                }

                if (static_pso) {
                    @autoreleasepool {
                        id<MTLCommandQueue> queue =
                            (__bridge id<MTLCommandQueue>)backend->command_queue;
                        id<MTLComputePipelineState> pso =
                            (__bridge id<MTLComputePipelineState>)static_pso;

                        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                        [enc setComputePipelineState:pso];

                        NSUInteger aidx = 0;

                        if (node->type == UOP_FILL) {
                            [enc setBuffer:(__bridge id<MTLBuffer>)gpu_out offset:0 atIndex:0];
                            float val = node->params ? *(float*)node->params : 0.0f;
                            uint32_t un = (uint32_t)out->numel;
                            [enc setBytes:&val length:sizeof(float) atIndex:1];
                            [enc setBytes:&un  length:sizeof(uint32_t) atIndex:2];
                            NSUInteger gsize = ((NSUInteger)out->numel + 255) / 256;
                            MTLSize gs = MTLSizeMake(gsize, 1, 1);
                            MTLSize ls = MTLSizeMake(256, 1, 1);
                            [enc dispatchThreadgroups:gs threadsPerThreadgroup:ls];
                        } else {
                            [enc setBuffer:(__bridge id<MTLBuffer>)bufs_in[0] offset:0 atIndex:0];
                            [enc setBuffer:(__bridge id<MTLBuffer>)gpu_out     offset:0 atIndex:1];
                            uint32_t un = (uint32_t)out->numel;
                            [enc setBytes:&un length:sizeof(uint32_t) atIndex:2];
                            NSUInteger gsize = ((NSUInteger)out->numel + 255) / 256;
                            MTLSize gs = MTLSizeMake(gsize, 1, 1);
                            MTLSize ls = MTLSizeMake(256, 1, 1);
                            [enc dispatchThreadgroups:gs threadsPerThreadgroup:ls];
                        }
                        (void)aidx;

                        [enc endEncoding];
                        [cmdBuf commit];
                        [cmdBuf waitUntilCompleted];
                        rc = [cmdBuf error] ? -1 : 0;
                    }
                } else {
                    void* cached_dpso = mtl_dyn_cache_lookup(backend, (int)node->type);

                    if (!cached_dpso) {
                        char* msl = cml_metal_generate_msl(node);
                        if (!msl) { rc = -1; goto check_rc; }
                        CMLMetalKernel* k = cml_metal_compile_msl(backend, msl, "cml_kernel");
                        free(msl);
                        if (!k) { rc = -1; goto check_rc; }
                        mtl_dyn_cache_insert(backend, (int)node->type, k->pipeline);
                        cml_metal_kernel_free(k);
                        cached_dpso = mtl_dyn_cache_lookup(backend, (int)node->type);
                        if (!cached_dpso) { rc = -1; goto check_rc; }
                    }

                    @autoreleasepool {
                        id<MTLCommandQueue> queue =
                            (__bridge id<MTLCommandQueue>)backend->command_queue;
                        id<MTLComputePipelineState> pso =
                            (__bridge id<MTLComputePipelineState>)cached_dpso;

                        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                        [enc setComputePipelineState:pso];

                        int bidx = 0;
                        for (int i = 0; i < node->num_inputs && i < 8; i++) {
                            if (!node->inputs[i]) continue;
                            [enc setBuffer:(__bridge id<MTLBuffer>)bufs_in[i]
                                   offset:0 atIndex:(NSUInteger)bidx++];
                        }
                        [enc setBuffer:(__bridge id<MTLBuffer>)gpu_out
                               offset:0 atIndex:(NSUInteger)bidx++];

                        uint32_t un = (uint32_t)out->numel;
                        [enc setBytes:&un length:sizeof(uint32_t) atIndex:(NSUInteger)bidx++];

                        bool is_reduce = (node->type == UOP_MIN_REDUCE ||
                                          node->type == UOP_PROD);
                        if (is_reduce)
                            [enc setThreadgroupMemoryLength:REDUCE_WG*sizeof(float) atIndex:0];

                        if (is_reduce) {
                            size_t n_elem = node->num_inputs >= 1 && node->inputs[0]
                                            ? node->inputs[0]->numel : out->numel;
                            NSUInteger lsize = REDUCE_WG;
                            while (lsize > n_elem) lsize >>= 1;
                            if (lsize < 1) lsize = 1;
                            NSUInteger ng = (n_elem + lsize - 1) / lsize;
                            [enc dispatchThreadgroups:MTLSizeMake(ng, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(lsize, 1, 1)];
                        } else {
                            NSUInteger ng = ((NSUInteger)out->numel + 255) / 256;
                            [enc dispatchThreadgroups:MTLSizeMake(ng, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        }

                        [enc endEncoding];
                        [cmdBuf commit];
                        [cmdBuf waitUntilCompleted];
                        rc = [cmdBuf error] ? -1 : 0;
                    }
                }
            }

        check_rc:
            if (rc != 0) goto cpu_fallback;

            node->is_executed = true;
            out->is_executed  = true;
            node = node->next;
            continue;
        }

    cpu_fallback:
        for (int i = 0; i < node->num_inputs; i++) {
            Tensor* inp = node->inputs[i];
            if (inp && !inp->data) {
                void* gbuf = find_gpu(live, n_live, inp);
                if (gbuf) download_tensor(backend, inp, live, n_live);
            }
        }
        cpu_execute_node(node);
        node->is_executed = true;
        out->is_executed  = true;
        if (out->data && out->numel > 0) {
            size_t bytes = out->numel * sizeof(float);
            void* gbuf = cml_metal_get_or_upload_buffer(backend, out->data, bytes, false);
            if (gbuf) {
                LIVE_GROW();
                live[n_live].tensor  = out;
                live[n_live].gpu_buf = gbuf;
                n_live++;
            }
        }
        node = node->next;
        continue;
    }

    node = graph->head;
    while (node) {
        if (node->output && node->is_executed) {
            Tensor* t = node->output;
            void* gbuf = find_gpu(live, n_live, t);
            if (gbuf) {
                bool consumed = false;
                struct IRNode* later = node->next;
                while (later) {
                    if (later->is_executed) {
                        for (int i = 0; i < later->num_inputs; i++) {
                            if (later->inputs[i] == t) { consumed = true; break; }
                        }
                    }
                    if (consumed) break;
                    later = later->next;
                }
                if (!consumed) {
                    size_t bytes = t->numel * sizeof(float);
                    if (!t->data) {
                        t->data = malloc(bytes);
                        if (t->data) t->owns_data = true;
                    }
                    if (t->data) {
                        cml_metal_download_buffer(backend, gbuf, t->data, bytes);
                    }
                }
            }
        }
        node = node->next;
    }

    cml_metal_release_intermediate_buffers(backend);

    graph->is_executed = true;
    free(live);

#undef LIVE_GROW
    return 0;
}

#else /* !CML_HAS_METAL */
int cml_metal_execute_graph(CMLMetalBackend* backend, CMLGraph_t graph) {
    (void)backend; (void)graph; return -1;
}
#endif /* CML_HAS_METAL */
