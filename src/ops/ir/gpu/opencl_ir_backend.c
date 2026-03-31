/*
 * OpenCL IR backend — executes CML IR graphs on GPU via OpenCL.
 * Uses static kernel strings compiled at init time.
 * Keeps intermediate buffers on GPU, only does D2H for final outputs.
 */

#include "ops/ir/gpu/opencl_ir_backend.h"
#include "ops/ir/internal.h"
#include "ops/ir/ir.h"
#include "ops/uops.h"
#include "tensor/tensor.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef CML_HAS_OPENCL


static const char* g_ocl_kernel_src =
"#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n"

"__attribute__((intel_reqd_sub_group_size(8)))\n"
"__kernel void matmul(__global const float* restrict A,\n"
"                        __global const float* restrict B,\n"
"                        __global float* restrict C, int M, int N, int K) {\n"
"    int tidC = get_local_id(0);\n"
"    int tidR = get_local_id(1);\n"
"    int gidR = get_group_id(1) * 128;\n"
"    int gidC = get_group_id(0) * 128;\n"
"    __local float As_T[16][129];\n"
"    __local float Bs[16][129];\n"
"    float acc[8][8];\n"
"    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;\n"
"    int tid = tidR * 16 + tidC;\n"
"    for (int t = 0; t < K / 16; t++) {\n"
"        int tK = t * 16;\n"
"        for (int l = 0; l < 2; l++) {\n"
"            int idx = tid + l * 256;\n"
"            int lr  = idx >> 2;\n"
"            int lc4 = (idx & 3) * 4;\n"
"            __global const float4* a4 = (__global const float4*)(A + (gidR + lr) * K + tK + lc4);\n"
"            float4 v = *a4;\n"
"            As_T[lc4+0][lr] = v.x; As_T[lc4+1][lr] = v.y;\n"
"            As_T[lc4+2][lr] = v.z; As_T[lc4+3][lr] = v.w;\n"
"        }\n"
"        for (int l = 0; l < 2; l++) {\n"
"            int idx = tid + l * 256;\n"
"            int lr  = idx >> 5;\n"
"            int lc4 = (idx & 31) * 4;\n"
"            __global const float4* b4 = (__global const float4*)(B + (tK + lr) * N + gidC + lc4);\n"
"            float4 v = *b4;\n"
"            Bs[lr][lc4+0] = v.x; Bs[lr][lc4+1] = v.y;\n"
"            Bs[lr][lc4+2] = v.z; Bs[lr][lc4+3] = v.w;\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        #pragma unroll\n"
"        for (int k = 0; k < 16; k++) {\n"
"            float a0=As_T[k][tidR*8],a1=As_T[k][tidR*8+1],a2=As_T[k][tidR*8+2],a3=As_T[k][tidR*8+3];\n"
"            float a4=As_T[k][tidR*8+4],a5=As_T[k][tidR*8+5],a6=As_T[k][tidR*8+6],a7=As_T[k][tidR*8+7];\n"
"            float b0=Bs[k][tidC*8], b1=Bs[k][tidC*8+1], b2=Bs[k][tidC*8+2], b3=Bs[k][tidC*8+3];\n"
"            float b4=Bs[k][tidC*8+4], b5=Bs[k][tidC*8+5], b6=Bs[k][tidC*8+6], b7=Bs[k][tidC*8+7];\n"
"            acc[0][0]=mad(a0,b0,acc[0][0]); acc[0][1]=mad(a0,b1,acc[0][1]); acc[0][2]=mad(a0,b2,acc[0][2]); acc[0][3]=mad(a0,b3,acc[0][3]);\n"
"            acc[0][4]=mad(a0,b4,acc[0][4]); acc[0][5]=mad(a0,b5,acc[0][5]); acc[0][6]=mad(a0,b6,acc[0][6]); acc[0][7]=mad(a0,b7,acc[0][7]);\n"
"            acc[1][0]=mad(a1,b0,acc[1][0]); acc[1][1]=mad(a1,b1,acc[1][1]); acc[1][2]=mad(a1,b2,acc[1][2]); acc[1][3]=mad(a1,b3,acc[1][3]);\n"
"            acc[1][4]=mad(a1,b4,acc[1][4]); acc[1][5]=mad(a1,b5,acc[1][5]); acc[1][6]=mad(a1,b6,acc[1][6]); acc[1][7]=mad(a1,b7,acc[1][7]);\n"
"            acc[2][0]=mad(a2,b0,acc[2][0]); acc[2][1]=mad(a2,b1,acc[2][1]); acc[2][2]=mad(a2,b2,acc[2][2]); acc[2][3]=mad(a2,b3,acc[2][3]);\n"
"            acc[2][4]=mad(a2,b4,acc[2][4]); acc[2][5]=mad(a2,b5,acc[2][5]); acc[2][6]=mad(a2,b6,acc[2][6]); acc[2][7]=mad(a2,b7,acc[2][7]);\n"
"            acc[3][0]=mad(a3,b0,acc[3][0]); acc[3][1]=mad(a3,b1,acc[3][1]); acc[3][2]=mad(a3,b2,acc[3][2]); acc[3][3]=mad(a3,b3,acc[3][3]);\n"
"            acc[3][4]=mad(a3,b4,acc[3][4]); acc[3][5]=mad(a3,b5,acc[3][5]); acc[3][6]=mad(a3,b6,acc[3][6]); acc[3][7]=mad(a3,b7,acc[3][7]);\n"
"            acc[4][0]=mad(a4,b0,acc[4][0]); acc[4][1]=mad(a4,b1,acc[4][1]); acc[4][2]=mad(a4,b2,acc[4][2]); acc[4][3]=mad(a4,b3,acc[4][3]);\n"
"            acc[4][4]=mad(a4,b4,acc[4][4]); acc[4][5]=mad(a4,b5,acc[4][5]); acc[4][6]=mad(a4,b6,acc[4][6]); acc[4][7]=mad(a4,b7,acc[4][7]);\n"
"            acc[5][0]=mad(a5,b0,acc[5][0]); acc[5][1]=mad(a5,b1,acc[5][1]); acc[5][2]=mad(a5,b2,acc[5][2]); acc[5][3]=mad(a5,b3,acc[5][3]);\n"
"            acc[5][4]=mad(a5,b4,acc[5][4]); acc[5][5]=mad(a5,b5,acc[5][5]); acc[5][6]=mad(a5,b6,acc[5][6]); acc[5][7]=mad(a5,b7,acc[5][7]);\n"
"            acc[6][0]=mad(a6,b0,acc[6][0]); acc[6][1]=mad(a6,b1,acc[6][1]); acc[6][2]=mad(a6,b2,acc[6][2]); acc[6][3]=mad(a6,b3,acc[6][3]);\n"
"            acc[6][4]=mad(a6,b4,acc[6][4]); acc[6][5]=mad(a6,b5,acc[6][5]); acc[6][6]=mad(a6,b6,acc[6][6]); acc[6][7]=mad(a6,b7,acc[6][7]);\n"
"            acc[7][0]=mad(a7,b0,acc[7][0]); acc[7][1]=mad(a7,b1,acc[7][1]); acc[7][2]=mad(a7,b2,acc[7][2]); acc[7][3]=mad(a7,b3,acc[7][3]);\n"
"            acc[7][4]=mad(a7,b4,acc[7][4]); acc[7][5]=mad(a7,b5,acc[7][5]); acc[7][6]=mad(a7,b6,acc[7][6]); acc[7][7]=mad(a7,b7,acc[7][7]);\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    for (int i = 0; i < 8; i++)\n"
"        for (int j = 0; j < 8; j++)\n"
"            C[(gidR+tidR*8+i)*N+gidC+tidC*8+j] = acc[i][j];\n"
"}\n"

"__attribute__((intel_reqd_sub_group_size(8)))\n"
"__kernel void matmul_fused_bias_relu(__global const float* restrict A,\n"
"                                     __global const float* restrict B,\n"
"                                     __global const float* restrict bias,\n"
"                                     __global float* restrict C,\n"
"                                     int M, int N, int K) {\n"
"    int tidC = get_local_id(0);\n"
"    int tidR = get_local_id(1);\n"
"    int gidR = get_group_id(1) * 128;\n"
"    int gidC = get_group_id(0) * 128;\n"
"    __local float As_T[16][129];\n"
"    __local float Bs[16][129];\n"
"    float acc[8][8];\n"
"    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;\n"
"    int tid = tidR * 16 + tidC;\n"
"    for (int t = 0; t < K / 16; t++) {\n"
"        int tK = t * 16;\n"
"        for (int l = 0; l < 2; l++) {\n"
"            int idx = tid + l * 256;\n"
"            int lr  = idx >> 2;\n"
"            int lc4 = (idx & 3) * 4;\n"
"            __global const float4* a4 = (__global const float4*)(A + (gidR + lr) * K + tK + lc4);\n"
"            float4 v = *a4;\n"
"            As_T[lc4+0][lr] = v.x; As_T[lc4+1][lr] = v.y;\n"
"            As_T[lc4+2][lr] = v.z; As_T[lc4+3][lr] = v.w;\n"
"        }\n"
"        for (int l = 0; l < 2; l++) {\n"
"            int idx = tid + l * 256;\n"
"            int lr  = idx >> 5;\n"
"            int lc4 = (idx & 31) * 4;\n"
"            __global const float4* b4 = (__global const float4*)(B + (tK + lr) * N + gidC + lc4);\n"
"            float4 v = *b4;\n"
"            Bs[lr][lc4+0] = v.x; Bs[lr][lc4+1] = v.y;\n"
"            Bs[lr][lc4+2] = v.z; Bs[lr][lc4+3] = v.w;\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        #pragma unroll\n"
"        for (int k = 0; k < 16; k++) {\n"
"            float a0=As_T[k][tidR*8],a1=As_T[k][tidR*8+1],a2=As_T[k][tidR*8+2],a3=As_T[k][tidR*8+3];\n"
"            float a4=As_T[k][tidR*8+4],a5=As_T[k][tidR*8+5],a6=As_T[k][tidR*8+6],a7=As_T[k][tidR*8+7];\n"
"            float b0=Bs[k][tidC*8], b1=Bs[k][tidC*8+1], b2=Bs[k][tidC*8+2], b3=Bs[k][tidC*8+3];\n"
"            float b4=Bs[k][tidC*8+4], b5=Bs[k][tidC*8+5], b6=Bs[k][tidC*8+6], b7=Bs[k][tidC*8+7];\n"
"            acc[0][0]=mad(a0,b0,acc[0][0]); acc[0][1]=mad(a0,b1,acc[0][1]); acc[0][2]=mad(a0,b2,acc[0][2]); acc[0][3]=mad(a0,b3,acc[0][3]);\n"
"            acc[0][4]=mad(a0,b4,acc[0][4]); acc[0][5]=mad(a0,b5,acc[0][5]); acc[0][6]=mad(a0,b6,acc[0][6]); acc[0][7]=mad(a0,b7,acc[0][7]);\n"
"            acc[1][0]=mad(a1,b0,acc[1][0]); acc[1][1]=mad(a1,b1,acc[1][1]); acc[1][2]=mad(a1,b2,acc[1][2]); acc[1][3]=mad(a1,b3,acc[1][3]);\n"
"            acc[1][4]=mad(a1,b4,acc[1][4]); acc[1][5]=mad(a1,b5,acc[1][5]); acc[1][6]=mad(a1,b6,acc[1][6]); acc[1][7]=mad(a1,b7,acc[1][7]);\n"
"            acc[2][0]=mad(a2,b0,acc[2][0]); acc[2][1]=mad(a2,b1,acc[2][1]); acc[2][2]=mad(a2,b2,acc[2][2]); acc[2][3]=mad(a2,b3,acc[2][3]);\n"
"            acc[2][4]=mad(a2,b4,acc[2][4]); acc[2][5]=mad(a2,b5,acc[2][5]); acc[2][6]=mad(a2,b6,acc[2][6]); acc[2][7]=mad(a2,b7,acc[2][7]);\n"
"            acc[3][0]=mad(a3,b0,acc[3][0]); acc[3][1]=mad(a3,b1,acc[3][1]); acc[3][2]=mad(a3,b2,acc[3][2]); acc[3][3]=mad(a3,b3,acc[3][3]);\n"
"            acc[3][4]=mad(a3,b4,acc[3][4]); acc[3][5]=mad(a3,b5,acc[3][5]); acc[3][6]=mad(a3,b6,acc[3][6]); acc[3][7]=mad(a3,b7,acc[3][7]);\n"
"            acc[4][0]=mad(a4,b0,acc[4][0]); acc[4][1]=mad(a4,b1,acc[4][1]); acc[4][2]=mad(a4,b2,acc[4][2]); acc[4][3]=mad(a4,b3,acc[4][3]);\n"
"            acc[4][4]=mad(a4,b4,acc[4][4]); acc[4][5]=mad(a4,b5,acc[4][5]); acc[4][6]=mad(a4,b6,acc[4][6]); acc[4][7]=mad(a4,b7,acc[4][7]);\n"
"            acc[5][0]=mad(a5,b0,acc[5][0]); acc[5][1]=mad(a5,b1,acc[5][1]); acc[5][2]=mad(a5,b2,acc[5][2]); acc[5][3]=mad(a5,b3,acc[5][3]);\n"
"            acc[5][4]=mad(a5,b4,acc[5][4]); acc[5][5]=mad(a5,b5,acc[5][5]); acc[5][6]=mad(a5,b6,acc[5][6]); acc[5][7]=mad(a5,b7,acc[5][7]);\n"
"            acc[6][0]=mad(a6,b0,acc[6][0]); acc[6][1]=mad(a6,b1,acc[6][1]); acc[6][2]=mad(a6,b2,acc[6][2]); acc[6][3]=mad(a6,b3,acc[6][3]);\n"
"            acc[6][4]=mad(a6,b4,acc[6][4]); acc[6][5]=mad(a6,b5,acc[6][5]); acc[6][6]=mad(a6,b6,acc[6][6]); acc[6][7]=mad(a6,b7,acc[6][7]);\n"
"            acc[7][0]=mad(a7,b0,acc[7][0]); acc[7][1]=mad(a7,b1,acc[7][1]); acc[7][2]=mad(a7,b2,acc[7][2]); acc[7][3]=mad(a7,b3,acc[7][3]);\n"
"            acc[7][4]=mad(a7,b4,acc[7][4]); acc[7][5]=mad(a7,b5,acc[7][5]); acc[7][6]=mad(a7,b6,acc[7][6]); acc[7][7]=mad(a7,b7,acc[7][7]);\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    for (int i = 0; i < 8; i++) {\n"
"        int row = gidR + tidR * 8 + i;\n"
"        for (int j = 0; j < 8; j++) {\n"
"            int col = gidC + tidC * 8 + j;\n"
"            float v = acc[i][j] + bias[col];\n"
"            C[row * N + col] = v > 0.0f ? v : 0.0f;\n"
"        }\n"
"    }\n"
"}\n"

"__kernel void matmul_naive(__global const float* A, __global const float* B,\n"
"                           __global float* C, int M, int N, int K) {\n"
"    int row = get_global_id(0);\n"
"    int col = get_global_id(1);\n"
"    if (row < M && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int i = 0; i < K; i++)\n"
"            sum += A[row * K + i] * B[i * N + col];\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n"

"__kernel void ew_add(__global const float* a, int sa,\n"
"                     __global const float* b, int sb,\n"
"                     __global float* out, int n) {\n"
"    int i = get_global_id(0);\n"
"    if (i < n) out[i] = a[sa ? (i % sa) : i] + b[sb ? (i % sb) : i];\n"
"}\n"
"__kernel void ew_sub(__global const float* a, int sa,\n"
"                     __global const float* b, int sb,\n"
"                     __global float* out, int n) {\n"
"    int i = get_global_id(0);\n"
"    if (i < n) out[i] = a[sa ? (i % sa) : i] - b[sb ? (i % sb) : i];\n"
"}\n"
"__kernel void ew_mul(__global const float* a, int sa,\n"
"                     __global const float* b, int sb,\n"
"                     __global float* out, int n) {\n"
"    int i = get_global_id(0);\n"
"    if (i < n) out[i] = a[sa ? (i % sa) : i] * b[sb ? (i % sb) : i];\n"
"}\n"
"__kernel void ew_div(__global const float* a, int sa,\n"
"                     __global const float* b, int sb,\n"
"                     __global float* out, int n) {\n"
"    int i = get_global_id(0);\n"
"    if (i < n) out[i] = a[sa ? (i % sa) : i] / b[sb ? (i % sb) : i];\n"
"}\n"

"__kernel void ew_neg(__global const float* x, __global float* out, int n) {\n"
"    int i = get_global_id(0); if (i < n) out[i] = -x[i];\n"
"}\n"
"__kernel void ew_relu(__global const float* x, __global float* out, int n) {\n"
"    int i = get_global_id(0); if (i < n) out[i] = x[i] > 0.0f ? x[i] : 0.0f;\n"
"}\n"
"__kernel void ew_sigmoid(__global const float* x, __global float* out, int n) {\n"
"    int i = get_global_id(0); if (i < n) out[i] = 1.0f / (1.0f + exp(-x[i]));\n"
"}\n"
"__kernel void ew_tanh_k(__global const float* x, __global float* out, int n) {\n"
"    int i = get_global_id(0); if (i < n) out[i] = tanh(x[i]);\n"
"}\n"
"__kernel void ew_exp(__global const float* x, __global float* out, int n) {\n"
"    int i = get_global_id(0); if (i < n) out[i] = exp(x[i]);\n"
"}\n"
"__kernel void ew_log(__global const float* x, __global float* out, int n) {\n"
"    int i = get_global_id(0); if (i < n) out[i] = log(x[i]);\n"
"}\n"
"__kernel void ew_sqrt_k(__global const float* x, __global float* out, int n) {\n"
"    int i = get_global_id(0); if (i < n) out[i] = sqrt(x[i]);\n"
"}\n"

"__kernel void ew_fill(__global float* out, float val, int n) {\n"
"    int i = get_global_id(0); if (i < n) out[i] = val;\n"
"}\n"

"__kernel void reduce_sum(__global const float* x, __global float* out,\n"
"                         __local float* scratch, int n) {\n"
"    int lid = get_local_id(0);\n"
"    int gid = get_global_id(0);\n"
"    scratch[lid] = (gid < n) ? x[gid] : 0.0f;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {\n"
"        if (lid < s) scratch[lid] += scratch[lid + s];\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (lid == 0) out[get_group_id(0)] = scratch[0];\n"
"}\n"
"__kernel void reduce_max(__global const float* x, __global float* out,\n"
"                         __local float* scratch, int n) {\n"
"    int lid = get_local_id(0);\n"
"    int gid = get_global_id(0);\n"
"    scratch[lid] = (gid < n) ? x[gid] : -INFINITY;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {\n"
"        if (lid < s && scratch[lid + s] > scratch[lid])\n"
"            scratch[lid] = scratch[lid + s];\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (lid == 0) out[get_group_id(0)] = scratch[0];\n"
"}\n"
;

/* ─── Buffer tracker helpers ───────────────────────────────────────────── */

static CMLOCLBufferEntry* ocl_find_buffer(CMLOpenCLIRBackend* b, Tensor* t) {
    for (int i = 0; i < b->buffer_count; i++) {
        if (b->buffers[i].tensor == t)
            return &b->buffers[i];
    }
    return NULL;
}

/* Find cached input buffer by CPU data pointer (survives tensor recreation) */
static CMLOCLBufferEntry* ocl_find_cached_input(CMLOpenCLIRBackend* b, void* data, size_t size) {
    for (int i = 0; i < b->buffer_count; i++) {
        if (b->buffers[i].is_input && b->buffers[i].data_ptr == data &&
            b->buffers[i].size == size && b->buffers[i].valid)
            return &b->buffers[i];
    }
    return NULL;
}

static cl_mem ocl_ensure_gpu(CMLOpenCLIRBackend* b, Tensor* t) {
    if (!t || !t->data) return NULL;

    size_t bytes = t->numel * cml_dtype_size(t->dtype);
    if (bytes == 0) return NULL;

    CMLOCLBufferEntry* e = ocl_find_buffer(b, t);
    if (e && e->valid)
        return e->gpu_buf;

    CMLOCLBufferEntry* cached = ocl_find_cached_input(b, t->data, bytes);
    if (cached) {
        cached->tensor = t;
        return cached->gpu_buf;
    }

    cl_int err;
    cl_mem buf;

    if (e) {
        if (e->size == bytes) {
            err = clEnqueueWriteBuffer(b->queue, e->gpu_buf, CL_FALSE, 0, bytes, t->data,
                                       0, NULL, NULL);
            if (err == CL_SUCCESS) { e->valid = true; e->data_ptr = t->data; return e->gpu_buf; }
        }
        clReleaseMemObject(e->gpu_buf);
        buf = clCreateBuffer(b->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                             bytes, t->data, &err);
        if (err != CL_SUCCESS) return NULL;
        e->gpu_buf = buf;
        e->size = bytes;
        e->data_ptr = t->data;
        e->valid = true;
        return buf;
    }

    if (b->buffer_count >= CML_OCL_MAX_TRACKED_BUFFERS) {
        for (int i = 0; i < b->buffer_count; i++) {
            if (b->buffers[i].is_input) {
                clReleaseMemObject(b->buffers[i].gpu_buf);
                for (int j = i; j < b->buffer_count - 1; j++)
                    b->buffers[j] = b->buffers[j + 1];
                b->buffer_count--;
                break;
            }
        }
        if (b->buffer_count >= CML_OCL_MAX_TRACKED_BUFFERS) {
            LOG_ERROR("OpenCL buffer tracker full");
            return NULL;
        }
    }
    buf = clCreateBuffer(b->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                         bytes, t->data, &err);
    if (err != CL_SUCCESS) return NULL;

    e = &b->buffers[b->buffer_count++];
    e->tensor   = t;
    e->data_ptr = t->data;
    e->gpu_buf  = buf;
    e->size     = bytes;
    e->valid    = true;
    e->is_input = (t->ir_node == NULL);  /* only cache leaf inputs across executions */
    return buf;
}

static cl_mem ocl_alloc_output(CMLOpenCLIRBackend* b, Tensor* t) {
    size_t bytes = t->numel * cml_dtype_size(t->dtype);
    if (bytes == 0) return NULL;

    CMLOCLBufferEntry* e = ocl_find_buffer(b, t);
    if (e && e->size >= bytes) {
        e->valid = true;
        return e->gpu_buf;
    }

    for (int i = 0; i < b->buffer_count; i++) {
        CMLOCLBufferEntry* p = &b->buffers[i];
        if (!p->is_input && !p->valid && p->gpu_buf && p->size == bytes) {
            p->tensor = t;
            p->valid = true;
            return p->gpu_buf;
        }
    }

    if (b->buffer_count >= CML_OCL_MAX_TRACKED_BUFFERS) {
        LOG_ERROR("OpenCL buffer tracker full");
        return NULL;
    }

    cl_int err;
    cl_mem buf = clCreateBuffer(b->context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    if (err != CL_SUCCESS) return NULL;

    if (e) {
        clReleaseMemObject(e->gpu_buf);
        e->gpu_buf = buf;
        e->size    = bytes;
        e->valid   = true;
    } else {
        e = &b->buffers[b->buffer_count++];
        e->tensor  = t;
        e->gpu_buf = buf;
        e->size    = bytes;
        e->valid   = true;
    }
    return buf;
}

static int ocl_download(CMLOpenCLIRBackend* b, Tensor* t) {
    CMLOCLBufferEntry* e = ocl_find_buffer(b, t);
    if (!e || !e->valid) return -1;

    size_t bytes = t->numel * cml_dtype_size(t->dtype);

    if (!t->data) {
        t->data = malloc(bytes);
        if (!t->data) return -1;
        t->owns_data = true;
    }
    cl_int err = clEnqueueReadBuffer(b->queue, e->gpu_buf, CL_FALSE, 0, bytes, t->data,
                                     0, NULL, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

static void ocl_release_intermediate_buffers(CMLOpenCLIRBackend* b) {
    /* Keep all buffers for reuse (pooling). Just mark intermediates as invalid
     * and clear tensor pointers so they can be reused by ocl_alloc_output. */
    for (int i = 0; i < b->buffer_count; i++) {
        if (b->buffers[i].is_input) {
            b->buffers[i].tensor = NULL;  /* prevent stale pointer match */
        } else {
            b->buffers[i].tensor = NULL;
            b->buffers[i].data_ptr = NULL;
            b->buffers[i].valid = false;
            /* Keep gpu_buf alive for reuse */
        }
    }
}

static void ocl_release_all_buffers(CMLOpenCLIRBackend* b) {
    for (int i = 0; i < b->buffer_count; i++) {
        if (b->buffers[i].gpu_buf)
            clReleaseMemObject(b->buffers[i].gpu_buf);
    }
    b->buffer_count = 0;
}

/* ─── BEAM autotuner for GEMM kernels ─────────────────────────────────── */

/* Generate OpenCL source for a parameterized GEMM kernel.
 * Returns heap-allocated string. Caller must free(). */
static char* ocl_beam_generate_gemm(const CMLGemmVariantParams* p, int id) {
    int wg_x = p->tsn / p->reg_n;
    int wg_y = p->tsm / p->reg_m;
    int wg_total = wg_x * wg_y;
    int a_tile = p->tsm * p->tsk;
    int b_tile = p->tsk * p->tsn;
    int a_loads = a_tile / wg_total;
    int b_loads = b_tile / wg_total;
    int slm_w_a = p->transpose_a ? (p->tsm + p->slm_pad) : (p->tsk + p->slm_pad);
    int slm_h_a = p->transpose_a ? p->tsk : p->tsm;
    int slm_w_b = p->tsn + p->slm_pad;

    char* buf = (char*)malloc(16384);
    if (!buf) return NULL;
    int off = 0;

#define P(...) off += snprintf(buf + off, 16384 - off, __VA_ARGS__)

    P("#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n");
    P("__attribute__((intel_reqd_sub_group_size(8)))\n");
    P("__kernel void beam_gemm_%d(\n", id);
    P("    __global const float* restrict A,\n");
    P("    __global const float* restrict B,\n");
    P("    __global float* restrict C, int M, int N, int K) {\n");
    P("    int tidC = get_local_id(0), tidR = get_local_id(1);\n");
    P("    int gidR = get_group_id(1) * %d, gidC = get_group_id(0) * %d;\n", p->tsm, p->tsn);

    if (p->transpose_a)
        P("    __local float As_T[%d][%d];\n", slm_h_a, slm_w_a);
    else
        P("    __local float As[%d][%d];\n", slm_h_a, slm_w_a);
    P("    __local float Bs[%d][%d];\n", p->tsk, slm_w_b);

    P("    float acc[%d][%d];\n", p->reg_m, p->reg_n);
    P("    for (int i = 0; i < %d; i++) for (int j = 0; j < %d; j++) acc[i][j] = 0.0f;\n",
      p->reg_m, p->reg_n);
    P("    int tid = tidR * %d + tidC;\n", wg_x);

    P("    for (int t = 0; t < K / %d; t++) {\n", p->tsk);
    P("        int tK = t * %d;\n", p->tsk);

    if (p->transpose_a) {
        /* Cooperative coalesced loading: As_T[col][row] = A[row, col] */
        P("        for (int l = 0; l < %d; l++) {\n", a_loads);
        P("            int idx = tid + l * %d;\n", wg_total);
        if (p->tsk == 16) {
            P("            int lr = idx >> 4, lc = idx & 15;\n");
        } else {
            P("            int lr = idx / %d, lc = idx %% %d;\n", p->tsk, p->tsk);
        }
        P("            As_T[lc][lr] = A[(gidR + lr) * K + tK + lc];\n");
        P("        }\n");
    } else {
        P("        for (int l = 0; l < %d; l++) {\n", a_loads);
        P("            int idx = tid + l * %d;\n", wg_total);
        P("            int lr = idx / %d, lc = idx %% %d;\n", p->tsk, p->tsk);
        P("            As[lr][lc] = A[(gidR + lr) * K + tK + lc];\n");
        P("        }\n");
    }

    P("        for (int l = 0; l < %d; l++) {\n", b_loads);
    P("            int idx = tid + l * %d;\n", wg_total);
    if (p->tsn == 128) {
        P("            Bs[idx >> 7][idx & 127] = B[(tK + (idx >> 7)) * N + gidC + (idx & 127)];\n");
    } else if (p->tsn == 64) {
        P("            Bs[idx >> 6][idx & 63] = B[(tK + (idx >> 6)) * N + gidC + (idx & 63)];\n");
    } else {
        P("            int kk = idx / %d, cc = idx %% %d;\n", p->tsn, p->tsn);
        P("            Bs[kk][cc] = B[(tK + kk) * N + gidC + cc];\n");
    }
    P("        }\n");
    P("        barrier(CLK_LOCAL_MEM_FENCE);\n");

    P("        for (int k = 0; k < %d; k++) {\n", p->tsk);

    for (int i = 0; i < p->reg_m; i++) {
        if (p->transpose_a)
            P("            float a%d = As_T[k][tidR*%d+%d];\n", i, p->reg_m, i);
        else
            P("            float a%d = As[tidR*%d+%d][k];\n", i, p->reg_m, i);
    }
    for (int j = 0; j < p->reg_n; j++)
        P("            float b%d = Bs[k][tidC*%d+%d];\n", j, p->reg_n, j);

    for (int i = 0; i < p->reg_m; i++) {
        for (int j = 0; j < p->reg_n; j++)
            P("            acc[%d][%d]=mad(a%d,b%d,acc[%d][%d]);\n", i, j, i, j, i, j);
    }
    P("        }\n");
    P("        barrier(CLK_LOCAL_MEM_FENCE);\n");
    P("    }\n");

    P("    for (int i = 0; i < %d; i++)\n", p->reg_m);
    P("        for (int j = 0; j < %d; j++)\n", p->reg_n);
    P("            C[(gidR+tidR*%d+i)*N+gidC+tidC*%d+j] = acc[i][j];\n", p->reg_m, p->reg_n);
    P("}\n");

#undef P
    return buf;
}

static bool ocl_beam_params_valid(const CMLGemmVariantParams* p) {
    int wg_x = p->tsn / p->reg_n;
    int wg_y = p->tsm / p->reg_m;
    int wg_total = wg_x * wg_y;
    if (wg_x < 4 || wg_y < 4 || wg_total > 512) return false;
    if ((p->tsm * p->tsk) % wg_total != 0) return false;
    if ((p->tsk * p->tsn) % wg_total != 0) return false;
    int regs = p->reg_m * p->reg_n + p->reg_m + p->reg_n;
    if (regs > 80) return false;
    int slm_a = (p->transpose_a ? p->tsk : p->tsm) * (p->transpose_a ? (p->tsm + p->slm_pad) : (p->tsk + p->slm_pad));
    int slm_b = p->tsk * (p->tsn + p->slm_pad);
    if ((slm_a + slm_b) * 4 > 65536) return false;
    return true;
}

/* Compile all BEAM GEMM variants. Called during backend init.
 * Order matters: known-best configs first (128x128 8x8, 64x64 8x4/4x8)
 * so autotuning with early-exit finds the winner quickly. */
static void ocl_beam_compile_variants(CMLOpenCLIRBackend* b) {
    /* Priority order: large tiles first (for big GEMM), then small tiles */
    static const int tile_sizes[][2] = {{128,128}, {128,64}, {64,128}, {64,64}};
    static const int tsk_values[] = {16, 8};
    /* Priority: high-compute register blocks first */
    static const int reg_blocks[][2] = {{8,8}, {8,4}, {4,8}, {4,4}};
    static const int pad_values[] = {0, 1};

    b->gemm_variant_count = 0;

    for (int ti = 0; ti < 4; ti++)
    for (int ki = 0; ki < 2; ki++)
    for (int ri = 0; ri < 4; ri++)
    for (int pi = 0; pi < 2; pi++) {
        if (b->gemm_variant_count >= CML_OCL_MAX_GEMM_VARIANTS) break;

        CMLGemmVariantParams p = {
            .tsm = tile_sizes[ti][0], .tsn = tile_sizes[ti][1],
            .tsk = tsk_values[ki],
            .reg_m = reg_blocks[ri][0], .reg_n = reg_blocks[ri][1],
            .slm_pad = pad_values[pi],
            .transpose_a = true  /* always use A-transposed — proven faster */
        };
        if (!ocl_beam_params_valid(&p)) continue;

        int idx = b->gemm_variant_count;
        char* src = ocl_beam_generate_gemm(&p, idx);
        if (!src) continue;

        cl_int err;
        cl_program prog = clCreateProgramWithSource(b->context, 1, (const char**)&src, NULL, &err);
        free(src);
        if (err != CL_SUCCESS) continue;

        err = clBuildProgram(prog, 1, &b->device, "-cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
        if (err != CL_SUCCESS) {
            clReleaseProgram(prog);
            continue;
        }

        char kname[32];
        snprintf(kname, sizeof(kname), "beam_gemm_%d", idx);
        cl_kernel kern = clCreateKernel(prog, kname, &err);
        if (err != CL_SUCCESS) {
            clReleaseProgram(prog);
            continue;
        }

        CMLGemmVariant* v = &b->gemm_variants[idx];
        v->params = p;
        v->program = prog;
        v->kernel = kern;
        v->local_size[0] = p.tsn / p.reg_n;
        v->local_size[1] = p.tsm / p.reg_m;
        v->valid = true;
        b->gemm_variant_count++;
    }

    LOG_INFO("BEAM: compiled %d GEMM variants", b->gemm_variant_count);
}

static int ocl_beam_cache_lookup(CMLOpenCLIRBackend* b, int M, int N, int K) {
    uint64_t key = ((uint64_t)M << 40) | ((uint64_t)N << 20) | (uint64_t)K;
    for (int i = 0; i < CML_OCL_GEMM_CACHE_SIZE; i++) {
        if (b->gemm_cache[i].occupied && b->gemm_cache[i].key == key)
            return b->gemm_cache[i].variant_idx;
    }
    return -1;
}

static void ocl_beam_cache_store(CMLOpenCLIRBackend* b, int M, int N, int K, int vidx) {
    uint64_t key = ((uint64_t)M << 40) | ((uint64_t)N << 20) | (uint64_t)K;
    for (int i = 0; i < CML_OCL_GEMM_CACHE_SIZE; i++) {
        if (!b->gemm_cache[i].occupied) {
            b->gemm_cache[i].key = key;
            b->gemm_cache[i].variant_idx = vidx;
            b->gemm_cache[i].occupied = true;
            return;
        }
    }
    b->gemm_cache[0].key = key;
    b->gemm_cache[0].variant_idx = vidx;
}

static int ocl_beam_autotune(CMLOpenCLIRBackend* b, int M, int N, int K,
                              cl_mem buf_a, cl_mem buf_b) {
    int cached = ocl_beam_cache_lookup(b, M, N, K);
    if (cached >= 0) return cached;

    size_t out_bytes = (size_t)M * N * sizeof(float);
    cl_int err;
    cl_mem tmp_out = clCreateBuffer(b->context, CL_MEM_READ_WRITE, out_bytes, NULL, &err);
    if (err != CL_SUCCESS) return -1;

    int best_idx = -1;
    double best_time = 1e18;

    int max_try = b->beam_width > 0 ? b->beam_width : b->gemm_variant_count;
    LOG_INFO("BEAM: autotuning GEMM M=%d N=%d K=%d (width=%d, %d variants)...",
             M, N, K, max_try, b->gemm_variant_count);

    int tried = 0;
    for (int v = 0; v < b->gemm_variant_count && tried < max_try; v++) {
        CMLGemmVariant* var = &b->gemm_variants[v];
        if (!var->valid) continue;
        CMLGemmVariantParams* p = &var->params;

        if (M < p->tsm || N < p->tsn) continue;
        if (M % p->tsm != 0 || N % p->tsn != 0 || K % p->tsk != 0) continue;
        tried++;

        size_t global[2] = {
            (size_t)(N / p->tsn) * var->local_size[0],
            (size_t)(M / p->tsm) * var->local_size[1]
        };

        clSetKernelArg(var->kernel, 0, sizeof(cl_mem), &buf_a);
        clSetKernelArg(var->kernel, 1, sizeof(cl_mem), &buf_b);
        clSetKernelArg(var->kernel, 2, sizeof(cl_mem), &tmp_out);
        clSetKernelArg(var->kernel, 3, sizeof(int), &M);
        clSetKernelArg(var->kernel, 4, sizeof(int), &N);
        clSetKernelArg(var->kernel, 5, sizeof(int), &K);

        clEnqueueNDRangeKernel(b->profiling_queue, var->kernel, 2, NULL,
                               global, var->local_size, 0, NULL, NULL);
        clFinish(b->profiling_queue);

        cl_event probe_ev;
        clEnqueueNDRangeKernel(b->profiling_queue, var->kernel, 2, NULL,
                               global, var->local_size, 0, NULL, &probe_ev);
        clFinish(b->profiling_queue);
        {
            cl_ulong pt0, pt1;
            clGetEventProfilingInfo(probe_ev, CL_PROFILING_COMMAND_START, sizeof(pt0), &pt0, NULL);
            clGetEventProfilingInfo(probe_ev, CL_PROFILING_COMMAND_END, sizeof(pt1), &pt1, NULL);
            clReleaseEvent(probe_ev);
            double probe_ns = (double)(pt1 - pt0);
            /* Skip if >4x slower than best (register spill or bad config) */
            if (best_time < 1e17 && probe_ns > best_time * 4.0) {
                LOG_INFO("BEAM:   V%d: TSM=%d TSN=%d TSK=%d reg=%dx%d pad=%d -> SKIP (probe %.1fms, best %.1fms)",
                         v, p->tsm, p->tsn, p->tsk, p->reg_m, p->reg_n, p->slm_pad,
                         probe_ns/1e6, best_time/1e6);
                continue;
            }
        }

        clEnqueueNDRangeKernel(b->profiling_queue, var->kernel, 2, NULL,
                               global, var->local_size, 0, NULL, NULL);
        clFinish(b->profiling_queue);

        cl_event events[3];
        for (int r = 0; r < 3; r++) {
            clEnqueueNDRangeKernel(b->profiling_queue, var->kernel, 2, NULL,
                                   global, var->local_size, 0, NULL, &events[r]);
        }
        clFinish(b->profiling_queue);

        double times[3];
        for (int r = 0; r < 3; r++) {
            cl_ulong t0, t1;
            clGetEventProfilingInfo(events[r], CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
            clGetEventProfilingInfo(events[r], CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL);
            times[r] = (double)(t1 - t0);
            clReleaseEvent(events[r]);
        }

        if (times[0] > times[1]) { double t = times[0]; times[0] = times[1]; times[1] = t; }
        if (times[1] > times[2]) { double t = times[1]; times[1] = times[2]; times[2] = t; }
        if (times[0] > times[1]) { double t = times[0]; times[0] = times[1]; times[1] = t; }
        double median_ns = times[1];

        double gflops = 2.0 * M * N * K / median_ns;
        LOG_INFO("BEAM:   V%d: TSM=%d TSN=%d TSK=%d reg=%dx%d pad=%d -> %.1f GFLOPS (%.2f ms)",
                 v, p->tsm, p->tsn, p->tsk, p->reg_m, p->reg_n, p->slm_pad,
                 gflops, median_ns / 1e6);

        if (median_ns < best_time) {
            best_time = median_ns;
            best_idx = v;
        }
    }

    clReleaseMemObject(tmp_out);

    if (best_idx >= 0) {
        CMLGemmVariantParams* bp = &b->gemm_variants[best_idx].params;
        double gflops = 2.0 * M * N * K / best_time;
        LOG_INFO("BEAM: WINNER V%d (TSM=%d TSN=%d TSK=%d reg=%dx%d pad=%d) -> %.1f GFLOPS",
                 best_idx, bp->tsm, bp->tsn, bp->tsk, bp->reg_m, bp->reg_n, bp->slm_pad, gflops);
        ocl_beam_cache_store(b, M, N, K, best_idx);
    }

    return best_idx;
}

/* ─── Backend lifecycle ────────────────────────────────────────────────── */

bool cml_opencl_ir_available(void) {
    cl_uint n = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &n);
    if (err != CL_SUCCESS || n == 0) return false;

    cl_platform_id plat;
    clGetPlatformIDs(1, &plat, NULL);
    cl_uint nd = 0;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, NULL, &nd);
    return (err == CL_SUCCESS && nd > 0);
}

CMLOpenCLIRBackend* cml_opencl_ir_backend_create(void) {
    CMLOpenCLIRBackend* b = (CMLOpenCLIRBackend*)calloc(1, sizeof(CMLOpenCLIRBackend));
    return b;
}

int cml_opencl_ir_backend_init(CMLOpenCLIRBackend* b) {
    if (!b) return -1;
    if (b->initialized) return 0;

    cl_int err;

    cl_uint np;
    err = clGetPlatformIDs(0, NULL, &np);
    if (err != CL_SUCCESS || np == 0) { LOG_ERROR("No OpenCL platforms"); return -1; }

    cl_platform_id platforms[8];
    clGetPlatformIDs(np > 8 ? 8 : np, platforms, NULL);

    bool found = false;
    for (cl_uint p = 0; p < np && p < 8; p++) {
        cl_uint nd;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 1, &b->device, &nd);
        if (err == CL_SUCCESS && nd > 0) {
            b->platform = platforms[p];
            found = true;
            break;
        }
    }
    if (!found) { LOG_ERROR("No OpenCL GPU device found"); return -1; }

    clGetDeviceInfo(b->device, CL_DEVICE_NAME, sizeof(b->device_name), b->device_name, NULL);
    cl_ulong mem;
    clGetDeviceInfo(b->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL);
    b->total_memory = (size_t)mem;
    size_t wgs;
    clGetDeviceInfo(b->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wgs), &wgs, NULL);
    b->max_work_group_size = (uint32_t)wgs;
    cl_uint cu;
    clGetDeviceInfo(b->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    b->max_compute_units = cu;

    LOG_INFO("OpenCL IR backend: %s (%u CUs, %zu MB, max WG %u)",
             b->device_name, cu, b->total_memory / (1024 * 1024), b->max_work_group_size);

    b->context = clCreateContext(NULL, 1, &b->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) { LOG_ERROR("clCreateContext failed: %d", err); return -1; }

#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, 0, 0 };
    b->queue = clCreateCommandQueueWithProperties(b->context, b->device, props, &err);
#else
    b->queue = clCreateCommandQueue(b->context, b->device, 0, &err);
#endif
    if (err != CL_SUCCESS) { LOG_ERROR("clCreateCommandQueue failed: %d", err); return -1; }

    const char* src = g_ocl_kernel_src;
    size_t len = strlen(src);
    b->program = clCreateProgramWithSource(b->context, 1, &src, &len, &err);
    if (err != CL_SUCCESS) { LOG_ERROR("clCreateProgramWithSource failed: %d", err); return -1; }

    err = clBuildProgram(b->program, 1, &b->device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log_buf[4096];
        clGetProgramBuildInfo(b->program, b->device, CL_PROGRAM_BUILD_LOG,
                              sizeof(log_buf), log_buf, NULL);
        LOG_ERROR("OpenCL kernel build failed:\n%s", log_buf);
        return -1;
    }

#define GET_KERNEL(field, name) \
    b->field = clCreateKernel(b->program, name, &err); \
    if (err != CL_SUCCESS) { LOG_ERROR("Missing kernel: " name); return -1; }

    GET_KERNEL(k_matmul,      "matmul");
    GET_KERNEL(k_matmul_fused_bias_relu, "matmul_fused_bias_relu");
    GET_KERNEL(k_matmul_naive, "matmul_naive");
    GET_KERNEL(k_add,          "ew_add");
    GET_KERNEL(k_sub,          "ew_sub");
    GET_KERNEL(k_mul,          "ew_mul");
    GET_KERNEL(k_div,          "ew_div");
    GET_KERNEL(k_neg,          "ew_neg");
    GET_KERNEL(k_relu,         "ew_relu");
    GET_KERNEL(k_sigmoid,      "ew_sigmoid");
    GET_KERNEL(k_tanh,         "ew_tanh_k");
    GET_KERNEL(k_exp,          "ew_exp");
    GET_KERNEL(k_log,          "ew_log");
    GET_KERNEL(k_sqrt,         "ew_sqrt_k");
    GET_KERNEL(k_fill,         "ew_fill");
    GET_KERNEL(k_sum_reduce,   "reduce_sum");
    GET_KERNEL(k_max_reduce,   "reduce_max");
#undef GET_KERNEL

    b->k_mean_reduce = NULL; /* mean = sum / n, composed from sum_reduce */

    b->initialized = true;

    const char* beam_env = getenv("CML_BEAM");
    int beam_width = 0;  /* default: all variants */
    if (beam_env) {
        beam_width = atoi(beam_env);
        if (beam_width < 0) beam_width = 0;
    }

    /* CML_BEAM=0 explicitly disables, absent means auto-enable */
    if (beam_env && beam_width == 0) {
        b->beam_width = 0;  /* explicitly disabled */
    } else {
#ifdef CL_VERSION_2_0
        cl_queue_properties prof_props[] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
        };
        b->profiling_queue = clCreateCommandQueueWithProperties(b->context, b->device, prof_props, &err);
#else
        b->profiling_queue = clCreateCommandQueue(b->context, b->device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
        if (err != CL_SUCCESS) {
            LOG_WARNING("BEAM: failed to create profiling queue, disabling");
            b->beam_width = 0;
        } else {
            ocl_beam_compile_variants(b);
            if (b->gemm_variant_count == 0) {
                LOG_WARNING("BEAM: no valid variants compiled, disabling");
                b->beam_width = 0;
            } else {
                b->beam_width = beam_width > 0 ? beam_width : b->gemm_variant_count;
                LOG_INFO("BEAM: enabled with width=%d (%d variants compiled)",
                         b->beam_width, b->gemm_variant_count);
            }
        }
    }

    return 0;
}

void cml_opencl_ir_backend_free(CMLOpenCLIRBackend* b) {
    if (!b) return;

    ocl_release_all_buffers(b);

#define REL_KERNEL(k) if (b->k) clReleaseKernel(b->k)
    REL_KERNEL(k_matmul);
    REL_KERNEL(k_matmul_fused_bias_relu);
    REL_KERNEL(k_matmul_naive);
    REL_KERNEL(k_add);
    REL_KERNEL(k_sub);
    REL_KERNEL(k_mul);
    REL_KERNEL(k_div);
    REL_KERNEL(k_neg);
    REL_KERNEL(k_relu);
    REL_KERNEL(k_sigmoid);
    REL_KERNEL(k_tanh);
    REL_KERNEL(k_exp);
    REL_KERNEL(k_log);
    REL_KERNEL(k_sqrt);
    REL_KERNEL(k_fill);
    REL_KERNEL(k_sum_reduce);
    REL_KERNEL(k_max_reduce);
#undef REL_KERNEL

    for (int i = 0; i < b->gemm_variant_count; i++) {
        if (b->gemm_variants[i].valid) {
            if (b->gemm_variants[i].kernel)  clReleaseKernel(b->gemm_variants[i].kernel);
            if (b->gemm_variants[i].program) clReleaseProgram(b->gemm_variants[i].program);
        }
    }
    if (b->profiling_queue) clReleaseCommandQueue(b->profiling_queue);

    if (b->program) clReleaseProgram(b->program);
    if (b->queue)   clReleaseCommandQueue(b->queue);
    if (b->context) clReleaseContext(b->context);

    free(b);
}

/* ─── Node execution helpers ───────────────────────────────────────────── */

static int ocl_exec_matmul(CMLOpenCLIRBackend* b, struct IRNode* node,
                            cl_mem buf_a, cl_mem buf_b, cl_mem buf_out) {
    Tensor* a = node->inputs[0];
    Tensor* bb = node->inputs[1];

    int M, K, N;
    if (a->ndim == 2 && bb->ndim == 2) {
        M = a->shape[0]; K = a->shape[1]; N = bb->shape[1];
    } else if (a->ndim == 1 && bb->ndim == 2) {
        M = 1; K = a->shape[0]; N = bb->shape[1];
    } else if (a->ndim == 2 && bb->ndim == 1) {
        M = a->shape[0]; K = a->shape[1]; N = 1;
    } else {
        return -1; /* batched matmul not yet supported */
    }

    cl_kernel kernel;
    size_t global[2], local[2];

    /* BEAM autotuner dispatch */
    if (b->beam_width > 0 && M >= 64 && N >= 64) {
        int vidx = ocl_beam_autotune(b, M, N, K, buf_a, buf_b);
        if (vidx >= 0) {
            CMLGemmVariant* var = &b->gemm_variants[vidx];
            CMLGemmVariantParams* p = &var->params;
            if (M % p->tsm == 0 && N % p->tsn == 0 && K % p->tsk == 0) {
                global[0] = (size_t)(N / p->tsn) * var->local_size[0];
                global[1] = (size_t)(M / p->tsm) * var->local_size[1];
                clSetKernelArg(var->kernel, 0, sizeof(cl_mem), &buf_a);
                clSetKernelArg(var->kernel, 1, sizeof(cl_mem), &buf_b);
                clSetKernelArg(var->kernel, 2, sizeof(cl_mem), &buf_out);
                clSetKernelArg(var->kernel, 3, sizeof(int), &M);
                clSetKernelArg(var->kernel, 4, sizeof(int), &N);
                clSetKernelArg(var->kernel, 5, sizeof(int), &K);
                cl_int err = clEnqueueNDRangeKernel(b->queue, var->kernel, 2, NULL,
                                                     global, var->local_size, 0, NULL, NULL);
                return (err == CL_SUCCESS) ? 0 : -1;
            }
        }
    }

    if ((M % 128) == 0 && (N % 128) == 0 && (K % 16) == 0) {
        /* V3 GEMM: float4 loads, A-transposed SLM, 8×8 register block. */
        kernel = b->k_matmul;
        global[0] = (size_t)(N / 128) * 16;
        global[1] = (size_t)(M / 128) * 16;
        local[0] = 16; local[1] = 16;
    } else {
        /* Naive fallback: one thread per output element (small or non-aligned). */
        kernel = b->k_matmul_naive;
        global[0] = M;
        global[1] = N;
        local[0] = 0; local[1] = 0;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_out);
    clSetKernelArg(kernel, 3, sizeof(int), &M);
    clSetKernelArg(kernel, 4, sizeof(int), &N);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    cl_int err;
    if ((M % 128) == 0 && (N % 128) == 0 && (K % 16) == 0) {
        err = clEnqueueNDRangeKernel(b->queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
    } else {
        err = clEnqueueNDRangeKernel(b->queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    }
    return (err == CL_SUCCESS) ? 0 : -1;
}

static int ocl_exec_binary(CMLOpenCLIRBackend* b, cl_kernel kernel,
                            struct IRNode* node,
                            cl_mem buf_a, cl_mem buf_b, cl_mem buf_out) {
    Tensor* a = node->inputs[0];
    Tensor* bb = node->inputs[1];
    Tensor* out = node->output;
    int n = (int)out->numel;

    /* stride params: 0 = no broadcast (use i directly), >0 = modulo wrap */
    int sa = (a->numel == (size_t)n) ? 0 : (int)a->numel;
    int sb = (bb->numel == (size_t)n) ? 0 : (int)bb->numel;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(kernel, 1, sizeof(int), &sa);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_b);
    clSetKernelArg(kernel, 3, sizeof(int), &sb);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &buf_out);
    clSetKernelArg(kernel, 5, sizeof(int), &n);

    size_t global = (size_t)n;
    cl_int err = clEnqueueNDRangeKernel(b->queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

static int ocl_exec_unary(CMLOpenCLIRBackend* b, cl_kernel kernel,
                           cl_mem buf_in, cl_mem buf_out, int n) {
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    size_t global = (size_t)n;
    cl_int err = clEnqueueNDRangeKernel(b->queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

static int ocl_exec_reduce(CMLOpenCLIRBackend* b, cl_kernel kernel,
                            cl_mem buf_in, cl_mem buf_out, int n, bool is_mean) {
    /* Multi-pass reduction: reduce n elements via work-groups of 256 */
    int wg_size = 256;
    if (wg_size > (int)b->max_work_group_size)
        wg_size = (int)b->max_work_group_size;

    /* Round up to power of 2 for reduction */
    int actual_wg = wg_size;
    while (actual_wg > n) actual_wg >>= 1;
    if (actual_wg < 1) actual_wg = 1;

    int num_groups = (n + actual_wg - 1) / actual_wg;
    size_t global = (size_t)(num_groups * actual_wg);
    size_t local = (size_t)actual_wg;

    cl_int err;

    if (num_groups == 1) {
        /* Single pass — output directly */
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
        clSetKernelArg(kernel, 2, local * sizeof(float), NULL);
        clSetKernelArg(kernel, 3, sizeof(int), &n);
        err = clEnqueueNDRangeKernel(b->queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        if (err != CL_SUCCESS) return -1;
    } else {
        /* Multi-pass: first pass into temp, then reduce temp */
        cl_mem temp = clCreateBuffer(b->context, CL_MEM_READ_WRITE,
                                     num_groups * sizeof(float), NULL, &err);
        if (err != CL_SUCCESS) return -1;

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &temp);
        clSetKernelArg(kernel, 2, local * sizeof(float), NULL);
        clSetKernelArg(kernel, 3, sizeof(int), &n);
        err = clEnqueueNDRangeKernel(b->queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        if (err != CL_SUCCESS) { clReleaseMemObject(temp); return -1; }

        /* Second pass */
        int n2 = num_groups;
        int wg2 = actual_wg;
        while (wg2 > n2) wg2 >>= 1;
        if (wg2 < 1) wg2 = 1;
        size_t global2 = (size_t)wg2;
        size_t local2 = (size_t)wg2;

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &temp);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
        clSetKernelArg(kernel, 2, local2 * sizeof(float), NULL);
        clSetKernelArg(kernel, 3, sizeof(int), &n2);
        err = clEnqueueNDRangeKernel(b->queue, kernel, 1, NULL, &global2, &local2, 0, NULL, NULL);
        clReleaseMemObject(temp);
        if (err != CL_SUCCESS) return -1;
    }

    /* For mean: read back, divide, write back */
    if (is_mean && n > 0) {
        float result;
        clEnqueueReadBuffer(b->queue, buf_out, CL_TRUE, 0, sizeof(float), &result, 0, NULL, NULL);
        result /= (float)n;
        clEnqueueWriteBuffer(b->queue, buf_out, CL_TRUE, 0, sizeof(float), &result, 0, NULL, NULL);
    }

    return 0;
}

/* ─── Graph execution ──────────────────────────────────────────────────── */

static bool is_gpu_supported(UOpType type) {
    switch (type) {
    case UOP_MATMUL:
    case UOP_ADD: case UOP_SUB: case UOP_MUL: case UOP_DIV:
    case UOP_NEG: case UOP_EXP: case UOP_LOG: case UOP_SQRT:
    case UOP_RELU: case UOP_SIGMOID: case UOP_TANH:
    case UOP_SUM: case UOP_MAX_REDUCE: case UOP_MEAN:
    case UOP_FILL:
    case UOP_RESHAPE: case UOP_EXPAND: case UOP_PERMUTE:
        return true;
    default:
        return false;
    }
}

static bool is_view_op(UOpType type) {
    return type == UOP_RESHAPE || type == UOP_EXPAND || type == UOP_PERMUTE ||
           type == UOP_STRIDE || type == UOP_SLICE;
}

int cml_opencl_execute_graph(CMLOpenCLIRBackend* b, CMLGraph_t ir) {
    if (!b || !b->initialized || !ir)
        return -1;

    /* Estimate total FLOPS. Skip GPU for small graphs where kernel launch
     * overhead dominates — CPU is faster for MLP forward, small conv, etc. */
    {
        int64_t total_flops = 0;
        struct IRNode* scan = ir->head;
        while (scan) {
            if (scan->type == UOP_MATMUL && scan->num_inputs >= 2) {
                Tensor* a = scan->inputs[0];
                Tensor* bb = scan->inputs[1];
                int64_t m = (a->ndim >= 2) ? a->shape[0] : 1;
                int64_t k = (a->ndim >= 2) ? a->shape[a->ndim - 1] : a->shape[0];
                int64_t n = (bb->ndim >= 2) ? bb->shape[bb->ndim - 1] : 1;
                total_flops += 2 * m * k * n;
            } else if (scan->output) {
                total_flops += (int64_t)scan->output->numel;
            }
            scan = scan->next;
        }
        /* Below 50M FLOPS, CPU is faster (avoids kernel launch + D2H/H2D overhead) */
        if (total_flops < 50000000LL)
            return -1;
    }

    struct IRNode* node = ir->head;
    int executed = 0;

    while (node) {
        if (node->is_executed && node->output && node->output->is_executed) {
            node = node->next;
            continue;
        }
        if (!node->output) {
            node = node->next;
            continue;
        }

        Tensor* out = node->output;

        /* View ops: share the input buffer, just update tensor metadata */
        if (is_view_op(node->type)) {
            /* Fall back to CPU for view ops — they're just metadata updates */
            /* Ensure inputs have CPU data */
            for (int i = 0; i < node->num_inputs; i++) {
                Tensor* inp = node->inputs[i];
                if (inp && !inp->data && ocl_find_buffer(b, inp)) {
                    ocl_download(b, inp);
                }
            }
            cpu_execute_node(node);
            node->is_executed = true;
            out->is_executed = true;
            node = node->next;
            continue;
        }

        /* Unsupported op: fall back to CPU */
        if (!is_gpu_supported(node->type)) {
            /* D2H inputs if they're on GPU */
            for (int i = 0; i < node->num_inputs; i++) {
                Tensor* inp = node->inputs[i];
                if (inp && !inp->data && ocl_find_buffer(b, inp)) {
                    ocl_download(b, inp);
                }
            }
            cpu_execute_node(node);
            node->is_executed = true;
            out->is_executed = true;
            node = node->next;
            executed++;
            continue;
        }

        /* FILL op */
        if (node->type == UOP_FILL) {
            /* Allocate output on GPU and fill */
            cl_mem buf_out = ocl_alloc_output(b, out);
            if (!buf_out) goto cpu_fallback;

            float val = 0.0f;
            if (node->params) val = *(float*)node->params;
            int n = (int)out->numel;

            clSetKernelArg(b->k_fill, 0, sizeof(cl_mem), &buf_out);
            clSetKernelArg(b->k_fill, 1, sizeof(float), &val);
            clSetKernelArg(b->k_fill, 2, sizeof(int), &n);

            size_t global = (size_t)n;
            clEnqueueNDRangeKernel(b->queue, b->k_fill, 1, NULL, &global, NULL, 0, NULL, NULL);

            node->is_executed = true;
            out->is_executed = true;
            node = node->next;
            executed++;
            continue;
        }

        if (node->type == UOP_MATMUL && b->k_matmul_fused_bias_relu &&
            node->num_inputs >= 2 && node->inputs[0] && node->inputs[1]) {
            struct IRNode* add_n  = node->next;
            struct IRNode* relu_n = add_n ? add_n->next : NULL;
            if (add_n && relu_n &&
                add_n->type  == UOP_ADD  && !add_n->is_executed &&
                relu_n->type == UOP_RELU && !relu_n->is_executed &&
                add_n->num_inputs == 2 && relu_n->num_inputs == 1 &&
                relu_n->output) {
                Tensor* mm_out   = out; /* matmul output */
                Tensor* bias_t   = (add_n->inputs[0] == mm_out) ? add_n->inputs[1]
                                                                 : add_n->inputs[0];
                Tensor* final_out = relu_n->output;
                Tensor* ta = node->inputs[0], *tb = node->inputs[1];
                if (mm_out && mm_out->ndim == 2 && bias_t && ta && ta->ndim >= 2) {
                    int fM = mm_out->shape[0], fN = mm_out->shape[1];
                    int fK = ta->shape[ta->ndim - 1];
                    if ((fM % 128) == 0 && (fN % 128) == 0 && (fK % 16) == 0 &&
                        (int)bias_t->numel == fN) {
                        cl_mem ba = ocl_ensure_gpu(b, ta);
                        cl_mem bb2 = ocl_ensure_gpu(b, tb);
                        cl_mem bbias = ocl_ensure_gpu(b, bias_t);
                        cl_mem bfinal = ocl_alloc_output(b, final_out);
                        if (ba && bb2 && bbias && bfinal) {
                            size_t gs[2] = {(size_t)(fN/128)*16, (size_t)(fM/128)*16};
                            size_t ls[2] = {16, 16};
                            clSetKernelArg(b->k_matmul_fused_bias_relu, 0, sizeof(cl_mem), &ba);
                            clSetKernelArg(b->k_matmul_fused_bias_relu, 1, sizeof(cl_mem), &bb2);
                            clSetKernelArg(b->k_matmul_fused_bias_relu, 2, sizeof(cl_mem), &bbias);
                            clSetKernelArg(b->k_matmul_fused_bias_relu, 3, sizeof(cl_mem), &bfinal);
                            clSetKernelArg(b->k_matmul_fused_bias_relu, 4, sizeof(int), &fM);
                            clSetKernelArg(b->k_matmul_fused_bias_relu, 5, sizeof(int), &fN);
                            clSetKernelArg(b->k_matmul_fused_bias_relu, 6, sizeof(int), &fK);
                            cl_int ferr = clEnqueueNDRangeKernel(b->queue,
                                              b->k_matmul_fused_bias_relu,
                                              2, NULL, gs, ls, 0, NULL, NULL);
                            if (ferr == CL_SUCCESS) {
                                node->is_executed = out->is_executed = true;
                                add_n->is_executed = true;
                                if (add_n->output) add_n->output->is_executed = true;
                                relu_n->is_executed = final_out->is_executed = true;
                                node = relu_n->next;
                                executed += 3;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        /* Ensure inputs are on GPU */
        cl_mem bufs_in[8] = {0};
        bool input_ok = true;
        for (int i = 0; i < node->num_inputs && i < 8; i++) {
            Tensor* inp = node->inputs[i];
            if (!inp) continue;
            bufs_in[i] = ocl_ensure_gpu(b, inp);
            if (!bufs_in[i]) { input_ok = false; break; }
        }
        if (!input_ok) goto cpu_fallback;

        /* Allocate output on GPU (CPU-side alloc deferred to download) */
        cl_mem buf_out = ocl_alloc_output(b, out);
        if (!buf_out) goto cpu_fallback;

        int rc = -1;
        switch (node->type) {
        case UOP_MATMUL:
            rc = ocl_exec_matmul(b, node, bufs_in[0], bufs_in[1], buf_out);
            break;
        case UOP_ADD:
            rc = ocl_exec_binary(b, b->k_add, node, bufs_in[0], bufs_in[1], buf_out);
            break;
        case UOP_SUB:
            rc = ocl_exec_binary(b, b->k_sub, node, bufs_in[0], bufs_in[1], buf_out);
            break;
        case UOP_MUL:
            rc = ocl_exec_binary(b, b->k_mul, node, bufs_in[0], bufs_in[1], buf_out);
            break;
        case UOP_DIV:
            rc = ocl_exec_binary(b, b->k_div, node, bufs_in[0], bufs_in[1], buf_out);
            break;
        case UOP_NEG:
            rc = ocl_exec_unary(b, b->k_neg, bufs_in[0], buf_out, (int)out->numel);
            break;
        case UOP_EXP:
            rc = ocl_exec_unary(b, b->k_exp, bufs_in[0], buf_out, (int)out->numel);
            break;
        case UOP_LOG:
            rc = ocl_exec_unary(b, b->k_log, bufs_in[0], buf_out, (int)out->numel);
            break;
        case UOP_SQRT:
            rc = ocl_exec_unary(b, b->k_sqrt, bufs_in[0], buf_out, (int)out->numel);
            break;
        case UOP_RELU:
            rc = ocl_exec_unary(b, b->k_relu, bufs_in[0], buf_out, (int)out->numel);
            break;
        case UOP_SIGMOID:
            rc = ocl_exec_unary(b, b->k_sigmoid, bufs_in[0], buf_out, (int)out->numel);
            break;
        case UOP_TANH:
            rc = ocl_exec_unary(b, b->k_tanh, bufs_in[0], buf_out, (int)out->numel);
            break;
        case UOP_SUM:
            rc = ocl_exec_reduce(b, b->k_sum_reduce, bufs_in[0], buf_out,
                                 (int)node->inputs[0]->numel, false);
            break;
        case UOP_MEAN:
            rc = ocl_exec_reduce(b, b->k_sum_reduce, bufs_in[0], buf_out,
                                 (int)node->inputs[0]->numel, true);
            break;
        case UOP_MAX_REDUCE:
            rc = ocl_exec_reduce(b, b->k_max_reduce, bufs_in[0], buf_out,
                                 (int)node->inputs[0]->numel, false);
            break;
        default:
            goto cpu_fallback;
        }

        if (rc != 0) goto cpu_fallback;

        node->is_executed = true;
        out->is_executed = true;
        node = node->next;
        executed++;
        continue;

    cpu_fallback:
        /* Download inputs, execute on CPU */
        for (int i = 0; i < node->num_inputs; i++) {
            Tensor* inp = node->inputs[i];
            if (inp && !inp->data && ocl_find_buffer(b, inp))
                ocl_download(b, inp);
        }
        cpu_execute_node(node);
        node->is_executed = true;
        out->is_executed = true;
        node = node->next;
        executed++;
        continue;
    }

    /* D2H — only download leaf outputs (not consumed by later GPU nodes).
     * Intermediates stay on GPU; they'll be downloaded on-demand if needed. */
    node = ir->head;
    while (node) {
        if (node->output && node->is_executed) {
            Tensor* t = node->output;
            CMLOCLBufferEntry* e = ocl_find_buffer(b, t);
            if (e && e->valid) {
                /* Check if this output feeds into any later executed node */
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
                if (!consumed)
                    ocl_download(b, t);
            }
        }
        node = node->next;
    }

    /* Flush and sync */
    clFinish(b->queue);

    /* Release intermediate GPU buffers, keep input buffers cached */
    ocl_release_intermediate_buffers(b);

    ir->is_executed = true;
    return 0;
}

#endif /* CML_HAS_OPENCL */
