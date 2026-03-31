 * Objective-C implementation using the Metal framework.
 * Entirely guarded by CML_HAS_METAL -- on non-Apple platforms every
 * public function compiles to a safe stub that returns false/NULL/-1.

#include "ops/ir/gpu/metal_backend.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* CML_HAS_METAL -- full implementation */
#ifdef CML_HAS_METAL

#import <Metal/Metal.h>


static const char* g_mtl_static_kernels =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"kernel void k_fill(\n"
"    device float* out      [[buffer(0)]],\n"
"    constant float& val    [[buffer(1)]],\n"
"    constant uint& n       [[buffer(2)]],\n"
"    uint i [[thread_position_in_grid]]) {\n"
"    if (i < n) out[i] = val;\n"
"}\n"
"\n"
"kernel void k_relu(\n"
"    device const float* a  [[buffer(0)]],\n"
"    device float* out      [[buffer(1)]],\n"
"    constant uint& n       [[buffer(2)]],\n"
"    uint i [[thread_position_in_grid]]) {\n"
"    if (i < n) { float x = a[i]; out[i] = x > 0.0f ? x : 0.0f; }\n"
"}\n"
"\n"
"kernel void k_sigmoid(\n"
"    device const float* a  [[buffer(0)]],\n"
"    device float* out      [[buffer(1)]],\n"
"    constant uint& n       [[buffer(2)]],\n"
"    uint i [[thread_position_in_grid]]) {\n"
"    if (i < n) out[i] = 1.0f / (1.0f + exp(-a[i]));\n"
"}\n"
"\n"
"kernel void k_tanh_k(\n"
"    device const float* a  [[buffer(0)]],\n"
"    device float* out      [[buffer(1)]],\n"
"    constant uint& n       [[buffer(2)]],\n"
"    uint i [[thread_position_in_grid]]) {\n"
"    if (i < n) out[i] = tanh(a[i]);\n"
"}\n"
"\n"
"kernel void k_sum_reduce(\n"
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
"}\n"
"\n"
"kernel void k_max_reduce(\n"
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
"}\n"
"\n"
"kernel void k_matmul_opt(\n"
"    device const float* A      [[buffer(0)]],\n"
"    device const float* B      [[buffer(1)]],\n"
"    device float* C            [[buffer(2)]],\n"
"    constant uint& M           [[buffer(3)]],\n"
"    constant uint& N           [[buffer(4)]],\n"
"    constant uint& K           [[buffer(5)]],\n"
"    threadgroup float* As_T    [[threadgroup(0)]],\n"   /* [TSK][TSM+1] = [16][65] */
"    threadgroup float* Bs      [[threadgroup(1)]],\n"   /* [TSK][TSN+1] = [16][65] */
"    uint2 lid [[thread_position_in_threadgroup]],\n"
"    uint2 gid [[threadgroup_position_in_grid]]) {\n"
"    const uint TSM = 64, TSN = 64, TSK = 16;\n"
"    const uint REG = 8;\n"    /* register block side */
"    uint tidC = lid.x, tidR = lid.y;\n"
"    uint gidR = gid.y * TSM, gidC = gid.x * TSN;\n"
"    float acc[8][8];\n"
"    for (uint i = 0; i < 8; i++) for (uint j = 0; j < 8; j++) acc[i][j] = 0.0f;\n"
"    uint tid = tidR * 8 + tidC;\n"    /* linear thread index 0..63 */
"    uint num_tiles = K / TSK;\n"
"    for (uint t = 0; t < num_tiles; t++) {\n"
"        uint tK = t * TSK;\n"
"        for (uint l = 0; l < 16; l++) {\n"
"            uint idx = tid + l * 64;\n"
"            uint lr = idx >> 4;\n"    /* 0..63: row in tile */
"            uint lc = idx & 15;\n"   /* 0..15: col in tile */
"            As_T[lc * 65 + lr] = A[(gidR + lr) * K + tK + lc];\n"
"        }\n"
"        for (uint l = 0; l < 16; l++) {\n"
"            uint idx = tid + l * 64;\n"
"            uint lr = idx >> 6;\n"   /* 0..15: row in tile (K dim) */
"            uint lc = idx & 63;\n"  /* 0..63: col in tile (N dim) */
"            Bs[lr * 65 + lc] = B[(tK + lr) * N + gidC + lc];\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint k = 0; k < TSK; k++) {\n"
"            float a0=As_T[k*65+tidR*8+0], a1=As_T[k*65+tidR*8+1];\n"
"            float a2=As_T[k*65+tidR*8+2], a3=As_T[k*65+tidR*8+3];\n"
"            float a4=As_T[k*65+tidR*8+4], a5=As_T[k*65+tidR*8+5];\n"
"            float a6=As_T[k*65+tidR*8+6], a7=As_T[k*65+tidR*8+7];\n"
"            float b0=Bs[k*65+tidC*8+0], b1=Bs[k*65+tidC*8+1];\n"
"            float b2=Bs[k*65+tidC*8+2], b3=Bs[k*65+tidC*8+3];\n"
"            float b4=Bs[k*65+tidC*8+4], b5=Bs[k*65+tidC*8+5];\n"
"            float b6=Bs[k*65+tidC*8+6], b7=Bs[k*65+tidC*8+7];\n"
"            acc[0][0]+=a0*b0; acc[0][1]+=a0*b1; acc[0][2]+=a0*b2; acc[0][3]+=a0*b3;\n"
"            acc[0][4]+=a0*b4; acc[0][5]+=a0*b5; acc[0][6]+=a0*b6; acc[0][7]+=a0*b7;\n"
"            acc[1][0]+=a1*b0; acc[1][1]+=a1*b1; acc[1][2]+=a1*b2; acc[1][3]+=a1*b3;\n"
"            acc[1][4]+=a1*b4; acc[1][5]+=a1*b5; acc[1][6]+=a1*b6; acc[1][7]+=a1*b7;\n"
"            acc[2][0]+=a2*b0; acc[2][1]+=a2*b1; acc[2][2]+=a2*b2; acc[2][3]+=a2*b3;\n"
"            acc[2][4]+=a2*b4; acc[2][5]+=a2*b5; acc[2][6]+=a2*b6; acc[2][7]+=a2*b7;\n"
"            acc[3][0]+=a3*b0; acc[3][1]+=a3*b1; acc[3][2]+=a3*b2; acc[3][3]+=a3*b3;\n"
"            acc[3][4]+=a3*b4; acc[3][5]+=a3*b5; acc[3][6]+=a3*b6; acc[3][7]+=a3*b7;\n"
"            acc[4][0]+=a4*b0; acc[4][1]+=a4*b1; acc[4][2]+=a4*b2; acc[4][3]+=a4*b3;\n"
"            acc[4][4]+=a4*b4; acc[4][5]+=a4*b5; acc[4][6]+=a4*b6; acc[4][7]+=a4*b7;\n"
"            acc[5][0]+=a5*b0; acc[5][1]+=a5*b1; acc[5][2]+=a5*b2; acc[5][3]+=a5*b3;\n"
"            acc[5][4]+=a5*b4; acc[5][5]+=a5*b5; acc[5][6]+=a5*b6; acc[5][7]+=a5*b7;\n"
"            acc[6][0]+=a6*b0; acc[6][1]+=a6*b1; acc[6][2]+=a6*b2; acc[6][3]+=a6*b3;\n"
"            acc[6][4]+=a6*b4; acc[6][5]+=a6*b5; acc[6][6]+=a6*b6; acc[6][7]+=a6*b7;\n"
"            acc[7][0]+=a7*b0; acc[7][1]+=a7*b1; acc[7][2]+=a7*b2; acc[7][3]+=a7*b3;\n"
"            acc[7][4]+=a7*b4; acc[7][5]+=a7*b5; acc[7][6]+=a7*b6; acc[7][7]+=a7*b7;\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    for (uint i = 0; i < 8; i++)\n"
"        for (uint j = 0; j < 8; j++)\n"
"            C[(gidR + tidR*8 + i) * N + gidC + tidC*8 + j] = acc[i][j];\n"
"}\n"
"\n"
"kernel void k_matmul_fused_bias_relu(\n"
"    device const float* A      [[buffer(0)]],\n"
"    device const float* B      [[buffer(1)]],\n"
"    device const float* bias   [[buffer(2)]],\n"
"    device float* C            [[buffer(3)]],\n"
"    constant uint& M           [[buffer(4)]],\n"
"    constant uint& N           [[buffer(5)]],\n"
"    constant uint& K           [[buffer(6)]],\n"
"    threadgroup float* As_T    [[threadgroup(0)]],\n"
"    threadgroup float* Bs      [[threadgroup(1)]],\n"
"    uint2 lid [[thread_position_in_threadgroup]],\n"
"    uint2 gid [[threadgroup_position_in_grid]]) {\n"
"    const uint TSK = 16;\n"
"    uint tidC = lid.x, tidR = lid.y;\n"
"    uint gidR = gid.y * 64, gidC = gid.x * 64;\n"
"    float acc[8][8];\n"
"    for (uint i = 0; i < 8; i++) for (uint j = 0; j < 8; j++) acc[i][j] = 0.0f;\n"
"    uint tid = tidR * 8 + tidC;\n"
"    uint num_tiles = K / TSK;\n"
"    for (uint t = 0; t < num_tiles; t++) {\n"
"        uint tK = t * TSK;\n"
"        for (uint l = 0; l < 16; l++) {\n"
"            uint idx = tid + l * 64;\n"
"            uint lr = idx >> 4; uint lc = idx & 15;\n"
"            As_T[lc * 65 + lr] = A[(gidR + lr) * K + tK + lc];\n"
"        }\n"
"        for (uint l = 0; l < 16; l++) {\n"
"            uint idx = tid + l * 64;\n"
"            uint lr = idx >> 6; uint lc = idx & 63;\n"
"            Bs[lr * 65 + lc] = B[(tK + lr) * N + gidC + lc];\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint k = 0; k < TSK; k++) {\n"
"            float a0=As_T[k*65+tidR*8+0], a1=As_T[k*65+tidR*8+1];\n"
"            float a2=As_T[k*65+tidR*8+2], a3=As_T[k*65+tidR*8+3];\n"
"            float a4=As_T[k*65+tidR*8+4], a5=As_T[k*65+tidR*8+5];\n"
"            float a6=As_T[k*65+tidR*8+6], a7=As_T[k*65+tidR*8+7];\n"
"            float b0=Bs[k*65+tidC*8+0], b1=Bs[k*65+tidC*8+1];\n"
"            float b2=Bs[k*65+tidC*8+2], b3=Bs[k*65+tidC*8+3];\n"
"            float b4=Bs[k*65+tidC*8+4], b5=Bs[k*65+tidC*8+5];\n"
"            float b6=Bs[k*65+tidC*8+6], b7=Bs[k*65+tidC*8+7];\n"
"            acc[0][0]+=a0*b0; acc[0][1]+=a0*b1; acc[0][2]+=a0*b2; acc[0][3]+=a0*b3;\n"
"            acc[0][4]+=a0*b4; acc[0][5]+=a0*b5; acc[0][6]+=a0*b6; acc[0][7]+=a0*b7;\n"
"            acc[1][0]+=a1*b0; acc[1][1]+=a1*b1; acc[1][2]+=a1*b2; acc[1][3]+=a1*b3;\n"
"            acc[1][4]+=a1*b4; acc[1][5]+=a1*b5; acc[1][6]+=a1*b6; acc[1][7]+=a1*b7;\n"
"            acc[2][0]+=a2*b0; acc[2][1]+=a2*b1; acc[2][2]+=a2*b2; acc[2][3]+=a2*b3;\n"
"            acc[2][4]+=a2*b4; acc[2][5]+=a2*b5; acc[2][6]+=a2*b6; acc[2][7]+=a2*b7;\n"
"            acc[3][0]+=a3*b0; acc[3][1]+=a3*b1; acc[3][2]+=a3*b2; acc[3][3]+=a3*b3;\n"
"            acc[3][4]+=a3*b4; acc[3][5]+=a3*b5; acc[3][6]+=a3*b6; acc[3][7]+=a3*b7;\n"
"            acc[4][0]+=a4*b0; acc[4][1]+=a4*b1; acc[4][2]+=a4*b2; acc[4][3]+=a4*b3;\n"
"            acc[4][4]+=a4*b4; acc[4][5]+=a4*b5; acc[4][6]+=a4*b6; acc[4][7]+=a4*b7;\n"
"            acc[5][0]+=a5*b0; acc[5][1]+=a5*b1; acc[5][2]+=a5*b2; acc[5][3]+=a5*b3;\n"
"            acc[5][4]+=a5*b4; acc[5][5]+=a5*b5; acc[5][6]+=a5*b6; acc[5][7]+=a5*b7;\n"
"            acc[6][0]+=a6*b0; acc[6][1]+=a6*b1; acc[6][2]+=a6*b2; acc[6][3]+=a6*b3;\n"
"            acc[6][4]+=a6*b4; acc[6][5]+=a6*b5; acc[6][6]+=a6*b6; acc[6][7]+=a6*b7;\n"
"            acc[7][0]+=a7*b0; acc[7][1]+=a7*b1; acc[7][2]+=a7*b2; acc[7][3]+=a7*b3;\n"
"            acc[7][4]+=a7*b4; acc[7][5]+=a7*b5; acc[7][6]+=a7*b6; acc[7][7]+=a7*b7;\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    for (uint i = 0; i < 8; i++) {\n"
"        uint row = gidR + tidR*8 + i;\n"
"        for (uint j = 0; j < 8; j++) {\n"
"            uint col = gidC + tidC*8 + j;\n"
"            float v = acc[i][j] + bias[col];\n"
"            C[row * N + col] = v > 0.0f ? v : 0.0f;\n"
"        }\n"
"    }\n"
"}\n";

static id<MTLComputePipelineState> compile_pipeline(id<MTLDevice> device,
                                                     id<MTLLibrary> lib,
                                                     const char* fn_name) {
    NSString* name = [NSString stringWithUTF8String:fn_name];
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        LOG_ERROR("Metal static kernel '%s' not found", fn_name);
        return nil;
    }
    NSError* err = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        LOG_ERROR("Metal pipeline '%s' failed: %s", fn_name,
                  err ? [[err localizedDescription] UTF8String] : "unknown");
    }
    return pso;
}

/* ── Availability ── */

bool cml_metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
}

/* ── Lifecycle ── */

CMLMetalBackend* cml_metal_backend_create(void) {
    CMLMetalBackend* backend = (CMLMetalBackend*)calloc(1, sizeof(CMLMetalBackend));
    if (!backend) {
        LOG_ERROR("Failed to allocate Metal backend");
        return NULL;
    }
    return backend;
}

int cml_metal_backend_init(CMLMetalBackend* backend) {
    if (!backend) return -1;

    if (backend->initialized) {
        LOG_DEBUG("Metal backend already initialized");
        return 0;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            LOG_ERROR("MTLCreateSystemDefaultDevice() returned nil");
            return -1;
        }

        CFRetain((__bridge CFTypeRef)device);
        backend->device = (__bridge void*)device;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            LOG_ERROR("Failed to create MTLCommandQueue");
            CFRelease((__bridge CFTypeRef)device);
            backend->device = NULL;
            return -1;
        }
        backend->command_queue = (__bridge_retained void*)queue;

        const char* name = [[device name] UTF8String];
        if (name) {
            strncpy(backend->device_name, name, sizeof(backend->device_name) - 1);
            backend->device_name[sizeof(backend->device_name) - 1] = '\0';
        } else {
            strncpy(backend->device_name, "Unknown Metal Device",
                    sizeof(backend->device_name) - 1);
        }

        if ([device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
            backend->total_memory = (uint64_t)[device recommendedMaxWorkingSetSize];
        } else {
            backend->total_memory = 0;
        }

        NSString* src = [NSString stringWithUTF8String:g_mtl_static_kernels];
        NSError* err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            LOG_ERROR("Metal static library compile failed: %s",
                      err ? [[err localizedDescription] UTF8String] : "unknown");
        } else {
#define COMPILE_PSO(field, name) \
            { id<MTLComputePipelineState> pso = compile_pipeline(device, lib, name); \
              if (pso) backend->field = (__bridge_retained void*)pso; }

            COMPILE_PSO(k_fill,                   "k_fill");
            COMPILE_PSO(k_relu,                   "k_relu");
            COMPILE_PSO(k_sigmoid,                "k_sigmoid");
            COMPILE_PSO(k_tanh_k,                 "k_tanh_k");
            COMPILE_PSO(k_sum_reduce,             "k_sum_reduce");
            COMPILE_PSO(k_max_reduce,             "k_max_reduce");
            COMPILE_PSO(k_matmul_opt,             "k_matmul_opt");
            COMPILE_PSO(k_matmul_fused_bias_relu, "k_matmul_fused_bias_relu");
#undef COMPILE_PSO
            LOG_INFO("Metal static kernels compiled successfully");
        }

        backend->initialized = true;
        backend->buffer_count = 0;

        LOG_INFO("Metal device: %s", backend->device_name);
        LOG_INFO("  Recommended working set: %.2f GB",
                 backend->total_memory / (1024.0 * 1024.0 * 1024.0));
    }

    return 0;
}

void cml_metal_backend_free(CMLMetalBackend* backend) {
    if (!backend) return;

    if (backend->initialized) {
        @autoreleasepool {
            for (int i = 0; i < backend->buffer_count; i++) {
                if (backend->buffers[i].gpu_buf) {
                    CFRelease((CFTypeRef)backend->buffers[i].gpu_buf);
                    backend->buffers[i].gpu_buf = NULL;
                }
            }
            backend->buffer_count = 0;

            for (int i = 0; i < backend->dyn_cache_count; i++) {
                if (backend->dyn_cache[i].pso) {
                    CFRelease((CFTypeRef)backend->dyn_cache[i].pso);
                    backend->dyn_cache[i].pso = NULL;
                }
            }
            backend->dyn_cache_count = 0;

#define REL_PSO(field) if (backend->field) { CFRelease((CFTypeRef)backend->field); backend->field = NULL; }
            REL_PSO(k_fill);
            REL_PSO(k_relu);
            REL_PSO(k_sigmoid);
            REL_PSO(k_tanh_k);
            REL_PSO(k_sum_reduce);
            REL_PSO(k_max_reduce);
            REL_PSO(k_matmul_opt);
            REL_PSO(k_matmul_fused_bias_relu);
#undef REL_PSO

            if (backend->command_queue) {
                CFRelease((CFTypeRef)backend->command_queue);
                backend->command_queue = NULL;
            }
            if (backend->device) {
                CFRelease((CFTypeRef)backend->device);
                backend->device = NULL;
            }
        }
        backend->initialized = false;
    }

    free(backend);
}

/* ── Compilation ── */

CMLMetalKernel* cml_metal_compile_msl(CMLMetalBackend* backend,
                                       const char* msl_source,
                                       const char* function_name) {
    if (!backend || !backend->initialized || !msl_source || !function_name) {
        LOG_ERROR("Invalid arguments to cml_metal_compile_msl");
        return NULL;
    }

    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)backend->device;

        NSString* src = [NSString stringWithUTF8String:msl_source];
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:src
                                                      options:nil
                                                        error:&error];
        if (!library) {
            LOG_ERROR("Metal library compilation failed: %s",
                      error ? [[error localizedDescription] UTF8String] : "unknown error");
            return NULL;
        }

        NSString* fnName = [NSString stringWithUTF8String:function_name];
        id<MTLFunction> function = [library newFunctionWithName:fnName];
        if (!function) {
            LOG_ERROR("Metal function '%s' not found in compiled library", function_name);
            return NULL;
        }

        error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            LOG_ERROR("Metal pipeline creation failed: %s",
                      error ? [[error localizedDescription] UTF8String] : "unknown error");
            return NULL;
        }

        CMLMetalKernel* kernel = (CMLMetalKernel*)calloc(1, sizeof(CMLMetalKernel));
        if (!kernel) {
            LOG_ERROR("Failed to allocate CMLMetalKernel");
            return NULL;
        }

        kernel->pipeline = (__bridge_retained void*)pipeline;
        kernel->library  = (__bridge_retained void*)library;
        kernel->function = (__bridge_retained void*)function;
        strncpy(kernel->name, function_name, sizeof(kernel->name) - 1);
        kernel->name[sizeof(kernel->name) - 1] = '\0';

        LOG_DEBUG("Compiled Metal kernel: %s", function_name);
        return kernel;
    }
}

void cml_metal_kernel_free(CMLMetalKernel* kernel) {
    if (!kernel) return;

    @autoreleasepool {
        if (kernel->pipeline) {
            CFRelease((CFTypeRef)kernel->pipeline);
            kernel->pipeline = NULL;
        }
        if (kernel->function) {
            CFRelease((CFTypeRef)kernel->function);
            kernel->function = NULL;
        }
        if (kernel->library) {
            CFRelease((CFTypeRef)kernel->library);
            kernel->library = NULL;
        }
    }

    free(kernel);
}


int cml_metal_encode_kernel(CMLMetalBackend* backend,
                             void* encoder_ptr,
                             CMLMetalKernel* kernel,
                             size_t grid[3], size_t block[3],
                             void** gpu_buffers, int num_gpu_buffers,
                             const void** bytes, const size_t* byte_sizes,
                             int num_bytes) {
    if (!backend || !backend->initialized || !kernel || !kernel->pipeline || !encoder_ptr)
        return -1;

    id<MTLComputeCommandEncoder> encoder =
        (__bridge id<MTLComputeCommandEncoder>)encoder_ptr;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)kernel->pipeline;

    [encoder setComputePipelineState:pipeline];

    NSUInteger idx = 0;
    for (int i = 0; i < num_gpu_buffers; i++, idx++) {
        id<MTLBuffer> mtlBuf = (__bridge id<MTLBuffer>)gpu_buffers[i];
        [encoder setBuffer:mtlBuf offset:0 atIndex:idx];
    }
    for (int i = 0; i < num_bytes; i++, idx++) {
        [encoder setBytes:bytes[i] length:byte_sizes[i] atIndex:idx];
    }

    MTLSize gridSize  = MTLSizeMake(grid[0],  grid[1],  grid[2]);
    MTLSize blockSize = MTLSizeMake(block[0], block[1], block[2]);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:blockSize];

    return 0;
}


int cml_metal_launch_kernel(CMLMetalBackend* backend,
                            CMLMetalKernel* kernel,
                            size_t grid[3],
                            size_t block[3],
                            void** buffers,
                            int num_buffers) {
    if (!backend || !backend->initialized || !kernel || !kernel->pipeline) {
        LOG_ERROR("Invalid arguments to cml_metal_launch_kernel");
        return -1;
    }

    @autoreleasepool {
        id<MTLCommandQueue> queue =
            (__bridge id<MTLCommandQueue>)backend->command_queue;
        id<MTLComputePipelineState> pipeline =
            (__bridge id<MTLComputePipelineState>)kernel->pipeline;

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        if (!cmdBuf) { LOG_ERROR("Failed to create Metal command buffer"); return -1; }

        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        if (!encoder) { LOG_ERROR("Failed to create Metal compute encoder"); return -1; }

        [encoder setComputePipelineState:pipeline];

        for (int i = 0; i < num_buffers; i++) {
            id<MTLBuffer> mtlBuf = (__bridge id<MTLBuffer>)buffers[i];
            [encoder setBuffer:mtlBuf offset:0 atIndex:(NSUInteger)i];
        }

        MTLSize gridSize  = MTLSizeMake(grid[0],  grid[1],  grid[2]);
        MTLSize blockSize = MTLSizeMake(block[0], block[1], block[2]);
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:blockSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if ([cmdBuf error]) {
            LOG_ERROR("Metal kernel execution failed: %s",
                      [[[cmdBuf error] localizedDescription] UTF8String]);
            return -1;
        }
    }

    return 0;
}

/* ── Memory ── */

void* cml_metal_alloc(CMLMetalBackend* backend, size_t size) {
    if (!backend || !backend->initialized || size == 0) return NULL;

    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)backend->device;

        id<MTLBuffer> buffer = [device newBufferWithLength:size
                                                   options:MTLResourceStorageModeShared];
        if (!buffer) {
            LOG_ERROR("Metal buffer allocation failed for %zu bytes", size);
            return NULL;
        }

        return (__bridge_retained void*)buffer;
    }
}

void cml_metal_free(CMLMetalBackend* backend, void* buffer) {
    (void)backend;
    if (!buffer) return;

    @autoreleasepool {
        CFRelease((CFTypeRef)buffer);
    }
}

int cml_metal_upload(CMLMetalBackend* backend,
                     void* dst_buffer,
                     const void* src_host,
                     size_t size) {
    if (!backend || !backend->initialized || !dst_buffer || !src_host || size == 0)
        return -1;

    @autoreleasepool {
        id<MTLBuffer> mtlBuf = (__bridge id<MTLBuffer>)dst_buffer;
        void* contents = [mtlBuf contents];
        if (!contents) { LOG_ERROR("Metal buffer contents pointer is NULL"); return -1; }
        memcpy(contents, src_host, size);
    }

    return 0;
}

int cml_metal_download(CMLMetalBackend* backend,
                       void* dst_host,
                       const void* src_buffer,
                       size_t size) {
    if (!backend || !backend->initialized || !dst_host || !src_buffer || size == 0)
        return -1;

    @autoreleasepool {
        id<MTLBuffer> mtlBuf = (__bridge id<MTLBuffer>)(void*)src_buffer;
        const void* contents = [mtlBuf contents];
        if (!contents) { LOG_ERROR("Metal buffer contents pointer is NULL"); return -1; }
        memcpy(dst_host, contents, size);
    }

    return 0;
}

static CMLMetalBufferEntry* mtl_find_cached_input(CMLMetalBackend* b,
                                                   void* data_ptr, size_t size) {
    for (int i = 0; i < b->buffer_count; i++) {
        CMLMetalBufferEntry* e = &b->buffers[i];
        if (e->is_input && e->data_ptr == data_ptr && e->size == size && e->valid)
            return e;
    }
    return NULL;
}

static CMLMetalBufferEntry* mtl_find_by_gpu(CMLMetalBackend* b, void* gpu_buf) {
    for (int i = 0; i < b->buffer_count; i++) {
        if (b->buffers[i].gpu_buf == gpu_buf)
            return &b->buffers[i];
    }
    return NULL;
}

static void mtl_evict_one(CMLMetalBackend* b) {
    for (int i = 0; i < b->buffer_count; i++) {
        if (!b->buffers[i].is_input) {
            if (b->buffers[i].gpu_buf) CFRelease((CFTypeRef)b->buffers[i].gpu_buf);
            /* Shift array down */
            for (int j = i; j < b->buffer_count - 1; j++)
                b->buffers[j] = b->buffers[j + 1];
            b->buffer_count--;
            return;
        }
    }
    if (b->buffer_count > 0) {
        if (b->buffers[0].gpu_buf) CFRelease((CFTypeRef)b->buffers[0].gpu_buf);
        for (int j = 0; j < b->buffer_count - 1; j++)
            b->buffers[j] = b->buffers[j + 1];
        b->buffer_count--;
    }
}

void* cml_metal_get_or_upload_buffer(CMLMetalBackend* backend,
                                      void* data_ptr, size_t size,
                                      bool is_input) {
    if (!backend || !backend->initialized || !data_ptr || size == 0) return NULL;

    @autoreleasepool {
        CMLMetalBufferEntry* cached = mtl_find_cached_input(backend, data_ptr, size);
        if (cached) return cached->gpu_buf;

        if (backend->buffer_count >= CML_MTL_MAX_BUFFERS)
            mtl_evict_one(backend);

        id<MTLDevice> device = (__bridge id<MTLDevice>)backend->device;
        id<MTLBuffer> buf = [device newBufferWithLength:size
                                               options:MTLResourceStorageModeShared];
        if (!buf) {
            LOG_ERROR("Metal: buffer alloc failed (%zu bytes)", size);
            return NULL;
        }
        memcpy([buf contents], data_ptr, size);

        void* retained_buf = (__bridge_retained void*)buf;

        CMLMetalBufferEntry* e = &backend->buffers[backend->buffer_count++];
        e->data_ptr = data_ptr;
        e->gpu_buf  = retained_buf;
        e->size     = size;
        e->valid    = true;
        e->is_input = is_input;

        return retained_buf;
    }
}

void* cml_metal_alloc_output_buffer(CMLMetalBackend* backend,
                                     void* tensor_key, size_t size) {
    if (!backend || !backend->initialized || size == 0) return NULL;

    @autoreleasepool {
        for (int i = 0; i < backend->buffer_count; i++) {
            CMLMetalBufferEntry* e = &backend->buffers[i];
            if (!e->is_input && !e->valid && e->gpu_buf && e->size == size) {
                e->data_ptr = tensor_key;
                e->valid    = true;
                return e->gpu_buf;
            }
        }

        if (backend->buffer_count >= CML_MTL_MAX_BUFFERS)
            mtl_evict_one(backend);

        id<MTLDevice> device = (__bridge id<MTLDevice>)backend->device;
        id<MTLBuffer> buf = [device newBufferWithLength:size
                                               options:MTLResourceStorageModeShared];
        if (!buf) {
            LOG_ERROR("Metal: output alloc failed (%zu bytes)", size);
            return NULL;
        }

        void* retained_buf = (__bridge_retained void*)buf;

        CMLMetalBufferEntry* e = &backend->buffers[backend->buffer_count++];
        e->data_ptr = tensor_key;
        e->gpu_buf  = retained_buf;
        e->size     = size;
        e->valid    = true;
        e->is_input = false;

        return retained_buf;
    }
}

void cml_metal_release_intermediate_buffers(CMLMetalBackend* backend) {
    if (!backend) return;
    for (int i = 0; i < backend->buffer_count; i++) {
        CMLMetalBufferEntry* e = &backend->buffers[i];
        if (e->is_input) {
            e->data_ptr = NULL; /* clear stale tensor pointer, keep buffer */
        } else {
            e->data_ptr = NULL;
            e->valid    = false;
        }
    }
}

int cml_metal_download_buffer(CMLMetalBackend* backend,
                               void* gpu_buf, void* dst, size_t size) {
    if (!backend || !backend->initialized || !gpu_buf || !dst || size == 0) return -1;

    @autoreleasepool {
        id<MTLBuffer> mtlBuf = (__bridge id<MTLBuffer>)gpu_buf;
        const void* contents = [mtlBuf contents];
        if (!contents) return -1;
        memcpy(dst, contents, size);
    }
    return 0;
}

#else

bool cml_metal_available(void) { return false; }
CMLMetalBackend* cml_metal_backend_create(void) { return NULL; }
int cml_metal_backend_init(CMLMetalBackend* backend) { (void)backend; return -1; }
void cml_metal_backend_free(CMLMetalBackend* backend) { (void)backend; }

CMLMetalKernel* cml_metal_compile_msl(CMLMetalBackend* backend,
                                       const char* msl_source,
                                       const char* function_name) {
    (void)backend; (void)msl_source; (void)function_name; return NULL;
}
void cml_metal_kernel_free(CMLMetalKernel* kernel) { (void)kernel; }

int cml_metal_encode_kernel(CMLMetalBackend* backend, void* encoder,
                             CMLMetalKernel* kernel,
                             size_t grid[3], size_t block[3],
                             void** gpu_buffers, int num_gpu_buffers,
                             const void** bytes, const size_t* byte_sizes,
                             int num_bytes) {
    (void)backend; (void)encoder; (void)kernel; (void)grid; (void)block;
    (void)gpu_buffers; (void)num_gpu_buffers;
    (void)bytes; (void)byte_sizes; (void)num_bytes;
    return -1;
}

int cml_metal_launch_kernel(CMLMetalBackend* backend, CMLMetalKernel* kernel,
                            size_t grid[3], size_t block[3],
                            void** buffers, int num_buffers) {
    (void)backend; (void)kernel; (void)grid; (void)block;
    (void)buffers; (void)num_buffers; return -1;
}

void* cml_metal_alloc(CMLMetalBackend* backend, size_t size) {
    (void)backend; (void)size; return NULL;
}
void cml_metal_free(CMLMetalBackend* backend, void* buffer) {
    (void)backend; (void)buffer;
}
int cml_metal_upload(CMLMetalBackend* backend, void* dst_buffer,
                     const void* src_host, size_t size) {
    (void)backend; (void)dst_buffer; (void)src_host; (void)size; return -1;
}
int cml_metal_download(CMLMetalBackend* backend, void* dst_host,
                       const void* src_buffer, size_t size) {
    (void)backend; (void)dst_host; (void)src_buffer; (void)size; return -1;
}

void* cml_metal_get_or_upload_buffer(CMLMetalBackend* backend,
                                      void* data_ptr, size_t size, bool is_input) {
    (void)backend; (void)data_ptr; (void)size; (void)is_input; return NULL;
}
void* cml_metal_alloc_output_buffer(CMLMetalBackend* backend,
                                     void* tensor_key, size_t size) {
    (void)backend; (void)tensor_key; (void)size; return NULL;
}
void cml_metal_release_intermediate_buffers(CMLMetalBackend* backend) { (void)backend; }
int cml_metal_download_buffer(CMLMetalBackend* backend, void* gpu_buf,
                               void* dst, size_t size) {
    (void)backend; (void)gpu_buf; (void)dst; (void)size; return -1;
}

#endif /* CML_HAS_METAL */
