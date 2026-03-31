/*
 * Metal Performance Shaders (MPS) backend for CML on macOS.
 * Uses Apple's optimized GPU kernels via MPS for high performance matmul.
 */

#include "ops/ir/gpu/metal_backend.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include "core/logging.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <CoreFoundation/CoreFoundation.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static bool g_mps_initialized = false;

static size_t row_bytes(int cols) {
    return cols * sizeof(float);
}

bool cml_mps_init(void) {
    if (g_mps_initialized) return true;
    
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            LOG_ERROR("MPS: No Metal device available");
            return false;
        }
        
        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            LOG_ERROR("MPS: Failed to create command queue");
            return false;
        }
        
        g_mps_initialized = true;
        LOG_INFO("MPS initialized on device: %s", [[g_device name] UTF8String]);
    }
    return true;
}

void cml_mps_cleanup(void) {
    g_mps_initialized = false;
    g_device = nil;
    g_queue = nil;
}

bool cml_mps_available(void) {
    return cml_mps_init();
}

int cml_mps_matmul(const float* A, const float* B, float* C,
                    int M, int N, int K) {
    if (!cml_mps_init()) return -1;
    
    @autoreleasepool {
        size_t ARowBytes = row_bytes(K);
        size_t BRowBytes = row_bytes(N);
        size_t CRowBytes = row_bytes(N);
        
        size_t sizeA = M * ARowBytes;
        size_t sizeB = K * BRowBytes;
        size_t sizeC = M * CRowBytes;
        
        id<MTLBuffer> bufA = [g_device newBufferWithBytes:A length:sizeA options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [g_device newBufferWithBytes:B length:sizeB options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [g_device newBufferWithLength:sizeC options:MTLResourceStorageModeShared];
        
        if (!bufA || !bufB || !bufC) {
            LOG_ERROR("MPS: Buffer allocation failed");
            return -1;
        }
        
        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithDimensions:M
                                                                                columns:K
                                                                               rowBytes:ARowBytes
                                                                               dataType:MPSDataTypeFloat32];
        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        
        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithDimensions:K
                                                                                columns:N
                                                                               rowBytes:BRowBytes
                                                                               dataType:MPSDataTypeFloat32];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
        
        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithDimensions:M
                                                                                columns:N
                                                                               rowBytes:CRowBytes
                                                                               dataType:MPSDataTypeFloat32];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];
        
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        
        MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc]
                                         initWithDevice:g_device
                                         transposeLeft:NO
                                         transposeRight:NO
                                         resultRows:M
                                         resultColumns:N
                                         interiorColumns:K
                                         alpha:1.0
                                         beta:0.0];
        
        [mul encodeToCommandBuffer:cmd
                        leftMatrix:matA
                       rightMatrix:matB
                      resultMatrix:matC];
        [cmd commit];
        [cmd waitUntilCompleted];
        
        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            LOG_ERROR("MPS: Command buffer failed");
            return -1;
        }
        
        memcpy(C, [bufC contents], sizeC);
    }
    
    return 0;
}

int cml_mps_matmul_fused_bias_relu(const float* A, const float* B, const float* bias,
                                    float* C, int M, int N, int K) {
    if (!cml_mps_init()) return -1;
    
    @autoreleasepool {
        size_t ARowBytes = row_bytes(K);
        size_t BRowBytes = row_bytes(N);
        size_t CRowBytes = row_bytes(N);
        
        size_t sizeA = M * ARowBytes;
        size_t sizeB = K * BRowBytes;
        size_t sizeC = M * CRowBytes;
        
        id<MTLBuffer> bufA = [g_device newBufferWithBytes:A length:sizeA options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [g_device newBufferWithBytes:B length:sizeB options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [g_device newBufferWithLength:sizeC options:MTLResourceStorageModeShared];
        
        if (!bufA || !bufB || !bufC) {
            LOG_ERROR("MPS: Buffer allocation failed");
            return -1;
        }
        
        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithDimensions:M
                                                                                columns:K
                                                                               rowBytes:ARowBytes
                                                                               dataType:MPSDataTypeFloat32];
        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        
        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithDimensions:K
                                                                                columns:N
                                                                               rowBytes:BRowBytes
                                                                               dataType:MPSDataTypeFloat32];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
        
        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithDimensions:M
                                                                                columns:N
                                                                               rowBytes:CRowBytes
                                                                               dataType:MPSDataTypeFloat32];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];
        
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        
        MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc]
                                         initWithDevice:g_device
                                         transposeLeft:NO
                                         transposeRight:NO
                                         resultRows:M
                                         resultColumns:N
                                         interiorColumns:K
                                         alpha:1.0
                                         beta:0.0];
        
        [mul encodeToCommandBuffer:cmd
                        leftMatrix:matA
                       rightMatrix:matB
                      resultMatrix:matC];
        [cmd commit];
        [cmd waitUntilCompleted];
        
        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            LOG_ERROR("MPS: Command buffer failed");
            return -1;
        }
        
        float* temp = (float*)[bufC contents];
        for (int i = 0; i < M * N; i++) {
            int row = i / N;
            float val = temp[i] + bias[row];
            C[i] = val > 0 ? val : 0;
        }
    }
    
    return 0;
}

int cml_mps_relu(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
    return 0;
}

int cml_mps_add(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
    return 0;
}
