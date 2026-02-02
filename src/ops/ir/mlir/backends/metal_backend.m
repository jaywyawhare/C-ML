/**
 * @file metal_backend.m
 * @brief Apple Metal backend implementation using Objective-C
 */

#include "ops/ir/mlir/backends/metal_backend.h"
#include "tensor/tensor.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// Metal backend structure
struct CMLMetalBackend {
    bool initialized;
    char device_name[256];
    size_t total_memory;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> current_command_buffer;
};

bool cml_metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
}

CMLMetalBackend* cml_metal_backend_create(void) {
    if (!cml_metal_available()) {
        LOG_ERROR("Metal is not available on this platform");
        return NULL;
    }

    CMLMetalBackend* backend = calloc(1, sizeof(CMLMetalBackend));
    if (!backend) {
        LOG_ERROR("Failed to allocate Metal backend");
        return NULL;
    }
    return backend;
}

int cml_metal_backend_init(CMLMetalBackend* backend) {
    if (!backend)
        return -1;
    if (backend->initialized)
        return 0;

    @autoreleasepool {
        // Get the default Metal device
        backend->device = MTLCreateSystemDefaultDevice();
        if (!backend->device) {
            LOG_ERROR("Failed to create Metal device");
            return -1;
        }

        // Create command queue
        backend->queue = [backend->device newCommandQueue];
        if (!backend->queue) {
            LOG_ERROR("Failed to create Metal command queue");
            backend->device = nil;
            return -1;
        }

        // Get device info
        const char* name = [[backend->device name] UTF8String];
        if (name) {
            strncpy(backend->device_name, name, sizeof(backend->device_name) - 1);
        }

        // Get recommended working set size as memory estimate
        backend->total_memory = [backend->device recommendedMaxWorkingSetSize];

        backend->initialized = true;
        LOG_INFO("Metal backend initialized: %s (%.2f GB recommended)", backend->device_name,
                 (double)backend->total_memory / (1024.0 * 1024.0 * 1024.0));

        return 0;
    }
}

void cml_metal_backend_free(CMLMetalBackend* backend) {
    if (!backend)
        return;

    if (backend->initialized) {
        @autoreleasepool {
            backend->queue  = nil;
            backend->device = nil;
        }
    }

    free(backend);
}

CMLMetalKernel* cml_metal_compile_source(CMLMetalBackend* backend, const char* msl_code,
                                         const char* kernel_name) {
    if (!backend || !backend->initialized || !msl_code || !kernel_name) {
        return NULL;
    }

    @autoreleasepool {
        NSError* error = nil;

        // Create Metal library from source
        NSString* source       = [NSString stringWithUTF8String:msl_code];
        id<MTLLibrary> library = [backend->device newLibraryWithSource:source
                                                               options:nil
                                                                 error:&error];
        if (!library) {
            LOG_ERROR("Failed to compile Metal shader: %s",
                      [[error localizedDescription] UTF8String]);
            return NULL;
        }

        // Get the kernel function
        NSString* functionName   = [NSString stringWithUTF8String:kernel_name];
        id<MTLFunction> function = [library newFunctionWithName:functionName];
        if (!function) {
            LOG_ERROR("Failed to find function '%s' in Metal library", kernel_name);
            return NULL;
        }

        // Create compute pipeline
        id<MTLComputePipelineState> pipeline =
            [backend->device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            LOG_ERROR("Failed to create compute pipeline: %s",
                      [[error localizedDescription] UTF8String]);
            return NULL;
        }

        // Create kernel structure
        CMLMetalKernel* kernel = calloc(1, sizeof(CMLMetalKernel));
        if (!kernel)
            return NULL;

        // Store Objective-C objects (need to retain them)
        kernel->pipeline    = (__bridge_retained void*)pipeline;
        kernel->function    = (__bridge_retained void*)function;
        kernel->kernel_name = strdup(kernel_name);

        // Set default thread group size
        NSUInteger maxThreads        = [pipeline maxTotalThreadsPerThreadgroup];
        kernel->threads_per_group[0] = (int)MIN(maxThreads, 256);
        kernel->threads_per_group[1] = 1;
        kernel->threads_per_group[2] = 1;

        kernel->grid_dim[0] = 1;
        kernel->grid_dim[1] = 1;
        kernel->grid_dim[2] = 1;

        LOG_INFO("Compiled Metal kernel: %s (max threads: %lu)", kernel_name,
                 (unsigned long)maxThreads);

        return kernel;
    }
}

void cml_metal_kernel_free(CMLMetalBackend* backend, CMLMetalKernel* kernel) {
    (void)backend;
    if (!kernel)
        return;

    @autoreleasepool {
        // Release Objective-C objects
        if (kernel->pipeline) {
            id obj = (__bridge_transfer id)kernel->pipeline;
            obj    = nil;
        }
        if (kernel->function) {
            id obj = (__bridge_transfer id)kernel->function;
            obj    = nil;
        }
    }

    free(kernel->kernel_name);
    free(kernel);
}

int cml_metal_launch_kernel(CMLMetalBackend* backend, CMLMetalKernel* kernel, void** args,
                            int num_args) {
    if (!backend || !backend->initialized || !kernel || !args)
        return -1;

    @autoreleasepool {
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [backend->queue commandBuffer];
        if (!commandBuffer) {
            LOG_ERROR("Failed to create command buffer");
            return -1;
        }

        // Create compute encoder
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            LOG_ERROR("Failed to create compute encoder");
            return -1;
        }

        // Set pipeline state
        id<MTLComputePipelineState> pipeline =
            (__bridge id<MTLComputePipelineState>)kernel->pipeline;
        [encoder setComputePipelineState:pipeline];

        // Set buffer arguments
        for (int i = 0; i < num_args; i++) {
            if (args[i]) {
                // Assuming args[i] is an MTLBuffer
                id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)args[i];
                [encoder setBuffer:buffer offset:0 atIndex:i];
            }
        }

        // Calculate grid and thread group sizes
        MTLSize threadgroupSize =
            MTLSizeMake(kernel->threads_per_group[0], kernel->threads_per_group[1],
                        kernel->threads_per_group[2]);

        MTLSize gridSize = MTLSizeMake(kernel->grid_dim[0] * kernel->threads_per_group[0],
                                       kernel->grid_dim[1] * kernel->threads_per_group[1],
                                       kernel->grid_dim[2] * kernel->threads_per_group[2]);

        // Dispatch threads
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        // Store current command buffer for synchronization
        backend->current_command_buffer = commandBuffer;

        // Commit command buffer
        [commandBuffer commit];

        return 0;
    }
}

int cml_metal_synchronize(CMLMetalBackend* backend) {
    if (!backend || !backend->initialized)
        return -1;

    @autoreleasepool {
        if (backend->current_command_buffer) {
            [backend->current_command_buffer waitUntilCompleted];
            backend->current_command_buffer = nil;
        }
        return 0;
    }
}

void* cml_metal_malloc(CMLMetalBackend* backend, size_t size) {
    if (!backend || !backend->initialized || size == 0)
        return NULL;

    @autoreleasepool {
        // Create a Metal buffer with shared storage mode (unified memory)
        id<MTLBuffer> buffer = [backend->device newBufferWithLength:size
                                                            options:MTLResourceStorageModeShared];
        if (!buffer) {
            LOG_ERROR("Failed to allocate Metal buffer of size %zu", size);
            return NULL;
        }

        // Return the buffer contents pointer (accessible from CPU)
        // Note: We need to keep the buffer alive, so we return the MTLBuffer handle
        return (__bridge_retained void*)buffer;
    }
}

void cml_metal_free(CMLMetalBackend* backend, void* ptr) {
    (void)backend;
    if (!ptr)
        return;

    @autoreleasepool {
        // Release the MTLBuffer
        id obj = (__bridge_transfer id)ptr;
        obj    = nil;
    }
}

// Helper to create buffer from data
static id<MTLBuffer> create_buffer_with_data(CMLMetalBackend* backend, const void* data,
                                             size_t size) {
    @autoreleasepool {
        id<MTLBuffer> buffer = [backend->device newBufferWithBytes:data
                                                            length:size
                                                           options:MTLResourceStorageModeShared];
        return buffer;
    }
}

int cml_metal_upload_tensor(CMLMetalBackend* backend, Tensor* tensor) {
    if (!backend || !tensor || !tensor->data)
        return -1;

    @autoreleasepool {
        size_t size = tensor->numel * sizeof(float);

        if (!tensor->buffer_handle) {
            // Create a new Metal buffer with the tensor data
            id<MTLBuffer> buffer = create_buffer_with_data(backend, tensor->data, size);
            if (!buffer)
                return -1;

            tensor->buffer_handle = (__bridge_retained void*)buffer;
        } else {
            // Update existing buffer
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)tensor->buffer_handle;
            void* contents       = [buffer contents];
            memcpy(contents, tensor->data, size);
        }

        return 0;
    }
}

int cml_metal_download_tensor(CMLMetalBackend* backend, Tensor* tensor) {
    if (!backend || !tensor || !tensor->buffer_handle)
        return -1;

    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)tensor->buffer_handle;

        if (!tensor->data) {
            tensor->data = malloc(tensor->numel * sizeof(float));
            if (!tensor->data)
                return -1;
            tensor->owns_data = true;
        }

        size_t size    = tensor->numel * sizeof(float);
        void* contents = [buffer contents];
        memcpy(tensor->data, contents, size);

        return 0;
    }
}

#else

// Non-Apple platform stubs

bool cml_metal_available(void) { return false; }

CMLMetalBackend* cml_metal_backend_create(void) {
    LOG_ERROR("Metal is not available on this platform");
    return NULL;
}

int cml_metal_backend_init(CMLMetalBackend* backend) {
    (void)backend;
    return -1;
}

void cml_metal_backend_free(CMLMetalBackend* backend) { (void)backend; }

CMLMetalKernel* cml_metal_compile_source(CMLMetalBackend* backend, const char* msl_code,
                                         const char* kernel_name) {
    (void)backend;
    (void)msl_code;
    (void)kernel_name;
    return NULL;
}

void cml_metal_kernel_free(CMLMetalBackend* backend, CMLMetalKernel* kernel) {
    (void)backend;
    (void)kernel;
}

int cml_metal_launch_kernel(CMLMetalBackend* backend, CMLMetalKernel* kernel, void** args,
                            int num_args) {
    (void)backend;
    (void)kernel;
    (void)args;
    (void)num_args;
    return -1;
}

int cml_metal_synchronize(CMLMetalBackend* backend) {
    (void)backend;
    return -1;
}

void* cml_metal_malloc(CMLMetalBackend* backend, size_t size) {
    (void)backend;
    (void)size;
    return NULL;
}

void cml_metal_free(CMLMetalBackend* backend, void* ptr) {
    (void)backend;
    (void)ptr;
}

int cml_metal_upload_tensor(CMLMetalBackend* backend, Tensor* tensor) {
    (void)backend;
    (void)tensor;
    return -1;
}

int cml_metal_download_tensor(CMLMetalBackend* backend, Tensor* tensor) {
    (void)backend;
    (void)tensor;
    return -1;
}

#endif // __APPLE__
