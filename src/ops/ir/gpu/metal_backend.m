/**
 * @file metal_backend.m
 * @brief Metal GPU backend implementation for macOS/iOS
 *
 * Objective-C implementation using the Metal framework.
 * Entirely guarded by CML_HAS_METAL -- on non-Apple platforms every
 * public function compiles to a safe stub that returns false/NULL/-1.
 */

#include "ops/ir/gpu/metal_backend.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ========================================================================
 * CML_HAS_METAL -- full implementation
 * ======================================================================== */
#ifdef CML_HAS_METAL

#import <Metal/Metal.h>

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
        /* Create the system default Metal device */
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            LOG_ERROR("MTLCreateSystemDefaultDevice() returned nil");
            return -1;
        }

        /* Retain the device so it persists beyond the autorelease pool */
        CFRetain((__bridge CFTypeRef)device);
        backend->device = (__bridge void*)device;

        /* Create a command queue */
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            LOG_ERROR("Failed to create MTLCommandQueue");
            CFRelease((__bridge CFTypeRef)device);
            backend->device = NULL;
            return -1;
        }
        /* newCommandQueue already returns a +1 object; store as void* */
        backend->command_queue = (__bridge_retained void*)queue;

        /* Query device name */
        const char* name = [[device name] UTF8String];
        if (name) {
            strncpy(backend->device_name, name, sizeof(backend->device_name) - 1);
            backend->device_name[sizeof(backend->device_name) - 1] = '\0';
        } else {
            strncpy(backend->device_name, "Unknown Metal Device",
                    sizeof(backend->device_name) - 1);
        }

        /* Query recommended max working-set size as a proxy for total memory */
        if ([device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
            backend->total_memory = (uint64_t)[device recommendedMaxWorkingSetSize];
        } else {
            backend->total_memory = 0;
        }

        backend->initialized = true;

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

        /* Compile the MSL source into a library */
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

        /* Retrieve the named function */
        NSString* fnName = [NSString stringWithUTF8String:function_name];
        id<MTLFunction> function = [library newFunctionWithName:fnName];
        if (!function) {
            LOG_ERROR("Metal function '%s' not found in compiled library", function_name);
            return NULL;
        }

        /* Create a compute pipeline state from the function */
        error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            LOG_ERROR("Metal pipeline creation failed: %s",
                      error ? [[error localizedDescription] UTF8String] : "unknown error");
            return NULL;
        }

        /* Allocate and fill the kernel structure */
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

/* ── Execution ── */

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

        /* Create a command buffer */
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        if (!cmdBuf) {
            LOG_ERROR("Failed to create Metal command buffer");
            return -1;
        }

        /* Create a compute command encoder */
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        if (!encoder) {
            LOG_ERROR("Failed to create Metal compute encoder");
            return -1;
        }

        [encoder setComputePipelineState:pipeline];

        /* Bind all buffers */
        for (int i = 0; i < num_buffers; i++) {
            id<MTLBuffer> mtlBuf = (__bridge id<MTLBuffer>)buffers[i];
            [encoder setBuffer:mtlBuf offset:0 atIndex:(NSUInteger)i];
        }

        /* Dispatch thread groups */
        MTLSize gridSize  = MTLSizeMake(grid[0],  grid[1],  grid[2]);
        MTLSize blockSize = MTLSizeMake(block[0], block[1], block[2]);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:blockSize];
        [encoder endEncoding];

        /* Commit and wait for completion */
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

        /* Transfer ownership to caller via __bridge_retained */
        return (__bridge_retained void*)buffer;
    }
}

void cml_metal_free(CMLMetalBackend* backend, void* buffer) {
    (void)backend;
    if (!buffer) return;

    @autoreleasepool {
        /* Release the retained Metal buffer */
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
        if (!contents) {
            LOG_ERROR("Metal buffer contents pointer is NULL");
            return -1;
        }
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
        if (!contents) {
            LOG_ERROR("Metal buffer contents pointer is NULL");
            return -1;
        }
        memcpy(dst_host, contents, size);
    }

    return 0;
}

/* ========================================================================
 * Stubs -- when CML_HAS_METAL is NOT defined
 * ======================================================================== */
#else /* !CML_HAS_METAL */

bool cml_metal_available(void) {
    return false;
}

CMLMetalBackend* cml_metal_backend_create(void) {
    return NULL;
}

int cml_metal_backend_init(CMLMetalBackend* backend) {
    (void)backend;
    return -1;
}

void cml_metal_backend_free(CMLMetalBackend* backend) {
    (void)backend;
}

CMLMetalKernel* cml_metal_compile_msl(CMLMetalBackend* backend,
                                       const char* msl_source,
                                       const char* function_name) {
    (void)backend;
    (void)msl_source;
    (void)function_name;
    return NULL;
}

void cml_metal_kernel_free(CMLMetalKernel* kernel) {
    (void)kernel;
}

int cml_metal_launch_kernel(CMLMetalBackend* backend,
                            CMLMetalKernel* kernel,
                            size_t grid[3],
                            size_t block[3],
                            void** buffers,
                            int num_buffers) {
    (void)backend;
    (void)kernel;
    (void)grid;
    (void)block;
    (void)buffers;
    (void)num_buffers;
    return -1;
}

void* cml_metal_alloc(CMLMetalBackend* backend, size_t size) {
    (void)backend;
    (void)size;
    return NULL;
}

void cml_metal_free(CMLMetalBackend* backend, void* buffer) {
    (void)backend;
    (void)buffer;
}

int cml_metal_upload(CMLMetalBackend* backend,
                     void* dst_buffer,
                     const void* src_host,
                     size_t size) {
    (void)backend;
    (void)dst_buffer;
    (void)src_host;
    (void)size;
    return -1;
}

int cml_metal_download(CMLMetalBackend* backend,
                       void* dst_host,
                       const void* src_buffer,
                       size_t size) {
    (void)backend;
    (void)dst_host;
    (void)src_buffer;
    (void)size;
    return -1;
}

#endif /* CML_HAS_METAL */
