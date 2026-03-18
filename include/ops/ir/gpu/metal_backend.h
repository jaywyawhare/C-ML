/*
 * Metal GPU backend for macOS.
 * MSL codegen + Metal API (MTLDevice, MTLCommandQueue, MTLComputePipelineState).
 */

#ifndef CML_OPS_IR_GPU_METAL_BACKEND_H
#define CML_OPS_IR_GPU_METAL_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct IRNode;
struct CMLGraph;
typedef struct CMLGraph* CMLGraph_t;

typedef struct CMLMetalBackend {
    void* device;           /* id<MTLDevice> */
    void* command_queue;    /* id<MTLCommandQueue> */
    char device_name[128];
    uint64_t total_memory;
    bool initialized;
} CMLMetalBackend;

typedef struct CMLMetalKernel {
    void* pipeline;    /* id<MTLComputePipelineState> */
    void* library;     /* id<MTLLibrary> */
    void* function;    /* id<MTLFunction> */
    char name[64];
} CMLMetalKernel;

bool cml_metal_available(void);

CMLMetalBackend* cml_metal_backend_create(void);
int cml_metal_backend_init(CMLMetalBackend* backend);
void cml_metal_backend_free(CMLMetalBackend* backend);

CMLMetalKernel* cml_metal_compile_msl(CMLMetalBackend* backend, const char* msl_source,
                                       const char* function_name);
void cml_metal_kernel_free(CMLMetalKernel* kernel);

int cml_metal_launch_kernel(CMLMetalBackend* backend, CMLMetalKernel* kernel,
                            size_t grid[3], size_t block[3],
                            void** buffers, int num_buffers);

void* cml_metal_alloc(CMLMetalBackend* backend, size_t size);
void cml_metal_free(CMLMetalBackend* backend, void* buffer);
int cml_metal_upload(CMLMetalBackend* backend, void* dst_buffer,
                     const void* src_host, size_t size);
int cml_metal_download(CMLMetalBackend* backend, void* dst_host,
                       const void* src_buffer, size_t size);

char* cml_metal_generate_msl(struct IRNode* node);

int cml_metal_execute_graph(CMLMetalBackend* backend, CMLGraph_t graph);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_GPU_METAL_BACKEND_H */
