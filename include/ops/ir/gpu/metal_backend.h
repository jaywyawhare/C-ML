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

#define CML_MTL_MAX_BUFFERS 256

typedef struct CMLMetalBufferEntry {
    void*   data_ptr;
    void*   gpu_buf;
    size_t  size;
    bool    valid;
    bool    is_input;
} CMLMetalBufferEntry;

#define CML_MTL_DYN_CACHE_SIZE 48

typedef struct CMLMetalDynKernelEntry {
    int   uop_type;
    void* pso;
} CMLMetalDynKernelEntry;

typedef struct CMLMetalBackend {
    void* device;           /* id<MTLDevice> */
    void* command_queue;    /* id<MTLCommandQueue> */
    char device_name[128];
    uint64_t total_memory;
    bool initialized;

    void* k_fill;
    void* k_relu;
    void* k_sigmoid;
    void* k_tanh_k;
    void* k_sum_reduce;
    void* k_max_reduce;
    void* k_matmul_opt;
    void* k_matmul_fused_bias_relu;

    CMLMetalBufferEntry buffers[CML_MTL_MAX_BUFFERS];
    int buffer_count;

    CMLMetalDynKernelEntry dyn_cache[CML_MTL_DYN_CACHE_SIZE];
    int dyn_cache_count;
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

int cml_metal_encode_kernel(CMLMetalBackend* backend,
                             void* encoder,
                             CMLMetalKernel* kernel,
                             size_t grid[3], size_t block[3],
                             void** gpu_buffers, int num_gpu_buffers,
                             const void** bytes,  const size_t* byte_sizes,
                             int num_bytes);

int cml_metal_launch_kernel(CMLMetalBackend* backend, CMLMetalKernel* kernel,
                            size_t grid[3], size_t block[3],
                            void** buffers, int num_buffers);

void* cml_metal_alloc(CMLMetalBackend* backend, size_t size);
void cml_metal_free(CMLMetalBackend* backend, void* buffer);
int cml_metal_upload(CMLMetalBackend* backend, void* dst_buffer,
                     const void* src_host, size_t size);
int cml_metal_download(CMLMetalBackend* backend, void* dst_host,
                       const void* src_buffer, size_t size);

void* cml_metal_get_or_upload_buffer(CMLMetalBackend* backend,
                                      void* data_ptr, size_t size,
                                      bool is_input);

void* cml_metal_alloc_output_buffer(CMLMetalBackend* backend,
                                     void* tensor_key, size_t size);

void cml_metal_release_intermediate_buffers(CMLMetalBackend* backend);

int cml_metal_download_buffer(CMLMetalBackend* backend,
                               void* gpu_buf, void* dst, size_t size);

char* cml_metal_generate_msl(struct IRNode* node);

int cml_metal_execute_graph(CMLMetalBackend* backend, CMLGraph_t graph);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_GPU_METAL_BACKEND_H */
