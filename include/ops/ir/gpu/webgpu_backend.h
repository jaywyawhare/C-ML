/*
 * WebGPU backend via wgpu-native (dynamic loading).
 * WGSL codegen + wgpu-native C bindings. Cross-platform via dlopen.
 */

#ifndef CML_OPS_IR_GPU_WEBGPU_BACKEND_H
#define CML_OPS_IR_GPU_WEBGPU_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct IRNode;
struct CMLGraph;
typedef struct CMLGraph* CMLGraph_t;

typedef struct CMLWebGPUBackend {
    void* instance;     /* WGPUInstance */
    void* adapter;      /* WGPUAdapter */
    void* device;       /* WGPUDevice */
    void* queue;        /* WGPUQueue */
    void* lib_handle;   /* dlopen handle */
    char device_name[128];
    bool initialized;

    /* dlsym'd function pointers */
    void* fn_create_instance;
    void* fn_instance_request_adapter;
    void* fn_adapter_request_device;
    void* fn_device_get_queue;
    void* fn_device_create_shader_module;
    void* fn_device_create_compute_pipeline;
    void* fn_device_create_bind_group_layout;
    void* fn_device_create_bind_group;
    void* fn_device_create_pipeline_layout;
    void* fn_device_create_buffer;
    void* fn_device_create_command_encoder;
    void* fn_command_encoder_begin_compute_pass;
    void* fn_compute_pass_set_pipeline;
    void* fn_compute_pass_set_bind_group;
    void* fn_compute_pass_dispatch_workgroups;
    void* fn_compute_pass_end;
    void* fn_command_encoder_finish;
    void* fn_queue_submit;
    void* fn_queue_write_buffer;
    void* fn_buffer_map_async;
    void* fn_buffer_get_mapped_range;
    void* fn_buffer_unmap;
    void* fn_buffer_destroy;
    void* fn_device_poll;
    void* fn_instance_release;
} CMLWebGPUBackend;

typedef struct CMLWebGPUKernel {
    void* pipeline;         /* WGPUComputePipeline */
    void* bind_group_layout; /* WGPUBindGroupLayout */
    void* pipeline_layout;  /* WGPUPipelineLayout */
    void* shader_module;    /* WGPUShaderModule */
    char name[64];
} CMLWebGPUKernel;

bool cml_webgpu_available(void);

CMLWebGPUBackend* cml_webgpu_backend_create(void);
int cml_webgpu_backend_init(CMLWebGPUBackend* backend);
void cml_webgpu_backend_free(CMLWebGPUBackend* backend);

CMLWebGPUKernel* cml_webgpu_compile_wgsl(CMLWebGPUBackend* backend,
                                           const char* wgsl_source,
                                           const char* entry_point);
void cml_webgpu_kernel_free(CMLWebGPUKernel* kernel);

int cml_webgpu_launch_kernel(CMLWebGPUBackend* backend, CMLWebGPUKernel* kernel,
                             size_t workgroup_count[3],
                             void** buffers, size_t* buffer_sizes, int num_buffers);

void* cml_webgpu_alloc(CMLWebGPUBackend* backend, size_t size);
void cml_webgpu_free(CMLWebGPUBackend* backend, void* buffer);
int cml_webgpu_upload(CMLWebGPUBackend* backend, void* dst_buffer,
                      const void* src_host, size_t size);
int cml_webgpu_download(CMLWebGPUBackend* backend, void* dst_host,
                        void* src_buffer, size_t size);

char* cml_wgsl_generate(struct IRNode* node);

int cml_webgpu_execute_graph(CMLWebGPUBackend* backend, CMLGraph_t graph);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_GPU_WEBGPU_BACKEND_H */
