#ifndef CML_GPU_ROCM_BACKEND_H
#define CML_GPU_ROCM_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Tensor;
typedef struct Tensor Tensor;

// HIP types (avoid requiring hip_runtime.h)
typedef int hipDevice_t;
typedef void* hipCtx_t;
typedef void* hipModule_t;
typedef void* hipFunction_t;
typedef void* hipStream_t;
typedef void* hipDeviceptr_t;
typedef int hipError_t;

typedef struct CMLROCmBackend {
    void* hip_lib;    // libamdhip64.so handle
    void* hiprtc_lib; // libhiprtc.so handle

    hipDevice_t device;
    hipCtx_t context;
    hipStream_t stream;
    bool initialized;

    char device_name[256];
    size_t total_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;

    hipError_t (*hipInit)(unsigned int flags);
    hipError_t (*hipGetDeviceCount)(int* count);
    hipError_t (*hipSetDevice)(int deviceId);
    hipError_t (*hipGetDeviceProperties)(void* prop, int deviceId);
    hipError_t (*hipModuleLoad)(hipModule_t* module, const char* fname);
    hipError_t (*hipModuleLoadData)(hipModule_t* module, const void* image);
    hipError_t (*hipModuleUnload)(hipModule_t module);
    hipError_t (*hipModuleGetFunction)(hipFunction_t* function, hipModule_t module,
                                       const char* kname);
    hipError_t (*hipModuleLaunchKernel)(hipFunction_t f, unsigned int gridDimX,
                                        unsigned int gridDimY, unsigned int gridDimZ,
                                        unsigned int blockDimX, unsigned int blockDimY,
                                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                                        hipStream_t stream, void** kernelParams, void** extra);
    hipError_t (*hipMalloc)(void** ptr, size_t size);
    hipError_t (*hipFree)(void* ptr);
    hipError_t (*hipMemcpy)(void* dst, const void* src, size_t sizeBytes, int kind);
    hipError_t (*hipStreamCreate)(hipStream_t* stream);
    hipError_t (*hipStreamDestroy)(hipStream_t stream);
    hipError_t (*hipStreamSynchronize)(hipStream_t stream);
    hipError_t (*hipDeviceSynchronize)(void);
} CMLROCmBackend;

typedef struct CMLROCmKernel {
    hipModule_t module;
    hipFunction_t function;
    char* kernel_name;
    int grid_dim[3];
    int block_dim[3];
    size_t shared_mem_size;
} CMLROCmKernel;

bool cml_rocm_available(void);
CMLROCmBackend* cml_rocm_backend_create(void);
int cml_rocm_backend_init(CMLROCmBackend* backend, int device_ordinal);
void cml_rocm_backend_free(CMLROCmBackend* backend);

CMLROCmKernel* cml_rocm_compile_hsaco(CMLROCmBackend* backend, const char* hsaco_code,
                                      const char* kernel_name);
void cml_rocm_kernel_free(CMLROCmBackend* backend, CMLROCmKernel* kernel);
int cml_rocm_launch_kernel(CMLROCmBackend* backend, CMLROCmKernel* kernel, void** args,
                           int num_args);
int cml_rocm_synchronize(CMLROCmBackend* backend);

hipDeviceptr_t cml_rocm_malloc(CMLROCmBackend* backend, size_t size);
void cml_rocm_free(CMLROCmBackend* backend, hipDeviceptr_t ptr);
int cml_rocm_memcpy_h2d(CMLROCmBackend* backend, hipDeviceptr_t dst, const void* src, size_t size);
int cml_rocm_memcpy_d2h(CMLROCmBackend* backend, void* dst, hipDeviceptr_t src, size_t size);

int cml_rocm_upload_tensor(CMLROCmBackend* backend, Tensor* tensor);
int cml_rocm_download_tensor(CMLROCmBackend* backend, Tensor* tensor);

#ifdef __cplusplus
}
#endif

#endif // CML_GPU_ROCM_BACKEND_H
