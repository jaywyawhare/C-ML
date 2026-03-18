/*
 * CUDA backend for JIT kernel execution via dynamic loading of the CUDA driver API.
 */

#ifndef CML_GPU_CUDA_BACKEND_H
#define CML_GPU_CUDA_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Tensor;
typedef struct Tensor Tensor;

// CUDA types (avoid requiring cuda.h)
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef void* CUdeviceptr;
typedef void* CUevent;
typedef int CUresult;

typedef struct CMLCUDABackend {
    void* cuda_lib;  // libcuda.so handle
    void* nvrtc_lib; // libnvrtc.so handle

    CUdevice device;
    CUcontext context;
    CUstream stream;
    bool initialized;

    char device_name[256];
    size_t total_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;

    // CUDA driver API function pointers
    CUresult (*cuInit)(unsigned int flags);
    CUresult (*cuDeviceGet)(CUdevice* device, int ordinal);
    CUresult (*cuDeviceGetCount)(int* count);
    CUresult (*cuDeviceGetName)(char* name, int len, CUdevice dev);
    CUresult (*cuDeviceTotalMem)(size_t* bytes, CUdevice dev);
    CUresult (*cuDeviceGetAttribute)(int* pi, int attrib, CUdevice dev);
    CUresult (*cuCtxCreate)(CUcontext* pctx, unsigned int flags, CUdevice dev);
    CUresult (*cuCtxDestroy)(CUcontext ctx);
    CUresult (*cuCtxSetCurrent)(CUcontext ctx);
    CUresult (*cuModuleLoadData)(CUmodule* module, const void* image);
    CUresult (*cuModuleUnload)(CUmodule hmod);
    CUresult (*cuModuleGetFunction)(CUfunction* hfunc, CUmodule hmod, const char* name);
    CUresult (*cuLaunchKernel)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                               unsigned int gridDimZ, unsigned int blockDimX,
                               unsigned int blockDimY, unsigned int blockDimZ,
                               unsigned int sharedMemBytes, CUstream hStream, void** kernelParams,
                               void** extra);
    CUresult (*cuMemAlloc)(CUdeviceptr* dptr, size_t bytesize);
    CUresult (*cuMemFree)(CUdeviceptr dptr);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
    CUresult (*cuMemcpyDtoH)(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
    CUresult (*cuMemcpyDtoD)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
    CUresult (*cuStreamCreate)(CUstream* phStream, unsigned int Flags);
    CUresult (*cuStreamDestroy)(CUstream hStream);
    CUresult (*cuStreamSynchronize)(CUstream hStream);
    CUresult (*cuCtxSynchronize)(void);

    // Event / async function pointers (for HCQ)
    CUresult (*cuEventCreate)(CUevent* phEvent, unsigned int Flags);
    CUresult (*cuEventDestroy)(CUevent hEvent);
    CUresult (*cuEventRecord)(CUevent hEvent, CUstream hStream);
    CUresult (*cuEventSynchronize)(CUevent hEvent);
    CUresult (*cuEventQuery)(CUevent hEvent);
    CUresult (*cuStreamWaitEvent)(CUstream hStream, CUevent hEvent, unsigned int Flags);
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr dst, const void* src, size_t n, CUstream s);
    CUresult (*cuMemcpyDtoHAsync)(void* dst, CUdeviceptr src, size_t n, CUstream s);
    CUresult (*cuEventElapsedTime)(float* ms, CUevent start, CUevent end);

    // NVRTC function pointers
    void* (*nvrtcCreateProgram)(void** prog, const char* src, const char* name, int numHeaders,
                                const char** headers, const char** includeNames);
    void* (*nvrtcCompileProgram)(void* prog, int numOptions, const char** options);
    void* (*nvrtcGetPTXSize)(void* prog, size_t* ptxSizeRet);
    void* (*nvrtcGetPTX)(void* prog, char* ptx);
    void* (*nvrtcDestroyProgram)(void** prog);
    void* (*nvrtcGetProgramLog)(void* prog, char* log);
    void* (*nvrtcGetProgramLogSize)(void* prog, size_t* logSizeRet);
} CMLCUDABackend;

typedef struct CMLCUDAKernel {
    CUmodule module;
    CUfunction function;
    char* kernel_name;

    int num_args;
    size_t shared_mem_size;
    int max_threads_per_block;

    int grid_dim[3];
    int block_dim[3];
} CMLCUDAKernel;

bool cml_cuda_available(void);
CMLCUDABackend* cml_cuda_backend_create(void);
int cml_cuda_backend_init(CMLCUDABackend* backend, int device_ordinal);
void cml_cuda_backend_free(CMLCUDABackend* backend);
int cml_cuda_get_device_count(CMLCUDABackend* backend);

CMLCUDAKernel* cml_cuda_compile_ptx(CMLCUDABackend* backend, const char* ptx_code,
                                    const char* kernel_name);
CMLCUDAKernel* cml_cuda_compile_source(CMLCUDABackend* backend, const char* cuda_code,
                                       const char* kernel_name);
void cml_cuda_kernel_free(CMLCUDABackend* backend, CMLCUDAKernel* kernel);

void cml_cuda_kernel_set_launch_config(CMLCUDAKernel* kernel, int grid_x, int grid_y, int grid_z,
                                       int block_x, int block_y, int block_z);
int cml_cuda_launch_kernel(CMLCUDABackend* backend, CMLCUDAKernel* kernel, void** args,
                           int num_args);
int cml_cuda_synchronize(CMLCUDABackend* backend);

CUdeviceptr cml_cuda_malloc(CMLCUDABackend* backend, size_t size);
void cml_cuda_free(CMLCUDABackend* backend, CUdeviceptr ptr);
int cml_cuda_memcpy_h2d(CMLCUDABackend* backend, CUdeviceptr dst, const void* src, size_t size);
int cml_cuda_memcpy_d2h(CMLCUDABackend* backend, void* dst, CUdeviceptr src, size_t size);

int cml_cuda_upload_tensor(CMLCUDABackend* backend, Tensor* tensor);
int cml_cuda_download_tensor(CMLCUDABackend* backend, Tensor* tensor);

#ifdef __cplusplus
}
#endif

#endif // CML_GPU_CUDA_BACKEND_H
