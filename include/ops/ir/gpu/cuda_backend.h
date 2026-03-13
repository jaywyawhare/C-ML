/**
 * @file cuda_backend.h
 * @brief CUDA backend for JIT kernel execution
 *
 * This module provides CUDA GPU execution via dynamic loading of the CUDA
 * driver API. It supports:
 * - Runtime detection of CUDA availability
 * - PTX code compilation via cuModuleLoadData
 * - Kernel execution via cuLaunchKernel
 * - Memory management via cuMemAlloc/cuMemFree
 */

#ifndef CML_GPU_CUDA_BACKEND_H
#define CML_GPU_CUDA_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
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

/**
 * @brief CUDA backend context
 */
typedef struct CMLCUDABackend {
    // Dynamic library handle
    void* cuda_lib;  // libcuda.so handle
    void* nvrtc_lib; // libnvrtc.so handle (for runtime compilation)

    // Device state
    CUdevice device;
    CUcontext context;
    CUstream stream;
    bool initialized;

    // Device info
    char device_name[256];
    size_t total_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;

    // Function pointers (dynamically loaded from libcuda.so)
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

    // NVRTC function pointers (for runtime compilation)
    void* (*nvrtcCreateProgram)(void** prog, const char* src, const char* name, int numHeaders,
                                const char** headers, const char** includeNames);
    void* (*nvrtcCompileProgram)(void* prog, int numOptions, const char** options);
    void* (*nvrtcGetPTXSize)(void* prog, size_t* ptxSizeRet);
    void* (*nvrtcGetPTX)(void* prog, char* ptx);
    void* (*nvrtcDestroyProgram)(void** prog);
    void* (*nvrtcGetProgramLog)(void* prog, char* log);
    void* (*nvrtcGetProgramLogSize)(void* prog, size_t* logSizeRet);
} CMLCUDABackend;

/**
 * @brief Compiled CUDA kernel
 */
typedef struct CMLCUDAKernel {
    CUmodule module;
    CUfunction function;
    char* kernel_name;

    // Kernel metadata
    int num_args;
    size_t shared_mem_size;
    int max_threads_per_block;

    // Optimal launch configuration
    int grid_dim[3];
    int block_dim[3];
} CMLCUDAKernel;

// Backend Lifecycle

/**
 * @brief Check if CUDA is available on the system
 * @return true if CUDA is available, false otherwise
 */
bool cml_cuda_available(void);

/**
 * @brief Create CUDA backend context
 * @return New backend context or NULL on failure
 */
CMLCUDABackend* cml_cuda_backend_create(void);

/**
 * @brief Initialize CUDA backend
 * @param backend Backend context
 * @param device_ordinal Device ordinal (0 for first GPU)
 * @return 0 on success, -1 on failure
 */
int cml_cuda_backend_init(CMLCUDABackend* backend, int device_ordinal);

/**
 * @brief Free CUDA backend and all resources
 * @param backend Backend context to free
 */
void cml_cuda_backend_free(CMLCUDABackend* backend);

/**
 * @brief Get device count
 * @param backend Backend context
 * @return Number of CUDA devices, or -1 on error
 */
int cml_cuda_get_device_count(CMLCUDABackend* backend);

// Kernel Compilation

/**
 * @brief Compile PTX code to CUDA kernel
 * @param backend Backend context
 * @param ptx_code PTX code string
 * @param kernel_name Name of the kernel function
 * @return Compiled kernel or NULL on failure
 */
CMLCUDAKernel* cml_cuda_compile_ptx(CMLCUDABackend* backend, const char* ptx_code,
                                    const char* kernel_name);

/**
 * @brief Compile CUDA C code to kernel (via NVRTC)
 * @param backend Backend context
 * @param cuda_code CUDA C code string
 * @param kernel_name Name of the kernel function
 * @return Compiled kernel or NULL on failure
 */
CMLCUDAKernel* cml_cuda_compile_source(CMLCUDABackend* backend, const char* cuda_code,
                                       const char* kernel_name);

/**
 * @brief Free compiled kernel
 * @param backend Backend context
 * @param kernel Kernel to free
 */
void cml_cuda_kernel_free(CMLCUDABackend* backend, CMLCUDAKernel* kernel);

// Kernel Execution

/**
 * @brief Set kernel launch configuration
 * @param kernel Kernel to configure
 * @param grid_x Grid dimension X
 * @param grid_y Grid dimension Y
 * @param grid_z Grid dimension Z
 * @param block_x Block dimension X
 * @param block_y Block dimension Y
 * @param block_z Block dimension Z
 */
void cml_cuda_kernel_set_launch_config(CMLCUDAKernel* kernel, int grid_x, int grid_y, int grid_z,
                                       int block_x, int block_y, int block_z);

/**
 * @brief Launch CUDA kernel
 * @param backend Backend context
 * @param kernel Compiled kernel
 * @param args Kernel arguments (device pointers)
 * @param num_args Number of arguments
 * @return 0 on success, -1 on failure
 */
int cml_cuda_launch_kernel(CMLCUDABackend* backend, CMLCUDAKernel* kernel, void** args,
                           int num_args);

/**
 * @brief Synchronize CUDA stream (wait for kernel completion)
 * @param backend Backend context
 * @return 0 on success, -1 on failure
 */
int cml_cuda_synchronize(CMLCUDABackend* backend);

// Memory Management

/**
 * @brief Allocate device memory
 * @param backend Backend context
 * @param size Size in bytes
 * @return Device pointer or 0 on failure
 */
CUdeviceptr cml_cuda_malloc(CMLCUDABackend* backend, size_t size);

/**
 * @brief Free device memory
 * @param backend Backend context
 * @param ptr Device pointer
 */
void cml_cuda_free(CMLCUDABackend* backend, CUdeviceptr ptr);

/**
 * @brief Copy data from host to device
 * @param backend Backend context
 * @param dst Device destination
 * @param src Host source
 * @param size Size in bytes
 * @return 0 on success, -1 on failure
 */
int cml_cuda_memcpy_h2d(CMLCUDABackend* backend, CUdeviceptr dst, const void* src, size_t size);

/**
 * @brief Copy data from device to host
 * @param backend Backend context
 * @param dst Host destination
 * @param src Device source
 * @param size Size in bytes
 * @return 0 on success, -1 on failure
 */
int cml_cuda_memcpy_d2h(CMLCUDABackend* backend, void* dst, CUdeviceptr src, size_t size);

// Tensor Operations

/**
 * @brief Upload tensor data to device
 * @param backend Backend context
 * @param tensor Tensor to upload
 * @return 0 on success, -1 on failure
 */
int cml_cuda_upload_tensor(CMLCUDABackend* backend, Tensor* tensor);

/**
 * @brief Download tensor data from device
 * @param backend Backend context
 * @param tensor Tensor to download
 * @return 0 on success, -1 on failure
 */
int cml_cuda_download_tensor(CMLCUDABackend* backend, Tensor* tensor);

#ifdef __cplusplus
}
#endif

#endif // CML_GPU_CUDA_BACKEND_H
