/**
 * @file cuda_backend.c
 * @brief CUDA backend implementation via dynamic loading
 */

#include "ops/ir/gpu/cuda_backend.h"
#include "tensor/tensor.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __linux__
#include <dlfcn.h>
#define CUDA_LIB_NAME "libcuda.so.1"
#define NVRTC_LIB_NAME "libnvrtc.so"
#elif defined(__APPLE__)
// CUDA is not supported on modern macOS
#define CUDA_LIB_NAME NULL
#define NVRTC_LIB_NAME NULL
#elif defined(_WIN32)
#include <windows.h>
#define CUDA_LIB_NAME "nvcuda.dll"
#define NVRTC_LIB_NAME "nvrtc64_*.dll"
#endif

#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK 1
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76

#define CUDA_SUCCESS 0

#ifdef __linux__
static void* load_library(const char* name) {
    void* lib = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        LOG_DEBUG("Failed to load %s: %s", name, dlerror());
    }
    return lib;
}

static void* get_symbol(void* lib, const char* name) { return dlsym(lib, name); }

static void unload_library(void* lib) {
    if (lib)
        dlclose(lib);
}
#elif defined(_WIN32)
static void* load_library(const char* name) {
    HMODULE lib = LoadLibraryA(name);
    if (!lib) {
        LOG_DEBUG("Failed to load %s: error %d", name, GetLastError());
    }
    return lib;
}

static void* get_symbol(void* lib, const char* name) {
    return (void*)GetProcAddress((HMODULE)lib, name);
}

static void unload_library(void* lib) {
    if (lib)
        FreeLibrary((HMODULE)lib);
}
#else
static void* load_library(const char* name) {
    (void)name;
    return NULL;
}
static void* get_symbol(void* lib, const char* name) {
    (void)lib;
    (void)name;
    return NULL;
}
static void unload_library(void* lib) { (void)lib; }
#endif

bool cml_cuda_available(void) {
#if defined(__APPLE__)
    return false; // CUDA not supported on modern macOS
#else
    if (!CUDA_LIB_NAME)
        return false;

    void* lib = load_library(CUDA_LIB_NAME);
    if (!lib)
        return false;

    // Try to get cuInit
    void* cuInit = get_symbol(lib, "cuInit");
    unload_library(lib);

    return cuInit != NULL;
#endif
}

CMLCUDABackend* cml_cuda_backend_create(void) {
    CMLCUDABackend* backend = calloc(1, sizeof(CMLCUDABackend));
    if (!backend) {
        LOG_ERROR("Failed to allocate CUDA backend");
        return NULL;
    }
    return backend;
}

static int load_cuda_functions(CMLCUDABackend* backend) {
    if (!CUDA_LIB_NAME) {
        LOG_ERROR("CUDA not supported on this platform");
        return -1;
    }

    backend->cuda_lib = load_library(CUDA_LIB_NAME);
    if (!backend->cuda_lib) {
        LOG_ERROR("Failed to load CUDA driver library");
        return -1;
    }

#define LOAD_FUNC(name)                                                                            \
    backend->name = get_symbol(backend->cuda_lib, #name);                                          \
    if (!backend->name) {                                                                          \
        LOG_ERROR("Failed to load CUDA function: %s", #name);                                      \
        return -1;                                                                                 \
    }

    LOAD_FUNC(cuInit);
    LOAD_FUNC(cuDeviceGet);
    LOAD_FUNC(cuDeviceGetCount);
    LOAD_FUNC(cuDeviceGetName);
    LOAD_FUNC(cuDeviceTotalMem);
    LOAD_FUNC(cuDeviceGetAttribute);
    LOAD_FUNC(cuCtxCreate);
    LOAD_FUNC(cuCtxDestroy);
    LOAD_FUNC(cuCtxSetCurrent);
    LOAD_FUNC(cuModuleLoadData);
    LOAD_FUNC(cuModuleUnload);
    LOAD_FUNC(cuModuleGetFunction);
    LOAD_FUNC(cuLaunchKernel);
    LOAD_FUNC(cuMemAlloc);
    LOAD_FUNC(cuMemFree);
    LOAD_FUNC(cuMemcpyHtoD);
    LOAD_FUNC(cuMemcpyDtoH);
    LOAD_FUNC(cuMemcpyDtoD);
    LOAD_FUNC(cuStreamCreate);
    LOAD_FUNC(cuStreamDestroy);
    LOAD_FUNC(cuStreamSynchronize);
    LOAD_FUNC(cuCtxSynchronize);

    /* Event and async transfer functions (best-effort, non-fatal) */
#define LOAD_FUNC_OPTIONAL(name) \
    backend->name = get_symbol(backend->cuda_lib, #name);

    LOAD_FUNC_OPTIONAL(cuEventCreate);
    LOAD_FUNC_OPTIONAL(cuEventDestroy);
    LOAD_FUNC_OPTIONAL(cuEventRecord);
    LOAD_FUNC_OPTIONAL(cuEventSynchronize);
    LOAD_FUNC_OPTIONAL(cuEventQuery);
    LOAD_FUNC_OPTIONAL(cuStreamWaitEvent);
    LOAD_FUNC_OPTIONAL(cuMemcpyHtoDAsync);
    LOAD_FUNC_OPTIONAL(cuMemcpyDtoHAsync);
    LOAD_FUNC_OPTIONAL(cuEventElapsedTime);

#undef LOAD_FUNC_OPTIONAL

#undef LOAD_FUNC

    // Optionally load NVRTC for runtime compilation
    if (NVRTC_LIB_NAME) {
        backend->nvrtc_lib = load_library(NVRTC_LIB_NAME);
        if (backend->nvrtc_lib) {
            backend->nvrtcCreateProgram  = get_symbol(backend->nvrtc_lib, "nvrtcCreateProgram");
            backend->nvrtcCompileProgram = get_symbol(backend->nvrtc_lib, "nvrtcCompileProgram");
            backend->nvrtcGetPTXSize     = get_symbol(backend->nvrtc_lib, "nvrtcGetPTXSize");
            backend->nvrtcGetPTX         = get_symbol(backend->nvrtc_lib, "nvrtcGetPTX");
            backend->nvrtcDestroyProgram = get_symbol(backend->nvrtc_lib, "nvrtcDestroyProgram");
            backend->nvrtcGetProgramLog  = get_symbol(backend->nvrtc_lib, "nvrtcGetProgramLog");
            backend->nvrtcGetProgramLogSize =
                get_symbol(backend->nvrtc_lib, "nvrtcGetProgramLogSize");
            LOG_DEBUG("NVRTC loaded for runtime CUDA compilation");
        }
    }

    return 0;
}

int cml_cuda_backend_init(CMLCUDABackend* backend, int device_ordinal) {
    if (!backend)
        return -1;

    if (backend->initialized) {
        LOG_DEBUG("CUDA backend already initialized");
        return 0;
    }

    // Load CUDA functions
    if (load_cuda_functions(backend) != 0) {
        return -1;
    }

    // Initialize CUDA
    CUresult err = backend->cuInit(0);
    if (err != CUDA_SUCCESS) {
        LOG_ERROR("cuInit failed with error %d", err);
        return -1;
    }

    // Get device count
    int device_count = 0;
    err              = backend->cuDeviceGetCount(&device_count);
    if (err != CUDA_SUCCESS || device_count == 0) {
        LOG_ERROR("No CUDA devices found");
        return -1;
    }

    if (device_ordinal >= device_count) {
        LOG_ERROR("Invalid device ordinal %d (only %d devices)", device_ordinal, device_count);
        return -1;
    }

    // Get device
    err = backend->cuDeviceGet(&backend->device, device_ordinal);
    if (err != CUDA_SUCCESS) {
        LOG_ERROR("cuDeviceGet failed");
        return -1;
    }

    // Get device info
    backend->cuDeviceGetName(backend->device_name, sizeof(backend->device_name), backend->device);
    backend->cuDeviceTotalMem(&backend->total_memory, backend->device);
    backend->cuDeviceGetAttribute(&backend->compute_capability_major,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, backend->device);
    backend->cuDeviceGetAttribute(&backend->compute_capability_minor,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, backend->device);
    backend->cuDeviceGetAttribute(&backend->multiprocessor_count,
                                  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, backend->device);
    backend->cuDeviceGetAttribute(&backend->max_threads_per_block,
                                  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, backend->device);

    LOG_INFO("CUDA device %d: %s", device_ordinal, backend->device_name);
    LOG_INFO("  Compute capability: %d.%d", backend->compute_capability_major,
             backend->compute_capability_minor);
    LOG_INFO("  Total memory: %.2f GB", backend->total_memory / (1024.0 * 1024.0 * 1024.0));
    LOG_INFO("  Multiprocessors: %d", backend->multiprocessor_count);

    // Create context
    err = backend->cuCtxCreate(&backend->context, 0, backend->device);
    if (err != CUDA_SUCCESS) {
        LOG_ERROR("cuCtxCreate failed with error %d", err);
        return -1;
    }

    // Create default stream
    err = backend->cuStreamCreate(&backend->stream, 0);
    if (err != CUDA_SUCCESS) {
        LOG_WARNING("cuStreamCreate failed, using default stream");
        backend->stream = NULL;
    }

    backend->initialized = true;
    return 0;
}

void cml_cuda_backend_free(CMLCUDABackend* backend) {
    if (!backend)
        return;

    if (backend->initialized) {
        if (backend->stream && backend->cuStreamDestroy) {
            backend->cuStreamDestroy(backend->stream);
        }
        if (backend->context && backend->cuCtxDestroy) {
            backend->cuCtxDestroy(backend->context);
        }
    }

    if (backend->nvrtc_lib) {
        unload_library(backend->nvrtc_lib);
    }
    if (backend->cuda_lib) {
        unload_library(backend->cuda_lib);
    }

    free(backend);
}

int cml_cuda_get_device_count(CMLCUDABackend* backend) {
    if (!backend || !backend->cuDeviceGetCount)
        return -1;

    int count    = 0;
    CUresult err = backend->cuDeviceGetCount(&count);
    return (err == CUDA_SUCCESS) ? count : -1;
}

CMLCUDAKernel* cml_cuda_compile_ptx(CMLCUDABackend* backend, const char* ptx_code,
                                    const char* kernel_name) {
    if (!backend || !backend->initialized || !ptx_code || !kernel_name) {
        LOG_ERROR("Invalid arguments to cml_cuda_compile_ptx");
        return NULL;
    }

    CMLCUDAKernel* kernel = calloc(1, sizeof(CMLCUDAKernel));
    if (!kernel) {
        LOG_ERROR("Failed to allocate CUDA kernel");
        return NULL;
    }

    // Load PTX module
    CUresult err = backend->cuModuleLoadData(&kernel->module, ptx_code);
    if (err != CUDA_SUCCESS) {
        LOG_ERROR("cuModuleLoadData failed with error %d", err);
        free(kernel);
        return NULL;
    }

    // Get kernel function
    err = backend->cuModuleGetFunction(&kernel->function, kernel->module, kernel_name);
    if (err != CUDA_SUCCESS) {
        LOG_ERROR("cuModuleGetFunction failed for '%s' with error %d", kernel_name, err);
        backend->cuModuleUnload(kernel->module);
        free(kernel);
        return NULL;
    }

    kernel->kernel_name = strdup(kernel_name);

    // Default launch configuration
    kernel->grid_dim[0]  = 1;
    kernel->grid_dim[1]  = 1;
    kernel->grid_dim[2]  = 1;
    kernel->block_dim[0] = 256;
    kernel->block_dim[1] = 1;
    kernel->block_dim[2] = 1;

    LOG_DEBUG("Compiled CUDA kernel: %s", kernel_name);

    return kernel;
}

CMLCUDAKernel* cml_cuda_compile_source(CMLCUDABackend* backend, const char* cuda_code,
                                       const char* kernel_name) {
    if (!backend || !backend->initialized || !cuda_code || !kernel_name) {
        return NULL;
    }

    // Check if NVRTC is available
    if (!backend->nvrtcCreateProgram || !backend->nvrtcCompileProgram) {
        LOG_ERROR("NVRTC not available for runtime compilation");
        return NULL;
    }

    // Create NVRTC program
    void* prog = NULL;
    backend->nvrtcCreateProgram(&prog, cuda_code, "cml_kernel.cu", 0, NULL, NULL);
    if (!prog) {
        LOG_ERROR("nvrtcCreateProgram failed");
        return NULL;
    }

    // Compile with target compute capability
    char arch_flag[32];
    snprintf(arch_flag, sizeof(arch_flag), "--gpu-architecture=compute_%d%d",
             backend->compute_capability_major, backend->compute_capability_minor);

    const char* options[] = {arch_flag, "-default-device"};
    void* result          = backend->nvrtcCompileProgram(prog, 2, options);

    // Check for compilation errors
    if (result) {
        size_t log_size = 0;
        backend->nvrtcGetProgramLogSize(prog, &log_size);
        if (log_size > 1) {
            char* log = malloc(log_size);
            backend->nvrtcGetProgramLog(prog, log);
            LOG_ERROR("NVRTC compilation failed:\n%s", log);
            free(log);
        }
        backend->nvrtcDestroyProgram(&prog);
        return NULL;
    }

    // Get PTX
    size_t ptx_size = 0;
    backend->nvrtcGetPTXSize(prog, &ptx_size);
    char* ptx = malloc(ptx_size);
    backend->nvrtcGetPTX(prog, ptx);
    backend->nvrtcDestroyProgram(&prog);

    // Compile PTX to kernel
    CMLCUDAKernel* kernel = cml_cuda_compile_ptx(backend, ptx, kernel_name);
    free(ptx);

    return kernel;
}

void cml_cuda_kernel_free(CMLCUDABackend* backend, CMLCUDAKernel* kernel) {
    if (!backend || !kernel)
        return;

    if (kernel->module && backend->cuModuleUnload) {
        backend->cuModuleUnload(kernel->module);
    }
    free(kernel->kernel_name);
    free(kernel);
}

void cml_cuda_kernel_set_launch_config(CMLCUDAKernel* kernel, int grid_x, int grid_y, int grid_z,
                                       int block_x, int block_y, int block_z) {
    if (!kernel)
        return;

    kernel->grid_dim[0]  = grid_x;
    kernel->grid_dim[1]  = grid_y;
    kernel->grid_dim[2]  = grid_z;
    kernel->block_dim[0] = block_x;
    kernel->block_dim[1] = block_y;
    kernel->block_dim[2] = block_z;
}

int cml_cuda_launch_kernel(CMLCUDABackend* backend, CMLCUDAKernel* kernel, void** args,
                           int num_args) {
    if (!backend || !backend->initialized || !kernel || !kernel->function) {
        LOG_ERROR("Invalid arguments to cml_cuda_launch_kernel");
        return -1;
    }

    (void)num_args; // Args passed via kernelParams

    CUresult err = backend->cuLaunchKernel(
        kernel->function, kernel->grid_dim[0], kernel->grid_dim[1], kernel->grid_dim[2],
        kernel->block_dim[0], kernel->block_dim[1], kernel->block_dim[2], kernel->shared_mem_size,
        backend->stream, args, NULL);

    if (err != CUDA_SUCCESS) {
        LOG_ERROR("cuLaunchKernel failed with error %d", err);
        return -1;
    }

    return 0;
}

int cml_cuda_synchronize(CMLCUDABackend* backend) {
    if (!backend || !backend->initialized)
        return -1;

    CUresult err;
    if (backend->stream) {
        err = backend->cuStreamSynchronize(backend->stream);
    } else {
        err = backend->cuCtxSynchronize();
    }

    return (err == CUDA_SUCCESS) ? 0 : -1;
}

CUdeviceptr cml_cuda_malloc(CMLCUDABackend* backend, size_t size) {
    if (!backend || !backend->initialized || size == 0)
        return 0;

    CUdeviceptr ptr = 0;
    CUresult err    = backend->cuMemAlloc(&ptr, size);

    if (err != CUDA_SUCCESS) {
        LOG_ERROR("cuMemAlloc failed for %zu bytes with error %d", size, err);
        return 0;
    }

    return ptr;
}

void cml_cuda_free(CMLCUDABackend* backend, CUdeviceptr ptr) {
    if (!backend || !backend->initialized || !ptr)
        return;
    backend->cuMemFree(ptr);
}

int cml_cuda_memcpy_h2d(CMLCUDABackend* backend, CUdeviceptr dst, const void* src, size_t size) {
    if (!backend || !backend->initialized || !dst || !src || size == 0)
        return -1;

    CUresult err = backend->cuMemcpyHtoD(dst, src, size);
    return (err == CUDA_SUCCESS) ? 0 : -1;
}

int cml_cuda_memcpy_d2h(CMLCUDABackend* backend, void* dst, CUdeviceptr src, size_t size) {
    if (!backend || !backend->initialized || !dst || !src || size == 0)
        return -1;

    CUresult err = backend->cuMemcpyDtoH(dst, src, size);
    return (err == CUDA_SUCCESS) ? 0 : -1;
}

int cml_cuda_upload_tensor(CMLCUDABackend* backend, Tensor* tensor) {
    if (!backend || !backend->initialized || !tensor || !tensor->data) {
        return -1;
    }

    size_t size = tensor->numel * cml_dtype_size(tensor->dtype);

    // Allocate device memory if not already allocated
    if (!tensor->buffer_handle) {
        CUdeviceptr ptr = cml_cuda_malloc(backend, size);
        if (!ptr)
            return -1;
        tensor->buffer_handle = (void*)ptr;
    }

    // Copy data to device
    return cml_cuda_memcpy_h2d(backend, (CUdeviceptr)tensor->buffer_handle, tensor->data, size);
}

int cml_cuda_download_tensor(CMLCUDABackend* backend, Tensor* tensor) {
    if (!backend || !backend->initialized || !tensor || !tensor->buffer_handle) {
        return -1;
    }

    size_t size = tensor->numel * cml_dtype_size(tensor->dtype);

    // Allocate host memory if needed
    if (!tensor->data) {
        tensor->data = malloc(size);
        if (!tensor->data)
            return -1;
        tensor->owns_data = true;
    }

    // Copy data from device
    return cml_cuda_memcpy_d2h(backend, tensor->data, (CUdeviceptr)tensor->buffer_handle, size);
}
