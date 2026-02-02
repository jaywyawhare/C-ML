/**
 * @file rocm_backend.c
 * @brief AMD ROCm/HIP backend implementation via dynamic loading
 */

#include "ops/ir/mlir/backends/rocm_backend.h"
#include "tensor/tensor.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __linux__
#include <dlfcn.h>
#define HIP_LIB_NAME "libamdhip64.so"
#define HIPRTC_LIB_NAME "libhiprtc.so"
#else
#define HIP_LIB_NAME NULL
#define HIPRTC_LIB_NAME NULL
#endif

#define HIP_SUCCESS 0
#define hipMemcpyHostToDevice 1
#define hipMemcpyDeviceToHost 2

// Dynamic library functions
#ifdef __linux__
static void* load_library(const char* name) {
    void* lib = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (!lib)
        LOG_DEBUG("Failed to load %s: %s", name, dlerror());
    return lib;
}
static void* get_symbol(void* lib, const char* name) { return dlsym(lib, name); }
static void unload_library(void* lib) {
    if (lib)
        dlclose(lib);
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

bool cml_rocm_available(void) {
#ifdef __linux__
    if (!HIP_LIB_NAME)
        return false;
    void* lib = load_library(HIP_LIB_NAME);
    if (!lib)
        return false;
    void* hipInit = get_symbol(lib, "hipInit");
    unload_library(lib);
    return hipInit != NULL;
#else
    return false;
#endif
}

CMLROCmBackend* cml_rocm_backend_create(void) {
    CMLROCmBackend* backend = calloc(1, sizeof(CMLROCmBackend));
    if (!backend)
        LOG_ERROR("Failed to allocate ROCm backend");
    return backend;
}

static int load_hip_functions(CMLROCmBackend* backend) {
    if (!HIP_LIB_NAME)
        return -1;

    backend->hip_lib = load_library(HIP_LIB_NAME);
    if (!backend->hip_lib)
        return -1;

#define LOAD_FUNC(name) backend->name = get_symbol(backend->hip_lib, #name)

    LOAD_FUNC(hipInit);
    LOAD_FUNC(hipGetDeviceCount);
    LOAD_FUNC(hipSetDevice);
    LOAD_FUNC(hipGetDeviceProperties);
    LOAD_FUNC(hipModuleLoad);
    LOAD_FUNC(hipModuleLoadData);
    LOAD_FUNC(hipModuleUnload);
    LOAD_FUNC(hipModuleGetFunction);
    LOAD_FUNC(hipModuleLaunchKernel);
    LOAD_FUNC(hipMalloc);
    LOAD_FUNC(hipFree);
    LOAD_FUNC(hipMemcpy);
    LOAD_FUNC(hipStreamCreate);
    LOAD_FUNC(hipStreamDestroy);
    LOAD_FUNC(hipStreamSynchronize);
    LOAD_FUNC(hipDeviceSynchronize);

#undef LOAD_FUNC

    if (!backend->hipInit || !backend->hipModuleLaunchKernel) {
        LOG_ERROR("Failed to load required HIP functions");
        return -1;
    }

    return 0;
}

int cml_rocm_backend_init(CMLROCmBackend* backend, int device_ordinal) {
    if (!backend)
        return -1;
    if (backend->initialized)
        return 0;

    if (load_hip_functions(backend) != 0)
        return -1;

    hipError_t err = backend->hipInit(0);
    if (err != HIP_SUCCESS) {
        LOG_ERROR("hipInit failed with error %d", err);
        return -1;
    }

    int device_count = 0;
    backend->hipGetDeviceCount(&device_count);
    if (device_count == 0 || device_ordinal >= device_count) {
        LOG_ERROR("No ROCm devices found or invalid ordinal");
        return -1;
    }

    backend->hipSetDevice(device_ordinal);
    backend->device = device_ordinal;

    // Get device properties via a temporary struct
    // In a full implementation, we'd call hipGetDeviceProperties

    backend->hipStreamCreate(&backend->stream);
    backend->initialized = true;

    LOG_INFO("ROCm backend initialized on device %d", device_ordinal);
    return 0;
}

void cml_rocm_backend_free(CMLROCmBackend* backend) {
    if (!backend)
        return;

    if (backend->initialized) {
        if (backend->stream && backend->hipStreamDestroy)
            backend->hipStreamDestroy(backend->stream);
    }

    unload_library(backend->hiprtc_lib);
    unload_library(backend->hip_lib);
    free(backend);
}

CMLROCmKernel* cml_rocm_compile_hsaco(CMLROCmBackend* backend, const char* hsaco_code,
                                      const char* kernel_name) {
    if (!backend || !backend->initialized || !hsaco_code || !kernel_name)
        return NULL;

    CMLROCmKernel* kernel = calloc(1, sizeof(CMLROCmKernel));
    if (!kernel)
        return NULL;

    hipError_t err = backend->hipModuleLoadData(&kernel->module, hsaco_code);
    if (err != HIP_SUCCESS) {
        free(kernel);
        return NULL;
    }

    err = backend->hipModuleGetFunction(&kernel->function, kernel->module, kernel_name);
    if (err != HIP_SUCCESS) {
        backend->hipModuleUnload(kernel->module);
        free(kernel);
        return NULL;
    }

    kernel->kernel_name = strdup(kernel_name);
    kernel->grid_dim[0] = kernel->grid_dim[1] = kernel->grid_dim[2] = 1;
    kernel->block_dim[0]                                            = 256;
    kernel->block_dim[1] = kernel->block_dim[2] = 1;

    return kernel;
}

void cml_rocm_kernel_free(CMLROCmBackend* backend, CMLROCmKernel* kernel) {
    if (!backend || !kernel)
        return;
    if (kernel->module)
        backend->hipModuleUnload(kernel->module);
    free(kernel->kernel_name);
    free(kernel);
}

int cml_rocm_launch_kernel(CMLROCmBackend* backend, CMLROCmKernel* kernel, void** args,
                           int num_args) {
    if (!backend || !backend->initialized || !kernel)
        return -1;
    (void)num_args;

    hipError_t err = backend->hipModuleLaunchKernel(
        kernel->function, kernel->grid_dim[0], kernel->grid_dim[1], kernel->grid_dim[2],
        kernel->block_dim[0], kernel->block_dim[1], kernel->block_dim[2], kernel->shared_mem_size,
        backend->stream, args, NULL);

    return (err == HIP_SUCCESS) ? 0 : -1;
}

int cml_rocm_synchronize(CMLROCmBackend* backend) {
    if (!backend || !backend->initialized)
        return -1;

    hipError_t err = backend->stream ? backend->hipStreamSynchronize(backend->stream)
                                     : backend->hipDeviceSynchronize();

    return (err == HIP_SUCCESS) ? 0 : -1;
}

hipDeviceptr_t cml_rocm_malloc(CMLROCmBackend* backend, size_t size) {
    if (!backend || !backend->initialized || size == 0)
        return NULL;
    void* ptr      = NULL;
    hipError_t err = backend->hipMalloc(&ptr, size);
    return (err == HIP_SUCCESS) ? ptr : NULL;
}

void cml_rocm_free(CMLROCmBackend* backend, hipDeviceptr_t ptr) {
    if (backend && backend->initialized && ptr)
        backend->hipFree(ptr);
}

int cml_rocm_memcpy_h2d(CMLROCmBackend* backend, hipDeviceptr_t dst, const void* src, size_t size) {
    if (!backend || !backend->initialized)
        return -1;
    hipError_t err = backend->hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
    return (err == HIP_SUCCESS) ? 0 : -1;
}

int cml_rocm_memcpy_d2h(CMLROCmBackend* backend, void* dst, hipDeviceptr_t src, size_t size) {
    if (!backend || !backend->initialized)
        return -1;
    hipError_t err = backend->hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
    return (err == HIP_SUCCESS) ? 0 : -1;
}

int cml_rocm_upload_tensor(CMLROCmBackend* backend, Tensor* tensor) {
    if (!backend || !tensor || !tensor->data)
        return -1;
    size_t size = tensor->numel * cml_dtype_size(tensor->dtype);
    if (!tensor->buffer_handle) {
        tensor->buffer_handle = cml_rocm_malloc(backend, size);
        if (!tensor->buffer_handle)
            return -1;
    }
    return cml_rocm_memcpy_h2d(backend, tensor->buffer_handle, tensor->data, size);
}

int cml_rocm_download_tensor(CMLROCmBackend* backend, Tensor* tensor) {
    if (!backend || !tensor || !tensor->buffer_handle)
        return -1;
    size_t size = tensor->numel * cml_dtype_size(tensor->dtype);
    if (!tensor->data) {
        tensor->data = malloc(size);
        if (!tensor->data)
            return -1;
        tensor->owns_data = true;
    }
    return cml_rocm_memcpy_d2h(backend, tensor->data, tensor->buffer_handle, size);
}
