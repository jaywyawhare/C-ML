/**
 * @file device.c
 * @brief Automatic device detection and management implementation
 */

#include "Core/device.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include "tensor/tensor.h"
#include "nn/layers/linear.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef __linux__
#include <dlfcn.h>
#define CML_DLOPEN(path, mode) dlopen(path, mode)
#define CML_DLSYM(handle, symbol) dlsym(handle, symbol)
#define CML_DLCLOSE(handle) dlclose(handle)
#ifndef RTLD_LAZY
#define RTLD_LAZY 1
#endif
#elif defined(__APPLE__)
#include <dlfcn.h>
#include <objc/objc.h>
#include <objc/runtime.h>
#include <objc/message.h>
#define CML_DLOPEN(path, mode) dlopen(path, mode)
#define CML_DLSYM(handle, symbol) dlsym(handle, symbol)
#define CML_DLCLOSE(handle) dlclose(handle)
#ifndef RTLD_LAZY
#define RTLD_LAZY 1
#endif
#elif defined(_WIN32)
#include <windows.h>
#define CML_DLOPEN(path, mode) LoadLibraryA(path)
#define CML_DLSYM(handle, symbol) GetProcAddress((HMODULE)handle, symbol)
#define CML_DLCLOSE(handle) FreeLibrary((HMODULE)handle)
#define RTLD_LAZY 0
#else
#define CML_DLOPEN(path, mode) NULL
#define CML_DLSYM(handle, symbol) NULL
#define CML_DLCLOSE(handle) ((void)0)
#define RTLD_LAZY 0
#endif

static DeviceType g_default_device = DEVICE_CPU;
static DeviceType g_current_device = DEVICE_CPU;
static bool g_cuda_available       = false;
static bool g_metal_available      = false;
static bool g_rocm_available       = false;
static bool g_device_initialized   = false;

// CUDA detection function pointers (loaded dynamically)
typedef int (*cudaRuntimeGetVersion_fn)(int*);
typedef int (*cudaGetDeviceCount_fn)(int*);
typedef int (*cudaMalloc_fn)(void**, size_t);
typedef int (*cudaFree_fn)(void*);
typedef int (*cudaMemcpy_fn)(void*, const void*, size_t, int);
typedef int (*cudaDeviceSynchronize_fn)(void);
typedef int (*cudaMemGetInfo_fn)(size_t*, size_t*);

static cudaRuntimeGetVersion_fn cudaRuntimeGetVersion = NULL;
static cudaGetDeviceCount_fn cudaGetDeviceCount       = NULL;
static cudaMalloc_fn cudaMalloc                       = NULL;
static cudaFree_fn cudaFree                           = NULL;
static cudaMemcpy_fn cudaMemcpy                       = NULL;
static cudaDeviceSynchronize_fn cudaDeviceSynchronize = NULL;
static cudaMemGetInfo_fn cudaMemGetInfo               = NULL;
static void* g_cuda_lib                               = NULL;

#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define cudaMemcpyDeviceToDevice 3

// Metal detection (macOS/iOS) - handled in check_metal_available()

/**
 * @brief Try to load CUDA library dynamically
 */
static bool try_load_cuda(void) {
    const char* lib_paths[] = {
#ifdef __linux__
        "libcudart.so", "libcudart.so.12", "libcudart.so.11",
#elif defined(__APPLE__)
        "libcudart.dylib", "/usr/local/cuda/lib/libcudart.dylib",
#elif defined(_WIN32)
        "cudart64_12.dll", "cudart64_11.dll", "cudart.dll",
#endif
        NULL};

    for (int i = 0; lib_paths[i] != NULL; i++) {
        g_cuda_lib = CML_DLOPEN(lib_paths[i], RTLD_LAZY);
        if (g_cuda_lib) {
            break;
        }
    }

    if (!g_cuda_lib) {
        return false;
    }

    cudaRuntimeGetVersion =
        (cudaRuntimeGetVersion_fn)CML_DLSYM(g_cuda_lib, "cudaRuntimeGetVersion");
    cudaGetDeviceCount = (cudaGetDeviceCount_fn)CML_DLSYM(g_cuda_lib, "cudaGetDeviceCount");
    cudaMalloc         = (cudaMalloc_fn)CML_DLSYM(g_cuda_lib, "cudaMalloc");
    cudaFree           = (cudaFree_fn)CML_DLSYM(g_cuda_lib, "cudaFree");
    cudaMemcpy         = (cudaMemcpy_fn)CML_DLSYM(g_cuda_lib, "cudaMemcpy");
    cudaDeviceSynchronize =
        (cudaDeviceSynchronize_fn)CML_DLSYM(g_cuda_lib, "cudaDeviceSynchronize");
    cudaMemGetInfo = (cudaMemGetInfo_fn)CML_DLSYM(g_cuda_lib, "cudaMemGetInfo");

    if (!cudaRuntimeGetVersion || !cudaGetDeviceCount) {
        CML_DLCLOSE(g_cuda_lib);
        g_cuda_lib = NULL;
        return false;
    }

    return true;
}

/**
 * @brief Check if CUDA is actually available
 */
static bool check_cuda_available(void) {
    if (!try_load_cuda()) {
        return false;
    }

    int version      = 0;
    int device_count = 0;

    if (cudaRuntimeGetVersion && cudaRuntimeGetVersion(&version) == 0) {
        if (cudaGetDeviceCount && cudaGetDeviceCount(&device_count) == 0) {
            if (device_count > 0) {
                LOG_INFO("CUDA detected: version %d, %d device(s) available", version,
                         device_count);
                return true;
            }
        }
    }

    if (g_cuda_lib) {
        CML_DLCLOSE(g_cuda_lib);
        g_cuda_lib = NULL;
    }

    return false;
}

/**
 * @brief Check if Metal is available (macOS/iOS)
 */
static bool check_metal_available(void) {
#ifdef __APPLE__
    // Try to load Metal framework dynamically
    void* metal_framework =
        CML_DLOPEN("/System/Library/Frameworks/Metal.framework/Metal", RTLD_LAZY);
    if (!metal_framework) {
        return false;
    }

    // Try to get MTLCreateSystemDefaultDevice function
    void* (*MTLCreateSystemDefaultDevice)(void) =
        (void* (*)(void))CML_DLSYM(metal_framework, "MTLCreateSystemDefaultDevice");
    if (MTLCreateSystemDefaultDevice) {
        void* device = MTLCreateSystemDefaultDevice();
        if (device) {
            LOG_INFO("Metal detected: device available");
            CML_DLCLOSE(metal_framework);
            return true;
        }
    }

    CML_DLCLOSE(metal_framework);
#endif
    return false;
}

/**
 * @brief Check if ROCm is available
 */
static bool check_rocm_available(void) {
    const char* lib_paths[] = {
#ifdef __linux__
        "libhip_hcc.so", "libamdhip64.so",
#elif defined(__APPLE__)
        "libhip_hcc.dylib", "libamdhip64.dylib",
#elif defined(_WIN32)
        "hip64.dll", "amdhip64.dll",
#endif
        NULL};

    for (int i = 0; lib_paths[i] != NULL; i++) {
        void* rocm_lib = CML_DLOPEN(lib_paths[i], RTLD_LAZY);
        if (rocm_lib) {
            LOG_INFO("ROCm detected: library available");
            CML_DLCLOSE(rocm_lib);
            return true;
        }
    }
    return false;
}

bool device_cuda_available(void) { return g_cuda_available; }

bool device_metal_available(void) { return g_metal_available; }

bool device_rocm_available(void) { return g_rocm_available; }

DeviceType device_get_best_available(void) {
    // Priority order: CUDA > Metal > ROCm > CPU
    if (g_cuda_available) {
        return DEVICE_CUDA;
    } else if (g_metal_available) {
        return DEVICE_METAL;
    } else if (g_rocm_available) {
        return DEVICE_ROCM;
    } else {
        return DEVICE_CPU;
    }
}

bool device_detect_available(void) {
    // Check for all available accelerators
    g_cuda_available  = check_cuda_available();
    g_metal_available = check_metal_available();
    g_rocm_available  = check_rocm_available();

    // Set default device based on best available accelerator
    g_default_device = device_get_best_available();
    g_current_device = g_default_device;

    const char* device_name = device_get_name(g_default_device);
    LOG_INFO("Default device set to %s", device_name);

    return true;
}

void device_init(void) {
    if (g_device_initialized) {
        return;
    }

    LOG_DEBUG("Initializing device management system");
    device_detect_available();
    g_device_initialized = true;
}

void device_cleanup(void) {
    if (g_cuda_lib) {
        CML_DLCLOSE(g_cuda_lib);
        g_cuda_lib = NULL;
    }
    g_cuda_available     = false;
    g_metal_available    = false;
    g_rocm_available     = false;
    g_device_initialized = false;
}

DeviceType device_get_default(void) {
    if (!g_device_initialized) {
        device_init();
    }
    // If DEVICE_AUTO is set, return the best available device
    if (g_default_device == DEVICE_AUTO) {
        return device_get_best_available();
    }
    return g_default_device;
}

void device_set_default(DeviceType device) {
    g_default_device = device;
    g_current_device = device;
    LOG_INFO("Default device set to %s", device_get_name(device));
}

DeviceType device_get_current(void) { return g_current_device; }

void device_set_current(DeviceType device) {
    g_current_device = device;
    LOG_DEBUG("Set current device to: %s", device_get_name(device));
}

const char* device_get_name(DeviceType device) {
    switch (device) {
    case DEVICE_CPU:
        return "CPU";
    case DEVICE_CUDA:
        return "CUDA";
    case DEVICE_METAL:
        return "Metal";
    case DEVICE_ROCM:
        return "ROCm";
    case DEVICE_AUTO:
        return "Auto";
    default:
        return "UNKNOWN";
    }
}

void device_print_info(void) {
    printf("\n=== Device Information ===\n");
    printf("Default device: %s\n", device_get_name(device_get_default()));
    printf("Current device: %s\n", device_get_name(device_get_current()));
    printf("\nAvailable accelerators:\n");
    printf("  CPU: Always available\n");
    printf("  CUDA: %s\n", g_cuda_available ? "Yes" : "No");
    printf("  Metal: %s\n", g_metal_available ? "Yes" : "No");
    printf("  ROCm: %s\n", g_rocm_available ? "Yes" : "No");

    if (g_cuda_available) {
        int device_count = 0;
        if (cudaGetDeviceCount) {
            cudaGetDeviceCount(&device_count);
            printf("\nCUDA devices: %d\n", device_count);
        }
    }
    printf("========================\n\n");
}

// Automatic device-aware wrappers
Tensor* tensor_empty_auto(int* shape, int ndim, int dtype) {
    TensorConfig config = tensor_config_with_dtype_device((DType)dtype, device_get_default());
    return tensor_empty(shape, ndim, &config);
}

Tensor* tensor_zeros_auto(int* shape, int ndim, int dtype) {
    TensorConfig config = tensor_config_with_dtype_device((DType)dtype, device_get_default());
    return tensor_zeros(shape, ndim, &config);
}

Tensor* tensor_ones_auto(int* shape, int ndim, int dtype) {
    TensorConfig config = tensor_config_with_dtype_device((DType)dtype, device_get_default());
    return tensor_ones(shape, ndim, &config);
}

struct Linear* nn_linear_auto(int in_features, int out_features, DType dtype, bool use_bias) {
    extern Linear* nn_linear(int in_features, int out_features, DType dtype, DeviceType device,
                             bool use_bias);
    return nn_linear(in_features, out_features, dtype, device_get_default(), use_bias);
}

// ============================================================================
// Device-Specific Memory Management
// ============================================================================

void* device_alloc(size_t size, DeviceType device) {
    if (size == 0) {
        LOG_ERROR("Cannot allocate zero bytes");
        return NULL;
    }

    void* ptr = NULL;

    switch (device) {
    case DEVICE_CPU:
        ptr = CM_MALLOC(size);
        if (!ptr) {
            LOG_ERROR("Failed to allocate %zu bytes on CPU", size);
        }
        break;

    case DEVICE_CUDA:
        if (g_cuda_available && cudaMalloc) {
            int result = cudaMalloc(&ptr, size);
            if (result != 0 || !ptr) {
                LOG_ERROR("Failed to allocate %zu bytes on CUDA device", size);
                ptr = NULL;
            }
        } else {
            LOG_WARNING("CUDA not available, falling back to CPU");
            ptr = CM_MALLOC(size);
        }
        break;

    case DEVICE_METAL:
#ifdef __APPLE__
        // Metal allocation would go here
        // For now, fall back to CPU
        LOG_WARNING("Metal allocation not yet implemented, falling back to CPU");
        ptr = CM_MALLOC(size);
#else
        LOG_WARNING("Metal not available on this platform, falling back to CPU");
        ptr = CM_MALLOC(size);
#endif
        break;

    case DEVICE_ROCM:
        // ROCm allocation would go here
        LOG_WARNING("ROCm allocation not yet implemented, falling back to CPU");
        ptr = CM_MALLOC(size);
        break;

    case DEVICE_AUTO:
        return device_alloc(size, device_get_best_available());

    default:
        LOG_ERROR("Unknown device type: %d", device);
        ptr = CM_MALLOC(size);
        break;
    }

    return ptr;
}

void device_free(void* ptr, DeviceType device) {
    if (!ptr) {
        return;
    }

    switch (device) {
    case DEVICE_CPU:
        CM_FREE(ptr);
        break;

    case DEVICE_CUDA:
        if (g_cuda_available && cudaFree) {
            int result = cudaFree(ptr);
            if (result != 0) {
                LOG_ERROR("Failed to free CUDA memory");
            }
        } else {
            CM_FREE(ptr);
        }
        break;

    case DEVICE_METAL:
#ifdef __APPLE__
        // Metal free would go here
        CM_FREE(ptr);
#else
        CM_FREE(ptr);
#endif
        break;

    case DEVICE_ROCM:
        // ROCm free would go here
        CM_FREE(ptr);
        break;

    case DEVICE_AUTO:
        device_free(ptr, device_get_best_available());
        break;

    default:
        CM_FREE(ptr);
        break;
    }
}

void* device_alloc_default(size_t size) { return device_alloc(size, device_get_default()); }

void device_free_default(void* ptr) { device_free(ptr, device_get_default()); }

// ============================================================================
// Device Data Transfer
// ============================================================================

int device_copy(void* dst, const void* src, size_t size, DeviceType dst_device,
                DeviceType src_device) {
    if (!dst || !src || size == 0) {
        LOG_ERROR("Invalid parameters for device_copy");
        return -1;
    }

    // Same device - direct copy
    if (dst_device == src_device) {
        if (dst_device == DEVICE_CPU) {
            memcpy(dst, src, size);
            return 0;
        } else if (dst_device == DEVICE_CUDA && g_cuda_available && cudaMemcpy) {
            int result = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
            return (result == 0) ? 0 : -1;
        }
        // For other devices, fall through to cross-device copy
    }

    // Cross-device copy
    if (src_device == DEVICE_CPU && dst_device == DEVICE_CUDA) {
        if (g_cuda_available && cudaMemcpy) {
            int result = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
            return (result == 0) ? 0 : -1;
        } else {
            LOG_ERROR("CUDA not available for copy");
            return -1;
        }
    } else if (src_device == DEVICE_CUDA && dst_device == DEVICE_CPU) {
        if (g_cuda_available && cudaMemcpy) {
            int result = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
            return (result == 0) ? 0 : -1;
        } else {
            LOG_ERROR("CUDA not available for copy");
            return -1;
        }
    } else if (src_device == DEVICE_CPU) {
        // CPU to other device - copy to CPU first, then to device
        // For now, just copy to CPU buffer
        memcpy(dst, src, size);
        return 0;
    } else if (dst_device == DEVICE_CPU) {
        // Other device to CPU - copy from device to CPU
        // For now, just copy from CPU buffer
        memcpy(dst, src, size);
        return 0;
    } else {
        // Device to device - use staging buffer
        void* staging = CM_MALLOC(size);
        if (!staging) {
            LOG_ERROR("Failed to allocate staging buffer for device copy");
            return -1;
        }

        // Copy from source device to CPU
        int result1 = device_copy(staging, src, size, DEVICE_CPU, src_device);
        if (result1 != 0) {
            CM_FREE(staging);
            return -1;
        }

        // Copy from CPU to destination device
        int result2 = device_copy(dst, staging, size, dst_device, DEVICE_CPU);
        CM_FREE(staging);

        return (result2 == 0) ? 0 : -1;
    }
}

int device_copy_to_device(void* dst, const void* src, size_t size, DeviceType device) {
    return device_copy(dst, src, size, device, DEVICE_CPU);
}

int device_copy_from_device(void* dst, const void* src, size_t size, DeviceType device) {
    return device_copy(dst, src, size, DEVICE_CPU, device);
}

int device_move_tensor(Tensor* tensor, DeviceType device) {
    if (!tensor) {
        LOG_ERROR("Cannot move NULL tensor");
        return -1;
    }

    // If already on target device, nothing to do
    if (tensor->device == device) {
        return 0;
    }

    size_t data_size = tensor->numel * dtype_size(tensor->dtype);

    // Allocate new memory on target device
    void* new_data = device_alloc(data_size, device);
    if (!new_data) {
        LOG_ERROR("Failed to allocate memory on target device");
        return -1;
    }

    // Copy data from old device to new device
    int result = device_copy(new_data, tensor->data, data_size, device, tensor->device);
    if (result != 0) {
        device_free(new_data, device);
        LOG_ERROR("Failed to copy tensor data to target device");
        return -1;
    }

    // Free old memory
    if (tensor->owns_data && tensor->data) {
        device_free(tensor->data, tensor->device);
    }

    // Update tensor
    tensor->data      = new_data;
    tensor->device    = device;
    tensor->owns_data = true;

    LOG_DEBUG("Moved tensor to device: %s", device_get_name(device));
    return 0;
}

int device_move_tensor_to_default(Tensor* tensor) {
    return device_move_tensor(tensor, device_get_default());
}

int device_auto_load_tensor(Tensor* tensor) {
    if (!tensor) {
        LOG_ERROR("Cannot auto-load NULL tensor");
        return -1;
    }

    DeviceType best_device = device_get_best_available();
    return device_move_tensor(tensor, best_device);
}

// ============================================================================
// Device-Specific Computation
// ============================================================================

int device_set_compute_device(DeviceType device) {
    if (!g_device_initialized) {
        device_init();
    }

    g_current_device = device;
    LOG_DEBUG("Set compute device to: %s", device_get_name(device));
    return 0;
}

int device_synchronize(DeviceType device) {
    switch (device) {
    case DEVICE_CPU:
        // CPU is synchronous, nothing to do
        return 0;

    case DEVICE_CUDA:
        if (g_cuda_available && cudaDeviceSynchronize) {
            int result = cudaDeviceSynchronize();
            return (result == 0) ? 0 : -1;
        }
        return 0;

    case DEVICE_METAL:
    case DEVICE_ROCM:
        // Device-specific synchronization would go here
        // For now, assume synchronous
        return 0;

    case DEVICE_AUTO:
        return device_synchronize(device_get_best_available());

    default:
        return 0;
    }
}

int device_synchronize_default(void) { return device_synchronize(device_get_default()); }

// ============================================================================
// Device Information
// ============================================================================

int device_get_info(DeviceType device, DeviceInfo* info) {
    if (!info) {
        LOG_ERROR("DeviceInfo pointer is NULL");
        return -1;
    }

    info->type         = device;
    info->device_id    = 0;
    info->total_memory = 0;
    info->free_memory  = 0;
    info->name         = device_get_name(device);
    info->available    = false;

    switch (device) {
    case DEVICE_CPU:
        info->available = true;
        // CPU memory info would require system-specific calls
        info->total_memory = 0; // Unknown
        info->free_memory  = 0; // Unknown
        break;

    case DEVICE_CUDA:
        info->available = g_cuda_available;
        if (g_cuda_available && cudaMemGetInfo) {
            size_t free = 0, total = 0;
            if (cudaMemGetInfo(&free, &total) == 0) {
                info->free_memory  = free;
                info->total_memory = total;
            }
        }
        break;

    case DEVICE_METAL:
        info->available = g_metal_available;
        break;

    case DEVICE_ROCM:
        info->available = g_rocm_available;
        break;

    case DEVICE_AUTO:
        return device_get_info(device_get_best_available(), info);

    default:
        break;
    }

    return 0;
}

size_t device_get_total_memory(DeviceType device) {
    DeviceInfo info;
    if (device_get_info(device, &info) == 0) {
        return info.total_memory;
    }
    return 0;
}

size_t device_get_free_memory(DeviceType device) {
    DeviceInfo info;
    if (device_get_info(device, &info) == 0) {
        return info.free_memory;
    }
    return 0;
}
