/**
 * @file device.c
 * @brief Automatic device detection and management implementation
 */

#include "backend/device.h"
#include "core/logging.h"
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

// ROCm detection function pointers (loaded dynamically)
typedef int (*hipMalloc_fn)(void**, size_t);
typedef int (*hipFree_fn)(void*);
typedef int (*hipMemcpy_fn)(void*, const void*, size_t, int);
typedef int (*hipDeviceSynchronize_fn)(void);
typedef int (*hipMemGetInfo_fn)(size_t*, size_t*);

static hipMalloc_fn hipMalloc                       = NULL;
static hipFree_fn hipFree                           = NULL;
static hipMemcpy_fn hipMemcpy                       = NULL;
static hipDeviceSynchronize_fn hipDeviceSynchronize = NULL;
static hipMemGetInfo_fn hipMemGetInfo               = NULL;
static void* g_rocm_lib                             = NULL;

#define hipMemcpyHostToDevice 1
#define hipMemcpyDeviceToHost 2
#define hipMemcpyDeviceToDevice 3

// Metal detection (macOS/iOS) - handled in check_metal_available()
// Metal uses Objective-C runtime, so we'll use a simpler approach

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
        (cudaRuntimeGetVersion_fn)(uintptr_t)CML_DLSYM(g_cuda_lib, "cudaRuntimeGetVersion");
    cudaGetDeviceCount =
        (cudaGetDeviceCount_fn)(uintptr_t)CML_DLSYM(g_cuda_lib, "cudaGetDeviceCount");
    cudaMalloc = (cudaMalloc_fn)(uintptr_t)CML_DLSYM(g_cuda_lib, "cudaMalloc");
    cudaFree   = (cudaFree_fn)(uintptr_t)CML_DLSYM(g_cuda_lib, "cudaFree");
    cudaMemcpy = (cudaMemcpy_fn)(uintptr_t)CML_DLSYM(g_cuda_lib, "cudaMemcpy");
    cudaDeviceSynchronize =
        (cudaDeviceSynchronize_fn)(uintptr_t)CML_DLSYM(g_cuda_lib, "cudaDeviceSynchronize");
    cudaMemGetInfo = (cudaMemGetInfo_fn)(uintptr_t)CML_DLSYM(g_cuda_lib, "cudaMemGetInfo");

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
 * @brief Try to load ROCm library dynamically
 */
static bool try_load_rocm(void) {
    const char* lib_paths[] = {
#ifdef __linux__
        "libamdhip64.so", "libhip_hcc.so", "libhip64.so",
#elif defined(__APPLE__)
        "libamdhip64.dylib", "libhip_hcc.dylib",
#elif defined(_WIN32)
        "amdhip64.dll", "hip64.dll",
#endif
        NULL};

    for (int i = 0; lib_paths[i] != NULL; i++) {
        g_rocm_lib = CML_DLOPEN(lib_paths[i], RTLD_LAZY);
        if (g_rocm_lib) {
            break;
        }
    }

    if (!g_rocm_lib) {
        return false;
    }

    // Load ROCm functions
    hipMalloc = (hipMalloc_fn)(uintptr_t)CML_DLSYM(g_rocm_lib, "hipMalloc");
    hipFree   = (hipFree_fn)(uintptr_t)CML_DLSYM(g_rocm_lib, "hipFree");
    hipMemcpy = (hipMemcpy_fn)(uintptr_t)CML_DLSYM(g_rocm_lib, "hipMemcpy");
    hipDeviceSynchronize =
        (hipDeviceSynchronize_fn)(uintptr_t)CML_DLSYM(g_rocm_lib, "hipDeviceSynchronize");
    hipMemGetInfo = (hipMemGetInfo_fn)(uintptr_t)CML_DLSYM(g_rocm_lib, "hipMemGetInfo");

    if (hipMalloc && hipFree && hipMemcpy) {
        LOG_INFO("ROCm library loaded successfully");
        return true;
    } else {
        LOG_WARNING("ROCm library loaded but some functions missing");
        CML_DLCLOSE(g_rocm_lib);
        g_rocm_lib = NULL;
        return false;
    }
}

/**
 * @brief Check if ROCm is available
 */
static bool check_rocm_available(void) {
    if (g_rocm_lib) {
        return true; // Already loaded
    }
    return try_load_rocm();
}

bool device_cuda_available(void) { return g_cuda_available; }

int device_cuda_get_count(void) {
    if (!g_cuda_available || !cudaGetDeviceCount)
        return 0;
    int count = 0;
    if (cudaGetDeviceCount(&count) == 0) {
        return count;
    }
    return 0;
}

bool device_metal_available(void) { return g_metal_available; }

bool device_rocm_available(void) { return g_rocm_available; }

int device_rocm_get_count(void) {
    // ROCm doesn't have a get device count loaded currently
    // Return 1 if ROCm is available (could be extended later)
    return g_rocm_available ? 1 : 0;
}

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
    // Initialize device system if not already done, but don't let it override our setting
    if (!g_device_initialized) {
        // Just do detection without setting defaults
        g_cuda_available     = check_cuda_available();
        g_metal_available    = check_metal_available();
        g_rocm_available     = check_rocm_available();
        g_device_initialized = true;
    }
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
    TensorConfig config = {.dtype      = (DType)dtype,
                           .device     = device_get_default(),
                           .has_dtype  = true,
                           .has_device = true};
    return tensor_empty(shape, ndim, &config);
}

Tensor* tensor_zeros_auto(int* shape, int ndim, int dtype) {
    TensorConfig config = {.dtype      = (DType)dtype,
                           .device     = device_get_default(),
                           .has_dtype  = true,
                           .has_device = true};
    return tensor_zeros(shape, ndim, &config);
}

Tensor* tensor_ones_auto(int* shape, int ndim, int dtype) {
    TensorConfig config = {.dtype      = (DType)dtype,
                           .device     = device_get_default(),
                           .has_dtype  = true,
                           .has_device = true};
    return tensor_ones(shape, ndim, &config);
}

void* device_alloc(size_t size, DeviceType device) {
    if (size == 0) {
        LOG_ERROR("Cannot allocate zero bytes");
        return NULL;
    }

    void* ptr = NULL;

    switch (device) {
    case DEVICE_CPU:
        ptr = malloc(size);
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
            ptr = malloc(size);
        }
        break;

    case DEVICE_METAL:
#ifdef __APPLE__
        // Metal on Apple Silicon uses unified memory architecture
        // Allocate aligned memory for better Metal performance
        // Metal prefers 256-byte alignment for optimal performance
        if (g_metal_available) {
            // Use posix_memalign for aligned allocation (Metal prefers 256-byte alignment)
            // On Apple Silicon, unified memory is automatically accessible by GPU
            size_t alignment = 256; // Metal preferred alignment
            if (size % alignment != 0) {
                size = ((size / alignment) + 1) * alignment; // Round up to alignment
            }

            int result = posix_memalign(&ptr, alignment, size);
            if (result != 0 || !ptr) {
                LOG_ERROR("Failed to allocate %zu bytes for Metal (aligned memory, error: %d)",
                          size, result);
                ptr = NULL;
            } else {
                LOG_DEBUG("Metal: allocated %zu bytes using aligned unified memory", size);
            }
        } else {
            LOG_WARNING("Metal not available, falling back to CPU");
            ptr = malloc(size);
        }
#else
        LOG_WARNING("Metal not available on this platform, falling back to CPU");
        ptr = malloc(size);
#endif
        break;

    case DEVICE_ROCM:
        if (g_rocm_available && hipMalloc) {
            int result = hipMalloc(&ptr, size);
            if (result != 0 || !ptr) {
                LOG_ERROR("Failed to allocate %zu bytes on ROCm device (error: %d)", size, result);
                ptr = NULL;
            } else {
                LOG_DEBUG("ROCm: allocated %zu bytes", size);
            }
        } else {
            if (!g_rocm_available) {
                LOG_WARNING("ROCm not available, falling back to CPU");
            } else {
                LOG_WARNING("ROCm library not loaded, falling back to CPU");
            }
            ptr = malloc(size);
        }
        break;

    case DEVICE_AUTO:
        return device_alloc(size, device_get_best_available());

    default:
        LOG_ERROR("Unknown device type: %d", device);
        ptr = malloc(size);
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
        free(ptr);
        break;

    case DEVICE_CUDA:
        if (g_cuda_available && cudaFree) {
            int result = cudaFree(ptr);
            if (result != 0) {
                LOG_ERROR("Failed to free CUDA memory");
            }
        } else {
            free(ptr);
        }
        break;

    case DEVICE_METAL:
#ifdef __APPLE__
        // Metal uses aligned unified memory (allocated with posix_memalign)
        // Use free() for posix_memalign-allocated memory
        if (ptr) {
            free(ptr); // posix_memalign requires free(), not free
            LOG_DEBUG("Metal: freed aligned unified memory");
        }
#else
        free(ptr);
#endif
        break;

    case DEVICE_ROCM:
        if (g_rocm_available && hipFree) {
            int result = hipFree(ptr);
            if (result != 0) {
                LOG_ERROR("Failed to free ROCm memory (error: %d)", result);
            } else {
                LOG_DEBUG("ROCm: freed memory");
            }
        } else {
            free(ptr);
        }
        break;

    case DEVICE_AUTO:
        device_free(ptr, device_get_best_available());
        break;

    default:
        free(ptr);
        break;
    }
}

void* device_alloc_default(size_t size) { return device_alloc(size, device_get_default()); }

void device_free_default(void* ptr) { device_free(ptr, device_get_default()); }

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
        } else if (dst_device == DEVICE_ROCM && g_rocm_available && hipMemcpy) {
            int result = hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
            return (result == 0) ? 0 : -1;
        } else if (dst_device == DEVICE_METAL) {
            // Metal uses unified memory, so we can use memcpy
            memcpy(dst, src, size);
            return 0;
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
    } else if (src_device == DEVICE_CPU && dst_device == DEVICE_ROCM) {
        if (g_rocm_available && hipMemcpy) {
            int result = hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
            return (result == 0) ? 0 : -1;
        } else {
            LOG_ERROR("ROCm not available for copy");
            return -1;
        }
    } else if (src_device == DEVICE_ROCM && dst_device == DEVICE_CPU) {
        if (g_rocm_available && hipMemcpy) {
            int result = hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
            return (result == 0) ? 0 : -1;
        } else {
            LOG_ERROR("ROCm not available for copy");
            return -1;
        }
    } else if (src_device == DEVICE_CPU && dst_device == DEVICE_METAL) {
        // Metal uses unified memory, so we can use memcpy
        memcpy(dst, src, size);
        return 0;
    } else if (src_device == DEVICE_METAL && dst_device == DEVICE_CPU) {
        // Metal uses unified memory, so we can use memcpy
        memcpy(dst, src, size);
        return 0;
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
        void* staging = malloc(size);
        if (!staging) {
            LOG_ERROR("Failed to allocate staging buffer for device copy");
            return -1;
        }

        // Copy from source device to CPU
        int result1 = device_copy(staging, src, size, DEVICE_CPU, src_device);
        if (result1 != 0) {
            free(staging);
            return -1;
        }

        // Copy from CPU to destination device
        int result2 = device_copy(dst, staging, size, dst_device, DEVICE_CPU);
        free(staging);

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

    size_t data_size = tensor->numel * cml_dtype_size(tensor->dtype);

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
