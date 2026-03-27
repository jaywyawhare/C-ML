#ifndef CML_CORE_DEVICE_H
#define CML_CORE_DEVICE_H

#include <stdbool.h>
#include <stddef.h>

struct Tensor;

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DEVICE_CPU,     // CPU (always available)
    DEVICE_CUDA,    // NVIDIA CUDA GPUs
    DEVICE_METAL,   // Apple Metal GPUs
    DEVICE_ROCM,    // AMD ROCm GPUs
    DEVICE_OPENCL,  // OpenCL GPUs
    DEVICE_SIM_GPU, // Simulated GPU (CPU-backed, for testing multi-GPU without hardware)
    DEVICE_AUTO     // Auto-select best available device
} DeviceType;

struct Tensor;

typedef struct {
    DeviceType type;
    int device_id;       // Device ID (for multi-device systems)
    size_t total_memory; // Total memory in bytes
    size_t free_memory;  // Free memory in bytes
    const char* name;    // Device name
    bool available;      // Is device available?
} DeviceInfo;

/* Detects all available devices; sets default to best available
   (CUDA > Metal > ROCm > Vulkan > OpenCL > oneAPI > CPU) */
void device_init(void);
void device_cleanup(void);
bool device_detect_available(void);

/* Priority: CUDA > Metal > ROCm > Vulkan > OpenCL > oneAPI > CPU */
DeviceType device_get_best_available(void);

bool device_cuda_available(void);
int device_cuda_get_count(void);
bool device_metal_available(void);
bool device_rocm_available(void);
int device_rocm_get_count(void);

/* If DEVICE_AUTO is set, returns the best available device */
DeviceType device_get_default(void);
void device_set_default(DeviceType device);
DeviceType device_get_current(void);
void device_set_current(DeviceType device);
const char* device_get_name(DeviceType device);
void device_print_info(void);

void* device_alloc(size_t size, DeviceType device);
void device_free(void* ptr, DeviceType device);
void* device_alloc_default(size_t size);
void device_free_default(void* ptr);

int device_copy(void* dst, const void* src, size_t size, DeviceType dst_device,
                DeviceType src_device);
int device_copy_to_device(void* dst, const void* src, size_t size, DeviceType device);
int device_copy_from_device(void* dst, const void* src, size_t size, DeviceType device);

/* Allocates memory on target device, copies data, and frees old memory */
int device_move_tensor(struct Tensor* tensor, DeviceType device);
int device_move_tensor_to_default(struct Tensor* tensor);

struct Tensor* tensor_empty_auto(int* shape, int ndim, int dtype);
struct Tensor* tensor_zeros_auto(int* shape, int ndim, int dtype);
struct Tensor* tensor_ones_auto(int* shape, int ndim, int dtype);

/* Creates N virtual GPU devices backed by CPU memory.
   Allows testing multi-GPU code paths without real hardware. */
int device_sim_gpu_enable(int num_devices, size_t memory_per_device);
void device_sim_gpu_disable(void);
bool device_sim_gpu_available(void);
int device_sim_gpu_get_count(void);
int device_sim_gpu_set_device(int device_id);
int device_sim_gpu_get_device(void);
int device_sim_gpu_get_info(int device_id, DeviceInfo* info);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_DEVICE_H
