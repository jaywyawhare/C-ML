/**
 * @file device.h
 * @brief Device management and automatic device detection
 *
 * Provides support for multiple compute devices:
 * - CPU (always available)
 * - CUDA (NVIDIA GPUs)
 * - Metal (Apple GPUs)
 * - ROCm (AMD GPUs)
 *
 * Features:
 * - Automatic device detection
 * - Automatic data loading to devices
 * - Device-specific memory allocation
 * - Device-specific computation dispatch
 */

#ifndef CML_CORE_DEVICE_H
#define CML_CORE_DEVICE_H

#include <stdbool.h>
#include <stddef.h>

// Forward declarations
struct Tensor;
// DType is defined in tensor/tensor.h - we'll use int in function signatures
// and cast appropriately in implementations

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Device types supported by C-ML
 */
typedef enum {
    DEVICE_CPU,   // CPU (always available)
    DEVICE_CUDA,  // NVIDIA CUDA GPUs
    DEVICE_METAL, // Apple Metal GPUs
    DEVICE_ROCM,  // AMD ROCm GPUs
    DEVICE_AUTO   // Auto-select best available device
} DeviceType;

// Forward declaration
struct Tensor;

/**
 * @brief Device information structure
 */
typedef struct {
    DeviceType type;
    int device_id;       // Device ID (for multi-device systems)
    size_t total_memory; // Total memory in bytes
    size_t free_memory;  // Free memory in bytes
    const char* name;    // Device name
    bool available;      // Is device available?
} DeviceInfo;

// ============================================================================
// Device Detection and Initialization
// ============================================================================

/**
 * @brief Initialize device management system
 *
 * Automatically detects all available devices and sets the default device
 * to the best available accelerator (CUDA > Metal > ROCm > Vulkan > OpenCL > oneAPI > CPU)
 */
void device_init(void);

/**
 * @brief Cleanup device management system
 *
 * Releases all device resources and closes device libraries
 */
void device_cleanup(void);

/**
 * @brief Detect all available devices
 *
 * @return true if detection successful, false otherwise
 */
bool device_detect_available(void);

/**
 * @brief Get the best available device
 *
 * Priority: CUDA > Metal > ROCm > Vulkan > OpenCL > oneAPI > CPU
 *
 * @return Best available DeviceType
 */
DeviceType device_get_best_available(void);

/**
 * @brief Check if CUDA is available
 */
bool device_cuda_available(void);

/**
 * @brief Check if Metal is available
 */
bool device_metal_available(void);

/**
 * @brief Check if ROCm is available
 */
bool device_rocm_available(void);

// ============================================================================
// Device Selection
// ============================================================================

/**
 * @brief Get the default device
 *
 * If DEVICE_AUTO is set, returns the best available device
 *
 * @return Default DeviceType
 */
DeviceType device_get_default(void);

/**
 * @brief Set the default device
 *
 * @param device DeviceType to set as default
 */
void device_set_default(DeviceType device);

/**
 * @brief Get the current active device
 *
 * @return Current DeviceType
 */
DeviceType device_get_current(void);

/**
 * @brief Set the current active device
 *
 * @param device DeviceType to set as current
 */
void device_set_current(DeviceType device);

/**
 * @brief Get device name as string
 *
 * @param device DeviceType
 * @return Device name string
 */
const char* device_get_name(DeviceType device);

/**
 * @brief Print device information
 */
void device_print_info(void);

// ============================================================================
// Device-Specific Memory Management
// ============================================================================

/**
 * @brief Allocate memory on a specific device
 *
 * @param size Size in bytes to allocate
 * @param device Target device
 * @return Pointer to allocated memory, or NULL on failure
 */
void* device_alloc(size_t size, DeviceType device);

/**
 * @brief Free memory on a specific device
 *
 * @param ptr Pointer to memory to free
 * @param device Device where memory was allocated
 */
void device_free(void* ptr, DeviceType device);

/**
 * @brief Allocate memory on default device
 *
 * @param size Size in bytes to allocate
 * @return Pointer to allocated memory, or NULL on failure
 */
void* device_alloc_default(size_t size);

/**
 * @brief Free memory on default device
 *
 * @param ptr Pointer to memory to free
 */
void device_free_default(void* ptr);

// ============================================================================
// Device Data Transfer
// ============================================================================

/**
 * @brief Copy data from source device to destination device
 *
 * @param dst Destination pointer
 * @param src Source pointer
 * @param size Size in bytes to copy
 * @param dst_device Destination device
 * @param src_device Source device
 * @return 0 on success, negative value on failure
 */
int device_copy(void* dst, const void* src, size_t size, DeviceType dst_device,
                DeviceType src_device);

/**
 * @brief Copy data to device (from CPU)
 *
 * @param dst Destination pointer on device
 * @param src Source pointer on CPU
 * @param size Size in bytes to copy
 * @param device Target device
 * @return 0 on success, negative value on failure
 */
int device_copy_to_device(void* dst, const void* src, size_t size, DeviceType device);

/**
 * @brief Copy data from device (to CPU)
 *
 * @param dst Destination pointer on CPU
 * @param src Source pointer on device
 * @param size Size in bytes to copy
 * @param device Source device
 * @return 0 on success, negative value on failure
 */
int device_copy_from_device(void* dst, const void* src, size_t size, DeviceType device);

/**
 * @brief Move tensor to a specific device
 *
 * Automatically allocates memory on target device, copies data, and frees old memory
 *
 * @param tensor Tensor to move
 * @param device Target device
 * @return 0 on success, negative value on failure
 */
int device_move_tensor(struct Tensor* tensor, DeviceType device);

/**
 * @brief Move tensor to default device
 *
 * @param tensor Tensor to move
 * @return 0 on success, negative value on failure
 */
int device_move_tensor_to_default(struct Tensor* tensor);

/**
 * @brief Auto-load tensor to best available device
 *
 * Detects best device and moves tensor to it
 *
 * @param tensor Tensor to auto-load
 * @return 0 on success, negative value on failure
 */
int device_auto_load_tensor(struct Tensor* tensor);

// ============================================================================
// Device-Specific Computation
// ============================================================================

/**
 * @brief Execute computation on a specific device
 *
 * This is a generic dispatch function that routes computations to device-specific
 * implementations. Operations should check tensor->device and dispatch accordingly.
 *
 * @param device Target device for computation
 * @return 0 on success, negative value on failure
 */
int device_set_compute_device(DeviceType device);

/**
 * @brief Synchronize device operations
 *
 * Waits for all operations on the specified device to complete
 *
 * @param device Device to synchronize
 * @return 0 on success, negative value on failure
 */
int device_synchronize(DeviceType device);

/**
 * @brief Synchronize default device
 *
 * @return 0 on success, negative value on failure
 */
int device_synchronize_default(void);

// ============================================================================
// Device Information
// ============================================================================

/**
 * @brief Get device information
 *
 * @param device DeviceType to query
 * @param info Output structure to fill with device information
 * @return 0 on success, negative value on failure
 */
int device_get_info(DeviceType device, DeviceInfo* info);

/**
 * @brief Get total memory on device
 *
 * @param device DeviceType to query
 * @return Total memory in bytes, or 0 on failure
 */
size_t device_get_total_memory(DeviceType device);

/**
 * @brief Get free memory on device
 *
 * @param device DeviceType to query
 * @return Free memory in bytes, or 0 on failure
 */
size_t device_get_free_memory(DeviceType device);

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Create tensor on default device (auto-detected)
 *
 * @param shape Shape array
 * @param ndim Number of dimensions
 * @param dtype Data type (DType enum from tensor/tensor.h)
 * @return New tensor on default device, or NULL on failure
 */
struct Tensor* tensor_empty_auto(int* shape, int ndim, int dtype);

/**
 * @brief Create zeros tensor on default device
 *
 * @param shape Shape array
 * @param ndim Number of dimensions
 * @param dtype Data type (DType enum from tensor/tensor.h)
 * @return New tensor on default device, or NULL on failure
 */
struct Tensor* tensor_zeros_auto(int* shape, int ndim, int dtype);

/**
 * @brief Create ones tensor on default device
 *
 * @param shape Shape array
 * @param ndim Number of dimensions
 * @param dtype Data type (DType enum from tensor/tensor.h)
 * @return New tensor on default device, or NULL on failure
 */
struct Tensor* tensor_ones_auto(int* shape, int ndim, int dtype);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_DEVICE_H
