/**
 * @file vulkan_backend.h
 * @brief Vulkan backend for cross-platform GPU compute
 */

#ifndef CML_MLIR_BACKENDS_VULKAN_BACKEND_H
#define CML_MLIR_BACKENDS_VULKAN_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct Tensor;
typedef struct Tensor Tensor;

// Vulkan types (avoid requiring vulkan.h)
typedef void* VkInstance;
typedef void* VkPhysicalDevice;
typedef void* VkDevice;
typedef void* VkQueue;
typedef void* VkCommandPool;
typedef void* VkCommandBuffer;
typedef void* VkShaderModule;
typedef void* VkPipeline;
typedef void* VkPipelineLayout;
typedef void* VkDescriptorPool;
typedef void* VkDescriptorSet;
typedef void* VkDescriptorSetLayout;
typedef void* VkBuffer;
typedef uint64_t VkDeviceMemory;
typedef int32_t VkResult;

/**
 * @brief Vulkan backend context
 */
typedef struct CMLVulkanBackend {
    void* vulkan_lib; // libvulkan.so handle

    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    VkCommandPool command_pool;
    VkDescriptorPool descriptor_pool;

    uint32_t compute_queue_family;
    bool initialized;

    // Device info
    char device_name[256];
    size_t total_memory;
    uint32_t max_compute_work_group_count[3];
    uint32_t max_compute_work_group_size[3];

    // Function pointers (dynamically loaded)
    VkResult (*vkCreateInstance)(const void*, const void*, VkInstance*);
    void (*vkDestroyInstance)(VkInstance, const void*);
    VkResult (*vkEnumeratePhysicalDevices)(VkInstance, uint32_t*, VkPhysicalDevice*);
    void (*vkGetPhysicalDeviceProperties)(VkPhysicalDevice, void*);
    void (*vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice, uint32_t*, void*);
    VkResult (*vkCreateDevice)(VkPhysicalDevice, const void*, const void*, VkDevice*);
    void (*vkDestroyDevice)(VkDevice, const void*);
    void (*vkGetDeviceQueue)(VkDevice, uint32_t, uint32_t, VkQueue*);
    VkResult (*vkCreateShaderModule)(VkDevice, const void*, const void*, VkShaderModule*);
    void (*vkDestroyShaderModule)(VkDevice, VkShaderModule, const void*);
    VkResult (*vkCreateComputePipelines)(VkDevice, void*, uint32_t, const void*, const void*,
                                         VkPipeline*);
    void (*vkDestroyPipeline)(VkDevice, VkPipeline, const void*);
    VkResult (*vkCreatePipelineLayout)(VkDevice, const void*, const void*, VkPipelineLayout*);
    void (*vkDestroyPipelineLayout)(VkDevice, VkPipelineLayout, const void*);
    VkResult (*vkCreateDescriptorSetLayout)(VkDevice, const void*, const void*,
                                            VkDescriptorSetLayout*);
    void (*vkDestroyDescriptorSetLayout)(VkDevice, VkDescriptorSetLayout, const void*);
    VkResult (*vkCreateDescriptorPool)(VkDevice, const void*, const void*, VkDescriptorPool*);
    void (*vkDestroyDescriptorPool)(VkDevice, VkDescriptorPool, const void*);
    VkResult (*vkAllocateDescriptorSets)(VkDevice, const void*, VkDescriptorSet*);
    void (*vkUpdateDescriptorSets)(VkDevice, uint32_t, const void*, uint32_t, const void*);
    VkResult (*vkCreateCommandPool)(VkDevice, const void*, const void*, VkCommandPool*);
    void (*vkDestroyCommandPool)(VkDevice, VkCommandPool, const void*);
    VkResult (*vkAllocateCommandBuffers)(VkDevice, const void*, VkCommandBuffer*);
    VkResult (*vkBeginCommandBuffer)(VkCommandBuffer, const void*);
    VkResult (*vkEndCommandBuffer)(VkCommandBuffer);
    void (*vkCmdBindPipeline)(VkCommandBuffer, int, VkPipeline);
    void (*vkCmdBindDescriptorSets)(VkCommandBuffer, int, VkPipelineLayout, uint32_t, uint32_t,
                                    const VkDescriptorSet*, uint32_t, const uint32_t*);
    void (*vkCmdDispatch)(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
    VkResult (*vkQueueSubmit)(VkQueue, uint32_t, const void*, void*);
    VkResult (*vkQueueWaitIdle)(VkQueue);
    VkResult (*vkCreateBuffer)(VkDevice, const void*, const void*, VkBuffer*);
    void (*vkDestroyBuffer)(VkDevice, VkBuffer, const void*);
    VkResult (*vkAllocateMemory)(VkDevice, const void*, const void*, VkDeviceMemory*);
    void (*vkFreeMemory)(VkDevice, VkDeviceMemory, const void*);
    VkResult (*vkBindBufferMemory)(VkDevice, VkBuffer, VkDeviceMemory, uint64_t);
    VkResult (*vkMapMemory)(VkDevice, VkDeviceMemory, uint64_t, uint64_t, uint32_t, void**);
    void (*vkUnmapMemory)(VkDevice, VkDeviceMemory);
    void (*vkGetBufferMemoryRequirements)(VkDevice, VkBuffer, void*);
    void (*vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, void*);
} CMLVulkanBackend;

/**
 * @brief Vulkan compiled kernel (SPIR-V based)
 */
typedef struct CMLVulkanKernel {
    VkShaderModule shader;
    VkPipeline pipeline;
    VkPipelineLayout pipeline_layout;
    VkDescriptorSetLayout descriptor_layout;
    char* kernel_name;
    int num_bindings;
    uint32_t workgroup_size[3];
} CMLVulkanKernel;

// Backend lifecycle
bool cml_vulkan_available(void);
CMLVulkanBackend* cml_vulkan_backend_create(void);
int cml_vulkan_backend_init(CMLVulkanBackend* backend);
void cml_vulkan_backend_free(CMLVulkanBackend* backend);

// Kernel operations
CMLVulkanKernel* cml_vulkan_compile_spirv(CMLVulkanBackend* backend, const uint32_t* spirv_code,
                                          size_t spirv_size, const char* entry_point);
void cml_vulkan_kernel_free(CMLVulkanBackend* backend, CMLVulkanKernel* kernel);
int cml_vulkan_launch_kernel(CMLVulkanBackend* backend, CMLVulkanKernel* kernel, VkBuffer* buffers,
                             int num_buffers, uint32_t group_count_x, uint32_t group_count_y,
                             uint32_t group_count_z);
int cml_vulkan_synchronize(CMLVulkanBackend* backend);

// Memory operations
VkBuffer cml_vulkan_create_buffer(CMLVulkanBackend* backend, size_t size, VkDeviceMemory* memory);
void cml_vulkan_destroy_buffer(CMLVulkanBackend* backend, VkBuffer buffer, VkDeviceMemory memory);
int cml_vulkan_upload_data(CMLVulkanBackend* backend, VkDeviceMemory memory, const void* data,
                           size_t size);
int cml_vulkan_download_data(CMLVulkanBackend* backend, VkDeviceMemory memory, void* data,
                             size_t size);

// Tensor operations
int cml_vulkan_upload_tensor(CMLVulkanBackend* backend, Tensor* tensor);
int cml_vulkan_download_tensor(CMLVulkanBackend* backend, Tensor* tensor);

#ifdef __cplusplus
}
#endif

#endif // CML_MLIR_BACKENDS_VULKAN_BACKEND_H
