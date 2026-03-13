/**
 * @file vulkan_backend.h
 * @brief Vulkan compute backend via dynamic loading
 *
 * Provides GPU compute execution through the Vulkan API without compile-time
 * dependencies. All Vulkan functions are loaded at runtime via dlopen/dlsym.
 */

#ifndef CML_GPU_VULKAN_BACKEND_H
#define CML_GPU_VULKAN_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
struct Tensor;
typedef struct Tensor Tensor;
struct CMLGraph;
typedef struct CMLGraph* CMLGraph_t;

/* ── Vulkan type aliases (avoid requiring vulkan.h) ── */

typedef uint32_t VkFlags;
typedef uint32_t VkBool32;
typedef uint64_t VkDeviceSize;
typedef void*    VkInstance;
typedef void*    VkPhysicalDevice;
typedef void*    VkDevice;
typedef void*    VkQueue;
typedef void*    VkCommandPool;
typedef void*    VkCommandBuffer;
typedef void*    VkFence;
typedef void*    VkBuffer;
typedef void*    VkDeviceMemory;
typedef void*    VkShaderModule;
typedef void*    VkPipelineLayout;
typedef void*    VkPipeline;
typedef void*    VkDescriptorSetLayout;
typedef void*    VkDescriptorPool;
typedef void*    VkDescriptorSet;
typedef int32_t  VkResult;

#define VK_SUCCESS 0
#define VK_MAX_BUFFERS_PER_KERNEL 16

/* ── Vulkan buffer ── */

typedef struct CMLVulkanBuffer {
    VkBuffer       buffer;
    VkDeviceMemory memory;
    VkDeviceSize   size;
    bool           is_device_local;  /* true = GPU-only, false = host-visible staging */
    void*          mapped;           /* non-NULL if persistently mapped (staging only) */
} CMLVulkanBuffer;

/* ── Vulkan compiled kernel ── */

typedef struct CMLVulkanKernel {
    VkShaderModule        shader_module;
    VkPipelineLayout      pipeline_layout;
    VkPipeline            pipeline;
    VkDescriptorSetLayout desc_layout;
    VkDescriptorPool      desc_pool;
    VkDescriptorSet       desc_set;
    char*                 name;
    int                   num_buffers;
} CMLVulkanKernel;

/* ── Vulkan backend context ── */

typedef struct CMLVulkanBackend {
    void* vulkan_lib;  /* libvulkan.so.1 handle */
    bool  initialized;

    /* Vulkan objects */
    VkInstance       instance;
    VkPhysicalDevice physical_device;
    VkDevice         device;
    VkQueue          compute_queue;
    uint32_t         compute_queue_family;
    VkCommandPool    command_pool;

    /* Device info */
    char     device_name[256];
    uint32_t api_version;
    size_t   total_memory;
    uint32_t max_compute_work_group_count[3];
    uint32_t max_compute_work_group_size[3];
    uint32_t max_compute_work_group_invocations;
    uint32_t memory_type_device_local;
    uint32_t memory_type_host_visible;

    /* ── Vulkan function pointers (loaded via dlsym) ── */

    /* Instance */
    VkResult (*vkCreateInstance)(const void* pCreateInfo, const void* pAllocator,
                                 VkInstance* pInstance);
    void     (*vkDestroyInstance)(VkInstance instance, const void* pAllocator);

    /* Physical device */
    VkResult (*vkEnumeratePhysicalDevices)(VkInstance instance, uint32_t* pCount,
                                           VkPhysicalDevice* pDevices);
    void     (*vkGetPhysicalDeviceProperties)(VkPhysicalDevice dev, void* pProps);
    void     (*vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice dev, void* pProps);
    void     (*vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice dev, uint32_t* pCount,
                                                          void* pProps);

    /* Logical device */
    VkResult (*vkCreateDevice)(VkPhysicalDevice phys, const void* pCreateInfo,
                                const void* pAllocator, VkDevice* pDevice);
    void     (*vkDestroyDevice)(VkDevice device, const void* pAllocator);
    void     (*vkGetDeviceQueue)(VkDevice device, uint32_t family, uint32_t index,
                                  VkQueue* pQueue);

    /* Memory */
    VkResult (*vkAllocateMemory)(VkDevice device, const void* pAllocInfo,
                                  const void* pAllocator, VkDeviceMemory* pMemory);
    void     (*vkFreeMemory)(VkDevice device, VkDeviceMemory memory, const void* pAllocator);
    VkResult (*vkMapMemory)(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
                             VkDeviceSize size, VkFlags flags, void** ppData);
    void     (*vkUnmapMemory)(VkDevice device, VkDeviceMemory memory);

    /* Buffers */
    VkResult (*vkCreateBuffer)(VkDevice device, const void* pCreateInfo,
                                const void* pAllocator, VkBuffer* pBuffer);
    void     (*vkDestroyBuffer)(VkDevice device, VkBuffer buffer, const void* pAllocator);
    void     (*vkGetBufferMemoryRequirements)(VkDevice device, VkBuffer buffer, void* pReqs);
    VkResult (*vkBindBufferMemory)(VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                                    VkDeviceSize offset);

    /* Command buffers */
    VkResult (*vkCreateCommandPool)(VkDevice device, const void* pCreateInfo,
                                     const void* pAllocator, VkCommandPool* pPool);
    void     (*vkDestroyCommandPool)(VkDevice device, VkCommandPool pool,
                                      const void* pAllocator);
    VkResult (*vkAllocateCommandBuffers)(VkDevice device, const void* pAllocInfo,
                                          VkCommandBuffer* pBuffers);
    void     (*vkFreeCommandBuffers)(VkDevice device, VkCommandPool pool, uint32_t count,
                                      const VkCommandBuffer* pBuffers);
    VkResult (*vkBeginCommandBuffer)(VkCommandBuffer buf, const void* pBeginInfo);
    VkResult (*vkEndCommandBuffer)(VkCommandBuffer buf);
    VkResult (*vkResetCommandBuffer)(VkCommandBuffer buf, VkFlags flags);

    /* Queue submit */
    VkResult (*vkQueueSubmit)(VkQueue queue, uint32_t submitCount, const void* pSubmits,
                               VkFence fence);
    VkResult (*vkQueueWaitIdle)(VkQueue queue);

    /* Fences */
    VkResult (*vkCreateFence)(VkDevice device, const void* pCreateInfo,
                               const void* pAllocator, VkFence* pFence);
    void     (*vkDestroyFence)(VkDevice device, VkFence fence, const void* pAllocator);
    VkResult (*vkWaitForFences)(VkDevice device, uint32_t count, const VkFence* pFences,
                                 VkBool32 waitAll, uint64_t timeout);
    VkResult (*vkResetFences)(VkDevice device, uint32_t count, const VkFence* pFences);

    /* Shader / pipeline */
    VkResult (*vkCreateShaderModule)(VkDevice device, const void* pCreateInfo,
                                      const void* pAllocator, VkShaderModule* pModule);
    void     (*vkDestroyShaderModule)(VkDevice device, VkShaderModule module,
                                       const void* pAllocator);
    VkResult (*vkCreateComputePipelines)(VkDevice device, void* pipelineCache, uint32_t count,
                                          const void* pCreateInfos, const void* pAllocator,
                                          VkPipeline* pPipelines);
    void     (*vkDestroyPipeline)(VkDevice device, VkPipeline pipeline, const void* pAllocator);
    VkResult (*vkCreatePipelineLayout)(VkDevice device, const void* pCreateInfo,
                                        const void* pAllocator, VkPipelineLayout* pLayout);
    void     (*vkDestroyPipelineLayout)(VkDevice device, VkPipelineLayout layout,
                                         const void* pAllocator);

    /* Descriptors */
    VkResult (*vkCreateDescriptorSetLayout)(VkDevice device, const void* pCreateInfo,
                                              const void* pAllocator,
                                              VkDescriptorSetLayout* pLayout);
    void     (*vkDestroyDescriptorSetLayout)(VkDevice device, VkDescriptorSetLayout layout,
                                               const void* pAllocator);
    VkResult (*vkCreateDescriptorPool)(VkDevice device, const void* pCreateInfo,
                                        const void* pAllocator, VkDescriptorPool* pPool);
    void     (*vkDestroyDescriptorPool)(VkDevice device, VkDescriptorPool pool,
                                         const void* pAllocator);
    VkResult (*vkAllocateDescriptorSets)(VkDevice device, const void* pAllocInfo,
                                          VkDescriptorSet* pSets);
    void     (*vkUpdateDescriptorSets)(VkDevice device, uint32_t writeCount,
                                        const void* pWrites, uint32_t copyCount,
                                        const void* pCopies);

    /* Dispatch */
    void (*vkCmdBindPipeline)(VkCommandBuffer buf, int bindPoint, VkPipeline pipeline);
    void (*vkCmdBindDescriptorSets)(VkCommandBuffer buf, int bindPoint,
                                     VkPipelineLayout layout, uint32_t firstSet,
                                     uint32_t count, const VkDescriptorSet* pSets,
                                     uint32_t dynOffsetCount, const uint32_t* pDynOffsets);
    void (*vkCmdDispatch)(VkCommandBuffer buf, uint32_t gx, uint32_t gy, uint32_t gz);
    void (*vkCmdCopyBuffer)(VkCommandBuffer buf, VkBuffer src, VkBuffer dst,
                             uint32_t regionCount, const void* pRegions);
    void (*vkCmdPipelineBarrier)(VkCommandBuffer buf, VkFlags srcStage, VkFlags dstStage,
                                  VkFlags depFlags, uint32_t memBarrierCount,
                                  const void* pMemBarriers, uint32_t bufBarrierCount,
                                  const void* pBufBarriers, uint32_t imgBarrierCount,
                                  const void* pImgBarriers);

    VkResult (*vkDeviceWaitIdle)(VkDevice device);
} CMLVulkanBackend;

/* ── Backend lifecycle ── */

bool              cml_vulkan_available(void);
CMLVulkanBackend* cml_vulkan_backend_create(void);
int               cml_vulkan_backend_init(CMLVulkanBackend* backend);
void              cml_vulkan_backend_free(CMLVulkanBackend* backend);

/* ── Buffer management ── */

CMLVulkanBuffer* cml_vulkan_buffer_create(CMLVulkanBackend* backend, VkDeviceSize size,
                                           bool device_local);
void             cml_vulkan_buffer_free(CMLVulkanBackend* backend, CMLVulkanBuffer* buf);
int              cml_vulkan_buffer_upload(CMLVulkanBackend* backend, CMLVulkanBuffer* dst,
                                          const void* src, size_t size);
int              cml_vulkan_buffer_download(CMLVulkanBackend* backend, CMLVulkanBuffer* src,
                                             void* dst, size_t size);

/* ── Kernel management ── */

CMLVulkanKernel* cml_vulkan_kernel_create(CMLVulkanBackend* backend, const uint32_t* spirv,
                                            size_t spirv_size, const char* entry_point,
                                            int num_buffers);
void             cml_vulkan_kernel_free(CMLVulkanBackend* backend, CMLVulkanKernel* kernel);
int              cml_vulkan_kernel_bind_buffer(CMLVulkanBackend* backend,
                                                CMLVulkanKernel* kernel, int binding,
                                                CMLVulkanBuffer* buffer);
int              cml_vulkan_kernel_dispatch(CMLVulkanBackend* backend,
                                             CMLVulkanKernel* kernel,
                                             uint32_t gx, uint32_t gy, uint32_t gz);

/* ── Graph execution ── */

int cml_vulkan_execute_graph(CMLVulkanBackend* backend, CMLGraph_t ir);
int cml_vulkan_synchronize(CMLVulkanBackend* backend);

#ifdef __cplusplus
}
#endif

#endif /* CML_GPU_VULKAN_BACKEND_H */
