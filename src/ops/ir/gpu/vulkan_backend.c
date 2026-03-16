/**
 * @file vulkan_backend.c
 * @brief Vulkan compute backend implementation
 *
 * All Vulkan API calls loaded at runtime via dlopen("libvulkan.so.1").
 * No compile-time Vulkan SDK dependency.
 */

#include "ops/ir/gpu/vulkan_backend.h"
#include "ops/ir/gpu/spirv_codegen.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/ir/schedule.h"
#include "tensor/tensor.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __linux__
#include <dlfcn.h>
#define VULKAN_LIB_NAME "libvulkan.so.1"
#elif defined(__APPLE__)
#include <dlfcn.h>
#define VULKAN_LIB_NAME "libvulkan.1.dylib"
#else
#define VULKAN_LIB_NAME NULL
#endif


/* VkApplicationInfo */
typedef struct {
    uint32_t sType;        /* VK_STRUCTURE_TYPE_APPLICATION_INFO = 0 */
    const void* pNext;
    const char* pApplicationName;
    uint32_t applicationVersion;
    const char* pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
} VkApplicationInfo_t;

/* VkInstanceCreateInfo */
typedef struct {
    uint32_t sType;        /* VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1 */
    const void* pNext;
    VkFlags flags;
    const VkApplicationInfo_t* pApplicationInfo;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
} VkInstanceCreateInfo_t;

/* VkDeviceQueueCreateInfo */
typedef struct {
    uint32_t sType;        /* VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2 */
    const void* pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
    uint32_t queueCount;
    const float* pQueuePriorities;
} VkDeviceQueueCreateInfo_t;

/* VkDeviceCreateInfo */
typedef struct {
    uint32_t sType;        /* VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3 */
    const void* pNext;
    VkFlags flags;
    uint32_t queueCreateInfoCount;
    const VkDeviceQueueCreateInfo_t* pQueueCreateInfos;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
    const void* pEnabledFeatures;
} VkDeviceCreateInfo_t;

/* VkPhysicalDeviceProperties (partial) */
typedef struct {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t deviceType;
    char     deviceName[256];
    uint8_t  pipelineCacheUUID[16];
    /* VkPhysicalDeviceLimits (504 bytes) + VkPhysicalDeviceSparseProperties (20 bytes) */
    uint8_t  _limits_and_sparse[1024]; /* padding for limits + sparse properties */
} VkPhysicalDeviceProperties_t;

/* VkQueueFamilyProperties */
typedef struct {
    VkFlags  queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    uint32_t minImageTransferGranularity[3];
} VkQueueFamilyProperties_t;

#define VK_QUEUE_COMPUTE_BIT 0x00000002

/* VkMemoryType */
typedef struct {
    VkFlags  propertyFlags;
    uint32_t heapIndex;
} VkMemoryType_t;

/* VkMemoryHeap */
typedef struct {
    VkDeviceSize size;
    VkFlags      flags;
} VkMemoryHeap_t;

/* VkPhysicalDeviceMemoryProperties */
typedef struct {
    uint32_t      memoryTypeCount;
    VkMemoryType_t memoryTypes[32];
    uint32_t      memoryHeapCount;
    VkMemoryHeap_t memoryHeaps[16];
} VkPhysicalDeviceMemoryProperties_t;

#define VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT  0x01
#define VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT  0x02
#define VK_MEMORY_PROPERTY_HOST_COHERENT_BIT 0x04

/* VkBufferCreateInfo */
typedef struct {
    uint32_t     sType;   /* VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12 */
    const void*  pNext;
    VkFlags      flags;
    VkDeviceSize size;
    VkFlags      usage;
    uint32_t     sharingMode;
    uint32_t     queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
} VkBufferCreateInfo_t;

#define VK_BUFFER_USAGE_TRANSFER_SRC_BIT  0x01
#define VK_BUFFER_USAGE_TRANSFER_DST_BIT  0x02
#define VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 0x20

/* VkMemoryRequirements */
typedef struct {
    VkDeviceSize size;
    VkDeviceSize alignment;
    uint32_t     memoryTypeBits;
} VkMemoryRequirements_t;

/* VkMemoryAllocateInfo */
typedef struct {
    uint32_t     sType;   /* VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5 */
    const void*  pNext;
    VkDeviceSize allocationSize;
    uint32_t     memoryTypeIndex;
} VkMemoryAllocateInfo_t;

/* VkCommandPoolCreateInfo */
typedef struct {
    uint32_t    sType;    /* VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39 */
    const void* pNext;
    VkFlags     flags;
    uint32_t    queueFamilyIndex;
} VkCommandPoolCreateInfo_t;

#define VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT 0x02

/* VkCommandBufferAllocateInfo */
typedef struct {
    uint32_t      sType;  /* VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40 */
    const void*   pNext;
    VkCommandPool commandPool;
    uint32_t      level;  /* VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0 */
    uint32_t      commandBufferCount;
} VkCommandBufferAllocateInfo_t;

/* VkCommandBufferBeginInfo */
typedef struct {
    uint32_t    sType;    /* VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42 */
    const void* pNext;
    VkFlags     flags;
    const void* pInheritanceInfo;
} VkCommandBufferBeginInfo_t;

#define VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT 0x01

/* VkFenceCreateInfo */
typedef struct {
    uint32_t    sType;    /* VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8 */
    const void* pNext;
    VkFlags     flags;
} VkFenceCreateInfo_t;

/* VkSubmitInfo */
typedef struct {
    uint32_t           sType;  /* VK_STRUCTURE_TYPE_SUBMIT_INFO = 4 */
    const void*        pNext;
    uint32_t           waitSemaphoreCount;
    const void*        pWaitSemaphores;
    const VkFlags*     pWaitDstStageMask;
    uint32_t           commandBufferCount;
    const VkCommandBuffer* pCommandBuffers;
    uint32_t           signalSemaphoreCount;
    const void*        pSignalSemaphores;
} VkSubmitInfo_t;

/* VkShaderModuleCreateInfo */
typedef struct {
    uint32_t    sType;    /* VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16 */
    const void* pNext;
    VkFlags     flags;
    size_t      codeSize;
    const uint32_t* pCode;
} VkShaderModuleCreateInfo_t;

/* VkDescriptorSetLayoutBinding */
typedef struct {
    uint32_t binding;
    uint32_t descriptorType;
    uint32_t descriptorCount;
    VkFlags  stageFlags;
    const void* pImmutableSamplers;
} VkDescriptorSetLayoutBinding_t;

#define VK_DESCRIPTOR_TYPE_STORAGE_BUFFER 7
#define VK_SHADER_STAGE_COMPUTE_BIT       0x20

/* VkDescriptorSetLayoutCreateInfo */
typedef struct {
    uint32_t sType;       /* VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32 */
    const void* pNext;
    VkFlags flags;
    uint32_t bindingCount;
    const VkDescriptorSetLayoutBinding_t* pBindings;
} VkDescriptorSetLayoutCreateInfo_t;

/* VkPipelineLayoutCreateInfo */
typedef struct {
    uint32_t sType;       /* VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30 */
    const void* pNext;
    VkFlags flags;
    uint32_t setLayoutCount;
    const VkDescriptorSetLayout* pSetLayouts;
    uint32_t pushConstantRangeCount;
    const void* pPushConstantRanges;
} VkPipelineLayoutCreateInfo_t;

/* VkPipelineShaderStageCreateInfo */
typedef struct {
    uint32_t sType;       /* VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18 */
    const void* pNext;
    VkFlags flags;
    uint32_t stage;       /* VK_SHADER_STAGE_COMPUTE_BIT */
    VkShaderModule module;
    const char* pName;
    const void* pSpecializationInfo;
} VkPipelineShaderStageCreateInfo_t;

/* VkComputePipelineCreateInfo */
typedef struct {
    uint32_t sType;       /* VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29 */
    const void* pNext;
    VkFlags flags;
    VkPipelineShaderStageCreateInfo_t stage;
    VkPipelineLayout layout;
    VkPipeline basePipelineHandle;
    int32_t basePipelineIndex;
} VkComputePipelineCreateInfo_t;

/* VkDescriptorPoolSize */
typedef struct {
    uint32_t type;
    uint32_t descriptorCount;
} VkDescriptorPoolSize_t;

/* VkDescriptorPoolCreateInfo */
typedef struct {
    uint32_t sType;       /* VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33 */
    const void* pNext;
    VkFlags flags;
    uint32_t maxSets;
    uint32_t poolSizeCount;
    const VkDescriptorPoolSize_t* pPoolSizes;
} VkDescriptorPoolCreateInfo_t;

/* VkDescriptorSetAllocateInfo */
typedef struct {
    uint32_t sType;       /* VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34 */
    const void* pNext;
    VkDescriptorPool descriptorPool;
    uint32_t descriptorSetCount;
    const VkDescriptorSetLayout* pSetLayouts;
} VkDescriptorSetAllocateInfo_t;

/* VkDescriptorBufferInfo */
typedef struct {
    VkBuffer     buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
} VkDescriptorBufferInfo_t;

/* VkWriteDescriptorSet */
typedef struct {
    uint32_t sType;       /* VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35 */
    const void* pNext;
    VkDescriptorSet dstSet;
    uint32_t dstBinding;
    uint32_t dstArrayElement;
    uint32_t descriptorCount;
    uint32_t descriptorType;
    const void* pImageInfo;
    const VkDescriptorBufferInfo_t* pBufferInfo;
    const void* pTexelBufferView;
} VkWriteDescriptorSet_t;

/* VkBufferCopy */
typedef struct {
    VkDeviceSize srcOffset;
    VkDeviceSize dstOffset;
    VkDeviceSize size;
} VkBufferCopy_t;

/* VkMemoryBarrier */
typedef struct {
    uint32_t    sType;    /* VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46 */
    const void* pNext;
    VkFlags     srcAccessMask;
    VkFlags     dstAccessMask;
} VkMemoryBarrier_t;

#define VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT 0x00000800
#define VK_PIPELINE_STAGE_TRANSFER_BIT       0x00001000
#define VK_ACCESS_SHADER_WRITE_BIT           0x00000040
#define VK_ACCESS_SHADER_READ_BIT            0x00000020
#define VK_ACCESS_TRANSFER_WRITE_BIT         0x00000100
#define VK_ACCESS_TRANSFER_READ_BIT          0x00000080
#define VK_PIPELINE_BIND_POINT_COMPUTE       1


static void* vk_load_library(const char* name) {
    if (!name) return NULL;
    void* lib = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        LOG_DEBUG("Failed to load %s: %s", name, dlerror());
    }
    return lib;
}

static void* vk_get_symbol(void* lib, const char* name) {
    return dlsym(lib, name);
}

static void vk_unload_library(void* lib) {
    if (lib) dlclose(lib);
}


#define VK_LOAD_FUNC(name) \
    backend->name = vk_get_symbol(backend->vulkan_lib, #name); \
    if (!backend->name) { \
        LOG_ERROR("Failed to load Vulkan function: %s", #name); \
        return -1; \
    }

static int load_vulkan_functions(CMLVulkanBackend* backend) {
    VK_LOAD_FUNC(vkCreateInstance);
    VK_LOAD_FUNC(vkDestroyInstance);
    VK_LOAD_FUNC(vkEnumeratePhysicalDevices);
    VK_LOAD_FUNC(vkGetPhysicalDeviceProperties);
    VK_LOAD_FUNC(vkGetPhysicalDeviceMemoryProperties);
    VK_LOAD_FUNC(vkGetPhysicalDeviceQueueFamilyProperties);
    VK_LOAD_FUNC(vkCreateDevice);
    VK_LOAD_FUNC(vkDestroyDevice);
    VK_LOAD_FUNC(vkGetDeviceQueue);
    VK_LOAD_FUNC(vkAllocateMemory);
    VK_LOAD_FUNC(vkFreeMemory);
    VK_LOAD_FUNC(vkMapMemory);
    VK_LOAD_FUNC(vkUnmapMemory);
    VK_LOAD_FUNC(vkCreateBuffer);
    VK_LOAD_FUNC(vkDestroyBuffer);
    VK_LOAD_FUNC(vkGetBufferMemoryRequirements);
    VK_LOAD_FUNC(vkBindBufferMemory);
    VK_LOAD_FUNC(vkCreateCommandPool);
    VK_LOAD_FUNC(vkDestroyCommandPool);
    VK_LOAD_FUNC(vkAllocateCommandBuffers);
    VK_LOAD_FUNC(vkFreeCommandBuffers);
    VK_LOAD_FUNC(vkBeginCommandBuffer);
    VK_LOAD_FUNC(vkEndCommandBuffer);
    VK_LOAD_FUNC(vkResetCommandBuffer);
    VK_LOAD_FUNC(vkQueueSubmit);
    VK_LOAD_FUNC(vkQueueWaitIdle);
    VK_LOAD_FUNC(vkCreateFence);
    VK_LOAD_FUNC(vkDestroyFence);
    VK_LOAD_FUNC(vkWaitForFences);
    VK_LOAD_FUNC(vkResetFences);
    VK_LOAD_FUNC(vkCreateShaderModule);
    VK_LOAD_FUNC(vkDestroyShaderModule);
    VK_LOAD_FUNC(vkCreateComputePipelines);
    VK_LOAD_FUNC(vkDestroyPipeline);
    VK_LOAD_FUNC(vkCreatePipelineLayout);
    VK_LOAD_FUNC(vkDestroyPipelineLayout);
    VK_LOAD_FUNC(vkCreateDescriptorSetLayout);
    VK_LOAD_FUNC(vkDestroyDescriptorSetLayout);
    VK_LOAD_FUNC(vkCreateDescriptorPool);
    VK_LOAD_FUNC(vkDestroyDescriptorPool);
    VK_LOAD_FUNC(vkAllocateDescriptorSets);
    VK_LOAD_FUNC(vkUpdateDescriptorSets);
    VK_LOAD_FUNC(vkCmdBindPipeline);
    VK_LOAD_FUNC(vkCmdBindDescriptorSets);
    VK_LOAD_FUNC(vkCmdDispatch);
    VK_LOAD_FUNC(vkCmdCopyBuffer);
    VK_LOAD_FUNC(vkCmdPipelineBarrier);
    VK_LOAD_FUNC(vkDeviceWaitIdle);
    return 0;
}

#undef VK_LOAD_FUNC


bool cml_vulkan_available(void) {
#ifndef __linux__
    /* Currently only Linux is supported for Vulkan dynamic loading */
    (void)VULKAN_LIB_NAME;
    return false;
#else
    if (!VULKAN_LIB_NAME) return false;
    void* lib = vk_load_library(VULKAN_LIB_NAME);
    if (!lib) return false;
    void* fn = vk_get_symbol(lib, "vkCreateInstance");
    vk_unload_library(lib);
    return fn != NULL;
#endif
}


CMLVulkanBackend* cml_vulkan_backend_create(void) {
    CMLVulkanBackend* backend = (CMLVulkanBackend*)calloc(1, sizeof(CMLVulkanBackend));
    if (!backend) {
        LOG_ERROR("Failed to allocate Vulkan backend");
        return NULL;
    }
    return backend;
}

static uint32_t find_memory_type(VkPhysicalDeviceMemoryProperties_t* mem_props,
                                  uint32_t type_bits, VkFlags required_flags) {
    if (!mem_props) return UINT32_MAX;
    for (uint32_t i = 0; i < mem_props->memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mem_props->memoryTypes[i].propertyFlags & required_flags) == required_flags) {
            return i;
        }
    }
    return UINT32_MAX;
}

int cml_vulkan_backend_init(CMLVulkanBackend* backend) {
    if (!backend) return -1;
    if (backend->initialized) return 0;

    backend->vulkan_lib = vk_load_library(VULKAN_LIB_NAME);
    if (!backend->vulkan_lib) {
        LOG_ERROR("Failed to load Vulkan library");
        return -1;
    }

    if (load_vulkan_functions(backend) != 0) {
        vk_unload_library(backend->vulkan_lib);
        backend->vulkan_lib = NULL;
        return -1;
    }

    VkApplicationInfo_t app_info = {0};
    app_info.sType = 0; /* VK_STRUCTURE_TYPE_APPLICATION_INFO */
    app_info.pApplicationName = "C-ML";
    app_info.applicationVersion = 1;
    app_info.pEngineName = "C-ML";
    app_info.engineVersion = 1;
    app_info.apiVersion = (1u << 22) | (2u << 12); /* VK_API_VERSION_1_2 */

    VkInstanceCreateInfo_t inst_info = {0};
    inst_info.sType = 1; /* VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO */
    inst_info.pApplicationInfo = &app_info;

    VkResult res = backend->vkCreateInstance(&inst_info, NULL, &backend->instance);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkCreateInstance failed: %d", res);
        vk_unload_library(backend->vulkan_lib);
        backend->vulkan_lib = NULL;
        return -1;
    }

    uint32_t dev_count = 0;
    backend->vkEnumeratePhysicalDevices(backend->instance, &dev_count, NULL);
    if (dev_count == 0) {
        LOG_ERROR("No Vulkan physical devices found");
        backend->vkDestroyInstance(backend->instance, NULL);
        vk_unload_library(backend->vulkan_lib);
        backend->vulkan_lib = NULL;
        return -1;
    }

    VkPhysicalDevice* devices = (VkPhysicalDevice*)calloc(dev_count, sizeof(VkPhysicalDevice));
    backend->vkEnumeratePhysicalDevices(backend->instance, &dev_count, devices);
    backend->physical_device = devices[0]; /* Use first device */
    free(devices);

    VkPhysicalDeviceProperties_t props = {0};
    backend->vkGetPhysicalDeviceProperties(backend->physical_device, &props);
    strncpy(backend->device_name, props.deviceName, sizeof(backend->device_name) - 1);
    backend->api_version = props.apiVersion;

    VkPhysicalDeviceMemoryProperties_t mem_props = {0};
    backend->vkGetPhysicalDeviceMemoryProperties(backend->physical_device, &mem_props);

    backend->memory_type_device_local = find_memory_type(
        &mem_props, UINT32_MAX, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    backend->memory_type_host_visible = find_memory_type(
        &mem_props, UINT32_MAX,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    backend->total_memory = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        if (mem_props.memoryHeaps[i].flags & 0x01) /* VK_MEMORY_HEAP_DEVICE_LOCAL_BIT */
            backend->total_memory += mem_props.memoryHeaps[i].size;
    }

    uint32_t qf_count = 0;
    backend->vkGetPhysicalDeviceQueueFamilyProperties(backend->physical_device, &qf_count, NULL);
    VkQueueFamilyProperties_t* qf_props =
        (VkQueueFamilyProperties_t*)calloc(qf_count, sizeof(VkQueueFamilyProperties_t));
    backend->vkGetPhysicalDeviceQueueFamilyProperties(backend->physical_device, &qf_count,
                                                       qf_props);

    backend->compute_queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            backend->compute_queue_family = i;
            break;
        }
    }
    free(qf_props);

    if (backend->compute_queue_family == UINT32_MAX) {
        LOG_ERROR("No Vulkan compute queue family found");
        backend->vkDestroyInstance(backend->instance, NULL);
        vk_unload_library(backend->vulkan_lib);
        backend->vulkan_lib = NULL;
        return -1;
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo_t queue_info = {0};
    queue_info.sType = 2; /* VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO */
    queue_info.queueFamilyIndex = backend->compute_queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo_t dev_info = {0};
    dev_info.sType = 3; /* VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO */
    dev_info.queueCreateInfoCount = 1;
    dev_info.pQueueCreateInfos = &queue_info;

    res = backend->vkCreateDevice(backend->physical_device, &dev_info, NULL, &backend->device);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkCreateDevice failed: %d", res);
        backend->vkDestroyInstance(backend->instance, NULL);
        vk_unload_library(backend->vulkan_lib);
        backend->vulkan_lib = NULL;
        return -1;
    }

    backend->vkGetDeviceQueue(backend->device, backend->compute_queue_family, 0,
                               &backend->compute_queue);

    VkCommandPoolCreateInfo_t pool_info = {0};
    pool_info.sType = 39; /* VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO */
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = backend->compute_queue_family;

    res = backend->vkCreateCommandPool(backend->device, &pool_info, NULL,
                                        &backend->command_pool);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkCreateCommandPool failed: %d", res);
        backend->vkDestroyDevice(backend->device, NULL);
        backend->vkDestroyInstance(backend->instance, NULL);
        vk_unload_library(backend->vulkan_lib);
        backend->vulkan_lib = NULL;
        return -1;
    }

    backend->initialized = true;
    LOG_INFO("Vulkan backend initialized: %s (API %u.%u.%u, %zu MB VRAM)",
             backend->device_name,
             (backend->api_version >> 22), (backend->api_version >> 12) & 0x3FF,
             backend->api_version & 0xFFF,
             backend->total_memory / (1024 * 1024));
    return 0;
}

void cml_vulkan_backend_free(CMLVulkanBackend* backend) {
    if (!backend) return;

    if (backend->initialized) {
        if (backend->device) {
            backend->vkDeviceWaitIdle(backend->device);
            if (backend->command_pool)
                backend->vkDestroyCommandPool(backend->device, backend->command_pool, NULL);
            backend->vkDestroyDevice(backend->device, NULL);
        }
        if (backend->instance)
            backend->vkDestroyInstance(backend->instance, NULL);
    }

    if (backend->vulkan_lib)
        vk_unload_library(backend->vulkan_lib);

    free(backend);
}


CMLVulkanBuffer* cml_vulkan_buffer_create(CMLVulkanBackend* backend, VkDeviceSize size,
                                           bool device_local) {
    if (!backend || !backend->initialized || size == 0) return NULL;

    CMLVulkanBuffer* buf = (CMLVulkanBuffer*)calloc(1, sizeof(CMLVulkanBuffer));
    if (!buf) return NULL;

    buf->size = size;
    buf->is_device_local = device_local;

    VkBufferCreateInfo_t buf_info = {0};
    buf_info.sType = 12; /* VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO */
    buf_info.size = size;
    buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_info.sharingMode = 0; /* VK_SHARING_MODE_EXCLUSIVE */

    VkResult res = backend->vkCreateBuffer(backend->device, &buf_info, NULL, &buf->buffer);
    if (res != VK_SUCCESS) {
        free(buf);
        return NULL;
    }

    VkMemoryRequirements_t mem_reqs = {0};
    backend->vkGetBufferMemoryRequirements(backend->device, buf->buffer, &mem_reqs);

    VkMemoryAllocateInfo_t alloc_info = {0};
    alloc_info.sType = 5; /* VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO */
    alloc_info.allocationSize = mem_reqs.size;
    /* Use cached memory types from backend */
    alloc_info.memoryTypeIndex = device_local ?
        backend->memory_type_device_local : backend->memory_type_host_visible;

    res = backend->vkAllocateMemory(backend->device, &alloc_info, NULL, &buf->memory);
    if (res != VK_SUCCESS) {
        backend->vkDestroyBuffer(backend->device, buf->buffer, NULL);
        free(buf);
        return NULL;
    }

    res = backend->vkBindBufferMemory(backend->device, buf->buffer, buf->memory, 0);
    if (res != VK_SUCCESS) {
        backend->vkFreeMemory(backend->device, buf->memory, NULL);
        backend->vkDestroyBuffer(backend->device, buf->buffer, NULL);
        free(buf);
        return NULL;
    }

    /* Map host-visible memory persistently */
    if (!device_local) {
        res = backend->vkMapMemory(backend->device, buf->memory, 0, size, 0, &buf->mapped);
        if (res != VK_SUCCESS) {
            buf->mapped = NULL;
        }
    }

    return buf;
}

void cml_vulkan_buffer_free(CMLVulkanBackend* backend, CMLVulkanBuffer* buf) {
    if (!backend || !buf) return;

    if (buf->mapped)
        backend->vkUnmapMemory(backend->device, buf->memory);
    if (buf->memory)
        backend->vkFreeMemory(backend->device, buf->memory, NULL);
    if (buf->buffer)
        backend->vkDestroyBuffer(backend->device, buf->buffer, NULL);
    free(buf);
}

int cml_vulkan_buffer_upload(CMLVulkanBackend* backend, CMLVulkanBuffer* dst,
                              const void* src, size_t size) {
    if (!backend || !dst || !src) return -1;

    if (dst->mapped) {
        /* Host-visible: direct memcpy */
        memcpy(dst->mapped, src, size);
        return 0;
    }

    /* Device-local: use staging buffer */
    CMLVulkanBuffer* staging = cml_vulkan_buffer_create(backend, size, false);
    if (!staging) return -1;

    memcpy(staging->mapped, src, size);

    /* Record and submit copy command */
    VkCommandBufferAllocateInfo_t alloc_info = {0};
    alloc_info.sType = 40;
    alloc_info.commandPool = backend->command_pool;
    alloc_info.level = 0; /* PRIMARY */
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd = NULL;
    backend->vkAllocateCommandBuffers(backend->device, &alloc_info, &cmd);

    VkCommandBufferBeginInfo_t begin_info = {0};
    begin_info.sType = 42;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    backend->vkBeginCommandBuffer(cmd, &begin_info);

    VkBufferCopy_t region = {0, 0, size};
    backend->vkCmdCopyBuffer(cmd, staging->buffer, dst->buffer, 1, &region);

    backend->vkEndCommandBuffer(cmd);

    VkSubmitInfo_t submit = {0};
    submit.sType = 4;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    backend->vkQueueSubmit(backend->compute_queue, 1, &submit, NULL);
    backend->vkQueueWaitIdle(backend->compute_queue);

    backend->vkFreeCommandBuffers(backend->device, backend->command_pool, 1, &cmd);
    cml_vulkan_buffer_free(backend, staging);
    return 0;
}

int cml_vulkan_buffer_download(CMLVulkanBackend* backend, CMLVulkanBuffer* src,
                                void* dst, size_t size) {
    if (!backend || !src || !dst) return -1;

    if (src->mapped) {
        memcpy(dst, src->mapped, size);
        return 0;
    }

    /* Device-local: use staging buffer */
    CMLVulkanBuffer* staging = cml_vulkan_buffer_create(backend, size, false);
    if (!staging) return -1;

    VkCommandBufferAllocateInfo_t alloc_info = {0};
    alloc_info.sType = 40;
    alloc_info.commandPool = backend->command_pool;
    alloc_info.level = 0;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd = NULL;
    backend->vkAllocateCommandBuffers(backend->device, &alloc_info, &cmd);

    VkCommandBufferBeginInfo_t begin_info = {0};
    begin_info.sType = 42;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    backend->vkBeginCommandBuffer(cmd, &begin_info);

    VkBufferCopy_t region = {0, 0, size};
    backend->vkCmdCopyBuffer(cmd, src->buffer, staging->buffer, 1, &region);

    backend->vkEndCommandBuffer(cmd);

    VkSubmitInfo_t submit = {0};
    submit.sType = 4;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    backend->vkQueueSubmit(backend->compute_queue, 1, &submit, NULL);
    backend->vkQueueWaitIdle(backend->compute_queue);

    memcpy(dst, staging->mapped, size);

    backend->vkFreeCommandBuffers(backend->device, backend->command_pool, 1, &cmd);
    cml_vulkan_buffer_free(backend, staging);
    return 0;
}


CMLVulkanKernel* cml_vulkan_kernel_create(CMLVulkanBackend* backend, const uint32_t* spirv,
                                            size_t spirv_size, const char* entry_point,
                                            int num_buffers) {
    if (!backend || !backend->initialized || !spirv || spirv_size == 0) return NULL;
    if (num_buffers < 1 || num_buffers > VK_MAX_BUFFERS_PER_KERNEL) return NULL;

    CMLVulkanKernel* kernel = (CMLVulkanKernel*)calloc(1, sizeof(CMLVulkanKernel));
    if (!kernel) return NULL;
    kernel->num_buffers = num_buffers;
    kernel->name = strdup(entry_point ? entry_point : "main");

    VkShaderModuleCreateInfo_t sm_info = {0};
    sm_info.sType = 16;
    sm_info.codeSize = spirv_size;
    sm_info.pCode = spirv;

    VkResult res = backend->vkCreateShaderModule(backend->device, &sm_info, NULL,
                                                   &kernel->shader_module);
    if (res != VK_SUCCESS) goto fail;

    /* Create descriptor set layout: N storage buffers */
    VkDescriptorSetLayoutBinding_t* bindings =
        (VkDescriptorSetLayoutBinding_t*)calloc(num_buffers,
                                                  sizeof(VkDescriptorSetLayoutBinding_t));
    for (int i = 0; i < num_buffers; i++) {
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo_t dsl_info = {0};
    dsl_info.sType = 32;
    dsl_info.bindingCount = (uint32_t)num_buffers;
    dsl_info.pBindings = bindings;

    res = backend->vkCreateDescriptorSetLayout(backend->device, &dsl_info, NULL,
                                                 &kernel->desc_layout);
    free(bindings);
    if (res != VK_SUCCESS) goto fail;

    VkPipelineLayoutCreateInfo_t pl_info = {0};
    pl_info.sType = 30;
    pl_info.setLayoutCount = 1;
    pl_info.pSetLayouts = &kernel->desc_layout;

    res = backend->vkCreatePipelineLayout(backend->device, &pl_info, NULL,
                                            &kernel->pipeline_layout);
    if (res != VK_SUCCESS) goto fail;

    VkComputePipelineCreateInfo_t cp_info = {0};
    cp_info.sType = 29;
    cp_info.stage.sType = 18;
    cp_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp_info.stage.module = kernel->shader_module;
    cp_info.stage.pName = kernel->name;
    cp_info.layout = kernel->pipeline_layout;
    cp_info.basePipelineIndex = -1;

    res = backend->vkCreateComputePipelines(backend->device, NULL, 1, &cp_info, NULL,
                                              &kernel->pipeline);
    if (res != VK_SUCCESS) goto fail;

    VkDescriptorPoolSize_t pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)num_buffers};
    VkDescriptorPoolCreateInfo_t dp_info = {0};
    dp_info.sType = 33;
    dp_info.maxSets = 1;
    dp_info.poolSizeCount = 1;
    dp_info.pPoolSizes = &pool_size;

    res = backend->vkCreateDescriptorPool(backend->device, &dp_info, NULL,
                                            &kernel->desc_pool);
    if (res != VK_SUCCESS) goto fail;

    VkDescriptorSetAllocateInfo_t ds_info = {0};
    ds_info.sType = 34;
    ds_info.descriptorPool = kernel->desc_pool;
    ds_info.descriptorSetCount = 1;
    ds_info.pSetLayouts = &kernel->desc_layout;

    res = backend->vkAllocateDescriptorSets(backend->device, &ds_info, &kernel->desc_set);
    if (res != VK_SUCCESS) goto fail;

    return kernel;

fail:
    cml_vulkan_kernel_free(backend, kernel);
    return NULL;
}

void cml_vulkan_kernel_free(CMLVulkanBackend* backend, CMLVulkanKernel* kernel) {
    if (!backend || !kernel) return;

    if (kernel->desc_pool)
        backend->vkDestroyDescriptorPool(backend->device, kernel->desc_pool, NULL);
    if (kernel->pipeline)
        backend->vkDestroyPipeline(backend->device, kernel->pipeline, NULL);
    if (kernel->pipeline_layout)
        backend->vkDestroyPipelineLayout(backend->device, kernel->pipeline_layout, NULL);
    if (kernel->desc_layout)
        backend->vkDestroyDescriptorSetLayout(backend->device, kernel->desc_layout, NULL);
    if (kernel->shader_module)
        backend->vkDestroyShaderModule(backend->device, kernel->shader_module, NULL);
    free(kernel->name);
    free(kernel);
}

int cml_vulkan_kernel_bind_buffer(CMLVulkanBackend* backend, CMLVulkanKernel* kernel,
                                   int binding, CMLVulkanBuffer* buffer) {
    if (!backend || !kernel || !buffer) return -1;
    if (binding < 0 || binding >= kernel->num_buffers) return -1;

    VkDescriptorBufferInfo_t buf_info = {0};
    buf_info.buffer = buffer->buffer;
    buf_info.offset = 0;
    buf_info.range = buffer->size;

    VkWriteDescriptorSet_t write = {0};
    write.sType = 35;
    write.dstSet = kernel->desc_set;
    write.dstBinding = (uint32_t)binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buf_info;

    backend->vkUpdateDescriptorSets(backend->device, 1, &write, 0, NULL);
    return 0;
}

int cml_vulkan_kernel_dispatch(CMLVulkanBackend* backend, CMLVulkanKernel* kernel,
                                uint32_t gx, uint32_t gy, uint32_t gz) {
    if (!backend || !kernel) return -1;

    VkCommandBufferAllocateInfo_t alloc_info = {0};
    alloc_info.sType = 40;
    alloc_info.commandPool = backend->command_pool;
    alloc_info.level = 0;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd = NULL;
    VkResult res = backend->vkAllocateCommandBuffers(backend->device, &alloc_info, &cmd);
    if (res != VK_SUCCESS) return -1;

    VkCommandBufferBeginInfo_t begin_info = {0};
    begin_info.sType = 42;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    backend->vkBeginCommandBuffer(cmd, &begin_info);

    backend->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline);
    backend->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      kernel->pipeline_layout, 0, 1,
                                      &kernel->desc_set, 0, NULL);

    backend->vkCmdDispatch(cmd, gx, gy, gz);

    /* Memory barrier for shader writes */
    VkMemoryBarrier_t barrier = {0};
    barrier.sType = 46;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    backend->vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);

    backend->vkEndCommandBuffer(cmd);

    VkFenceCreateInfo_t fence_info = {0};
    fence_info.sType = 8;
    VkFence fence = NULL;
    backend->vkCreateFence(backend->device, &fence_info, NULL, &fence);

    VkSubmitInfo_t submit = {0};
    submit.sType = 4;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    backend->vkQueueSubmit(backend->compute_queue, 1, &submit, fence);
    backend->vkWaitForFences(backend->device, 1, &fence, 1, UINT64_MAX);

    backend->vkDestroyFence(backend->device, fence, NULL);
    backend->vkFreeCommandBuffers(backend->device, backend->command_pool, 1, &cmd);
    return 0;
}


int cml_vulkan_execute_graph(CMLVulkanBackend* backend, CMLGraph_t ir) {
    if (!backend || !backend->initialized || !ir) return -1;

    /* Walk the IR graph and execute each node via SPIR-V codegen + Vulkan dispatch */
    CMLSPIRVCodegen* cg = cml_spirv_codegen_create();
    if (!cg) return -1;

    struct IRNode* node = ir->head;
    int result = 0;
    int local_size = 256;

    while (node && result == 0) {
        if (!node->output || !node->output->data) {
            /* Allocate output if needed */
            if (node->output && node->output->numel > 0 && !node->output->data) {
                node->output->data = calloc(node->output->numel, sizeof(float));
            }
        }

        /* Determine operation and generate SPIR-V */
        size_t spirv_size = 0;
        uint32_t* spirv = NULL;
        int num_bufs = 0;
        UOpType op = node->type;

        if (cml_schedule_is_elementwise(op) && node->num_inputs == 1) {
            spirv = cml_spirv_gen_unary(cg, op, "main", &spirv_size);
            num_bufs = 3; /* in, out, params */
        } else if (cml_schedule_is_elementwise(op) && node->num_inputs == 2) {
            spirv = cml_spirv_gen_binary(cg, op, "main", &spirv_size);
            num_bufs = 3;
        } else if (op == UOP_MATMUL) {
            spirv = cml_spirv_gen_matmul(cg, "main", &spirv_size);
            num_bufs = 3;
        }

        if (!spirv) {
            /* Fall back to CPU for unsupported ops */
            node = node->next;
            continue;
        }

        /* Create kernel and execute */
        CMLVulkanKernel* kernel = cml_vulkan_kernel_create(backend, spirv, spirv_size,
                                                             "main", num_bufs);
        free(spirv);
        if (!kernel) {
            node = node->next;
            continue;
        }

        /* Create GPU buffers, upload data, bind, dispatch, download */
        size_t n = node->output->numel;
        uint32_t groups = ((uint32_t)n + local_size - 1) / local_size;

            CMLVulkanBuffer* in_bufs[2] = {NULL, NULL};
        for (int i = 0; i < node->num_inputs && i < 2; i++) {
            if (node->inputs[i] && node->inputs[i]->data) {
                size_t sz = node->inputs[i]->numel * sizeof(float);
                in_bufs[i] = cml_vulkan_buffer_create(backend, sz, true);
                if (in_bufs[i])
                    cml_vulkan_buffer_upload(backend, in_bufs[i], node->inputs[i]->data, sz);
            }
        }

        CMLVulkanBuffer* out_buf = cml_vulkan_buffer_create(backend, n * sizeof(float), true);

            if (in_bufs[0]) cml_vulkan_kernel_bind_buffer(backend, kernel, 0, in_bufs[0]);
        if (out_buf) cml_vulkan_kernel_bind_buffer(backend, kernel, 1, out_buf);
        if (node->num_inputs > 1 && in_bufs[1])
            cml_vulkan_kernel_bind_buffer(backend, kernel, 2, in_bufs[1]);

        cml_vulkan_kernel_dispatch(backend, kernel, groups, 1, 1);

            if (out_buf && node->output->data)
            cml_vulkan_buffer_download(backend, out_buf, node->output->data, n * sizeof(float));

            for (int i = 0; i < 2; i++)
            if (in_bufs[i]) cml_vulkan_buffer_free(backend, in_bufs[i]);
        if (out_buf) cml_vulkan_buffer_free(backend, out_buf);
        cml_vulkan_kernel_free(backend, kernel);

        node = node->next;
    }

    cml_spirv_codegen_destroy(cg);
    return result;
}

int cml_vulkan_synchronize(CMLVulkanBackend* backend) {
    if (!backend || !backend->initialized) return -1;
    return backend->vkDeviceWaitIdle(backend->device) == VK_SUCCESS ? 0 : -1;
}
