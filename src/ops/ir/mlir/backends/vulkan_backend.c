/**
 * @file vulkan_backend.c
 * @brief Vulkan backend implementation via dynamic loading
 */

#include "ops/ir/mlir/backends/vulkan_backend.h"
#include "tensor/tensor.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#ifdef __linux__
#include <dlfcn.h>
#define VULKAN_LIB_NAME "libvulkan.so.1"
#elif defined(_WIN32)
#include <windows.h>
#define VULKAN_LIB_NAME "vulkan-1.dll"
#elif defined(__APPLE__)
#include <dlfcn.h>
#define VULKAN_LIB_NAME "libvulkan.dylib" // MoltenVK
#else
#define VULKAN_LIB_NAME NULL
#endif

#define VK_SUCCESS 0
#define VK_STRUCTURE_TYPE_APPLICATION_INFO 0
#define VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO 1
#define VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO 2
#define VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO 3
#define VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO 39
#define VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO 40
#define VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO 42
#define VK_STRUCTURE_TYPE_SUBMIT_INFO 4
#define VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO 16
#define VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO 29
#define VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO 30
#define VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO 32
#define VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO 33
#define VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO 34
#define VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET 35
#define VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO 12
#define VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO 5
#define VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO 18

#define VK_QUEUE_COMPUTE_BIT 0x00000002
#define VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT 0x00000002
#define VK_COMMAND_BUFFER_LEVEL_PRIMARY 0
#define VK_PIPELINE_BIND_POINT_COMPUTE 1
#define VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 0x00000020
#define VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT 0x00000002
#define VK_MEMORY_PROPERTY_HOST_COHERENT_BIT 0x00000004
#define VK_SHARING_MODE_EXCLUSIVE 0
#define VK_DESCRIPTOR_TYPE_STORAGE_BUFFER 7
#define VK_SHADER_STAGE_COMPUTE_BIT 0x00000020

// Vulkan structures (minimal definitions)
typedef struct VkApplicationInfo {
    int sType;
    const void* pNext;
    const char* pApplicationName;
    uint32_t applicationVersion;
    const char* pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
} VkApplicationInfo;

typedef struct VkInstanceCreateInfo {
    int sType;
    const void* pNext;
    uint32_t flags;
    const VkApplicationInfo* pApplicationInfo;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef struct VkDeviceQueueCreateInfo {
    int sType;
    const void* pNext;
    uint32_t flags;
    uint32_t queueFamilyIndex;
    uint32_t queueCount;
    const float* pQueuePriorities;
} VkDeviceQueueCreateInfo;

typedef struct VkDeviceCreateInfo {
    int sType;
    const void* pNext;
    uint32_t flags;
    uint32_t queueCreateInfoCount;
    const VkDeviceQueueCreateInfo* pQueueCreateInfos;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
    const void* pEnabledFeatures;
} VkDeviceCreateInfo;

// VkPhysicalDeviceProperties is ~824 bytes total:
// - Basic fields: ~292 bytes
// - VkPhysicalDeviceLimits: ~504 bytes
// - VkPhysicalDeviceSparseProperties: ~20 bytes
// We allocate extra space to be safe
typedef struct VkPhysicalDeviceProperties {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    int deviceType;
    char deviceName[256];
    uint8_t pipelineCacheUUID[16];
    // VkPhysicalDeviceLimits (~504 bytes) + VkPhysicalDeviceSparseProperties (~20 bytes)
    // Use 256 uint32_t = 1024 bytes to be safe
    uint32_t limits_and_sparse[256];
} VkPhysicalDeviceProperties;

typedef struct VkQueueFamilyProperties {
    uint32_t queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    uint32_t minImageTransferGranularity[3];
} VkQueueFamilyProperties;

typedef struct VkCommandPoolCreateInfo {
    int sType;
    const void* pNext;
    uint32_t flags;
    uint32_t queueFamilyIndex;
} VkCommandPoolCreateInfo;

typedef struct VkCommandBufferAllocateInfo {
    int sType;
    const void* pNext;
    VkCommandPool commandPool;
    int level;
    uint32_t commandBufferCount;
} VkCommandBufferAllocateInfo;

typedef struct VkCommandBufferBeginInfo {
    int sType;
    const void* pNext;
    uint32_t flags;
    const void* pInheritanceInfo;
} VkCommandBufferBeginInfo;

typedef struct VkSubmitInfo {
    int sType;
    const void* pNext;
    uint32_t waitSemaphoreCount;
    const void* pWaitSemaphores;
    const uint32_t* pWaitDstStageMask;
    uint32_t commandBufferCount;
    const VkCommandBuffer* pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const void* pSignalSemaphores;
} VkSubmitInfo;

typedef struct VkShaderModuleCreateInfo {
    int sType;
    const void* pNext;
    uint32_t flags;
    size_t codeSize;
    const uint32_t* pCode;
} VkShaderModuleCreateInfo;

typedef struct VkDescriptorSetLayoutBinding {
    uint32_t binding;
    int descriptorType;
    uint32_t descriptorCount;
    uint32_t stageFlags;
    const void* pImmutableSamplers;
} VkDescriptorSetLayoutBinding;

typedef struct VkDescriptorSetLayoutCreateInfo {
    int sType;
    const void* pNext;
    uint32_t flags;
    uint32_t bindingCount;
    const VkDescriptorSetLayoutBinding* pBindings;
} VkDescriptorSetLayoutCreateInfo;

typedef struct VkPipelineLayoutCreateInfo {
    int sType;
    const void* pNext;
    uint32_t flags;
    uint32_t setLayoutCount;
    const VkDescriptorSetLayout* pSetLayouts;
    uint32_t pushConstantRangeCount;
    const void* pPushConstantRanges;
} VkPipelineLayoutCreateInfo;

typedef struct VkPipelineShaderStageCreateInfo {
    int sType;
    const void* pNext;
    uint32_t flags;
    int stage;
    VkShaderModule module;
    const char* pName;
    const void* pSpecializationInfo;
} VkPipelineShaderStageCreateInfo;

typedef struct VkComputePipelineCreateInfo {
    int sType;
    const void* pNext;
    uint32_t flags;
    VkPipelineShaderStageCreateInfo stage;
    VkPipelineLayout layout;
    VkPipeline basePipelineHandle;
    int32_t basePipelineIndex;
} VkComputePipelineCreateInfo;

typedef struct VkDescriptorPoolSize {
    int type;
    uint32_t descriptorCount;
} VkDescriptorPoolSize;

typedef struct VkDescriptorPoolCreateInfo {
    int sType;
    const void* pNext;
    uint32_t flags;
    uint32_t maxSets;
    uint32_t poolSizeCount;
    const VkDescriptorPoolSize* pPoolSizes;
} VkDescriptorPoolCreateInfo;

typedef struct VkDescriptorSetAllocateInfo {
    int sType;
    const void* pNext;
    VkDescriptorPool descriptorPool;
    uint32_t descriptorSetCount;
    const VkDescriptorSetLayout* pSetLayouts;
} VkDescriptorSetAllocateInfo;

typedef struct VkDescriptorBufferInfo {
    VkBuffer buffer;
    uint64_t offset;
    uint64_t range;
} VkDescriptorBufferInfo;

typedef struct VkWriteDescriptorSet {
    int sType;
    const void* pNext;
    VkDescriptorSet dstSet;
    uint32_t dstBinding;
    uint32_t dstArrayElement;
    uint32_t descriptorCount;
    int descriptorType;
    const void* pImageInfo;
    const VkDescriptorBufferInfo* pBufferInfo;
    const void* pTexelBufferView;
} VkWriteDescriptorSet;

typedef struct VkBufferCreateInfo {
    int sType;
    const void* pNext;
    uint32_t flags;
    uint64_t size;
    uint32_t usage;
    int sharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
} VkBufferCreateInfo;

typedef struct VkMemoryRequirements {
    uint64_t size;
    uint64_t alignment;
    uint32_t memoryTypeBits;
} VkMemoryRequirements;

typedef struct VkMemoryType {
    uint32_t propertyFlags;
    uint32_t heapIndex;
} VkMemoryType;

typedef struct VkMemoryHeap {
    uint64_t size;
    uint32_t flags;
} VkMemoryHeap;

typedef struct VkPhysicalDeviceMemoryProperties {
    uint32_t memoryTypeCount;
    VkMemoryType memoryTypes[32];
    uint32_t memoryHeapCount;
    VkMemoryHeap memoryHeaps[16];
} VkPhysicalDeviceMemoryProperties;

typedef struct VkMemoryAllocateInfo {
    int sType;
    const void* pNext;
    uint64_t allocationSize;
    uint32_t memoryTypeIndex;
} VkMemoryAllocateInfo;

// Dynamic library functions
#if defined(__linux__) || defined(__APPLE__)
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
#elif defined(_WIN32)
static void* load_library(const char* name) { return (void*)LoadLibraryA(name); }
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

bool cml_vulkan_available(void) {
    if (!VULKAN_LIB_NAME)
        return false;

    void* lib = load_library(VULKAN_LIB_NAME);
    if (!lib)
        return false;

    void* vkCreateInstance = get_symbol(lib, "vkCreateInstance");
    unload_library(lib);

    return vkCreateInstance != NULL;
}

CMLVulkanBackend* cml_vulkan_backend_create(void) {
    CMLVulkanBackend* backend = calloc(1, sizeof(CMLVulkanBackend));
    if (!backend) {
        LOG_ERROR("Failed to allocate Vulkan backend");
        return NULL;
    }
    return backend;
}

static int load_vulkan_functions(CMLVulkanBackend* backend) {
    if (!VULKAN_LIB_NAME)
        return -1;

    backend->vulkan_lib = load_library(VULKAN_LIB_NAME);
    if (!backend->vulkan_lib) {
        LOG_ERROR("Failed to load Vulkan library");
        return -1;
    }

#define LOAD_FUNC(name) backend->name = get_symbol(backend->vulkan_lib, #name)

    LOAD_FUNC(vkCreateInstance);
    LOAD_FUNC(vkDestroyInstance);
    LOAD_FUNC(vkEnumeratePhysicalDevices);
    LOAD_FUNC(vkGetPhysicalDeviceProperties);
    LOAD_FUNC(vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD_FUNC(vkCreateDevice);
    LOAD_FUNC(vkDestroyDevice);
    LOAD_FUNC(vkGetDeviceQueue);
    LOAD_FUNC(vkCreateShaderModule);
    LOAD_FUNC(vkDestroyShaderModule);
    LOAD_FUNC(vkCreateComputePipelines);
    LOAD_FUNC(vkDestroyPipeline);
    LOAD_FUNC(vkCreatePipelineLayout);
    LOAD_FUNC(vkDestroyPipelineLayout);
    LOAD_FUNC(vkCreateDescriptorSetLayout);
    LOAD_FUNC(vkDestroyDescriptorSetLayout);
    LOAD_FUNC(vkCreateDescriptorPool);
    LOAD_FUNC(vkDestroyDescriptorPool);
    LOAD_FUNC(vkAllocateDescriptorSets);
    LOAD_FUNC(vkUpdateDescriptorSets);
    LOAD_FUNC(vkCreateCommandPool);
    LOAD_FUNC(vkDestroyCommandPool);
    LOAD_FUNC(vkAllocateCommandBuffers);
    LOAD_FUNC(vkBeginCommandBuffer);
    LOAD_FUNC(vkEndCommandBuffer);
    LOAD_FUNC(vkCmdBindPipeline);
    LOAD_FUNC(vkCmdBindDescriptorSets);
    LOAD_FUNC(vkCmdDispatch);
    LOAD_FUNC(vkQueueSubmit);
    LOAD_FUNC(vkQueueWaitIdle);
    LOAD_FUNC(vkCreateBuffer);
    LOAD_FUNC(vkDestroyBuffer);
    LOAD_FUNC(vkAllocateMemory);
    LOAD_FUNC(vkFreeMemory);
    LOAD_FUNC(vkBindBufferMemory);
    LOAD_FUNC(vkMapMemory);
    LOAD_FUNC(vkUnmapMemory);
    LOAD_FUNC(vkGetBufferMemoryRequirements);
    LOAD_FUNC(vkGetPhysicalDeviceMemoryProperties);

#undef LOAD_FUNC

    if (!backend->vkCreateInstance || !backend->vkCmdDispatch) {
        LOG_ERROR("Failed to load required Vulkan functions");
        return -1;
    }

    return 0;
}

static int find_compute_queue_family(CMLVulkanBackend* backend) {
    uint32_t count = 0;
    backend->vkGetPhysicalDeviceQueueFamilyProperties(backend->physical_device, &count, NULL);

    if (count == 0)
        return -1;

    VkQueueFamilyProperties* props = malloc(count * sizeof(VkQueueFamilyProperties));
    if (!props)
        return -1;

    backend->vkGetPhysicalDeviceQueueFamilyProperties(backend->physical_device, &count, props);

    int result = -1;
    for (uint32_t i = 0; i < count; i++) {
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            result = (int)i;
            break;
        }
    }

    free(props);
    return result;
}

static uint32_t find_memory_type(CMLVulkanBackend* backend, uint32_t type_filter,
                                 uint32_t properties) {
    VkPhysicalDeviceMemoryProperties mem_props;
    backend->vkGetPhysicalDeviceMemoryProperties(backend->physical_device, &mem_props);

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    return UINT32_MAX;
}

int cml_vulkan_backend_init(CMLVulkanBackend* backend) {
    if (!backend)
        return -1;
    if (backend->initialized)
        return 0;

    if (load_vulkan_functions(backend) != 0)
        return -1;

    // 1. Create Vulkan instance
    VkApplicationInfo app_info = {
        .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName   = "CML",
        .applicationVersion = 1,
        .pEngineName        = "CML MLIR",
        .engineVersion      = 1,
        .apiVersion         = (1 << 22) | (2 << 12) // Vulkan 1.2
    };

    VkInstanceCreateInfo instance_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                          .pApplicationInfo      = &app_info,
                                          .enabledLayerCount     = 0,
                                          .enabledExtensionCount = 0};

    VkResult result = backend->vkCreateInstance(&instance_info, NULL, &backend->instance);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create Vulkan instance: %d", result);
        return -1;
    }

    // 2. Enumerate physical devices
    uint32_t device_count = 0;
    backend->vkEnumeratePhysicalDevices(backend->instance, &device_count, NULL);
    if (device_count == 0) {
        LOG_ERROR("No Vulkan-capable devices found");
        backend->vkDestroyInstance(backend->instance, NULL);
        return -1;
    }

    VkPhysicalDevice* devices = malloc(device_count * sizeof(VkPhysicalDevice));
    if (!devices) {
        backend->vkDestroyInstance(backend->instance, NULL);
        return -1;
    }

    backend->vkEnumeratePhysicalDevices(backend->instance, &device_count, devices);
    backend->physical_device = devices[0]; // Use first device
    free(devices);

    // Get device properties
    VkPhysicalDeviceProperties props;
    backend->vkGetPhysicalDeviceProperties(backend->physical_device, &props);
    strncpy(backend->device_name, props.deviceName, sizeof(backend->device_name) - 1);
    backend->device_name[sizeof(backend->device_name) - 1] = '\0';
    LOG_INFO("Vulkan device: %s", backend->device_name);

    // 3. Find compute queue family
    int queue_family = find_compute_queue_family(backend);
    if (queue_family < 0) {
        LOG_ERROR("No compute queue family found");
        backend->vkDestroyInstance(backend->instance, NULL);
        return -1;
    }
    backend->compute_queue_family = (uint32_t)queue_family;

    // 4. Create logical device
    float queue_priority               = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                          .queueFamilyIndex = backend->compute_queue_family,
                                          .queueCount       = 1,
                                          .pQueuePriorities = &queue_priority};

    VkDeviceCreateInfo device_info = {.sType                 = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                      .queueCreateInfoCount  = 1,
                                      .pQueueCreateInfos     = &queue_info,
                                      .enabledLayerCount     = 0,
                                      .enabledExtensionCount = 0,
                                      .pEnabledFeatures      = NULL};

    result =
        backend->vkCreateDevice(backend->physical_device, &device_info, NULL, &backend->device);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create Vulkan device: %d", result);
        backend->vkDestroyInstance(backend->instance, NULL);
        return -1;
    }

    // 5. Get compute queue
    backend->vkGetDeviceQueue(backend->device, backend->compute_queue_family, 0,
                              &backend->compute_queue);

    // 6. Create command pool
    VkCommandPoolCreateInfo pool_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                         .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                         .queueFamilyIndex = backend->compute_queue_family};

    result =
        backend->vkCreateCommandPool(backend->device, &pool_info, NULL, &backend->command_pool);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create command pool: %d", result);
        backend->vkDestroyDevice(backend->device, NULL);
        backend->vkDestroyInstance(backend->instance, NULL);
        return -1;
    }

    // 7. Create descriptor pool
    VkDescriptorPoolSize pool_sizes[] = {
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 256}};

    VkDescriptorPoolCreateInfo desc_pool_info = {.sType =
                                                     VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                 .maxSets       = 64,
                                                 .poolSizeCount = 1,
                                                 .pPoolSizes    = pool_sizes};

    result = backend->vkCreateDescriptorPool(backend->device, &desc_pool_info, NULL,
                                             &backend->descriptor_pool);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create descriptor pool: %d", result);
        backend->vkDestroyCommandPool(backend->device, backend->command_pool, NULL);
        backend->vkDestroyDevice(backend->device, NULL);
        backend->vkDestroyInstance(backend->instance, NULL);
        return -1;
    }

    backend->initialized = true;
    LOG_INFO("Vulkan backend initialized successfully");

    return 0;
}

void cml_vulkan_backend_free(CMLVulkanBackend* backend) {
    if (!backend)
        return;

    if (backend->initialized) {
        if (backend->descriptor_pool) {
            backend->vkDestroyDescriptorPool(backend->device, backend->descriptor_pool, NULL);
        }
        if (backend->command_pool) {
            backend->vkDestroyCommandPool(backend->device, backend->command_pool, NULL);
        }
        if (backend->device) {
            backend->vkDestroyDevice(backend->device, NULL);
        }
        if (backend->instance) {
            backend->vkDestroyInstance(backend->instance, NULL);
        }
    }

    unload_library(backend->vulkan_lib);
    free(backend);
}

CMLVulkanKernel* cml_vulkan_compile_spirv(CMLVulkanBackend* backend, const uint32_t* spirv_code,
                                          size_t spirv_size, const char* entry_point) {
    if (!backend || !backend->initialized || !spirv_code || !entry_point)
        return NULL;

    CMLVulkanKernel* kernel = calloc(1, sizeof(CMLVulkanKernel));
    if (!kernel)
        return NULL;

    // Create shader module
    VkShaderModuleCreateInfo shader_info = {.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                            .codeSize = spirv_size,
                                            .pCode    = spirv_code};

    VkResult result =
        backend->vkCreateShaderModule(backend->device, &shader_info, NULL, &kernel->shader);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create shader module: %d", result);
        free(kernel);
        return NULL;
    }

    // Create descriptor set layout (assume 3 storage buffers: 2 inputs + 1 output)
    VkDescriptorSetLayoutBinding bindings[3] = {
        {.binding         = 0,
         .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding         = 1,
         .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding         = 2,
         .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT}};

    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings    = bindings};

    result = backend->vkCreateDescriptorSetLayout(backend->device, &layout_info, NULL,
                                                  &kernel->descriptor_layout);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create descriptor set layout: %d", result);
        backend->vkDestroyShaderModule(backend->device, kernel->shader, NULL);
        free(kernel);
        return NULL;
    }

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts    = &kernel->descriptor_layout};

    result = backend->vkCreatePipelineLayout(backend->device, &pipeline_layout_info, NULL,
                                             &kernel->pipeline_layout);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create pipeline layout: %d", result);
        backend->vkDestroyDescriptorSetLayout(backend->device, kernel->descriptor_layout, NULL);
        backend->vkDestroyShaderModule(backend->device, kernel->shader, NULL);
        free(kernel);
        return NULL;
    }

    // Create compute pipeline
    VkComputePipelineCreateInfo pipeline_info = {
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = {.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                   .module = kernel->shader,
                   .pName  = entry_point},
        .layout = kernel->pipeline_layout};

    result = backend->vkCreateComputePipelines(backend->device, NULL, 1, &pipeline_info, NULL,
                                               &kernel->pipeline);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create compute pipeline: %d", result);
        backend->vkDestroyPipelineLayout(backend->device, kernel->pipeline_layout, NULL);
        backend->vkDestroyDescriptorSetLayout(backend->device, kernel->descriptor_layout, NULL);
        backend->vkDestroyShaderModule(backend->device, kernel->shader, NULL);
        free(kernel);
        return NULL;
    }

    kernel->kernel_name       = strdup(entry_point);
    kernel->num_bindings      = 3;
    kernel->workgroup_size[0] = 256;
    kernel->workgroup_size[1] = kernel->workgroup_size[2] = 1;

    LOG_INFO("Compiled Vulkan SPIR-V kernel: %s", entry_point);
    return kernel;
}

void cml_vulkan_kernel_free(CMLVulkanBackend* backend, CMLVulkanKernel* kernel) {
    if (!backend || !kernel)
        return;

    if (kernel->pipeline) {
        backend->vkDestroyPipeline(backend->device, kernel->pipeline, NULL);
    }
    if (kernel->pipeline_layout) {
        backend->vkDestroyPipelineLayout(backend->device, kernel->pipeline_layout, NULL);
    }
    if (kernel->descriptor_layout) {
        backend->vkDestroyDescriptorSetLayout(backend->device, kernel->descriptor_layout, NULL);
    }
    if (kernel->shader) {
        backend->vkDestroyShaderModule(backend->device, kernel->shader, NULL);
    }

    free(kernel->kernel_name);
    free(kernel);
}

int cml_vulkan_launch_kernel(CMLVulkanBackend* backend, CMLVulkanKernel* kernel, VkBuffer* buffers,
                             int num_buffers, uint32_t group_count_x, uint32_t group_count_y,
                             uint32_t group_count_z) {
    if (!backend || !backend->initialized || !kernel || !buffers)
        return -1;

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo alloc_info = {.sType =
                                                  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                              .descriptorPool     = backend->descriptor_pool,
                                              .descriptorSetCount = 1,
                                              .pSetLayouts        = &kernel->descriptor_layout};

    VkDescriptorSet descriptor_set;
    VkResult result =
        backend->vkAllocateDescriptorSets(backend->device, &alloc_info, &descriptor_set);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to allocate descriptor set: %d", result);
        return -1;
    }

    // Update descriptor set with buffers
    VkDescriptorBufferInfo* buffer_infos = malloc(num_buffers * sizeof(VkDescriptorBufferInfo));
    VkWriteDescriptorSet* writes         = malloc(num_buffers * sizeof(VkWriteDescriptorSet));
    if (!buffer_infos || !writes) {
        free(buffer_infos);
        free(writes);
        return -1;
    }

    for (int i = 0; i < num_buffers; i++) {
        buffer_infos[i] = (VkDescriptorBufferInfo){
            .buffer = buffers[i],
            .offset = 0,
            .range  = (uint64_t)-1 // VK_WHOLE_SIZE
        };

        writes[i] = (VkWriteDescriptorSet){.sType      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                           .dstSet     = descriptor_set,
                                           .dstBinding = (uint32_t)i,
                                           .dstArrayElement = 0,
                                           .descriptorCount = 1,
                                           .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                           .pBufferInfo     = &buffer_infos[i]};
    }

    backend->vkUpdateDescriptorSets(backend->device, (uint32_t)num_buffers, writes, 0, NULL);

    // Allocate command buffer
    VkCommandBufferAllocateInfo cmd_alloc_info = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = backend->command_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1};

    VkCommandBuffer cmd_buffer;
    result = backend->vkAllocateCommandBuffers(backend->device, &cmd_alloc_info, &cmd_buffer);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to allocate command buffer: %d", result);
        free(buffer_infos);
        free(writes);
        return -1;
    }

    // Record command buffer
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0x00000001 // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

    backend->vkBeginCommandBuffer(cmd_buffer, &begin_info);
    backend->vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline);
    backend->vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     kernel->pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
    backend->vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, group_count_z);
    backend->vkEndCommandBuffer(cmd_buffer);

    // Submit
    VkSubmitInfo submit_info = {.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                .commandBufferCount = 1,
                                .pCommandBuffers    = &cmd_buffer};

    result = backend->vkQueueSubmit(backend->compute_queue, 1, &submit_info, NULL);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to submit command buffer: %d", result);
        free(buffer_infos);
        free(writes);
        return -1;
    }

    free(buffer_infos);
    free(writes);

    return 0;
}

int cml_vulkan_synchronize(CMLVulkanBackend* backend) {
    if (!backend || !backend->initialized)
        return -1;
    return backend->vkQueueWaitIdle(backend->compute_queue);
}

VkBuffer cml_vulkan_create_buffer(CMLVulkanBackend* backend, size_t size, VkDeviceMemory* memory) {
    if (!backend || !backend->initialized || size == 0 || !memory)
        return NULL;

    // Create buffer
    VkBufferCreateInfo buffer_info = {.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                      .size        = size,
                                      .usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      .sharingMode = VK_SHARING_MODE_EXCLUSIVE};

    VkBuffer buffer;
    VkResult result = backend->vkCreateBuffer(backend->device, &buffer_info, NULL, &buffer);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create buffer: %d", result);
        *memory = 0;
        return NULL;
    }

    // Get memory requirements
    VkMemoryRequirements mem_reqs;
    backend->vkGetBufferMemoryRequirements(backend->device, buffer, &mem_reqs);

    // Find suitable memory type
    uint32_t mem_type = find_memory_type(backend, mem_reqs.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (mem_type == UINT32_MAX) {
        LOG_ERROR("Failed to find suitable memory type");
        backend->vkDestroyBuffer(backend->device, buffer, NULL);
        *memory = 0;
        return NULL;
    }

    // Allocate memory
    VkMemoryAllocateInfo alloc_info = {.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                       .allocationSize  = mem_reqs.size,
                                       .memoryTypeIndex = mem_type};

    result = backend->vkAllocateMemory(backend->device, &alloc_info, NULL, memory);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to allocate device memory: %d", result);
        backend->vkDestroyBuffer(backend->device, buffer, NULL);
        *memory = 0;
        return NULL;
    }

    // Bind memory to buffer
    result = backend->vkBindBufferMemory(backend->device, buffer, *memory, 0);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to bind buffer memory: %d", result);
        backend->vkFreeMemory(backend->device, *memory, NULL);
        backend->vkDestroyBuffer(backend->device, buffer, NULL);
        *memory = 0;
        return NULL;
    }

    return buffer;
}

void cml_vulkan_destroy_buffer(CMLVulkanBackend* backend, VkBuffer buffer, VkDeviceMemory memory) {
    if (!backend || !backend->initialized)
        return;

    if (buffer) {
        backend->vkDestroyBuffer(backend->device, buffer, NULL);
    }
    if (memory) {
        backend->vkFreeMemory(backend->device, memory, NULL);
    }
}

int cml_vulkan_upload_data(CMLVulkanBackend* backend, VkDeviceMemory memory, const void* data,
                           size_t size) {
    if (!backend || !backend->initialized || !memory || !data)
        return -1;

    void* mapped;
    VkResult result = backend->vkMapMemory(backend->device, memory, 0, size, 0, &mapped);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to map memory: %d", result);
        return -1;
    }

    memcpy(mapped, data, size);
    backend->vkUnmapMemory(backend->device, memory);

    return 0;
}

int cml_vulkan_download_data(CMLVulkanBackend* backend, VkDeviceMemory memory, void* data,
                             size_t size) {
    if (!backend || !backend->initialized || !memory || !data)
        return -1;

    void* mapped;
    VkResult result = backend->vkMapMemory(backend->device, memory, 0, size, 0, &mapped);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to map memory: %d", result);
        return -1;
    }

    memcpy(data, mapped, size);
    backend->vkUnmapMemory(backend->device, memory);

    return 0;
}

// Simple buffer-memory association (for tensors)
// In production, use a proper hash map
#define MAX_BUFFER_MAPPINGS 256
static struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
} g_buffer_memory_map[MAX_BUFFER_MAPPINGS];
static int g_buffer_map_count = 0;

static void register_buffer_memory(VkBuffer buffer, VkDeviceMemory memory) {
    if (g_buffer_map_count < MAX_BUFFER_MAPPINGS) {
        g_buffer_memory_map[g_buffer_map_count].buffer = buffer;
        g_buffer_memory_map[g_buffer_map_count].memory = memory;
        g_buffer_map_count++;
    }
}

static VkDeviceMemory find_memory_for_buffer(VkBuffer buffer) {
    for (int i = 0; i < g_buffer_map_count; i++) {
        if (g_buffer_memory_map[i].buffer == buffer) {
            return g_buffer_memory_map[i].memory;
        }
    }
    return 0;
}

int cml_vulkan_upload_tensor(CMLVulkanBackend* backend, Tensor* tensor) {
    if (!backend || !tensor || !tensor->data)
        return -1;

    size_t size = tensor->numel * sizeof(float);

    // Create buffer if needed
    if (!tensor->buffer_handle) {
        VkDeviceMemory memory;
        VkBuffer buffer = cml_vulkan_create_buffer(backend, size, &memory);
        if (!buffer)
            return -1;

        tensor->buffer_handle = buffer;
        register_buffer_memory(buffer, memory);
    }

    // Upload data
    VkDeviceMemory memory = find_memory_for_buffer((VkBuffer)tensor->buffer_handle);
    if (!memory)
        return -1;

    return cml_vulkan_upload_data(backend, memory, tensor->data, size);
}

int cml_vulkan_download_tensor(CMLVulkanBackend* backend, Tensor* tensor) {
    if (!backend || !tensor || !tensor->buffer_handle)
        return -1;

    if (!tensor->data) {
        tensor->data = malloc(tensor->numel * sizeof(float));
        if (!tensor->data)
            return -1;
        tensor->owns_data = true;
    }

    VkDeviceMemory memory = find_memory_for_buffer((VkBuffer)tensor->buffer_handle);
    if (!memory)
        return -1;

    size_t size = tensor->numel * sizeof(float);
    return cml_vulkan_download_data(backend, memory, tensor->data, size);
}
