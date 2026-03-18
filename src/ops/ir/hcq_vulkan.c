#include "ops/ir/hcq.h"
#include "core/logging.h"

#ifdef CML_HAS_VULKAN
#include "ops/ir/gpu/vulkan_backend.h"
#include <string.h>

extern CMLVulkanBackend* cml_dispatch_get_vulkan_backend(void);

/* ── Vulkan structure type constants (matching vulkan_core.h) ── */
#define VK_STYPE_CMD_BUF_ALLOC_INFO    40
#define VK_STYPE_CMD_BUF_BEGIN_INFO    42
#define VK_STYPE_SUBMIT_INFO            4
#define VK_STYPE_FENCE_CREATE_INFO      8
#define VK_STYPE_BUFFER_CREATE_INFO    12
#define VK_STYPE_MEMORY_ALLOC_INFO      5

#define VK_CMD_BUF_LEVEL_PRIMARY        0
#define VK_CMD_BUF_USAGE_ONE_TIME       1
#define VK_BIND_POINT_COMPUTE           1
#define VK_BUF_USAGE_TRANSFER_SRC       1
#define VK_BUF_USAGE_TRANSFER_DST       2

#define VK_PIPELINE_STAGE_TRANSFER  0x1000
#define VK_ACCESS_TRANSFER_WRITE    0x1000
#define VK_ACCESS_TRANSFER_READ     0x0800

/* ── Inline Vulkan struct layouts ── */

typedef struct {
    uint32_t    sType;
    const void* pNext;
    void*       commandPool;
    uint32_t    level;
    uint32_t    commandBufferCount;
} HCQ_VkCmdBufAllocInfo;

typedef struct {
    uint32_t    sType;
    const void* pNext;
    uint32_t    flags;
    const void* pInheritanceInfo;
} HCQ_VkCmdBufBeginInfo;

typedef struct {
    uint32_t    sType;
    const void* pNext;
    uint32_t    flags;
} HCQ_VkFenceCreateInfo;

typedef struct {
    uint32_t     sType;
    const void*  pNext;
    uint32_t     waitSemaphoreCount;
    const void*  pWaitSemaphores;
    const void*  pWaitDstStageMask;
    uint32_t     commandBufferCount;
    const void*  pCommandBuffers;
    uint32_t     signalSemaphoreCount;
    const void*  pSignalSemaphores;
} HCQ_VkSubmitInfo;

typedef struct {
    uint32_t    sType;
    const void* pNext;
    uint32_t    flags;
    uint64_t    size;
    uint32_t    usage;
    uint32_t    sharingMode;
    uint32_t    queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
} HCQ_VkBufferCreateInfo;

typedef struct {
    uint32_t    sType;
    const void* pNext;
    uint64_t    allocationSize;
    uint32_t    memoryTypeIndex;
} HCQ_VkMemoryAllocInfo;

typedef struct {
    uint64_t size;
    uint64_t alignment;
    uint32_t memoryTypeBits;
} HCQ_VkMemReqs;

typedef struct {
    uint64_t srcOffset;
    uint64_t dstOffset;
    uint64_t size;
} HCQ_VkBufferCopy;

typedef struct {
    uint32_t    sType;
    const void* pNext;
    uint32_t    srcAccessMask;
    uint32_t    dstAccessMask;
} HCQ_VkMemoryBarrier;

/* ── Helper: one-shot command buffer submit and wait ── */

static int vk_submit_and_wait(CMLVulkanBackend* vk, VkCommandBuffer cmd) {
    VkResult res;

    res = vk->vkEndCommandBuffer(cmd);
    if (res != VK_SUCCESS) return -1;

    /* Create fence */
    HCQ_VkFenceCreateInfo fence_ci = { VK_STYPE_FENCE_CREATE_INFO, NULL, 0 };
    VkFence fence = NULL;
    res = vk->vkCreateFence(vk->device, &fence_ci, NULL, &fence);
    if (res != VK_SUCCESS) return -1;

    /* Submit */
    HCQ_VkSubmitInfo submit = {
        .sType = VK_STYPE_SUBMIT_INFO,
        .pNext = NULL,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = NULL,
        .pWaitDstStageMask = NULL,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = NULL
    };

    res = vk->vkQueueSubmit(vk->compute_queue, 1, &submit, fence);
    if (res != VK_SUCCESS) {
        vk->vkDestroyFence(vk->device, fence, NULL);
        return -1;
    }

    /* Wait (UINT64_MAX timeout) */
    res = vk->vkWaitForFences(vk->device, 1, &fence, 1, UINT64_MAX);
    vk->vkDestroyFence(vk->device, fence, NULL);
    return (res == VK_SUCCESS) ? 0 : -1;
}

/* ── Helper: allocate and begin a one-time command buffer ── */

static VkCommandBuffer vk_begin_one_time_cmd(CMLVulkanBackend* vk) {
    HCQ_VkCmdBufAllocInfo alloc_info = {
        .sType = VK_STYPE_CMD_BUF_ALLOC_INFO,
        .pNext = NULL,
        .commandPool = vk->command_pool,
        .level = VK_CMD_BUF_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    VkCommandBuffer cmd = NULL;
    VkResult res = vk->vkAllocateCommandBuffers(vk->device, &alloc_info, &cmd);
    if (res != VK_SUCCESS) return NULL;

    HCQ_VkCmdBufBeginInfo begin = {
        .sType = VK_STYPE_CMD_BUF_BEGIN_INFO,
        .pNext = NULL,
        .flags = VK_CMD_BUF_USAGE_ONE_TIME,
        .pInheritanceInfo = NULL
    };

    res = vk->vkBeginCommandBuffer(cmd, &begin);
    if (res != VK_SUCCESS) {
        vk->vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &cmd);
        return NULL;
    }
    return cmd;
}

/* ── Helper: create a host-visible staging buffer ── */

static int vk_create_staging_buffer(CMLVulkanBackend* vk, size_t bytes,
                                    uint32_t usage_bits,
                                    VkBuffer* out_buf, VkDeviceMemory* out_mem) {
    HCQ_VkBufferCreateInfo buf_ci = {
        .sType = VK_STYPE_BUFFER_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .size = (uint64_t)bytes,
        .usage = usage_bits,
        .sharingMode = 0, /* VK_SHARING_MODE_EXCLUSIVE */
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL
    };

    VkResult res = vk->vkCreateBuffer(vk->device, &buf_ci, NULL, out_buf);
    if (res != VK_SUCCESS) return -1;

    HCQ_VkMemReqs reqs;
    vk->vkGetBufferMemoryRequirements(vk->device, *out_buf, &reqs);

    HCQ_VkMemoryAllocInfo alloc = {
        .sType = VK_STYPE_MEMORY_ALLOC_INFO,
        .pNext = NULL,
        .allocationSize = reqs.size,
        .memoryTypeIndex = vk->memory_type_host_visible
    };

    res = vk->vkAllocateMemory(vk->device, &alloc, NULL, out_mem);
    if (res != VK_SUCCESS) {
        vk->vkDestroyBuffer(vk->device, *out_buf, NULL);
        return -1;
    }

    res = vk->vkBindBufferMemory(vk->device, *out_buf, *out_mem, 0);
    if (res != VK_SUCCESS) {
        vk->vkFreeMemory(vk->device, *out_mem, NULL);
        vk->vkDestroyBuffer(vk->device, *out_buf, NULL);
        return -1;
    }
    return 0;
}

int cml_hcq_vulkan_queue_init(CMLHCQQueue* queue) {
    CMLVulkanBackend* vk = cml_dispatch_get_vulkan_backend();
    if (!vk || !vk->initialized) return -1;

    /* Use the backend's compute queue as the native handle */
    queue->native_handle = vk->compute_queue;
    queue->active = true;
    return 0;
}

void cml_hcq_vulkan_queue_destroy(CMLHCQQueue* queue) {
    if (!queue) return;
    queue->native_handle = NULL;
    queue->active = false;
}

int cml_hcq_vulkan_submit_kernel(CMLHCQQueue* queue, const CMLHCQKernelDesc* desc) {
    if (!queue || !desc || !desc->compiled_kernel) return -1;

    CMLVulkanBackend* vk = cml_dispatch_get_vulkan_backend();
    if (!vk || !vk->initialized) return -1;

    CMLVulkanKernel* kernel = (CMLVulkanKernel*)desc->compiled_kernel;

    VkCommandBuffer cmd = vk_begin_one_time_cmd(vk);
    if (!cmd) return -1;

    /* Bind compute pipeline */
    vk->vkCmdBindPipeline(cmd, VK_BIND_POINT_COMPUTE, kernel->pipeline);

    /* Bind descriptor sets */
    vk->vkCmdBindDescriptorSets(cmd, VK_BIND_POINT_COMPUTE,
                                kernel->pipeline_layout, 0, 1,
                                &kernel->desc_set, 0, NULL);

    /* Dispatch */
    vk->vkCmdDispatch(cmd,
                      (uint32_t)desc->grid[0],
                      (uint32_t)desc->grid[1],
                      (uint32_t)desc->grid[2]);

    /* Submit, wait, clean up */
    int rc = vk_submit_and_wait(vk, cmd);
    vk->vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &cmd);
    return rc;
}

int cml_hcq_vulkan_memcpy_h2d(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes) {
    if (!queue || !dst || !src || bytes == 0) return -1;

    CMLVulkanBackend* vk = cml_dispatch_get_vulkan_backend();
    if (!vk || !vk->initialized) return -1;

    VkBuffer dst_buf = (VkBuffer)dst;

    /* Create host-visible staging buffer */
    VkBuffer staging = NULL;
    VkDeviceMemory staging_mem = NULL;
    if (vk_create_staging_buffer(vk, bytes, VK_BUF_USAGE_TRANSFER_SRC,
                                 &staging, &staging_mem) != 0)
        return -1;

    /* Map, copy host data into staging, unmap */
    void* mapped = NULL;
    VkResult res = vk->vkMapMemory(vk->device, staging_mem, 0, (uint64_t)bytes, 0, &mapped);
    if (res != VK_SUCCESS) {
        vk->vkFreeMemory(vk->device, staging_mem, NULL);
        vk->vkDestroyBuffer(vk->device, staging, NULL);
        return -1;
    }
    memcpy(mapped, src, bytes);
    vk->vkUnmapMemory(vk->device, staging_mem);

    /* Record copy command */
    VkCommandBuffer cmd = vk_begin_one_time_cmd(vk);
    if (!cmd) {
        vk->vkFreeMemory(vk->device, staging_mem, NULL);
        vk->vkDestroyBuffer(vk->device, staging, NULL);
        return -1;
    }

    HCQ_VkBufferCopy region = { 0, 0, (uint64_t)bytes };
    vk->vkCmdCopyBuffer(cmd, staging, dst_buf, 1, &region);

    /* Submit and wait */
    int rc = vk_submit_and_wait(vk, cmd);
    vk->vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &cmd);

    /* Clean up staging */
    vk->vkFreeMemory(vk->device, staging_mem, NULL);
    vk->vkDestroyBuffer(vk->device, staging, NULL);
    return rc;
}

int cml_hcq_vulkan_memcpy_d2h(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes) {
    if (!queue || !dst || !src || bytes == 0) return -1;

    CMLVulkanBackend* vk = cml_dispatch_get_vulkan_backend();
    if (!vk || !vk->initialized) return -1;

    VkBuffer src_buf = (VkBuffer)src;

    /* Create host-visible staging buffer */
    VkBuffer staging = NULL;
    VkDeviceMemory staging_mem = NULL;
    if (vk_create_staging_buffer(vk, bytes, VK_BUF_USAGE_TRANSFER_DST,
                                 &staging, &staging_mem) != 0)
        return -1;

    /* Record copy command: device buffer -> staging */
    VkCommandBuffer cmd = vk_begin_one_time_cmd(vk);
    if (!cmd) {
        vk->vkFreeMemory(vk->device, staging_mem, NULL);
        vk->vkDestroyBuffer(vk->device, staging, NULL);
        return -1;
    }

    HCQ_VkBufferCopy region = { 0, 0, (uint64_t)bytes };
    vk->vkCmdCopyBuffer(cmd, src_buf, staging, 1, &region);

    /* Submit and wait */
    int rc = vk_submit_and_wait(vk, cmd);
    vk->vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &cmd);

    if (rc != 0) {
        vk->vkFreeMemory(vk->device, staging_mem, NULL);
        vk->vkDestroyBuffer(vk->device, staging, NULL);
        return -1;
    }

    /* Map staging, copy to host, unmap */
    void* mapped = NULL;
    VkResult res = vk->vkMapMemory(vk->device, staging_mem, 0, (uint64_t)bytes, 0, &mapped);
    if (res != VK_SUCCESS) {
        vk->vkFreeMemory(vk->device, staging_mem, NULL);
        vk->vkDestroyBuffer(vk->device, staging, NULL);
        return -1;
    }
    memcpy(dst, mapped, bytes);
    vk->vkUnmapMemory(vk->device, staging_mem);

    /* Clean up staging */
    vk->vkFreeMemory(vk->device, staging_mem, NULL);
    vk->vkDestroyBuffer(vk->device, staging, NULL);
    return 0;
}

int cml_hcq_vulkan_signal_create(CMLHCQSignal* signal) {
    CMLVulkanBackend* vk = cml_dispatch_get_vulkan_backend();
    if (!vk || !vk->initialized || !signal) return -1;

    /* Use a VkFence as the signal primitive */
    typedef struct {
        uint32_t sType;
        const void* pNext;
        uint32_t flags;
    } VkFenceCI;
    VkFenceCI ci = {8, NULL, 0}; /* VK_STRUCTURE_TYPE_FENCE_CREATE_INFO */

    void* fence = NULL;
    int res = vk->vkCreateFence(vk->device, &ci, NULL, &fence);
    if (res != 0) return -1;

    signal->native_handle = fence;
    signal->signaled = false;
    return 0;
}

void cml_hcq_vulkan_signal_destroy(CMLHCQSignal* signal) {
    if (!signal || !signal->native_handle) return;

    CMLVulkanBackend* vk = cml_dispatch_get_vulkan_backend();
    if (vk && vk->initialized) {
        vk->vkDestroyFence(vk->device, signal->native_handle, NULL);
    }
    signal->native_handle = NULL;
}

int cml_hcq_vulkan_signal_wait(CMLHCQSignal* signal, uint64_t timeout_ms) {
    if (!signal || !signal->native_handle) return -1;

    CMLVulkanBackend* vk = cml_dispatch_get_vulkan_backend();
    if (!vk || !vk->initialized) return -1;

    uint64_t timeout_ns = timeout_ms * 1000000ULL;
    void* fence = signal->native_handle;
    int res = vk->vkWaitForFences(vk->device, 1, &fence, 1, timeout_ns);
    if (res == 0) {
        signal->signaled = true;
        return 0;
    }
    return -1;
}

int cml_hcq_vulkan_synchronize(CMLHCQQueue* queue) {
    (void)queue;
    CMLVulkanBackend* vk = cml_dispatch_get_vulkan_backend();
    if (!vk || !vk->initialized) return -1;
    return cml_vulkan_synchronize(vk);
}

#else /* !CML_HAS_VULKAN */

/* Stubs when Vulkan is not compiled in */
int  cml_hcq_vulkan_queue_init(CMLHCQQueue* q)    { (void)q; return -1; }
void cml_hcq_vulkan_queue_destroy(CMLHCQQueue* q)  { (void)q; }
int  cml_hcq_vulkan_submit_kernel(CMLHCQQueue* q, const CMLHCQKernelDesc* d)
    { (void)q; (void)d; return -1; }
int  cml_hcq_vulkan_memcpy_h2d(CMLHCQQueue* q, void* d, const void* s, size_t n)
    { (void)q; (void)d; (void)s; (void)n; return -1; }
int  cml_hcq_vulkan_memcpy_d2h(CMLHCQQueue* q, void* d, const void* s, size_t n)
    { (void)q; (void)d; (void)s; (void)n; return -1; }
int  cml_hcq_vulkan_signal_create(CMLHCQSignal* s)  { (void)s; return -1; }
void cml_hcq_vulkan_signal_destroy(CMLHCQSignal* s) { (void)s; }
int  cml_hcq_vulkan_signal_wait(CMLHCQSignal* s, uint64_t t)
    { (void)s; (void)t; return -1; }
int  cml_hcq_vulkan_synchronize(CMLHCQQueue* q)     { (void)q; return -1; }

#endif /* CML_HAS_VULKAN */
