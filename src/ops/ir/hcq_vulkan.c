/**
 * @file hcq_vulkan.c
 * @brief HCQ backend for Vulkan — VkCommandPool per queue, VkFence for signals
 */

#include "ops/ir/hcq.h"
#include "core/logging.h"

#ifdef CML_HAS_VULKAN
#include "ops/ir/gpu/vulkan_backend.h"

extern CMLVulkanBackend* cml_dispatch_get_vulkan_backend(void);

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
    (void)queue; (void)desc;
    /* Kernel dispatch is handled directly by vulkan_backend.c
     * via cml_vulkan_kernel_dispatch(). HCQ integration would
     * record into a persistent command buffer instead. */
    return 0;
}

int cml_hcq_vulkan_memcpy_h2d(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes) {
    (void)queue; (void)dst; (void)src; (void)bytes;
    return 0;
}

int cml_hcq_vulkan_memcpy_d2h(CMLHCQQueue* queue, void* dst, const void* src, size_t bytes) {
    (void)queue; (void)dst; (void)src; (void)bytes;
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
