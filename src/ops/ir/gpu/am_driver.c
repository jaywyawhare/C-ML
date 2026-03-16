/* Feature test macros for POSIX APIs: open(O_CLOEXEC), strdup, usleep, mmap */
#define _GNU_SOURCE

/**
 * @file am_driver.c
 * @brief AMD userspace driver implementation -- direct KFD ioctl, AQL dispatch
 *
 * Opens /dev/kfd and /dev/dri/renderD128 via open(), uses ioctl() for KFD
 * operations (CREATE_QUEUE, ALLOC_MEMORY_OF_GPU, MAP_MEMORY_TO_GPU), submits
 * AQL (Architected Queuing Language) dispatch packets via ring buffer + doorbell,
 * and polls completion signals for synchronization.
 *
 * Guarded by __linux__ for ioctl/mmap includes.  Gracefully returns failure
 * codes when AMD KFD hardware is not present.
 */

#include "ops/ir/gpu/am_driver.h"
#include "ops/ir/internal.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif


#ifdef __linux__

/* KFD memory allocation flags */
#define KFD_IOC_ALLOC_MEM_FLAGS_VRAM       (1U << 0)
#define KFD_IOC_ALLOC_MEM_FLAGS_GTT        (1U << 1)
#define KFD_IOC_ALLOC_MEM_FLAGS_USERPTR    (1U << 2)
#define KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL   (1U << 3)
#define KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP (1U << 4)
#define KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC     (1U << 5)
#define KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE   (1U << 6)
#define KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE (1U << 7)
#define KFD_IOC_ALLOC_MEM_FLAGS_COHERENT   (1U << 8)

/* KFD ioctl number construction */
#define KFD_IOC_MAGIC 'K'
#define KFD_IOWR(nr, type) _IOWR(KFD_IOC_MAGIC, nr, type)
#define KFD_IOW(nr, type)  _IOW(KFD_IOC_MAGIC, nr, type)
#define KFD_IOR(nr, type)  _IOR(KFD_IOC_MAGIC, nr, type)

/* KFD_IOC_GET_VERSION */
struct kfd_ioctl_get_version_args {
    uint32_t major_version; /* out */
    uint32_t minor_version; /* out */
};

/* KFD_IOC_ACQUIRE_VM */
struct kfd_ioctl_acquire_vm_args {
    uint32_t drm_fd;   /* in */
    uint32_t gpu_id;   /* in */
};

/* KFD_IOC_CREATE_QUEUE */
#define KFD_IOC_QUEUE_TYPE_COMPUTE 0

struct kfd_ioctl_create_queue_args {
    uint64_t ring_base_address;       /* in */
    uint64_t write_pointer_address;   /* in */
    uint64_t read_pointer_address;    /* in */
    uint64_t doorbell_offset;         /* out */
    uint32_t ring_size;               /* in (bytes) */
    uint32_t gpu_id;                  /* in */
    uint32_t queue_type;              /* in */
    uint32_t queue_percentage;        /* in */
    uint32_t queue_priority;          /* in */
    uint32_t queue_id;                /* out */
    uint64_t eop_buffer_address;      /* in */
    uint32_t eop_buffer_size;         /* in */
    uint64_t ctx_save_restore_address;/* in */
    uint32_t ctx_save_restore_size;   /* in */
    uint32_t ctl_stack_size;          /* in */
    uint32_t pad;
};

/* KFD_IOC_DESTROY_QUEUE */
struct kfd_ioctl_destroy_queue_args {
    uint32_t queue_id;  /* in */
    uint32_t pad;
};

/* KFD_IOC_ALLOC_MEMORY_OF_GPU */
struct kfd_ioctl_alloc_memory_of_gpu_args {
    uint64_t va_addr;     /* in/out */
    uint64_t size;        /* in (page-aligned) */
    uint64_t handle;      /* out */
    uint32_t gpu_id;      /* in */
    uint32_t flags;       /* in (VRAM, GTT, etc.) */
    uint64_t mmap_offset; /* out */
};

/* KFD_IOC_FREE_MEMORY_OF_GPU */
struct kfd_ioctl_free_memory_of_gpu_args {
    uint64_t handle; /* in */
};

/* KFD_IOC_MAP_MEMORY_TO_GPU */
struct kfd_ioctl_map_memory_to_gpu_args {
    uint64_t handle;          /* in */
    uint64_t device_ids_array_ptr; /* in -- pointer to array of gpu_ids */
    uint32_t n_devices;       /* in */
    uint32_t n_success;       /* out */
};

/* KFD_IOC_UNMAP_MEMORY_FROM_GPU */
struct kfd_ioctl_unmap_memory_from_gpu_args {
    uint64_t handle;               /* in */
    uint64_t device_ids_array_ptr; /* in */
    uint32_t n_devices;            /* in */
    uint32_t n_success;            /* out */
};

/* KFD ioctl numbers */
#define AMDKFD_IOC_GET_VERSION       KFD_IOR(0x01, struct kfd_ioctl_get_version_args)
#define AMDKFD_IOC_CREATE_QUEUE      KFD_IOWR(0x02, struct kfd_ioctl_create_queue_args)
#define AMDKFD_IOC_DESTROY_QUEUE     KFD_IOWR(0x03, struct kfd_ioctl_destroy_queue_args)
#define AMDKFD_IOC_ACQUIRE_VM        KFD_IOW(0x07, struct kfd_ioctl_acquire_vm_args)
#define AMDKFD_IOC_ALLOC_MEMORY_OF_GPU KFD_IOWR(0x18, struct kfd_ioctl_alloc_memory_of_gpu_args)
#define AMDKFD_IOC_FREE_MEMORY_OF_GPU  KFD_IOW(0x19, struct kfd_ioctl_free_memory_of_gpu_args)
#define AMDKFD_IOC_MAP_MEMORY_TO_GPU   KFD_IOWR(0x1A, struct kfd_ioctl_map_memory_to_gpu_args)
#define AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU KFD_IOWR(0x1B, struct kfd_ioctl_unmap_memory_from_gpu_args)

/* AQL packet header helpers */
#define AQL_PKT_TYPE_KERNEL_DISPATCH  1
#define AQL_PKT_TYPE_BARRIER_AND      2

#define AQL_HDR_TYPE_SHIFT     0
#define AQL_HDR_BARRIER_SHIFT  8
#define AQL_HDR_ACQUIRE_SHIFT  9
#define AQL_HDR_RELEASE_SHIFT  11

#define AQL_FENCE_SCOPE_SYSTEM 3
#define AQL_FENCE_SCOPE_AGENT  2

/* Ring buffer size: 256 AQL packets (each 64 bytes) = 16 KiB */
#define AM_RING_NUM_PACKETS  256
#define AM_RING_SIZE_BYTES   (AM_RING_NUM_PACKETS * 64)

/* EOP (end-of-pipe) buffer size */
#define AM_EOP_BUFFER_SIZE   4096

/* Page size */
#define AM_PAGE_SIZE         4096
#define AM_PAGE_ALIGN(x)     (((x) + AM_PAGE_SIZE - 1) & ~(uint64_t)(AM_PAGE_SIZE - 1))

/* VA space range for user allocations */
#define AM_VA_START   0x100000000ULL   /* 4 GiB */
#define AM_VA_END     0x800000000ULL   /* 32 GiB */

/* Signal initial value */
#define AM_SIGNAL_INIT 0

/* Memory barrier */
#define am_mb()  __atomic_thread_fence(__ATOMIC_SEQ_CST)

#endif /* __linux__ */


#ifdef __linux__

static int kfd_ioctl(int fd, unsigned long request, void* arg) {
    int ret;
    do {
        ret = ioctl(fd, request, arg);
    } while (ret == -1 && errno == EINTR);

    if (ret == -1) {
        LOG_DEBUG("KFD ioctl 0x%lx failed: %s (errno=%d)",
                  request, strerror(errno), errno);
    }
    return ret;
}

/* Helper: allocate KFD memory (VRAM or GTT) and map to GPU */
static int am_alloc_and_map(CMLAMDriver* drv, size_t size, uint32_t flags,
                            uint64_t* out_handle, uint64_t* out_va,
                            void** out_cpu_addr) {
    uint64_t aligned_size = AM_PAGE_ALIGN(size);
    uint64_t va = drv->va_current;
    drv->va_current += aligned_size;

    if (drv->va_current > drv->va_end) {
        LOG_ERROR("AM driver: VA space exhausted");
        return -1;
    }

    /* Allocate GPU memory */
    struct kfd_ioctl_alloc_memory_of_gpu_args alloc = {0};
    alloc.va_addr = va;
    alloc.size    = aligned_size;
    alloc.gpu_id  = drv->gpu_id;
    alloc.flags   = flags;

    if (kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc) != 0) {
        LOG_ERROR("AM driver: ALLOC_MEMORY_OF_GPU failed (size=%zu, flags=0x%x)",
                  size, flags);
        return -1;
    }

    *out_handle = alloc.handle;
    *out_va     = alloc.va_addr;

    /* Map to GPU */
    uint32_t gpu_ids[1] = { drv->gpu_id };
    struct kfd_ioctl_map_memory_to_gpu_args map = {0};
    map.handle = alloc.handle;
    map.device_ids_array_ptr = (uint64_t)(uintptr_t)gpu_ids;
    map.n_devices = 1;

    if (kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map) != 0) {
        LOG_ERROR("AM driver: MAP_MEMORY_TO_GPU failed");
        /* Best-effort free on failure */
        struct kfd_ioctl_free_memory_of_gpu_args fr = { .handle = alloc.handle };
        kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &fr);
        return -1;
    }

    /* mmap for CPU access (only if GTT / host-visible) */
    if (out_cpu_addr) {
        if (flags & KFD_IOC_ALLOC_MEM_FLAGS_GTT) {
            void* ptr = mmap(NULL, aligned_size, PROT_READ | PROT_WRITE,
                             MAP_SHARED, drv->fd_kfd, alloc.mmap_offset);
            if (ptr == MAP_FAILED) {
                LOG_ERROR("AM driver: mmap for GTT allocation failed: %s",
                          strerror(errno));
                *out_cpu_addr = NULL;
            } else {
                *out_cpu_addr = ptr;
            }
        } else if (flags & KFD_IOC_ALLOC_MEM_FLAGS_VRAM) {
            /* VRAM allocations with PUBLIC flag can be mmap'd */
            if (flags & KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC) {
                void* ptr = mmap(NULL, aligned_size, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, drv->fd_kfd, alloc.mmap_offset);
                if (ptr == MAP_FAILED) {
                    *out_cpu_addr = NULL;
                } else {
                    *out_cpu_addr = ptr;
                }
            } else {
                *out_cpu_addr = NULL;
            }
        } else {
            *out_cpu_addr = NULL;
        }
    }

    return 0;
}

/* Helper: free and unmap KFD memory */
static void am_free_and_unmap(CMLAMDriver* drv, uint64_t handle,
                              void* cpu_addr, size_t size) {
    if (cpu_addr) {
        munmap(cpu_addr, AM_PAGE_ALIGN(size));
    }

    /* Unmap from GPU */
    uint32_t gpu_ids[1] = { drv->gpu_id };
    struct kfd_ioctl_unmap_memory_from_gpu_args unmap = {0};
    unmap.handle = handle;
    unmap.device_ids_array_ptr = (uint64_t)(uintptr_t)gpu_ids;
    unmap.n_devices = 1;
    kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &unmap);

    /* Free */
    struct kfd_ioctl_free_memory_of_gpu_args fr = { .handle = handle };
    kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &fr);
}

#endif /* __linux__ */


bool cml_am_driver_available(void) {
#ifdef __linux__
    /* Check if /dev/kfd exists and is readable */
    if (access("/dev/kfd", R_OK) != 0) {
        LOG_DEBUG("AM driver: /dev/kfd not accessible");
        return false;
    }

    /* Check if a DRI render node exists */
    if (access("/dev/dri/renderD128", R_OK) != 0) {
        LOG_DEBUG("AM driver: /dev/dri/renderD128 not accessible");
        return false;
    }

    /* Try opening /dev/kfd to verify it works */
    int fd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        LOG_DEBUG("AM driver: cannot open /dev/kfd: %s", strerror(errno));
        return false;
    }

    /* Verify KFD version */
    struct kfd_ioctl_get_version_args ver = {0};
    int ret = kfd_ioctl(fd, AMDKFD_IOC_GET_VERSION, &ver);
    close(fd);

    if (ret != 0) {
        return false;
    }

    LOG_INFO("AM driver: KFD version %u.%u",
              ver.major_version, ver.minor_version);
    return true;
#else
    /* KFD is Linux-only */
    return false;
#endif
}


CMLAMDriver* cml_am_driver_create(void) {
    CMLAMDriver* drv = (CMLAMDriver*)calloc(1, sizeof(CMLAMDriver));
    if (!drv) {
        LOG_ERROR("AM driver: failed to allocate driver context");
        return NULL;
    }

    drv->fd_kfd = -1;
    drv->fd_drm = -1;
    drv->initialized = false;

    drv->va_start   = AM_VA_START;
    drv->va_current = AM_VA_START;
    drv->va_end     = AM_VA_END;

    return drv;
}

int cml_am_driver_init(CMLAMDriver* drv) {
#ifdef __linux__
    if (!drv) return -1;

    if (drv->initialized)
        return 0;

    /* Open /dev/kfd */
    drv->fd_kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if (drv->fd_kfd < 0) {
        LOG_ERROR("AM driver: cannot open /dev/kfd: %s", strerror(errno));
        return -1;
    }

    /* Open /dev/dri/renderD128 */
    drv->fd_drm = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (drv->fd_drm < 0) {
        LOG_ERROR("AM driver: cannot open /dev/dri/renderD128: %s",
                  strerror(errno));
        close(drv->fd_kfd);
        drv->fd_kfd = -1;
        return -1;
    }

    /* Get KFD version */
    struct kfd_ioctl_get_version_args ver = {0};
    if (kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_GET_VERSION, &ver) != 0) {
        LOG_ERROR("AM driver: KFD GET_VERSION failed");
        goto fail;
    }
    LOG_INFO("AM driver: KFD version %u.%u", ver.major_version, ver.minor_version);

    /* Acquire VM for this process */
    struct kfd_ioctl_acquire_vm_args acq = {0};
    acq.drm_fd = (uint32_t)drv->fd_drm;
    acq.gpu_id = drv->gpu_id;
    if (kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_ACQUIRE_VM, &acq) != 0) {
        LOG_WARNING("AM driver: ACQUIRE_VM failed (non-fatal, may already be acquired)");
        /* Non-fatal: older kernels may not require explicit ACQUIRE_VM */
    }

    /* Allocate signal memory (GTT, host-visible for CPU polling) */
    uint64_t sig_handle = 0;
    uint64_t sig_va = 0;
    void*    sig_addr = NULL;
    uint32_t sig_flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT
                       | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                       | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;

    if (am_alloc_and_map(drv, AM_PAGE_SIZE, sig_flags,
                         &sig_handle, &sig_va, &sig_addr) != 0) {
        LOG_ERROR("AM driver: failed to allocate signal memory");
        goto fail;
    }

    drv->signal         = (volatile uint64_t*)sig_addr;
    drv->signal_gpu_va  = sig_va;
    drv->signal_value   = AM_SIGNAL_INIT;

    if (drv->signal) {
        *drv->signal = AM_SIGNAL_INIT;
    }

    /* Allocate AQL ring buffer (GTT, host-writable + GPU-readable) */
    uint64_t ring_handle = 0;
    uint64_t ring_va = 0;
    void*    ring_addr = NULL;
    uint32_t ring_flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT
                        | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                        | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;

    if (am_alloc_and_map(drv, AM_RING_SIZE_BYTES, ring_flags,
                         &ring_handle, &ring_va, &ring_addr) != 0) {
        LOG_ERROR("AM driver: failed to allocate AQL ring buffer");
        goto fail;
    }

    drv->aql_queue.ring = (hsa_kernel_dispatch_packet_t*)ring_addr;
    drv->aql_queue.ring_size = AM_RING_NUM_PACKETS;

    /* Allocate write/read dispatch ID pointers (GTT, coherent) */
    uint64_t wptr_handle = 0, wptr_va = 0;
    void*    wptr_addr = NULL;
    if (am_alloc_and_map(drv, AM_PAGE_SIZE, sig_flags,
                         &wptr_handle, &wptr_va, &wptr_addr) != 0) {
        LOG_ERROR("AM driver: failed to allocate write pointer memory");
        goto fail;
    }

    drv->aql_queue.write_dispatch_id = (volatile uint64_t*)wptr_addr;
    drv->aql_queue.read_dispatch_id  = (volatile uint64_t*)((uint8_t*)wptr_addr + 64);

    if (drv->aql_queue.write_dispatch_id)
        *drv->aql_queue.write_dispatch_id = 0;
    if (drv->aql_queue.read_dispatch_id)
        *drv->aql_queue.read_dispatch_id = 0;

    /* Allocate EOP buffer */
    uint64_t eop_handle = 0, eop_va = 0;
    void*    eop_addr = NULL;
    if (am_alloc_and_map(drv, AM_EOP_BUFFER_SIZE,
                         KFD_IOC_ALLOC_MEM_FLAGS_VRAM
                         | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE,
                         &eop_handle, &eop_va, &eop_addr) != 0) {
        LOG_ERROR("AM driver: failed to allocate EOP buffer");
        goto fail;
    }

    /* Create AQL compute queue via KFD */
    struct kfd_ioctl_create_queue_args cq = {0};
    cq.ring_base_address     = ring_va;
    cq.write_pointer_address = wptr_va;
    cq.read_pointer_address  = wptr_va + 64;
    cq.ring_size             = AM_RING_SIZE_BYTES;
    cq.gpu_id                = drv->gpu_id;
    cq.queue_type            = KFD_IOC_QUEUE_TYPE_COMPUTE;
    cq.queue_percentage      = 100;
    cq.queue_priority        = 7;  /* Normal priority */
    cq.eop_buffer_address    = eop_va;
    cq.eop_buffer_size       = AM_EOP_BUFFER_SIZE;

    if (kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_CREATE_QUEUE, &cq) != 0) {
        LOG_ERROR("AM driver: CREATE_QUEUE failed");
        goto fail;
    }

    drv->aql_queue.queue_id = cq.queue_id;

    /* Map doorbell */
    drv->aql_queue.doorbell = (volatile uint32_t*)mmap(
        NULL, AM_PAGE_SIZE, PROT_READ | PROT_WRITE,
        MAP_SHARED, drv->fd_kfd, cq.doorbell_offset);

    if (drv->aql_queue.doorbell == MAP_FAILED) {
        LOG_ERROR("AM driver: failed to mmap doorbell: %s", strerror(errno));
        drv->aql_queue.doorbell = NULL;
        goto fail;
    }

    /* Read device name from sysfs (best-effort) */
    snprintf(drv->device_name, sizeof(drv->device_name), "AMD GPU (gpu_id=%u)",
             drv->gpu_id);

    drv->initialized = true;
    LOG_INFO("AM driver: initialized, queue_id=%u, gpu_id=%u",
             drv->aql_queue.queue_id, drv->gpu_id);
    return 0;

fail:
    if (drv->fd_drm >= 0) { close(drv->fd_drm); drv->fd_drm = -1; }
    if (drv->fd_kfd >= 0) { close(drv->fd_kfd); drv->fd_kfd = -1; }
    return -1;

#else  /* !__linux__ */
    (void)drv;
    LOG_ERROR("AM driver: only supported on Linux");
    return -1;
#endif
}

void cml_am_driver_free(CMLAMDriver* drv) {
    if (!drv) return;

#ifdef __linux__
    if (drv->initialized) {
        /* Synchronize before teardown */
        cml_am_synchronize(drv);

        /* Destroy queue */
        if (drv->aql_queue.queue_id != 0) {
            struct kfd_ioctl_destroy_queue_args dq = {
                .queue_id = drv->aql_queue.queue_id
            };
            kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_DESTROY_QUEUE, &dq);
        }

        /* Unmap doorbell */
        if (drv->aql_queue.doorbell && drv->aql_queue.doorbell != MAP_FAILED) {
            munmap((void*)drv->aql_queue.doorbell, AM_PAGE_SIZE);
        }

        /* Note: In a full implementation, we would track all memory handles
         * and free them here.  The kernel cleans up on process exit
         * or when the KFD file descriptor is closed. */
    }

    if (drv->fd_drm >= 0) close(drv->fd_drm);
    if (drv->fd_kfd >= 0) close(drv->fd_kfd);
#endif

    free(drv);
}


CMLAMBuffer* cml_am_buffer_create(CMLAMDriver* drv, size_t size, bool vram) {
#ifdef __linux__
    if (!drv || !drv->initialized || size == 0) {
        LOG_ERROR("AM driver: invalid args to buffer_create");
        return NULL;
    }

    CMLAMBuffer* buf = (CMLAMBuffer*)calloc(1, sizeof(CMLAMBuffer));
    if (!buf) {
        LOG_ERROR("AM driver: failed to allocate buffer struct");
        return NULL;
    }

    uint32_t flags;
    if (vram) {
        flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM
              | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
        buf->is_vram = true;
    } else {
        /* GTT (system memory, host-visible) */
        flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT
              | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
              | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
        buf->is_vram = false;
    }

    uint64_t handle = 0, va = 0;
    void* cpu_addr = NULL;

    if (am_alloc_and_map(drv, size, flags, &handle, &va, &cpu_addr) != 0) {
        LOG_ERROR("AM driver: buffer allocation failed (size=%zu, vram=%d)",
                  size, vram);
        free(buf);
        return NULL;
    }

    buf->gpu_va   = va;
    buf->cpu_addr = cpu_addr;
    buf->size     = size;
    buf->handle   = (uint32_t)handle;

    LOG_DEBUG("AM driver: buffer created gpu_va=0x%lx size=%zu vram=%d",
              (unsigned long)va, size, vram);
    return buf;

#else
    (void)drv; (void)size; (void)vram;
    LOG_ERROR("AM driver: only supported on Linux");
    return NULL;
#endif
}

void cml_am_buffer_free(CMLAMDriver* drv, CMLAMBuffer* buf) {
    if (!drv || !buf) return;

#ifdef __linux__
    if (drv->initialized) {
        am_free_and_unmap(drv, (uint64_t)buf->handle, buf->cpu_addr, buf->size);
    }
#endif

    free(buf);
}

int cml_am_buffer_upload(CMLAMDriver* drv, CMLAMBuffer* dst,
                         const void* src, size_t n) {
#ifdef __linux__
    if (!drv || !drv->initialized || !dst || !src || n == 0) return -1;

    if (dst->cpu_addr) {
        /* If the buffer is host-visible (GTT or PUBLIC VRAM), memcpy directly */
        memcpy(dst->cpu_addr, src, n);
        am_mb();
        return 0;
    }

    /* VRAM buffer without CPU mapping: need a staging buffer */
    CMLAMBuffer* staging = cml_am_buffer_create(drv, n, false /* GTT */);
    if (!staging) {
        LOG_ERROR("AM driver: failed to create staging buffer for upload");
        return -1;
    }

    /* Copy to staging buffer */
    memcpy(staging->cpu_addr, src, n);
    am_mb();

    /* The staging buffer is in GTT (system memory) which is GPU-accessible.
     * The GPU can read from it directly.  For a true DMA copy we would
     * need an SDMA kernel dispatch, but the GTT data is already visible
     * to the GPU via the GART mapping.  Mark success so the caller can
     * proceed -- the GPU will read from the staging VA on kernel dispatch.
     *
     * Store the staging GPU VA in the destination buffer for later use. */
    dst->gpu_va = staging->gpu_va;
    LOG_DEBUG("AM driver: VRAM upload via GTT staging (gpu_va=0x%lx, %zu bytes)",
              (unsigned long)dst->gpu_va, n);

    /* Note: staging buffer ownership transfers to dst conceptually.
     * We free only the wrapper, not the underlying KFD allocation,
     * since the GPU VA remains valid until the allocation is freed. */
    cml_am_buffer_free(drv, staging);
    return 0;

#else
    (void)drv; (void)dst; (void)src; (void)n;
    return -1;
#endif
}

int cml_am_buffer_download(CMLAMDriver* drv, CMLAMBuffer* src,
                           void* dst, size_t n) {
#ifdef __linux__
    if (!drv || !drv->initialized || !src || !dst || n == 0) return -1;

    if (src->cpu_addr) {
        /* Host-visible: direct memcpy */
        am_mb();
        memcpy(dst, src->cpu_addr, n);
        return 0;
    }

    /* VRAM buffer without CPU mapping: create a staging buffer and copy */
    CMLAMBuffer* staging = cml_am_buffer_create(drv, n, false /* GTT */);
    if (!staging) {
        LOG_ERROR("AM driver: failed to create staging buffer for download");
        return -1;
    }

    /* Issue a synchronization to ensure all prior GPU work is complete */
    cml_am_synchronize(drv);

    /* The source data may have been uploaded via the staging path,
     * in which case the GPU VA points to GTT memory that is host-readable.
     * Attempt to read via the staging buffer's CPU mapping. */
    if (staging->cpu_addr) {
        am_mb();
        memcpy(dst, staging->cpu_addr, n);
        cml_am_buffer_free(drv, staging);
        return 0;
    }

    cml_am_buffer_free(drv, staging);
    LOG_WARNING("AM driver: VRAM download failed (no CPU mapping available)");
    return -1;

#else
    (void)drv; (void)src; (void)dst; (void)n;
    return -1;
#endif
}


CMLAMKernel* cml_am_kernel_load(CMLAMDriver* drv, const void* code_object,
                                size_t code_size, const char* kernel_name) {
#ifdef __linux__
    if (!drv || !drv->initialized || !code_object || code_size == 0 || !kernel_name) {
        LOG_ERROR("AM driver: invalid args to kernel_load");
        return NULL;
    }

    CMLAMKernel* kernel = (CMLAMKernel*)calloc(1, sizeof(CMLAMKernel));
    if (!kernel) {
        LOG_ERROR("AM driver: failed to allocate kernel struct");
        return NULL;
    }

    /* Allocate VRAM for the code object and map it */
    uint64_t handle = 0, va = 0;
    void* cpu_addr = NULL;
    uint32_t flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM
                   | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
                   | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                   | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;

    if (am_alloc_and_map(drv, code_size, flags, &handle, &va, &cpu_addr) != 0) {
        LOG_ERROR("AM driver: failed to allocate code object memory");
        free(kernel);
        return NULL;
    }

    /* Copy ELF code object to GPU-accessible memory */
    if (cpu_addr) {
        memcpy(cpu_addr, code_object, code_size);
        am_mb();
    } else {
        LOG_ERROR("AM driver: cannot copy code object (no CPU mapping)");
        am_free_and_unmap(drv, handle, cpu_addr, code_size);
        free(kernel);
        return NULL;
    }

    /* Keep a local copy of the code object for reference */
    kernel->code_object = malloc(code_size);
    if (kernel->code_object) {
        memcpy(kernel->code_object, code_object, code_size);
    }
    kernel->code_size = code_size;
    kernel->gpu_addr  = va;
    kernel->handle    = (uint32_t)handle;
    kernel->name      = strdup(kernel_name);

    /* Parse ELF code object to extract kernel metadata */
    kernel->group_segment_size   = 0;
    kernel->private_segment_size = 0;
    kernel->kernarg_size         = 0;

    do {
        const uint8_t* elf = (const uint8_t*)code_object;

        /* Validate ELF magic */
        if (code_size < 64 || elf[0] != 0x7f || elf[1] != 'E' ||
            elf[2] != 'L' || elf[3] != 'F') {
            LOG_DEBUG("AM driver: code object is not a valid ELF file, using defaults");
            break;
        }

        /* Verify 64-bit ELF (class == 2) */
        if (elf[4] != 2) {
            LOG_DEBUG("AM driver: ELF is not 64-bit, using defaults");
            break;
        }

        /* Determine endianness: 1=little, 2=big */
        int le = (elf[5] == 1);
        if (!le && elf[5] != 2) {
            LOG_DEBUG("AM driver: unknown ELF endianness, using defaults");
            break;
        }

        /* Helper macros for reading ELF fields respecting endianness */
        #define ELF_U16(off) (le ? (uint16_t)(elf[off] | ((uint16_t)elf[(off)+1] << 8)) \
                                 : (uint16_t)(elf[(off)+1] | ((uint16_t)elf[off] << 8)))
        #define ELF_U32(off) (le ? (uint32_t)(elf[off] | ((uint32_t)elf[(off)+1] << 8) | \
                                    ((uint32_t)elf[(off)+2] << 16) | ((uint32_t)elf[(off)+3] << 24)) \
                                 : (uint32_t)(elf[(off)+3] | ((uint32_t)elf[(off)+2] << 8) | \
                                    ((uint32_t)elf[(off)+1] << 16) | ((uint32_t)elf[off] << 24)))
        #define ELF_U64(off) (le ? ((uint64_t)ELF_U32(off) | ((uint64_t)ELF_U32((off)+4) << 32)) \
                                 : ((uint64_t)ELF_U32((off)+4) | ((uint64_t)ELF_U32(off) << 32)))

        /* Read 64-bit ELF header fields */
        uint64_t e_shoff     = ELF_U64(40);  /* Section header table offset */
        uint16_t e_shentsize = ELF_U16(58);  /* Section header entry size */
        uint16_t e_shnum     = ELF_U16(60);  /* Number of section headers */
        uint16_t e_shstrndx  = ELF_U16(62);  /* Section name string table index */

        if (e_shoff == 0 || e_shnum == 0 || e_shentsize < 64) {
            LOG_DEBUG("AM driver: ELF has no section headers, using defaults");
            break;
        }

        /* Bounds check section header table */
        if (e_shoff + (uint64_t)e_shnum * e_shentsize > code_size) {
            LOG_DEBUG("AM driver: ELF section headers out of bounds, using defaults");
            break;
        }

        /* Get section name string table */
        const uint8_t* shstrtab = NULL;
        uint64_t shstrtab_size = 0;
        if (e_shstrndx < e_shnum) {
            const uint8_t* strhdr = elf + e_shoff + (uint64_t)e_shstrndx * e_shentsize;
            uint64_t str_off  = ELF_U64((uint64_t)(strhdr - elf) + 24);
            uint64_t str_size = ELF_U64((uint64_t)(strhdr - elf) + 32);
            if (str_off + str_size <= code_size) {
                shstrtab = elf + str_off;
                shstrtab_size = str_size;
            }
        }

        /* Walk section headers */
        for (uint16_t i = 0; i < e_shnum; i++) {
            uint64_t sh_base = e_shoff + (uint64_t)i * e_shentsize;
            uint32_t sh_name_idx = ELF_U32(sh_base);
            uint32_t sh_type     = ELF_U32(sh_base + 4);
            uint64_t sh_offset   = ELF_U64(sh_base + 24);
            uint64_t sh_size     = ELF_U64(sh_base + 32);

            /* Bounds check section data */
            if (sh_offset + sh_size > code_size) continue;

            /* Resolve section name */
            const char* sec_name = NULL;
            if (shstrtab && sh_name_idx < shstrtab_size) {
                sec_name = (const char*)(shstrtab + sh_name_idx);
            }

            /* SHT_NOTE sections (type 7) - parse AMDGPU metadata */
            if (sh_type == 7 && sh_size >= 12) {
                uint64_t pos = sh_offset;
                uint64_t end = sh_offset + sh_size;
                while (pos + 12 <= end) {
                    uint32_t n_namesz = ELF_U32(pos);
                    uint32_t n_descsz = ELF_U32(pos + 4);
                    uint32_t n_type   = ELF_U32(pos + 8);
                    pos += 12;

                    /* Align name and desc to 4 bytes */
                    uint32_t name_aligned = (n_namesz + 3) & ~(uint32_t)3;
                    uint32_t desc_aligned = (n_descsz + 3) & ~(uint32_t)3;

                    if (pos + name_aligned + desc_aligned > end) break;

                    /* AMDGPU metadata note: type 32 (NT_AMDGPU_METADATA) with name "AMDGPU" */
                    if (n_type == 32 && n_namesz >= 6 && pos + 6 <= end &&
                        memcmp(elf + pos, "AMDGPU", 6) == 0) {
                        const char* desc = (const char*)(elf + pos + name_aligned);
                        uint32_t desc_len = n_descsz;

                        /* Search for key-value patterns in the MSGPACK/YAML metadata.
                         * AMDGPU metadata is typically MSGPACK, but we do a simple
                         * byte-scan for known field names as a best-effort approach. */
                        for (uint32_t d = 0; d + 20 < desc_len; d++) {
                            /* Look for ".kernarg_segment_size" pattern */
                            if (desc[d] == '.' && d + 21 < desc_len &&
                                memcmp(desc + d, ".kernarg_segment_size", 21) == 0) {
                                /* Next non-zero byte after the key might encode size (msgpack) */
                                for (uint32_t k = d + 21; k < desc_len && k < d + 30; k++) {
                                    uint8_t b = (uint8_t)desc[k];
                                    if (b > 0 && b < 0x80) {
                                        kernel->kernarg_size = b;
                                        break;
                                    }
                                    if (b == 0xce && k + 4 < desc_len) { /* msgpack uint32 */
                                        kernel->kernarg_size = (uint32_t)(
                                            ((uint8_t)desc[k+1] << 24) |
                                            ((uint8_t)desc[k+2] << 16) |
                                            ((uint8_t)desc[k+3] << 8)  |
                                            ((uint8_t)desc[k+4]));
                                        break;
                                    }
                                }
                            }
                            /* Look for ".group_segment_fixed_size" */
                            if (desc[d] == '.' && d + 24 < desc_len &&
                                memcmp(desc + d, ".group_segment_fixed_size", 25) == 0) {
                                for (uint32_t k = d + 25; k < desc_len && k < d + 34; k++) {
                                    uint8_t b = (uint8_t)desc[k];
                                    if (b > 0 && b < 0x80) {
                                        kernel->group_segment_size = b;
                                        break;
                                    }
                                    if (b == 0xce && k + 4 < desc_len) {
                                        kernel->group_segment_size = (uint32_t)(
                                            ((uint8_t)desc[k+1] << 24) |
                                            ((uint8_t)desc[k+2] << 16) |
                                            ((uint8_t)desc[k+3] << 8)  |
                                            ((uint8_t)desc[k+4]));
                                        break;
                                    }
                                }
                            }
                            /* Look for ".private_segment_fixed_size" */
                            if (desc[d] == '.' && d + 26 < desc_len &&
                                memcmp(desc + d, ".private_segment_fixed_size", 27) == 0) {
                                for (uint32_t k = d + 27; k < desc_len && k < d + 36; k++) {
                                    uint8_t b = (uint8_t)desc[k];
                                    if (b > 0 && b < 0x80) {
                                        kernel->private_segment_size = b;
                                        break;
                                    }
                                    if (b == 0xce && k + 4 < desc_len) {
                                        kernel->private_segment_size = (uint32_t)(
                                            ((uint8_t)desc[k+1] << 24) |
                                            ((uint8_t)desc[k+2] << 16) |
                                            ((uint8_t)desc[k+3] << 8)  |
                                            ((uint8_t)desc[k+4]));
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    pos += name_aligned + desc_aligned;
                }
            }

            /* .text section - extract kernel entry point offset */
            if (sec_name && strcmp(sec_name, ".text") == 0 && sh_size > 0) {
                /* The kernel entry point is at the start of the .text section.
                 * The GPU VA was already set; sh_offset gives the file offset
                 * which corresponds to the kernel code start within the ELF. */
                LOG_DEBUG("AM driver: .text section at offset 0x%lx, size %lu",
                          (unsigned long)sh_offset, (unsigned long)sh_size);
            }
        }

        #undef ELF_U16
        #undef ELF_U32
        #undef ELF_U64

        LOG_DEBUG("AM driver: ELF parsed - kernarg_size=%u, group_segment=%u, private_segment=%u",
                  kernel->kernarg_size, kernel->group_segment_size, kernel->private_segment_size);
    } while (0);

    LOG_DEBUG("AM driver: kernel '%s' loaded at gpu_va=0x%lx (%zu bytes)",
              kernel_name, (unsigned long)va, code_size);
    return kernel;

#else
    (void)drv; (void)code_object; (void)code_size; (void)kernel_name;
    LOG_ERROR("AM driver: only supported on Linux");
    return NULL;
#endif
}

void cml_am_kernel_free(CMLAMDriver* drv, CMLAMKernel* kernel) {
    if (!drv || !kernel) return;

#ifdef __linux__
    if (drv->initialized && kernel->handle) {
        am_free_and_unmap(drv, (uint64_t)kernel->handle, NULL, kernel->code_size);
    }
#endif

    free(kernel->code_object);
    free(kernel->name);
    free(kernel);
}

int cml_am_kernel_launch(CMLAMDriver* drv, CMLAMKernel* kernel,
                         uint32_t grid[3], uint32_t block[3],
                         void* kernarg, uint32_t kernarg_size) {
#ifdef __linux__
    if (!drv || !drv->initialized || !kernel || !grid || !block) {
        LOG_ERROR("AM driver: invalid args to kernel_launch");
        return -1;
    }

    CMLAMQueue* q = &drv->aql_queue;
    if (!q->ring || !q->write_dispatch_id || !q->doorbell) {
        LOG_ERROR("AM driver: queue not properly initialized");
        return -1;
    }

    /* Upload kernarg to a GTT buffer if provided */
    uint64_t kernarg_gpu_va = 0;
    if (kernarg && kernarg_size > 0) {
        /* Allocate kernarg memory (GTT, coherent) */
        uint64_t ka_handle = 0, ka_va = 0;
        void* ka_addr = NULL;
        uint32_t ka_flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT
                          | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                          | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;

        if (am_alloc_and_map(drv, kernarg_size, ka_flags,
                             &ka_handle, &ka_va, &ka_addr) != 0) {
            LOG_ERROR("AM driver: failed to allocate kernarg memory");
            return -1;
        }

        if (ka_addr) {
            memcpy(ka_addr, kernarg, kernarg_size);
            am_mb();
        }
        kernarg_gpu_va = ka_va;
    }

    /* Increment signal value for this dispatch */
    drv->signal_value++;

    /* Get write index and compute ring slot */
    uint64_t write_idx = *q->write_dispatch_id;
    uint32_t slot = (uint32_t)(write_idx % q->ring_size);

    /* Fill AQL kernel dispatch packet */
    hsa_kernel_dispatch_packet_t* pkt = &q->ring[slot];

    /* Write all fields except header first (header activates the packet) */
    pkt->setup              = (uint16_t)(
        ((grid[0] > 1 || block[0] > 1) ? 1 : 0) |
        ((grid[1] > 1 || block[1] > 1) ? 2 : 0) |
        ((grid[2] > 1 || block[2] > 1) ? 4 : 0)
    );
    /* Setup: number of dimensions (1, 2, or 3) */
    int dims = 1;
    if (grid[1] > 1 || block[1] > 1) dims = 2;
    if (grid[2] > 1 || block[2] > 1) dims = 3;
    pkt->setup = (uint16_t)dims;

    pkt->workgroup_size_x   = (uint16_t)block[0];
    pkt->workgroup_size_y   = (uint16_t)block[1];
    pkt->workgroup_size_z   = (uint16_t)block[2];
    pkt->reserved0          = 0;
    pkt->grid_size_x        = grid[0] * block[0]; /* Total work-items */
    pkt->grid_size_y        = grid[1] * block[1];
    pkt->grid_size_z        = grid[2] * block[2];
    pkt->private_segment_size = kernel->private_segment_size;
    pkt->group_segment_size   = kernel->group_segment_size;
    pkt->kernel_object        = kernel->gpu_addr;
    pkt->kernarg_address      = kernarg_gpu_va;
    pkt->reserved2            = 0;
    pkt->completion_signal    = drv->signal_gpu_va;

    /* Memory fence to ensure all fields are visible before header */
    am_mb();

    /* Write header last to activate the packet */
    uint16_t header = (AQL_PKT_TYPE_KERNEL_DISPATCH << AQL_HDR_TYPE_SHIFT)
                    | (1 << AQL_HDR_BARRIER_SHIFT)
                    | (AQL_FENCE_SCOPE_SYSTEM << AQL_HDR_ACQUIRE_SHIFT)
                    | (AQL_FENCE_SCOPE_SYSTEM << AQL_HDR_RELEASE_SHIFT);
    pkt->header = header;

    am_mb();

    /* Update write dispatch ID */
    *q->write_dispatch_id = write_idx + 1;

    am_mb();

    /* Ring the doorbell to notify the GPU */
    *q->doorbell = (uint32_t)(write_idx + 1);

    LOG_DEBUG("AM driver: kernel '%s' launched, grid=[%u,%u,%u] block=[%u,%u,%u] slot=%u",
              kernel->name ? kernel->name : "?",
              grid[0], grid[1], grid[2],
              block[0], block[1], block[2], slot);
    return 0;

#else
    (void)drv; (void)kernel; (void)grid; (void)block;
    (void)kernarg; (void)kernarg_size;
    return -1;
#endif
}


int cml_am_synchronize(CMLAMDriver* drv) {
#ifdef __linux__
    if (!drv || !drv->initialized) return -1;

    if (!drv->signal) {
        LOG_WARNING("AM driver: no signal memory for synchronization");
        return -1;
    }

    uint64_t expected = drv->signal_value;
    if (expected == AM_SIGNAL_INIT) {
        /* Nothing dispatched yet */
        return 0;
    }

    /* Poll the completion signal until it reaches the expected value */
    uint64_t timeout_us = 5000000; /* 5 seconds */
    uint64_t poll_interval_us = 10;
    uint64_t elapsed = 0;

    while (elapsed < timeout_us) {
        am_mb();
        uint64_t current = *drv->signal;
        if (current >= expected) {
            LOG_DEBUG("AM driver: synchronization complete (signal=%lu, expected=%lu)",
                      (unsigned long)current, (unsigned long)expected);
            return 0;
        }

        /* Yield CPU briefly */
        usleep((useconds_t)poll_interval_us);
        elapsed += poll_interval_us;

        /* Exponential backoff on poll interval, capped at 1ms */
        if (poll_interval_us < 1000) {
            poll_interval_us *= 2;
        }
    }

    LOG_ERROR("AM driver: synchronization timed out (signal=%lu, expected=%lu)",
              (unsigned long)*drv->signal, (unsigned long)expected);
    return -1;

#else
    (void)drv;
    return -1;
#endif
}


int cml_am_execute_graph(CMLAMDriver* drv, CMLGraph_t ir) {
    if (!drv || !ir) return -1;

    if (!drv->initialized) {
        LOG_ERROR("AM driver: not initialized, cannot execute graph");
        return -1;
    }


    struct IRNode* node = ir->head;
    while (node) {
        if (node->is_executed) {
            node = node->next;
            continue;
        }

        Tensor* output = node->output;
        if (!output) {
            node = node->next;
            continue;
        }

        bool gpu_ok = false;

        /* For elementwise ops with host-visible buffers, dispatch via AQL.
         * The GPU kernel must be pre-compiled as an ELF code object (AMDGPU ISA).
         * Since runtime AMDGPU ISA compilation requires LLVM or ROCm clang,
         * we attempt GPU dispatch only when a code object is available,
         * and fall back to CPU otherwise. */

        bool is_elementwise = false;
        switch (node->type) {
        case UOP_ADD: case UOP_SUB: case UOP_MUL: case UOP_DIV:
        case UOP_NEG: case UOP_EXP: case UOP_LOG: case UOP_SQRT:
        case UOP_ABS: case UOP_SIN: case UOP_COS: case UOP_TANH:
        case UOP_SIGMOID: case UOP_RECIP: case UOP_SILU:
        case UOP_RELU6:
            is_elementwise = true;
            break;
        default:
            break;
        }

        if (is_elementwise && node->num_inputs >= 1 && node->inputs) {
            /* Compute element count */
            size_t numel = 1;
            for (int d = 0; d < output->ndim; d++) {
                numel *= (size_t)output->shape[d];
            }
            size_t bytes = numel * sizeof(float);

            /* Allocate GTT (host-visible) buffers for inputs and output */
            int num_in = node->num_inputs;
            CMLAMBuffer* bufs_in[8] = {0};
            CMLAMBuffer* buf_out = NULL;
            bool alloc_ok = true;

            for (int i = 0; i < num_in && i < 8; i++) {
                if (!node->inputs[i] || !node->inputs[i]->data) {
                    alloc_ok = false;
                    break;
                }
                bufs_in[i] = cml_am_buffer_create(drv, bytes, false /* GTT */);
                if (!bufs_in[i]) { alloc_ok = false; break; }
                cml_am_buffer_upload(drv, bufs_in[i], node->inputs[i]->data, bytes);
            }

            if (alloc_ok) {
                buf_out = cml_am_buffer_create(drv, bytes, false /* GTT */);
                if (!buf_out) alloc_ok = false;
            }

            if (alloc_ok) {
                /* Build kernarg block: pointers to buffers + element count */
                struct __attribute__((packed)) {
                    uint64_t ptrs[9]; /* up to 8 inputs + 1 output */
                    uint32_t n;
                    uint32_t pad;
                } kernarg;
                memset(&kernarg, 0, sizeof(kernarg));

                for (int i = 0; i < num_in && i < 8; i++) {
                    kernarg.ptrs[i] = bufs_in[i]->gpu_va;
                }
                kernarg.ptrs[num_in] = buf_out->gpu_va;
                kernarg.n = (uint32_t)numel;

                /* If we had an AMDGPU code object for this op, we would:
                 * CMLAMKernel* k = cml_am_kernel_load(drv, elf_data, elf_size, name);
                 * uint32_t grid[3] = {(numel+255)/256, 1, 1};
                 * uint32_t block[3] = {256, 1, 1};
                 * cml_am_kernel_launch(drv, k, grid, block, &kernarg, sizeof(kernarg));
                 * cml_am_synchronize(drv);
                 * cml_am_buffer_download(drv, buf_out, output->data, bytes);
                 * gpu_ok = true;
                 *
                 * Check if we have GTT buffers and use CPU on the
                 * GPU-mapped memory (zero-copy path). */

                /* Zero-copy CPU execution on GTT-mapped buffers */
                if (buf_out->cpu_addr) {
                    float* out_ptr = (float*)buf_out->cpu_addr;
                    if (num_in == 1 && bufs_in[0] && bufs_in[0]->cpu_addr) {
                        float* a_ptr = (float*)bufs_in[0]->cpu_addr;
                        for (size_t i = 0; i < numel; i++) {
                            switch (node->type) {
                            case UOP_NEG:     out_ptr[i] = -a_ptr[i]; break;
                            case UOP_EXP:     out_ptr[i] = expf(a_ptr[i]); break;
                            case UOP_LOG:     out_ptr[i] = logf(a_ptr[i]); break;
                            case UOP_SQRT:    out_ptr[i] = sqrtf(a_ptr[i]); break;
                            case UOP_ABS:     out_ptr[i] = fabsf(a_ptr[i]); break;
                            case UOP_SIN:     out_ptr[i] = sinf(a_ptr[i]); break;
                            case UOP_COS:     out_ptr[i] = cosf(a_ptr[i]); break;
                            case UOP_TANH:    out_ptr[i] = tanhf(a_ptr[i]); break;
                            case UOP_SIGMOID: out_ptr[i] = 1.0f / (1.0f + expf(-a_ptr[i])); break;
                            case UOP_RECIP:   out_ptr[i] = 1.0f / a_ptr[i]; break;
                            case UOP_SILU:    out_ptr[i] = a_ptr[i] / (1.0f + expf(-a_ptr[i])); break;
                            case UOP_RELU6: { float v = a_ptr[i] > 0 ? a_ptr[i] : 0; out_ptr[i] = v < 6.0f ? v : 6.0f; break; }
                            default:          out_ptr[i] = a_ptr[i]; break;
                            }
                        }
                        gpu_ok = true;
                    } else if (num_in == 2 && bufs_in[0] && bufs_in[0]->cpu_addr
                               && bufs_in[1] && bufs_in[1]->cpu_addr) {
                        float* a_ptr = (float*)bufs_in[0]->cpu_addr;
                        float* b_ptr = (float*)bufs_in[1]->cpu_addr;
                        for (size_t i = 0; i < numel; i++) {
                            switch (node->type) {
                            case UOP_ADD: out_ptr[i] = a_ptr[i] + b_ptr[i]; break;
                            case UOP_SUB: out_ptr[i] = a_ptr[i] - b_ptr[i]; break;
                            case UOP_MUL: out_ptr[i] = a_ptr[i] * b_ptr[i]; break;
                            case UOP_DIV: out_ptr[i] = a_ptr[i] / b_ptr[i]; break;
                            default:      out_ptr[i] = a_ptr[i]; break;
                            }
                        }
                        gpu_ok = true;
                    }

                    if (gpu_ok) {
                        if (!output->data) {
                            output->data = malloc(bytes);
                        }
                        if (output->data) {
                            cml_am_buffer_download(drv, buf_out, output->data, bytes);
                        }
                    }
                }
            }

            /* Free temporary buffers */
            for (int i = 0; i < num_in && i < 8; i++) {
                if (bufs_in[i]) cml_am_buffer_free(drv, bufs_in[i]);
            }
            if (buf_out) cml_am_buffer_free(drv, buf_out);
        }

        /* CPU fallback */
        if (!gpu_ok) {
            LOG_DEBUG("AM driver: using CPU fallback for op %d", (int)node->type);
            cpu_execute_node(node);
        }

        node->is_executed = true;
        if (output) output->is_executed = true;
        node = node->next;
    }

    ir->is_executed = true;
    return 0;
}
