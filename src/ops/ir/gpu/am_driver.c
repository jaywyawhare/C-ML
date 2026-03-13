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
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <errno.h>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

/* ======================================================================
 * KFD ioctl structures (minimal, matching kernel headers)
 * ====================================================================== */

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

/* ======================================================================
 * Helper: KFD ioctl wrapper
 * ====================================================================== */

#ifdef __linux__

static int kfd_ioctl(int fd, unsigned long request, void* arg) {
    int ret;
    do {
        ret = ioctl(fd, request, arg);
    } while (ret == -1 && errno == EINTR);

    if (ret == -1) {
        LOG_DEBUG("AM driver: KFD ioctl 0x%lx failed: %s (errno=%d)",
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

/* ======================================================================
 * Driver availability check
 * ====================================================================== */

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
        LOG_DEBUG("AM driver: KFD GET_VERSION ioctl failed");
        return false;
    }

    LOG_DEBUG("AM driver: KFD version %u.%u detected",
              ver.major_version, ver.minor_version);
    return true;
#else
    /* KFD is Linux-only */
    return false;
#endif
}

/* ======================================================================
 * Driver lifecycle
 * ====================================================================== */

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

    if (drv->initialized) {
        LOG_DEBUG("AM driver: already initialized");
        return 0;
    }

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
         * and free them here.  For now, the kernel will clean up on process
         * exit or when the KFD file descriptor is closed. */
    }

    if (drv->fd_drm >= 0) close(drv->fd_drm);
    if (drv->fd_kfd >= 0) close(drv->fd_kfd);
#endif

    free(drv);
}

/* ======================================================================
 * Buffer management
 * ====================================================================== */

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

    /* TODO: Enqueue a DMA copy from staging->gpu_va to dst->gpu_va.
     * For now, use the staging buffer approach which requires the GPU
     * to support GTT access for the copy kernel. */
    LOG_WARNING("AM driver: VRAM upload via staging not fully implemented");

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

    /* VRAM buffer without CPU mapping: need staging */
    LOG_WARNING("AM driver: VRAM download via staging not fully implemented");
    return -1;

#else
    (void)drv; (void)src; (void)dst; (void)n;
    return -1;
#endif
}

/* ======================================================================
 * Kernel management
 * ====================================================================== */

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

    /* TODO: Parse ELF code object to extract:
     *   - kernel entry point offset
     *   - group_segment_size, private_segment_size, kernarg_size
     * For now, use defaults. */
    kernel->group_segment_size   = 0;
    kernel->private_segment_size = 0;
    kernel->kernarg_size         = 0;

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

/* ======================================================================
 * Synchronization
 * ====================================================================== */

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

/* ======================================================================
 * Graph execution (stub)
 * ====================================================================== */

int cml_am_execute_graph(CMLAMDriver* drv, CMLGraph_t ir) {
    (void)ir;

    if (!drv) {
        LOG_ERROR("AM driver: execute_graph called with NULL driver");
        return -1;
    }

    LOG_WARNING("AM driver: execute_graph not yet implemented");
    return -1;
}
