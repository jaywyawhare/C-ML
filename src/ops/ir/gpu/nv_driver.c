#include "ops/ir/gpu/nv_driver.h"
#include "ops/ir/gpu/nv_qmd.h"
#include "ops/ir/internal.h"
#include "core/logging.h"

#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#endif

#ifdef CML_NV_MOCK_GPU
#include "ops/ir/gpu/nv_mock.h"
#define open(...)   cml_nv_mock_open(__VA_ARGS__)
#define close(...)  cml_nv_mock_close(__VA_ARGS__)
#define ioctl(...)  cml_nv_mock_ioctl(__VA_ARGS__)
#define mmap(...)   cml_nv_mock_mmap(__VA_ARGS__)
#define munmap(...) cml_nv_mock_munmap(__VA_ARGS__)
#endif

#define NV_PUSHBUF_DWORDS  4096
#define NV_ALIGN(x, a)     (((x) + ((a)-1)) & ~((uint64_t)(a)-1))

#ifdef __linux__

static uint32_t g_nv_next_handle = 0x10000;

typedef struct {
    uint32_t hRoot;
    uint32_t hObjectParent;
    uint32_t hObjectNew;
    uint32_t hClass;
    void*    pAllocParms;
    uint32_t status;
} NV_RM_ALLOC_PARAMS;

typedef struct {
    uint32_t hClient;
    uint32_t hObject;
    uint32_t cmd;
    uint32_t flags;
    void*    params;
    uint32_t paramsSize;
    uint32_t status;
} NV_RM_CONTROL_PARAMS;

typedef struct {
    uint32_t hRoot;
    uint32_t hObjectParent;
    uint32_t hObjectOld;
    uint32_t status;
} NV_RM_FREE_PARAMS;

#define NV_IOCTL_RM_ALLOC    _IOWR('F', NV_ESC_RM_ALLOC,   NV_RM_ALLOC_PARAMS)
#define NV_IOCTL_RM_CONTROL  _IOWR('F', NV_ESC_RM_CONTROL,  NV_RM_CONTROL_PARAMS)
#define NV_IOCTL_RM_FREE     _IOWR('F', NV_ESC_RM_FREE,     NV_RM_FREE_PARAMS)

/* RM alloc parameter structures for specific classes */

typedef struct {
    uint32_t hVASpace;
    uint32_t index;
    uint32_t flags;
    uint64_t vaSize;
    uint64_t vaStartInternal;
    uint64_t vaLimitInternal;
    uint32_t bigPageSize;
    uint64_t vaBase;
} NV_VASPACE_ALLOC_PARAMS;

typedef struct {
    uint32_t hObjectError;
    uint32_t hObjectBuffer;
    uint64_t gpFifoOffset;
    uint32_t gpFifoEntries;
    uint32_t flags;
    uint32_t hContextShare;
    uint32_t hVASpace;
    uint32_t hUserd;
    uint64_t userdOffset;
    uint32_t engineType;
    uint32_t subDeviceId;
    uint32_t internalFlags;
} NV_CHANNEL_ALLOC_PARAMS;

typedef struct {
    uint32_t engineType;
    uint32_t hVASpace;
} NV_CHANNEL_GROUP_ALLOC_PARAMS;

typedef struct {
    uint32_t attr;
    uint32_t attr2;
    uint32_t flags;
    uint32_t format;
    uint32_t height;
    uint32_t width;
    uint32_t size;
    uint32_t alignment;
    uint64_t offset;
    uint64_t limit;
    uint32_t hVASpace;
    uint32_t pteKind;
    uint32_t compTag;
    uint64_t address;
    uint32_t cTag;
} NV_MEMORY_ALLOC_PARAMS;

typedef struct {
    uint32_t type;
    uint32_t data;
} NV_GPU_INFO_ENTRY;

typedef struct {
    uint32_t gpuInfoListSize;
    NV_GPU_INFO_ENTRY *gpuInfoList;
} NV2080_CTRL_GPU_GET_INFO_PARAMS;

typedef struct {
    uint32_t gpuNameStringFlags;
    char     gpuNameString[256];
} NV2080_CTRL_GPU_GET_NAME_STRING_PARAMS;

typedef struct {
    uint32_t grInfoListSize;
    void     *grInfoList;
} NV2080_CTRL_GR_GET_INFO_PARAMS;

#define NV2080_GPU_INFO_INDEX_GPU_FLA_CAPABILITY 48
#define NV2080_GPU_INFO_INDEX_MINOR_REVISION_EXT 38
#define NV2080_GPU_INFO_INDEX_GPU_ARCH           52

static int nv_rm_alloc(int fd, uint32_t client, uint32_t parent,
                       uint32_t *out_handle, uint32_t nv_class,
                       void *alloc_params) {
    NV_RM_ALLOC_PARAMS p;
    memset(&p, 0, sizeof(p));
    p.hRoot         = client;
    p.hObjectParent = parent;
    p.hObjectNew    = *out_handle;
    p.hClass        = nv_class;
    p.pAllocParms   = alloc_params;

    int ret = ioctl(fd, NV_IOCTL_RM_ALLOC, &p);
    if (ret < 0) {
        LOG_DEBUG("NV RM alloc class 0x%08X failed: %s", nv_class, strerror(errno));
        return -1;
    }
    if (p.status != 0) {
        LOG_DEBUG("NV RM alloc class 0x%08X returned status 0x%X", nv_class, p.status);
        return -1;
    }
    *out_handle = p.hObjectNew;
    return 0;
}

static int nv_rm_control(int fd, uint32_t client, uint32_t object,
                         uint32_t cmd, void *params, uint32_t params_size) {
    NV_RM_CONTROL_PARAMS p;
    memset(&p, 0, sizeof(p));
    p.hClient    = client;
    p.hObject    = object;
    p.cmd        = cmd;
    p.params     = params;
    p.paramsSize = params_size;

    int ret = ioctl(fd, NV_IOCTL_RM_CONTROL, &p);
    if (ret < 0) {
        LOG_DEBUG("NV RM control cmd 0x%08X failed: %s", cmd, strerror(errno));
        return -1;
    }
    if (p.status != 0) {
        LOG_DEBUG("NV RM control cmd 0x%08X returned status 0x%X", cmd, p.status);
        return -1;
    }
    return 0;
}

static int nv_rm_free(int fd, uint32_t client, uint32_t parent, uint32_t handle) {
    NV_RM_FREE_PARAMS p;
    memset(&p, 0, sizeof(p));
    p.hRoot         = client;
    p.hObjectParent = parent;
    p.hObjectOld    = handle;

    int ret = ioctl(fd, NV_IOCTL_RM_FREE, &p);
    if (ret < 0) {
        LOG_DEBUG("NV RM free handle 0x%X failed: %s", handle, strerror(errno));
        return -1;
    }
    return 0;
}

static uint64_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

static uint64_t nv_va_alloc(CMLNVDriver *drv, size_t size, size_t align) {
    uint64_t aligned_size = NV_ALIGN(size, align > 0 ? align : 4096);
    uint64_t va = NV_ALIGN(drv->va_current, align > 0 ? align : 4096);
    if (va + aligned_size > drv->va_end) {
        LOG_ERROR("NV driver: GPU VA space exhausted");
        return 0;
    }
    drv->va_current = va + aligned_size;
    return va;
}

static uint32_t nv_gpfifo_class_for_arch(uint32_t arch) {
    if (arch >= NV_GPU_ARCH_BLACKWELL) return BLACKWELL_CHANNEL_GPFIFO_A;
    if (arch >= NV_GPU_ARCH_HOPPER) return HOPPER_CHANNEL_GPFIFO_A;
    if (arch >= NV_GPU_ARCH_AMPERE) return AMPERE_CHANNEL_GPFIFO_A;
    return TURING_CHANNEL_GPFIFO_A;
}

static uint32_t nv_compute_class_for_arch(uint32_t arch) {
    if (arch >= NV_GPU_ARCH_BLACKWELL) return BLACKWELL_COMPUTE_A;
    if (arch >= NV_GPU_ARCH_HOPPER) return HOPPER_COMPUTE_A;
    if (arch >= NV_GPU_ARCH_AMPERE) return AMPERE_COMPUTE_A;
    return TURING_COMPUTE_A;
}

static uint32_t nv_copy_class_for_arch(uint32_t arch) {
    if (arch >= NV_GPU_ARCH_BLACKWELL) return BLACKWELL_DMA_COPY_A;
    if (arch >= NV_GPU_ARCH_HOPPER) return HOPPER_DMA_COPY_A;
    if (arch >= NV_GPU_ARCH_AMPERE) return AMPERE_DMA_COPY_A;
    return TURING_DMA_COPY_A;
}

static int nv_query_gpu_info(CMLNVDriver *drv) {
    NV2080_CTRL_GPU_GET_NAME_STRING_PARAMS name_params;
    memset(&name_params, 0, sizeof(name_params));

    if (nv_rm_control(drv->fd_ctl, drv->client_handle, drv->subdevice_handle,
                      NV2080_CTRL_CMD_GPU_GET_NAME_STRING,
                      &name_params, sizeof(name_params)) == 0) {
        strncpy(drv->device_name, name_params.gpuNameString, sizeof(drv->device_name) - 1);
        drv->device_name[sizeof(drv->device_name) - 1] = '\0';
    }

    NV_GPU_INFO_ENTRY info_list[4];
    memset(info_list, 0, sizeof(info_list));
    info_list[0].type = NV2080_GPU_INFO_INDEX_GPU_ARCH;

    NV2080_CTRL_GPU_GET_INFO_PARAMS info_params;
    memset(&info_params, 0, sizeof(info_params));
    info_params.gpuInfoListSize = 1;
    info_params.gpuInfoList = info_list;

    if (nv_rm_control(drv->fd_ctl, drv->client_handle, drv->subdevice_handle,
                      NV2080_CTRL_CMD_GPU_GET_INFO,
                      &info_params, sizeof(info_params)) == 0) {
        drv->gpu_arch = info_list[0].data;
    }

    if (drv->gpu_arch == 0) {
        drv->gpu_arch = NV_GPU_ARCH_TURING;
    }

    if (drv->compute_cap_major == 0) {
        if (drv->gpu_arch >= NV_GPU_ARCH_BLACKWELL) {
            drv->compute_cap_major = 10;
            drv->compute_cap_minor = 0;
        } else if (drv->gpu_arch >= NV_GPU_ARCH_HOPPER) {
            drv->compute_cap_major = 9;
            drv->compute_cap_minor = 0;
        } else if (drv->gpu_arch >= NV_GPU_ARCH_ADA) {
            drv->compute_cap_major = 8;
            drv->compute_cap_minor = 9;
        } else if (drv->gpu_arch >= NV_GPU_ARCH_AMPERE) {
            drv->compute_cap_major = 8;
            drv->compute_cap_minor = 0;
        } else {
            drv->compute_cap_major = 7;
            drv->compute_cap_minor = 5;
        }
    }

    return 0;
}

static int nv_setup_vaspace(CMLNVDriver *drv) {
    NV_VASPACE_ALLOC_PARAMS va_params;
    memset(&va_params, 0, sizeof(va_params));
    va_params.index = 0;
    va_params.flags = 0;
    va_params.vaSize = drv->va_end - drv->va_start;
    va_params.vaBase = drv->va_start;
    va_params.bigPageSize = 0x20000;

    drv->vaspace_handle = g_nv_next_handle++;
    if (nv_rm_alloc(drv->fd_ctl, drv->client_handle, drv->device_handle,
                    &drv->vaspace_handle, FERMI_VASPACE_A, &va_params) != 0) {
        LOG_DEBUG("NV driver: VA space alloc via FERMI_VASPACE_A failed, using default");
        drv->vaspace_handle = 0;
    }

    return 0;
}

static CMLNVBuffer *nv_alloc_gpu_buffer(CMLNVDriver *drv, size_t size, bool host_visible) {
    CMLNVBuffer *buf = (CMLNVBuffer *)calloc(1, sizeof(CMLNVBuffer));
    if (!buf) return NULL;

    size_t aligned = NV_ALIGN(size, 4096);
    buf->size = aligned;
    buf->host_visible = host_visible;

    NV_MEMORY_ALLOC_PARAMS mem_params;
    memset(&mem_params, 0, sizeof(mem_params));
    mem_params.size = (uint32_t)aligned;
    mem_params.hVASpace = drv->vaspace_handle;

    if (host_visible) {
        mem_params.attr = 0x00010001;  /* UNCACHED, HOST_MEMORY */
        mem_params.flags = 0x00000001; /* PHYSICAL */
    } else {
        mem_params.attr = 0x00020001;  /* CACHED, VIDEO_MEMORY */
        mem_params.flags = 0x00000002;
        buf->is_vram = true;
    }

    buf->handle = g_nv_next_handle++;
    uint32_t mem_class = host_visible ? NV01_MEMORY_SYSTEM : NV01_MEMORY_LOCAL;

    if (nv_rm_alloc(drv->fd_ctl, drv->client_handle, drv->device_handle,
                    &buf->handle, mem_class, &mem_params) != 0) {
        if (!host_visible) {
            mem_params.attr = 0x00010001;
            mem_params.flags = 0x00000001;
            buf->handle = g_nv_next_handle++;
            buf->is_vram = false;

            if (nv_rm_alloc(drv->fd_ctl, drv->client_handle, drv->device_handle,
                            &buf->handle, NV01_MEMORY_SYSTEM, &mem_params) != 0) {
                goto fallback_mmap;
            }
        } else {
            goto fallback_mmap;
        }
    }

    buf->gpu_va = nv_va_alloc(drv, aligned, 4096);
    if (buf->gpu_va == 0) {
        nv_rm_free(drv->fd_ctl, drv->client_handle, drv->device_handle, buf->handle);
        free(buf);
        return NULL;
    }

    if (host_visible || !buf->is_vram) {
        buf->cpu_addr = mmap(NULL, aligned, PROT_READ | PROT_WRITE,
                             MAP_SHARED, drv->fd_dev, 0);
        if (buf->cpu_addr == MAP_FAILED) {
            buf->cpu_addr = mmap(NULL, aligned, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (buf->cpu_addr == MAP_FAILED) {
                buf->cpu_addr = NULL;
            }
        }
        if (buf->cpu_addr)
            memset(buf->cpu_addr, 0, aligned);
    }

    return buf;

fallback_mmap:
    buf->handle = 0;
    buf->gpu_va = nv_va_alloc(drv, aligned, 4096);
    if (buf->gpu_va == 0) {
        free(buf);
        return NULL;
    }
    if (host_visible) {
        buf->cpu_addr = mmap(NULL, aligned, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (buf->cpu_addr == MAP_FAILED) {
            buf->cpu_addr = NULL;
            free(buf);
            return NULL;
        }
        memset(buf->cpu_addr, 0, aligned);
    }
    return buf;
}

static int nv_setup_channel(CMLNVDriver *drv) {
    drv->gpfifo_buf = nv_alloc_gpu_buffer(drv,
        (size_t)NV_GPFIFO_DEFAULT_ENTRIES * NV_GPFIFO_ENTRY_BYTES, true);
    if (!drv->gpfifo_buf) {
        LOG_WARNING("NV driver: failed to allocate GPFIFO buffer");
        return -1;
    }

    drv->userd_buf = nv_alloc_gpu_buffer(drv, 4096, true);
    if (!drv->userd_buf) {
        LOG_WARNING("NV driver: failed to allocate USERD buffer");
        return -1;
    }

    NV_CHANNEL_GROUP_ALLOC_PARAMS cg_params;
    memset(&cg_params, 0, sizeof(cg_params));
    cg_params.engineType = 0x01; /* GR/COMPUTE */
    cg_params.hVASpace = drv->vaspace_handle;

    drv->channel_group_handle = g_nv_next_handle++;
    if (nv_rm_alloc(drv->fd_ctl, drv->client_handle, drv->device_handle,
                    &drv->channel_group_handle, KEPLER_CHANNEL_GROUP_A,
                    &cg_params) != 0) {
        LOG_DEBUG("NV driver: channel group alloc failed");
        drv->channel_group_handle = 0;
    }

    NV_CHANNEL_ALLOC_PARAMS ch_params;
    memset(&ch_params, 0, sizeof(ch_params));
    ch_params.gpFifoOffset = drv->gpfifo_buf->gpu_va;
    ch_params.gpFifoEntries = NV_GPFIFO_DEFAULT_ENTRIES;
    ch_params.hVASpace = drv->vaspace_handle;
    ch_params.hUserd = drv->userd_buf->handle;
    ch_params.userdOffset = 0;
    ch_params.engineType = 0x01;

    uint32_t gpfifo_class = nv_gpfifo_class_for_arch(drv->gpu_arch);
    uint32_t parent = drv->channel_group_handle ? drv->channel_group_handle : drv->device_handle;

    drv->channel_handle = g_nv_next_handle++;
    if (nv_rm_alloc(drv->fd_ctl, drv->client_handle, parent,
                    &drv->channel_handle, gpfifo_class, &ch_params) != 0) {
        LOG_WARNING("NV driver: channel alloc failed (class 0x%04X)", gpfifo_class);
        drv->channel_handle = 0;
    }

    drv->gpfifo.entries = (uint64_t *)drv->gpfifo_buf->cpu_addr;
    drv->gpfifo.num_entries = NV_GPFIFO_DEFAULT_ENTRIES;
    drv->gpfifo.put_offset = 0;
    drv->gpfifo.get_offset = 0;
    drv->gpfifo.gpu_va = drv->gpfifo_buf->gpu_va;
    drv->gpfifo.handle = drv->channel_handle;

    if (drv->userd_buf->cpu_addr) {
        drv->gpfifo.doorbell = (uint8_t *)drv->userd_buf->cpu_addr + NV_USERD_GP_PUT_OFFSET;
    }

    uint32_t compute_class = nv_compute_class_for_arch(drv->gpu_arch);
    drv->compute_obj_handle = g_nv_next_handle++;
    if (drv->channel_handle) {
        if (nv_rm_alloc(drv->fd_ctl, drv->client_handle, drv->channel_handle,
                        &drv->compute_obj_handle, compute_class, NULL) != 0) {
            LOG_DEBUG("NV driver: compute object alloc failed (class 0x%04X)", compute_class);
            drv->compute_obj_handle = 0;
        }
    }

    uint32_t copy_class = nv_copy_class_for_arch(drv->gpu_arch);
    drv->copy_obj_handle = g_nv_next_handle++;
    if (drv->channel_handle) {
        if (nv_rm_alloc(drv->fd_ctl, drv->client_handle, drv->channel_handle,
                        &drv->copy_obj_handle, copy_class, NULL) != 0) {
            LOG_DEBUG("NV driver: copy engine object alloc failed");
            drv->copy_obj_handle = 0;
        }
    }

    return 0;
}

static int nv_setup_pushbuf(CMLNVDriver *drv) {
    drv->pushbuf.backing = nv_alloc_gpu_buffer(drv,
        NV_PUSHBUF_DWORDS * sizeof(uint32_t), true);
    if (!drv->pushbuf.backing || !drv->pushbuf.backing->cpu_addr) {
        LOG_WARNING("NV driver: failed to allocate pushbuffer");
        return -1;
    }

    drv->pushbuf.buf = (uint32_t *)drv->pushbuf.backing->cpu_addr;
    drv->pushbuf.pos = 0;
    drv->pushbuf.capacity = NV_PUSHBUF_DWORDS;
    drv->pushbuf.gpu_va = drv->pushbuf.backing->gpu_va;

    return 0;
}

static int nv_setup_semaphore(CMLNVDriver *drv) {
    drv->semaphore_buf = nv_alloc_gpu_buffer(drv, 4096, true);
    if (!drv->semaphore_buf || !drv->semaphore_buf->cpu_addr) {
        drv->semaphore = (volatile uint32_t *)mmap(NULL, 4096,
            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (drv->semaphore == MAP_FAILED) {
            drv->semaphore = NULL;
            return -1;
        }
    } else {
        drv->semaphore = (volatile uint32_t *)drv->semaphore_buf->cpu_addr;
    }

    *drv->semaphore = 0;
    drv->semaphore_value = 0;

    if (drv->semaphore_buf)
        drv->semaphore_gpu_va = drv->semaphore_buf->gpu_va;
    else
        drv->semaphore_gpu_va = nv_va_alloc(drv, 4096, 4096);

    return 0;
}

static void pushbuf_reset(CMLNVPushbuf *pb) {
    pb->pos = 0;
}

static void pushbuf_emit(CMLNVPushbuf *pb, uint32_t val) {
    if (pb->pos < pb->capacity)
        pb->buf[pb->pos++] = val;
}

static void pushbuf_emit_method(CMLNVPushbuf *pb, int subchan, uint32_t reg, uint32_t count) {
    pushbuf_emit(pb, NV_FIFO_INCR(subchan, reg, count));
}

static void pushbuf_emit_data(CMLNVPushbuf *pb, const void *data, uint32_t dwords) {
    const uint32_t *src = (const uint32_t *)data;
    for (uint32_t i = 0; i < dwords; i++)
        pushbuf_emit(pb, src[i]);
}

static void nv_gpfifo_submit(CMLNVDriver *drv, uint64_t pb_gpu_va, uint32_t len_dwords) {
    if (!drv->gpfifo.entries) return;

    uint32_t put = drv->gpfifo.put_offset;
    drv->gpfifo.entries[put] = NV_GPFIFO_ENTRY(pb_gpu_va, len_dwords);

    put = (put + 1) % drv->gpfifo.num_entries;
    drv->gpfifo.put_offset = put;

    __sync_synchronize();

    if (drv->gpfifo.doorbell) {
        volatile uint32_t *bell = (volatile uint32_t *)drv->gpfifo.doorbell;
        *bell = put;
        __sync_synchronize();
    }
}

static void nv_push_semaphore_release(CMLNVPushbuf *pb, uint64_t sem_va, uint32_t value) {
    pushbuf_emit_method(pb, NVC0_SUBCHANNEL_COMPUTE, NVC3C0_SET_REPORT_SEMAPHORE_A, 4);
    pushbuf_emit(pb, (uint32_t)(sem_va >> 32));
    pushbuf_emit(pb, (uint32_t)(sem_va & 0xFFFFFFFF));
    pushbuf_emit(pb, value);
    pushbuf_emit(pb, NV_SEMAPHORE_RELEASE_WFI);
}

static void nv_push_semaphore_acquire(CMLNVPushbuf *pb, uint64_t sem_va, uint32_t value) {
    pushbuf_emit_method(pb, NVC0_SUBCHANNEL_COMPUTE, NVC3C0_SET_REPORT_SEMAPHORE_A, 4);
    pushbuf_emit(pb, (uint32_t)(sem_va >> 32));
    pushbuf_emit(pb, (uint32_t)(sem_va & 0xFFFFFFFF));
    pushbuf_emit(pb, value);
    pushbuf_emit(pb, NV_SEMAPHORE_ACQUIRE_GEQ);
}

static void nv_push_invalidate_caches(CMLNVPushbuf *pb) {
    pushbuf_emit_method(pb, NVC0_SUBCHANNEL_COMPUTE, NVC3C0_INVALIDATE_SHADER_CACHES, 1);
    pushbuf_emit(pb, 0x12);
}

#endif /* __linux__ */


bool cml_nv_driver_available(void) {
#ifdef CML_NV_MOCK_GPU
    return cml_nv_mock_get() != NULL;
#elif defined(__linux__)
    struct stat st;
    if (stat("/dev/nvidia0", &st) != 0)
        return false;
    int fd = open("/dev/nvidia0", O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return false;
    close(fd);
    return true;
#else
    return false;
#endif
}


CMLNVDriver* cml_nv_driver_create(void) {
    CMLNVDriver *drv = (CMLNVDriver *)calloc(1, sizeof(CMLNVDriver));
    if (!drv) {
        LOG_ERROR("NV driver: failed to allocate context");
        return NULL;
    }
    drv->fd_ctl = -1;
    drv->fd_dev = -1;
    drv->fd_uvm = -1;
    drv->va_start   = 0x100000000ULL;
    drv->va_current = drv->va_start;
    drv->va_end     = 0x800000000ULL;
    return drv;
}

int cml_nv_driver_init(CMLNVDriver *drv) {
    if (!drv) return -1;
    if (drv->initialized)
        return 0;

#ifndef __linux__
    LOG_ERROR("NV driver: only supported on Linux");
    return -1;
#else
    drv->fd_ctl = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    if (drv->fd_ctl < 0) {
        LOG_ERROR("NV driver: failed to open /dev/nvidiactl: %s", strerror(errno));
        return -1;
    }

    drv->fd_dev = open("/dev/nvidia0", O_RDWR | O_CLOEXEC);
    if (drv->fd_dev < 0) {
        LOG_ERROR("NV driver: failed to open /dev/nvidia0: %s", strerror(errno));
        close(drv->fd_ctl);
        drv->fd_ctl = -1;
        return -1;
    }

    drv->fd_uvm = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    if (drv->fd_uvm < 0) {
        LOG_DEBUG("NV driver: /dev/nvidia-uvm not available (non-fatal)");
        drv->fd_uvm = -1;
    }

    /* RM hierarchy: client -> device -> subdevice */
    drv->client_handle = g_nv_next_handle++;
    if (nv_rm_alloc(drv->fd_ctl, 0, 0,
                    &drv->client_handle, NV01_ROOT_CLIENT, NULL) != 0) {
        LOG_ERROR("NV driver: failed to create RM client");
        goto fail;
    }

    drv->device_handle = g_nv_next_handle++;
    if (nv_rm_alloc(drv->fd_ctl, drv->client_handle, drv->client_handle,
                    &drv->device_handle, NV01_DEVICE_0, NULL) != 0) {
        LOG_ERROR("NV driver: failed to create RM device");
        goto fail;
    }

    drv->subdevice_handle = g_nv_next_handle++;
    if (nv_rm_alloc(drv->fd_ctl, drv->client_handle, drv->device_handle,
                    &drv->subdevice_handle, NV20_SUBDEVICE_0, NULL) != 0) {
        LOG_ERROR("NV driver: failed to create RM subdevice");
        goto fail;
    }

    nv_query_gpu_info(drv);

    nv_setup_vaspace(drv);

    if (nv_setup_channel(drv) != 0)
        LOG_WARNING("NV driver: channel setup incomplete");

    if (nv_setup_pushbuf(drv) != 0)
        LOG_WARNING("NV driver: pushbuffer setup incomplete");

    if (nv_setup_semaphore(drv) != 0)
        LOG_WARNING("NV driver: semaphore setup incomplete");

    drv->initialized = true;
    LOG_INFO("NV driver: initialized (arch=0x%X, sm_%d%d, fd_ctl=%d, fd_dev=%d)",
             drv->gpu_arch, drv->compute_cap_major, drv->compute_cap_minor,
             drv->fd_ctl, drv->fd_dev);
    return 0;

fail:
    if (drv->fd_uvm >= 0) { close(drv->fd_uvm); drv->fd_uvm = -1; }
    if (drv->fd_dev >= 0) { close(drv->fd_dev); drv->fd_dev = -1; }
    if (drv->fd_ctl >= 0) { close(drv->fd_ctl); drv->fd_ctl = -1; }
    return -1;
#endif
}

void cml_nv_driver_free(CMLNVDriver *drv) {
    if (!drv) return;

#ifdef __linux__
    if (drv->initialized) {
        if (drv->semaphore_buf) {
            cml_nv_buffer_free(drv, drv->semaphore_buf);
            drv->semaphore_buf = NULL;
            drv->semaphore = NULL;
        } else if (drv->semaphore && drv->semaphore != MAP_FAILED) {
            munmap((void *)drv->semaphore, 4096);
            drv->semaphore = NULL;
        }

        if (drv->pushbuf.backing) {
            cml_nv_buffer_free(drv, drv->pushbuf.backing);
            drv->pushbuf.backing = NULL;
            drv->pushbuf.buf = NULL;
        }

        if (drv->copy_obj_handle && drv->channel_handle) {
            nv_rm_free(drv->fd_ctl, drv->client_handle,
                       drv->channel_handle, drv->copy_obj_handle);
        }
        if (drv->compute_obj_handle && drv->channel_handle) {
            nv_rm_free(drv->fd_ctl, drv->client_handle,
                       drv->channel_handle, drv->compute_obj_handle);
        }
        if (drv->channel_handle) {
            uint32_t parent = drv->channel_group_handle ? drv->channel_group_handle : drv->device_handle;
            nv_rm_free(drv->fd_ctl, drv->client_handle, parent, drv->channel_handle);
        }
        if (drv->channel_group_handle) {
            nv_rm_free(drv->fd_ctl, drv->client_handle,
                       drv->device_handle, drv->channel_group_handle);
        }

        if (drv->userd_buf) {
            cml_nv_buffer_free(drv, drv->userd_buf);
            drv->userd_buf = NULL;
        }
        if (drv->gpfifo_buf) {
            cml_nv_buffer_free(drv, drv->gpfifo_buf);
            drv->gpfifo_buf = NULL;
        }

        if (drv->vaspace_handle) {
            nv_rm_free(drv->fd_ctl, drv->client_handle,
                       drv->device_handle, drv->vaspace_handle);
        }
        if (drv->subdevice_handle) {
            nv_rm_free(drv->fd_ctl, drv->client_handle,
                       drv->device_handle, drv->subdevice_handle);
        }
        if (drv->device_handle) {
            nv_rm_free(drv->fd_ctl, drv->client_handle,
                       drv->client_handle, drv->device_handle);
        }
        if (drv->client_handle) {
            nv_rm_free(drv->fd_ctl, drv->client_handle, 0, drv->client_handle);
        }

        if (drv->fd_uvm >= 0) close(drv->fd_uvm);
        if (drv->fd_dev >= 0) close(drv->fd_dev);
        if (drv->fd_ctl >= 0) close(drv->fd_ctl);
    }
#endif

    free(drv);
}


CMLNVBuffer* cml_nv_buffer_create(CMLNVDriver *drv, size_t size, bool host_visible) {
    if (!drv || !drv->initialized || size == 0) return NULL;

#ifdef __linux__
    return nv_alloc_gpu_buffer(drv, size, host_visible);
#else
    (void)host_visible;
    return NULL;
#endif
}

CMLNVBuffer* cml_nv_buffer_create_vram(CMLNVDriver *drv, size_t size) {
    if (!drv || !drv->initialized || size == 0) return NULL;

#ifdef __linux__
    return nv_alloc_gpu_buffer(drv, size, false);
#else
    return NULL;
#endif
}

void cml_nv_buffer_free(CMLNVDriver *drv, CMLNVBuffer *buf) {
    if (!drv || !buf) return;

#ifdef __linux__
    if (buf->cpu_addr && buf->cpu_addr != MAP_FAILED) {
        size_t aligned = NV_ALIGN(buf->size, 4096);
        munmap(buf->cpu_addr, aligned);
    }

    if (drv->initialized && buf->handle) {
        nv_rm_free(drv->fd_ctl, drv->client_handle,
                   drv->device_handle, buf->handle);
    }
    if (drv->initialized && buf->va_handle) {
        nv_rm_free(drv->fd_ctl, drv->client_handle,
                   drv->device_handle, buf->va_handle);
    }
#endif

    free(buf);
}

int cml_nv_buffer_upload(CMLNVDriver *drv, CMLNVBuffer *dst,
                          const void *src, size_t n) {
    if (!drv || !dst || !src || n == 0) return -1;
    if (!drv->initialized) return -1;
    if (n > dst->size) return -1;

    if (dst->cpu_addr) {
        memcpy(dst->cpu_addr, src, n);
        return 0;
    }

    /* For device-local without host mapping, use staging + CE copy */
    CMLNVBuffer *staging = cml_nv_buffer_create(drv, n, true);
    if (!staging) return -1;

    memcpy(staging->cpu_addr, src, n);

    int ret = cml_nv_buffer_copy(drv, dst, staging, n);
    cml_nv_buffer_free(drv, staging);
    return ret;
}

int cml_nv_buffer_download(CMLNVDriver *drv, CMLNVBuffer *src,
                            void *dst, size_t n) {
    if (!drv || !src || !dst || n == 0) return -1;
    if (!drv->initialized) return -1;
    if (n > src->size) return -1;

    if (src->cpu_addr) {
        memcpy(dst, src->cpu_addr, n);
        return 0;
    }

    cml_nv_synchronize(drv);

    CMLNVBuffer *staging = cml_nv_buffer_create(drv, n, true);
    if (!staging) return -1;

    int ret = cml_nv_buffer_copy(drv, staging, src, n);
    if (ret == 0) {
        cml_nv_synchronize(drv);
        if (staging->cpu_addr)
            memcpy(dst, staging->cpu_addr, n);
        else
            ret = -1;
    }

    cml_nv_buffer_free(drv, staging);
    return ret;
}

int cml_nv_buffer_copy(CMLNVDriver *drv, CMLNVBuffer *dst,
                        CMLNVBuffer *src, size_t n) {
    if (!drv || !dst || !src || n == 0) return -1;
    if (!drv->initialized) return -1;

#ifdef __linux__
    CMLNVPushbuf *pb = &drv->pushbuf;
    if (!pb->buf) {
        if (dst->cpu_addr && src->cpu_addr) {
            memcpy(dst->cpu_addr, src->cpu_addr, n);
            return 0;
        }
        return -1;
    }

    pushbuf_reset(pb);

    /* CE copy: set src/dst addresses, line length, and launch */
    pushbuf_emit_method(pb, NVC0_SUBCHANNEL_COPY, NVC0B5_OFFSET_IN_UPPER, 4);
    pushbuf_emit(pb, (uint32_t)(src->gpu_va >> 32));
    pushbuf_emit(pb, (uint32_t)(src->gpu_va & 0xFFFFFFFF));
    pushbuf_emit(pb, (uint32_t)(dst->gpu_va >> 32));
    pushbuf_emit(pb, (uint32_t)(dst->gpu_va & 0xFFFFFFFF));

    pushbuf_emit_method(pb, NVC0_SUBCHANNEL_COPY, NVC0B5_PITCH_IN, 4);
    pushbuf_emit(pb, (uint32_t)n);
    pushbuf_emit(pb, (uint32_t)n);
    pushbuf_emit(pb, (uint32_t)n);
    pushbuf_emit(pb, 1);

    pushbuf_emit_method(pb, NVC0_SUBCHANNEL_COPY, NVC0B5_LAUNCH_DMA, 1);
    pushbuf_emit(pb, 0x00000182); /* SRC_TYPE_PHYS | DST_TYPE_PHYS | PIPELINED */

    drv->semaphore_value++;
    nv_push_semaphore_release(pb, drv->semaphore_gpu_va, (uint32_t)drv->semaphore_value);

    nv_gpfifo_submit(drv, pb->gpu_va, pb->pos);
    return 0;
#else
    return -1;
#endif
}


CMLNVKernel* cml_nv_kernel_compile_ptx(CMLNVDriver *drv, const char *ptx_code,
                                         const char *kernel_name) {
    if (!ptx_code || !kernel_name) return NULL;

    int sm = 75;
    if (drv && drv->compute_cap_major > 0)
        sm = drv->compute_cap_major * 10 + drv->compute_cap_minor;

    CMLNVKernel *kernel = (CMLNVKernel *)calloc(1, sizeof(CMLNVKernel));
    if (!kernel) return NULL;

    kernel->name = strdup(kernel_name);
    if (!kernel->name) {
        free(kernel);
        return NULL;
    }

    char ptx_path[256];
    char cubin_path[256];
    snprintf(ptx_path, sizeof(ptx_path), "/tmp/cml_nv_%s.ptx", kernel_name);
    snprintf(cubin_path, sizeof(cubin_path), "/tmp/cml_nv_%s.cubin", kernel_name);

    FILE *ptx_file = fopen(ptx_path, "w");
    if (!ptx_file) {
        LOG_ERROR("NV driver: failed to write PTX temp file: %s", strerror(errno));
        free(kernel->name);
        free(kernel);
        return NULL;
    }
    fputs(ptx_code, ptx_file);
    fclose(ptx_file);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "ptxas -arch=sm_%d -o %s %s 2>&1", sm, cubin_path, ptx_path);

    FILE *proc = popen(cmd, "r");
    if (!proc) {
        LOG_ERROR("NV driver: failed to run ptxas: %s", strerror(errno));
        unlink(ptx_path);
        free(kernel->name);
        free(kernel);
        return NULL;
    }

    char output[2048];
    size_t total_read = 0;
    while (total_read < sizeof(output) - 1) {
        size_t r = fread(output + total_read, 1, sizeof(output) - 1 - total_read, proc);
        if (r == 0) break;
        total_read += r;
    }
    output[total_read] = '\0';

    int status = pclose(proc);
    unlink(ptx_path);

    if (status != 0) {
        LOG_ERROR("NV driver: ptxas failed (status %d): %s", status, output);
        unlink(cubin_path);
        free(kernel->name);
        free(kernel);
        return NULL;
    }

    FILE *cubin_file = fopen(cubin_path, "rb");
    if (!cubin_file) {
        LOG_ERROR("NV driver: failed to read CUBIN file %s", cubin_path);
        free(kernel->name);
        free(kernel);
        return NULL;
    }

    fseek(cubin_file, 0, SEEK_END);
    long cubin_len = ftell(cubin_file);
    fseek(cubin_file, 0, SEEK_SET);

    if (cubin_len <= 0) {
        fclose(cubin_file);
        unlink(cubin_path);
        free(kernel->name);
        free(kernel);
        return NULL;
    }

    kernel->cubin_data = malloc((size_t)cubin_len);
    if (!kernel->cubin_data) {
        fclose(cubin_file);
        unlink(cubin_path);
        free(kernel->name);
        free(kernel);
        return NULL;
    }

    kernel->cubin_size = (size_t)cubin_len;
    if (fread(kernel->cubin_data, 1, kernel->cubin_size, cubin_file) != kernel->cubin_size) {
        fclose(cubin_file);
        unlink(cubin_path);
        free(kernel->cubin_data);
        free(kernel->name);
        free(kernel);
        return NULL;
    }
    fclose(cubin_file);
    unlink(cubin_path);

    if (nv_parse_cubin(kernel->cubin_data, kernel->cubin_size,
                       kernel_name, &kernel->meta) == 0) {
        kernel->num_regs = (int)kernel->meta.num_registers;
        kernel->shared_mem = (int)kernel->meta.shared_mem_size;
        kernel->param_size = (int)kernel->meta.param_size;
        kernel->bar_count = (int)kernel->meta.bar_count;
    } else {
        kernel->num_regs = 16;
        kernel->bar_count = 1;
    }

    if (drv && drv->initialized) {
        size_t code_alloc = NV_ALIGN(kernel->cubin_size, 256);
        CMLNVBuffer *code_buf = cml_nv_buffer_create(drv, code_alloc, true);
        if (code_buf) {
            memcpy(code_buf->cpu_addr, kernel->cubin_data, kernel->cubin_size);

            if (kernel->meta.code_offset > 0)
                kernel->gpu_addr = code_buf->gpu_va + kernel->meta.code_offset;
            else
                kernel->gpu_addr = code_buf->gpu_va;

            kernel->handle = code_buf->handle;
        }
    }

    LOG_INFO("NV driver: compiled kernel '%s' -> %zu bytes CUBIN (sm_%d, regs=%d, smem=%d)",
             kernel_name, kernel->cubin_size, sm, kernel->num_regs, kernel->shared_mem);
    return kernel;
}

CMLNVKernel* cml_nv_kernel_load_cubin(CMLNVDriver *drv, const void *cubin, size_t size,
                                        const char *kernel_name) {
    if (!cubin || size == 0 || !kernel_name) return NULL;

    CMLNVKernel *kernel = (CMLNVKernel *)calloc(1, sizeof(CMLNVKernel));
    if (!kernel) return NULL;

    kernel->name = strdup(kernel_name);
    if (!kernel->name) {
        free(kernel);
        return NULL;
    }

    kernel->cubin_data = malloc(size);
    if (!kernel->cubin_data) {
        free(kernel->name);
        free(kernel);
        return NULL;
    }
    memcpy(kernel->cubin_data, cubin, size);
    kernel->cubin_size = size;

    if (nv_parse_cubin(cubin, size, kernel_name, &kernel->meta) == 0) {
        kernel->num_regs = (int)kernel->meta.num_registers;
        kernel->shared_mem = (int)kernel->meta.shared_mem_size;
        kernel->param_size = (int)kernel->meta.param_size;
        kernel->bar_count = (int)kernel->meta.bar_count;
    } else {
        kernel->num_regs = 16;
        kernel->bar_count = 1;
    }

    if (drv && drv->initialized) {
        size_t code_alloc = NV_ALIGN(size, 256);
        CMLNVBuffer *code_buf = cml_nv_buffer_create(drv, code_alloc, true);
        if (code_buf) {
            memcpy(code_buf->cpu_addr, cubin, size);
            if (kernel->meta.code_offset > 0)
                kernel->gpu_addr = code_buf->gpu_va + kernel->meta.code_offset;
            else
                kernel->gpu_addr = code_buf->gpu_va;
            kernel->handle = code_buf->handle;
        }
    }

    return kernel;
}

void cml_nv_kernel_free(CMLNVDriver *drv, CMLNVKernel *kernel) {
    if (!kernel) return;

#ifdef __linux__
    if (drv && drv->initialized && kernel->handle) {
        nv_rm_free(drv->fd_ctl, drv->client_handle,
                   drv->device_handle, kernel->handle);
    }
#else
    (void)drv;
#endif

    free(kernel->cubin_data);
    free(kernel->name);
    free(kernel);
}


int cml_nv_kernel_launch(CMLNVDriver *drv, CMLNVKernel *kernel,
                          uint32_t grid[3], uint32_t block[3],
                          void **args, int num_args) {
    if (!drv || !drv->initialized || !kernel) return -1;

    LOG_DEBUG("NV driver: launching kernel '%s' grid=[%u,%u,%u] block=[%u,%u,%u]",
              kernel->name, grid[0], grid[1], grid[2], block[0], block[1], block[2]);

#ifdef __linux__
    CMLNVPushbuf *pb = &drv->pushbuf;
    if (!pb->buf || !drv->gpfifo.entries)
        return -1;

    /* Build QMD */
    NVQmd qmd;
    nv_qmd_init(&qmd, drv->gpu_arch);
    nv_qmd_set_program_address(&qmd, kernel->gpu_addr);
    nv_qmd_set_grid_dim(&qmd, grid[0], grid[1], grid[2]);
    nv_qmd_set_block_dim(&qmd, block[0], block[1], block[2]);
    nv_qmd_set_register_count(&qmd, (uint32_t)kernel->num_regs);
    nv_qmd_set_shared_memory(&qmd, (uint32_t)kernel->shared_mem);
    nv_qmd_set_barrier_count(&qmd, (uint32_t)kernel->bar_count);

    /* Build CB0 (constant buffer 0) with kernel arguments */
    CMLNVBuffer *cb0_buf = NULL;
    if (args && num_args > 0) {
        size_t cb0_size = NV_ALIGN((size_t)num_args * 8, 256);
        cb0_buf = cml_nv_buffer_create(drv, cb0_size, true);
        if (cb0_buf) {
            uint8_t *cb0_data = (uint8_t *)cb0_buf->cpu_addr;
            size_t offset = 0;
            for (int i = 0; i < num_args; i++) {
                if (args[i]) {
                    memcpy(cb0_data + offset, args[i], 8);
                }
                offset += 8;
            }
            nv_qmd_set_constant_buffer(&qmd, 0, cb0_buf->gpu_va, (uint32_t)cb0_size);
        }
    }

    /* Allocate QMD in GPU-visible memory */
    CMLNVBuffer *qmd_buf = cml_nv_buffer_create(drv, NV_ALIGN(NV_QMD_BYTES, 256), true);
    if (!qmd_buf) {
        if (cb0_buf) cml_nv_buffer_free(drv, cb0_buf);
        return -1;
    }
    memcpy(qmd_buf->cpu_addr, qmd.data, NV_QMD_BYTES);

    __sync_synchronize();

    pushbuf_reset(pb);

    nv_push_invalidate_caches(pb);

    /* Send QMD address and launch */
    pushbuf_emit_method(pb, NVC0_SUBCHANNEL_COMPUTE, NVC3C0_SEND_PCAS_A, 1);
    pushbuf_emit(pb, (uint32_t)(qmd_buf->gpu_va >> 8));

    pushbuf_emit_method(pb, NVC0_SUBCHANNEL_COMPUTE, NVC3C0_SEND_SIGNALING_PCAS_B, 1);
    pushbuf_emit(pb, 0x1);

    /* Alternatively, inline the QMD data directly into the pushbuffer:
     * This avoids a separate QMD buffer allocation. */
    pushbuf_emit_method(pb, NVC0_SUBCHANNEL_COMPUTE, NVC3C0_LOAD_INLINE_QMD_DATA(0), NV_QMD_DWORDS);
    pushbuf_emit_data(pb, qmd.data, NV_QMD_DWORDS);

    pushbuf_emit_method(pb, NVC0_SUBCHANNEL_COMPUTE, NVC3C0_LAUNCH, 1);
    pushbuf_emit(pb, 0x1);

    drv->semaphore_value++;
    nv_push_semaphore_release(pb, drv->semaphore_gpu_va, (uint32_t)drv->semaphore_value);

    __sync_synchronize();

    nv_gpfifo_submit(drv, pb->gpu_va, pb->pos);

    cml_nv_buffer_free(drv, qmd_buf);
    if (cb0_buf) cml_nv_buffer_free(drv, cb0_buf);

    return 0;
#else
    (void)args;
    (void)num_args;
    return -1;
#endif
}


int cml_nv_synchronize(CMLNVDriver *drv) {
    if (!drv || !drv->initialized) return -1;

#ifdef __linux__
    if (!drv->semaphore) {
        LOG_WARNING("NV driver: no semaphore mapped, sync is a no-op");
        return 0;
    }

    uint64_t deadline = now_ms() + 5000;
    uint32_t target = (uint32_t)drv->semaphore_value;

    while (*drv->semaphore < target) {
        __sync_synchronize();

        if (now_ms() >= deadline) {
            LOG_ERROR("NV driver: synchronize timed out (current=%u, target=%u)",
                      *drv->semaphore, target);
            return -1;
        }

        struct timespec ts = {0, 100000}; /* 100us */
        nanosleep(&ts, NULL);
    }

    return 0;
#else
    return 0;
#endif
}


static char* nv_gen_ptx_for_node(struct IRNode *node, int sm) {
    char *ptx = NULL;
    const char *kname = "nv_auto_kernel";
    size_t buf_size = 4096;

    switch (node->type) {
    case UOP_NEG: case UOP_EXP: case UOP_LOG: case UOP_SQRT:
    case UOP_ABS: case UOP_SIN: case UOP_COS: case UOP_TANH:
    case UOP_SIGMOID: case UOP_RECIP: case UOP_SILU: {
        ptx = (char *)malloc(buf_size);
        if (!ptx) return NULL;
        const char *op_ptx;
        switch (node->type) {
            case UOP_NEG:   op_ptx = "neg.f32 %%f1, %%f0;"; break;
            case UOP_EXP:   op_ptx = "ex2.approx.f32 %%f1, %%f0;"; break;
            case UOP_LOG:   op_ptx = "lg2.approx.f32 %%f1, %%f0;"; break;
            case UOP_SQRT:  op_ptx = "sqrt.approx.f32 %%f1, %%f0;"; break;
            case UOP_ABS:   op_ptx = "abs.f32 %%f1, %%f0;"; break;
            case UOP_SIN:   op_ptx = "sin.approx.f32 %%f1, %%f0;"; break;
            case UOP_COS:   op_ptx = "cos.approx.f32 %%f1, %%f0;"; break;
            case UOP_RECIP: op_ptx = "rcp.approx.f32 %%f1, %%f0;"; break;
            default:        op_ptx = "mov.f32 %%f1, %%f0;"; break;
        }
        snprintf(ptx, buf_size,
            ".version 7.0\n.target sm_%d\n.address_size 64\n\n"
            ".visible .entry %s(\n"
            "    .param .u64 param_in,\n"
            "    .param .u64 param_out,\n"
            "    .param .u32 param_n\n"
            ") {\n"
            "    .reg .pred %%p<2>;\n"
            "    .reg .b32 %%r<8>;\n"
            "    .reg .b64 %%rd<8>;\n"
            "    .reg .f32 %%f<4>;\n\n"
            "    mov.u32 %%r0, %%tid.x;\n"
            "    mov.u32 %%r1, %%ctaid.x;\n"
            "    mov.u32 %%r2, %%ntid.x;\n"
            "    mad.lo.u32 %%r3, %%r1, %%r2, %%r0;\n"
            "    ld.param.u32 %%r4, [param_n];\n"
            "    setp.ge.u32 %%p0, %%r3, %%r4;\n"
            "    @%%p0 ret;\n\n"
            "    ld.param.u64 %%rd0, [param_in];\n"
            "    ld.param.u64 %%rd1, [param_out];\n"
            "    cvt.u64.u32 %%rd2, %%r3;\n"
            "    shl.b64 %%rd2, %%rd2, 2;\n"
            "    add.u64 %%rd0, %%rd0, %%rd2;\n"
            "    add.u64 %%rd1, %%rd1, %%rd2;\n"
            "    ld.global.f32 %%f0, [%%rd0];\n"
            "    %s\n"
            "    st.global.f32 [%%rd1], %%f1;\n"
            "    ret;\n}\n",
            sm, kname, op_ptx);
        break;
    }

    case UOP_ADD: case UOP_SUB: case UOP_MUL: case UOP_DIV: {
        ptx = (char *)malloc(buf_size);
        if (!ptx) return NULL;
        const char *op_ptx;
        switch (node->type) {
            case UOP_ADD: op_ptx = "add.f32 %%f2, %%f0, %%f1;"; break;
            case UOP_SUB: op_ptx = "sub.f32 %%f2, %%f0, %%f1;"; break;
            case UOP_MUL: op_ptx = "mul.f32 %%f2, %%f0, %%f1;"; break;
            case UOP_DIV: op_ptx = "div.approx.f32 %%f2, %%f0, %%f1;"; break;
            default:      op_ptx = "add.f32 %%f2, %%f0, %%f1;"; break;
        }
        snprintf(ptx, buf_size,
            ".version 7.0\n.target sm_%d\n.address_size 64\n\n"
            ".visible .entry %s(\n"
            "    .param .u64 param_a,\n"
            "    .param .u64 param_b,\n"
            "    .param .u64 param_out,\n"
            "    .param .u32 param_n\n"
            ") {\n"
            "    .reg .pred %%p<2>;\n"
            "    .reg .b32 %%r<8>;\n"
            "    .reg .b64 %%rd<8>;\n"
            "    .reg .f32 %%f<4>;\n\n"
            "    mov.u32 %%r0, %%tid.x;\n"
            "    mov.u32 %%r1, %%ctaid.x;\n"
            "    mov.u32 %%r2, %%ntid.x;\n"
            "    mad.lo.u32 %%r3, %%r1, %%r2, %%r0;\n"
            "    ld.param.u32 %%r4, [param_n];\n"
            "    setp.ge.u32 %%p0, %%r3, %%r4;\n"
            "    @%%p0 ret;\n\n"
            "    ld.param.u64 %%rd0, [param_a];\n"
            "    ld.param.u64 %%rd1, [param_b];\n"
            "    ld.param.u64 %%rd2, [param_out];\n"
            "    cvt.u64.u32 %%rd3, %%r3;\n"
            "    shl.b64 %%rd3, %%rd3, 2;\n"
            "    add.u64 %%rd0, %%rd0, %%rd3;\n"
            "    add.u64 %%rd1, %%rd1, %%rd3;\n"
            "    add.u64 %%rd2, %%rd2, %%rd3;\n"
            "    ld.global.f32 %%f0, [%%rd0];\n"
            "    ld.global.f32 %%f1, [%%rd1];\n"
            "    %s\n"
            "    st.global.f32 [%%rd2], %%f2;\n"
            "    ret;\n}\n",
            sm, kname, op_ptx);
        break;
    }

    default:
        break;
    }

    return ptx;
}

int cml_nv_gpu_wait_semaphore(CMLNVDriver *drv, uint64_t sem_va, uint32_t value) {
#ifdef __linux__
    if (!drv || !drv->initialized) return -1;

    CMLNVPushbuf *pb = &drv->pushbuf;
    pb->pos = 0;

    nv_push_semaphore_acquire(pb, sem_va, value);

    __sync_synchronize();
    nv_gpfifo_submit(drv, pb->gpu_va, pb->pos);
    return 0;
#else
    (void)drv; (void)sem_va; (void)value;
    return -1;
#endif
}

int cml_nv_execute_graph(CMLNVDriver *drv, CMLGraph_t ir) {
    if (!drv || !ir) return -1;
    if (!drv->initialized) {
        LOG_ERROR("NV driver: not initialized");
        return -1;
    }

    int sm = 75;
    if (drv->compute_cap_major > 0)
        sm = drv->compute_cap_major * 10 + drv->compute_cap_minor;

    struct IRNode *node = ir->head;
    while (node) {
        if (node->is_executed) {
            node = node->next;
            continue;
        }

        Tensor *output = node->output;
        if (!output) {
            node = node->next;
            continue;
        }

        bool gpu_ok = false;
        char *ptx = nv_gen_ptx_for_node(node, sm);

        if (ptx) {
            CMLNVKernel *kernel = cml_nv_kernel_compile_ptx(drv, ptx, "nv_auto_kernel");
            if (kernel) {
                size_t numel = 1;
                for (int d = 0; d < output->ndim; d++)
                    numel *= (size_t)output->shape[d];

                bool is_unary = (node->num_inputs == 1);
                bool is_binary = (node->num_inputs == 2);

                if (is_unary && node->inputs && node->inputs[0]) {
                    Tensor *a = node->inputs[0];
                    size_t bytes = numel * sizeof(float);

                    CMLNVBuffer *buf_in  = cml_nv_buffer_create(drv, bytes, true);
                    CMLNVBuffer *buf_out = cml_nv_buffer_create(drv, bytes, true);

                    if (buf_in && buf_out && a->data) {
                        cml_nv_buffer_upload(drv, buf_in, a->data, bytes);

                        uint32_t n32 = (uint32_t)numel;
                        void *kargs[] = { &buf_in->gpu_va, &buf_out->gpu_va, &n32 };
                        uint32_t block_dim[3] = {256, 1, 1};
                        uint32_t grid_dim[3]  = {(uint32_t)((numel + 255) / 256), 1, 1};

                        if (cml_nv_kernel_launch(drv, kernel, grid_dim, block_dim, kargs, 3) == 0) {
                            cml_nv_synchronize(drv);
                            if (!output->data)
                                output->data = malloc(bytes);
                            if (output->data) {
                                cml_nv_buffer_download(drv, buf_out, output->data, bytes);
                                gpu_ok = true;
                            }
                        }
                    }

                    if (buf_in)  cml_nv_buffer_free(drv, buf_in);
                    if (buf_out) cml_nv_buffer_free(drv, buf_out);

                } else if (is_binary && node->inputs && node->inputs[0] && node->inputs[1]) {
                    Tensor *a = node->inputs[0];
                    Tensor *b = node->inputs[1];
                    size_t bytes = numel * sizeof(float);

                    CMLNVBuffer *buf_a   = cml_nv_buffer_create(drv, bytes, true);
                    CMLNVBuffer *buf_b   = cml_nv_buffer_create(drv, bytes, true);
                    CMLNVBuffer *buf_out = cml_nv_buffer_create(drv, bytes, true);

                    if (buf_a && buf_b && buf_out && a->data && b->data) {
                        cml_nv_buffer_upload(drv, buf_a, a->data, bytes);
                        cml_nv_buffer_upload(drv, buf_b, b->data, bytes);

                        uint32_t n32 = (uint32_t)numel;
                        void *kargs[] = { &buf_a->gpu_va, &buf_b->gpu_va,
                                          &buf_out->gpu_va, &n32 };
                        uint32_t block_dim[3] = {256, 1, 1};
                        uint32_t grid_dim[3]  = {(uint32_t)((numel + 255) / 256), 1, 1};

                        if (cml_nv_kernel_launch(drv, kernel, grid_dim, block_dim, kargs, 4) == 0) {
                            cml_nv_synchronize(drv);
                            if (!output->data)
                                output->data = malloc(bytes);
                            if (output->data) {
                                cml_nv_buffer_download(drv, buf_out, output->data, bytes);
                                gpu_ok = true;
                            }
                        }
                    }

                    if (buf_a)   cml_nv_buffer_free(drv, buf_a);
                    if (buf_b)   cml_nv_buffer_free(drv, buf_b);
                    if (buf_out) cml_nv_buffer_free(drv, buf_out);
                }

                cml_nv_kernel_free(drv, kernel);
            }
            free(ptx);
        }

        if (!gpu_ok) {
            LOG_DEBUG("NV driver: GPU path failed for op %d, using CPU fallback",
                      (int)node->type);
            cpu_execute_node(node);
        }

        node->is_executed = true;
        if (output) output->is_executed = true;
        node = node->next;
    }

    ir->is_executed = true;
    return 0;
}
