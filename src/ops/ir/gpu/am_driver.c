#define _GNU_SOURCE

#include "ops/ir/gpu/am_driver.h"
#include "ops/ir/gpu/amdgpu_kd.h"
#include "ops/ir/internal.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>
#include <dirent.h>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#endif

#ifdef CML_AM_MOCK_GPU
#include "ops/ir/gpu/am_mock.h"
#define open(...)    cml_am_mock_open(__VA_ARGS__)
#define close(...)   cml_am_mock_close(__VA_ARGS__)
#define ioctl(...)   cml_am_mock_ioctl(__VA_ARGS__)
#define mmap(...)    cml_am_mock_mmap(__VA_ARGS__)
#define munmap(...)  cml_am_mock_munmap(__VA_ARGS__)
#define fopen(...)   cml_am_mock_fopen(__VA_ARGS__)
#define access(...)  cml_am_mock_access(__VA_ARGS__)
#define opendir(...) cml_am_mock_opendir(__VA_ARGS__)
#endif


#ifdef __linux__

#define KFD_IOC_ALLOC_MEM_FLAGS_VRAM       (1U << 0)
#define KFD_IOC_ALLOC_MEM_FLAGS_GTT        (1U << 1)
#define KFD_IOC_ALLOC_MEM_FLAGS_USERPTR    (1U << 2)
#define KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL   (1U << 3)
#define KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP (1U << 4)
#define KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC     (1U << 5)
#define KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE   (1U << 6)
#define KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE (1U << 7)
#define KFD_IOC_ALLOC_MEM_FLAGS_COHERENT   (1U << 8)

#define KFD_IOC_MAGIC 'K'
#define KFD_IOWR(nr, type) _IOWR(KFD_IOC_MAGIC, nr, type)
#define KFD_IOW(nr, type)  _IOW(KFD_IOC_MAGIC, nr, type)
#define KFD_IOR(nr, type)  _IOR(KFD_IOC_MAGIC, nr, type)

struct kfd_ioctl_get_version_args {
    uint32_t major_version;
    uint32_t minor_version;
};

struct kfd_ioctl_acquire_vm_args {
    uint32_t drm_fd;
    uint32_t gpu_id;
};

#define KFD_IOC_QUEUE_TYPE_COMPUTE      0
#define KFD_IOC_QUEUE_TYPE_SDMA         1
#define KFD_IOC_QUEUE_TYPE_COMPUTE_AQL  2
#define KFD_IOC_QUEUE_TYPE_SDMA_XGMI    3

struct kfd_ioctl_create_queue_args {
    uint64_t ring_base_address;
    uint64_t write_pointer_address;
    uint64_t read_pointer_address;
    uint64_t doorbell_offset;
    uint32_t ring_size;
    uint32_t gpu_id;
    uint32_t queue_type;
    uint32_t queue_percentage;
    uint32_t queue_priority;
    uint32_t queue_id;
    uint64_t eop_buffer_address;
    uint32_t eop_buffer_size;
    uint64_t ctx_save_restore_address;
    uint32_t ctx_save_restore_size;
    uint32_t ctl_stack_size;
    uint32_t pad;
};

struct kfd_ioctl_destroy_queue_args {
    uint32_t queue_id;
    uint32_t pad;
};

struct kfd_ioctl_alloc_memory_of_gpu_args {
    uint64_t va_addr;
    uint64_t size;
    uint64_t handle;
    uint32_t gpu_id;
    uint32_t flags;
    uint64_t mmap_offset;
};

struct kfd_ioctl_free_memory_of_gpu_args {
    uint64_t handle;
};

struct kfd_ioctl_map_memory_to_gpu_args {
    uint64_t handle;
    uint64_t device_ids_array_ptr;
    uint32_t n_devices;
    uint32_t n_success;
};

struct kfd_ioctl_unmap_memory_from_gpu_args {
    uint64_t handle;
    uint64_t device_ids_array_ptr;
    uint32_t n_devices;
    uint32_t n_success;
};

#define AMDKFD_IOC_GET_VERSION       KFD_IOR(0x01, struct kfd_ioctl_get_version_args)
#define AMDKFD_IOC_CREATE_QUEUE      KFD_IOWR(0x02, struct kfd_ioctl_create_queue_args)
#define AMDKFD_IOC_DESTROY_QUEUE     KFD_IOWR(0x03, struct kfd_ioctl_destroy_queue_args)
#define AMDKFD_IOC_ACQUIRE_VM        KFD_IOW(0x07, struct kfd_ioctl_acquire_vm_args)
#define AMDKFD_IOC_ALLOC_MEMORY_OF_GPU KFD_IOWR(0x18, struct kfd_ioctl_alloc_memory_of_gpu_args)
#define AMDKFD_IOC_FREE_MEMORY_OF_GPU  KFD_IOW(0x19, struct kfd_ioctl_free_memory_of_gpu_args)
#define AMDKFD_IOC_MAP_MEMORY_TO_GPU   KFD_IOWR(0x1A, struct kfd_ioctl_map_memory_to_gpu_args)
#define AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU KFD_IOWR(0x1B, struct kfd_ioctl_unmap_memory_from_gpu_args)

/* AQL packet types */
#define AQL_PKT_TYPE_KERNEL_DISPATCH  1
#define AQL_PKT_TYPE_BARRIER_AND      2
#define AQL_PKT_TYPE_BARRIER_OR       3

#define AQL_HDR_TYPE_SHIFT     0
#define AQL_HDR_BARRIER_SHIFT  8
#define AQL_HDR_ACQUIRE_SHIFT  9
#define AQL_HDR_RELEASE_SHIFT  11

#define AQL_FENCE_SCOPE_SYSTEM 3
#define AQL_FENCE_SCOPE_AGENT  2

/* SDMA packet opcodes */
#define SDMA_OP_COPY  1
#define SDMA_OP_FENCE 5
#define SDMA_OP_TRAP  6
#define SDMA_SUBOP_COPY_LINEAR 0

#define AM_RING_NUM_PACKETS  256
#define AM_RING_SIZE_BYTES   (AM_RING_NUM_PACKETS * 64)

#define AM_SDMA_RING_SIZE    (64 * 1024)

#define AM_EOP_BUFFER_SIZE   4096

#define AM_PAGE_SIZE         4096
#define AM_PAGE_ALIGN(x)     (((x) + AM_PAGE_SIZE - 1) & ~(uint64_t)(AM_PAGE_SIZE - 1))

#define AM_VA_START   0x100000000ULL
#define AM_VA_END     0x800000000ULL

#define AM_SIGNAL_INIT 0

#define am_mb()  __atomic_thread_fence(__ATOMIC_SEQ_CST)

/* SDMA packet structures */
typedef struct __attribute__((packed)) {
    uint32_t op_subop;
    uint32_t count_minus_one;
    uint32_t pad;
    uint32_t src_addr_lo;
    uint32_t src_addr_hi;
    uint32_t dst_addr_lo;
    uint32_t dst_addr_hi;
} SDMACopyPacket;

typedef struct __attribute__((packed)) {
    uint32_t op_subop;
    uint32_t addr_lo;
    uint32_t addr_hi;
    uint32_t value;
} SDMAFencePacket;

typedef struct __attribute__((packed)) {
    uint32_t op_subop;
    uint32_t pad;
} SDMATrapPacket;

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

    uint32_t gpu_ids[1] = { drv->gpu_id };
    struct kfd_ioctl_map_memory_to_gpu_args map = {0};
    map.handle = alloc.handle;
    map.device_ids_array_ptr = (uint64_t)(uintptr_t)gpu_ids;
    map.n_devices = 1;

    if (kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map) != 0) {
        LOG_ERROR("AM driver: MAP_MEMORY_TO_GPU failed");
        struct kfd_ioctl_free_memory_of_gpu_args fr = { .handle = alloc.handle };
        kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &fr);
        return -1;
    }

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

static void am_free_and_unmap(CMLAMDriver* drv, uint64_t handle,
                              void* cpu_addr, size_t size) {
    if (cpu_addr) {
        munmap(cpu_addr, AM_PAGE_ALIGN(size));
    }

    uint32_t gpu_ids[1] = { drv->gpu_id };
    struct kfd_ioctl_unmap_memory_from_gpu_args unmap = {0};
    unmap.handle = handle;
    unmap.device_ids_array_ptr = (uint64_t)(uintptr_t)gpu_ids;
    unmap.n_devices = 1;
    kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &unmap);

    struct kfd_ioctl_free_memory_of_gpu_args fr = { .handle = handle };
    kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &fr);
}

/* Read a uint64 property from a sysfs file. Returns 0 on success. */
static int sysfs_read_u64(const char* path, uint64_t* val) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    int ret = (fscanf(f, "%lu", (unsigned long*)val) == 1) ? 0 : -1;
    fclose(f);
    return ret;
}

/* Read a uint32 property from a sysfs file. */
static int sysfs_read_u32(const char* path, uint32_t* val) {
    uint64_t v;
    int ret = sysfs_read_u64(path, &v);
    if (ret == 0) *val = (uint32_t)v;
    return ret;
}

/* Read a string property from a sysfs file. */
static int sysfs_read_str(const char* path, char* buf, size_t buflen)
    __attribute__((unused));
static int sysfs_read_str(const char* path, char* buf, size_t buflen) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    if (!fgets(buf, (int)buflen, f)) {
        fclose(f);
        return -1;
    }
    fclose(f);
    size_t len = strlen(buf);
    while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r'))
        buf[--len] = '\0';
    return 0;
}

/* Parse a single KFD topology node's properties file into key=value pairs.
 * Calls cb for each pair. Returns 0 on success. */
typedef void (*prop_cb_t)(const char* key, const char* value, void* ctx);

static int parse_topology_properties(const char* props_path, prop_cb_t cb, void* ctx) {
    FILE* f = fopen(props_path, "r");
    if (!f) return -1;

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';

        char* eq = strchr(line, ' ');
        if (!eq) continue;
        *eq = '\0';
        cb(line, eq + 1, ctx);
    }

    fclose(f);
    return 0;
}

typedef struct {
    CMLAMGPUInfo* info;
    bool has_cpu_cores;
} gpu_parse_ctx;

static void gpu_prop_cb(const char* key, const char* value, void* ctx) {
    gpu_parse_ctx* pc = (gpu_parse_ctx*)ctx;
    CMLAMGPUInfo* g = pc->info;

    if (strcmp(key, "cpu_cores_count") == 0) {
        int v = atoi(value);
        if (v > 0) pc->has_cpu_cores = true;
    } else if (strcmp(key, "name") == 0) {
        snprintf(g->name, sizeof(g->name), "%s", value);
    } else if (strcmp(key, "simd_count") == 0) {
        g->cu_count = atoi(value);
    } else if (strcmp(key, "simd_per_cu") == 0) {
        g->simd_per_cu = (uint32_t)atoi(value);
    } else if (strcmp(key, "max_waves_per_simd") == 0) {
        g->max_waves_per_simd = (uint32_t)atoi(value);
    } else if (strcmp(key, "max_slots_scratch_cu") == 0) {
        g->max_slots_scratch = (uint32_t)atoi(value);
    } else if (strcmp(key, "vendor_id") == 0) {
        g->vendor_id = (uint32_t)strtoul(value, NULL, 0);
    } else if (strcmp(key, "device_id") == 0) {
        g->device_id = (uint32_t)strtoul(value, NULL, 0);
    } else if (strcmp(key, "domain") == 0) {
        g->domain = (uint32_t)strtoul(value, NULL, 0);
    } else if (strcmp(key, "location_id") == 0) {
        g->location_id = (uint32_t)strtoul(value, NULL, 0);
    } else if (strcmp(key, "fw_version") == 0) {
        g->fw_version = (uint32_t)strtoul(value, NULL, 0);
    } else if (strcmp(key, "max_engine_clk_fcompute") == 0) {
        g->max_engine_clk = (uint32_t)atoi(value);
    } else if (strcmp(key, "local_mem_size") == 0) {
        g->vram_size = (size_t)strtoull(value, NULL, 0);
    } else if (strcmp(key, "lds_size_in_kb") == 0) {
        g->lds_size_per_cu = (uint32_t)atoi(value) * 1024;
    } else if (strcmp(key, "gfx_target_version") == 0) {
        uint32_t ver = (uint32_t)strtoul(value, NULL, 0);
        uint32_t major = ver / 10000;
        uint32_t minor = (ver / 100) % 100;
        uint32_t step  = ver % 100;
        snprintf(g->gfx_version, sizeof(g->gfx_version), "gfx%u%u%u",
                 major, minor, step);
    } else if (strcmp(key, "sdma_fw_version") == 0) {
        if (atoi(value) > 0) g->sdma_count++;
    }
}

#endif /* __linux__ */


int cml_am_enumerate_gpus(CMLAMGPUInfo** gpus, int* count) {
#ifdef __linux__
    if (!gpus || !count) return -1;
    *gpus = NULL;
    *count = 0;

    const char* topo_base = "/sys/devices/virtual/kfd/kfd/topology/nodes";
    DIR* dir = opendir(topo_base);
    if (!dir) {
        LOG_ERROR("AM driver: cannot open KFD topology at %s", topo_base);
        return -1;
    }

    CMLAMGPUInfo* list = NULL;
    int num = 0;
    int cap = 0;

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;

        int node_id = atoi(entry->d_name);
        char props_path[512];
        snprintf(props_path, sizeof(props_path), "%s/%s/properties", topo_base, entry->d_name);

        CMLAMGPUInfo info;
        memset(&info, 0, sizeof(info));
        info.node_id = node_id;
        info.sdma_count = 0;

        gpu_parse_ctx pc = { .info = &info, .has_cpu_cores = false };
        if (parse_topology_properties(props_path, gpu_prop_cb, &pc) != 0)
            continue;

        /* CPU nodes have cpu_cores_count > 0; skip them */
        if (pc.has_cpu_cores) continue;

        /* Read gpu_id */
        char gpu_id_path[512];
        snprintf(gpu_id_path, sizeof(gpu_id_path), "%s/%s/gpu_id", topo_base, entry->d_name);
        uint32_t gid = 0;
        sysfs_read_u32(gpu_id_path, &gid);
        if (gid == 0) continue;
        info.gpu_id = gid;

        if (num >= cap) {
            cap = cap ? cap * 2 : 4;
            CMLAMGPUInfo* tmp = realloc(list, (size_t)cap * sizeof(CMLAMGPUInfo));
            if (!tmp) { free(list); closedir(dir); return -1; }
            list = tmp;
        }
        list[num++] = info;
    }

    closedir(dir);
    *gpus = list;
    *count = num;

    LOG_INFO("AM driver: enumerated %d GPU(s)", num);
    for (int i = 0; i < num; i++) {
        LOG_INFO("  GPU %d: %s [%s] gpu_id=%u CUs=%d VRAM=%zuMB",
                 i, list[i].name, list[i].gfx_version, list[i].gpu_id,
                 list[i].cu_count, list[i].vram_size / (1024*1024));
    }

    return 0;
#else
    (void)gpus; (void)count;
    return -1;
#endif
}


CMLAMChipletConfig cml_am_get_chiplet_config(const char* gfx_version) {
    CMLAMChipletConfig cfg = {0};

    if (!gfx_version)
        return cfg;

    if (strcmp(gfx_version, "gfx942") == 0) {
        cfg.num_xcd = 8;
        cfg.cu_per_xcd = 38;
        cfg.sdma_per_xcd = 1;
        cfg.unified_memory = true;
    } else if (strcmp(gfx_version, "gfx950") == 0) {
        cfg.num_xcd = 8;
        cfg.cu_per_xcd = 48;
        cfg.sdma_per_xcd = 2;
        cfg.unified_memory = true;
    } else if (strncmp(gfx_version, "gfx12", 5) == 0) {
        cfg.num_xcd = 1;
        cfg.cu_per_xcd = 32;
        cfg.sdma_per_xcd = 2;
        cfg.unified_memory = false;
    }

    return cfg;
}

int cml_am_sdma_copy_nearest_xcd(CMLAMDriver* drv, int xcd_idx,
                                  uint64_t dst_va, uint64_t src_va, size_t size) {
    if (!drv || !drv->initialized)
        return -1;

    if (!drv->is_chiplet_gpu || xcd_idx < 0 ||
        xcd_idx >= drv->chiplet_config.num_xcd) {
        return cml_am_sdma_copy(drv, dst_va, src_va, size);
    }

    return cml_am_sdma_copy(drv, dst_va, src_va, size);
}

bool cml_am_driver_available(void) {
#ifdef __linux__
    if (access("/dev/kfd", R_OK) != 0)
        return false;

    if (access("/dev/dri/renderD128", R_OK) != 0)
        return false;

    int fd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if (fd < 0)
        return false;

    struct kfd_ioctl_get_version_args ver = {0};
    int ret = kfd_ioctl(fd, AMDKFD_IOC_GET_VERSION, &ver);
    close(fd);

    if (ret != 0)
        return false;

    LOG_INFO("AM driver: KFD version %u.%u",
              ver.major_version, ver.minor_version);
    return true;
#else
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

#ifdef __linux__
static int am_init_aql_queue(CMLAMDriver* drv, CMLAMQueue* q) {
    uint32_t gtt_flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT
                       | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                       | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;

    uint64_t ring_handle = 0, ring_va = 0;
    void*    ring_addr = NULL;
    uint32_t ring_flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT
                        | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                        | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;

    if (am_alloc_and_map(drv, AM_RING_SIZE_BYTES, ring_flags,
                         &ring_handle, &ring_va, &ring_addr) != 0)
        return -1;

    q->ring = (hsa_kernel_dispatch_packet_t*)ring_addr;
    q->ring_size = AM_RING_NUM_PACKETS;

    uint64_t wptr_handle = 0, wptr_va = 0;
    void*    wptr_addr = NULL;
    if (am_alloc_and_map(drv, AM_PAGE_SIZE, gtt_flags,
                         &wptr_handle, &wptr_va, &wptr_addr) != 0)
        return -1;

    q->write_dispatch_id = (volatile uint64_t*)wptr_addr;
    q->read_dispatch_id  = (volatile uint64_t*)((uint8_t*)wptr_addr + 64);

    if (q->write_dispatch_id) *q->write_dispatch_id = 0;
    if (q->read_dispatch_id)  *q->read_dispatch_id = 0;

    uint64_t eop_handle = 0, eop_va = 0;
    void*    eop_addr = NULL;
    if (am_alloc_and_map(drv, AM_EOP_BUFFER_SIZE,
                         KFD_IOC_ALLOC_MEM_FLAGS_VRAM
                         | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE,
                         &eop_handle, &eop_va, &eop_addr) != 0)
        return -1;

    struct kfd_ioctl_create_queue_args cq = {0};
    cq.ring_base_address     = ring_va;
    cq.write_pointer_address = wptr_va;
    cq.read_pointer_address  = wptr_va + 64;
    cq.ring_size             = AM_RING_SIZE_BYTES;
    cq.gpu_id                = drv->gpu_id;
    cq.queue_type            = KFD_IOC_QUEUE_TYPE_COMPUTE;
    cq.queue_percentage      = 100;
    cq.queue_priority        = 7;
    cq.eop_buffer_address    = eop_va;
    cq.eop_buffer_size       = AM_EOP_BUFFER_SIZE;

    if (kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_CREATE_QUEUE, &cq) != 0) {
        LOG_ERROR("AM driver: CREATE_QUEUE failed");
        return -1;
    }

    q->queue_id = cq.queue_id;

    q->doorbell = (volatile uint32_t*)mmap(
        NULL, AM_PAGE_SIZE, PROT_READ | PROT_WRITE,
        MAP_SHARED, drv->fd_kfd, cq.doorbell_offset);

    if (q->doorbell == MAP_FAILED) {
        LOG_ERROR("AM driver: failed to mmap doorbell: %s", strerror(errno));
        q->doorbell = NULL;
        return -1;
    }

    q->active = true;
    return 0;
}
#endif

int cml_am_driver_init(CMLAMDriver* drv) {
#ifdef __linux__
    if (!drv) return -1;

    if (drv->initialized)
        return 0;

    /* Discover GPUs and pick the first one if gpu_id not set */
    if (drv->gpu_id == 0) {
        CMLAMGPUInfo* gpus = NULL;
        int gpu_count = 0;
        if (cml_am_enumerate_gpus(&gpus, &gpu_count) == 0 && gpu_count > 0) {
            drv->gpu_id    = gpus[0].gpu_id;
            drv->gpu_info  = gpus[0];
            drv->cu_count  = gpus[0].cu_count;
            drv->total_vram = gpus[0].vram_size;
            snprintf(drv->device_name, sizeof(drv->device_name), "%s", gpus[0].name);
            snprintf(drv->gfx_version, sizeof(drv->gfx_version), "%s", gpus[0].gfx_version);
        }
        free(gpus);

        if (drv->gpu_id == 0) {
            LOG_ERROR("AM driver: no GPU found via KFD topology");
            return -1;
        }
    }

    drv->chiplet_config = cml_am_get_chiplet_config(drv->gfx_version);
    drv->is_chiplet_gpu = (drv->chiplet_config.num_xcd > 1);

    if (drv->is_chiplet_gpu) {
        LOG_INFO("AM driver: chiplet GPU detected (%s): %d XCDs, %d CUs/XCD, unified_mem=%d",
                 drv->gfx_version, drv->chiplet_config.num_xcd,
                 drv->chiplet_config.cu_per_xcd,
                 drv->chiplet_config.unified_memory);
    }

    drv->fd_kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if (drv->fd_kfd < 0) {
        LOG_ERROR("AM driver: cannot open /dev/kfd: %s", strerror(errno));
        return -1;
    }

    drv->fd_drm = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (drv->fd_drm < 0) {
        LOG_ERROR("AM driver: cannot open /dev/dri/renderD128: %s",
                  strerror(errno));
        close(drv->fd_kfd);
        drv->fd_kfd = -1;
        return -1;
    }

    struct kfd_ioctl_get_version_args ver = {0};
    if (kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_GET_VERSION, &ver) != 0) {
        LOG_ERROR("AM driver: KFD GET_VERSION failed");
        goto fail;
    }
    LOG_INFO("AM driver: KFD version %u.%u", ver.major_version, ver.minor_version);

    struct kfd_ioctl_acquire_vm_args acq = {0};
    acq.drm_fd = (uint32_t)drv->fd_drm;
    acq.gpu_id = drv->gpu_id;
    if (kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_ACQUIRE_VM, &acq) != 0) {
        LOG_WARNING("AM driver: ACQUIRE_VM failed (non-fatal)");
    }

    /* Signal memory */
    uint64_t sig_handle = 0, sig_va = 0;
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
    if (drv->signal)
        *drv->signal = AM_SIGNAL_INIT;

    /* Primary AQL queue */
    if (am_init_aql_queue(drv, &drv->aql_queue) != 0) {
        LOG_ERROR("AM driver: failed to create primary AQL queue");
        goto fail;
    }

    /* Also store as compute_queues[0] */
    drv->compute_queues[0] = drv->aql_queue;
    drv->num_compute_queues = 1;

    /* Attempt SDMA queue creation */
    if (cml_am_sdma_queue_create(drv) == 0) {
        drv->has_sdma = true;
        LOG_INFO("AM driver: SDMA queue created");
    } else {
        LOG_DEBUG("AM driver: SDMA queue not available, using CPU copies");
    }

    drv->initialized = true;
    LOG_INFO("AM driver: initialized [%s] gpu_id=%u CUs=%d VRAM=%zuMB",
             drv->gfx_version, drv->gpu_id, drv->cu_count,
             drv->total_vram / (1024*1024));
    return 0;

fail:
    if (drv->fd_drm >= 0) { close(drv->fd_drm); drv->fd_drm = -1; }
    if (drv->fd_kfd >= 0) { close(drv->fd_kfd); drv->fd_kfd = -1; }
    return -1;

#else
    (void)drv;
    LOG_ERROR("AM driver: only supported on Linux");
    return -1;
#endif
}

void cml_am_driver_free(CMLAMDriver* drv) {
    if (!drv) return;

#ifdef __linux__
    if (drv->initialized) {
        cml_am_synchronize(drv);

        /* Destroy compute queues */
        for (int i = 0; i < drv->num_compute_queues; i++) {
            CMLAMQueue* q = &drv->compute_queues[i];
            if (q->queue_id != 0) {
                struct kfd_ioctl_destroy_queue_args dq = {
                    .queue_id = (uint32_t)q->queue_id
                };
                kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_DESTROY_QUEUE, &dq);
            }
            if (q->doorbell && q->doorbell != MAP_FAILED) {
                munmap((void*)q->doorbell, AM_PAGE_SIZE);
            }
        }

        /* Also destroy legacy aql_queue if it differs from compute_queues[0] */
        if (drv->aql_queue.queue_id != 0 &&
            (drv->num_compute_queues == 0 ||
             drv->aql_queue.queue_id != drv->compute_queues[0].queue_id)) {
            struct kfd_ioctl_destroy_queue_args dq = {
                .queue_id = (uint32_t)drv->aql_queue.queue_id
            };
            kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_DESTROY_QUEUE, &dq);
            if (drv->aql_queue.doorbell && drv->aql_queue.doorbell != MAP_FAILED)
                munmap((void*)drv->aql_queue.doorbell, AM_PAGE_SIZE);
        }

        /* Destroy SDMA queue */
        if (drv->has_sdma && drv->sdma_queue.queue_id != 0) {
            struct kfd_ioctl_destroy_queue_args dq = {
                .queue_id = (uint32_t)drv->sdma_queue.queue_id
            };
            kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_DESTROY_QUEUE, &dq);
            if (drv->sdma_queue.doorbell && drv->sdma_queue.doorbell != MAP_FAILED)
                munmap((void*)drv->sdma_queue.doorbell, AM_PAGE_SIZE);
        }

        /* Free scratch */
        if (drv->has_scratch) {
            am_free_and_unmap(drv, drv->scratch.handle, NULL, drv->scratch.size);
        }
    }

    if (drv->fd_drm >= 0) close(drv->fd_drm);
    if (drv->fd_kfd >= 0) close(drv->fd_kfd);
#endif

    free(drv);
}


/* Multi-queue */

int cml_am_create_compute_queue(CMLAMDriver* drv, int queue_index) {
#ifdef __linux__
    if (!drv || !drv->initialized) return -1;
    if (queue_index < 0 || queue_index >= AM_MAX_COMPUTE_QUEUES) return -1;

    if (drv->compute_queues[queue_index].active) {
        LOG_DEBUG("AM driver: compute queue %d already active", queue_index);
        return 0;
    }

    CMLAMQueue q;
    memset(&q, 0, sizeof(q));
    if (am_init_aql_queue(drv, &q) != 0) {
        LOG_ERROR("AM driver: failed to create compute queue %d", queue_index);
        return -1;
    }

    drv->compute_queues[queue_index] = q;
    if (queue_index >= drv->num_compute_queues)
        drv->num_compute_queues = queue_index + 1;

    LOG_INFO("AM driver: compute queue %d created (queue_id=%lu)",
             queue_index, (unsigned long)q.queue_id);
    return 0;
#else
    (void)drv; (void)queue_index;
    return -1;
#endif
}


/* SDMA queue */

int cml_am_sdma_queue_create(CMLAMDriver* drv) {
#ifdef __linux__
    if (!drv || drv->fd_kfd < 0) return -1;

    uint32_t gtt_flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT
                       | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                       | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;

    uint64_t ring_handle = 0, ring_va = 0;
    void*    ring_addr = NULL;
    if (am_alloc_and_map(drv, AM_SDMA_RING_SIZE, gtt_flags,
                         &ring_handle, &ring_va, &ring_addr) != 0)
        return -1;

    drv->sdma_queue.ring = (uint32_t*)ring_addr;
    drv->sdma_queue.ring_size = AM_SDMA_RING_SIZE;

    uint64_t wptr_handle = 0, wptr_va = 0;
    void*    wptr_addr = NULL;
    if (am_alloc_and_map(drv, AM_PAGE_SIZE, gtt_flags,
                         &wptr_handle, &wptr_va, &wptr_addr) != 0)
        return -1;

    drv->sdma_queue.write_ptr = (volatile uint64_t*)wptr_addr;
    drv->sdma_queue.read_ptr  = (volatile uint64_t*)((uint8_t*)wptr_addr + 64);

    if (drv->sdma_queue.write_ptr) *drv->sdma_queue.write_ptr = 0;
    if (drv->sdma_queue.read_ptr)  *drv->sdma_queue.read_ptr = 0;

    struct kfd_ioctl_create_queue_args cq = {0};
    cq.ring_base_address     = ring_va;
    cq.write_pointer_address = wptr_va;
    cq.read_pointer_address  = wptr_va + 64;
    cq.ring_size             = AM_SDMA_RING_SIZE;
    cq.gpu_id                = drv->gpu_id;
    cq.queue_type            = KFD_IOC_QUEUE_TYPE_SDMA;
    cq.queue_percentage      = 100;
    cq.queue_priority        = 7;

    if (kfd_ioctl(drv->fd_kfd, AMDKFD_IOC_CREATE_QUEUE, &cq) != 0) {
        LOG_DEBUG("AM driver: SDMA CREATE_QUEUE failed");
        return -1;
    }

    drv->sdma_queue.queue_id = cq.queue_id;

    drv->sdma_queue.doorbell = (volatile uint32_t*)mmap(
        NULL, AM_PAGE_SIZE, PROT_READ | PROT_WRITE,
        MAP_SHARED, drv->fd_kfd, cq.doorbell_offset);

    if (drv->sdma_queue.doorbell == MAP_FAILED) {
        LOG_ERROR("AM driver: SDMA doorbell mmap failed: %s", strerror(errno));
        drv->sdma_queue.doorbell = NULL;
        return -1;
    }

    drv->sdma_queue.active = true;
    return 0;
#else
    (void)drv;
    return -1;
#endif
}

int cml_am_sdma_copy(CMLAMDriver* drv, uint64_t dst_va, uint64_t src_va, size_t size) {
#ifdef __linux__
    if (!drv || !drv->has_sdma || !drv->sdma_queue.active) return -1;
    if (size == 0) return 0;

    CMLAMSDMAQueue* sq = &drv->sdma_queue;
    if (!sq->ring || !sq->write_ptr || !sq->doorbell) return -1;

    uint64_t wp = *sq->write_ptr;
    uint32_t byte_offset = (uint32_t)(wp % sq->ring_size);

    /* SDMA linear copy can do up to 2^26 - 1 bytes per packet. Split large copies. */
    size_t remaining = size;
    uint64_t src = src_va;
    uint64_t dst = dst_va;

    while (remaining > 0) {
        size_t chunk = remaining;
        if (chunk > (1 << 26) - 1)
            chunk = (1 << 26) - 1;

        SDMACopyPacket pkt;
        pkt.op_subop = (SDMA_OP_COPY) | ((uint32_t)SDMA_SUBOP_COPY_LINEAR << 8);
        pkt.count_minus_one = (uint32_t)(chunk - 1);
        pkt.pad = 0;
        pkt.src_addr_lo = (uint32_t)(src & 0xFFFFFFFF);
        pkt.src_addr_hi = (uint32_t)(src >> 32);
        pkt.dst_addr_lo = (uint32_t)(dst & 0xFFFFFFFF);
        pkt.dst_addr_hi = (uint32_t)(dst >> 32);

        uint32_t pkt_dwords = sizeof(SDMACopyPacket) / 4;
        uint8_t* ring_base = (uint8_t*)sq->ring;

        for (uint32_t d = 0; d < pkt_dwords; d++) {
            uint32_t off = (byte_offset + d * 4) % sq->ring_size;
            memcpy(ring_base + off, (uint8_t*)&pkt + d * 4, 4);
        }

        byte_offset = (byte_offset + pkt_dwords * 4) % sq->ring_size;
        wp += pkt_dwords * 4;

        remaining -= chunk;
        src += chunk;
        dst += chunk;
    }

    am_mb();
    *sq->write_ptr = wp;
    am_mb();
    *sq->doorbell = (uint32_t)(wp);

    return 0;
#else
    (void)drv; (void)dst_va; (void)src_va; (void)size;
    return -1;
#endif
}

int cml_am_sdma_fence(CMLAMDriver* drv, uint64_t signal_va, uint64_t value) {
#ifdef __linux__
    if (!drv || !drv->has_sdma || !drv->sdma_queue.active) return -1;

    CMLAMSDMAQueue* sq = &drv->sdma_queue;
    if (!sq->ring || !sq->write_ptr || !sq->doorbell) return -1;

    uint64_t wp = *sq->write_ptr;
    uint32_t byte_offset = (uint32_t)(wp % sq->ring_size);

    SDMAFencePacket pkt;
    pkt.op_subop = SDMA_OP_FENCE;
    pkt.addr_lo  = (uint32_t)(signal_va & 0xFFFFFFFF);
    pkt.addr_hi  = (uint32_t)(signal_va >> 32);
    pkt.value    = (uint32_t)value;

    uint32_t pkt_dwords = sizeof(SDMAFencePacket) / 4;
    uint8_t* ring_base = (uint8_t*)sq->ring;

    for (uint32_t d = 0; d < pkt_dwords; d++) {
        uint32_t off = (byte_offset + d * 4) % sq->ring_size;
        memcpy(ring_base + off, (uint8_t*)&pkt + d * 4, 4);
    }

    wp += pkt_dwords * 4;
    am_mb();
    *sq->write_ptr = wp;
    am_mb();
    *sq->doorbell = (uint32_t)(wp);

    return 0;
#else
    (void)drv; (void)signal_va; (void)value;
    return -1;
#endif
}

int cml_am_sdma_synchronize(CMLAMDriver* drv) {
#ifdef __linux__
    if (!drv || !drv->has_sdma || !drv->sdma_queue.active) return -1;

    CMLAMSDMAQueue* sq = &drv->sdma_queue;
    if (!sq->write_ptr || !sq->read_ptr) return -1;

    uint64_t expected = *sq->write_ptr;
    uint64_t timeout_us = 5000000;
    uint64_t elapsed = 0;
    uint64_t poll_us = 10;

    while (elapsed < timeout_us) {
        am_mb();
        uint64_t current = *sq->read_ptr;
        if (current >= expected)
            return 0;

        usleep((useconds_t)poll_us);
        elapsed += poll_us;
        if (poll_us < 1000) poll_us *= 2;
    }

    LOG_ERROR("AM driver: SDMA synchronize timed out");
    return -1;
#else
    (void)drv;
    return -1;
#endif
}


/* Signal system */

CMLAMSignal* cml_am_signal_create(CMLAMDriver* drv, uint64_t initial_value) {
#ifdef __linux__
    if (!drv || !drv->initialized) return NULL;

    CMLAMSignal* sig = (CMLAMSignal*)calloc(1, sizeof(CMLAMSignal));
    if (!sig) return NULL;

    uint64_t handle = 0, va = 0;
    void*    addr = NULL;
    uint32_t flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT
                   | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                   | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;

    if (am_alloc_and_map(drv, sizeof(uint64_t) * 8, flags, &handle, &va, &addr) != 0) {
        free(sig);
        return NULL;
    }

    sig->value  = (volatile uint64_t*)addr;
    sig->gpu_va = va;
    sig->handle = handle;
    sig->target = initial_value;

    if (sig->value)
        *sig->value = initial_value;

    return sig;
#else
    (void)drv; (void)initial_value;
    return NULL;
#endif
}

void cml_am_signal_free(CMLAMDriver* drv, CMLAMSignal* signal) {
    if (!drv || !signal) return;

#ifdef __linux__
    if (drv->initialized) {
        am_free_and_unmap(drv, signal->handle,
                          (void*)signal->value, sizeof(uint64_t) * 8);
    }
#endif

    free(signal);
}

int cml_am_signal_wait(CMLAMSignal* signal, uint64_t expected, uint64_t timeout_ns) {
#ifdef __linux__
    if (!signal || !signal->value) return -1;

    uint64_t timeout_us = timeout_ns / 1000;
    if (timeout_us == 0) timeout_us = 1;
    uint64_t elapsed = 0;
    uint64_t poll_us = 10;

    while (elapsed < timeout_us) {
        am_mb();
        uint64_t current = *signal->value;
        if (current >= expected) {
            signal->target = current;
            return 0;
        }

        usleep((useconds_t)poll_us);
        elapsed += poll_us;
        if (poll_us < 1000) poll_us *= 2;
    }

    return -1;
#else
    (void)signal; (void)expected; (void)timeout_ns;
    return -1;
#endif
}


/* Barrier packets */

#ifdef __linux__
static uint16_t am_make_barrier_header(int pkt_type) {
    return (uint16_t)(
        (pkt_type << AQL_HDR_TYPE_SHIFT)
      | (1 << AQL_HDR_BARRIER_SHIFT)
      | (AQL_FENCE_SCOPE_SYSTEM << AQL_HDR_ACQUIRE_SHIFT)
      | (AQL_FENCE_SCOPE_SYSTEM << AQL_HDR_RELEASE_SHIFT)
    );
}

static int am_submit_barrier(CMLAMDriver* drv, int queue_idx, int pkt_type,
                             CMLAMSignal** deps, int num_deps,
                             CMLAMSignal* completion) {
    if (!drv || !drv->initialized) return -1;
    if (queue_idx < 0 || queue_idx >= drv->num_compute_queues) return -1;

    CMLAMQueue* q = &drv->compute_queues[queue_idx];
    if (!q->active || !q->ring || !q->write_dispatch_id || !q->doorbell)
        return -1;

    if (num_deps > 5) num_deps = 5;

    uint64_t write_idx = *q->write_dispatch_id;
    uint32_t slot = (uint32_t)(write_idx % q->ring_size);

    uint8_t* pkt = (uint8_t*)&q->ring[slot];
    memset(pkt, 0, 64);

    for (int i = 0; i < num_deps; i++) {
        if (deps[i]) {
            uint64_t dep_va = deps[i]->gpu_va;
            memcpy(pkt + 8 + i * 8, &dep_va, sizeof(uint64_t));
        }
    }

    if (completion) {
        uint64_t comp_va = completion->gpu_va;
        memcpy(pkt + 56, &comp_va, sizeof(uint64_t));
    }

    am_mb();

    uint16_t header = am_make_barrier_header(pkt_type);
    memcpy(pkt, &header, sizeof(uint16_t));

    am_mb();

    *q->write_dispatch_id = write_idx + 1;
    am_mb();
    *q->doorbell = (uint32_t)(write_idx + 1);

    return 0;
}
#endif

int cml_am_barrier_and(CMLAMDriver* drv, int queue_idx,
                       CMLAMSignal** deps, int num_deps,
                       CMLAMSignal* completion) {
#ifdef __linux__
    return am_submit_barrier(drv, queue_idx, AQL_PKT_TYPE_BARRIER_AND,
                             deps, num_deps, completion);
#else
    (void)drv; (void)queue_idx; (void)deps; (void)num_deps; (void)completion;
    return -1;
#endif
}

int cml_am_barrier_or(CMLAMDriver* drv, int queue_idx,
                      CMLAMSignal** deps, int num_deps,
                      CMLAMSignal* completion) {
#ifdef __linux__
    return am_submit_barrier(drv, queue_idx, AQL_PKT_TYPE_BARRIER_OR,
                             deps, num_deps, completion);
#else
    (void)drv; (void)queue_idx; (void)deps; (void)num_deps; (void)completion;
    return -1;
#endif
}


/* Scratch and LDS */

int cml_am_alloc_scratch(CMLAMDriver* drv, size_t per_thread_size, uint32_t max_waves) {
#ifdef __linux__
    if (!drv || !drv->initialized) return -1;

    if (drv->has_scratch) {
        am_free_and_unmap(drv, drv->scratch.handle, NULL, drv->scratch.size);
        drv->has_scratch = false;
    }

    size_t wave_size = 64;
    size_t total = per_thread_size * wave_size * max_waves;
    total = AM_PAGE_ALIGN(total);

    uint64_t handle = 0, va = 0;
    void*    addr = NULL;
    uint32_t flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM
                   | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;

    if (am_alloc_and_map(drv, total, flags, &handle, &va, &addr) != 0) {
        LOG_ERROR("AM driver: scratch allocation failed (%zu bytes)", total);
        return -1;
    }

    drv->scratch.gpu_va = va;
    drv->scratch.handle = handle;
    drv->scratch.size = total;
    drv->scratch.per_thread_size = per_thread_size;
    drv->scratch.max_waves = max_waves;
    drv->has_scratch = true;

    LOG_DEBUG("AM driver: scratch allocated %zu bytes at 0x%lx",
              total, (unsigned long)va);
    return 0;
#else
    (void)drv; (void)per_thread_size; (void)max_waves;
    return -1;
#endif
}

bool cml_am_validate_lds(CMLAMDriver* drv, uint32_t requested_bytes) {
#ifdef __linux__
    if (!drv || !drv->initialized) return false;

    uint32_t max_lds = drv->gpu_info.lds_size_per_cu;
    if (max_lds == 0) {
        /* Default to 64KB if topology didn't report it */
        max_lds = 65536;
    }

    if (requested_bytes > max_lds) {
        LOG_ERROR("AM driver: LDS request %u exceeds hardware limit %u",
                  requested_bytes, max_lds);
        return false;
    }

    return true;
#else
    (void)drv; (void)requested_bytes;
    return false;
#endif
}


/* Error recovery */

int cml_am_check_gpu_hang(CMLAMDriver* drv) {
#ifdef __linux__
    if (!drv || !drv->initialized) return -1;

    char path[256];
    snprintf(path, sizeof(path),
             "/sys/class/drm/renderD128/device/gpu_busy_percent");
    uint32_t busy = 0;
    if (sysfs_read_u32(path, &busy) != 0) {
        LOG_DEBUG("AM driver: cannot read GPU busy percent");
        return -1;
    }

    /* Check if queue has progressed */
    CMLAMQueue* q = &drv->aql_queue;
    if (q->write_dispatch_id && q->read_dispatch_id) {
        uint64_t write = *q->write_dispatch_id;
        uint64_t read  = *q->read_dispatch_id;

        if (write > read && busy == 100) {
            LOG_WARNING("AM driver: possible GPU hang (write=%lu read=%lu busy=100%%)",
                        (unsigned long)write, (unsigned long)read);
            return 1;
        }
    }

    return 0;
#else
    (void)drv;
    return -1;
#endif
}

int cml_am_gpu_reset(CMLAMDriver* drv) {
#ifdef __linux__
    if (!drv || !drv->initialized) return -1;

    /* KFD does not expose a direct reset ioctl to userspace.
     * The kernel handles GPU resets internally when it detects a hang.
     * We can trigger a reset by writing to the amdgpu debugfs reset file
     * if available and if we have root privileges. */
    const char* reset_path = "/sys/kernel/debug/dri/0/amdgpu_gpu_recover";
    int fd = open(reset_path, O_WRONLY);
    if (fd < 0) {
        LOG_ERROR("AM driver: cannot open GPU reset file (requires root): %s",
                  strerror(errno));
        return -1;
    }

    if (write(fd, "1", 1) != 1) {
        LOG_ERROR("AM driver: GPU reset write failed: %s", strerror(errno));
        close(fd);
        return -1;
    }

    close(fd);
    LOG_WARNING("AM driver: GPU reset triggered");

    /* Driver state is now invalid; mark as uninitialized */
    drv->initialized = false;
    return 0;
#else
    (void)drv;
    return -1;
#endif
}

int cml_am_dump_wave_status(CMLAMDriver* drv) {
#ifdef __linux__
    if (!drv) return -1;

    /* Read wave status from debugfs */
    const char* wave_path = "/sys/kernel/debug/dri/0/amdgpu_wave_status";
    FILE* f = fopen(wave_path, "r");
    if (!f) {
        LOG_DEBUG("AM driver: cannot open wave status (requires root)");
        return -1;
    }

    char line[256];
    int count = 0;
    while (fgets(line, sizeof(line), f) && count < 64) {
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
        LOG_INFO("wave: %s", line);
        count++;
    }

    fclose(f);
    return 0;
#else
    (void)drv;
    return -1;
#endif
}


/* Buffer management */

CMLAMBuffer* cml_am_buffer_create(CMLAMDriver* drv, size_t size, bool vram) {
#ifdef __linux__
    if (!drv || !drv->initialized || size == 0) {
        LOG_ERROR("AM driver: invalid args to buffer_create");
        return NULL;
    }

    CMLAMBuffer* buf = (CMLAMBuffer*)calloc(1, sizeof(CMLAMBuffer));
    if (!buf) return NULL;

    uint32_t flags;
    if (vram) {
        flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM
              | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
        buf->is_vram = true;
    } else {
        flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT
              | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
              | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
        buf->is_vram = false;
    }

    uint64_t handle = 0, va = 0;
    void* cpu_addr = NULL;

    if (am_alloc_and_map(drv, size, flags, &handle, &va, &cpu_addr) != 0) {
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
        memcpy(dst->cpu_addr, src, n);
        am_mb();
        return 0;
    }

    /* VRAM without CPU mapping: use SDMA if available, else staging */
    CMLAMBuffer* staging = cml_am_buffer_create(drv, n, false);
    if (!staging) return -1;

    memcpy(staging->cpu_addr, src, n);
    am_mb();

    if (drv->has_sdma) {
        int ret = cml_am_sdma_copy(drv, dst->gpu_va, staging->gpu_va, n);
        if (ret == 0) {
            cml_am_sdma_synchronize(drv);
            cml_am_buffer_free(drv, staging);
            return 0;
        }
    }

    /* Fallback: remap dst's va to staging for GPU access */
    dst->gpu_va = staging->gpu_va;
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
        am_mb();
        memcpy(dst, src->cpu_addr, n);
        return 0;
    }

    /* VRAM: use SDMA to copy to a staging buffer */
    CMLAMBuffer* staging = cml_am_buffer_create(drv, n, false);
    if (!staging) return -1;

    if (drv->has_sdma) {
        int ret = cml_am_sdma_copy(drv, staging->gpu_va, src->gpu_va, n);
        if (ret == 0) {
            cml_am_sdma_synchronize(drv);
            am_mb();
            memcpy(dst, staging->cpu_addr, n);
            cml_am_buffer_free(drv, staging);
            return 0;
        }
    }

    /* Fallback: try direct copy if staging has CPU addr */
    if (staging->cpu_addr) {
        am_mb();
        memcpy(dst, staging->cpu_addr, n);
        cml_am_buffer_free(drv, staging);
        return 0;
    }

    cml_am_buffer_free(drv, staging);
    LOG_WARNING("AM driver: VRAM download failed (no CPU mapping)");
    return -1;
#else
    (void)drv; (void)src; (void)dst; (void)n;
    return -1;
#endif
}


/* ELF parsing and kernel loading */

/* ELF field reading macros (little-endian assumed for AMDGPU) */
#define ELF_U16(elf, off) ((uint16_t)((elf)[off] | ((uint16_t)(elf)[(off)+1] << 8)))
#define ELF_U32(elf, off) ((uint32_t)((elf)[off] | ((uint32_t)(elf)[(off)+1] << 8) | \
                           ((uint32_t)(elf)[(off)+2] << 16) | ((uint32_t)(elf)[(off)+3] << 24)))
#define ELF_U64(elf, off) ((uint64_t)ELF_U32(elf, off) | ((uint64_t)ELF_U32(elf, (off)+4) << 32))
#define ELF_I64(elf, off) ((int64_t)ELF_U64(elf, off))

int am_parse_kernel_descriptor(const void* code_object, size_t code_size,
                               const char* kernel_name, AMDGPUKernelDescriptor* kd) {
#ifdef __linux__
    if (!code_object || code_size < 64 || !kd) return -1;

    const uint8_t* elf = (const uint8_t*)code_object;

    if (elf[0] != 0x7f || elf[1] != 'E' || elf[2] != 'L' || elf[3] != 'F')
        return -1;
    if (elf[4] != 2) return -1;

    uint64_t e_shoff     = ELF_U64(elf, 40);
    uint16_t e_shentsize = ELF_U16(elf, 58);
    uint16_t e_shnum     = ELF_U16(elf, 60);
    uint16_t e_shstrndx  = ELF_U16(elf, 62);

    if (e_shoff == 0 || e_shnum == 0 || e_shentsize < 64) return -1;
    if (e_shoff + (uint64_t)e_shnum * e_shentsize > code_size) return -1;

    const uint8_t* shstrtab = NULL;
    uint64_t shstrtab_size = 0;
    if (e_shstrndx < e_shnum) {
        const uint8_t* strhdr = elf + e_shoff + (uint64_t)e_shstrndx * e_shentsize;
        uint64_t str_off  = ELF_U64(elf, (uint64_t)(strhdr - elf) + 24);
        uint64_t str_size = ELF_U64(elf, (uint64_t)(strhdr - elf) + 32);
        if (str_off + str_size <= code_size) {
            shstrtab = elf + str_off;
            shstrtab_size = str_size;
        }
    }

    /* Find .text section */
    for (uint16_t i = 0; i < e_shnum; i++) {
        uint64_t sh_base = e_shoff + (uint64_t)i * e_shentsize;
        uint32_t sh_name_idx = ELF_U32(elf, sh_base);
        uint64_t sh_offset   = ELF_U64(elf, sh_base + 24);
        uint64_t sh_size     = ELF_U64(elf, sh_base + 32);

        if (sh_offset + sh_size > code_size) continue;

        const char* sec_name = NULL;
        if (shstrtab && sh_name_idx < shstrtab_size)
            sec_name = (const char*)(shstrtab + sh_name_idx);

        if (!sec_name || strcmp(sec_name, ".text") != 0) continue;

        /* Find the symbol for this kernel if name is provided */
        uint64_t kd_offset = sh_offset;

        if (kernel_name) {
            /* Look through symtab to find the kernel symbol */
            for (uint16_t j = 0; j < e_shnum; j++) {
                uint64_t sym_base = e_shoff + (uint64_t)j * e_shentsize;
                uint32_t sym_type = ELF_U32(elf, sym_base + 4);
                if (sym_type != 2 && sym_type != 11) continue; /* SHT_SYMTAB or SHT_DYNSYM */

                uint64_t sym_off  = ELF_U64(elf, sym_base + 24);
                uint64_t sym_size = ELF_U64(elf, sym_base + 32);
                uint32_t sym_entsize = ELF_U32(elf, sym_base + 56);
                uint32_t sym_link = ELF_U32(elf, sym_base + 40);

                if (sym_entsize < 24 || sym_off + sym_size > code_size) continue;

                /* Get string table for symbols */
                const uint8_t* sym_strtab = NULL;
                uint64_t sym_strtab_size = 0;
                if (sym_link < e_shnum) {
                    uint64_t sl_base = e_shoff + (uint64_t)sym_link * e_shentsize;
                    uint64_t sl_off  = ELF_U64(elf, sl_base + 24);
                    uint64_t sl_size = ELF_U64(elf, sl_base + 32);
                    if (sl_off + sl_size <= code_size) {
                        sym_strtab = elf + sl_off;
                        sym_strtab_size = sl_size;
                    }
                }

                if (!sym_strtab) continue;

                uint64_t num_syms = sym_size / sym_entsize;
                for (uint64_t s = 0; s < num_syms; s++) {
                    uint64_t se = sym_off + s * sym_entsize;
                    uint32_t st_name = ELF_U32(elf, se);
                    uint64_t st_value = ELF_U64(elf, se + 8);

                    if (st_name >= sym_strtab_size) continue;
                    const char* sname = (const char*)(sym_strtab + st_name);

                    /* Match kernel name; AMDGPU appends ".kd" suffix */
                    size_t nlen = strlen(kernel_name);
                    if (strncmp(sname, kernel_name, nlen) == 0) {
                        if (sname[nlen] == '\0' || strcmp(sname + nlen, ".kd") == 0) {
                            kd_offset = sh_offset + st_value;
                            goto found;
                        }
                    }
                }
            }
        }

found:
        if (kd_offset + sizeof(AMDGPUKernelDescriptor) > code_size)
            return -1;

        memcpy(kd, elf + kd_offset, sizeof(AMDGPUKernelDescriptor));
        return 0;
    }

    return -1;
#else
    (void)code_object; (void)code_size; (void)kernel_name; (void)kd;
    return -1;
#endif
}


CMLAMKernel* cml_am_kernel_load(CMLAMDriver* drv, const void* code_object,
                                size_t code_size, const char* kernel_name) {
#ifdef __linux__
    if (!drv || !drv->initialized || !code_object || code_size == 0 || !kernel_name)
        return NULL;

    CMLAMKernel* kernel = (CMLAMKernel*)calloc(1, sizeof(CMLAMKernel));
    if (!kernel) return NULL;

    uint64_t handle = 0, va = 0;
    void* cpu_addr = NULL;
    uint32_t flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM
                   | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
                   | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                   | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;

    if (am_alloc_and_map(drv, code_size, flags, &handle, &va, &cpu_addr) != 0) {
        free(kernel);
        return NULL;
    }

    if (cpu_addr) {
        memcpy(cpu_addr, code_object, code_size);
        am_mb();
    } else {
        am_free_and_unmap(drv, handle, cpu_addr, code_size);
        free(kernel);
        return NULL;
    }

    kernel->code_object = malloc(code_size);
    if (kernel->code_object)
        memcpy(kernel->code_object, code_object, code_size);
    kernel->code_size = code_size;
    kernel->gpu_addr  = va;
    kernel->handle    = (uint32_t)handle;
    kernel->name      = strdup(kernel_name);

    /* Try kernel descriptor parsing first */
    AMDGPUKernelDescriptor kd;
    if (am_parse_kernel_descriptor(code_object, code_size, kernel_name, &kd) == 0) {
        kernel->group_segment_size   = kd.group_segment_fixed_size;
        kernel->private_segment_size = kd.private_segment_fixed_size;
        kernel->kernarg_size         = kd.kernarg_size;
        kernel->kern_code_entry_offset = kd.kernel_code_entry_byte_offset;

        bool is_gfx10_plus = (drv->gfx_version[3] >= '1' && drv->gfx_version[4] >= '0');
        kernel->vgpr_count = amdgpu_vgpr_count(kd.compute_pgm_rsrc1, is_gfx10_plus);
        kernel->sgpr_count = amdgpu_sgpr_count(kd.compute_pgm_rsrc1);

        LOG_DEBUG("AM driver: KD parsed - group=%u private=%u kernarg=%u vgpr=%u sgpr=%u entry_off=%ld",
                  kernel->group_segment_size, kernel->private_segment_size,
                  kernel->kernarg_size, kernel->vgpr_count, kernel->sgpr_count,
                  (long)kernel->kern_code_entry_offset);
    } else {
        /* Fallback: parse AMDGPU metadata note (msgpack) */
        kernel->group_segment_size   = 0;
        kernel->private_segment_size = 0;
        kernel->kernarg_size         = 0;

        const uint8_t* elf = (const uint8_t*)code_object;
        if (code_size >= 64 && elf[0] == 0x7f && elf[1] == 'E' &&
            elf[2] == 'L' && elf[3] == 'F' && elf[4] == 2) {

            uint64_t e_shoff     = ELF_U64(elf, 40);
            uint16_t e_shentsize = ELF_U16(elf, 58);
            uint16_t e_shnum     = ELF_U16(elf, 60);

            if (e_shoff > 0 && e_shnum > 0 && e_shentsize >= 64 &&
                e_shoff + (uint64_t)e_shnum * e_shentsize <= code_size) {

                for (uint16_t i = 0; i < e_shnum; i++) {
                    uint64_t sh_base = e_shoff + (uint64_t)i * e_shentsize;
                    uint32_t sh_type   = ELF_U32(elf, sh_base + 4);
                    uint64_t sh_offset = ELF_U64(elf, sh_base + 24);
                    uint64_t sh_size   = ELF_U64(elf, sh_base + 32);

                    if (sh_offset + sh_size > code_size) continue;
                    if (sh_type != 7 || sh_size < 12) continue;

                    uint64_t pos = sh_offset;
                    uint64_t end = sh_offset + sh_size;
                    while (pos + 12 <= end) {
                        uint32_t n_namesz = ELF_U32(elf, pos);
                        uint32_t n_descsz = ELF_U32(elf, pos + 4);
                        uint32_t n_type   = ELF_U32(elf, pos + 8);
                        pos += 12;

                        uint32_t name_aligned = (n_namesz + 3) & ~(uint32_t)3;
                        uint32_t desc_aligned = (n_descsz + 3) & ~(uint32_t)3;
                        if (pos + name_aligned + desc_aligned > end) break;

                        if (n_type == 32 && n_namesz >= 6 && pos + 6 <= end &&
                            memcmp(elf + pos, "AMDGPU", 6) == 0) {
                            const char* desc = (const char*)(elf + pos + name_aligned);
                            uint32_t desc_len = n_descsz;

                            for (uint32_t d = 0; d + 20 < desc_len; d++) {
                                if (desc[d] != '.') continue;

                                if (d + 21 < desc_len &&
                                    memcmp(desc + d, ".kernarg_segment_size", 21) == 0) {
                                    for (uint32_t k = d + 21; k < desc_len && k < d + 30; k++) {
                                        uint8_t b = (uint8_t)desc[k];
                                        if (b > 0 && b < 0x80) { kernel->kernarg_size = b; break; }
                                        if (b == 0xce && k + 4 < desc_len) {
                                            kernel->kernarg_size = (uint32_t)(
                                                ((uint8_t)desc[k+1] << 24) | ((uint8_t)desc[k+2] << 16) |
                                                ((uint8_t)desc[k+3] << 8)  | ((uint8_t)desc[k+4]));
                                            break;
                                        }
                                    }
                                }
                                if (d + 25 < desc_len &&
                                    memcmp(desc + d, ".group_segment_fixed_size", 25) == 0) {
                                    for (uint32_t k = d + 25; k < desc_len && k < d + 34; k++) {
                                        uint8_t b = (uint8_t)desc[k];
                                        if (b > 0 && b < 0x80) { kernel->group_segment_size = b; break; }
                                        if (b == 0xce && k + 4 < desc_len) {
                                            kernel->group_segment_size = (uint32_t)(
                                                ((uint8_t)desc[k+1] << 24) | ((uint8_t)desc[k+2] << 16) |
                                                ((uint8_t)desc[k+3] << 8)  | ((uint8_t)desc[k+4]));
                                            break;
                                        }
                                    }
                                }
                                if (d + 27 < desc_len &&
                                    memcmp(desc + d, ".private_segment_fixed_size", 27) == 0) {
                                    for (uint32_t k = d + 27; k < desc_len && k < d + 36; k++) {
                                        uint8_t b = (uint8_t)desc[k];
                                        if (b > 0 && b < 0x80) { kernel->private_segment_size = b; break; }
                                        if (b == 0xce && k + 4 < desc_len) {
                                            kernel->private_segment_size = (uint32_t)(
                                                ((uint8_t)desc[k+1] << 24) | ((uint8_t)desc[k+2] << 16) |
                                                ((uint8_t)desc[k+3] << 8)  | ((uint8_t)desc[k+4]));
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        pos += name_aligned + desc_aligned;
                    }
                }
            }
        }

        LOG_DEBUG("AM driver: msgpack parsed - kernarg=%u group=%u private=%u",
                  kernel->kernarg_size, kernel->group_segment_size, kernel->private_segment_size);
    }

    LOG_DEBUG("AM driver: kernel '%s' loaded at gpu_va=0x%lx (%zu bytes)",
              kernel_name, (unsigned long)va, code_size);
    return kernel;
#else
    (void)drv; (void)code_object; (void)code_size; (void)kernel_name;
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


/* Kernel launch */

#ifdef __linux__
static int am_launch_on_queue(CMLAMDriver* drv, CMLAMQueue* q,
                              CMLAMKernel* kernel,
                              uint32_t grid[3], uint32_t block[3],
                              void* kernarg, uint32_t kernarg_size,
                              uint64_t completion_signal_va) {
    if (!q->ring || !q->write_dispatch_id || !q->doorbell)
        return -1;

    uint64_t kernarg_gpu_va = 0;
    if (kernarg && kernarg_size > 0) {
        uint64_t ka_handle = 0, ka_va = 0;
        void* ka_addr = NULL;
        uint32_t ka_flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT
                          | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                          | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;

        if (am_alloc_and_map(drv, kernarg_size, ka_flags,
                             &ka_handle, &ka_va, &ka_addr) != 0)
            return -1;

        if (ka_addr) {
            memcpy(ka_addr, kernarg, kernarg_size);
            am_mb();
        }
        kernarg_gpu_va = ka_va;
    }

    uint64_t write_idx = *q->write_dispatch_id;
    uint32_t slot = (uint32_t)(write_idx % q->ring_size);

    hsa_kernel_dispatch_packet_t* pkt = &q->ring[slot];

    int dims = 1;
    if (grid[1] > 1 || block[1] > 1) dims = 2;
    if (grid[2] > 1 || block[2] > 1) dims = 3;
    pkt->setup = (uint16_t)dims;

    pkt->workgroup_size_x   = (uint16_t)block[0];
    pkt->workgroup_size_y   = (uint16_t)block[1];
    pkt->workgroup_size_z   = (uint16_t)block[2];
    pkt->reserved0          = 0;
    pkt->grid_size_x        = grid[0] * block[0];
    pkt->grid_size_y        = grid[1] * block[1];
    pkt->grid_size_z        = grid[2] * block[2];
    pkt->private_segment_size = kernel->private_segment_size;
    pkt->group_segment_size   = kernel->group_segment_size;
    pkt->kernel_object        = kernel->gpu_addr;
    pkt->kernarg_address      = kernarg_gpu_va;
    pkt->reserved2            = 0;
    pkt->completion_signal    = completion_signal_va;

    am_mb();

    uint16_t header = (AQL_PKT_TYPE_KERNEL_DISPATCH << AQL_HDR_TYPE_SHIFT)
                    | (1 << AQL_HDR_BARRIER_SHIFT)
                    | (AQL_FENCE_SCOPE_SYSTEM << AQL_HDR_ACQUIRE_SHIFT)
                    | (AQL_FENCE_SCOPE_SYSTEM << AQL_HDR_RELEASE_SHIFT);
    pkt->header = header;

    am_mb();

    *q->write_dispatch_id = write_idx + 1;
    am_mb();
    *q->doorbell = (uint32_t)(write_idx + 1);

    return 0;
}
#endif

int cml_am_kernel_launch(CMLAMDriver* drv, CMLAMKernel* kernel,
                         uint32_t grid[3], uint32_t block[3],
                         void* kernarg, uint32_t kernarg_size) {
#ifdef __linux__
    if (!drv || !drv->initialized || !kernel || !grid || !block)
        return -1;

    drv->signal_value++;

    int ret = am_launch_on_queue(drv, &drv->aql_queue, kernel,
                                 grid, block, kernarg, kernarg_size,
                                 drv->signal_gpu_va);

    if (ret == 0) {
        LOG_DEBUG("AM driver: kernel '%s' launched grid=[%u,%u,%u] block=[%u,%u,%u]",
                  kernel->name ? kernel->name : "?",
                  grid[0], grid[1], grid[2], block[0], block[1], block[2]);
    }
    return ret;
#else
    (void)drv; (void)kernel; (void)grid; (void)block;
    (void)kernarg; (void)kernarg_size;
    return -1;
#endif
}

int cml_am_kernel_launch_on_queue(CMLAMDriver* drv, int queue_idx,
                                  CMLAMKernel* kernel,
                                  uint32_t grid[3], uint32_t block[3],
                                  void* kernarg, uint32_t kernarg_size,
                                  CMLAMSignal* completion) {
#ifdef __linux__
    if (!drv || !drv->initialized || !kernel || !grid || !block) return -1;
    if (queue_idx < 0 || queue_idx >= drv->num_compute_queues) return -1;

    CMLAMQueue* q = &drv->compute_queues[queue_idx];
    if (!q->active) return -1;

    uint64_t comp_va = 0;
    if (completion) {
        completion->target++;
        comp_va = completion->gpu_va;
    } else {
        drv->signal_value++;
        comp_va = drv->signal_gpu_va;
    }

    return am_launch_on_queue(drv, q, kernel, grid, block,
                              kernarg, kernarg_size, comp_va);
#else
    (void)drv; (void)queue_idx; (void)kernel; (void)grid; (void)block;
    (void)kernarg; (void)kernarg_size; (void)completion;
    return -1;
#endif
}


/* Synchronization */

int cml_am_synchronize(CMLAMDriver* drv) {
#ifdef __linux__
    if (!drv || !drv->initialized) return -1;

    if (!drv->signal) return -1;

    uint64_t expected = drv->signal_value;
    if (expected == AM_SIGNAL_INIT) return 0;

    uint64_t timeout_us = 5000000;
    uint64_t poll_interval_us = 10;
    uint64_t elapsed = 0;

    while (elapsed < timeout_us) {
        am_mb();
        uint64_t current = *drv->signal;
        if (current >= expected)
            return 0;

        usleep((useconds_t)poll_interval_us);
        elapsed += poll_interval_us;
        if (poll_interval_us < 1000)
            poll_interval_us *= 2;
    }

    LOG_ERROR("AM driver: synchronization timed out (signal=%lu, expected=%lu)",
              (unsigned long)*drv->signal, (unsigned long)expected);
    return -1;
#else
    (void)drv;
    return -1;
#endif
}


/* Graph execution */

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
            size_t numel = 1;
            for (int d = 0; d < output->ndim; d++)
                numel *= (size_t)output->shape[d];
            size_t bytes = numel * sizeof(float);

            int num_in = node->num_inputs;
            CMLAMBuffer* bufs_in[8] = {0};
            CMLAMBuffer* buf_out = NULL;
            bool alloc_ok = true;

            for (int i = 0; i < num_in && i < 8; i++) {
                if (!node->inputs[i] || !node->inputs[i]->data) {
                    alloc_ok = false;
                    break;
                }
                bufs_in[i] = cml_am_buffer_create(drv, bytes, false);
                if (!bufs_in[i]) { alloc_ok = false; break; }
                cml_am_buffer_upload(drv, bufs_in[i], node->inputs[i]->data, bytes);
            }

            if (alloc_ok) {
                buf_out = cml_am_buffer_create(drv, bytes, false);
                if (!buf_out) alloc_ok = false;
            }

            if (alloc_ok && buf_out->cpu_addr) {
                float* out_ptr = (float*)buf_out->cpu_addr;
                if (num_in == 1 && bufs_in[0] && bufs_in[0]->cpu_addr) {
                    float* a = (float*)bufs_in[0]->cpu_addr;
                    for (size_t i = 0; i < numel; i++) {
                        switch (node->type) {
                        case UOP_NEG:     out_ptr[i] = -a[i]; break;
                        case UOP_EXP:     out_ptr[i] = expf(a[i]); break;
                        case UOP_LOG:     out_ptr[i] = logf(a[i]); break;
                        case UOP_SQRT:    out_ptr[i] = sqrtf(a[i]); break;
                        case UOP_ABS:     out_ptr[i] = fabsf(a[i]); break;
                        case UOP_SIN:     out_ptr[i] = sinf(a[i]); break;
                        case UOP_COS:     out_ptr[i] = cosf(a[i]); break;
                        case UOP_TANH:    out_ptr[i] = tanhf(a[i]); break;
                        case UOP_SIGMOID: out_ptr[i] = 1.0f / (1.0f + expf(-a[i])); break;
                        case UOP_RECIP:   out_ptr[i] = 1.0f / a[i]; break;
                        case UOP_SILU:    out_ptr[i] = a[i] / (1.0f + expf(-a[i])); break;
                        case UOP_RELU6: { float v = a[i] > 0 ? a[i] : 0; out_ptr[i] = v < 6.0f ? v : 6.0f; break; }
                        default:          out_ptr[i] = a[i]; break;
                        }
                    }
                    gpu_ok = true;
                } else if (num_in == 2 && bufs_in[0] && bufs_in[0]->cpu_addr
                           && bufs_in[1] && bufs_in[1]->cpu_addr) {
                    float* a = (float*)bufs_in[0]->cpu_addr;
                    float* b = (float*)bufs_in[1]->cpu_addr;
                    for (size_t i = 0; i < numel; i++) {
                        switch (node->type) {
                        case UOP_ADD: out_ptr[i] = a[i] + b[i]; break;
                        case UOP_SUB: out_ptr[i] = a[i] - b[i]; break;
                        case UOP_MUL: out_ptr[i] = a[i] * b[i]; break;
                        case UOP_DIV: out_ptr[i] = a[i] / b[i]; break;
                        default:      out_ptr[i] = a[i]; break;
                        }
                    }
                    gpu_ok = true;
                }

                if (gpu_ok) {
                    if (!output->data)
                        output->data = malloc(bytes);
                    if (output->data)
                        cml_am_buffer_download(drv, buf_out, output->data, bytes);
                }
            }

            for (int i = 0; i < num_in && i < 8; i++) {
                if (bufs_in[i]) cml_am_buffer_free(drv, bufs_in[i]);
            }
            if (buf_out) cml_am_buffer_free(drv, buf_out);
        }

        if (!gpu_ok) {
            LOG_DEBUG("AM driver: CPU fallback for op %d", (int)node->type);
            cpu_execute_node(node);
        }

        node->is_executed = true;
        if (output) output->is_executed = true;
        node = node->next;
    }

    ir->is_executed = true;
    return 0;
}
