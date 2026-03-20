#define _GNU_SOURCE

#include "ops/ir/gpu/am_mock.h"

#ifdef CML_AM_MOCK_GPU

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <dirent.h>

#define MOCK_FD_KFD  200
#define MOCK_FD_DRM  201

#define MOCK_ALLOC_INIT_CAP 64

static CMLAMMockGPU g_mock;
static bool g_mock_active = false;

static void mock_create_topology(void);
static void mock_remove_topology(void);

void cml_am_mock_init(CMLAMMockGPU* config) {
    if (g_mock_active)
        cml_am_mock_shutdown();

    memset(&g_mock, 0, sizeof(g_mock));

    if (config) {
        g_mock = *config;
    } else {
        g_mock.gpu_id             = 12345;
        snprintf(g_mock.name, sizeof(g_mock.name), "Mock RDNA3");
        snprintf(g_mock.gfx_version, sizeof(g_mock.gfx_version), "gfx1100");
        g_mock.cu_count           = 48;
        g_mock.vram_size          = (size_t)8 * 1024 * 1024 * 1024;
        g_mock.lds_size_per_cu    = 65536;
        g_mock.sdma_count         = 2;
        g_mock.simd_per_cu        = 2;
        g_mock.max_waves_per_simd = 16;
        g_mock.auto_complete      = true;
    }

    g_mock.next_handle  = 1000;
    g_mock.next_queue_id = 1;

    g_mock.alloc_capacity = MOCK_ALLOC_INIT_CAP;
    g_mock.alloc_table = (void**)calloc((size_t)g_mock.alloc_capacity, sizeof(void*));
    g_mock.num_allocs = 0;

    mock_create_topology();
    g_mock_active = true;
}

void cml_am_mock_shutdown(void) {
    if (!g_mock_active) return;

    for (int i = 0; i < g_mock.num_allocs; i++) {
        free(g_mock.alloc_table[i]);
    }
    free(g_mock.alloc_table);
    g_mock.alloc_table = NULL;
    g_mock.num_allocs = 0;
    g_mock.alloc_capacity = 0;

    mock_remove_topology();

    g_mock_active = false;
}

CMLAMMockGPU* cml_am_mock_get(void) {
    return g_mock_active ? &g_mock : NULL;
}

static void mock_track_alloc(void* ptr) {
    if (!ptr) return;

    if (g_mock.num_allocs >= g_mock.alloc_capacity) {
        int new_cap = g_mock.alloc_capacity * 2;
        void** tmp = (void**)realloc(g_mock.alloc_table,
                                     (size_t)new_cap * sizeof(void*));
        if (!tmp) return;
        g_mock.alloc_table = tmp;
        g_mock.alloc_capacity = new_cap;
    }

    g_mock.alloc_table[g_mock.num_allocs++] = ptr;
}

static bool mock_untrack_alloc(void* ptr) {
    for (int i = 0; i < g_mock.num_allocs; i++) {
        if (g_mock.alloc_table[i] == ptr) {
            g_mock.alloc_table[i] = g_mock.alloc_table[--g_mock.num_allocs];
            return true;
        }
    }
    return false;
}


/* Fake sysfs topology */

static void write_file(const char* path, const char* content) {
    FILE* f = fopen(path, "w");
    if (f) {
        fputs(content, f);
        fclose(f);
    }
}

static int mkpath(const char* path, mode_t mode) {
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, mode);
            *p = '/';
        }
    }
    return mkdir(tmp, mode);
}

static void mock_create_topology(void) {
    char template[] = "/tmp/cml_am_mock_XXXXXX";
    char* dir = mkdtemp(template);
    if (!dir) return;
    snprintf(g_mock.topology_dir, sizeof(g_mock.topology_dir), "%s", dir);

    char path[512];

    /* CPU node (node 0) */
    snprintf(path, sizeof(path), "%s/topology/nodes/0", dir);
    mkpath(path, 0755);

    snprintf(path, sizeof(path), "%s/topology/nodes/0/properties", dir);
    write_file(path, "cpu_cores_count 8\n");

    snprintf(path, sizeof(path), "%s/topology/nodes/0/gpu_id", dir);
    write_file(path, "0\n");

    /* GPU node (node 1) */
    snprintf(path, sizeof(path), "%s/topology/nodes/1", dir);
    mkpath(path, 0755);

    char props[2048];
    int simd_count = g_mock.cu_count * (int)g_mock.simd_per_cu;
    uint32_t gfx_target = 0;
    if (strcmp(g_mock.gfx_version, "gfx1100") == 0) gfx_target = 110000;
    else if (strcmp(g_mock.gfx_version, "gfx1030") == 0) gfx_target = 100300;
    else if (strcmp(g_mock.gfx_version, "gfx942") == 0) gfx_target = 90402;
    else gfx_target = 110000;

    int array_count = g_mock.cu_count > 0 ? 6 : 0;
    int cu_per_array = g_mock.cu_count / (array_count > 0 ? array_count : 1);

    snprintf(props, sizeof(props),
        "cpu_cores_count 0\n"
        "simd_count %d\n"
        "simd_per_cu %u\n"
        "max_waves_per_simd %u\n"
        "gfx_target_version %u\n"
        "gpu_id %u\n"
        "array_count %d\n"
        "cu_per_simd_array %d\n"
        "local_mem_size %u\n"
        "fw_version 123\n"
        "max_engine_clk_fcompute 2500\n"
        "vendor_id 4098\n"
        "device_id 29772\n"
        "location_id 256\n"
        "domain 0\n"
        "sdma_fw_version 2\n"
        "lds_size_in_kb %u\n",
        simd_count,
        g_mock.simd_per_cu,
        g_mock.max_waves_per_simd,
        gfx_target,
        g_mock.gpu_id,
        array_count,
        cu_per_array,
        g_mock.lds_size_per_cu,
        g_mock.lds_size_per_cu / 1024);

    snprintf(path, sizeof(path), "%s/topology/nodes/1/properties", dir);
    write_file(path, props);

    snprintf(path, sizeof(path), "%s/topology/nodes/1/gpu_id", dir);
    char gpu_id_str[32];
    snprintf(gpu_id_str, sizeof(gpu_id_str), "%u\n", g_mock.gpu_id);
    write_file(path, gpu_id_str);
}

static void mock_remove_topology(void) {
    if (g_mock.topology_dir[0] == '\0') return;

    char cmd[512];
    snprintf(cmd, sizeof(cmd), "rm -rf %s", g_mock.topology_dir);
    (void)system(cmd);
    g_mock.topology_dir[0] = '\0';
}


/* Mock syscalls */

int cml_am_mock_open(const char* path, int flags, ...) {
    if (!g_mock_active) {
        va_list ap;
        va_start(ap, flags);
        int mode = va_arg(ap, int);
        va_end(ap);
        return open(path, flags, mode);
    }

    if (strcmp(path, "/dev/kfd") == 0) return MOCK_FD_KFD;
    if (strcmp(path, "/dev/dri/renderD128") == 0) return MOCK_FD_DRM;

    va_list ap;
    va_start(ap, flags);
    int mode = va_arg(ap, int);
    va_end(ap);
    return open(path, flags, mode);
}

int cml_am_mock_close(int fd) {
    if (!g_mock_active) return close(fd);
    if (fd == MOCK_FD_KFD || fd == MOCK_FD_DRM) return 0;
    return close(fd);
}

/* KFD ioctl structures -- must match am_driver.c definitions */
#define KFD_IOC_MAGIC 'K'
#define KFD_IOWR(nr, type) _IOWR(KFD_IOC_MAGIC, nr, type)
#define KFD_IOW(nr, type)  _IOW(KFD_IOC_MAGIC, nr, type)
#define KFD_IOR(nr, type)  _IOR(KFD_IOC_MAGIC, nr, type)

struct mock_kfd_get_version {
    uint32_t major_version;
    uint32_t minor_version;
};

struct mock_kfd_acquire_vm {
    uint32_t drm_fd;
    uint32_t gpu_id;
};

struct mock_kfd_create_queue {
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

struct mock_kfd_destroy_queue {
    uint32_t queue_id;
    uint32_t pad;
};

struct mock_kfd_alloc_memory {
    uint64_t va_addr;
    uint64_t size;
    uint64_t handle;
    uint32_t gpu_id;
    uint32_t flags;
    uint64_t mmap_offset;
};

struct mock_kfd_free_memory {
    uint64_t handle;
};

struct mock_kfd_map_memory {
    uint64_t handle;
    uint64_t device_ids_array_ptr;
    uint32_t n_devices;
    uint32_t n_success;
};

struct mock_kfd_unmap_memory {
    uint64_t handle;
    uint64_t device_ids_array_ptr;
    uint32_t n_devices;
    uint32_t n_success;
};

#define MOCK_IOC_GET_VERSION       KFD_IOR(0x01, struct mock_kfd_get_version)
#define MOCK_IOC_CREATE_QUEUE      KFD_IOWR(0x02, struct mock_kfd_create_queue)
#define MOCK_IOC_DESTROY_QUEUE     KFD_IOWR(0x03, struct mock_kfd_destroy_queue)
#define MOCK_IOC_ACQUIRE_VM        KFD_IOW(0x07, struct mock_kfd_acquire_vm)
#define MOCK_IOC_ALLOC_MEMORY      KFD_IOWR(0x18, struct mock_kfd_alloc_memory)
#define MOCK_IOC_FREE_MEMORY       KFD_IOW(0x19, struct mock_kfd_free_memory)
#define MOCK_IOC_MAP_MEMORY        KFD_IOWR(0x1A, struct mock_kfd_map_memory)
#define MOCK_IOC_UNMAP_MEMORY      KFD_IOWR(0x1B, struct mock_kfd_unmap_memory)

int cml_am_mock_ioctl(int fd, unsigned long request, void* arg) {
    if (!g_mock_active || (fd != MOCK_FD_KFD && fd != MOCK_FD_DRM))
        return ioctl(fd, request, arg);

    if (request == MOCK_IOC_GET_VERSION) {
        struct mock_kfd_get_version* v = (struct mock_kfd_get_version*)arg;
        v->major_version = 1;
        v->minor_version = 14;
        return 0;
    }

    if (request == MOCK_IOC_ACQUIRE_VM) {
        return 0;
    }

    if (request == MOCK_IOC_CREATE_QUEUE) {
        struct mock_kfd_create_queue* cq = (struct mock_kfd_create_queue*)arg;
        cq->queue_id = g_mock.next_queue_id++;
        cq->doorbell_offset = (uint64_t)cq->queue_id * 4096;
        return 0;
    }

    if (request == MOCK_IOC_DESTROY_QUEUE) {
        return 0;
    }

    if (request == MOCK_IOC_ALLOC_MEMORY) {
        struct mock_kfd_alloc_memory* a = (struct mock_kfd_alloc_memory*)arg;
        a->handle = g_mock.next_handle++;
        a->mmap_offset = a->handle * 4096;
        return 0;
    }

    if (request == MOCK_IOC_FREE_MEMORY) {
        return 0;
    }

    if (request == MOCK_IOC_MAP_MEMORY) {
        struct mock_kfd_map_memory* m = (struct mock_kfd_map_memory*)arg;
        m->n_success = m->n_devices;
        return 0;
    }

    if (request == MOCK_IOC_UNMAP_MEMORY) {
        return 0;
    }

    errno = EINVAL;
    return -1;
}

void* cml_am_mock_mmap(void* addr, size_t length, int prot, int flags,
                        int fd, off_t offset) {
    (void)addr; (void)prot; (void)flags; (void)offset;

    if (!g_mock_active || (fd != MOCK_FD_KFD && fd != MOCK_FD_DRM))
        return mmap(addr, length, prot, flags, fd, offset);

    size_t aligned = (length + 4095) & ~(size_t)4095;
    void* ptr = NULL;
    if (posix_memalign(&ptr, 4096, aligned) != 0)
        return MAP_FAILED;
    memset(ptr, 0, aligned);
    mock_track_alloc(ptr);
    return ptr;
}

int cml_am_mock_munmap(void* addr, size_t length) {
    (void)length;
    if (!g_mock_active) return munmap(addr, length);

    if (mock_untrack_alloc(addr)) {
        free(addr);
        return 0;
    }

    return munmap(addr, length);
}

FILE* cml_am_mock_fopen(const char* path, const char* mode) {
    if (!g_mock_active) return fopen(path, mode);

    const char* sysfs_prefix = "/sys/devices/virtual/kfd/kfd/topology/nodes";
    size_t prefix_len = strlen(sysfs_prefix);

    if (strncmp(path, sysfs_prefix, prefix_len) == 0) {
        char mock_path[512];
        snprintf(mock_path, sizeof(mock_path), "%s/topology/nodes%s",
                 g_mock.topology_dir, path + prefix_len);
        return fopen(mock_path, mode);
    }

    return fopen(path, mode);
}


/* AQL auto-completion */

void cml_am_mock_complete_dispatch(void) {
    if (!g_mock_active) return;
    g_mock.dispatches_seen++;
}

/* Called from the mock when auto_complete is enabled.
 * Scans known signal addresses and writes completion values. */

int cml_am_mock_access(const char* path, int mode) {
    if (!g_mock_active) return access(path, mode);

    if (strcmp(path, "/dev/kfd") == 0) return 0;
    if (strcmp(path, "/dev/dri/renderD128") == 0) return 0;

    return access(path, mode);
}

DIR* cml_am_mock_opendir(const char* path) {
    if (!g_mock_active) return opendir(path);

    const char* sysfs_prefix = "/sys/devices/virtual/kfd/kfd/topology/nodes";
    if (strcmp(path, sysfs_prefix) == 0) {
        char mock_path[512];
        snprintf(mock_path, sizeof(mock_path), "%s/topology/nodes",
                 g_mock.topology_dir);
        return opendir(mock_path);
    }

    return opendir(path);
}

#endif /* CML_AM_MOCK_GPU */
