#ifdef CML_NV_MOCK_GPU

#include "ops/ir/gpu/nv_mock.h"
#include "ops/ir/gpu/nv_driver.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#define MOCK_FD_CTL  100
#define MOCK_FD_DEV  101
#define MOCK_FD_UVM  102

#define MOCK_INITIAL_ALLOC_CAP 64

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

typedef struct {
    uint32_t type;
    uint32_t data;
} NV_GPU_INFO_ENTRY;

typedef struct {
    uint32_t gpuInfoListSize;
    NV_GPU_INFO_ENTRY *gpuInfoList;
} NV2080_CTRL_GPU_GET_INFO_PARAMS_M;

typedef struct {
    uint32_t gpuNameStringFlags;
    char     gpuNameString[256];
} NV2080_CTRL_GPU_GET_NAME_STRING_PARAMS_M;

#define NV2080_GPU_INFO_INDEX_GPU_ARCH 52

#define MOCK_NV_IOCTL_RM_ALLOC   _IOWR('F', NV_ESC_RM_ALLOC,   NV_RM_ALLOC_PARAMS)
#define MOCK_NV_IOCTL_RM_CONTROL _IOWR('F', NV_ESC_RM_CONTROL,  NV_RM_CONTROL_PARAMS)
#define MOCK_NV_IOCTL_RM_FREE    _IOWR('F', NV_ESC_RM_FREE,     NV_RM_FREE_PARAMS)

static CMLNVMockGPU g_mock;
static bool g_mock_active = false;

static void mock_track_alloc(void *ptr) {
    if (g_mock.num_allocs >= g_mock.alloc_capacity) {
        int new_cap = g_mock.alloc_capacity * 2;
        void **new_table = (void **)realloc(g_mock.alloc_table, (size_t)new_cap * sizeof(void *));
        if (!new_table) return;
        g_mock.alloc_table = new_table;
        g_mock.alloc_capacity = new_cap;
    }
    g_mock.alloc_table[g_mock.num_allocs++] = ptr;
}

static bool mock_untrack_alloc(void *ptr) {
    for (int i = 0; i < g_mock.num_allocs; i++) {
        if (g_mock.alloc_table[i] == ptr) {
            g_mock.alloc_table[i] = g_mock.alloc_table[--g_mock.num_allocs];
            return true;
        }
    }
    return false;
}

void cml_nv_mock_init(CMLNVMockGPU *config) {
    memset(&g_mock, 0, sizeof(g_mock));

    if (config) {
        g_mock.gpu_arch         = config->gpu_arch;
        g_mock.compute_cap_major = config->compute_cap_major;
        g_mock.compute_cap_minor = config->compute_cap_minor;
        g_mock.vram_size        = config->vram_size;
        memcpy(g_mock.name, config->name, sizeof(g_mock.name));
        g_mock.auto_complete    = config->auto_complete;
    }

    if (g_mock.gpu_arch == 0)
        g_mock.gpu_arch = 0x190;
    if (g_mock.compute_cap_major == 0) {
        g_mock.compute_cap_major = 7;
        g_mock.compute_cap_minor = 5;
    }
    if (g_mock.vram_size == 0)
        g_mock.vram_size = (size_t)8 * 1024 * 1024 * 1024;
    if (g_mock.name[0] == '\0')
        snprintf(g_mock.name, sizeof(g_mock.name), "CML Mock GPU (Turing)");

    g_mock.next_handle = 1;
    g_mock.alloc_capacity = MOCK_INITIAL_ALLOC_CAP;
    g_mock.alloc_table = (void **)calloc((size_t)g_mock.alloc_capacity, sizeof(void *));
    g_mock.num_allocs = 0;
    g_mock.last_semaphore = NULL;
    g_mock.last_semaphore_value = 0;

    g_mock_active = true;
}

void cml_nv_mock_shutdown(void) {
    if (!g_mock_active) return;

    for (int i = 0; i < g_mock.num_allocs; i++)
        free(g_mock.alloc_table[i]);

    free(g_mock.alloc_table);
    memset(&g_mock, 0, sizeof(g_mock));
    g_mock_active = false;
}

CMLNVMockGPU* cml_nv_mock_get(void) {
    return g_mock_active ? &g_mock : NULL;
}

void cml_nv_mock_complete_kernel(void) {
    if (!g_mock_active) return;
    if (g_mock.last_semaphore) {
        *g_mock.last_semaphore = g_mock.last_semaphore_value;
        __sync_synchronize();
    }
}

static bool is_mock_fd(int fd) {
    return fd == MOCK_FD_CTL || fd == MOCK_FD_DEV || fd == MOCK_FD_UVM;
}

int cml_nv_mock_open(const char *path, int flags, ...) {
    if (!g_mock_active || !path)
        goto real_open;

    if (strstr(path, "nvidiactl"))
        return MOCK_FD_CTL;
    if (strstr(path, "nvidia-uvm"))
        return MOCK_FD_UVM;
    if (strstr(path, "nvidia0"))
        return MOCK_FD_DEV;

real_open:;
    va_list ap;
    va_start(ap, flags);
    int mode = 0;
    if (flags & O_CREAT)
        mode = va_arg(ap, int);
    va_end(ap);
    return open(path, flags, mode);
}

int cml_nv_mock_close(int fd) {
    if (g_mock_active && is_mock_fd(fd))
        return 0;
    return close(fd);
}

static uint32_t mock_handle_for_class(uint32_t nv_class) {
    switch (nv_class) {
    case NV01_ROOT_CLIENT:        return 1;
    case NV01_DEVICE_0:           return 2;
    case NV20_SUBDEVICE_0:        return 3;
    case FERMI_VASPACE_A:         return 4;
    case KEPLER_CHANNEL_GROUP_A:  return 5;
    case KEPLER_CHANNEL_GPFIFO_A:
    case TURING_CHANNEL_GPFIFO_A:
    case AMPERE_CHANNEL_GPFIFO_A:
    case HOPPER_CHANNEL_GPFIFO_A: return 6;
    case TURING_COMPUTE_A:
    case AMPERE_COMPUTE_A:
    case HOPPER_COMPUTE_A:        return 7;
    case TURING_DMA_COPY_A:
    case AMPERE_DMA_COPY_A:
    case HOPPER_DMA_COPY_A:       return 8;
    default:                      return g_mock.next_handle++;
    }
}

static int mock_ioctl_rm_alloc(NV_RM_ALLOC_PARAMS *p) {
    p->hObjectNew = mock_handle_for_class(p->hClass);
    p->status = 0;
    return 0;
}

static int mock_ioctl_rm_control(NV_RM_CONTROL_PARAMS *p) {
    p->status = 0;

    switch (p->cmd) {
    case NV2080_CTRL_CMD_GPU_GET_INFO: {
        if (!p->params) break;
        NV2080_CTRL_GPU_GET_INFO_PARAMS_M *info = (NV2080_CTRL_GPU_GET_INFO_PARAMS_M *)p->params;
        for (uint32_t i = 0; i < info->gpuInfoListSize; i++) {
            if (info->gpuInfoList[i].type == NV2080_GPU_INFO_INDEX_GPU_ARCH)
                info->gpuInfoList[i].data = g_mock.gpu_arch;
        }
        break;
    }
    case NV2080_CTRL_CMD_GPU_GET_NAME_STRING: {
        if (!p->params) break;
        NV2080_CTRL_GPU_GET_NAME_STRING_PARAMS_M *name =
            (NV2080_CTRL_GPU_GET_NAME_STRING_PARAMS_M *)p->params;
        strncpy(name->gpuNameString, g_mock.name, sizeof(name->gpuNameString) - 1);
        name->gpuNameString[sizeof(name->gpuNameString) - 1] = '\0';
        break;
    }
    default:
        break;
    }

    return 0;
}

static int mock_ioctl_rm_free(NV_RM_FREE_PARAMS *p) {
    p->status = 0;
    return 0;
}

int cml_nv_mock_ioctl(int fd, unsigned long request, void *arg) {
    if (!g_mock_active || !is_mock_fd(fd))
        return ioctl(fd, request, arg);

    if (!arg) return 0;

    if (request == MOCK_NV_IOCTL_RM_ALLOC)
        return mock_ioctl_rm_alloc((NV_RM_ALLOC_PARAMS *)arg);
    if (request == MOCK_NV_IOCTL_RM_CONTROL)
        return mock_ioctl_rm_control((NV_RM_CONTROL_PARAMS *)arg);
    if (request == MOCK_NV_IOCTL_RM_FREE)
        return mock_ioctl_rm_free((NV_RM_FREE_PARAMS *)arg);

    return 0;
}

void* cml_nv_mock_mmap(void *addr, size_t length, int prot, int flags,
                        int fd, off_t offset) {
    (void)addr;
    (void)prot;
    (void)offset;

    if (!g_mock_active || !is_mock_fd(fd)) {
        return mmap(addr, length, prot, flags, fd, offset);
    }

    size_t aligned = (length + 4095) & ~(size_t)4095;
    void *ptr = aligned_alloc(4096, aligned);
    if (!ptr) return MAP_FAILED;

    memset(ptr, 0, aligned);
    mock_track_alloc(ptr);
    return ptr;
}

int cml_nv_mock_munmap(void *addr, size_t length) {
    if (!g_mock_active)
        return munmap(addr, length);

    if (mock_untrack_alloc(addr)) {
        free(addr);
        return 0;
    }

    return munmap(addr, length);
}

#endif /* CML_NV_MOCK_GPU */
