/**
 * @file nv_driver.c
 * @brief NVIDIA userspace driver -- direct ioctl interface to kernel module
 *
 * Opens /dev/nvidiactl and /dev/nvidia0 via open(), uses ioctl() for RM
 * operations (NV_ESC_RM_ALLOC, NV_ESC_RM_CONTROL, NV_ESC_RM_FREE).
 * Dispatches kernels through a GPFIFO ring buffer and synchronises via
 * semaphore polling.  PTX is compiled to CUBIN by invoking the `ptxas`
 * subprocess through popen/pclose.
 *
 * The actual RM ioctl structures are simplified stubs that demonstrate the
 * architecture; the code returns gracefully when hardware is not present.
 */

#include "ops/ir/gpu/nv_driver.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#ifdef __linux__
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#endif

/* ── Local constants ────────────────────────────────────────────────────── */

#define NV_GPFIFO_DEFAULT_ENTRIES  128
#define NV_GPFIFO_ENTRY_BYTES       8   /* 64-bit per GPFIFO entry */

/* ══════════════════════════════════════════════════════════════════════════
 * RM ioctl parameter structures (simplified for stub implementation)
 *
 * Real RM ioctls carry much larger payloads; these capture the essential
 * fields needed to demonstrate the alloc -> control -> free lifecycle.
 * ══════════════════════════════════════════════════════════════════════════ */

#ifdef __linux__

/* File-local handle counter for RM allocations */
static uint32_t g_nv_next_handle = 0x10000;

typedef struct {
    uint32_t hRoot;         /* client handle   */
    uint32_t hObjectParent; /* parent handle   */
    uint32_t hObjectNew;    /* output handle   */
    uint32_t hClass;        /* NV class id     */
    void*    pAllocParms;   /* class-specific  */
    uint32_t status;        /* out: RM status  */
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

/* RM NV class IDs (subset) */
#define NV01_ROOT_CLIENT           0x00000041
#define NV01_DEVICE_0              0x00000080
#define NV20_SUBDEVICE_0           0x00002080
#define KEPLER_CHANNEL_GPFIFO_A    0x0000A06F
#define TURING_CHANNEL_GPFIFO_A    0x0000C46F
#define NV50_MEMORY_VIRTUAL        0x000050A0

/* ioctl request codes -- _IOWR('F', escape_nr, param_struct) */
#define NV_IOCTL_RM_ALLOC    _IOWR('F', NV_ESC_RM_ALLOC,   NV_RM_ALLOC_PARAMS)
#define NV_IOCTL_RM_CONTROL  _IOWR('F', NV_ESC_RM_CONTROL,  NV_RM_CONTROL_PARAMS)
#define NV_IOCTL_RM_FREE     _IOWR('F', NV_ESC_RM_FREE,     NV_RM_FREE_PARAMS)

/* ── Helper: perform an RM alloc ioctl ──────────────────────────────────── */

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

/* ── Helper: perform an RM control ioctl ────────────────────────────────── */

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

/* ── Helper: perform an RM free ioctl ───────────────────────────────────── */

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

/* ── Helper: monotonic clock (ms) ───────────────────────────────────────── */

static uint64_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

/* Suppress unused-function warning for nv_rm_control in stub builds */
static int nv_rm_control_unused_guard(void) __attribute__((unused));
static int nv_rm_control_unused_guard(void) {
    return nv_rm_control(0, 0, 0, 0, NULL, 0);
}

#endif /* __linux__ */

/* ══════════════════════════════════════════════════════════════════════════
 * Availability check
 * ══════════════════════════════════════════════════════════════════════════ */

bool cml_nv_driver_available(void) {
#ifdef __linux__
    /* Quick stat check -- does not open the device */
    struct stat st;
    if (stat("/dev/nvidia0", &st) != 0) {
        return false;
    }
    /* Verify we can open it for reading */
    int fd = open("/dev/nvidia0", O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    close(fd);
    return true;
#else
    return false;  /* Only Linux supports NVIDIA RM ioctls */
#endif
}

/* ══════════════════════════════════════════════════════════════════════════
 * Driver lifecycle
 * ══════════════════════════════════════════════════════════════════════════ */

CMLNVDriver* cml_nv_driver_create(void) {
    CMLNVDriver* drv = (CMLNVDriver*)calloc(1, sizeof(CMLNVDriver));
    if (!drv) {
        LOG_ERROR("NV driver: failed to allocate context");
        return NULL;
    }
    drv->fd_ctl = -1;
    drv->fd_dev = -1;
    drv->va_start   = 0x100000000ULL;  /* 4 GiB, typical GPU VA start */
    drv->va_current = drv->va_start;
    drv->va_end     = 0x800000000ULL;  /* 32 GiB VA space */
    return drv;
}

int cml_nv_driver_init(CMLNVDriver* drv) {
    if (!drv) return -1;
    if (drv->initialized) {
        LOG_DEBUG("NV driver: already initialized");
        return 0;
    }

#ifndef __linux__
    LOG_ERROR("NV driver: only supported on Linux");
    return -1;
#else
    /* ── Open device nodes ── */
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

    /* ── Allocate RM client ── */
    drv->client_handle = g_nv_next_handle++;
    if (nv_rm_alloc(drv->fd_ctl, 0, 0,
                    &drv->client_handle, NV01_ROOT_CLIENT, NULL) != 0) {
        LOG_ERROR("NV driver: failed to create RM client");
        goto fail;
    }
    LOG_DEBUG("NV driver: RM client handle = 0x%X", drv->client_handle);

    /* ── Allocate RM device (GPU 0) ── */
    drv->device_handle = g_nv_next_handle++;
    if (nv_rm_alloc(drv->fd_ctl, drv->client_handle, drv->client_handle,
                    &drv->device_handle, NV01_DEVICE_0, NULL) != 0) {
        LOG_ERROR("NV driver: failed to create RM device");
        goto fail;
    }
    LOG_DEBUG("NV driver: RM device handle = 0x%X", drv->device_handle);

    /* ── Allocate RM subdevice ── */
    drv->subdevice_handle = g_nv_next_handle++;
    if (nv_rm_alloc(drv->fd_ctl, drv->client_handle, drv->device_handle,
                    &drv->subdevice_handle, NV20_SUBDEVICE_0, NULL) != 0) {
        LOG_ERROR("NV driver: failed to create RM subdevice");
        goto fail;
    }
    LOG_DEBUG("NV driver: RM subdevice handle = 0x%X", drv->subdevice_handle);

    /* ── Setup GPFIFO channel ──
     * In a real driver we would:
     * 1. Allocate a channel group
     * 2. Allocate a GPFIFO channel (KEPLER_CHANNEL_GPFIFO_A or newer)
     * 3. mmap the GPFIFO entries and doorbell register
     * 4. Map the channel's notifier and semaphore pages
     *
     * Here we attempt the RM alloc and fall back gracefully.
     */
    drv->gpfifo.num_entries = NV_GPFIFO_DEFAULT_ENTRIES;
    drv->channel_group_handle = g_nv_next_handle++;
    drv->gpfifo.handle = g_nv_next_handle++;

    if (nv_rm_alloc(drv->fd_ctl, drv->client_handle, drv->device_handle,
                    &drv->gpfifo.handle, TURING_CHANNEL_GPFIFO_A, NULL) != 0) {
        LOG_WARNING("NV driver: GPFIFO channel alloc failed (expected without real GPU)");
        /* Continue -- we still mark as initialized for the ptxas path */
    } else {
        /* mmap the GPFIFO entry buffer from the channel */
        size_t gpfifo_size = (size_t)drv->gpfifo.num_entries * NV_GPFIFO_ENTRY_BYTES;
        drv->gpfifo.entries = (uint64_t*)mmap(NULL, gpfifo_size,
                                              PROT_READ | PROT_WRITE,
                                              MAP_SHARED, drv->fd_dev, 0);
        if (drv->gpfifo.entries == MAP_FAILED) {
            LOG_WARNING("NV driver: GPFIFO mmap failed: %s", strerror(errno));
            drv->gpfifo.entries = NULL;
        } else {
            drv->gpfifo.gpu_va = drv->va_current;
            drv->va_current += gpfifo_size;
            LOG_DEBUG("NV driver: GPFIFO mapped at %p, GPU VA 0x%llX",
                      (void*)drv->gpfifo.entries,
                      (unsigned long long)drv->gpfifo.gpu_va);
        }
    }

    /* ── Allocate semaphore for synchronization ── */
    size_t sem_size = 4096;  /* one page */
    drv->semaphore = (volatile uint32_t*)mmap(NULL, sem_size,
                                              PROT_READ | PROT_WRITE,
                                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (drv->semaphore == MAP_FAILED) {
        LOG_WARNING("NV driver: semaphore mmap failed");
        drv->semaphore = NULL;
    } else {
        *drv->semaphore = 0;
        drv->semaphore_gpu_va = drv->va_current;
        drv->va_current += sem_size;
        drv->semaphore_value = 0;
    }

    drv->initialized = true;
    LOG_INFO("NV driver: initialized (fd_ctl=%d, fd_dev=%d)", drv->fd_ctl, drv->fd_dev);
    return 0;

fail:
    if (drv->fd_dev >= 0) { close(drv->fd_dev); drv->fd_dev = -1; }
    if (drv->fd_ctl >= 0) { close(drv->fd_ctl); drv->fd_ctl = -1; }
    return -1;
#endif /* __linux__ */
}

void cml_nv_driver_free(CMLNVDriver* drv) {
    if (!drv) return;

#ifdef __linux__
    if (drv->initialized) {
        /* Unmap semaphore */
        if (drv->semaphore && drv->semaphore != MAP_FAILED) {
            munmap((void*)drv->semaphore, 4096);
        }

        /* Unmap GPFIFO */
        if (drv->gpfifo.entries && drv->gpfifo.entries != MAP_FAILED) {
            size_t gpfifo_size = (size_t)drv->gpfifo.num_entries * NV_GPFIFO_ENTRY_BYTES;
            munmap((void*)drv->gpfifo.entries, gpfifo_size);
        }

        /* Free RM objects in reverse order */
        if (drv->gpfifo.handle) {
            nv_rm_free(drv->fd_ctl, drv->client_handle,
                       drv->device_handle, drv->gpfifo.handle);
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

        /* Close device nodes */
        if (drv->fd_dev >= 0) close(drv->fd_dev);
        if (drv->fd_ctl >= 0) close(drv->fd_ctl);
    }
#endif

    free(drv);
}

/* ══════════════════════════════════════════════════════════════════════════
 * Buffer management
 * ══════════════════════════════════════════════════════════════════════════ */

CMLNVBuffer* cml_nv_buffer_create(CMLNVDriver* drv, size_t size, bool host_visible) {
    if (!drv || !drv->initialized || size == 0) return NULL;

    CMLNVBuffer* buf = (CMLNVBuffer*)calloc(1, sizeof(CMLNVBuffer));
    if (!buf) {
        LOG_ERROR("NV driver: failed to allocate buffer descriptor");
        return NULL;
    }

    buf->size         = size;
    buf->host_visible = host_visible;

#ifdef __linux__
    buf->handle = g_nv_next_handle++;

    /* Attempt RM alloc for the buffer.
     * In a real implementation this would use NV50_MEMORY_VIRTUAL + bind,
     * or NV01_MEMORY_SYSTEM / NV01_MEMORY_LOCAL classes.  Here we attempt
     * the alloc and fall back to a host-side mmap for testing purposes.
     */
    if (nv_rm_alloc(drv->fd_ctl, drv->client_handle, drv->device_handle,
                    &buf->handle, NV50_MEMORY_VIRTUAL, NULL) != 0) {
        LOG_DEBUG("NV driver: RM buffer alloc failed, using host fallback");
    }

    /* Assign a GPU VA from our allocator */
    uint64_t aligned_size = (size + 4095) & ~4095ULL;
    if (drv->va_current + aligned_size > drv->va_end) {
        LOG_ERROR("NV driver: GPU VA space exhausted");
        free(buf);
        return NULL;
    }
    buf->gpu_va = drv->va_current;
    drv->va_current += aligned_size;

    if (host_visible) {
        /* mmap a host-visible mapping for CPU access */
        buf->cpu_addr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (buf->cpu_addr == MAP_FAILED) {
            LOG_ERROR("NV driver: buffer mmap failed: %s", strerror(errno));
            free(buf);
            return NULL;
        }
        memset(buf->cpu_addr, 0, size);
    } else {
        buf->cpu_addr = NULL;
    }
#else
    (void)host_visible;
    buf->gpu_va   = 0;
    buf->cpu_addr = NULL;
    buf->handle   = 0;
#endif

    LOG_DEBUG("NV driver: buffer created, size=%zu, gpu_va=0x%llX, host=%s",
              size, (unsigned long long)buf->gpu_va,
              host_visible ? "yes" : "no");
    return buf;
}

void cml_nv_buffer_free(CMLNVDriver* drv, CMLNVBuffer* buf) {
    if (!drv || !buf) return;

#ifdef __linux__
    /* Unmap host-visible mapping */
    if (buf->cpu_addr && buf->cpu_addr != MAP_FAILED) {
        munmap(buf->cpu_addr, buf->size);
    }

    /* Free RM allocation */
    if (drv->initialized && buf->handle) {
        nv_rm_free(drv->fd_ctl, drv->client_handle,
                   drv->device_handle, buf->handle);
    }
#endif

    free(buf);
}

int cml_nv_buffer_upload(CMLNVDriver* drv, CMLNVBuffer* dst,
                         const void* src, size_t n) {
    if (!drv || !dst || !src || n == 0) return -1;
    if (!drv->initialized) {
        LOG_ERROR("NV driver: not initialized");
        return -1;
    }
    if (n > dst->size) {
        LOG_ERROR("NV driver: upload size %zu exceeds buffer size %zu", n, dst->size);
        return -1;
    }

    if (dst->host_visible && dst->cpu_addr) {
        /* For host-visible buffers we can memcpy directly, then the GPU
         * would see it via the coherent mapping.  In a real driver we
         * would issue a DMA copy for device-local buffers. */
        memcpy(dst->cpu_addr, src, n);
        LOG_DEBUG("NV driver: uploaded %zu bytes to buffer GPU VA 0x%llX",
                  n, (unsigned long long)dst->gpu_va);
        return 0;
    }

    /* Device-local buffer without host mapping -- would need DMA engine.
     * For now, log and return error. */
    LOG_WARNING("NV driver: upload to device-local buffer not implemented (no DMA)");
    return -1;
}

int cml_nv_buffer_download(CMLNVDriver* drv, CMLNVBuffer* src,
                           void* dst, size_t n) {
    if (!drv || !src || !dst || n == 0) return -1;
    if (!drv->initialized) {
        LOG_ERROR("NV driver: not initialized");
        return -1;
    }
    if (n > src->size) {
        LOG_ERROR("NV driver: download size %zu exceeds buffer size %zu", n, src->size);
        return -1;
    }

    if (src->host_visible && src->cpu_addr) {
        memcpy(dst, src->cpu_addr, n);
        LOG_DEBUG("NV driver: downloaded %zu bytes from buffer GPU VA 0x%llX",
                  n, (unsigned long long)src->gpu_va);
        return 0;
    }

    LOG_WARNING("NV driver: download from device-local buffer not implemented (no DMA)");
    return -1;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Kernel compilation: PTX -> CUBIN via ptxas subprocess
 * ══════════════════════════════════════════════════════════════════════════ */

CMLNVKernel* cml_nv_kernel_compile_ptx(CMLNVDriver* drv, const char* ptx_code,
                                        const char* kernel_name) {
    if (!ptx_code || !kernel_name) return NULL;

    /* drv can be NULL if we just want to compile without a driver context */
    int sm = 75;  /* default: Turing */
    if (drv && drv->compute_cap_major > 0) {
        sm = drv->compute_cap_major * 10 + drv->compute_cap_minor;
    }

    CMLNVKernel* kernel = (CMLNVKernel*)calloc(1, sizeof(CMLNVKernel));
    if (!kernel) {
        LOG_ERROR("NV driver: failed to allocate kernel descriptor");
        return NULL;
    }
    kernel->name = strdup(kernel_name);
    if (!kernel->name) {
        free(kernel);
        return NULL;
    }

    /* Write PTX to a temporary file */
    char ptx_path[256];
    char cubin_path[256];
    snprintf(ptx_path, sizeof(ptx_path), "/tmp/cml_nv_%s.ptx", kernel_name);
    snprintf(cubin_path, sizeof(cubin_path), "/tmp/cml_nv_%s.cubin", kernel_name);

    FILE* ptx_file = fopen(ptx_path, "w");
    if (!ptx_file) {
        LOG_ERROR("NV driver: failed to write PTX temp file %s: %s",
                  ptx_path, strerror(errno));
        free(kernel->name);
        free(kernel);
        return NULL;
    }
    fputs(ptx_code, ptx_file);
    fclose(ptx_file);

    /* Invoke ptxas to compile PTX -> CUBIN */
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "ptxas -arch=sm_%d -o %s %s 2>&1",
             sm, cubin_path, ptx_path);

    LOG_DEBUG("NV driver: compiling PTX with: %s", cmd);

    FILE* proc = popen(cmd, "r");
    if (!proc) {
        LOG_ERROR("NV driver: failed to run ptxas: %s", strerror(errno));
        unlink(ptx_path);
        free(kernel->name);
        free(kernel);
        return NULL;
    }

    /* Read ptxas output (errors/warnings) */
    char output[2048];
    size_t total_read = 0;
    while (total_read < sizeof(output) - 1) {
        size_t r = fread(output + total_read, 1, sizeof(output) - 1 - total_read, proc);
        if (r == 0) break;
        total_read += r;
    }
    output[total_read] = '\0';

    int status = pclose(proc);
    unlink(ptx_path);  /* Clean up temp PTX file */

    if (status != 0) {
        LOG_ERROR("NV driver: ptxas failed (status %d): %s", status, output);
        unlink(cubin_path);
        free(kernel->name);
        free(kernel);
        return NULL;
    }

    if (total_read > 0) {
        LOG_DEBUG("NV driver: ptxas output: %s", output);
    }

    /* Read the compiled CUBIN */
    FILE* cubin_file = fopen(cubin_path, "rb");
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
        LOG_ERROR("NV driver: CUBIN file is empty");
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
        LOG_ERROR("NV driver: short read on CUBIN file");
        fclose(cubin_file);
        unlink(cubin_path);
        free(kernel->cubin_data);
        free(kernel->name);
        free(kernel);
        return NULL;
    }
    fclose(cubin_file);
    unlink(cubin_path);

    LOG_INFO("NV driver: compiled kernel '%s' -> %zu bytes CUBIN (sm_%d)",
             kernel_name, kernel->cubin_size, sm);
    return kernel;
}

void cml_nv_kernel_free(CMLNVDriver* drv, CMLNVKernel* kernel) {
    if (!kernel) return;

#ifdef __linux__
    /* If the kernel was loaded into the GPU, free the RM handle */
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

/* ══════════════════════════════════════════════════════════════════════════
 * Kernel launch via GPFIFO
 * ══════════════════════════════════════════════════════════════════════════ */

int cml_nv_kernel_launch(CMLNVDriver* drv, CMLNVKernel* kernel,
                         uint32_t grid[3], uint32_t block[3],
                         void** args, int num_args) {
    if (!drv || !drv->initialized || !kernel) return -1;
    (void)args;
    (void)num_args;

    LOG_DEBUG("NV driver: launching kernel '%s' grid=[%u,%u,%u] block=[%u,%u,%u]",
              kernel->name,
              grid[0], grid[1], grid[2],
              block[0], block[1], block[2]);

#ifdef __linux__
    if (!drv->gpfifo.entries) {
        LOG_WARNING("NV driver: GPFIFO not available, cannot dispatch");
        return -1;
    }

    /* ── Build a simplified command buffer entry ──
     *
     * A real GPFIFO entry is a 64-bit word: [GPU_VA_of_pushbuffer : length].
     * The pushbuffer itself would contain NV method calls to:
     *   1. Bind the CUBIN module
     *   2. Set grid/block dimensions
     *   3. Set kernel arguments (constant buffer bindings)
     *   4. Issue a LAUNCH method
     *   5. Release a semaphore on completion
     *
     * Here we write a placeholder entry to demonstrate the ring buffer
     * mechanics (put pointer advance, wrap, doorbell ring).
     */

    uint32_t put = drv->gpfifo.put_offset;

    /* Write GPFIFO entry: GPU VA of the "command stream" (placeholder) */
    drv->gpfifo.entries[put] = kernel->gpu_addr | ((uint64_t)0x40 << 40);

    /* Advance put pointer with wrap */
    put = (put + 1) % drv->gpfifo.num_entries;
    drv->gpfifo.put_offset = put;

    /* Ring the doorbell -- in a real driver, we would write the new put
     * offset to a mmap'd doorbell register to notify the GPU. */
    if (drv->gpfifo.doorbell) {
        volatile uint32_t* bell = (volatile uint32_t*)drv->gpfifo.doorbell;
        *bell = put;
    }

    /* Bump the semaphore value so cml_nv_synchronize() knows what to wait for */
    drv->semaphore_value++;

    LOG_DEBUG("NV driver: GPFIFO put=%u, semaphore target=%llu",
              put, (unsigned long long)drv->semaphore_value);
    return 0;
#else
    LOG_ERROR("NV driver: GPFIFO dispatch only supported on Linux");
    return -1;
#endif
}

/* ══════════════════════════════════════════════════════════════════════════
 * Synchronization: poll the semaphore
 * ══════════════════════════════════════════════════════════════════════════ */

int cml_nv_synchronize(CMLNVDriver* drv) {
    if (!drv || !drv->initialized) return -1;

#ifdef __linux__
    if (!drv->semaphore) {
        LOG_WARNING("NV driver: no semaphore mapped, sync is a no-op");
        return 0;
    }

    /* Poll the semaphore until it reaches the expected value.
     *
     * In a real driver the GPU writes to this memory location when it
     * finishes processing the GPFIFO entries up to the semaphore release
     * method.  Here we simulate immediate completion for the stub.
     */
    uint64_t deadline = now_ms() + 5000;  /* 5-second timeout */
    uint64_t target = drv->semaphore_value;

    while ((uint64_t)(*drv->semaphore) < target) {
        if (now_ms() >= deadline) {
            LOG_ERROR("NV driver: synchronize timed out waiting for semaphore "
                      "(current=%u, target=%llu)",
                      *drv->semaphore, (unsigned long long)target);
            return -1;
        }
        /* Spin-wait.  A real driver might use usleep() or futex(). */
    }

    LOG_DEBUG("NV driver: synchronized (semaphore=%llu)",
              (unsigned long long)*drv->semaphore);
    return 0;
#else
    return 0;
#endif
}

/* ══════════════════════════════════════════════════════════════════════════
 * Graph execution (stub)
 * ══════════════════════════════════════════════════════════════════════════ */

int cml_nv_execute_graph(CMLNVDriver* drv, CMLGraph_t ir) {
    if (!drv || !ir) return -1;

    LOG_INFO("NV driver: cml_nv_execute_graph() called (stub, not yet implemented)");
    LOG_INFO("NV driver: full implementation would walk IR, compile PTX per-op "
             "via ptx_codegen, and launch kernels through GPFIFO");

    /* TODO: Implementation deferred.  The flow would be:
     *
     * 1. Walk the IR graph nodes
     * 2. For each compute node, call the appropriate cml_ptx_gen_*()
     *    function from ptx_codegen.h to get PTX source
     * 3. Compile PTX to CUBIN via cml_nv_kernel_compile_ptx()
     * 4. Allocate device buffers for inputs/outputs
     * 5. Upload input data
     * 6. Launch kernels via cml_nv_kernel_launch()
     * 7. Synchronize via cml_nv_synchronize()
     * 8. Download output data
     * 9. Free temporary buffers and kernels
     */

    return -1;
}
