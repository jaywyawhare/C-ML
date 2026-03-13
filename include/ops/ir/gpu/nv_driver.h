/**
 * @file nv_driver.h
 * @brief NVIDIA userspace driver — bypasses libcuda.so, direct ioctl to kernel driver
 *
 * Opens /dev/nvidiactl and /dev/nvidia0 directly, uses RM ioctls for
 * resource management, GPFIFO for kernel dispatch, and semaphore polling
 * for synchronization. Reuses PTX codegen + ptxas for CUBIN compilation.
 */

#ifndef CML_GPU_NV_DRIVER_H
#define CML_GPU_NV_DRIVER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CMLGraph;
typedef struct CMLGraph* CMLGraph_t;

/* ── NV RM ioctl constants ── */

#define NV_ESC_RM_ALLOC      0x2B
#define NV_ESC_RM_CONTROL    0x2A
#define NV_ESC_RM_FREE       0x29

/* ── NV buffer ── */

typedef struct CMLNVBuffer {
    uint64_t gpu_va;       /* GPU virtual address */
    void*    cpu_addr;     /* mmap'd host pointer (NULL if device-only) */
    size_t   size;
    uint32_t handle;       /* RM allocation handle */
    bool     host_visible;
} CMLNVBuffer;

/* ── NV compiled kernel ── */

typedef struct CMLNVKernel {
    void*    cubin_data;   /* Compiled CUBIN blob */
    size_t   cubin_size;
    uint64_t gpu_addr;     /* GPU address of loaded CUBIN */
    uint32_t handle;       /* RM handle for the loaded module */
    char*    name;
    int      num_regs;
    int      shared_mem;
} CMLNVKernel;

/* ── GPFIFO ring buffer ── */

typedef struct CMLNVGPFIFO {
    uint64_t* entries;     /* Ring buffer of GP entries (GPU VA + size) */
    uint32_t  num_entries;
    uint32_t  put_offset;  /* Write pointer */
    uint32_t  get_offset;  /* Read pointer (GPU updates this) */
    void*     doorbell;    /* Doorbell register (mmap'd) */
    uint64_t  gpu_va;      /* GPU VA of the GPFIFO itself */
    uint32_t  handle;      /* RM channel handle */
} CMLNVGPFIFO;

/* ── NV driver context ── */

typedef struct CMLNVDriver {
    int      fd_ctl;         /* /dev/nvidiactl */
    int      fd_dev;         /* /dev/nvidia0 */
    bool     initialized;

    /* RM object hierarchy: client -> device -> subdevice -> channel */
    uint32_t client_handle;
    uint32_t device_handle;
    uint32_t subdevice_handle;
    uint32_t channel_group_handle;

    /* GPU info */
    char     device_name[256];
    uint32_t gpu_arch;       /* e.g. 0x190 for Turing, 0x1A0 for Ampere */
    size_t   total_memory;
    int      sm_count;
    int      compute_cap_major;
    int      compute_cap_minor;

    /* GPFIFO */
    CMLNVGPFIFO gpfifo;

    /* Virtual address allocator */
    uint64_t va_start;
    uint64_t va_current;
    uint64_t va_end;

    /* Semaphore for synchronization */
    volatile uint32_t* semaphore;
    uint64_t           semaphore_gpu_va;
    uint64_t           semaphore_value;
} CMLNVDriver;

/* ── Driver lifecycle ── */

bool         cml_nv_driver_available(void);
CMLNVDriver* cml_nv_driver_create(void);
int          cml_nv_driver_init(CMLNVDriver* drv);
void         cml_nv_driver_free(CMLNVDriver* drv);

/* ── Buffer management ── */

CMLNVBuffer* cml_nv_buffer_create(CMLNVDriver* drv, size_t size, bool host_visible);
void         cml_nv_buffer_free(CMLNVDriver* drv, CMLNVBuffer* buf);
int          cml_nv_buffer_upload(CMLNVDriver* drv, CMLNVBuffer* dst, const void* src, size_t n);
int          cml_nv_buffer_download(CMLNVDriver* drv, CMLNVBuffer* src, void* dst, size_t n);

/* ── Kernel management ── */

CMLNVKernel* cml_nv_kernel_compile_ptx(CMLNVDriver* drv, const char* ptx_code,
                                         const char* kernel_name);
void         cml_nv_kernel_free(CMLNVDriver* drv, CMLNVKernel* kernel);
int          cml_nv_kernel_launch(CMLNVDriver* drv, CMLNVKernel* kernel,
                                   uint32_t grid[3], uint32_t block[3],
                                   void** args, int num_args);

/* ── Synchronization ── */

int cml_nv_synchronize(CMLNVDriver* drv);

/* ── Graph execution ── */

int cml_nv_execute_graph(CMLNVDriver* drv, CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif /* CML_GPU_NV_DRIVER_H */
