/**
 * @file am_driver.h
 * @brief AMD userspace driver — bypasses libamdhip64.so, direct KFD ioctl
 *
 * Opens /dev/kfd directly, uses KFD ioctls for queue creation, memory
 * allocation, and AQL (Architected Queuing Language) packet submission.
 * Targets RDNA3/RDNA4 (gfx11.x) via LLVM AMDGPU codegen.
 */

#ifndef CML_GPU_AM_DRIVER_H
#define CML_GPU_AM_DRIVER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CMLGraph;
typedef struct CMLGraph* CMLGraph_t;

/* ── AQL dispatch packet (HSA spec, 64 bytes) ── */

typedef struct __attribute__((aligned(64))) hsa_kernel_dispatch_packet_t {
    uint16_t header;
    uint16_t setup;
    uint16_t workgroup_size_x;
    uint16_t workgroup_size_y;
    uint16_t workgroup_size_z;
    uint16_t reserved0;
    uint32_t grid_size_x;
    uint32_t grid_size_y;
    uint32_t grid_size_z;
    uint32_t private_segment_size;
    uint32_t group_segment_size;
    uint64_t kernel_object;
    uint64_t kernarg_address;
    uint64_t reserved2;
    uint64_t completion_signal;
} hsa_kernel_dispatch_packet_t;

/* ── AM buffer ── */

typedef struct CMLAMBuffer {
    uint64_t gpu_va;       /* GPU virtual address */
    void*    cpu_addr;     /* mmap'd host pointer (NULL if VRAM-only) */
    size_t   size;
    uint32_t handle;       /* KFD memory handle */
    bool     is_vram;      /* true = VRAM, false = GTT (system memory) */
} CMLAMBuffer;

/* ── AM compiled kernel ── */

typedef struct CMLAMKernel {
    void*    code_object;  /* ELF code object blob */
    size_t   code_size;
    uint64_t gpu_addr;     /* GPU address of loaded code */
    uint32_t handle;       /* KFD handle */
    char*    name;
    uint32_t group_segment_size;
    uint32_t private_segment_size;
    uint32_t kernarg_size;
} CMLAMKernel;

/* ── AQL queue ── */

typedef struct CMLAMQueue {
    hsa_kernel_dispatch_packet_t* ring;  /* Ring buffer of AQL packets */
    uint32_t ring_size;                   /* Number of packets in ring */
    volatile uint64_t* write_dispatch_id; /* Write pointer */
    volatile uint64_t* read_dispatch_id;  /* Read pointer (GPU updates) */
    volatile uint32_t* doorbell;          /* Doorbell register */
    uint64_t queue_id;                    /* KFD queue ID */
    uint32_t queue_handle;               /* KFD handle */
} CMLAMQueue;

/* ── AM driver context ── */

typedef struct CMLAMDriver {
    int      fd_kfd;         /* /dev/kfd */
    int      fd_drm;         /* /dev/dri/renderD128 */
    bool     initialized;

    /* GPU info */
    uint32_t gpu_id;
    char     device_name[256];
    char     gfx_version[32]; /* e.g. "gfx1100" */
    size_t   total_vram;
    int      cu_count;        /* Compute units */

    /* AQL queue */
    CMLAMQueue aql_queue;

    /* Completion signal for synchronization */
    volatile uint64_t* signal;
    uint64_t           signal_gpu_va;
    uint64_t           signal_value;

    /* Virtual address range */
    uint64_t va_start;
    uint64_t va_current;
    uint64_t va_end;
} CMLAMDriver;

/* ── Driver lifecycle ── */

bool          cml_am_driver_available(void);
CMLAMDriver*  cml_am_driver_create(void);
int           cml_am_driver_init(CMLAMDriver* drv);
void          cml_am_driver_free(CMLAMDriver* drv);

/* ── Buffer management ── */

CMLAMBuffer* cml_am_buffer_create(CMLAMDriver* drv, size_t size, bool vram);
void         cml_am_buffer_free(CMLAMDriver* drv, CMLAMBuffer* buf);
int          cml_am_buffer_upload(CMLAMDriver* drv, CMLAMBuffer* dst, const void* src, size_t n);
int          cml_am_buffer_download(CMLAMDriver* drv, CMLAMBuffer* src, void* dst, size_t n);

/* ── Kernel management ── */

CMLAMKernel* cml_am_kernel_load(CMLAMDriver* drv, const void* code_object, size_t code_size,
                                  const char* kernel_name);
void         cml_am_kernel_free(CMLAMDriver* drv, CMLAMKernel* kernel);
int          cml_am_kernel_launch(CMLAMDriver* drv, CMLAMKernel* kernel,
                                   uint32_t grid[3], uint32_t block[3],
                                   void* kernarg, uint32_t kernarg_size);

/* ── Synchronization ── */

int cml_am_synchronize(CMLAMDriver* drv);

/* ── Graph execution ── */

int cml_am_execute_graph(CMLAMDriver* drv, CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif /* CML_GPU_AM_DRIVER_H */
