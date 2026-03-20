/*
 * AMD userspace driver -- bypasses libamdhip64.so, direct KFD ioctl.
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

#define AM_GFX_MI300X   0x120100  /* gfx942 */
#define AM_GFX_MI350    0x120200  /* gfx950 */
#define AM_GFX_RDNA4    0x120300  /* gfx12  */

typedef struct CMLAMChipletConfig {
    int num_xcd;
    int cu_per_xcd;
    int sdma_per_xcd;
    bool unified_memory;
} CMLAMChipletConfig;

/* AQL dispatch packet (HSA spec, 64 bytes) */
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

/* AQL barrier-AND packet (HSA spec, 64 bytes) */
typedef struct __attribute__((aligned(64))) hsa_barrier_and_packet_t {
    uint16_t header;
    uint8_t  reserved0[6];
    uint64_t dep_signal[5];
    uint64_t reserved1;
    uint64_t completion_signal;
} hsa_barrier_and_packet_t;

/* AQL barrier-OR packet (HSA spec, 64 bytes) */
typedef struct __attribute__((aligned(64))) hsa_barrier_or_packet_t {
    uint16_t header;
    uint8_t  reserved0[6];
    uint64_t dep_signal[5];
    uint64_t reserved1;
    uint64_t completion_signal;
} hsa_barrier_or_packet_t;

typedef struct CMLAMBuffer {
    uint64_t gpu_va;
    void*    cpu_addr;
    size_t   size;
    uint32_t handle;
    bool     is_vram;
} CMLAMBuffer;

typedef struct CMLAMKernel {
    void*    code_object;
    size_t   code_size;
    uint64_t gpu_addr;
    uint32_t handle;
    char*    name;
    uint32_t group_segment_size;
    uint32_t private_segment_size;
    uint32_t kernarg_size;
    uint32_t vgpr_count;
    uint32_t sgpr_count;
    int64_t  kern_code_entry_offset;
} CMLAMKernel;

typedef struct CMLAMQueue {
    hsa_kernel_dispatch_packet_t* ring;
    uint32_t ring_size;
    volatile uint64_t* write_dispatch_id;
    volatile uint64_t* read_dispatch_id;
    volatile uint32_t* doorbell;
    uint64_t queue_id;
    uint32_t queue_handle;
    bool     active;
} CMLAMQueue;

/* SDMA (System DMA) queue for async H2D/D2H transfers */
typedef struct CMLAMSDMAQueue {
    uint32_t* ring;
    uint32_t  ring_size;
    volatile uint64_t* write_ptr;
    volatile uint64_t* read_ptr;
    volatile uint32_t* doorbell;
    uint32_t  queue_handle;
    uint64_t  queue_id;
    bool      active;
} CMLAMSDMAQueue;

/* HSA-style signal backed by GPU-visible GTT memory */
typedef struct CMLAMSignal {
    volatile uint64_t* value;
    uint64_t           gpu_va;
    uint64_t           handle;
    uint64_t           target;
} CMLAMSignal;

/* GPU information from KFD topology sysfs */
typedef struct CMLAMGPUInfo {
    uint32_t gpu_id;
    char     name[64];
    char     gfx_version[32];
    int      cu_count;
    size_t   vram_size;
    uint32_t simd_per_cu;
    uint32_t max_waves_per_simd;
    uint32_t max_slots_scratch;
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t domain;
    uint32_t location_id;
    uint32_t fw_version;
    uint32_t max_engine_clk;
    uint32_t local_mem_size;
    uint32_t lds_size_per_cu;
    int      sdma_count;
    int      node_id;
} CMLAMGPUInfo;

/* Scratch memory state */
typedef struct CMLAMScratch {
    uint64_t gpu_va;
    uint64_t handle;
    size_t   size;
    size_t   per_thread_size;
    uint32_t max_waves;
} CMLAMScratch;

#define AM_MAX_COMPUTE_QUEUES 4

typedef struct CMLAMDriver {
    int      fd_kfd;
    int      fd_drm;
    bool     initialized;

    uint32_t gpu_id;
    char     device_name[256];
    char     gfx_version[32];
    size_t   total_vram;
    int      cu_count;

    CMLAMGPUInfo gpu_info;

    CMLAMQueue   compute_queues[AM_MAX_COMPUTE_QUEUES];
    int          num_compute_queues;
    CMLAMQueue   aql_queue;

    CMLAMSDMAQueue sdma_queue;
    bool           has_sdma;

    volatile uint64_t* signal;
    uint64_t           signal_gpu_va;
    uint64_t           signal_value;

    CMLAMScratch scratch;
    bool         has_scratch;

    CMLAMChipletConfig chiplet_config;
    bool               is_chiplet_gpu;

    uint64_t va_start;
    uint64_t va_current;
    uint64_t va_end;
} CMLAMDriver;

/* Driver lifecycle */
bool          cml_am_driver_available(void);
CMLAMDriver*  cml_am_driver_create(void);
int           cml_am_driver_init(CMLAMDriver* drv);
void          cml_am_driver_free(CMLAMDriver* drv);

/* GPU discovery */
int cml_am_enumerate_gpus(CMLAMGPUInfo** gpus, int* count);
CMLAMChipletConfig cml_am_get_chiplet_config(const char* gfx_version);
int cml_am_sdma_copy_nearest_xcd(CMLAMDriver* drv, int xcd_idx,
                                  uint64_t dst_va, uint64_t src_va, size_t size);

/* Buffer management */
CMLAMBuffer* cml_am_buffer_create(CMLAMDriver* drv, size_t size, bool vram);
void         cml_am_buffer_free(CMLAMDriver* drv, CMLAMBuffer* buf);
int          cml_am_buffer_upload(CMLAMDriver* drv, CMLAMBuffer* dst, const void* src, size_t n);
int          cml_am_buffer_download(CMLAMDriver* drv, CMLAMBuffer* src, void* dst, size_t n);

/* Kernel loading and dispatch */
CMLAMKernel* cml_am_kernel_load(CMLAMDriver* drv, const void* code_object, size_t code_size,
                                  const char* kernel_name);
void         cml_am_kernel_free(CMLAMDriver* drv, CMLAMKernel* kernel);
int          cml_am_kernel_launch(CMLAMDriver* drv, CMLAMKernel* kernel,
                                   uint32_t grid[3], uint32_t block[3],
                                   void* kernarg, uint32_t kernarg_size);
int          cml_am_kernel_launch_on_queue(CMLAMDriver* drv, int queue_idx,
                                           CMLAMKernel* kernel,
                                           uint32_t grid[3], uint32_t block[3],
                                           void* kernarg, uint32_t kernarg_size,
                                           CMLAMSignal* completion);

/* Multi-queue */
int cml_am_create_compute_queue(CMLAMDriver* drv, int queue_index);

/* SDMA (DMA engine) */
int cml_am_sdma_queue_create(CMLAMDriver* drv);
int cml_am_sdma_copy(CMLAMDriver* drv, uint64_t dst_va, uint64_t src_va, size_t size);
int cml_am_sdma_fence(CMLAMDriver* drv, uint64_t signal_va, uint64_t value);
int cml_am_sdma_synchronize(CMLAMDriver* drv);

/* Signal system */
CMLAMSignal* cml_am_signal_create(CMLAMDriver* drv, uint64_t initial_value);
void         cml_am_signal_free(CMLAMDriver* drv, CMLAMSignal* signal);
int          cml_am_signal_wait(CMLAMSignal* signal, uint64_t expected, uint64_t timeout_ns);

/* Barrier packets */
int cml_am_barrier_and(CMLAMDriver* drv, int queue_idx,
                       CMLAMSignal** deps, int num_deps,
                       CMLAMSignal* completion);
int cml_am_barrier_or(CMLAMDriver* drv, int queue_idx,
                      CMLAMSignal** deps, int num_deps,
                      CMLAMSignal* completion);

/* Synchronization */
int cml_am_synchronize(CMLAMDriver* drv);

/* Scratch and LDS */
int  cml_am_alloc_scratch(CMLAMDriver* drv, size_t per_thread_size, uint32_t max_waves);
bool cml_am_validate_lds(CMLAMDriver* drv, uint32_t requested_bytes);

/* Error recovery */
int cml_am_check_gpu_hang(CMLAMDriver* drv);
int cml_am_gpu_reset(CMLAMDriver* drv);
int cml_am_dump_wave_status(CMLAMDriver* drv);

/* Graph execution */
int cml_am_execute_graph(CMLAMDriver* drv, CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif /* CML_GPU_AM_DRIVER_H */
