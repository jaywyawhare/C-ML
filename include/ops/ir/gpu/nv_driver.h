/*
 * NVIDIA userspace driver -- bypasses libcuda.so, direct ioctl to kernel driver.
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

#include "ops/ir/gpu/nv_qmd.h"

#ifdef __cplusplus
extern "C" {
#endif

struct CMLGraph;
typedef struct CMLGraph* CMLGraph_t;

/* RM ioctl escape numbers */
#define NV_ESC_RM_ALLOC      0x2B
#define NV_ESC_RM_CONTROL    0x2A
#define NV_ESC_RM_FREE       0x29
#define NV_ESC_RM_MAP_MEMORY 0x4E
#define NV_ESC_RM_UNMAP_MEMORY 0x4F

/* RM NV class IDs */
#define NV01_ROOT_CLIENT            0x00000041
#define NV01_DEVICE_0               0x00000080
#define NV20_SUBDEVICE_0            0x00002080
#define NV50_MEMORY_VIRTUAL         0x000050A0
#define NV01_MEMORY_SYSTEM          0x0000003E
#define NV01_MEMORY_LOCAL           0x0000003D
#define FERMI_VASPACE_A             0x000090F1
#define KEPLER_CHANNEL_GROUP_A      0x0000A06C
#define KEPLER_CHANNEL_GPFIFO_A     0x0000A06F
#define TURING_CHANNEL_GPFIFO_A     0x0000C46F
#define AMPERE_CHANNEL_GPFIFO_A     0x0000C56F
#define HOPPER_CHANNEL_GPFIFO_A     0x0000C86F
#define BLACKWELL_CHANNEL_GPFIFO_A  0x0000C96F
#define TURING_COMPUTE_A            0x0000C3C0
#define AMPERE_COMPUTE_A            0x0000C6C0
#define HOPPER_COMPUTE_A            0x0000C9C0
#define BLACKWELL_COMPUTE_A         0x0000CAC0
#define TURING_DMA_COPY_A           0x0000C3B5
#define AMPERE_DMA_COPY_A           0x0000C6B5
#define HOPPER_DMA_COPY_A           0x0000C8B5
#define BLACKWELL_DMA_COPY_A        0x0000C9B5

/* RM control commands */
#define NV2080_CTRL_CMD_GPU_GET_INFO         0x20800101
#define NV2080_CTRL_CMD_GPU_GET_NAME_STRING  0x20800110
#define NV0080_CTRL_CMD_FIFO_GET_CHANNELLIST 0x00801713
#define NV2080_CTRL_CMD_GR_GET_INFO          0x20801201
#define NV2080_CTRL_CMD_FB_GET_INFO          0x20801301
#define NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES 0x90F10106

/* Compute subchannel and method registers */
#define NVC0_SUBCHANNEL_COMPUTE  1
#define NVC0_SUBCHANNEL_COPY     4

#define NVC0B5_OFFSET_IN_UPPER         0x0400
#define NVC0B5_OFFSET_IN_LOWER         0x0404
#define NVC0B5_OFFSET_OUT_UPPER        0x0408
#define NVC0B5_OFFSET_OUT_LOWER        0x040C
#define NVC0B5_PITCH_IN                0x0410
#define NVC0B5_PITCH_OUT               0x0414
#define NVC0B5_LINE_LENGTH_IN          0x0418
#define NVC0B5_LINE_COUNT              0x041C
#define NVC0B5_LAUNCH_DMA              0x0300

#define NVC3C0_SET_OBJECT              0x0000
#define NVC3C0_INVALIDATE_SHADER_CACHES 0x021C
#define NVC3C0_SET_SHADER_LOCAL_MEMORY_A 0x0204
#define NVC3C0_SET_SHADER_LOCAL_MEMORY_B 0x0208
#define NVC3C0_SEND_PCAS_A             0x0D00
#define NVC3C0_SEND_SIGNALING_PCAS_B   0x0D04
#define NVC3C0_LAUNCH                  0x0368
#define NVC3C0_LOAD_INLINE_QMD_DATA(i) (0x0B00 + (i) * 4)

#define NVC3C0_SET_REPORT_SEMAPHORE_A  0x06C0
#define NVC3C0_SET_REPORT_SEMAPHORE_B  0x06C4
#define NVC3C0_SET_REPORT_SEMAPHORE_C  0x06C8
#define NVC3C0_SET_REPORT_SEMAPHORE_D  0x06CC

#define NV_SEMAPHORE_RELEASE_WFI       0x04
#define NV_SEMAPHORE_RELEASE           0x00
#define NV_SEMAPHORE_ACQUIRE_GEQ      0x01

/* GPFIFO entry encoding */
#define NV_GPFIFO_ENTRY(gpu_va, len_dwords) \
    (((uint64_t)(len_dwords) << 42) | (((gpu_va) >> 2) & 0x3FFFFFFFFFFULL))

#define NV_GPFIFO_DEFAULT_ENTRIES  64
#define NV_GPFIFO_ENTRY_BYTES      8

/* NV pushbuffer method encoding */
#define NV_FIFO_INCR(subchan, reg, count) \
    (0x20000000U | ((unsigned)(count) << 16) | ((unsigned)(subchan) << 13) | ((unsigned)(reg) >> 2))
#define NV_FIFO_NONINCR(subchan, reg, count) \
    (0x60000000U | ((unsigned)(count) << 16) | ((unsigned)(subchan) << 13) | ((unsigned)(reg) >> 2))
#define NV_FIFO_INLINE(subchan, reg, value) \
    (0x80000000U | ((unsigned)(value) << 16) | ((unsigned)(subchan) << 13) | ((unsigned)(reg) >> 2))
#define NV_FIFO_IMM(subchan, reg, value) \
    NV_FIFO_INLINE(subchan, reg, value)

/* USERD doorbell layout */
#define NV_USERD_GP_PUT_OFFSET   0x90

typedef struct CMLNVBuffer {
    uint64_t gpu_va;
    void*    cpu_addr;
    size_t   size;
    uint32_t handle;
    uint32_t va_handle;
    bool     host_visible;
    bool     is_vram;
} CMLNVBuffer;

typedef struct CMLNVKernel {
    void*    cubin_data;
    size_t   cubin_size;
    uint64_t gpu_addr;
    uint32_t handle;
    char*    name;
    int      num_regs;
    int      shared_mem;
    int      param_size;
    int      bar_count;
    NVKernelMeta meta;
    struct CMLNVBuffer* code_buffer; 
} CMLNVKernel;

typedef struct CMLNVGPFIFO {
    uint64_t* entries;
    uint32_t  num_entries;
    uint32_t  put_offset;
    uint32_t  get_offset;
    void*     doorbell;
    uint64_t  gpu_va;
    uint32_t  handle;
} CMLNVGPFIFO;

typedef struct CMLNVPushbuf {
    uint32_t *buf;
    uint32_t  pos;
    uint32_t  capacity;
    uint64_t  gpu_va;
    CMLNVBuffer *backing;
} CMLNVPushbuf;

typedef struct CMLNVDriver {
    int      fd_ctl;
    int      fd_dev;
    int      fd_uvm;
    bool     initialized;

    uint32_t client_handle;
    uint32_t device_handle;
    uint32_t subdevice_handle;
    uint32_t vaspace_handle;
    uint32_t channel_group_handle;
    uint32_t channel_handle;
    uint32_t compute_obj_handle;
    uint32_t copy_obj_handle;

    char     device_name[256];
    uint32_t gpu_arch;
    size_t   total_memory;
    int      sm_count;
    int      compute_cap_major;
    int      compute_cap_minor;

    CMLNVGPFIFO gpfifo;
    CMLNVBuffer *gpfifo_buf;
    CMLNVBuffer *userd_buf;

    CMLNVPushbuf pushbuf;

    uint64_t va_start;
    uint64_t va_current;
    uint64_t va_end;

    CMLNVBuffer *semaphore_buf;
    volatile uint32_t *semaphore;
    uint64_t           semaphore_gpu_va;
    uint64_t           semaphore_value;
} CMLNVDriver;

bool         cml_nv_driver_available(void);
CMLNVDriver* cml_nv_driver_create(void);
int          cml_nv_driver_init(CMLNVDriver* drv);
void         cml_nv_driver_free(CMLNVDriver* drv);

CMLNVBuffer* cml_nv_buffer_create(CMLNVDriver* drv, size_t size, bool host_visible);
CMLNVBuffer* cml_nv_buffer_create_vram(CMLNVDriver* drv, size_t size);
void         cml_nv_buffer_free(CMLNVDriver* drv, CMLNVBuffer* buf);
int          cml_nv_buffer_upload(CMLNVDriver* drv, CMLNVBuffer* dst, const void* src, size_t n);
int          cml_nv_buffer_download(CMLNVDriver* drv, CMLNVBuffer* src, void* dst, size_t n);
int          cml_nv_buffer_copy(CMLNVDriver* drv, CMLNVBuffer* dst, CMLNVBuffer* src, size_t n);

CMLNVKernel* cml_nv_kernel_compile_ptx(CMLNVDriver* drv, const char* ptx_code,
                                         const char* kernel_name);
CMLNVKernel* cml_nv_kernel_load_cubin(CMLNVDriver* drv, const void* cubin, size_t size,
                                        const char* kernel_name);
void         cml_nv_kernel_free(CMLNVDriver* drv, CMLNVKernel* kernel);
int          cml_nv_kernel_launch(CMLNVDriver* drv, CMLNVKernel* kernel,
                                   uint32_t grid[3], uint32_t block[3],
                                   void** args, int num_args);

int cml_nv_synchronize(CMLNVDriver* drv);
int cml_nv_gpu_wait_semaphore(CMLNVDriver* drv, uint64_t sem_va, uint32_t value);

int cml_nv_execute_graph(CMLNVDriver* drv, CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif /* CML_GPU_NV_DRIVER_H */
