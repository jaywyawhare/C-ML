#ifndef CML_GPU_NV_QMD_H
#define CML_GPU_NV_QMD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NV_QMD_DWORDS 64
#define NV_QMD_BYTES  (NV_QMD_DWORDS * 4)

#define QMD_VERSION_TURING    0x03
#define QMD_VERSION_AMPERE    0x03
#define QMD_VERSION_HOPPER    0x04
#define QMD_VERSION_BLACKWELL 0x05

#define NV_GPU_ARCH_TURING    0x190
#define NV_GPU_ARCH_AMPERE    0x1A0
#define NV_GPU_ARCH_ADA       0x1B0
#define NV_GPU_ARCH_HOPPER    0x1C0
#define NV_GPU_ARCH_BLACKWELL 0x1D0

typedef struct NVQmd {
    uint32_t data[NV_QMD_DWORDS];
} NVQmd;

void nv_qmd_init(NVQmd *qmd, uint32_t arch);
void nv_qmd_set_program_address(NVQmd *qmd, uint64_t addr);
void nv_qmd_set_grid_dim(NVQmd *qmd, uint32_t x, uint32_t y, uint32_t z);
void nv_qmd_set_block_dim(NVQmd *qmd, uint32_t x, uint32_t y, uint32_t z);
void nv_qmd_set_shared_memory(NVQmd *qmd, uint32_t bytes);
void nv_qmd_set_constant_buffer(NVQmd *qmd, int index, uint64_t addr, uint32_t size);
void nv_qmd_set_register_count(NVQmd *qmd, uint32_t count);
void nv_qmd_set_barrier_count(NVQmd *qmd, uint32_t count);
void nv_qmd_set_sass_version(NVQmd *qmd, uint32_t major, uint32_t minor);

typedef struct NVKernelMeta {
    uint64_t code_offset;
    uint32_t code_size;
    uint32_t num_registers;
    uint32_t shared_mem_size;
    uint32_t param_size;
    uint32_t num_params;
    uint32_t bar_count;
} NVKernelMeta;

int nv_parse_cubin(const void *cubin, size_t size, const char *kernel_name,
                   NVKernelMeta *meta);

#ifdef __cplusplus
}
#endif

#endif /* CML_GPU_NV_QMD_H */
