/*
 * AMDGPU Kernel Descriptor (KD) -- 64-byte structure at the start of
 * each kernel's code in the .text section of an AMDGPU ELF code object.
 *
 * Ref: LLVM AMDGPU Backend, AMD GCN/RDNA ISA manuals.
 */

#ifndef CML_GPU_AMDGPU_KD_H
#define CML_GPU_AMDGPU_KD_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct __attribute__((packed)) AMDGPUKernelDescriptor {
    uint32_t group_segment_fixed_size;
    uint32_t private_segment_fixed_size;
    uint32_t kernarg_size;
    uint8_t  reserved0[4];
    int64_t  kernel_code_entry_byte_offset;
    uint8_t  reserved1[20];
    uint32_t compute_pgm_rsrc3;
    uint32_t compute_pgm_rsrc1;
    uint32_t compute_pgm_rsrc2;
    uint16_t kernel_code_properties;
    uint8_t  reserved2[6];
} AMDGPUKernelDescriptor;

/* compute_pgm_rsrc1 field extraction */
#define AMDGPU_RSRC1_VGPRS(rsrc1)      (((rsrc1) & 0x3F) + 1)
#define AMDGPU_RSRC1_SGPRS(rsrc1)      ((((rsrc1) >> 6) & 0xF) + 1)
#define AMDGPU_RSRC1_FLOAT_MODE(rsrc1) (((rsrc1) >> 16) & 0xFF)
#define AMDGPU_RSRC1_PRIV(rsrc1)       (((rsrc1) >> 24) & 0x1)
#define AMDGPU_RSRC1_DX10_CLAMP(rsrc1) (((rsrc1) >> 25) & 0x1)
#define AMDGPU_RSRC1_IEEE_MODE(rsrc1)  (((rsrc1) >> 26) & 0x1)

/* VGPR/SGPR granularity by architecture generation */
#define AMDGPU_VGPR_GRANULARITY_GFX9   4
#define AMDGPU_VGPR_GRANULARITY_GFX10  8
#define AMDGPU_SGPR_GRANULARITY        8

static inline uint32_t amdgpu_vgpr_count(uint32_t rsrc1, bool is_gfx10_plus) {
    uint32_t granularity = is_gfx10_plus ? AMDGPU_VGPR_GRANULARITY_GFX10
                                         : AMDGPU_VGPR_GRANULARITY_GFX9;
    return AMDGPU_RSRC1_VGPRS(rsrc1) * granularity;
}

static inline uint32_t amdgpu_sgpr_count(uint32_t rsrc1) {
    return AMDGPU_RSRC1_SGPRS(rsrc1) * AMDGPU_SGPR_GRANULARITY;
}

/* compute_pgm_rsrc2 field extraction */
#define AMDGPU_RSRC2_SCRATCH_EN(rsrc2)     ((rsrc2) & 0x1)
#define AMDGPU_RSRC2_USER_SGPR(rsrc2)      (((rsrc2) >> 1) & 0x1F)
#define AMDGPU_RSRC2_TRAP_PRESENT(rsrc2)   (((rsrc2) >> 6) & 0x1)
#define AMDGPU_RSRC2_TGID_X_EN(rsrc2)      (((rsrc2) >> 7) & 0x1)
#define AMDGPU_RSRC2_TGID_Y_EN(rsrc2)      (((rsrc2) >> 8) & 0x1)
#define AMDGPU_RSRC2_TGID_Z_EN(rsrc2)      (((rsrc2) >> 9) & 0x1)
#define AMDGPU_RSRC2_LDS_SIZE(rsrc2)       (((rsrc2) >> 15) & 0x1FF)

/* kernel_code_properties flags */
#define AMDGPU_KCP_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER   (1 << 0)
#define AMDGPU_KCP_ENABLE_SGPR_DISPATCH_PTR             (1 << 1)
#define AMDGPU_KCP_ENABLE_SGPR_QUEUE_PTR                (1 << 2)
#define AMDGPU_KCP_ENABLE_SGPR_KERNARG_SEGMENT_PTR      (1 << 3)
#define AMDGPU_KCP_ENABLE_SGPR_DISPATCH_ID              (1 << 4)
#define AMDGPU_KCP_ENABLE_SGPR_FLAT_SCRATCH_INIT        (1 << 5)
#define AMDGPU_KCP_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE     (1 << 6)

int am_parse_kernel_descriptor(const void* code_object, size_t code_size,
                               const char* kernel_name, AMDGPUKernelDescriptor* kd);

#ifdef __cplusplus
}
#endif

#endif /* CML_GPU_AMDGPU_KD_H */
