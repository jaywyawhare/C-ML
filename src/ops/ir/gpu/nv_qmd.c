#include "ops/ir/gpu/nv_qmd.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/*
 * QMD field manipulation helpers.
 * Fields are packed into 32-bit words at specific bit offsets.
 */

static void qmd_set_field(uint32_t *data, int word, int lo, int hi, uint32_t val) {
    uint32_t mask = ((1U << (hi - lo + 1)) - 1) << lo;
    data[word] = (data[word] & ~mask) | ((val << lo) & mask);
}

static void qmd_set_word(uint32_t *data, int word, uint32_t val) {
    data[word] = val;
}

/*
 * QMD v03_00 (Turing/Ampere) layout:
 *   word 0: [1:0] outer_put_api_call_mode, [7:4] qmd_version, [13:8] qmd_major_version
 *   word 1: [5:0] qmd_group_id, [6] sm_global_caching, [8:7] api_visible_call_limit
 *   word 2: sampler_index, inner_put
 *   word 3: shared_memory_layout config
 *   word 8: program_address_lower
 *   word 9: program_address_upper
 *   word 12: grid_dim_x
 *   word 13: grid_dim_y
 *   word 14: grid_dim_z
 *   word 15: [15:0] block_dim_x
 *   word 16: [15:0] block_dim_y
 *   word 17: [15:0] block_dim_z
 *   word 18: [7:0] register_count
 *   word 19: shared_memory_size (bytes, aligned to 256)
 *   word 20: [5:0] barrier_count
 *   word 22: constant_buffer_valid (bitmask)
 *   word 24: cb0_address_lower
 *   word 25: cb0_address_upper
 *   word 26: [16:0] cb0_size
 *
 * QMD v04_00 (Hopper) has mostly the same layout with added fields:
 *   word 0: [7:4] qmd_version = 4
 *   word 30-31: additional SM config
 *   Grid/block/program/CB layout preserved from v03.
 */

/* Turing/Ampere (QMD v03_00) */
#define QMD3_OUTER_PUT_MODE_WORD      0
#define QMD3_OUTER_PUT_MODE_LO        0
#define QMD3_OUTER_PUT_MODE_HI        1
#define QMD3_VERSION_WORD             0
#define QMD3_VERSION_LO               4
#define QMD3_VERSION_HI               7
#define QMD3_MAJOR_VERSION_WORD       0
#define QMD3_MAJOR_VERSION_LO         8
#define QMD3_MAJOR_VERSION_HI         13

#define QMD3_GROUP_ID_WORD            1
#define QMD3_GROUP_ID_LO              0
#define QMD3_GROUP_ID_HI              5
#define QMD3_SM_GLOBAL_CACHING_WORD   1
#define QMD3_SM_GLOBAL_CACHING_LO     6
#define QMD3_SM_GLOBAL_CACHING_HI     6
#define QMD3_API_CALL_LIMIT_WORD      1
#define QMD3_API_CALL_LIMIT_LO        7
#define QMD3_API_CALL_LIMIT_HI        8

#define QMD3_PROGRAM_ADDR_LO_WORD     8
#define QMD3_PROGRAM_ADDR_HI_WORD     9

#define QMD3_GRID_DIM_X_WORD          12
#define QMD3_GRID_DIM_Y_WORD          13
#define QMD3_GRID_DIM_Z_WORD          14

#define QMD3_BLOCK_DIM_X_WORD         15
#define QMD3_BLOCK_DIM_X_LO           0
#define QMD3_BLOCK_DIM_X_HI           15
#define QMD3_BLOCK_DIM_Y_WORD         16
#define QMD3_BLOCK_DIM_Y_LO           0
#define QMD3_BLOCK_DIM_Y_HI           15
#define QMD3_BLOCK_DIM_Z_WORD         17
#define QMD3_BLOCK_DIM_Z_LO           0
#define QMD3_BLOCK_DIM_Z_HI           15

#define QMD3_REGISTER_COUNT_WORD      18
#define QMD3_REGISTER_COUNT_LO        0
#define QMD3_REGISTER_COUNT_HI        7

#define QMD3_SHARED_MEM_SIZE_WORD     19

#define QMD3_BARRIER_COUNT_WORD       20
#define QMD3_BARRIER_COUNT_LO         0
#define QMD3_BARRIER_COUNT_HI         5

#define QMD3_CB_VALID_WORD            22

#define QMD3_SASS_VERSION_WORD        4
#define QMD3_SASS_MAJOR_LO            0
#define QMD3_SASS_MAJOR_HI            3
#define QMD3_SASS_MINOR_LO            4
#define QMD3_SASS_MINOR_HI            7

/*
 * Constant buffer descriptors start at word 24.
 * Each CB uses 3 words: addr_lo, addr_hi, size.
 * CB0 = words 24-26, CB1 = words 27-29, etc.
 */
#define QMD3_CB_BASE_WORD             24
#define QMD3_CB_STRIDE                3

/*
 * QMD v05_00 (Blackwell) layout differences from v04:
 *   word 0: [7:4] qmd_version = 5
 *   word 30: sm_disable_mask_lower
 *   word 31: sm_disable_mask_upper
 *   word 32: min_sm_config_shared_mem_size
 *   word 33: max_sm_config_shared_mem_size
 *   Grid/block/program/CB layout preserved from v03/v04.
 */
#define QMD5_SM_DISABLE_MASK_LO_WORD  30
#define QMD5_SM_DISABLE_MASK_HI_WORD  31
#define QMD5_MIN_SM_SMEM_WORD         32
#define QMD5_MAX_SM_SMEM_WORD         33

void nv_qmd_init(NVQmd *qmd, uint32_t arch) {
    memset(qmd, 0, NV_QMD_BYTES);

    uint32_t version = QMD_VERSION_TURING;
    if (arch >= NV_GPU_ARCH_BLACKWELL)
        version = QMD_VERSION_BLACKWELL;
    else if (arch >= NV_GPU_ARCH_HOPPER)
        version = QMD_VERSION_HOPPER;

    qmd_set_field(qmd->data, QMD3_OUTER_PUT_MODE_WORD,
                  QMD3_OUTER_PUT_MODE_LO, QMD3_OUTER_PUT_MODE_HI, 0x1);

    qmd_set_field(qmd->data, QMD3_VERSION_WORD,
                  QMD3_VERSION_LO, QMD3_VERSION_HI, version);

    qmd_set_field(qmd->data, QMD3_MAJOR_VERSION_WORD,
                  QMD3_MAJOR_VERSION_LO, QMD3_MAJOR_VERSION_HI, 0);

    qmd_set_field(qmd->data, QMD3_API_CALL_LIMIT_WORD,
                  QMD3_API_CALL_LIMIT_LO, QMD3_API_CALL_LIMIT_HI, 0x1);

    qmd_set_field(qmd->data, QMD3_SM_GLOBAL_CACHING_WORD,
                  QMD3_SM_GLOBAL_CACHING_LO, QMD3_SM_GLOBAL_CACHING_HI, 1);

    if (version == QMD_VERSION_BLACKWELL) {
        qmd_set_word(qmd->data, QMD5_SM_DISABLE_MASK_LO_WORD, 0);
        qmd_set_word(qmd->data, QMD5_SM_DISABLE_MASK_HI_WORD, 0);
        qmd_set_word(qmd->data, QMD5_MIN_SM_SMEM_WORD, 0);
        qmd_set_word(qmd->data, QMD5_MAX_SM_SMEM_WORD, 0);
    }

    nv_qmd_set_grid_dim(qmd, 1, 1, 1);
    nv_qmd_set_block_dim(qmd, 1, 1, 1);
    nv_qmd_set_register_count(qmd, 16);
    nv_qmd_set_barrier_count(qmd, 1);
}

void nv_qmd_set_program_address(NVQmd *qmd, uint64_t addr) {
    qmd_set_word(qmd->data, QMD3_PROGRAM_ADDR_LO_WORD, (uint32_t)(addr & 0xFFFFFFFF));
    qmd_set_word(qmd->data, QMD3_PROGRAM_ADDR_HI_WORD, (uint32_t)(addr >> 32));
}

void nv_qmd_set_grid_dim(NVQmd *qmd, uint32_t x, uint32_t y, uint32_t z) {
    qmd_set_word(qmd->data, QMD3_GRID_DIM_X_WORD, x);
    qmd_set_word(qmd->data, QMD3_GRID_DIM_Y_WORD, y);
    qmd_set_word(qmd->data, QMD3_GRID_DIM_Z_WORD, z);
}

void nv_qmd_set_block_dim(NVQmd *qmd, uint32_t x, uint32_t y, uint32_t z) {
    qmd_set_field(qmd->data, QMD3_BLOCK_DIM_X_WORD,
                  QMD3_BLOCK_DIM_X_LO, QMD3_BLOCK_DIM_X_HI, x);
    qmd_set_field(qmd->data, QMD3_BLOCK_DIM_Y_WORD,
                  QMD3_BLOCK_DIM_Y_LO, QMD3_BLOCK_DIM_Y_HI, y);
    qmd_set_field(qmd->data, QMD3_BLOCK_DIM_Z_WORD,
                  QMD3_BLOCK_DIM_Z_LO, QMD3_BLOCK_DIM_Z_HI, z);
}

void nv_qmd_set_shared_memory(NVQmd *qmd, uint32_t bytes) {
    uint32_t aligned = (bytes + 255) & ~255U;
    qmd_set_word(qmd->data, QMD3_SHARED_MEM_SIZE_WORD, aligned);
}

void nv_qmd_set_constant_buffer(NVQmd *qmd, int index, uint64_t addr, uint32_t size) {
    if (index < 0 || index > 7) return;

    qmd->data[QMD3_CB_VALID_WORD] |= (1U << index);

    int base = QMD3_CB_BASE_WORD + index * QMD3_CB_STRIDE;
    if (base + 2 >= NV_QMD_DWORDS) return;

    qmd_set_word(qmd->data, base, (uint32_t)(addr & 0xFFFFFFFF));
    qmd_set_word(qmd->data, base + 1, (uint32_t)(addr >> 32));
    qmd_set_word(qmd->data, base + 2, size);
}

void nv_qmd_set_register_count(NVQmd *qmd, uint32_t count) {
    if (count > 255) count = 255;
    qmd_set_field(qmd->data, QMD3_REGISTER_COUNT_WORD,
                  QMD3_REGISTER_COUNT_LO, QMD3_REGISTER_COUNT_HI, count);
}

void nv_qmd_set_barrier_count(NVQmd *qmd, uint32_t count) {
    if (count > 31) count = 31;
    qmd_set_field(qmd->data, QMD3_BARRIER_COUNT_WORD,
                  QMD3_BARRIER_COUNT_LO, QMD3_BARRIER_COUNT_HI, count);
}

void nv_qmd_set_sass_version(NVQmd *qmd, uint32_t major, uint32_t minor) {
    qmd_set_field(qmd->data, QMD3_SASS_VERSION_WORD,
                  QMD3_SASS_MAJOR_LO, QMD3_SASS_MAJOR_HI, major);
    qmd_set_field(qmd->data, QMD3_SASS_VERSION_WORD,
                  QMD3_SASS_MINOR_LO, QMD3_SASS_MINOR_HI, minor);
}

/*
 * Minimal ELF parsing to extract kernel metadata from CUBIN files.
 * CUBIN is an ELF file with NVIDIA-specific sections:
 *   .text.<name>          - kernel machine code
 *   .nv.info.<name>       - kernel attributes (regs, smem, params)
 *   .nv.constant0.<name>  - constant buffer data
 */

#define ELF_MAGIC 0x464C457F

typedef struct {
    uint8_t  e_ident[16];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
} Elf64_Ehdr_t;

typedef struct {
    uint32_t sh_name;
    uint32_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addr;
    uint64_t sh_offset;
    uint64_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint64_t sh_addralign;
    uint64_t sh_entsize;
} Elf64_Shdr_t;

/*
 * NV info attribute types found in .nv.info.<kernel> sections.
 * These are 12-byte records: { uint32_t format; uint32_t attr; uint32_t value; }
 */
#define EIATTR_REGCOUNT          0x1B
#define EIATTR_MIN_STACK_SIZE    0x12
#define EIATTR_FRAME_SIZE        0x11
#define EIATTR_MAX_THREADS       0x05
#define EIATTR_PARAM_CBANK       0x0A
#define EIATTR_CBANK_PARAM_SIZE  0x19
#define EIATTR_KPARAM_INFO       0x17
#define EIATTR_S2RCTAID_INSTR_OFFSETS 0x1D
#define EIATTR_EXIT_INSTR_OFFSETS     0x1C
#define EIATTR_CRS_STACK_SIZE    0x1E
#define EIATTR_REQNTID           0x10
#define EIATTR_SMEM_SIZE         0x0F
#define EIATTR_BAR_COUNT         0x1F

static const char *elf_get_shstrtab(const uint8_t *data, size_t size,
                                    const Elf64_Ehdr_t *ehdr) {
    if (ehdr->e_shstrndx == 0 || ehdr->e_shstrndx >= ehdr->e_shnum)
        return NULL;

    uint64_t off = ehdr->e_shoff + (uint64_t)ehdr->e_shstrndx * ehdr->e_shentsize;
    if (off + sizeof(Elf64_Shdr_t) > size) return NULL;

    const Elf64_Shdr_t *shstr = (const Elf64_Shdr_t *)(data + off);
    if (shstr->sh_offset + shstr->sh_size > size) return NULL;

    return (const char *)(data + shstr->sh_offset);
}

static const Elf64_Shdr_t *elf_find_section(const uint8_t *data, size_t size,
                                             const Elf64_Ehdr_t *ehdr,
                                             const char *shstrtab,
                                             const char *name) {
    for (int i = 0; i < ehdr->e_shnum; i++) {
        uint64_t off = ehdr->e_shoff + (uint64_t)i * ehdr->e_shentsize;
        if (off + sizeof(Elf64_Shdr_t) > size) continue;

        const Elf64_Shdr_t *sh = (const Elf64_Shdr_t *)(data + off);
        const char *sname = shstrtab + sh->sh_name;
        if (strcmp(sname, name) == 0)
            return sh;
    }
    return NULL;
}

int nv_parse_cubin(const void *cubin, size_t size, const char *kernel_name,
                   NVKernelMeta *meta) {
    if (!cubin || size < sizeof(Elf64_Ehdr_t) || !kernel_name || !meta)
        return -1;

    const uint8_t *data = (const uint8_t *)cubin;
    const Elf64_Ehdr_t *ehdr = (const Elf64_Ehdr_t *)data;

    uint32_t magic;
    memcpy(&magic, data, 4);
    if (magic != ELF_MAGIC)
        return -1;

    memset(meta, 0, sizeof(*meta));

    const char *shstrtab = elf_get_shstrtab(data, size, ehdr);
    if (!shstrtab) return -1;

    char text_name[512];
    char info_name[512];
    snprintf(text_name, sizeof(text_name), ".text.%s", kernel_name);
    snprintf(info_name, sizeof(info_name), ".nv.info.%s", kernel_name);

    const Elf64_Shdr_t *text_sh = elf_find_section(data, size, ehdr, shstrtab, text_name);
    if (!text_sh) {
        text_sh = elf_find_section(data, size, ehdr, shstrtab, ".text");
    }

    if (text_sh) {
        meta->code_offset = text_sh->sh_offset;
        meta->code_size = (uint32_t)text_sh->sh_size;
    }

    const Elf64_Shdr_t *info_sh = elf_find_section(data, size, ehdr, shstrtab, info_name);
    if (!info_sh) {
        info_sh = elf_find_section(data, size, ehdr, shstrtab, ".nv.info");
    }

    if (info_sh && info_sh->sh_offset + info_sh->sh_size <= size) {
        const uint8_t *attr_data = data + info_sh->sh_offset;
        size_t attr_size = (size_t)info_sh->sh_size;
        size_t pos = 0;

        while (pos + 12 <= attr_size) {
            uint32_t format, attr, value;
            memcpy(&format, attr_data + pos, 4);
            memcpy(&attr, attr_data + pos + 4, 4);
            memcpy(&value, attr_data + pos + 8, 4);

            uint32_t attr_id = attr & 0xFFFF;

            switch (attr_id) {
            case EIATTR_REGCOUNT:
                meta->num_registers = value;
                break;
            case EIATTR_SMEM_SIZE:
                meta->shared_mem_size = value;
                break;
            case EIATTR_CBANK_PARAM_SIZE:
                meta->param_size = value;
                break;
            case EIATTR_KPARAM_INFO:
                meta->num_params++;
                break;
            case EIATTR_BAR_COUNT:
                meta->bar_count = value;
                break;
            default:
                break;
            }

            uint32_t entry_size = 12;
            if ((format >> 16) == 0x0004) {
                uint32_t extra = value;
                entry_size = 8 + ((extra + 3) & ~3U);
            }
            pos += entry_size;
        }
    }

    if (meta->num_registers == 0)
        meta->num_registers = 16;
    if (meta->bar_count == 0)
        meta->bar_count = 1;

    return 0;
}
