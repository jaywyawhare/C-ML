#include "ops/ir/gpu/rdna3_emu.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── RDNA3 encoding format constants ────────────────────────────────────── */

/* SOP2: [31:30]=10, [29:23]=opcode, [22:16]=sdst, [15:8]=ssrc1, [7:0]=ssrc0 */
#define ENC_SOP2_MASK   0xC0000000
#define ENC_SOP2_TAG    0x80000000

/* SOP1: [31:23]=101111101, [22:16]=sdst, [15:8]=opcode, [7:0]=ssrc0 */
#define ENC_SOP1_MASK   0xFF800000
#define ENC_SOP1_TAG    0xBE800000

/* SOPC: [31:23]=101111110, [22:16]=unused, [15:8]=ssrc1, [7:0]=ssrc0, opcode in [22:16] */
#define ENC_SOPC_MASK   0xFF800000
#define ENC_SOPC_TAG    0xBF000000

/* SOPP: [31:16]=10111111_1xxxxxxx => top 9 bits = 101111111 */
#define ENC_SOPP_MASK   0xFF800000
#define ENC_SOPP_TAG    0xBF800000

/* VOP2: [31]=0, [30:25]=opcode, [24:17]=vdst, [16:9]=vsrc1(vgpr), [8:0]=src0 */
#define ENC_VOP2_MASK   0x80000000
#define ENC_VOP2_TAG    0x00000000

/* VOP1: [31:25]=0111111, [24:17]=vdst, [16:9]=opcode, [8:0]=src0 */
#define ENC_VOP1_MASK   0xFE000000
#define ENC_VOP1_TAG    0x3E000000

/* VOP3: 64-bit, top word [31:26]=110100 */
#define ENC_VOP3_MASK   0xFC000000
#define ENC_VOP3_TAG    0xD4000000

/* SMEM: 64-bit, top word [31:26]=111101 */
#define ENC_SMEM_MASK   0xFC000000
#define ENC_SMEM_TAG    0xF4000000

/* FLAT/GLOBAL: 64-bit, top word [31:26]=110111 */
#define ENC_FLAT_MASK   0xFC000000
#define ENC_FLAT_TAG    0xDC000000

/* DS (LDS): 64-bit, top word [31:26]=110110 */
#define ENC_DS_MASK     0xFC000000
#define ENC_DS_TAG      0xD8000000

/* ── SOP2 opcodes ──────────────────────────────────────────────────────── */
enum {
    SOP2_S_ADD_U32      = 0,
    SOP2_S_SUB_U32      = 1,
    SOP2_S_MUL_I32      = 2,
    SOP2_S_AND_B32      = 3,
    SOP2_S_OR_B32       = 4,
    SOP2_S_LSHL_B32     = 5,
    SOP2_S_LSHR_B32     = 6,
};

/* ── SOP1 opcodes ──────────────────────────────────────────────────────── */
enum {
    SOP1_S_MOV_B32      = 0,
    SOP1_S_MOV_B64      = 1,
};

/* ── SOPC opcodes ──────────────────────────────────────────────────────── */
enum {
    SOPC_S_CMP_EQ_I32   = 0,
    SOPC_S_CMP_LT_I32   = 1,
    SOPC_S_CMP_GT_I32   = 2,
};

/* ── VOP2 opcodes ──────────────────────────────────────────────────────── */
enum {
    VOP2_V_ADD_F32      = 3,
    VOP2_V_SUB_F32      = 4,
    VOP2_V_MUL_F32      = 8,
    VOP2_V_MAX_F32      = 16,
    VOP2_V_MIN_F32      = 17,
    VOP2_V_ADD_U32      = 25,
    VOP2_V_CNDMASK_B32  = 1,
};

/* ── VOP1 opcodes ──────────────────────────────────────────────────────── */
enum {
    VOP1_V_MOV_B32      = 1,
    VOP1_V_CVT_F32_I32  = 5,
    VOP1_V_CVT_I32_F32  = 8,
};

/* ── VOP3 opcodes ──────────────────────────────────────────────────────── */
enum {
    VOP3_V_FMA_F32      = 0x13,
    VOP3_V_MUL_LO_U32   = 0x69,
};

/* ── SMEM opcodes ──────────────────────────────────────────────────────── */
enum {
    SMEM_S_LOAD_DWORD      = 0,
    SMEM_S_LOAD_DWORDX2    = 1,
    SMEM_S_LOAD_DWORDX4    = 2,
};

/* ── FLAT/GLOBAL opcodes ───────────────────────────────────────────────── */
enum {
    GLOBAL_LOAD_DWORD   = 20,
    GLOBAL_STORE_DWORD  = 28,
};

/* ── DS opcodes ────────────────────────────────────────────────────────── */
enum {
    DS_READ_B32         = 54,
    DS_WRITE_B32        = 13,
};

/* ── SOPP opcodes ──────────────────────────────────────────────────────── */
enum {
    SOPP_S_ENDPGM       = 1,
    SOPP_S_BRANCH        = 2,
    SOPP_S_CBRANCH_SCC0  = 4,
    SOPP_S_CBRANCH_SCC1  = 5,
    SOPP_S_CBRANCH_EXECZ = 9,
};

/* ── Helpers ───────────────────────────────────────────────────────────── */

static inline float u2f(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline uint32_t f2u(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static int mem_check(CMLRDNA3Emu* emu, uint64_t addr, size_t size) {
    if (addr + size > emu->mem_size)
        return -1;
    return 0;
}

static uint32_t mem_read32(CMLRDNA3Emu* emu, uint64_t addr) {
    if (addr + 4 > emu->mem_size) return 0;
    uint32_t v;
    memcpy(&v, emu->memory + addr, 4);
    return v;
}

static void mem_write32(CMLRDNA3Emu* emu, uint64_t addr, uint32_t val) {
    if (addr + 4 > emu->mem_size) return;
    memcpy(emu->memory + addr, &val, 4);
}

static uint32_t lds_read32(CMLRDNA3Emu* emu, uint32_t offset) {
    if (offset + 4 > emu->lds_size) return 0;
    uint32_t v;
    memcpy(&v, emu->lds + offset, 4);
    return v;
}

static void lds_write32(CMLRDNA3Emu* emu, uint32_t offset, uint32_t val) {
    if (offset + 4 > emu->lds_size) return;
    memcpy(emu->lds + offset, &val, 4);
}

static uint32_t read_sgpr(CMLRDNA3Wave* w, int idx) {
    if (idx < 0 || idx >= RDNA3_NUM_SGPR) return 0;
    return w->sgpr[idx];
}

static uint64_t read_sgpr_pair(CMLRDNA3Wave* w, int idx) {
    uint64_t lo = read_sgpr(w, idx);
    uint64_t hi = read_sgpr(w, idx + 1);
    return lo | (hi << 32);
}

/* ── Create / Free ─────────────────────────────────────────────────────── */

CMLRDNA3Emu* cml_rdna3_emu_create(size_t mem_size) {
    CMLRDNA3Emu* emu = calloc(1, sizeof(CMLRDNA3Emu));
    if (!emu) return NULL;

    if (mem_size == 0) mem_size = 64 * 1024 * 1024;

    emu->memory = calloc(1, mem_size);
    if (!emu->memory) { free(emu); return NULL; }
    emu->mem_size = mem_size;

    emu->lds = calloc(1, RDNA3_LDS_SIZE);
    if (!emu->lds) { free(emu->memory); free(emu); return NULL; }
    emu->lds_size = RDNA3_LDS_SIZE;

    emu->wave_capacity = 64;
    emu->waves = calloc(emu->wave_capacity, sizeof(CMLRDNA3Wave));
    if (!emu->waves) { free(emu->lds); free(emu->memory); free(emu); return NULL; }

    return emu;
}

void cml_rdna3_emu_free(CMLRDNA3Emu* emu) {
    if (!emu) return;
    free(emu->waves);
    free(emu->lds);
    free(emu->memory);
    free(emu);
}

int cml_rdna3_emu_load(CMLRDNA3Emu* emu, const void* binary, size_t size, uint64_t addr) {
    if (!emu || !binary) return -1;
    if (mem_check(emu, addr, size) != 0) return -1;
    memcpy(emu->memory + addr, binary, size);
    return 0;
}

int cml_rdna3_emu_write(CMLRDNA3Emu* emu, uint64_t addr, const void* data, size_t size) {
    if (!emu || !data) return -1;
    if (mem_check(emu, addr, size) != 0) return -1;
    memcpy(emu->memory + addr, data, size);
    return 0;
}

int cml_rdna3_emu_read(CMLRDNA3Emu* emu, uint64_t addr, void* data, size_t size) {
    if (!emu || !data) return -1;
    if (mem_check(emu, addr, size) != 0) return -1;
    memcpy(data, emu->memory + addr, size);
    return 0;
}

/* ── Dispatch ──────────────────────────────────────────────────────────── */

int cml_rdna3_emu_dispatch(CMLRDNA3Emu* emu, uint32_t grid[3], uint32_t block[3],
                           uint64_t program_addr, uint64_t kernarg_addr) {
    if (!emu) return -1;

    uint32_t total_groups = grid[0] * grid[1] * grid[2];
    uint32_t threads_per_group = block[0] * block[1] * block[2];
    uint32_t waves_per_group = (threads_per_group + RDNA3_WAVE_SIZE - 1) / RDNA3_WAVE_SIZE;
    uint32_t total_waves = total_groups * waves_per_group;

    if ((int)total_waves > emu->wave_capacity) {
        int new_cap = (int)total_waves + 16;
        CMLRDNA3Wave* tmp = realloc(emu->waves, new_cap * sizeof(CMLRDNA3Wave));
        if (!tmp) return -1;
        emu->waves = tmp;
        emu->wave_capacity = new_cap;
    }

    emu->num_waves = (int)total_waves;
    int wid = 0;

    for (uint32_t gz = 0; gz < grid[2]; gz++) {
        for (uint32_t gy = 0; gy < grid[1]; gy++) {
            for (uint32_t gx = 0; gx < grid[0]; gx++) {
                for (uint32_t wi = 0; wi < waves_per_group; wi++, wid++) {
                    CMLRDNA3Wave* w = &emu->waves[wid];
                    memset(w, 0, sizeof(CMLRDNA3Wave));
                    w->wave_id = wid;
                    w->pc = (uint32_t)program_addr;

                    uint32_t lane_base = wi * RDNA3_WAVE_SIZE;
                    uint32_t active_lanes = threads_per_group - lane_base;
                    if (active_lanes > RDNA3_WAVE_SIZE)
                        active_lanes = RDNA3_WAVE_SIZE;

                    w->exec = ((uint64_t)1 << active_lanes) - 1;

                    /* SGPR setup:
                     * s[0:1] = kernarg_addr
                     * s[2]   = workgroup_id_x
                     * s[3]   = workgroup_id_y
                     * s[4]   = workgroup_id_z */
                    w->sgpr[0] = (uint32_t)(kernarg_addr & 0xFFFFFFFF);
                    w->sgpr[1] = (uint32_t)(kernarg_addr >> 32);
                    w->sgpr[2] = gx;
                    w->sgpr[3] = gy;
                    w->sgpr[4] = gz;

                    /* VGPR v0 = local thread id within workgroup */
                    for (uint32_t lane = 0; lane < active_lanes; lane++) {
                        w->vgpr[0][lane] = lane_base + lane;
                    }
                }
            }
        }
    }

    return 0;
}

/* ── Instruction execution for a single wave ───────────────────────────── */

static int exec_sop2(CMLRDNA3Wave* w, uint32_t inst) {
    int opcode = (inst >> 23) & 0x7F;
    int sdst   = (inst >> 16) & 0x7F;
    int ssrc1  = (inst >>  8) & 0xFF;
    int ssrc0  = inst & 0xFF;

    uint32_t s0 = read_sgpr(w, ssrc0);
    uint32_t s1 = read_sgpr(w, ssrc1);
    uint32_t result = 0;

    switch (opcode) {
    case SOP2_S_ADD_U32: {
        uint64_t r = (uint64_t)s0 + s1;
        result = (uint32_t)r;
        w->scc = (r >> 32) ? 1 : 0;
        break;
    }
    case SOP2_S_SUB_U32: {
        uint64_t r = (uint64_t)s0 - s1;
        result = (uint32_t)r;
        w->scc = (s0 < s1) ? 1 : 0;
        break;
    }
    case SOP2_S_MUL_I32:
        result = (uint32_t)((int32_t)s0 * (int32_t)s1);
        break;
    case SOP2_S_AND_B32:
        result = s0 & s1;
        w->scc = (result != 0) ? 1 : 0;
        break;
    case SOP2_S_OR_B32:
        result = s0 | s1;
        w->scc = (result != 0) ? 1 : 0;
        break;
    case SOP2_S_LSHL_B32:
        result = s0 << (s1 & 0x1F);
        w->scc = (result != 0) ? 1 : 0;
        break;
    case SOP2_S_LSHR_B32:
        result = s0 >> (s1 & 0x1F);
        w->scc = (result != 0) ? 1 : 0;
        break;
    default:
        return -1;
    }

    if (sdst < RDNA3_NUM_SGPR)
        w->sgpr[sdst] = result;
    return 0;
}

static int exec_sop1(CMLRDNA3Wave* w, uint32_t inst) {
    int sdst   = (inst >> 16) & 0x7F;
    int opcode = (inst >>  8) & 0xFF;
    int ssrc0  = inst & 0xFF;

    switch (opcode) {
    case SOP1_S_MOV_B32:
        if (sdst < RDNA3_NUM_SGPR)
            w->sgpr[sdst] = read_sgpr(w, ssrc0);
        break;
    case SOP1_S_MOV_B64: {
        uint64_t val = read_sgpr_pair(w, ssrc0);
        if (sdst < RDNA3_NUM_SGPR - 1) {
            w->sgpr[sdst]     = (uint32_t)(val & 0xFFFFFFFF);
            w->sgpr[sdst + 1] = (uint32_t)(val >> 32);
        }
        break;
    }
    default:
        return -1;
    }
    return 0;
}

static int exec_sopc(CMLRDNA3Wave* w, uint32_t inst) {
    int opcode = (inst >> 16) & 0x7F;
    int ssrc1  = (inst >>  8) & 0xFF;
    int ssrc0  = inst & 0xFF;

    int32_t s0 = (int32_t)read_sgpr(w, ssrc0);
    int32_t s1 = (int32_t)read_sgpr(w, ssrc1);

    switch (opcode) {
    case SOPC_S_CMP_EQ_I32: w->scc = (s0 == s1) ? 1 : 0; break;
    case SOPC_S_CMP_LT_I32: w->scc = (s0 < s1)  ? 1 : 0; break;
    case SOPC_S_CMP_GT_I32: w->scc = (s0 > s1)  ? 1 : 0; break;
    default: return -1;
    }
    return 0;
}

static int exec_sopp(CMLRDNA3Wave* w, uint32_t inst) {
    int opcode  = (inst >> 16) & 0x7F;
    int16_t imm = (int16_t)(inst & 0xFFFF);

    switch (opcode) {
    case SOPP_S_ENDPGM:
        w->halted = true;
        break;
    case SOPP_S_BRANCH:
        w->pc += (int32_t)imm * 4;
        return 1;
    case SOPP_S_CBRANCH_SCC0:
        if (!w->scc) { w->pc += (int32_t)imm * 4; return 1; }
        break;
    case SOPP_S_CBRANCH_SCC1:
        if (w->scc) { w->pc += (int32_t)imm * 4; return 1; }
        break;
    case SOPP_S_CBRANCH_EXECZ:
        if (w->exec == 0) { w->pc += (int32_t)imm * 4; return 1; }
        break;
    default:
        return -1;
    }
    return 0;
}

static int exec_vop2(CMLRDNA3Wave* w, uint32_t inst) {
    int opcode = (inst >> 25) & 0x3F;
    int vdst   = (inst >> 17) & 0xFF;
    int vsrc1  = (inst >>  9) & 0xFF;
    int src0   = inst & 0x1FF;

    for (int lane = 0; lane < RDNA3_WAVE_SIZE; lane++) {
        if (!((w->exec >> lane) & 1)) continue;

        uint32_t s0;
        if (src0 < 256)
            s0 = read_sgpr(w, src0);
        else
            s0 = w->vgpr[src0 - 256][lane];

        uint32_t s1 = w->vgpr[vsrc1][lane];
        uint32_t result = 0;

        switch (opcode) {
        case VOP2_V_CNDMASK_B32:
            result = ((w->vcc >> lane) & 1) ? s1 : s0;
            break;
        case VOP2_V_ADD_F32:
            result = f2u(u2f(s0) + u2f(s1));
            break;
        case VOP2_V_SUB_F32:
            result = f2u(u2f(s0) - u2f(s1));
            break;
        case VOP2_V_MUL_F32:
            result = f2u(u2f(s0) * u2f(s1));
            break;
        case VOP2_V_MAX_F32:
            result = f2u(fmaxf(u2f(s0), u2f(s1)));
            break;
        case VOP2_V_MIN_F32:
            result = f2u(fminf(u2f(s0), u2f(s1)));
            break;
        case VOP2_V_ADD_U32:
            result = s0 + s1;
            break;
        default:
            return -1;
        }

        if (vdst < RDNA3_NUM_VGPR)
            w->vgpr[vdst][lane] = result;
    }
    return 0;
}

static int exec_vop1(CMLRDNA3Wave* w, uint32_t inst) {
    int vdst   = (inst >> 17) & 0xFF;
    int opcode = (inst >>  9) & 0xFF;
    int src0   = inst & 0x1FF;

    for (int lane = 0; lane < RDNA3_WAVE_SIZE; lane++) {
        if (!((w->exec >> lane) & 1)) continue;

        uint32_t s0;
        if (src0 < 256)
            s0 = read_sgpr(w, src0);
        else
            s0 = w->vgpr[src0 - 256][lane];

        uint32_t result = 0;

        switch (opcode) {
        case VOP1_V_MOV_B32:
            result = s0;
            break;
        case VOP1_V_CVT_F32_I32:
            result = f2u((float)(int32_t)s0);
            break;
        case VOP1_V_CVT_I32_F32:
            result = (uint32_t)(int32_t)u2f(s0);
            break;
        default:
            return -1;
        }

        if (vdst < RDNA3_NUM_VGPR)
            w->vgpr[vdst][lane] = result;
    }
    return 0;
}

static int exec_vop3(CMLRDNA3Wave* w, CMLRDNA3Emu* emu, uint32_t hi, uint32_t lo) {
    (void)emu;
    int opcode = (hi >> 16) & 0x3FF;
    int vdst   = hi & 0xFF;
    int src0   = lo & 0x1FF;
    int src1   = (lo >>  9) & 0x1FF;
    int src2   = (lo >> 18) & 0x1FF;

    for (int lane = 0; lane < RDNA3_WAVE_SIZE; lane++) {
        if (!((w->exec >> lane) & 1)) continue;

        uint32_t s0 = (src0 < 256) ? read_sgpr(w, src0) : w->vgpr[src0 - 256][lane];
        uint32_t s1 = (src1 < 256) ? read_sgpr(w, src1) : w->vgpr[src1 - 256][lane];
        uint32_t s2 = (src2 < 256) ? read_sgpr(w, src2) : w->vgpr[src2 - 256][lane];
        uint32_t result = 0;

        switch (opcode) {
        case VOP3_V_FMA_F32:
            result = f2u(fmaf(u2f(s0), u2f(s1), u2f(s2)));
            break;
        case VOP3_V_MUL_LO_U32:
            result = s0 * s1;
            break;
        default:
            return -1;
        }

        if (vdst < RDNA3_NUM_VGPR)
            w->vgpr[vdst][lane] = result;
    }
    return 0;
}

static int exec_smem(CMLRDNA3Wave* w, CMLRDNA3Emu* emu, uint32_t hi, uint32_t lo) {
    int opcode = (hi >> 18) & 0xFF;
    int sdata  = (hi >> 6) & 0x7F;
    int sbase  = hi & 0x3F;
    uint32_t offset = lo & 0x1FFFFF;

    uint64_t base_addr = read_sgpr_pair(w, sbase & ~1);
    uint64_t addr = base_addr + offset;

    switch (opcode) {
    case SMEM_S_LOAD_DWORD:
        if (sdata < RDNA3_NUM_SGPR)
            w->sgpr[sdata] = mem_read32(emu, addr);
        break;
    case SMEM_S_LOAD_DWORDX2:
        for (int i = 0; i < 2 && sdata + i < RDNA3_NUM_SGPR; i++)
            w->sgpr[sdata + i] = mem_read32(emu, addr + i * 4);
        break;
    case SMEM_S_LOAD_DWORDX4:
        for (int i = 0; i < 4 && sdata + i < RDNA3_NUM_SGPR; i++)
            w->sgpr[sdata + i] = mem_read32(emu, addr + i * 4);
        break;
    default:
        return -1;
    }
    return 0;
}

static int exec_flat_global(CMLRDNA3Wave* w, CMLRDNA3Emu* emu, uint32_t hi, uint32_t lo) {
    int opcode = (hi >> 18) & 0xFF;
    int vdst   = lo & 0xFF;
    int vaddr  = (lo >> 8) & 0xFF;
    int vdata  = (lo >> 16) & 0xFF;
    int saddr  = (hi >> 0) & 0x3F;
    int16_t offset = (int16_t)((hi >> 6) & 0xFFF);
    if (offset & 0x800) offset |= 0xF000;

    uint64_t sbase = 0;
    if (saddr != 0x7F)
        sbase = read_sgpr_pair(w, saddr & ~1);

    switch (opcode) {
    case GLOBAL_LOAD_DWORD:
        for (int lane = 0; lane < RDNA3_WAVE_SIZE; lane++) {
            if (!((w->exec >> lane) & 1)) continue;
            uint64_t va = w->vgpr[vaddr][lane];
            if (vaddr + 1 < RDNA3_NUM_VGPR)
                va |= (uint64_t)w->vgpr[vaddr + 1][lane] << 32;
            uint64_t addr = sbase + va + offset;
            if (vdst < RDNA3_NUM_VGPR)
                w->vgpr[vdst][lane] = mem_read32(emu, addr);
        }
        break;
    case GLOBAL_STORE_DWORD:
        for (int lane = 0; lane < RDNA3_WAVE_SIZE; lane++) {
            if (!((w->exec >> lane) & 1)) continue;
            uint64_t va = w->vgpr[vaddr][lane];
            if (vaddr + 1 < RDNA3_NUM_VGPR)
                va |= (uint64_t)w->vgpr[vaddr + 1][lane] << 32;
            uint64_t addr = sbase + va + offset;
            mem_write32(emu, addr, w->vgpr[vdata][lane]);
        }
        break;
    default:
        return -1;
    }
    return 0;
}

static int exec_ds(CMLRDNA3Wave* w, CMLRDNA3Emu* emu, uint32_t hi, uint32_t lo) {
    int opcode = (hi >> 18) & 0xFF;
    int vdst   = lo & 0xFF;
    int addr_v = (lo >> 8) & 0xFF;
    int data0  = (lo >> 16) & 0xFF;
    uint16_t offset = hi & 0xFFFF;

    switch (opcode) {
    case DS_READ_B32:
        for (int lane = 0; lane < RDNA3_WAVE_SIZE; lane++) {
            if (!((w->exec >> lane) & 1)) continue;
            uint32_t a = w->vgpr[addr_v][lane] + offset;
            if (vdst < RDNA3_NUM_VGPR)
                w->vgpr[vdst][lane] = lds_read32(emu, a);
        }
        break;
    case DS_WRITE_B32:
        for (int lane = 0; lane < RDNA3_WAVE_SIZE; lane++) {
            if (!((w->exec >> lane) & 1)) continue;
            uint32_t a = w->vgpr[addr_v][lane] + offset;
            lds_write32(emu, a, w->vgpr[data0][lane]);
        }
        break;
    default:
        return -1;
    }
    return 0;
}

static int exec_wave_instruction(CMLRDNA3Wave* w, CMLRDNA3Emu* emu) {
    if (w->halted) return 0;
    if (w->pc + 4 > emu->mem_size) { w->halted = true; return -1; }

    uint32_t inst = mem_read32(emu, w->pc);
    int advance = 4;

    if ((inst & ENC_SOPP_MASK) == ENC_SOPP_TAG) {
        int r = exec_sopp(w, inst);
        if (r == 1) return 0;
        if (r < 0) return r;
    }
    else if ((inst & ENC_SOP1_MASK) == ENC_SOP1_TAG) {
        int r = exec_sop1(w, inst);
        if (r < 0) return r;
    }
    else if ((inst & ENC_SOPC_MASK) == ENC_SOPC_TAG) {
        int r = exec_sopc(w, inst);
        if (r < 0) return r;
    }
    else if ((inst & ENC_SOP2_MASK) == ENC_SOP2_TAG && (inst >> 30) == 2) {
        int r = exec_sop2(w, inst);
        if (r < 0) return r;
    }
    else if ((inst & ENC_VOP1_MASK) == ENC_VOP1_TAG) {
        int r = exec_vop1(w, inst);
        if (r < 0) return r;
    }
    else if ((inst & ENC_VOP3_MASK) == ENC_VOP3_TAG) {
        uint32_t lo = mem_read32(emu, w->pc + 4);
        advance = 8;
        int r = exec_vop3(w, emu, inst, lo);
        if (r < 0) return r;
    }
    else if ((inst & ENC_SMEM_MASK) == ENC_SMEM_TAG) {
        uint32_t lo = mem_read32(emu, w->pc + 4);
        advance = 8;
        int r = exec_smem(w, emu, inst, lo);
        if (r < 0) return r;
    }
    else if ((inst & ENC_FLAT_MASK) == ENC_FLAT_TAG) {
        uint32_t lo = mem_read32(emu, w->pc + 4);
        advance = 8;
        int r = exec_flat_global(w, emu, inst, lo);
        if (r < 0) return r;
    }
    else if ((inst & ENC_DS_MASK) == ENC_DS_TAG) {
        uint32_t lo = mem_read32(emu, w->pc + 4);
        advance = 8;
        int r = exec_ds(w, emu, inst, lo);
        if (r < 0) return r;
    }
    else if ((inst & ENC_VOP2_MASK) == ENC_VOP2_TAG) {
        int r = exec_vop2(w, inst);
        if (r < 0) return r;
    }
    else {
        w->halted = true;
        return -1;
    }

    w->pc += advance;
    emu->instructions_executed++;
    return 0;
}

/* ── Step / Run ────────────────────────────────────────────────────────── */

int cml_rdna3_emu_step(CMLRDNA3Emu* emu) {
    if (!emu) return -1;

    int active = 0;
    for (int i = 0; i < emu->num_waves; i++) {
        if (emu->waves[i].halted) continue;
        active++;
        exec_wave_instruction(&emu->waves[i], emu);
    }

    emu->cycle_count++;
    return active;
}

int cml_rdna3_emu_run(CMLRDNA3Emu* emu, int max_cycles) {
    if (!emu) return -1;

    for (int c = 0; c < max_cycles; c++) {
        int active = cml_rdna3_emu_step(emu);
        if (active <= 0) return 0;
    }

    return 1;
}
