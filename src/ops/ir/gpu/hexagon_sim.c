#include "ops/ir/gpu/hexagon_sim.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>

#define HVX_VECTOR_SIZE 128

/*
 * Instruction encoding (32-bit):
 *   [31:28] opcode class
 *   [27:24] sub-opcode
 *   [23:19] rd/vd (destination register)
 *   [18:14] rs/vs (source register 1)
 *   [13:9]  rt/vt (source register 2)
 *   [8:0]   immediate / flags
 *
 * Opcode classes:
 *   0x0 = NOP
 *   0x1 = ALU (add, sub, and, or, xor, shl, shr)
 *   0x2 = Load (word from memory)
 *   0x3 = Store (word to memory)
 *   0x4 = Branch (conditional on predicate register)
 *   0x5 = Compare (sets predicate register)
 *   0x6 = Move immediate (low 16 bits)
 *   0x7 = HVX vector ops
 *   0x8 = HVX memory load
 *   0x9 = HVX memory store
 *   0xF = HALT
 */

#define OP_NOP      0x0
#define OP_ALU      0x1
#define OP_LOAD     0x2
#define OP_STORE    0x3
#define OP_BRANCH   0x4
#define OP_COMPARE  0x5
#define OP_MOVI     0x6
#define OP_HVX_VEC  0x7
#define OP_HVX_LOAD 0x8
#define OP_HVX_STORE 0x9
#define OP_HALT     0xF

#define ALU_ADD 0x0
#define ALU_SUB 0x1
#define ALU_AND 0x2
#define ALU_OR  0x3
#define ALU_XOR 0x4
#define ALU_SHL 0x5
#define ALU_SHR 0x6
#define ALU_MUL 0x7

#define HVX_VADD 0x0
#define HVX_VMPY 0x1
#define HVX_VSUB 0x2

#define CMP_EQ  0x0
#define CMP_GT  0x1
#define CMP_LT  0x2
#define CMP_NE  0x3

static inline uint32_t decode_opclass(uint32_t insn) { return (insn >> 28) & 0xF; }
static inline uint32_t decode_subop(uint32_t insn)   { return (insn >> 24) & 0xF; }
static inline uint32_t decode_rd(uint32_t insn)      { return (insn >> 19) & 0x1F; }
static inline uint32_t decode_rs(uint32_t insn)      { return (insn >> 14) & 0x1F; }
static inline uint32_t decode_rt(uint32_t insn)      { return (insn >> 9) & 0x1F; }
static inline uint32_t decode_imm9(uint32_t insn)    { return insn & 0x1FF; }
static inline uint16_t decode_imm16(uint32_t insn)   { return (uint16_t)(insn & 0xFFFF); }
static inline int32_t int9_sign_extend(uint32_t v) { return (v & 0x100) ? (int32_t)(v | 0xFFFFFE00) : (int32_t)v; }

static inline int bounds_check(CMLHexagonSim* sim, uint64_t addr, size_t size) {
    if (addr + size > sim->mem_size) {
        LOG_ERROR("Hexagon sim: OOB access at 0x%" PRIx64 " (size=%zu, mem=%zu)",
                  addr, size, sim->mem_size);
        return -1;
    }
    return 0;
}

static inline uint32_t mem_read32(CMLHexagonSim* sim, uint64_t addr) {
    if (bounds_check(sim, addr, 4) != 0) { sim->running = false; return 0; }
    uint32_t v;
    memcpy(&v, sim->memory + addr, 4);
    return v;
}

static inline void mem_write32(CMLHexagonSim* sim, uint64_t addr, uint32_t val) {
    if (bounds_check(sim, addr, 4) != 0) { sim->running = false; return; }
    memcpy(sim->memory + addr, &val, 4);
}

CMLHexagonSim* cml_hexagon_sim_create(size_t mem_size) {
    if (mem_size == 0) mem_size = 1024 * 1024;

    CMLHexagonSim* sim = calloc(1, sizeof(CMLHexagonSim));
    if (!sim) return NULL;

    sim->memory = calloc(1, mem_size);
    if (!sim->memory) {
        free(sim);
        return NULL;
    }

    sim->mem_size = mem_size;
    sim->running = false;
    sim->cycle_count = 0;
    return sim;
}

void cml_hexagon_sim_free(CMLHexagonSim* sim) {
    if (!sim) return;
    free(sim->memory);
    free(sim);
}

int cml_hexagon_sim_load(CMLHexagonSim* sim, const void* program, size_t size, uint64_t load_addr) {
    if (!sim || !program || size == 0) return -1;
    if (bounds_check(sim, load_addr, size) != 0) return -1;

    memcpy(sim->memory + load_addr, program, size);
    sim->pc = load_addr;
    sim->running = true;
    return 0;
}

int cml_hexagon_sim_step(CMLHexagonSim* sim) {
    if (!sim || !sim->running) return -1;

    if (bounds_check(sim, sim->pc, 4) != 0) {
        sim->running = false;
        return -1;
    }

    uint32_t insn = mem_read32(sim, sim->pc);
    uint32_t opclass = decode_opclass(insn);
    uint32_t subop   = decode_subop(insn);
    uint32_t rd      = decode_rd(insn);
    uint32_t rs      = decode_rs(insn);
    uint32_t rt      = decode_rt(insn);

    sim->pc += 4;
    sim->cycle_count++;

    switch (opclass) {
    case OP_NOP:
        break;

    case OP_ALU: {
        uint32_t a = sim->regs[rs];
        uint32_t b = sim->regs[rt];
        uint32_t result = 0;
        switch (subop) {
        case ALU_ADD: result = a + b; break;
        case ALU_SUB: result = a - b; break;
        case ALU_AND: result = a & b; break;
        case ALU_OR:  result = a | b; break;
        case ALU_XOR: result = a ^ b; break;
        case ALU_SHL: result = a << (b & 31); break;
        case ALU_SHR: result = a >> (b & 31); break;
        case ALU_MUL: result = a * b; break;
        default:
            LOG_WARNING("Hexagon sim: unknown ALU subop 0x%x", subop);
            break;
        }
        if (rd != 0) sim->regs[rd] = result;
        break;
    }

    case OP_LOAD: {
        int32_t offset = (int32_t)(int16_t)(int9_sign_extend(decode_imm9(insn)));
        uint64_t addr = (uint64_t)((int64_t)sim->regs[rs] + offset);
        uint32_t val = mem_read32(sim, addr);
        if (rd != 0) sim->regs[rd] = val;
        break;
    }

    case OP_STORE: {
        int32_t offset = (int32_t)(int16_t)(int9_sign_extend(decode_imm9(insn)));
        uint64_t addr = (uint64_t)((int64_t)sim->regs[rs] + offset);
        mem_write32(sim, addr, sim->regs[rt]);
        break;
    }

    case OP_BRANCH: {
        uint32_t pred = rd & 3;
        if (sim->pred_regs[pred]) {
            int32_t offset = int9_sign_extend(decode_imm9(insn));
            sim->pc = (uint64_t)((int64_t)sim->pc + offset - 4);
        }
        break;
    }

    case OP_COMPARE: {
        uint32_t pred = rd & 3;
        uint32_t a = sim->regs[rs];
        uint32_t b = sim->regs[rt];
        switch (subop) {
        case CMP_EQ: sim->pred_regs[pred] = (a == b); break;
        case CMP_GT: sim->pred_regs[pred] = (a > b); break;
        case CMP_LT: sim->pred_regs[pred] = (a < b); break;
        case CMP_NE: sim->pred_regs[pred] = (a != b); break;
        default: break;
        }
        break;
    }

    case OP_MOVI: {
        uint16_t imm = decode_imm16(insn);
        if (rd != 0) sim->regs[rd] = (uint32_t)imm;
        break;
    }

    case OP_HVX_VEC: {
        uint32_t vd = rd;
        uint32_t vs = rs;
        uint32_t vt = rt;
        int elem_size = 1 << (subop & 0x3);
        int hvx_op = (subop >> 2) & 0x3;

        switch (hvx_op) {
        case HVX_VADD:
            cml_hexagon_sim_hvx_vadd(sim, (int)vd, (int)vs, (int)vt, elem_size);
            break;
        case HVX_VMPY:
            cml_hexagon_sim_hvx_vmpy(sim, (int)vd, (int)vs, (int)vt, elem_size);
            break;
        case HVX_VSUB: {
            uint8_t* dst = (uint8_t*)sim->hvx_regs[vd];
            uint8_t* s1  = (uint8_t*)sim->hvx_regs[vs];
            uint8_t* s2  = (uint8_t*)sim->hvx_regs[vt];
            for (int i = 0; i < HVX_VECTOR_SIZE; i += elem_size) {
                if (elem_size == 1) {
                    dst[i] = s1[i] - s2[i];
                } else if (elem_size == 2) {
                    int16_t a, b;
                    memcpy(&a, s1 + i, 2);
                    memcpy(&b, s2 + i, 2);
                    int16_t r = a - b;
                    memcpy(dst + i, &r, 2);
                } else {
                    int32_t a, b;
                    memcpy(&a, s1 + i, 4);
                    memcpy(&b, s2 + i, 4);
                    int32_t r = a - b;
                    memcpy(dst + i, &r, 4);
                }
            }
            break;
        }
        default:
            LOG_WARNING("Hexagon sim: unknown HVX op %d", hvx_op);
            break;
        }
        sim->cycle_count++;
        break;
    }

    case OP_HVX_LOAD: {
        uint64_t addr = (uint64_t)sim->regs[rs];
        cml_hexagon_sim_hvx_vmem_load(sim, (int)rd, addr);
        sim->cycle_count++;
        break;
    }

    case OP_HVX_STORE: {
        uint64_t addr = (uint64_t)sim->regs[rs];
        cml_hexagon_sim_hvx_vmem_store(sim, (int)rd, addr);
        sim->cycle_count++;
        break;
    }

    case OP_HALT:
        sim->running = false;
        break;

    default:
        LOG_WARNING("Hexagon sim: unknown opclass 0x%x at PC=0x%" PRIx64, opclass, sim->pc - 4);
        break;
    }

    return sim->running ? 0 : 1;
}

int cml_hexagon_sim_run(CMLHexagonSim* sim, int max_cycles) {
    if (!sim || !sim->running) return -1;

    int start_cycles = sim->cycle_count;

    while (sim->running && (sim->cycle_count - start_cycles) < max_cycles) {
        int rc = cml_hexagon_sim_step(sim);
        if (rc < 0) return -1;
        if (rc == 1) return 0;
    }

    return sim->running ? 1 : 0;
}

int cml_hexagon_sim_write(CMLHexagonSim* sim, uint64_t addr, const void* data, size_t size) {
    if (!sim || !data) return -1;
    if (bounds_check(sim, addr, size) != 0) return -1;
    memcpy(sim->memory + addr, data, size);
    return 0;
}

int cml_hexagon_sim_read(CMLHexagonSim* sim, uint64_t addr, void* data, size_t size) {
    if (!sim || !data) return -1;
    if (bounds_check(sim, addr, size) != 0) return -1;
    memcpy(data, sim->memory + addr, size);
    return 0;
}

int cml_hexagon_sim_hvx_vadd(CMLHexagonSim* sim, int vd, int vs, int vt, int elem_size) {
    if (!sim || vd < 0 || vd >= 32 || vs < 0 || vs >= 32 || vt < 0 || vt >= 32) return -1;
    if (elem_size != 1 && elem_size != 2 && elem_size != 4) return -1;

    uint8_t* dst = (uint8_t*)sim->hvx_regs[vd];
    uint8_t* s1  = (uint8_t*)sim->hvx_regs[vs];
    uint8_t* s2  = (uint8_t*)sim->hvx_regs[vt];

    for (int i = 0; i < HVX_VECTOR_SIZE; i += elem_size) {
        if (elem_size == 1) {
            dst[i] = s1[i] + s2[i];
        } else if (elem_size == 2) {
            int16_t a, b;
            memcpy(&a, s1 + i, 2);
            memcpy(&b, s2 + i, 2);
            int16_t r = a + b;
            memcpy(dst + i, &r, 2);
        } else {
            int32_t a, b;
            memcpy(&a, s1 + i, 4);
            memcpy(&b, s2 + i, 4);
            int32_t r = a + b;
            memcpy(dst + i, &r, 4);
        }
    }

    return 0;
}

int cml_hexagon_sim_hvx_vmpy(CMLHexagonSim* sim, int vd, int vs, int vt, int elem_size) {
    if (!sim || vd < 0 || vd >= 32 || vs < 0 || vs >= 32 || vt < 0 || vt >= 32) return -1;
    if (elem_size != 1 && elem_size != 2 && elem_size != 4) return -1;

    uint8_t* dst = (uint8_t*)sim->hvx_regs[vd];
    uint8_t* s1  = (uint8_t*)sim->hvx_regs[vs];
    uint8_t* s2  = (uint8_t*)sim->hvx_regs[vt];

    for (int i = 0; i < HVX_VECTOR_SIZE; i += elem_size) {
        if (elem_size == 1) {
            dst[i] = s1[i] * s2[i];
        } else if (elem_size == 2) {
            int16_t a, b;
            memcpy(&a, s1 + i, 2);
            memcpy(&b, s2 + i, 2);
            int16_t r = a * b;
            memcpy(dst + i, &r, 2);
        } else {
            int32_t a, b;
            memcpy(&a, s1 + i, 4);
            memcpy(&b, s2 + i, 4);
            int32_t r = a * b;
            memcpy(dst + i, &r, 4);
        }
    }

    return 0;
}

int cml_hexagon_sim_hvx_vmem_load(CMLHexagonSim* sim, int vd, uint64_t addr) {
    if (!sim || vd < 0 || vd >= 32) return -1;
    if (bounds_check(sim, addr, HVX_VECTOR_SIZE) != 0) return -1;
    memcpy(sim->hvx_regs[vd], sim->memory + addr, HVX_VECTOR_SIZE);
    return 0;
}

int cml_hexagon_sim_hvx_vmem_store(CMLHexagonSim* sim, int vs, uint64_t addr) {
    if (!sim || vs < 0 || vs >= 32) return -1;
    if (bounds_check(sim, addr, HVX_VECTOR_SIZE) != 0) return -1;
    memcpy(sim->memory + addr, sim->hvx_regs[vs], HVX_VECTOR_SIZE);
    return 0;
}
