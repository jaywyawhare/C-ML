#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "ops/ir/gpu/hexagon_sim.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  Testing: %s... ", #name); \
    tests_run++; \
    if (test_##name()) { printf("PASS\n"); tests_passed++; } \
    else { printf("FAIL\n"); } \
} while(0)

/* Instruction encoding helpers */
static uint32_t encode_nop(void) { return 0; }
static uint32_t encode_halt(void) { return 0xF0000000; }

static uint32_t encode_movi(int rd, uint16_t imm) {
    return (0x6 << 28) | ((uint32_t)(rd & 0x1F) << 19) | imm;
}

static uint32_t encode_alu(int subop, int rd, int rs, int rt) {
    return (0x1 << 28) | ((uint32_t)(subop & 0xF) << 24) |
           ((uint32_t)(rd & 0x1F) << 19) | ((uint32_t)(rs & 0x1F) << 14) |
           ((uint32_t)(rt & 0x1F) << 9);
}

static uint32_t encode_store(int rs, int rt, int16_t offset) {
    return (0x3 << 28) | ((uint32_t)(rs & 0x1F) << 14) |
           ((uint32_t)(rt & 0x1F) << 9) | ((uint32_t)offset & 0x1FF);
}

static uint32_t encode_load(int rd, int rs, int16_t offset) {
    return (0x2 << 28) | ((uint32_t)(rd & 0x1F) << 19) |
           ((uint32_t)(rs & 0x1F) << 14) | ((uint32_t)offset & 0x1FF);
}

static bool test_create_free(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(0);
    if (!sim) return false;
    if (sim->mem_size != 1024 * 1024) { cml_hexagon_sim_free(sim); return false; }
    if (sim->running) { cml_hexagon_sim_free(sim); return false; }
    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_create_custom_size(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);
    if (!sim) return false;
    if (sim->mem_size != 4096) { cml_hexagon_sim_free(sim); return false; }
    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_free_null(void) {
    cml_hexagon_sim_free(NULL);
    return true;
}

static bool test_load_program(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);
    if (!sim) return false;

    uint32_t prog[] = { encode_halt() };
    if (cml_hexagon_sim_load(sim, prog, sizeof(prog), 0) != 0) {
        cml_hexagon_sim_free(sim); return false;
    }
    if (sim->pc != 0) { cml_hexagon_sim_free(sim); return false; }
    if (!sim->running) { cml_hexagon_sim_free(sim); return false; }

    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_load_oob(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(64);
    if (!sim) return false;

    uint32_t prog[32];
    if (cml_hexagon_sim_load(sim, prog, sizeof(prog), 0) == 0) {
        cml_hexagon_sim_free(sim); return false;
    }

    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_halt(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);
    uint32_t prog[] = { encode_halt() };
    cml_hexagon_sim_load(sim, prog, sizeof(prog), 0);

    int rc = cml_hexagon_sim_step(sim);
    if (rc != 1) { cml_hexagon_sim_free(sim); return false; }
    if (sim->running) { cml_hexagon_sim_free(sim); return false; }

    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_nop(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);
    uint32_t prog[] = { encode_nop(), encode_halt() };
    cml_hexagon_sim_load(sim, prog, sizeof(prog), 0);

    int rc = cml_hexagon_sim_step(sim);
    if (rc != 0) { cml_hexagon_sim_free(sim); return false; }
    if (sim->pc != 4) { cml_hexagon_sim_free(sim); return false; }

    rc = cml_hexagon_sim_step(sim);
    if (rc != 1) { cml_hexagon_sim_free(sim); return false; }

    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_movi(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);
    uint32_t prog[] = { encode_movi(5, 42), encode_halt() };
    cml_hexagon_sim_load(sim, prog, sizeof(prog), 0);

    cml_hexagon_sim_run(sim, 100);
    if (sim->regs[5] != 42) { cml_hexagon_sim_free(sim); return false; }

    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_alu_add(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);
    uint32_t prog[] = {
        encode_movi(1, 10),
        encode_movi(2, 20),
        encode_alu(0, 3, 1, 2),  /* r3 = r1 + r2 */
        encode_halt()
    };
    cml_hexagon_sim_load(sim, prog, sizeof(prog), 0);
    cml_hexagon_sim_run(sim, 100);

    if (sim->regs[3] != 30) { cml_hexagon_sim_free(sim); return false; }
    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_alu_sub(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);
    uint32_t prog[] = {
        encode_movi(1, 50),
        encode_movi(2, 20),
        encode_alu(1, 3, 1, 2),  /* r3 = r1 - r2 */
        encode_halt()
    };
    cml_hexagon_sim_load(sim, prog, sizeof(prog), 0);
    cml_hexagon_sim_run(sim, 100);

    if (sim->regs[3] != 30) { cml_hexagon_sim_free(sim); return false; }
    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_alu_mul(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);
    uint32_t prog[] = {
        encode_movi(1, 7),
        encode_movi(2, 6),
        encode_alu(7, 3, 1, 2),  /* r3 = r1 * r2 */
        encode_halt()
    };
    cml_hexagon_sim_load(sim, prog, sizeof(prog), 0);
    cml_hexagon_sim_run(sim, 100);

    if (sim->regs[3] != 42) { cml_hexagon_sim_free(sim); return false; }
    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_load_store(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);

    /* Store 0xDEAD at addr 1024, then load it back */
    uint32_t prog[] = {
        encode_movi(1, 1024),      /* r1 = 1024 (base addr) */
        encode_movi(2, 0xDEAD),    /* r2 = value */
        encode_store(1, 2, 0),     /* mem[r1+0] = r2 */
        encode_load(3, 1, 0),      /* r3 = mem[r1+0] */
        encode_halt()
    };
    cml_hexagon_sim_load(sim, prog, sizeof(prog), 0);
    cml_hexagon_sim_run(sim, 100);

    if (sim->regs[3] != 0xDEAD) { cml_hexagon_sim_free(sim); return false; }
    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_memory_read_write(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);

    uint32_t val = 0xCAFEBABE;
    if (cml_hexagon_sim_write(sim, 100, &val, sizeof(val)) != 0) {
        cml_hexagon_sim_free(sim); return false;
    }

    uint32_t out = 0;
    if (cml_hexagon_sim_read(sim, 100, &out, sizeof(out)) != 0) {
        cml_hexagon_sim_free(sim); return false;
    }

    if (out != 0xCAFEBABE) { cml_hexagon_sim_free(sim); return false; }
    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_memory_oob(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(256);
    uint32_t val = 0;
    if (cml_hexagon_sim_write(sim, 300, &val, 4) == 0) {
        cml_hexagon_sim_free(sim); return false;
    }
    if (cml_hexagon_sim_read(sim, 300, &val, 4) == 0) {
        cml_hexagon_sim_free(sim); return false;
    }
    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_hvx_vadd_i32(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);

    int32_t a[32], b[32];
    for (int i = 0; i < 32; i++) { a[i] = i; b[i] = 100 + i; }
    memcpy(sim->hvx_regs[0], a, 128);
    memcpy(sim->hvx_regs[1], b, 128);

    if (cml_hexagon_sim_hvx_vadd(sim, 2, 0, 1, 4) != 0) {
        cml_hexagon_sim_free(sim); return false;
    }

    int32_t result[32];
    memcpy(result, sim->hvx_regs[2], 128);

    for (int i = 0; i < 32; i++) {
        if (result[i] != 100 + 2 * i) { cml_hexagon_sim_free(sim); return false; }
    }

    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_hvx_vmpy_i16(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);

    int16_t a[64], b[64];
    for (int i = 0; i < 64; i++) { a[i] = (int16_t)(i + 1); b[i] = 2; }
    memcpy(sim->hvx_regs[0], a, 128);
    memcpy(sim->hvx_regs[1], b, 128);

    if (cml_hexagon_sim_hvx_vmpy(sim, 2, 0, 1, 2) != 0) {
        cml_hexagon_sim_free(sim); return false;
    }

    int16_t result[64];
    memcpy(result, sim->hvx_regs[2], 128);

    for (int i = 0; i < 64; i++) {
        if (result[i] != (int16_t)(2 * (i + 1))) {
            cml_hexagon_sim_free(sim); return false;
        }
    }

    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_hvx_vmem_load_store(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);

    int32_t data[32];
    for (int i = 0; i < 32; i++) data[i] = i * 10;
    cml_hexagon_sim_write(sim, 512, data, 128);

    if (cml_hexagon_sim_hvx_vmem_load(sim, 5, 512) != 0) {
        cml_hexagon_sim_free(sim); return false;
    }

    if (cml_hexagon_sim_hvx_vmem_store(sim, 5, 1024) != 0) {
        cml_hexagon_sim_free(sim); return false;
    }

    int32_t out[32];
    cml_hexagon_sim_read(sim, 1024, out, 128);

    for (int i = 0; i < 32; i++) {
        if (out[i] != i * 10) { cml_hexagon_sim_free(sim); return false; }
    }

    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_hvx_invalid_reg(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);
    if (cml_hexagon_sim_hvx_vadd(sim, 32, 0, 0, 4) == 0) {
        cml_hexagon_sim_free(sim); return false;
    }
    if (cml_hexagon_sim_hvx_vadd(sim, -1, 0, 0, 4) == 0) {
        cml_hexagon_sim_free(sim); return false;
    }
    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_hvx_invalid_elem_size(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);
    if (cml_hexagon_sim_hvx_vadd(sim, 0, 1, 2, 3) == 0) {
        cml_hexagon_sim_free(sim); return false;
    }
    if (cml_hexagon_sim_hvx_vadd(sim, 0, 1, 2, 8) == 0) {
        cml_hexagon_sim_free(sim); return false;
    }
    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_cycle_counting(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);
    uint32_t prog[] = {
        encode_nop(),
        encode_nop(),
        encode_nop(),
        encode_halt()
    };
    cml_hexagon_sim_load(sim, prog, sizeof(prog), 0);
    cml_hexagon_sim_run(sim, 100);

    if (sim->cycle_count != 4) { cml_hexagon_sim_free(sim); return false; }
    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_run_max_cycles(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);

    /* Infinite loop: NOP then branch back */
    uint32_t prog[] = { encode_nop(), encode_nop(), encode_nop(), encode_nop() };
    /* Fill with NOPs - no halt so it runs until max_cycles */
    cml_hexagon_sim_load(sim, prog, sizeof(prog), 0);

    int rc = cml_hexagon_sim_run(sim, 3);
    if (sim->cycle_count < 3) { cml_hexagon_sim_free(sim); return false; }
    (void)rc;

    cml_hexagon_sim_free(sim);
    return true;
}

static bool test_r0_is_zero(void) {
    CMLHexagonSim* sim = cml_hexagon_sim_create(4096);
    uint32_t prog[] = {
        encode_movi(0, 999),
        encode_halt()
    };
    cml_hexagon_sim_load(sim, prog, sizeof(prog), 0);
    cml_hexagon_sim_run(sim, 100);

    if (sim->regs[0] != 0) { cml_hexagon_sim_free(sim); return false; }
    cml_hexagon_sim_free(sim);
    return true;
}

int main(void) {
    printf("=== Hexagon DSP Simulator Tests ===\n");

    TEST(create_free);
    TEST(create_custom_size);
    TEST(free_null);
    TEST(load_program);
    TEST(load_oob);
    TEST(halt);
    TEST(nop);
    TEST(movi);
    TEST(alu_add);
    TEST(alu_sub);
    TEST(alu_mul);
    TEST(load_store);
    TEST(memory_read_write);
    TEST(memory_oob);
    TEST(hvx_vadd_i32);
    TEST(hvx_vmpy_i16);
    TEST(hvx_vmem_load_store);
    TEST(hvx_invalid_reg);
    TEST(hvx_invalid_elem_size);
    TEST(cycle_counting);
    TEST(run_max_cycles);
    TEST(r0_is_zero);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
