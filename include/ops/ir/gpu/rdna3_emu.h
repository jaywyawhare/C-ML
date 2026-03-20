#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RDNA3_NUM_VGPR  256
#define RDNA3_NUM_SGPR  106
#define RDNA3_WAVE_SIZE  32
#define RDNA3_LDS_SIZE   65536
#define RDNA3_MEM_SIZE   (1ULL << 32)

typedef struct CMLRDNA3Wave {
    uint32_t vgpr[RDNA3_NUM_VGPR][RDNA3_WAVE_SIZE];
    uint32_t sgpr[RDNA3_NUM_SGPR];
    uint64_t exec;
    uint64_t vcc;
    uint32_t pc;
    uint32_t scc;
    int      wave_id;
    bool     halted;
} CMLRDNA3Wave;

typedef struct CMLRDNA3Emu {
    uint8_t* memory;
    size_t   mem_size;
    uint8_t* lds;
    size_t   lds_size;
    CMLRDNA3Wave* waves;
    int      num_waves;
    int      wave_capacity;
    uint64_t cycle_count;
    int      instructions_executed;
} CMLRDNA3Emu;

CMLRDNA3Emu* cml_rdna3_emu_create(size_t mem_size);
void cml_rdna3_emu_free(CMLRDNA3Emu* emu);

int cml_rdna3_emu_load(CMLRDNA3Emu* emu, const void* binary, size_t size, uint64_t addr);

int cml_rdna3_emu_dispatch(CMLRDNA3Emu* emu, uint32_t grid[3], uint32_t block[3],
                           uint64_t program_addr, uint64_t kernarg_addr);

int cml_rdna3_emu_run(CMLRDNA3Emu* emu, int max_cycles);
int cml_rdna3_emu_step(CMLRDNA3Emu* emu);

int cml_rdna3_emu_write(CMLRDNA3Emu* emu, uint64_t addr, const void* data, size_t size);
int cml_rdna3_emu_read(CMLRDNA3Emu* emu, uint64_t addr, void* data, size_t size);

#ifdef __cplusplus
}
#endif
