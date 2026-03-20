#ifndef CML_HEXAGON_SIM_H
#define CML_HEXAGON_SIM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLHexagonSim {
    uint8_t* memory;
    size_t mem_size;
    uint32_t regs[32];
    uint32_t pred_regs[4];
    uint64_t hvx_regs[32][16];
    uint64_t pc;
    bool running;
    int cycle_count;
} CMLHexagonSim;

CMLHexagonSim* cml_hexagon_sim_create(size_t mem_size);
void cml_hexagon_sim_free(CMLHexagonSim* sim);

int cml_hexagon_sim_load(CMLHexagonSim* sim, const void* program, size_t size, uint64_t load_addr);

int cml_hexagon_sim_run(CMLHexagonSim* sim, int max_cycles);
int cml_hexagon_sim_step(CMLHexagonSim* sim);

int cml_hexagon_sim_write(CMLHexagonSim* sim, uint64_t addr, const void* data, size_t size);
int cml_hexagon_sim_read(CMLHexagonSim* sim, uint64_t addr, void* data, size_t size);

int cml_hexagon_sim_hvx_vadd(CMLHexagonSim* sim, int vd, int vs, int vt, int elem_size);
int cml_hexagon_sim_hvx_vmpy(CMLHexagonSim* sim, int vd, int vs, int vt, int elem_size);
int cml_hexagon_sim_hvx_vmem_load(CMLHexagonSim* sim, int vd, uint64_t addr);
int cml_hexagon_sim_hvx_vmem_store(CMLHexagonSim* sim, int vs, uint64_t addr);

#ifdef __cplusplus
}
#endif

#endif /* CML_HEXAGON_SIM_H */
