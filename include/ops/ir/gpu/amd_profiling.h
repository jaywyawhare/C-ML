#ifndef CML_AMD_PROFILING_H
#define CML_AMD_PROFILING_H

#include "ops/ir/gpu/am_driver.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLAMDProfile {
    uint64_t* timestamps;
    uint32_t* wave_counts;
    uint64_t* busy_cycles;
    uint64_t* mem_reads;
    uint64_t* mem_writes;
    int num_entries;
    int capacity;
} CMLAMDProfile;

CMLAMDProfile* cml_amd_profile_create(void);
int cml_amd_profile_start(CMLAMDProfile* prof, CMLAMDriver* drv);
int cml_amd_profile_stop(CMLAMDProfile* prof, CMLAMDriver* drv);
void cml_amd_profile_free(CMLAMDProfile* prof);

int cml_amd_pmc_read(CMLAMDriver* drv, uint32_t counter_id, uint64_t* value);

typedef struct CMLAMDSQTTTrace {
    void* data;
    size_t size;
    int num_waves;
} CMLAMDSQTTTrace;

CMLAMDSQTTTrace* cml_amd_sqtt_capture(CMLAMDriver* drv, int num_dispatches);
void cml_amd_sqtt_free(CMLAMDSQTTTrace* trace);

void cml_amd_profile_print(const CMLAMDProfile* prof);

#ifdef __cplusplus
}
#endif

#endif /* CML_AMD_PROFILING_H */
