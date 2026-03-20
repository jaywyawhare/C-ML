#pragma once
#include "ops/ir/ir.h"
#include "ops/ir/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

int cml_multi_output_analyze(CMLScheduleV2* sched, int** merge_groups, int* num_merges);
int cml_multi_output_fuse(CMLScheduleV2* sched, int* merge_groups, int num_merges);

#ifdef __cplusplus
}
#endif
