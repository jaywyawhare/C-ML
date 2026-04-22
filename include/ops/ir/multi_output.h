#pragma once
#include "ops/ir/ir.h"
#include "ops/ir/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

int cml_multi_output_analyze(CMLFusionSchedule* sched, int** merge_groups, int* num_merges);
int cml_multi_output_fuse(CMLFusionSchedule* sched, int* merge_groups, int num_merges);

#ifdef __cplusplus
}
#endif
