#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLMemoryPlan {
    int num_buffers;
    size_t* buffer_sizes;
    size_t* buffer_offsets;
    int* buffer_reuse_map;     /* buffer_reuse_map[i] = j means i reuses j's slot (-1 = own slot) */
    int* buffer_first_use;
    int* buffer_last_use;
    size_t total_memory;
    size_t peak_memory;
    size_t saved_memory;
} CMLMemoryPlan;

CMLMemoryPlan* cml_memory_plan_create(int num_buffers, size_t* sizes,
                                       int* first_use, int* last_use);
void cml_memory_plan_free(CMLMemoryPlan* plan);
void cml_memory_plan_print(const CMLMemoryPlan* plan);

#ifdef __cplusplus
}
#endif
