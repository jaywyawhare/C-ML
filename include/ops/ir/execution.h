#ifndef CML_OPS_IR_EXECUTION_H
#define CML_OPS_IR_EXECUTION_H

#include "ops/ir/ir.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int cml_ir_execute(CMLGraph_t ir);

/* Simple scalar interpreter used as a fallback when other backends
   are not available or fail. */
int cpu_execute_ir(CMLGraph_t ir);

/* Executes up to and including the target node (for lazy evaluation). */
int cml_ir_execute_up_to(CMLGraph_t ir, struct IRNode* target_node);

void cml_print_exec_stats(void);
void cml_reset_exec_stats(void);

/* Uses cached buffers when available, otherwise allocates new memory.
   Buffers are allocated in power-of-2 size buckets for efficient reuse. */
void* cml_buffer_cache_alloc(size_t size);

/* Returns buffer to cache for future reuse. If cache is full, frees memory. */
void cml_buffer_cache_free(void* ptr, size_t size);

void cml_print_buffer_cache_stats(void);
void cml_cleanup_buffer_cache(void);

/* On first call: records kernel launch sequence.
   On subsequent calls with same graph structure: replays without re-scheduling. */
int cml_ir_execute_traced(CMLGraph_t ir);

int cml_ir_execute_v2(CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_EXECUTION_H
