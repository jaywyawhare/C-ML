/*
 * TinyJit -- capture-and-replay JIT for repeated graph execution.
 *
 * On the first call the graph is executed normally and the kernel launch
 * sequence is recorded via the trace infrastructure.  Subsequent calls
 * with the same graph structure (same hash) replay the cached trace,
 * bypassing scheduling and code generation entirely.
 */

#ifndef CML_OPS_IR_TINY_JIT_H
#define CML_OPS_IR_TINY_JIT_H

#include "ops/ir/ir.h"
#include "ops/ir/trace.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CML_JIT_CACHE_SIZE 64

typedef struct {
    uint64_t graph_hash;
    CMLTrace* trace;

    /* Shape signature for invalidation */
    int shape_sig[32];  /* flattened first-node output shape */
    int shape_len;

    bool occupied;
} CMLJitEntry;

typedef struct {
    CMLJitEntry entries[CML_JIT_CACHE_SIZE];
    int count;

    size_t hits;
    size_t misses;
    size_t invalidations;
} CMLTinyJit;

CMLTinyJit* cml_tinyjit_create(void);
void cml_tinyjit_free(CMLTinyJit* jit);

/* First call: execute normally + record trace.
   Subsequent calls: replay cached trace if graph hash + shapes match. */
int cml_tinyjit_execute(CMLTinyJit* jit, CMLGraph_t ir);

void cml_tinyjit_stats(const CMLTinyJit* jit,
                       size_t* hits, size_t* misses, size_t* invalidations);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_TINY_JIT_H */
