/**
 * @file tiny_jit.h
 * @brief TinyJit -- capture-and-replay JIT for repeated graph execution
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

/* ── JIT cache entry ── */

typedef struct {
    uint64_t graph_hash;
    CMLTrace* trace;

    /* Shape signature for invalidation */
    int shape_sig[32];  /* flattened first-node output shape */
    int shape_len;

    bool occupied;
} CMLJitEntry;

/* ── TinyJit context ── */

typedef struct {
    CMLJitEntry entries[CML_JIT_CACHE_SIZE];
    int count;

    /* Statistics */
    size_t hits;
    size_t misses;
    size_t invalidations;
} CMLTinyJit;

/* ── API ── */

/**
 * @brief Create a TinyJit context
 */
CMLTinyJit* cml_tinyjit_create(void);

/**
 * @brief Free a TinyJit context and all cached traces
 */
void cml_tinyjit_free(CMLTinyJit* jit);

/**
 * @brief Execute a graph through the JIT cache
 *
 * First call: execute normally + record trace.
 * Subsequent calls: replay cached trace if graph hash + shapes match.
 *
 * @param jit  TinyJit context
 * @param ir   IR graph to execute
 * @return 0 on success, negative on error
 */
int cml_tinyjit_execute(CMLTinyJit* jit, CMLGraph_t ir);

/**
 * @brief Get JIT statistics
 */
void cml_tinyjit_stats(const CMLTinyJit* jit,
                       size_t* hits, size_t* misses, size_t* invalidations);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_TINY_JIT_H */
