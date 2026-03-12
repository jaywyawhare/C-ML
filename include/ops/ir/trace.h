/**
 * @file trace.h
 * @brief Trace-and-Replay JIT
 *
 * Capture kernel launch sequence on first run, replay without
 * re-scheduling on subsequent runs.
 */

#ifndef CML_OPS_IR_TRACE_H
#define CML_OPS_IR_TRACE_H

#include "ops/ir/ir.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CML_TRACE_MAX_ENTRIES 256
#define CML_TRACE_MAX_ARGS    16
#define CML_TRACE_CACHE_SIZE  64

/* ── Trace entry types ── */

typedef enum {
    CML_TRACE_KERNEL = 0,
    CML_TRACE_MEMCPY_H2D,
    CML_TRACE_MEMCPY_D2H,
} CMLTraceEntryType;

/* ── Single trace entry ── */

typedef struct {
    CMLTraceEntryType type;
    uint64_t kernel_hash;
    void* compiled_kernel;       /* cached compiled kernel pointer */

    /* Launch config */
    size_t grid[3];
    size_t block[3];

    /* Argument slots: indices into the trace's tensor slot array */
    int arg_indices[CML_TRACE_MAX_ARGS];
    int num_args;

    /* Memcpy info */
    size_t memcpy_bytes;
    int src_slot;
    int dst_slot;
} CMLTraceEntry;

/* ── Trace: a recorded sequence of operations ── */

typedef struct {
    CMLTraceEntry entries[CML_TRACE_MAX_ENTRIES];
    int num_entries;

    /* Tensor slot mapping (slot index -> live tensor pointer) */
    void* tensor_slots[CML_TRACE_MAX_ENTRIES];
    int num_slots;

    /* State */
    bool is_recording;
    bool is_complete;
    uint64_t graph_hash;    /* hash of the IR graph that produced this trace */
} CMLTrace;

/* ── Trace cache: maps graph hash -> trace ── */

typedef struct {
    struct {
        uint64_t hash;
        CMLTrace* trace;
        bool occupied;
    } entries[CML_TRACE_CACHE_SIZE];
    int count;
} CMLTraceCache;

/* ── API ── */

CMLTrace* cml_trace_create(void);
void cml_trace_free(CMLTrace* trace);

int cml_trace_begin(CMLTrace* trace, uint64_t graph_hash);
int cml_trace_end(CMLTrace* trace);

int cml_trace_record_kernel(CMLTrace* trace, uint64_t kernel_hash,
                            void* compiled_kernel, const size_t grid[3],
                            const size_t block[3], int* arg_indices, int num_args);

int cml_trace_record_memcpy(CMLTrace* trace, CMLTraceEntryType type,
                            int src_slot, int dst_slot, size_t bytes);

/**
 * @brief Replay a recorded trace with updated tensor pointers
 */
int cml_trace_replay(CMLTrace* trace, void** tensor_ptrs, int num_tensors);

/* ── Cache ── */

CMLTraceCache* cml_trace_cache_create(void);
void cml_trace_cache_free(CMLTraceCache* cache);
CMLTrace* cml_trace_cache_lookup(CMLTraceCache* cache, uint64_t graph_hash);
int cml_trace_cache_insert(CMLTraceCache* cache, uint64_t graph_hash, CMLTrace* trace);

/* ── High-level traced execution ── */

int cml_ir_execute_traced(CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_TRACE_H */
