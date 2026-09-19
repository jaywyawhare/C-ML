#include "ops/ir/tiny_jit.h"
#include "ops/ir/internal.h"
#include "ops/ir/trace.h"
#include "ops/ir/execution.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include "alloc/cml_allocator.h"

static void compute_shape_sig(CMLGraph_t ir, int* sig, int* sig_len, int max_len) {
    *sig_len            = 0;
    struct IRNode* node = ir->head;
    if (node && node->output_shape && node->output_ndim > 0) {
        int n = node->output_ndim;
        if (n > max_len)
            n = max_len;
        memcpy(sig, node->output_shape, sizeof(int) * (size_t)n);
        *sig_len = n;
    }
}

static bool shape_matches(const CMLJitEntry* entry, const int* sig, int sig_len) {
    if (entry->shape_len != sig_len)
        return false;
    return memcmp(entry->shape_sig, sig, sizeof(int) * (size_t)sig_len) == 0;
}

CMLTinyJit* cml_tinyjit_create(void) {
    CMLTinyJit* jit = (CMLTinyJit*)cml_calloc(1, sizeof(CMLTinyJit));
    return jit;
}

void cml_tinyjit_free(CMLTinyJit* jit) {
    if (!jit)
        return;
    for (int i = 0; i < CML_JIT_CACHE_SIZE; i++) {
        if (jit->entries[i].occupied && jit->entries[i].trace) {
            cml_trace_free(jit->entries[i].trace);
        }
    }
    cml_free(jit);
}

int cml_tinyjit_execute(CMLTinyJit* jit, CMLGraph_t ir) {
    if (!jit || !ir)
        return -1;

    uint64_t hash = cml_ir_graph_hash(ir);
    int sig[32];
    int sig_len = 0;
    compute_shape_sig(ir, sig, &sig_len, 32);

    uint64_t idx = hash % CML_JIT_CACHE_SIZE;
    for (int probe = 0; probe < CML_JIT_CACHE_SIZE; probe++) {
        uint64_t slot      = (idx + (uint64_t)probe) % CML_JIT_CACHE_SIZE;
        CMLJitEntry* entry = &jit->entries[slot];

        if (!entry->occupied)
            break; /* empty slot -> not cached */

        if (entry->graph_hash == hash) {
            if (!shape_matches(entry, sig, sig_len)) {
                LOG_DEBUG("TinyJit: shape mismatch for hash 0x%016llx, re-recording",
                          (unsigned long long)hash);
                cml_trace_free(entry->trace);
                entry->trace    = NULL;
                entry->occupied = false;
                jit->count--;
                jit->invalidations++;
                break;
            }

            /* Only replay a trace that actually captured kernel launches.
             * Kernel recording is not yet wired into the CPU/JIT execution path,
             * so traces are currently empty; replaying an empty trace is a no-op
             * that leaves the output buffers at their first-run (STALE) values.
             * Requiring num_entries>0 makes replay faithful, and falling through
             * re-executes for real (correct) until recording is implemented. */
            if (entry->trace && entry->trace->is_complete && entry->trace->num_entries > 0) {
                void* tensor_ptrs[CML_TRACE_MAX_ENTRIES];
                int n = cml_ir_output_slots(ir, tensor_ptrs, CML_TRACE_MAX_ENTRIES);

                int rc = cml_trace_replay(entry->trace, tensor_ptrs, n);
                if (rc == 0) {
                    jit->hits++;
                    return 0;
                }
            }
            break;
        }
    }

    jit->misses++;

    CMLTrace* trace = cml_trace_create();
    if (!trace)
        return -2;

    cml_trace_begin(trace, hash);
    cml_trace_set_active(trace);

    int rc = cml_ir_execute(ir);

    cml_trace_set_active(NULL);

    if (rc != 0) {
        cml_trace_free(trace);
        return rc;
    }

    cml_trace_end(trace);

    trace->num_slots = cml_ir_output_slots(ir, trace->tensor_slots, CML_TRACE_MAX_ENTRIES);

    /* Don't cache an empty trace (no kernels were recorded): a cached empty
     * trace would be "replayed" as a no-op on the next same-hash call, returning
     * stale outputs. Leaving it uncached means the next call re-executes for real. */
    if (trace->num_entries == 0) {
        cml_trace_free(trace);
        return rc;
    }

    if (jit->count < CML_JIT_CACHE_SIZE) {
        idx = hash % CML_JIT_CACHE_SIZE;
        for (int probe = 0; probe < CML_JIT_CACHE_SIZE; probe++) {
            uint64_t slot      = (idx + (uint64_t)probe) % CML_JIT_CACHE_SIZE;
            CMLJitEntry* entry = &jit->entries[slot];

            if (!entry->occupied) {
                entry->graph_hash = hash;
                entry->trace      = trace;
                memcpy(entry->shape_sig, sig, sizeof(int) * (size_t)sig_len);
                entry->shape_len = sig_len;
                entry->occupied  = true;
                jit->count++;
                LOG_DEBUG("TinyJit: cached trace for hash 0x%016llx", (unsigned long long)hash);
                return 0;
            }
        }
    }

    cml_trace_free(trace);
    return 0;
}

void cml_tinyjit_stats(const CMLTinyJit* jit, size_t* hits, size_t* misses, size_t* invalidations) {
    if (!jit)
        return;
    if (hits)
        *hits = jit->hits;
    if (misses)
        *misses = jit->misses;
    if (invalidations)
        *invalidations = jit->invalidations;
}
