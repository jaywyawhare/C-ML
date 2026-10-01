#include "ops/ir/tiny_jit.h"
#include "ops/ir/internal.h"
#include "ops/ir/trace.h"
#include "ops/ir/execution.h"
#include "tensor/tensor.h"
#include "core/logging.h"

#include <stdio.h>
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

/* ---- Opt-in CPU replay (TINYJIT_REPLAY=1) ------------------------------------
 * Replay executes the *live* graph by a plain head->next walk of
 * cpu_execute_node, skipping cpu_execute_ir's DCE marking and scheduler
 * dispatch. Because it drives the real per-node kernels over the real nodes,
 * there is no serialized trace to go stale (no copied params, no slot binding,
 * no IRNode* outliving its graph) -- the hazards that made a serialized CPU
 * trace unsafe. A shape is only ever replayed after a self-check proves the
 * walk reproduces the scheduled execution bit-for-bit, so a graph whose correct
 * result depends on the scheduler's ordering is poisoned, not mis-run. Off by
 * default: the standard path is byte-for-byte unchanged. */
static int cml_tinyjit_replay_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("TINYJIT_REPLAY");
        v             = (e && e[0] == '1') ? 1 : 0;
    }
    return v;
}

static int cml_tinyjit_plain_walk(CMLGraph_t ir) {
    for (struct IRNode* node = ir->head; node; node = node->next) {
        if (!node->output)
            continue;
        if (cpu_execute_node(node) != 0)
            return -1;
    }
    return 0;
}

/* Recompute every node into scratch buffers and require bit-identical results
 * against the just-produced real outputs. The live outputs are swapped out and
 * restored, so a failed check never corrupts the current execution. */
static bool cml_tinyjit_verify(CMLGraph_t ir) {
    int n = 0;
    for (struct IRNode* nd = ir->head; nd; nd = nd->next)
        n++;
    if (n == 0)
        return false;

    typedef struct {
        Tensor* out;
        void* real;
        bool owns;
        bool cache;
        size_t bytes;
    } Saved;
    Saved* sv = (Saved*)cml_calloc((size_t)n, sizeof(Saved));
    if (!sv)
        return false;

    bool fatal = false;
    int idx    = 0;
    for (struct IRNode* nd = ir->head; nd; nd = nd->next, idx++) {
        Tensor* o   = nd->output;
        sv[idx].out = o;
        if (!o || !o->data || o->numel == 0)
            continue;
        size_t bytes  = o->numel * cml_dtype_size(o->dtype);
        void* scratch = cml_buffer_cache_alloc(bytes);
        if (!scratch) {
            fatal = true;
            break;
        }
        sv[idx].real         = o->data;
        sv[idx].owns         = o->owns_data;
        sv[idx].cache        = o->from_buffer_cache;
        sv[idx].bytes        = bytes;
        o->data              = scratch;
        o->owns_data         = true;
        o->from_buffer_cache = true;
    }

    bool verified = false;
    if (!fatal && cml_tinyjit_plain_walk(ir) == 0) {
        verified = true;
        idx      = 0;
        for (struct IRNode* nd = ir->head; nd; nd = nd->next, idx++) {
            if (!sv[idx].real)
                continue;
            Tensor* o = sv[idx].out;
            if (!o->data || memcmp(o->data, sv[idx].real, sv[idx].bytes) != 0) {
                verified = false;
                break;
            }
        }
    }

    /* Restore the real outputs; free whatever the walk left behind. */
    idx = 0;
    for (struct IRNode* nd = ir->head; nd; nd = nd->next, idx++) {
        if (!sv[idx].real)
            continue;
        Tensor* o = sv[idx].out;
        if (o->data && o->data != sv[idx].real) {
            if (o->from_buffer_cache)
                cml_buffer_cache_free(o->data, sv[idx].bytes);
            else if (o->owns_data)
                cml_free(o->data);
        }
        o->data              = sv[idx].real;
        o->owns_data         = sv[idx].owns;
        o->from_buffer_cache = sv[idx].cache;
    }
    cml_free(sv);
    return verified;
}

static CMLJitEntry* cml_tinyjit_find(CMLTinyJit* jit, uint64_t hash, const int* sig, int sig_len) {
    uint64_t idx = hash % CML_JIT_CACHE_SIZE;
    for (int probe = 0; probe < CML_JIT_CACHE_SIZE; probe++) {
        CMLJitEntry* e = &jit->entries[(idx + (uint64_t)probe) % CML_JIT_CACHE_SIZE];
        if (!e->occupied)
            return NULL;
        if (e->graph_hash == hash && shape_matches(e, sig, sig_len))
            return e;
    }
    return NULL;
}

static CMLJitEntry* cml_tinyjit_insert(CMLTinyJit* jit, uint64_t hash, const int* sig,
                                       int sig_len) {
    if (jit->count >= CML_JIT_CACHE_SIZE)
        return NULL;
    uint64_t idx = hash % CML_JIT_CACHE_SIZE;
    for (int probe = 0; probe < CML_JIT_CACHE_SIZE; probe++) {
        CMLJitEntry* e = &jit->entries[(idx + (uint64_t)probe) % CML_JIT_CACHE_SIZE];
        if (!e->occupied) {
            e->graph_hash   = hash;
            e->trace        = NULL;
            e->shape_len    = sig_len;
            e->replay_state = 0;
            memcpy(e->shape_sig, sig, sizeof(int) * (size_t)sig_len);
            e->occupied = true;
            jit->count++;
            return e;
        }
    }
    return NULL;
}

static int cml_tinyjit_cpu_replay(CMLTinyJit* jit, CMLGraph_t ir) {
    uint64_t hash = cml_ir_graph_hash(ir);
    int sig[32];
    int sig_len = 0;
    compute_shape_sig(ir, sig, &sig_len, 32);

    CMLJitEntry* e = cml_tinyjit_find(jit, hash, sig, sig_len);

    if (e && e->replay_state == 1) {
        if (cml_tinyjit_plain_walk(ir) == 0) {
            jit->hits++;
            return 0;
        }
        e->replay_state = 2; /* unexpected walk failure -> never replay again */
    }

    int rc = cml_ir_execute(ir);
    if (rc != 0)
        return rc;
    jit->misses++;

    if (!e)
        e = cml_tinyjit_insert(jit, hash, sig, sig_len);
    if (e && e->replay_state == 0)
        e->replay_state = cml_tinyjit_verify(ir) ? 1 : 2;
    return 0;
}

int cml_tinyjit_execute(CMLTinyJit* jit, CMLGraph_t ir) {
    if (!jit || !ir)
        return -1;

    if (cml_tinyjit_replay_enabled())
        return cml_tinyjit_cpu_replay(jit, ir);

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

            /* Only replay a trace that actually captured work. An empty trace
             * would "replay" as a no-op and leave the outputs at their first-run
             * values, so num_entries>0 is what keeps replay faithful; a graph that
             * recorded nothing falls through and re-executes for real.
             *
             * The CPU executor records its node order (CML_TRACE_CPU_NODE), so
             * replay skips the graph walk, DCE, fusion decisions and scheduling and
             * re-runs just those nodes. A truncated trace is never marked complete,
             * so it cannot be replayed in part. */
            /* Known to record nothing: run it directly rather than paying for
             * another trace allocation and hash on every execute. */
            if (entry->records_nothing)
                return cml_ir_execute(ir);

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

    /* Don't cache an empty or truncated trace: an empty one would "replay" as a
     * no-op and a truncated one would skip the graph's tail, either way returning
     * stale outputs. Leaving it uncached means the next call re-executes for real.
     *
     * An EMPTY trace is also worth remembering as such. The CPU path records
     * nothing, so without a negative entry every execute re-allocated a CMLTrace
     * and re-hashed the graph to rediscover that -- pure overhead on the default
     * path (about 37% of a small step, measured). A truncated trace is not
     * negatively cached: it recorded work, just not all of it, and a different
     * run could fit. */
    if (trace->num_entries == 0 || !trace->is_complete) {
        bool empty = (trace->num_entries == 0);
        cml_trace_free(trace);
        if (empty && jit->count < CML_JIT_CACHE_SIZE) {
            idx = hash % CML_JIT_CACHE_SIZE;
            for (int probe = 0; probe < CML_JIT_CACHE_SIZE; probe++) {
                uint64_t slot      = (idx + (uint64_t)probe) % CML_JIT_CACHE_SIZE;
                CMLJitEntry* entry = &jit->entries[slot];
                if (!entry->occupied) {
                    entry->graph_hash      = hash;
                    entry->trace           = NULL;
                    entry->records_nothing = true;
                    memcpy(entry->shape_sig, sig, sizeof(int) * (size_t)sig_len);
                    entry->shape_len = sig_len;
                    entry->occupied  = true;
                    jit->count++;
                    break;
                }
            }
        }
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
