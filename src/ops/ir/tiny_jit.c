/**
 * @file tiny_jit.c
 * @brief TinyJit -- capture-and-replay JIT implementation
 *
 * Wraps the trace infrastructure with a hash-table cache and shape
 * validation, providing a transparent JIT layer for repeated graph
 * execution.
 */

#include "ops/ir/tiny_jit.h"
#include "ops/ir/internal.h"
#include "ops/ir/trace.h"
#include "ops/ir/execution.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>

/* ── FNV-1a constants ── */

#define FNV_OFFSET_BASIS 0xcbf29ce484222325ULL
#define FNV_PRIME        0x100000001b3ULL

/* ── Graph hashing ── */

static uint64_t fnv1a_bytes(uint64_t hash, const void *data, size_t len)
{
    const uint8_t *bytes = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)bytes[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

static uint64_t jit_compute_hash(CMLGraph_t ir)
{
    if (!ir) return 0;

    uint64_t hash = FNV_OFFSET_BASIS;
    struct IRNode *node = ir->head;
    while (node) {
        int type_val = (int)node->type;
        hash = fnv1a_bytes(hash, &type_val, sizeof(type_val));

        if (node->output_shape && node->output_ndim > 0) {
            hash = fnv1a_bytes(hash, node->output_shape,
                               sizeof(int) * (size_t)node->output_ndim);
        }

        hash = fnv1a_bytes(hash, &node->num_inputs, sizeof(node->num_inputs));
        node = node->next;
    }
    return hash;
}

/* ── Shape signature ── */

static void compute_shape_sig(CMLGraph_t ir, int *sig, int *sig_len, int max_len)
{
    *sig_len = 0;
    struct IRNode *node = ir->head;
    /* Take the first node's output shape as the signature */
    if (node && node->output_shape && node->output_ndim > 0) {
        int n = node->output_ndim;
        if (n > max_len) n = max_len;
        memcpy(sig, node->output_shape, sizeof(int) * (size_t)n);
        *sig_len = n;
    }
}

static bool shape_matches(const CMLJitEntry *entry, const int *sig, int sig_len)
{
    if (entry->shape_len != sig_len) return false;
    return memcmp(entry->shape_sig, sig, sizeof(int) * (size_t)sig_len) == 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Public API
 * ══════════════════════════════════════════════════════════════════════════ */

CMLTinyJit *cml_tinyjit_create(void)
{
    CMLTinyJit *jit = (CMLTinyJit *)calloc(1, sizeof(CMLTinyJit));
    return jit;
}

void cml_tinyjit_free(CMLTinyJit *jit)
{
    if (!jit) return;
    for (int i = 0; i < CML_JIT_CACHE_SIZE; i++) {
        if (jit->entries[i].occupied && jit->entries[i].trace) {
            cml_trace_free(jit->entries[i].trace);
        }
    }
    free(jit);
}

int cml_tinyjit_execute(CMLTinyJit *jit, CMLGraph_t ir)
{
    if (!jit || !ir) return -1;

    /* Step 1 -- Compute graph hash and shape signature */
    uint64_t hash = jit_compute_hash(ir);
    int sig[32];
    int sig_len = 0;
    compute_shape_sig(ir, sig, &sig_len, 32);

    /* Step 2 -- Look up in cache */
    uint64_t idx = hash % CML_JIT_CACHE_SIZE;
    for (int probe = 0; probe < CML_JIT_CACHE_SIZE; probe++) {
        uint64_t slot = (idx + (uint64_t)probe) % CML_JIT_CACHE_SIZE;
        CMLJitEntry *entry = &jit->entries[slot];

        if (!entry->occupied) break;  /* empty slot -> not cached */

        if (entry->graph_hash == hash) {
            /* Hash match -- validate shapes */
            if (!shape_matches(entry, sig, sig_len)) {
                /* Shape mismatch -> invalidate and re-record */
                LOG_DEBUG("TinyJit: shape mismatch for hash 0x%016llx, re-recording",
                          (unsigned long long)hash);
                cml_trace_free(entry->trace);
                entry->trace = NULL;
                entry->occupied = false;
                jit->count--;
                jit->invalidations++;
                break;
            }

            /* Cache hit: replay */
            if (entry->trace && entry->trace->is_complete) {
                void *tensor_ptrs[CML_TRACE_MAX_ENTRIES];
                int n = 0;
                struct IRNode *node = ir->head;
                while (node && n < CML_TRACE_MAX_ENTRIES) {
                    tensor_ptrs[n++] = (node->output && node->output->data)
                                           ? node->output->data : NULL;
                    node = node->next;
                }

                int rc = cml_trace_replay(entry->trace, tensor_ptrs, n);
                if (rc == 0) {
                    jit->hits++;
                    return 0;
                }
            }
            break;
        }
    }

    /* Step 3 -- Cache miss: execute normally + record trace */
    jit->misses++;

    CMLTrace *trace = cml_trace_create();
    if (!trace) return -2;

    cml_trace_begin(trace, hash);
    cml_trace_set_active(trace);

    int rc = cml_ir_execute(ir);

    cml_trace_set_active(NULL);

    if (rc != 0) {
        cml_trace_free(trace);
        return rc;
    }

    cml_trace_end(trace);

    /* Populate slot table */
    {
        int n = 0;
        struct IRNode *node = ir->head;
        while (node && n < CML_TRACE_MAX_ENTRIES) {
            if (node->output && node->output->data) {
                trace->tensor_slots[n] = node->output->data;
            }
            n++;
            node = node->next;
        }
        trace->num_slots = n;
    }

    /* Step 4 -- Insert into cache */
    if (jit->count < CML_JIT_CACHE_SIZE) {
        idx = hash % CML_JIT_CACHE_SIZE;
        for (int probe = 0; probe < CML_JIT_CACHE_SIZE; probe++) {
            uint64_t slot = (idx + (uint64_t)probe) % CML_JIT_CACHE_SIZE;
            CMLJitEntry *entry = &jit->entries[slot];

            if (!entry->occupied) {
                entry->graph_hash = hash;
                entry->trace = trace;
                memcpy(entry->shape_sig, sig, sizeof(int) * (size_t)sig_len);
                entry->shape_len = sig_len;
                entry->occupied = true;
                jit->count++;
                LOG_DEBUG("TinyJit: cached trace for hash 0x%016llx",
                          (unsigned long long)hash);
                return 0;
            }
        }
    }

    /* Cache full -- execute succeeded but trace is discarded */
    cml_trace_free(trace);
    return 0;
}

void cml_tinyjit_stats(const CMLTinyJit *jit,
                       size_t *hits, size_t *misses, size_t *invalidations)
{
    if (!jit) return;
    if (hits) *hits = jit->hits;
    if (misses) *misses = jit->misses;
    if (invalidations) *invalidations = jit->invalidations;
}
