/**
 * @file trace.c
 * @brief Trace-and-Replay JIT implementation
 *
 * Captures the kernel launch sequence on the first run and replays it
 * without re-scheduling on subsequent runs with the same graph structure.
 */

#include "ops/ir/trace.h"
#include "ops/ir/internal.h"
#include "ops/ir/execution.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

static _Thread_local CMLTrace* g_active_trace = NULL;

CMLTrace* cml_trace_get_active(void) {
    return g_active_trace;
}

void cml_trace_set_active(CMLTrace* trace) {
    g_active_trace = trace;
}

#define FNV_OFFSET_BASIS 0xcbf29ce484222325ULL
#define FNV_PRIME        0x100000001b3ULL

/*  Trace lifecycle                                                        */

CMLTrace *cml_trace_create(void)
{
    CMLTrace *trace = (CMLTrace *)calloc(1, sizeof(CMLTrace));
    return trace; /* NULL on allocation failure */
}

void cml_trace_free(CMLTrace *trace)
{
    if (!trace) return;
    free(trace);
}

/*  Recording control                                                      */

int cml_trace_begin(CMLTrace *trace, uint64_t graph_hash)
{
    if (!trace) return -1;

    trace->is_recording = true;
    trace->is_complete  = false;
    trace->graph_hash   = graph_hash;
    trace->num_entries  = 0;
    trace->num_slots    = 0;
    memset(trace->entries, 0, sizeof(trace->entries));
    memset(trace->tensor_slots, 0, sizeof(trace->tensor_slots));

    return 0;
}

int cml_trace_end(CMLTrace *trace)
{
    if (!trace) return -1;

    trace->is_recording = false;
    trace->is_complete  = true;

    return 0;
}

/*  Recording helpers                                                      */

int cml_trace_record_kernel(CMLTrace *trace, uint64_t kernel_hash,
                            void *compiled_kernel, const size_t grid[3],
                            const size_t block[3], int *arg_indices,
                            int num_args)
{
    if (!trace || !trace->is_recording) return -1;
    if (trace->num_entries >= CML_TRACE_MAX_ENTRIES) return -2;
    if (num_args > CML_TRACE_MAX_ARGS) return -3;

    CMLTraceEntry *entry = &trace->entries[trace->num_entries];
    entry->type            = CML_TRACE_KERNEL;
    entry->kernel_hash     = kernel_hash;
    entry->compiled_kernel = compiled_kernel;

    /* Copy launch configuration */
    memcpy(entry->grid,  grid,  sizeof(size_t) * 3);
    memcpy(entry->block, block, sizeof(size_t) * 3);

    /* Copy argument slot indices */
    entry->num_args = num_args;
    if (num_args > 0 && arg_indices) {
        memcpy(entry->arg_indices, arg_indices, sizeof(int) * (size_t)num_args);
    }

    /* Clear memcpy-specific fields */
    entry->memcpy_bytes = 0;
    entry->src_slot     = -1;
    entry->dst_slot     = -1;

    trace->num_entries++;
    return 0;
}

int cml_trace_record_memcpy(CMLTrace *trace, CMLTraceEntryType type,
                            int src_slot, int dst_slot, size_t bytes)
{
    if (!trace || !trace->is_recording) return -1;
    if (trace->num_entries >= CML_TRACE_MAX_ENTRIES) return -2;
    if (type != CML_TRACE_MEMCPY_H2D && type != CML_TRACE_MEMCPY_D2H) return -3;

    CMLTraceEntry *entry = &trace->entries[trace->num_entries];
    entry->type        = type;
    entry->kernel_hash = 0;
    entry->compiled_kernel = NULL;

    memset(entry->grid,  0, sizeof(entry->grid));
    memset(entry->block, 0, sizeof(entry->block));
    entry->num_args = 0;

    entry->src_slot     = src_slot;
    entry->dst_slot     = dst_slot;
    entry->memcpy_bytes = bytes;

    trace->num_entries++;
    return 0;
}

/*  Replay                                                                 */

/**
 * Kernel function pointer type used during replay.
 * The compiled kernel is invoked with an array of data pointers and count.
 */
typedef void (*cml_kernel_fn_t)(void **args, int num_args,
                                const size_t grid[3],
                                const size_t block[3]);

int cml_trace_replay(CMLTrace *trace, void **tensor_ptrs, int num_tensors)
{
    if (!trace || !trace->is_complete) return -1;
    if (!tensor_ptrs && num_tensors > 0) return -2;

    /* Update the slot table with the fresh tensor pointers */
    int slots_to_copy = (num_tensors < trace->num_slots)
                            ? num_tensors
                            : trace->num_slots;
    for (int i = 0; i < slots_to_copy; i++) {
        trace->tensor_slots[i] = tensor_ptrs[i];
    }

    /* Walk every recorded entry */
    for (int i = 0; i < trace->num_entries; i++) {
        CMLTraceEntry *e = &trace->entries[i];

        switch (e->type) {

        case CML_TRACE_KERNEL: {
            if (!e->compiled_kernel) return -3;

            /* Build the argument pointer array from slot indices */
            void *args[CML_TRACE_MAX_ARGS];
            for (int a = 0; a < e->num_args; a++) {
                int slot = e->arg_indices[a];
                if (slot < 0 || slot >= num_tensors) return -4;
                args[a] = tensor_ptrs[slot];
            }

            /* Invoke the compiled kernel */
            cml_kernel_fn_t fn = (cml_kernel_fn_t)e->compiled_kernel;
            fn(args, e->num_args, e->grid, e->block);
            break;
        }

        case CML_TRACE_MEMCPY_H2D:
        case CML_TRACE_MEMCPY_D2H: {
            if (e->src_slot < 0 || e->src_slot >= num_tensors) return -4;
            if (e->dst_slot < 0 || e->dst_slot >= num_tensors) return -4;

            void *src = tensor_ptrs[e->src_slot];
            void *dst = tensor_ptrs[e->dst_slot];
            if (!src || !dst) return -5;

            memcpy(dst, src, e->memcpy_bytes);
            break;
        }

        default:
            return -6; /* unknown entry type */
        }
    }

    return 0;
}

/*  Trace cache                                                            */

CMLTraceCache *cml_trace_cache_create(void)
{
    CMLTraceCache *cache = (CMLTraceCache *)calloc(1, sizeof(CMLTraceCache));
    return cache;
}

void cml_trace_cache_free(CMLTraceCache *cache)
{
    if (!cache) return;

    for (int i = 0; i < CML_TRACE_CACHE_SIZE; i++) {
        if (cache->entries[i].occupied && cache->entries[i].trace) {
            cml_trace_free(cache->entries[i].trace);
        }
    }
    free(cache);
}

CMLTrace *cml_trace_cache_lookup(CMLTraceCache *cache, uint64_t graph_hash)
{
    if (!cache) return NULL;

    /* Linear-probe hash table lookup */
    uint64_t idx = graph_hash % CML_TRACE_CACHE_SIZE;

    for (int probe = 0; probe < CML_TRACE_CACHE_SIZE; probe++) {
        uint64_t slot = (idx + (uint64_t)probe) % CML_TRACE_CACHE_SIZE;

        if (!cache->entries[slot].occupied) {
            return NULL; /* empty slot -> not found */
        }
        if (cache->entries[slot].hash == graph_hash) {
            return cache->entries[slot].trace;
        }
    }

    return NULL; /* table full, not found */
}

int cml_trace_cache_insert(CMLTraceCache *cache, uint64_t graph_hash,
                           CMLTrace *trace)
{
    if (!cache || !trace) return -1;
    if (cache->count >= CML_TRACE_CACHE_SIZE) return -2; /* full */

    uint64_t idx = graph_hash % CML_TRACE_CACHE_SIZE;

    for (int probe = 0; probe < CML_TRACE_CACHE_SIZE; probe++) {
        uint64_t slot = (idx + (uint64_t)probe) % CML_TRACE_CACHE_SIZE;

        if (!cache->entries[slot].occupied) {
            cache->entries[slot].hash     = graph_hash;
            cache->entries[slot].trace    = trace;
            cache->entries[slot].occupied = true;
            cache->count++;
            return 0;
        }

        /* If the hash already exists, replace the trace */
        if (cache->entries[slot].hash == graph_hash) {
            if (cache->entries[slot].trace && cache->entries[slot].trace != trace) {
                cml_trace_free(cache->entries[slot].trace);
            }
            cache->entries[slot].trace = trace;
            return 0;
        }
    }

    return -2; /* should not reach here if count < size */
}

/*  Graph hashing (FNV-1a over node types + shapes)                        */

static uint64_t fnv1a_bytes(uint64_t hash, const void *data, size_t len)
{
    const uint8_t *bytes = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)bytes[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

static uint64_t compute_graph_hash(CMLGraph_t ir)
{
    if (!ir) return 0;

    uint64_t hash = FNV_OFFSET_BASIS;

    struct IRNode *node = ir->head;
    while (node) {
        /* Hash the UOp type */
        int type_val = (int)node->type;
        hash = fnv1a_bytes(hash, &type_val, sizeof(type_val));

        /* Hash the output shape */
        if (node->output_shape && node->output_ndim > 0) {
            hash = fnv1a_bytes(hash, node->output_shape,
                               sizeof(int) * (size_t)node->output_ndim);
        }

        /* Hash the number of inputs */
        hash = fnv1a_bytes(hash, &node->num_inputs, sizeof(node->num_inputs));

        node = node->next;
    }

    return hash;
}

/*  High-level traced execution                                            */

/**
 * Global trace cache.  Lazily initialised on first call.
 */
static CMLTraceCache *g_trace_cache = NULL;

int cml_ir_execute_traced(CMLGraph_t ir)
{
    if (!ir) return -1;

    /* Lazy-init the global cache */
    if (!g_trace_cache) {
        g_trace_cache = cml_trace_cache_create();
        if (!g_trace_cache) return -2;
    }

    /* Step 1 -- Compute graph hash */
    uint64_t hash = compute_graph_hash(ir);

    /* Step 2 -- Look up trace in cache */
    CMLTrace *trace = cml_trace_cache_lookup(g_trace_cache, hash);

    if (trace && trace->is_complete) {

        /*
         * Build a tensor_ptrs array from the current graph.
         * Walk the IR nodes and collect each node's output data pointer
         * so that the replay can use updated addresses.
         */
        void *tensor_ptrs[CML_TRACE_MAX_ENTRIES];
        int n = 0;
        struct IRNode *node = ir->head;
        while (node && n < CML_TRACE_MAX_ENTRIES) {
            if (node->output && node->output->data) {
                tensor_ptrs[n++] = node->output->data;
            } else {
                tensor_ptrs[n++] = NULL;
            }
            node = node->next;
        }

        return cml_trace_replay(trace, tensor_ptrs, n);
    }

    trace = cml_trace_create();
    if (!trace) return -3;

    cml_trace_begin(trace, hash);

    /* Set the active trace so the execution engine can record launches */
    cml_trace_set_active(trace);

    /* Execute the graph normally */
    int rc = cml_ir_execute(ir);

    /* Clear active trace */
    cml_trace_set_active(NULL);

    if (rc != 0) {
        cml_trace_free(trace);
        return rc;
    }

    cml_trace_end(trace);

    /*
     * Populate the slot table with the data pointers that were live during
     * this first execution so that future replays can diff against them.
     */
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

    /* Step 4 -- Insert trace into cache */
    cml_trace_cache_insert(g_trace_cache, hash, trace);

    return 0;
}
