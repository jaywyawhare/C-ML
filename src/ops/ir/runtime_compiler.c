/**
 * @file runtime_compiler.c
 * @brief Runtime kernel compilation pipeline implementation
 *
 * Full pipeline: IR -> schedule_v2 -> linearize -> fused codegen -> cache -> execute.
 */

#include "ops/ir/runtime_compiler.h"
#include "ops/ir/fused_codegen.h"
#include "ops/ir/schedule.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* ── FNV-1a hashing for LinearProgram ── */

#define FNV_OFFSET 0xcbf29ce484222325ULL
#define FNV_PRIME  0x100000001b3ULL

static uint64_t hash_bytes(uint64_t h, const void* data, size_t len) {
    const uint8_t* bytes = (const uint8_t*)data;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)bytes[i];
        h *= FNV_PRIME;
    }
    return h;
}

static uint64_t hash_linear_program(const CMLLinearProgram* prog) {
    uint64_t h = FNV_OFFSET;
    h = hash_bytes(h, &prog->num_ops, sizeof(prog->num_ops));
    h = hash_bytes(h, &prog->next_vreg, sizeof(prog->next_vreg));

    for (int i = 0; i < prog->num_ops; i++) {
        const CMLLinearOp* op = &prog->ops[i];
        h = hash_bytes(h, &op->kind, sizeof(op->kind));
        h = hash_bytes(h, &op->uop, sizeof(op->uop));
        h = hash_bytes(h, &op->dest_reg, sizeof(op->dest_reg));
        h = hash_bytes(h, &op->num_srcs, sizeof(op->num_srcs));
        for (int j = 0; j < op->num_srcs; j++) {
            h = hash_bytes(h, &op->src_regs[j], sizeof(op->src_regs[j]));
        }
    }
    return h;
}

/* ── Public API ── */

CMLRuntimeCompiler* cml_runtime_compiler_create(void) {
    CMLRuntimeCompiler* rc = calloc(1, sizeof(CMLRuntimeCompiler));
    if (!rc) return NULL;
    rc->preferred_backend = CML_FUSED_BACKEND_C;
    rc->enable_caching = true;
    rc->verbose = false;
    return rc;
}

void cml_runtime_compiler_free(CMLRuntimeCompiler* rc) {
    if (!rc) return;
    for (int i = 0; i < CML_COMPILED_CACHE_SIZE; i++) {
        CMLCompiledKernel* k = &rc->cache[i];
        if (k->valid) {
            free(k->source);
            free(k->binary);
        }
    }
    free(rc);
}

static CMLCompiledKernel* cache_lookup(CMLRuntimeCompiler* rc, uint64_t hash) {
    uint64_t idx = hash % CML_COMPILED_CACHE_SIZE;
    for (int probe = 0; probe < CML_COMPILED_CACHE_SIZE; probe++) {
        uint64_t slot = (idx + (uint64_t)probe) % CML_COMPILED_CACHE_SIZE;
        CMLCompiledKernel* k = &rc->cache[slot];
        if (!k->valid) return NULL;
        if (k->hash == hash) return k;
    }
    return NULL;
}

static CMLCompiledKernel* cache_insert(CMLRuntimeCompiler* rc, uint64_t hash) {
    uint64_t idx = hash % CML_COMPILED_CACHE_SIZE;
    for (int probe = 0; probe < CML_COMPILED_CACHE_SIZE; probe++) {
        uint64_t slot = (idx + (uint64_t)probe) % CML_COMPILED_CACHE_SIZE;
        CMLCompiledKernel* k = &rc->cache[slot];
        if (!k->valid) {
            k->hash = hash;
            k->valid = true;
            rc->num_cached++;
            return k;
        }
    }
    /* Cache full - evict slot 0 */
    CMLCompiledKernel* k = &rc->cache[hash % CML_COMPILED_CACHE_SIZE];
    free(k->source);
    free(k->binary);
    memset(k, 0, sizeof(*k));
    k->hash = hash;
    k->valid = true;
    return k;
}

const CMLCompiledKernel* cml_runtime_compile_program(CMLRuntimeCompiler* rc,
                                                       const CMLLinearProgram* prog,
                                                       size_t work_size) {
    if (!rc || !prog) return NULL;

    uint64_t hash = hash_linear_program(prog);

    /* Cache lookup */
    if (rc->enable_caching) {
        CMLCompiledKernel* cached = cache_lookup(rc, hash);
        if (cached) {
            rc->compile_hits++;
            return cached;
        }
    }

    rc->compile_misses++;

    /* Generate fused kernel */
    CMLFusedKernel* fused = cml_fused_codegen(prog, rc->preferred_backend, work_size);
    if (!fused) {
        LOG_ERROR("Runtime compiler: fused codegen failed");
        return NULL;
    }

    /* Insert into cache */
    CMLCompiledKernel* entry = cache_insert(rc, hash);
    if (!entry) {
        cml_fused_kernel_free(fused);
        return NULL;
    }

    entry->backend = fused->backend;
    entry->source = fused->source;
    fused->source = NULL;  /* Transfer ownership */
    entry->binary = fused->spirv_words;
    entry->binary_size = fused->spirv_words
        ? (size_t)fused->spirv_num_words * sizeof(uint32_t) : 0;
    fused->spirv_words = NULL;
    entry->num_inputs = fused->num_inputs;
    entry->num_outputs = fused->num_outputs;
    entry->work_size = work_size;

    rc->total_compilations++;

    cml_fused_kernel_free(fused);

    if (rc->verbose) {
        LOG_DEBUG("Runtime compiler: compiled kernel hash=0x%016llx, "
                  "%d inputs, %d outputs, work_size=%zu",
                  (unsigned long long)hash,
                  entry->num_inputs, entry->num_outputs, work_size);
    }

    return entry;
}

const CMLCompiledKernel* cml_runtime_compile_group(CMLRuntimeCompiler* rc,
                                                     const CMLFusionGroup* group) {
    if (!rc || !group) return NULL;

    CMLLinearProgram* prog = cml_linearize_group(group);
    if (!prog) return NULL;

    /* Estimate work size */
    size_t work_size = 0;
    if (group->num_nodes > 0 && group->nodes[0] && group->nodes[0]->output) {
        Tensor* t = group->nodes[0]->output;
        work_size = 1;
        for (int i = 0; i < t->ndim; i++) {
            work_size *= (size_t)t->shape[i];
        }
    }

    const CMLCompiledKernel* result = cml_runtime_compile_program(rc, prog, work_size);
    cml_linear_program_free(prog);
    return result;
}

int cml_runtime_execute_compiled(const CMLCompiledKernel* kernel,
                                  Tensor** inputs, int num_inputs,
                                  Tensor** outputs, int num_outputs) {
    if (!kernel || !inputs || !outputs) return -1;
    if (num_inputs < kernel->num_inputs || num_outputs < kernel->num_outputs) {
        LOG_ERROR("Runtime execute: buffer count mismatch");
        return -1;
    }

    /* For now, this is a placeholder. Real execution would:
     * - For PTX: use cuModuleLoadDataEx + cuLaunchKernel
     * - For SPIR-V: use vkCreateShaderModule + vkCmdDispatch
     * - For C: dlopen the compiled .so
     *
     * Currently we return 0 to indicate the infrastructure is wired up.
     */
    LOG_DEBUG("Runtime execute: %d inputs, %d outputs (stub)",
              num_inputs, num_outputs);
    return 0;
}

int cml_runtime_execute_graph(CMLRuntimeCompiler* rc, CMLGraph_t ir) {
    if (!rc || !ir) return -1;

    /* Step 1: Create V2 schedule with fusion groups */
    CMLScheduleV2* sched = cml_schedule_v2_create(ir, NULL);
    if (!sched) {
        LOG_ERROR("Runtime compiler: failed to create schedule");
        return -1;
    }

    int rc_val = 0;

    /* Step 2: For each fusion group, try fused compilation, fall back to CPU */
    for (int i = 0; i < sched->num_ordered && rc_val == 0; i++) {
        int idx = sched->execution_order[i];
        CMLFusionGroup* group = sched->groups[idx];
        if (!group) continue;

        /* Try fused compilation */
        const CMLCompiledKernel* compiled = cml_runtime_compile_group(rc, group);
        if (compiled) {
            /* TODO: launch compiled kernel on GPU
             * For now, fall through to CPU execution */
            LOG_DEBUG("Compiled group %d (%d nodes), falling back to CPU",
                      idx, group->num_nodes);
        }

        /* CPU fallback: execute each node */
        for (int j = 0; j < group->num_nodes && rc_val == 0; j++) {
            struct IRNode* node = group->nodes[j];
            if (!node || node->is_executed) continue;
            rc_val = cpu_execute_node(node);
            if (rc_val == 0) node->is_executed = true;
        }
    }

    cml_schedule_v2_free(sched);
    return rc_val;
}

void cml_runtime_compiler_stats(const CMLRuntimeCompiler* rc,
                                 size_t* hits, size_t* misses,
                                 size_t* compilations) {
    if (!rc) return;
    if (hits) *hits = rc->compile_hits;
    if (misses) *misses = rc->compile_misses;
    if (compilations) *compilations = rc->total_compilations;
}

void cml_runtime_compiler_clear_cache(CMLRuntimeCompiler* rc) {
    if (!rc) return;
    for (int i = 0; i < CML_COMPILED_CACHE_SIZE; i++) {
        CMLCompiledKernel* k = &rc->cache[i];
        if (k->valid) {
            free(k->source);
            free(k->binary);
            memset(k, 0, sizeof(*k));
        }
    }
    rc->num_cached = 0;
}

void cml_runtime_compiler_set_backend(CMLRuntimeCompiler* rc,
                                       CMLFusedBackend backend) {
    if (!rc) return;
    rc->preferred_backend = backend;
}
