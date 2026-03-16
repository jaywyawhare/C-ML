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
#include <math.h>

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
            free(k->ops);
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
    CMLCompiledKernel* k = &rc->cache[hash % CML_COMPILED_CACHE_SIZE];
    free(k->source);
    free(k->binary);
    free(k->ops);
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

    if (rc->enable_caching) {
        CMLCompiledKernel* cached = cache_lookup(rc, hash);
        if (cached) {
            rc->compile_hits++;
            return cached;
        }
    }

    rc->compile_misses++;

    CMLFusedKernel* fused = cml_fused_codegen(prog, rc->preferred_backend, work_size);
    if (!fused) {
        LOG_ERROR("Runtime compiler: fused codegen failed");
        return NULL;
    }

    CMLCompiledKernel* entry = cache_insert(rc, hash);
    if (!entry) {
        cml_fused_kernel_free(fused);
        return NULL;
    }

    entry->backend = fused->backend;
    entry->source = fused->source;
    fused->source = NULL;
    entry->binary = fused->spirv_words;
    entry->binary_size = fused->spirv_words
        ? (size_t)fused->spirv_num_words * sizeof(uint32_t) : 0;
    fused->spirv_words = NULL;
    entry->num_inputs = fused->num_inputs;
    entry->num_outputs = fused->num_outputs;
    entry->work_size = work_size;

    if (prog->num_ops > 0) {
        entry->ops = malloc((size_t)prog->num_ops * sizeof(CMLLinearOp));
        if (entry->ops) {
            memcpy(entry->ops, prog->ops, (size_t)prog->num_ops * sizeof(CMLLinearOp));
            entry->num_ops = prog->num_ops;
            entry->num_vregs = prog->next_vreg;
        }
    }

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

    if (kernel->backend == CML_FUSED_BACKEND_C && kernel->work_size > 0) {
        size_t n = kernel->work_size;
        int nregs = kernel->num_vregs > 0 ? kernel->num_vregs : 64;
        if (nregs > 256) nregs = 256;

        if (kernel->ops && kernel->num_ops > 0) {
            for (size_t i = 0; i < n; i++) {
                float vregs[256] = {0};
                int input_idx = 0, output_idx = 0;

                for (int op_i = 0; op_i < kernel->num_ops; op_i++) {
                    const CMLLinearOp* op = &kernel->ops[op_i];
                    if (op->is_eliminated) continue;

                    int d = op->dest_reg;
                    if (d < 0 || d >= nregs) continue;

                    if (op->kind == LINOP_LOAD) {
                        if (input_idx < num_inputs && inputs[input_idx]
                            && inputs[input_idx]->data
                            && i < (size_t)inputs[input_idx]->numel) {
                            vregs[d] = ((float*)inputs[input_idx]->data)[i];
                        }
                        input_idx++;
                    } else if (op->kind == LINOP_STORE) {
                        float val = (op->num_srcs > 0 && op->src_regs[0] >= 0
                                     && op->src_regs[0] < nregs)
                                    ? vregs[op->src_regs[0]] : 0.0f;
                        if (output_idx < num_outputs && outputs[output_idx]
                            && outputs[output_idx]->data
                            && i < (size_t)outputs[output_idx]->numel) {
                            ((float*)outputs[output_idx]->data)[i] = val;
                        }
                        output_idx++;
                    } else if (op->kind == LINOP_COMPUTE) {
                        float a = (op->num_srcs > 0 && op->src_regs[0] >= 0
                                   && op->src_regs[0] < nregs)
                                  ? vregs[op->src_regs[0]] : 0.0f;
                        float b = (op->num_srcs > 1 && op->src_regs[1] >= 0
                                   && op->src_regs[1] < nregs)
                                  ? vregs[op->src_regs[1]] : 0.0f;

                        switch (op->uop) {
                        case UOP_ADD:   vregs[d] = a + b; break;
                        case UOP_SUB:   vregs[d] = a - b; break;
                        case UOP_MUL:   vregs[d] = a * b; break;
                        case UOP_DIV:   vregs[d] = b != 0.0f ? a / b : 0.0f; break;
                        case UOP_NEG:   vregs[d] = -a; break;
                        case UOP_EXP:   vregs[d] = expf(a); break;
                        case UOP_LOG:   vregs[d] = a > 0.0f ? logf(a) : -INFINITY; break;
                        case UOP_LOG2:  vregs[d] = a > 0.0f ? log2f(a) : -INFINITY; break;
                        case UOP_LOG10: vregs[d] = a > 0.0f ? log10f(a) : -INFINITY; break;
                        case UOP_EXP2:  vregs[d] = exp2f(a); break;
                        case UOP_SQRT:  vregs[d] = a >= 0.0f ? sqrtf(a) : 0.0f; break;
                        case UOP_RECIP: vregs[d] = a != 0.0f ? 1.0f / a : 0.0f; break;
                        case UOP_ABS:   vregs[d] = fabsf(a); break;
                        case UOP_SIN:   vregs[d] = sinf(a); break;
                        case UOP_COS:   vregs[d] = cosf(a); break;
                        case UOP_TAN:   vregs[d] = tanf(a); break;
                        case UOP_POW:   vregs[d] = powf(a, b); break;
                        case UOP_MAX:   vregs[d] = a > b ? a : b; break;
                        case UOP_CMPLT: vregs[d] = a < b ? 1.0f : 0.0f; break;
                        default:        vregs[d] = a; break; /* passthrough */
                        }
                    }
                }
            }
        } else {
            for (size_t i = 0; i < n; i++) {
                float val = 0.0f;
                if (num_inputs > 0 && inputs[0] && inputs[0]->data
                    && i < (size_t)inputs[0]->numel) {
                    val = ((float*)inputs[0]->data)[i];
                }
                for (int j = 0; j < num_outputs; j++) {
                    if (outputs[j] && outputs[j]->data
                        && i < (size_t)outputs[j]->numel) {
                        ((float*)outputs[j]->data)[i] = val;
                    }
                }
            }
        }
        LOG_DEBUG("Runtime execute: fused C kernel, %zu elements, %d in, %d out",
                  n, num_inputs, num_outputs);
        return 0;
    }

    LOG_DEBUG("Runtime execute: %d inputs, %d outputs (backend=%d)",
              num_inputs, num_outputs, kernel->backend);
    return 0;
}

int cml_runtime_execute_graph(CMLRuntimeCompiler* rc, CMLGraph_t ir) {
    if (!rc || !ir) return -1;

    CMLScheduleV2* sched = cml_schedule_v2_create(ir, NULL);
    if (!sched) {
        LOG_ERROR("Runtime compiler: failed to create schedule");
        return -1;
    }

    int rc_val = 0;

    for (int i = 0; i < sched->num_ordered && rc_val == 0; i++) {
        int idx = sched->execution_order[i];
        CMLFusionGroup* group = sched->groups[idx];
        if (!group) continue;

        /* Try fused compilation and execution */
        const CMLCompiledKernel* compiled = cml_runtime_compile_group(rc, group);
        bool fused_executed = false;

        if (compiled && compiled->backend == CML_FUSED_BACKEND_C
            && compiled->work_size > 0
            && group->num_nodes > 1) {
            /* Collect input/output tensors for the fused group */
            Tensor* fused_inputs[64] = {0};
            Tensor* fused_outputs[64] = {0};
            int n_in = 0, n_out = 0;

            for (int j = 0; j < group->num_nodes; j++) {
                struct IRNode* nd = group->nodes[j];
                if (!nd) continue;

                /* Gather external inputs */
                for (int k = 0; k < nd->num_inputs && n_in < 64; k++) {
                    if (nd->inputs && nd->inputs[k]) {
                        /* Check if this input is produced by another node in the group */
                        bool internal = false;
                        for (int m = 0; m < group->num_nodes; m++) {
                            if (group->nodes[m] && group->nodes[m]->output == nd->inputs[k]) {
                                internal = true;
                                break;
                            }
                        }
                        if (!internal) {
                            fused_inputs[n_in++] = nd->inputs[k];
                        }
                    }
                }

                /* Last node's output is the group output */
                if (j == group->num_nodes - 1 && nd->output && n_out < 64) {
                    fused_outputs[n_out++] = nd->output;
                }
            }

            if (cml_runtime_execute_compiled(compiled, fused_inputs, n_in,
                                              fused_outputs, n_out) == 0) {
                /* Mark all nodes as executed */
                for (int j = 0; j < group->num_nodes; j++) {
                    if (group->nodes[j]) {
                        group->nodes[j]->is_executed = true;
                        if (group->nodes[j]->output) {
                            group->nodes[j]->output->is_executed = true;
                        }
                    }
                }
                fused_executed = true;
                LOG_DEBUG("Fused execution of group %d (%d nodes)", idx, group->num_nodes);
            }
        }

        /* CPU fallback: execute each node individually */
        if (!fused_executed) {
            for (int j = 0; j < group->num_nodes && rc_val == 0; j++) {
                struct IRNode* node = group->nodes[j];
                if (!node || node->is_executed) continue;
                rc_val = cpu_execute_node(node);
                if (rc_val == 0) node->is_executed = true;
            }
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
            free(k->ops);
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
