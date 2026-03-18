/*
 * Runtime kernel compilation pipeline.
 * Orchestrates: IR -> schedule -> linearize -> codegen -> compile -> cache -> launch.
 * Bridges the fused codegen with the dispatch and JIT cache systems.
 */

#ifndef CML_RUNTIME_COMPILER_H
#define CML_RUNTIME_COMPILER_H

#include "ops/ir/fused_codegen.h"
#include "ops/ir/schedule.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CML_COMPILED_CACHE_SIZE 256

typedef struct CMLCompiledKernel {
    uint64_t hash;
    CMLFusedBackend backend;
    char* source;
    void* binary;
    size_t binary_size;
    int num_inputs;
    int num_outputs;
    size_t work_size;
    bool valid;

    /* Stored linear ops for CPU interpreter execution */
    CMLLinearOp* ops;
    int num_ops;
    int num_vregs;
} CMLCompiledKernel;

typedef struct CMLRuntimeCompiler {
    CMLCompiledKernel cache[CML_COMPILED_CACHE_SIZE];
    int num_cached;

    size_t compile_hits;
    size_t compile_misses;
    size_t total_compilations;
    double total_compile_time_ms;

    CMLFusedBackend preferred_backend;
    bool enable_caching;
    bool verbose;
} CMLRuntimeCompiler;

CMLRuntimeCompiler* cml_runtime_compiler_create(void);
void cml_runtime_compiler_free(CMLRuntimeCompiler* rc);

/* Returns compiled kernel (owned by cache), or NULL on failure. */
const CMLCompiledKernel* cml_runtime_compile_group(CMLRuntimeCompiler* rc,
                                                     const CMLFusionGroup* group);
const CMLCompiledKernel* cml_runtime_compile_program(CMLRuntimeCompiler* rc,
                                                       const CMLLinearProgram* prog,
                                                       size_t work_size);

/* CPU fallback: interprets the LinearProgram operations directly. */
int cml_runtime_execute_compiled(const CMLCompiledKernel* kernel,
                                  Tensor** inputs, int num_inputs,
                                  Tensor** outputs, int num_outputs);

/* schedule_v2 -> linearize groups -> fused codegen -> execute */
int cml_runtime_execute_graph(CMLRuntimeCompiler* rc, CMLGraph_t ir);

void cml_runtime_compiler_stats(const CMLRuntimeCompiler* rc,
                                 size_t* hits, size_t* misses,
                                 size_t* compilations);
void cml_runtime_compiler_clear_cache(CMLRuntimeCompiler* rc);
void cml_runtime_compiler_set_backend(CMLRuntimeCompiler* rc,
                                       CMLFusedBackend backend);

#ifdef __cplusplus
}
#endif

#endif /* CML_RUNTIME_COMPILER_H */
