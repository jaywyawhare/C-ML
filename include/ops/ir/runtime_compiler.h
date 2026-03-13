/**
 * @file runtime_compiler.h
 * @brief Runtime kernel compilation pipeline
 *
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

/** Maximum cached compiled kernels */
#define CML_COMPILED_CACHE_SIZE 256

/** A compiled kernel ready for execution */
typedef struct CMLCompiledKernel {
    uint64_t hash;              /* Hash of the LinearProgram */
    CMLFusedBackend backend;
    char* source;               /* Source code (for recompilation) */
    void* binary;               /* Compiled binary (PTX cubin, SPIR-V, etc.) */
    size_t binary_size;
    int num_inputs;
    int num_outputs;
    size_t work_size;
    bool valid;
} CMLCompiledKernel;

/** Runtime compiler context */
typedef struct CMLRuntimeCompiler {
    CMLCompiledKernel cache[CML_COMPILED_CACHE_SIZE];
    int num_cached;

    /* Statistics */
    size_t compile_hits;
    size_t compile_misses;
    size_t total_compilations;
    double total_compile_time_ms;

    /* Configuration */
    CMLFusedBackend preferred_backend;
    bool enable_caching;
    bool verbose;
} CMLRuntimeCompiler;

/** Create runtime compiler */
CMLRuntimeCompiler* cml_runtime_compiler_create(void);

/** Free runtime compiler and all cached kernels */
void cml_runtime_compiler_free(CMLRuntimeCompiler* rc);

/** Compile a fusion group into a fused kernel (with caching)
 * Returns compiled kernel (owned by cache), or NULL on failure.
 */
const CMLCompiledKernel* cml_runtime_compile_group(CMLRuntimeCompiler* rc,
                                                     const CMLFusionGroup* group);

/** Compile a LinearProgram into a fused kernel (with caching) */
const CMLCompiledKernel* cml_runtime_compile_program(CMLRuntimeCompiler* rc,
                                                       const CMLLinearProgram* prog,
                                                       size_t work_size);

/** Execute a compiled kernel on CPU (fallback execution)
 * Interprets the LinearProgram operations directly.
 */
int cml_runtime_execute_compiled(const CMLCompiledKernel* kernel,
                                  Tensor** inputs, int num_inputs,
                                  Tensor** outputs, int num_outputs);

/** Execute a full IR graph via the runtime compiler pipeline
 * schedule_v2 -> linearize groups -> fused codegen -> execute
 */
int cml_runtime_execute_graph(CMLRuntimeCompiler* rc, CMLGraph_t ir);

/** Get compiler statistics */
void cml_runtime_compiler_stats(const CMLRuntimeCompiler* rc,
                                 size_t* hits, size_t* misses,
                                 size_t* compilations);

/** Clear the compilation cache */
void cml_runtime_compiler_clear_cache(CMLRuntimeCompiler* rc);

/** Set preferred backend for compilation */
void cml_runtime_compiler_set_backend(CMLRuntimeCompiler* rc,
                                       CMLFusedBackend backend);

#ifdef __cplusplus
}
#endif

#endif /* CML_RUNTIME_COMPILER_H */
