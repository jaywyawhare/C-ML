/**
 * @file aot.h
 * @brief Ahead-of-Time (AOT) compilation for CML models
 *
 * Compiles CML IR graphs to native shared libraries (.so/.dylib/.dll)
 * that can be loaded and executed without any compilation dependency at runtime.
 *
 * Features:
 * - Cross-compilation with target triple
 * - Weight bundling in the compiled artifact
 * - C header generation for the compiled model
 * - Zero dependency at runtime via dlopen loader
 */

#ifndef CML_AOT_H
#define CML_AOT_H

#include "ops/ir/ir.h"
#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief AOT output format
 */
typedef enum {
    AOT_FORMAT_OBJECT,      /* .o object file */
    AOT_FORMAT_SHARED_LIB,  /* .so / .dylib shared library */
    AOT_FORMAT_STATIC_LIB,  /* .a static library */
    AOT_FORMAT_HEADER_ONLY, /* .h header with function declarations */
    AOT_FORMAT_LLVM_IR      /* .ll LLVM IR text */
} AOTOutputFormat;

/**
 * @brief Optimization level for AOT compilation
 */
typedef enum {
    AOT_OPT_O0 = 0,
    AOT_OPT_O1 = 1,
    AOT_OPT_O2 = 2,
    AOT_OPT_O3 = 3,
    AOT_OPT_Os = 4  /* Optimize for size */
} AOTOptLevel;

/**
 * @brief AOT compilation options
 */
typedef struct {
    const char* target_triple;  /* e.g. "x86_64-unknown-linux-gnu" (NULL = host) */
    const char* cpu;            /* e.g. "skylake" (NULL = generic) */
    const char* features;       /* e.g. "+avx2,+fma" (NULL = default) */
    AOTOptLevel opt_level;      /* Optimization level */
    bool include_weights;       /* Bundle weights into the binary */
    bool generate_header;       /* Generate companion .h file */
    const char* function_name;  /* Entry point name (default: "cml_model_forward") */
    AOTOutputFormat format;     /* Output format */
    bool position_independent;  /* -fPIC (default: true for shared libs) */
} AOTCompileOptions;

/**
 * @brief AOT model artifact loaded at runtime
 */
typedef struct CMLAOTModel {
    void* handle;               /* dlopen handle */
    void* forward_fn;           /* Function pointer to forward pass */
    const char* path;           /* Path to the loaded library */
    int num_inputs;             /* Number of input tensors */
    int num_outputs;            /* Number of output tensors */
    int** input_shapes;         /* Expected input shapes */
    int* input_ndims;           /* Input dimensions */
    int** output_shapes;        /* Expected output shapes */
    int* output_ndims;          /* Output dimensions */
    bool weights_bundled;       /* Whether weights are in the binary */
} CMLAOTModel;

/**
 * @brief Create default AOT compilation options
 */
AOTCompileOptions cml_aot_default_options(void);

/**
 * @brief Compile an IR graph to an AOT artifact
 *
 * Traces the forward pass IR, compiles through LLVM, then generates
 * the requested output format.
 *
 * @param ir IR graph to compile
 * @param output_path Path for the output file
 * @param options Compilation options (NULL = defaults)
 * @return 0 on success, -1 on failure
 */
int cml_aot_compile(CMLGraph_t ir, const char* output_path, const AOTCompileOptions* options);

/**
 * @brief Compile a Module to an AOT artifact
 *
 * Traces the module's forward pass with sample input to capture IR,
 * then compiles it.
 *
 * @param module Module to compile
 * @param sample_input Sample input tensor (for shape inference)
 * @param output_path Output file path
 * @param options Compilation options (NULL = defaults)
 * @return 0 on success, -1 on failure
 */
int cml_aot_compile_module(struct Module* module, Tensor* sample_input,
                           const char* output_path, const AOTCompileOptions* options);

/**
 * @brief Load an AOT-compiled model at runtime
 *
 * Uses dlopen() to load the shared library and dlsym() to find the
 * entry function. Zero compilation dependency at runtime.
 *
 * @param path Path to the shared library
 * @return Loaded AOT model, or NULL on failure
 */
CMLAOTModel* cml_aot_load(const char* path);

/**
 * @brief Execute an AOT-compiled model
 *
 * @param model Loaded AOT model
 * @param inputs Input tensors
 * @param num_inputs Number of inputs
 * @param outputs Output tensors (pre-allocated)
 * @param num_outputs Number of outputs
 * @return 0 on success, -1 on failure
 */
int cml_aot_execute(CMLAOTModel* model, Tensor** inputs, int num_inputs,
                    Tensor** outputs, int num_outputs);

/**
 * @brief Free an AOT-compiled model
 *
 * @param model Model to free
 */
void cml_aot_free(CMLAOTModel* model);

/**
 * @brief Generate C header for an AOT-compiled model
 *
 * @param ir IR graph
 * @param header_path Output header path
 * @param function_name Function name
 * @return 0 on success, -1 on failure
 */
int cml_aot_generate_header(CMLGraph_t ir, const char* header_path, const char* function_name);

#ifdef __cplusplus
}
#endif

#endif /* CML_AOT_H */
