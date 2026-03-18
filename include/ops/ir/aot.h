/*
 * Ahead-of-Time (AOT) compilation for CML models.
 * Compiles CML IR graphs to native shared libraries (.so/.dylib/.dll)
 * that can be loaded and executed without any compilation dependency at runtime.
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

typedef enum {
    AOT_FORMAT_OBJECT,      /* .o object file */
    AOT_FORMAT_SHARED_LIB,  /* .so / .dylib shared library */
    AOT_FORMAT_STATIC_LIB,  /* .a static library */
    AOT_FORMAT_HEADER_ONLY, /* .h header with function declarations */
    AOT_FORMAT_LLVM_IR      /* .ll LLVM IR text */
} AOTOutputFormat;

typedef enum {
    AOT_OPT_O0 = 0,
    AOT_OPT_O1 = 1,
    AOT_OPT_O2 = 2,
    AOT_OPT_O3 = 3,
    AOT_OPT_Os = 4  /* Optimize for size */
} AOTOptLevel;

typedef struct {
    const char* target_triple;  /* e.g. "x86_64-unknown-linux-gnu" (NULL = host) */
    const char* cpu;            /* e.g. "skylake" (NULL = generic) */
    const char* features;       /* e.g. "+avx2,+fma" (NULL = default) */
    AOTOptLevel opt_level;
    bool include_weights;       /* Bundle weights into the binary */
    bool generate_header;       /* Generate companion .h file */
    const char* function_name;  /* Entry point name (default: "cml_model_forward") */
    AOTOutputFormat format;
    bool position_independent;  /* -fPIC (default: true for shared libs) */
} AOTCompileOptions;

typedef struct CMLAOTModel {
    void* handle;               /* dlopen handle */
    void* forward_fn;           /* Function pointer to forward pass */
    const char* path;
    int num_inputs;
    int num_outputs;
    int** input_shapes;
    int* input_ndims;
    int** output_shapes;
    int* output_ndims;
    bool weights_bundled;
} CMLAOTModel;

AOTCompileOptions cml_aot_default_options(void);

int cml_aot_compile(CMLGraph_t ir, const char* output_path, const AOTCompileOptions* options);

/* Traces the module's forward pass with sample input to capture IR, then compiles. */
int cml_aot_compile_module(struct Module* module, Tensor* sample_input,
                           const char* output_path, const AOTCompileOptions* options);

/* Uses dlopen/dlsym. Zero compilation dependency at runtime. */
CMLAOTModel* cml_aot_load(const char* path);

int cml_aot_execute(CMLAOTModel* model, Tensor** inputs, int num_inputs,
                    Tensor** outputs, int num_outputs);
void cml_aot_free(CMLAOTModel* model);
int cml_aot_generate_header(CMLGraph_t ir, const char* header_path, const char* function_name);

#ifdef __cplusplus
}
#endif

#endif /* CML_AOT_H */
