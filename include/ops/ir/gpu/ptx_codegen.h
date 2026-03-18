/*
 * Standalone PTX assembly code generation (no LLVM dependency).
 * Generates PTX assembly text using snprintf into dynamic buffers.
 * All generated kernels target sm_50+ and use .f32 precision.
 */

#ifndef CML_PTX_CODEGEN_H
#define CML_PTX_CODEGEN_H

#include "ops/uops.h"
#include "ops/ir/ir.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CMLCUDABackend;

typedef struct CMLPTXCodegen {
    int sm_version;       // e.g. 50 for sm_50
    int kernel_count;
    bool initialized;
    struct CMLCUDABackend* cuda;  // CUDA backend for compilation + launch
} CMLPTXCodegen;

/* @param sm_version Target compute capability (e.g. 50 for sm_50)
   @param cuda CUDA backend context (may be NULL for codegen-only use) */
CMLPTXCodegen* cml_ptx_codegen_create(int sm_version, struct CMLCUDABackend* cuda);
void cml_ptx_codegen_destroy(CMLPTXCodegen* cg);

/* All return heap-allocated PTX strings (caller must free), or NULL on error. */
char* cml_ptx_gen_unary(CMLPTXCodegen* cg, UOpType op, const char* kernel_name);
char* cml_ptx_gen_binary(CMLPTXCodegen* cg, UOpType op, const char* kernel_name);
char* cml_ptx_gen_fill(CMLPTXCodegen* cg, float value, const char* kernel_name);
char* cml_ptx_gen_where(CMLPTXCodegen* cg, const char* kernel_name);
char* cml_ptx_gen_reduction(CMLPTXCodegen* cg, UOpType op, const char* kernel_name);
char* cml_ptx_gen_matmul(CMLPTXCodegen* cg, const char* kernel_name);

/* CUDA C source compiled via NVRTC */
char* cml_ptx_gen_tiled_matmul(CMLPTXCodegen* cg, const char* kernel_name);
char* cml_ptx_gen_conv2d(CMLPTXCodegen* cg, const char* kernel_name);

int cml_ptx_execute_graph(CMLPTXCodegen* cg, CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif // CML_PTX_CODEGEN_H
