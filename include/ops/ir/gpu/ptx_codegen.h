/**
 * @file ptx_codegen.h
 * @brief Standalone PTX assembly code generation (no LLVM dependency)
 *
 * Generates PTX assembly text using snprintf into dynamic buffers.
 * The generated PTX can be compiled at runtime via cml_cuda_compile_ptx()
 * from the CUDA backend, or used purely for testing (string-based).
 *
 * Supports unary ops, binary ops, fill, where, reductions, and matmul.
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

/**
 * @brief PTX codegen context
 */
typedef struct CMLPTXCodegen {
    int sm_version;       // e.g. 50 for sm_50
    int kernel_count;
    bool initialized;
} CMLPTXCodegen;

/**
 * @brief Create PTX codegen context
 * @param sm_version Target compute capability (e.g. 50 for sm_50, 75 for sm_75)
 * @return New codegen context or NULL
 */
CMLPTXCodegen* cml_ptx_codegen_create(int sm_version);

/**
 * @brief Destroy PTX codegen context
 */
void cml_ptx_codegen_destroy(CMLPTXCodegen* cg);

/**
 * @brief Generate PTX for a unary elementwise kernel
 * @param cg Codegen context
 * @param op UOp type (UOP_NEG, UOP_EXP, UOP_LOG, UOP_SQRT, UOP_ABS, UOP_SIN, UOP_COS,
 *           UOP_SIGMOID, UOP_TANH)
 * @param kernel_name Name for the kernel entry point
 * @return Heap-allocated PTX string (caller must free), or NULL on error
 */
char* cml_ptx_gen_unary(CMLPTXCodegen* cg, UOpType op, const char* kernel_name);

/**
 * @brief Generate PTX for a binary elementwise kernel
 * @param cg Codegen context
 * @param op UOp type (UOP_ADD, UOP_SUB, UOP_MUL, UOP_DIV, UOP_MAX, UOP_POW, UOP_CMPLT)
 * @param kernel_name Name for the kernel entry point
 * @return Heap-allocated PTX string (caller must free), or NULL on error
 */
char* cml_ptx_gen_binary(CMLPTXCodegen* cg, UOpType op, const char* kernel_name);

/**
 * @brief Generate PTX for a fill kernel (set all elements to a constant)
 * @param cg Codegen context
 * @param value Constant float value
 * @param kernel_name Name for the kernel entry point
 * @return Heap-allocated PTX string (caller must free), or NULL on error
 */
char* cml_ptx_gen_fill(CMLPTXCodegen* cg, float value, const char* kernel_name);

/**
 * @brief Generate PTX for a where (ternary) kernel: out = cond ? a : b
 * @param cg Codegen context
 * @param kernel_name Name for the kernel entry point
 * @return Heap-allocated PTX string (caller must free), or NULL on error
 */
char* cml_ptx_gen_where(CMLPTXCodegen* cg, const char* kernel_name);

/**
 * @brief Generate PTX for a reduction kernel (SUM or MEAN)
 * @param cg Codegen context
 * @param op UOp type (UOP_SUM or UOP_MEAN)
 * @param kernel_name Name for the kernel entry point
 * @return Heap-allocated PTX string (caller must free), or NULL on error
 */
char* cml_ptx_gen_reduction(CMLPTXCodegen* cg, UOpType op, const char* kernel_name);

/**
 * @brief Generate PTX for a matrix multiplication kernel
 * @param cg Codegen context
 * @param kernel_name Name for the kernel entry point
 * @return Heap-allocated PTX string (caller must free), or NULL on error
 */
char* cml_ptx_gen_matmul(CMLPTXCodegen* cg, const char* kernel_name);

/**
 * @brief Execute an IR graph using PTX codegen + CUDA runtime
 * @param cg Codegen context
 * @param ir IR graph to execute
 * @return 0 on success, -1 on failure
 */
int cml_ptx_execute_graph(CMLPTXCodegen* cg, CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif // CML_PTX_CODEGEN_H
