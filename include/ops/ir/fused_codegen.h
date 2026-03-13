/**
 * @file fused_codegen.h
 * @brief Fused kernel code generation from LinearProgram
 *
 * Takes a LinearProgram (produced by the linearizer from a fusion group)
 * and emits fused kernel source code for each backend (PTX, SPIR-V, C).
 */

#ifndef CML_FUSED_CODEGEN_H
#define CML_FUSED_CODEGEN_H

#include "ops/ir/schedule.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations for LinearProgram (defined in linearize.c, now public) */
typedef enum {
    LINOP_LOAD,
    LINOP_COMPUTE,
    LINOP_STORE,
} CMLLinearOpKind;

typedef struct CMLLinearOp {
    CMLLinearOpKind kind;
    UOpType uop;
    int dest_reg;
    int src_regs[8];
    int num_srcs;
    Tensor* tensor;
    bool is_eliminated;
} CMLLinearOp;

typedef struct CMLLinearProgram {
    CMLLinearOp* ops;
    int num_ops;
    int capacity;
    int next_vreg;
} CMLLinearProgram;

/** Backend type for fused codegen */
typedef enum {
    CML_FUSED_BACKEND_C = 0,
    CML_FUSED_BACKEND_PTX,
    CML_FUSED_BACKEND_SPIRV,
    CML_FUSED_BACKEND_WGSL,
    CML_FUSED_BACKEND_METAL,
} CMLFusedBackend;

/** Result of fused code generation */
typedef struct CMLFusedKernel {
    CMLFusedBackend backend;
    char* source;           /* Generated source code (text for PTX/C/WGSL/Metal) */
    uint32_t* spirv_words;  /* SPIR-V binary words (if backend == SPIRV) */
    int spirv_num_words;
    int num_inputs;         /* Number of input buffers */
    int num_outputs;        /* Number of output buffers */
    int num_vregs;          /* Virtual registers used */
    size_t work_size;       /* Total elements to process */
} CMLFusedKernel;

/** Linearize a fusion group into a CMLLinearProgram (public API) */
CMLLinearProgram* cml_linearize_group(const CMLFusionGroup* group);

/** Free a linear program */
void cml_linear_program_free(CMLLinearProgram* prog);

/** Print a linear program for debugging */
void cml_linear_program_print(const CMLLinearProgram* prog);

/** Generate fused kernel from a LinearProgram
 * @param prog The linear program to codegen
 * @param backend Target backend
 * @param work_size Number of elements to process
 * @return Fused kernel source, or NULL on failure. Caller frees.
 */
CMLFusedKernel* cml_fused_codegen(const CMLLinearProgram* prog,
                                    CMLFusedBackend backend,
                                    size_t work_size);

/** Generate fused kernel from a fusion group (convenience) */
CMLFusedKernel* cml_fused_codegen_group(const CMLFusionGroup* group,
                                          CMLFusedBackend backend);

/** Free a fused kernel */
void cml_fused_kernel_free(CMLFusedKernel* kernel);

/** Print fused kernel source */
void cml_fused_kernel_print(const CMLFusedKernel* kernel);

/** Generate fused PTX kernel from LinearProgram */
char* cml_ptx_gen_fused_kernel(const CMLLinearProgram* prog, size_t work_size);

/** Generate fused SPIR-V kernel from LinearProgram */
uint32_t* cml_spirv_gen_fused_kernel(const CMLLinearProgram* prog,
                                      size_t work_size, int* out_num_words);

#ifdef __cplusplus
}
#endif

#endif /* CML_FUSED_CODEGEN_H */
