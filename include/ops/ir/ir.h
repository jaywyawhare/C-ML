#ifndef CML_CORE_IR_H
#define CML_CORE_IR_H

#include "ops/uops.h"
#include "backend/device.h"
#include "autograd/autograd.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    IR_TARGET_C,      // Plain C (scalar)
    IR_TARGET_C_SIMD, // C with SIMD intrinsics
    IR_TARGET_CUDA,   // CUDA kernels
    IR_TARGET_METAL,  // Metal shaders
    IR_TARGET_OPENCL, // OpenCL kernels
    IR_TARGET_WGSL,   // WebGPU shaders
} IRTarget;

typedef struct CMLGraph* CMLGraph_t;

struct IRNode;

CMLGraph_t cml_ir_new(IRTarget target);
void cml_ir_free(CMLGraph_t ir);
int cml_ir_add_uop(CMLGraph_t ir, UOpType type, Tensor** inputs, int num_inputs, void* params);
struct IRNode* cml_ir_get_tail(CMLGraph_t ir);
const char* uop_type_to_string(UOpType type);

/* @param output_file Output file path (NULL = return string) */
char* cml_ir_compile(CMLGraph_t ir, const char* output_file);
char* cml_ir_to_string(CMLGraph_t ir);

/* Computes the output shape from input shapes using NumPy-style broadcasting rules. */
int cml_ir_compute_broadcast_shape(struct IRNode* node);

#ifdef __cplusplus
}
#endif

// Include modular headers
#include "ops/ir/execution.h"
#include "ops/ir/optimization.h"
#include "ops/ir/backward.h"
#include "ops/ir/context.h"
#include "ops/ir/export.h"
#include "ops/ir/decompose.h"

#endif // CML_CORE_IR_H
