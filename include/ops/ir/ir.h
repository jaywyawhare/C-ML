/**
 * @file ir.h
 * @brief Intermediate Representation (IR) for code generation
 *
 * Generates optimized C/CUDA/Metal code from uops based on accelerator.
 */

#ifndef CML_CORE_IR_H
#define CML_CORE_IR_H

#include "ops/uops.h"
#include "backend/device.h"
#include "autograd/autograd.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief IR target language/accelerator
 */
typedef enum {
    IR_TARGET_C,      // Plain C (scalar)
    IR_TARGET_C_SIMD, // C with SIMD intrinsics
    IR_TARGET_CUDA,   // CUDA kernels
    IR_TARGET_METAL,  // Metal shaders
    IR_TARGET_OPENCL, // OpenCL kernels
    IR_TARGET_WGSL,   // WebGPU shaders
} IRTarget;

/**
 * @brief IR structure (opaque)
 */
typedef struct CMLGraph* CMLGraph_t;

/**
 * @brief IR Node structure (forward declaration)
 */
struct IRNode;

/**
 * @brief Create new IR for target
 */
CMLGraph_t cml_ir_new(IRTarget target);

/**
 * @brief Free IR
 */
void cml_ir_free(CMLGraph_t ir);

/**
 * @brief Add uop to IR
 */
int cml_ir_add_uop(CMLGraph_t ir, UOpType type, Tensor** inputs, int num_inputs, void* params);

/**
 * @brief Get the last (tail) node added to IR
 * @return IRNode pointer (opaque, use with cml_ir_node_* functions) or NULL
 */
struct IRNode* cml_ir_get_tail(CMLGraph_t ir);

/**
 * @brief Convert UOpType to string for debugging/visualization
 * @param type UOpType to convert
 * @return String representation of UOpType
 */
const char* uop_type_to_string(UOpType type);

/**
 * @brief Compile IR to code
 *
 * @param ir IR to compile
 * @param output_file Output file path (NULL = return string)
 * @return Generated code string (caller must free) or NULL on failure
 */
char* cml_ir_compile(CMLGraph_t ir, const char* output_file);

/**
 * @brief Convert IR to string representation (for debugging)
 *
 * @param ir IR context
 * @return String representation of IR (caller must free)
 */
char* cml_ir_to_string(CMLGraph_t ir);

/**
 * @brief Compute broadcast shape for IR node (semantic rule, not execution)
 *
 * Computes the output shape from input shapes using NumPy-style broadcasting rules.
 * This is a semantic operation, not an execution - it just computes shapes.
 *
 * @param node IR node to compute broadcast shape for
 * @return 0 on success, negative on failure
 */
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

#endif // CML_CORE_IR_H
