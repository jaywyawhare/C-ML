/**
 * @file backward.h
 * @brief IR backward pass construction and execution
 */

#ifndef CML_OPS_IR_BACKWARD_H
#define CML_OPS_IR_BACKWARD_H

#include "ops/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Build backward graph from forward graph
 *
 * Traverses the forward graph in reverse and creates backward nodes for autograd.
 * This builds the graph structure but doesn't execute it.
 *
 * @param ir IR context
 * @param output_node Output node to start backward pass from
 * @return 0 on success, negative on failure
 */
int cml_ir_build_backward(CMLIR_t ir, struct IRNode* output_node);

/**
 * @brief Execute backward graph
 *
 * Executes the backward graph that was built by cml_ir_build_backward.
 *
 * @param ir IR context
 * @return 0 on success, negative on failure
 */
int cml_ir_execute_backward(CMLIR_t ir);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_BACKWARD_H
