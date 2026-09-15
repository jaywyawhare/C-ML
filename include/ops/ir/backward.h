#ifndef CML_OPS_IR_BACKWARD_H
#define CML_OPS_IR_BACKWARD_H

#include "ops/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Traverses the forward graph in reverse and creates backward nodes for autograd.
   Builds the graph structure but doesn't execute it. */
int cml_ir_build_backward(CMLGraph_t ir, struct IRNode* output_node);

/* Backward from the graph tail. */
int cml_ir_execute_backward(CMLGraph_t ir);

/* Backward rooted at `loss_node` (need not be the tail): only the dependency
 * subgraph of loss_node runs. tensor_backward uses this so gradients are
 * computed for the tensor the caller named, not for whatever node happens to
 * have been built last. */
int cml_ir_execute_backward_from(CMLGraph_t ir, struct IRNode* loss_node);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_BACKWARD_H
