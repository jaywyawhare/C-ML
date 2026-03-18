#ifndef CML_OPS_IR_BACKWARD_H
#define CML_OPS_IR_BACKWARD_H

#include "ops/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Traverses the forward graph in reverse and creates backward nodes for autograd.
   Builds the graph structure but doesn't execute it. */
int cml_ir_build_backward(CMLGraph_t ir, struct IRNode* output_node);

int cml_ir_execute_backward(CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_BACKWARD_H
