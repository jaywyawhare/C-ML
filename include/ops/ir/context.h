#ifndef CML_OPS_IR_CONTEXT_H
#define CML_OPS_IR_CONTEXT_H

#include "ops/ir/ir.h"
#include "autograd/autograd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* When enabled, all tensor_* operations will automatically be added
   to the specified IR context. Pass NULL to disable. */
int cml_ir_enable_auto_capture(CMLGraph_t ir);
void cml_ir_disable_auto_capture(void);
CMLGraph_t cml_ir_get_auto_capture_context(void);

UOpType cml_ir_optype_to_uoptype(OpType op_type, int num_inputs);

/* Called internally from tensor_* functions when auto-capture is enabled. */
int cml_ir_auto_capture_tensor_op(OpType op_type, Tensor** inputs, int num_inputs, void* params);

/* Returns the global IR context used for lazy evaluation.
   Creates a new one if it doesn't exist. */
CMLGraph_t cml_ir_get_or_create_context(void);
void cml_ir_set_global_context(CMLGraph_t ir);

/* Frees the current global IR context and sets it to NULL.
   Useful for clearing the graph after a training step. */
void cml_ir_reset_global_context(void);

/* Ensure lazy gradient tensors have their data computed before
   resetting the IR context. */
void cml_ir_ensure_gradients_executed(CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_CONTEXT_H
