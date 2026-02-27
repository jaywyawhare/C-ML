/**
 * @file context.h
 * @brief IR auto-capture and global context management
 */

#ifndef CML_OPS_IR_CONTEXT_H
#define CML_OPS_IR_CONTEXT_H

#include "ops/ir/ir.h"
#include "autograd/autograd.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Enable automatic IR capture for tensor_* operations
 *
 * When enabled, all tensor_* operations will automatically be added
 * to the specified IR context. This allows users to write normal tensor_*
 * code and automatically generate backend code.
 *
 * @param ir IR context to capture operations to (NULL to disable)
 * @return 0 on success, negative on failure
 */
int cml_ir_enable_auto_capture(CMLGraph_t ir);

/**
 * @brief Disable automatic IR capture
 */
void cml_ir_disable_auto_capture(void);

/**
 * @brief Get current IR context for auto-capture (if enabled)
 * @return Current IR context or NULL if disabled
 */
CMLGraph_t cml_ir_get_auto_capture_context(void);

/**
 * @brief Convert OpType (from autograd) to UOpType (for IR)
 *
 * This allows automatic conversion when capturing tensor_* operations
 *
 * @param op_type Operation type from autograd
 * @param num_inputs Number of inputs to the operation
 * @return Corresponding UOpType, or UOP_COUNT if no mapping exists
 */
UOpType cml_ir_optype_to_uoptype(OpType op_type, int num_inputs);

/**
 * @brief Automatically capture a tensor operation to IR (internal use)
 *
 * This function is called from tensor_* functions when auto-capture is enabled.
 * Users should not call this directly - it's automatically invoked.
 *
 * @param op_type Operation type from autograd
 * @param inputs Array of input tensors
 * @param num_inputs Number of input tensors
 * @param params Operation-specific parameters (can be NULL)
 * @return 0 on success, negative on failure or if auto-capture is disabled
 */
int cml_ir_auto_capture_tensor_op(OpType op_type, Tensor** inputs, int num_inputs, void* params);

/**
 * @brief Get or create global lazy IR context
 *
 * Returns the global IR context used for lazy evaluation.
 * Creates a new one if it doesn't exist.
 *
 * @return Global IR context
 */
CMLGraph_t cml_ir_get_or_create_context(void);

/**
 * @brief Set global lazy IR context
 *
 * Sets the global IR context for lazy evaluation.
 *
 * @param ir IR context to set (can be NULL to clear)
 */
void cml_ir_set_global_context(CMLGraph_t ir);

/**
 * @brief Reset global lazy IR context
 *
 * Frees the current global IR context and sets it to NULL.
 * This is useful for clearing the graph after a training step.
 */
void cml_ir_reset_global_context(void);

/**
 * @brief Ensure all gradients in the IR context are executed (materialized)
 *
 * This is necessary before resetting the IR context, to ensure that
 * lazy gradient tensors have their data computed and stored.
 */
void cml_ir_ensure_gradients_executed(CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_CONTEXT_H
