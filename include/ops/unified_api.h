/**
 * @file unified_api.h
 * @brief Unified API for PyTorch-like and tinygrad-like operations
 *
 * This header provides a unified interface that supports:
 * 1. High-level tensor operations (PyTorch-like: tensor_*)
 * 2. Low-level uops (tinygrad-like: uop_*)
 * 3. User-defined custom operations/layers
 *
 * All operations support:
 * - Autograd (automatic differentiation)
 * - Backend code generation (CUDA, Metal, OpenCL, SIMD)
 * - IR capture for optimization
 */

#ifndef CML_UNIFIED_API_H
#define CML_UNIFIED_API_H

#include "tensor/tensor.h"
#include "ops/uops.h"
#include "ops/ir/ir.h"
#include "autograd/autograd.h"
#include "nn.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Forward function signature for custom operations
 *
 * @param inputs Array of input tensors
 * @param num_inputs Number of input tensors
 * @param params Operation-specific parameters (can be NULL)
 * @return Output tensor (caller must free)
 */
typedef Tensor* (*CustomForwardFn)(Tensor** inputs, int num_inputs, void* params);

/**
 * @brief Backward function signature for custom operations
 *
 * @param node IR node containing operation context
 * @param grad_output Gradient with respect to output
 */
typedef void (*CustomBackwardFn)(struct IRNode* node, Tensor* grad_output);

/**
 * @brief Code generator function for custom operations
 *
 * Generates backend-specific code for the custom operation
 *
 * @param op_name Name of the operation
 * @param inputs Array of input tensor names
 * @param num_inputs Number of inputs
 * @param output_name Output tensor name
 * @param params Operation parameters
 * @param target Target backend (CUDA, Metal, etc.)
 * @param code_buffer Buffer to write generated code
 * @param buffer_size Size of buffer
 * @return Number of bytes written, or negative on error
 */
typedef int (*CustomCodeGenFn)(const char* op_name, const char** inputs, int num_inputs,
                               const char* output_name, void* params, IRTarget target,
                               char* code_buffer, size_t buffer_size);

/**
 * @brief Custom operation descriptor
 */
typedef struct {
    const char* name;             // Operation name
    OpType op_type;               // Autograd operation type (OP_CUSTOM or specific)
    UOpType uop_type;             // UOp type (UOP_COUNT if not a uop)
    CustomForwardFn forward_fn;   // Forward pass implementation
    CustomBackwardFn backward_fn; // Backward pass implementation (can be NULL)
    CustomCodeGenFn codegen_fn;   // Code generator (can be NULL)
    void* default_params;         // Default parameters (can be NULL)
    size_t params_size;           // Size of parameters structure
} CustomOpDescriptor;

/**
 * @brief Register a custom operation
 *
 * This allows users to define custom operations that:
 * - Support autograd (if backward_fn provided)
 * - Can be captured to IR for code generation
 * - Work at tensor_* level (high-level) or uop_* level (low-level)
 *
 * @param desc Operation descriptor
 * @return 0 on success, negative on failure
 */
int cml_register_custom_op(const CustomOpDescriptor* desc);

/**
 * @brief Unregister a custom operation
 *
 * @param name Operation name
 * @return 0 on success, negative if not found
 */
int cml_unregister_custom_op(const char* name);

/**
 * @brief Execute a registered custom operation
 *
 * @param name Operation name
 * @param inputs Input tensors
 * @param num_inputs Number of inputs
 * @param params Operation parameters (can be NULL)
 * @return Output tensor or NULL on failure
 */
Tensor* cml_execute_custom_op(const char* name, Tensor** inputs, int num_inputs, void* params);

/**
 * @brief Operation level/abstraction
 */
typedef enum {
    CML_OP_LEVEL_USER,   // User-defined layers/modules (highest level)
    CML_OP_LEVEL_TENSOR, // High-level tensor operations (tensor_*)
    CML_OP_LEVEL_UOP     // Low-level micro-operations (uop_*)
} CMLOpLevel;

/**
 * @brief Unified operation context
 *
 * This structure allows operations to be executed at any level
 * while maintaining autograd and IR capture support.
 */
typedef struct {
    CMLOpLevel level;   // Operation level
    const char* name;   // Operation name
    OpType op_type;     // Autograd operation type
    UOpType uop_type;   // UOp type (if applicable)
    Tensor** inputs;    // Input tensors
    int num_inputs;     // Number of inputs
    void* params;       // Operation parameters
    bool requires_grad; // Whether gradients are needed
    bool capture_to_ir; // Whether to capture to IR
} CMLOpContext;

/**
 * @brief Execute operation at any level with unified interface
 *
 * This function provides a unified way to execute operations regardless
 * of whether they're user-defined, tensor_*, or uop_* operations.
 *
 * @param ctx Operation context
 * @return Output tensor or NULL on failure
 */
Tensor* cml_execute_unified_op(CMLOpContext* ctx);

/**
 * @brief Create operation context for tensor-level operation
 */
CMLOpContext cml_op_context_tensor(OpType op_type, Tensor** inputs, int num_inputs, void* params);

/**
 * @brief Create operation context for uop-level operation
 */
CMLOpContext cml_op_context_uop(UOpType uop_type, Tensor** inputs, int num_inputs, void* params);

/**
 * @brief Create operation context for user-defined operation
 */
CMLOpContext cml_op_context_user(const char* op_name, Tensor** inputs, int num_inputs,
                                 void* params);

/**
 * @brief Enable automatic IR capture for all operation levels
 *
 * When enabled, operations at all levels (user, tensor, uop) will
 * automatically be captured to the IR context for code generation.
 *
 * @param ir IR context to capture to
 * @return 0 on success, negative on failure
 */
int cml_enable_unified_ir_capture(CMLIR_t ir);

/**
 * @brief Disable automatic IR capture
 */
void cml_disable_unified_ir_capture(void);

/**
 * @brief Check if unified IR capture is enabled
 * @return true if enabled, false otherwise
 */
bool cml_is_unified_ir_capture_enabled(void);

/**
 * @brief Enable autograd for custom operation
 *
 * This ensures that custom operations participate in the autograd system
 * and gradients can flow through them.
 *
 * @param op_name Operation name
 * @param backward_fn Backward function
 * @return 0 on success, negative on failure
 */
int cml_enable_autograd_for_custom_op(const char* op_name, CustomBackwardFn backward_fn);

#ifdef __cplusplus
}
#endif

#endif // CML_UNIFIED_API_H
