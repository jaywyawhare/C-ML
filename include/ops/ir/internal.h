/**
 * @file ir_internal.h
 * @brief Internal IR structure definitions (for ops module only)
 *
 * This header exposes the internal structure of IRNode and CMLIR
 * for use within the ops module (ir.c and uops.c).
 * External code should use the opaque types and accessor functions.
 */

#ifndef CML_OPS_IR_INTERNAL_H
#define CML_OPS_IR_INTERNAL_H

#include "ops/uops.h"
#include "ops/ir/ir.h"
#include "tensor/tensor.h"
#include <stdbool.h>

// Fusion types
typedef enum {
    FUSION_NONE = 0,
    FUSION_FMA,               // MUL + ADD -> FMA
    FUSION_NEG_ADD,           // NEG + ADD -> SUB
    FUSION_EXP_LOG,           // EXP + LOG -> identity
    FUSION_MUL_DIV,           // MUL + DIV -> identity (if same operand)
    FUSION_SQRT_MUL,          // SQRT + MUL -> sqrt_mul
    FUSION_EXP_RECIP,         // EXP + RECIP -> exp_recip
    FUSION_CHAIN_ELEMENTWISE, // Multiple elementwise ops
    FUSION_REDUCE_ELEMENTWISE // Reduction + elementwise
} FusionType;

// Fused kernel structure
typedef struct FusedKernel {
    struct IRNode** ops;    // Operations in this fused kernel
    int num_ops;            // Number of operations
    int capacity;           // Capacity of ops array
    FusionType fusion_type; // Type of fusion
    bool is_chained;        // Whether operations are chained
} FusedKernel;

// Broadcasting info structure
typedef struct BroadcastInfo {
    int* broadcast_dims; // Which dimensions need broadcasting
    int num_broadcast_dims;
    size_t* broadcast_strides; // Strides for broadcasting
} BroadcastInfo;

/**
 * @brief IR Node structure (internal)
 */
struct IRNode {
    UOpType type;
    char** input_names;
    int num_inputs;
    char* output_name;
    void* params;
    struct IRNode* next;

    // Tensor references (for lazy evaluation)
    Tensor** inputs; // Input tensors (lazy)
    Tensor* output;  // Output tensor (lazy facade)

    // Broadcasting (semantic rule - computed, not executed)
    int** input_shapes;
    int* input_ndims;
    int* output_shape; // Computed from broadcasting rule
    int output_ndim;
    BroadcastInfo* broadcast; // Broadcasting metadata

    // Autograd (graph structure, not execution)
    bool requires_grad;
    bool needs_input_grad[8];     // Which inputs need gradients
    struct IRNode* backward_node; // Backward pass node (lazy)
    struct IRNode* forward_node;  // Forward pass node (for backward execution)
    void* saved_for_backward;     // Saved values for backward

    // Execution state
    bool is_executed;       // Has this node been executed?
    void* execution_result; // Cached result (if executed)

    // Optimization
    bool is_used;              // For dead code elimination
    bool is_fused;             // For operation fusion
    FusionType fusion_type;    // Type of fusion applied
    FusedKernel* fused_kernel; // If part of fused kernel
    int use_count;             // Number of nodes using this output
    struct IRNode** users;     // Array of nodes that use this output
    int users_capacity;        // Capacity of users array
    int chain_id;              // ID for chained callables
};

/**
 * @brief IR Structure (internal)
 */
struct CMLIR {
    IRTarget target;
    struct IRNode* head; // Forward graph (lazy)
    struct IRNode* tail;
    struct IRNode* backward_head; // Backward graph (lazy)
    int node_count;

    // Execution state
    bool is_executed;           // Has graph been executed?
    bool is_optimized;          // Has graph been optimized?
    Tensor** execution_results; // Cached execution results
    int execution_results_count;
    int execution_results_capacity;

    // Tensor tracking
    char** tensor_names;
    int tensor_count;
    int tensor_capacity;
    Tensor** tensor_refs; // Track actual tensors for proper name generation
    int tensor_refs_count;
    int tensor_refs_capacity;

    // MLIR Context (lazy initialized)
    void* mlir_ctx;
};

/**
 * @brief Get string name for UOpType
 * @param type The UOp type
 * @return String representation of the UOp type
 */
const char* uop_type_to_string(UOpType type);

/**
 * @brief Find the "other" input in a binary operation
 *
 * When fusing operations like MUL+ADD into FMA, we need to identify which
 * input of the ADD comes from outside the fusion (the "other" input).
 *
 * @param producer The producer node (e.g., MUL)
 * @param consumer The consumer node (e.g., ADD)
 * @return The name of the external input, or NULL if not found
 */
char* find_other_input(struct IRNode* producer, struct IRNode* consumer);

/**
 * @brief Free a fused kernel structure
 * @param kernel The fused kernel to free
 */
void free_fused_kernel(FusedKernel* kernel);

#endif // CML_OPS_IR_INTERNAL_H
