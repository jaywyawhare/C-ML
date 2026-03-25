/*
 * Internal IR structure definitions (for ops module only).
 * Exposes the internal structure of IRNode and CMLGraph for use
 * within the ops module. External code should use the opaque types
 * and accessor functions.
 */

#ifndef CML_OPS_IR_INTERNAL_H
#define CML_OPS_IR_INTERNAL_H

#include "ops/uops.h"
#include "ops/ir/ir.h"
#include "ops/ir/intern.h"
#include "tensor/tensor.h"
#include <stdbool.h>

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

typedef struct FusedKernel {
    struct IRNode** ops;    // Operations in this fused kernel
    int num_ops;
    int capacity;
    FusionType fusion_type;
    bool is_chained;
} FusedKernel;

typedef struct BroadcastInfo {
    int* broadcast_dims; // Which dimensions need broadcasting
    int num_broadcast_dims;
    size_t* broadcast_strides;
} BroadcastInfo;

struct IRNode {
    UOpType type;
    char** input_names;
    int num_inputs;
    char* output_name;
    void* params;
    struct IRNode* next;

    uint64_t hash;
    int ref_count;

    Tensor** inputs; // Input tensors (lazy)
    Tensor* output;  // Output tensor (lazy facade)

    int** input_shapes;
    int* input_ndims;
    int* output_shape; // Computed from broadcasting rule
    int output_ndim;
    BroadcastInfo* broadcast;

    bool requires_grad;
    bool needs_input_grad[8];     // Which inputs need gradients
    struct IRNode* backward_node; // Backward pass node (lazy)
    struct IRNode* forward_node;  // Forward pass node (for backward execution)
    void* saved_for_backward;

    bool is_executed;
    void* execution_result;

    bool is_used;              // For dead code elimination
    bool is_fused;             // For operation fusion
    FusionType fusion_type;
    FusedKernel* fused_kernel;
    int use_count;             // Number of nodes using this output
    struct IRNode** users;
    int users_capacity;
    int chain_id;              // ID for chained callables
};

struct CMLGraph {
    IRTarget target;
    struct IRNode* head; // Forward graph (lazy)
    struct IRNode* tail;
    struct IRNode* last_result; // Last node from cml_ir_add_uop (may be interned)
    struct IRNode* backward_head; // Backward graph (lazy)
    int node_count;

    bool is_executed;
    bool is_optimized;
    Tensor** execution_results;
    int execution_results_count;
    int execution_results_capacity;

    char** tensor_names;
    int tensor_count;
    int tensor_capacity;
    Tensor** tensor_refs;
    int tensor_refs_count;
    int tensor_refs_capacity;

    bool is_decomposed;

    CMLInternTable* intern_table;
};

const char* uop_type_to_string(UOpType type);
void free_fused_kernel(FusedKernel* kernel);
int cpu_execute_node(struct IRNode* node);

#endif // CML_OPS_IR_INTERNAL_H
