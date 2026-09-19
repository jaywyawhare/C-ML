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
    struct IRNode** ops; // Operations in this fused kernel
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
    char* scope; /* "Sequential/Linear" module path; NULL unless VIZ is on */
    /* Folded C call stack at the moment this node was built, root-first and
     * ';'-separated, e.g. "main;training_example;module_forward;nn_linear_forward".
     * A lazy graph has no stack to sample at execution time -- by then everything
     * runs from one executor loop -- so the meaningful stack is the one that
     * *created* the node. Captured only under FLAMEGRAPH; NULL otherwise. */
    char* build_stack;
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
    /* For zero-input creation ops (FILL, CONST, RAND, ARANGE, EYE, ALLOC):
     * dtype/device are set explicitly here since there are no input tensors to
     * infer from.  Zero value = DTYPE_FLOAT32 / DEVICE_CPU (correct defaults). */
    DType output_dtype;
    DeviceType output_device;
    BroadcastInfo* broadcast;

    bool requires_grad;
    bool needs_input_grad[8];     // Which inputs need gradients
    struct IRNode* backward_node; // Backward pass node (lazy)
    struct IRNode* forward_node;  // Forward pass node (for backward execution)
    void* saved_for_backward;

    bool is_executed;
    void* execution_result;

    bool is_used;  // For dead code elimination
    bool is_fused; // For operation fusion
    FusionType fusion_type;
    FusedKernel* fused_kernel;
    int use_count; // Number of nodes using this output
    struct IRNode** users;
    int users_capacity;
    int chain_id; // ID for chained callables
};

void cml_ir_free_node_params(struct IRNode* node);
/* Canonical teardown of a node's owned storage (params/scope/build_stack/
 * shapes/...) without freeing the node or touching ref_count. */
void cml_ir_release_node_storage(struct IRNode* node);

struct CMLGraph {
    IRTarget target;
    struct IRNode* head; // Forward graph (lazy)
    struct IRNode* tail;
    struct IRNode* last_result;   // Last node from cml_ir_add_uop (may be interned)
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

    /* Set once cml_ir_grad has emitted VJP nodes into this context. A
     * re-decompose over forward+backward nodes corrupts references (see the
     * note at the end of autodiff.c), so a second grad pass (double-backward)
     * lowers only nodes appended after decomposed_frontier. */
    bool has_backward_nodes;
    struct IRNode* decomposed_frontier;

    /* Values that received a lazy grad from a grad pass of THIS context.
     * Double-backward uses it to zero stale differentiable grads that a
     * later pass never reached (see autodiff.c). */
    Tensor** grad_publish_log;
    int grad_publish_count;
    int grad_publish_cap;

    CMLInternTable* intern_table;
};

/* FNV-1a, shared by the graph hash and the node intern table. */
#define CML_FNV_OFFSET_BASIS 0xcbf29ce484222325ULL
#define CML_FNV_PRIME 0x100000001b3ULL

uint64_t cml_fnv1a_bytes(uint64_t hash, const void* data, size_t len);

/* Structural hash of `ir`'s forward graph: op type, output shape and input
 * count of every node. Two graphs with the same hash replay interchangeably. */
uint64_t cml_ir_graph_hash(CMLGraph_t ir);

/* Collect the output data pointers of `ir`'s forward nodes into `ptrs` (at most
 * `max`), one slot per node, NULL where a node has no realized output. Returns
 * how many slots were written -- the tensor binding a trace replay expects. */
int cml_ir_output_slots(CMLGraph_t ir, void** ptrs, int max);

/* Graph-editing primitives shared by the IR rewrite passes (pattern matcher,
 * tree automaton, tensor-core opt, peephole optimizer). */
struct IRNode* cml_ir_find_by_output(CMLGraph_t ir, const char* output_name);
void cml_ir_unlink_node(CMLGraph_t ir, struct IRNode* node);

/* Repoint every input reference to `old_name` at `new_name`. */
void cml_ir_replace_refs(CMLGraph_t ir, const char* old_name, const char* new_name);

/* Splice `new_node` in just before `before`, or at the tail when `before` is
 * NULL or not in the graph. */
void cml_ir_insert_before(CMLGraph_t ir, struct IRNode* new_node, struct IRNode* before);

/* Module scope tracking for the graph view. Without it graph.json is a flat
 * list of primitives -- after decomposition a real model is thousands of nodes
 * and no one can find anything. module_forward pushes/pops around every layer,
 * so nesting (Sequential/Linear) falls out for free. Only records under VIZ:
 * outside the dashboard the strdup per node is pure overhead. */
void cml_ir_scope_push(const char* name);
void cml_ir_scope_pop(void);
const char* cml_ir_scope_current(void);
bool cml_ir_scope_enabled(void);

const char* uop_type_to_string(UOpType type);
void free_fused_kernel(FusedKernel* kernel);
int cpu_execute_node(struct IRNode* node);

#endif // CML_OPS_IR_INTERNAL_H
