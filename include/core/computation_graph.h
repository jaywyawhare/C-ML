/**
 * @file computation_graph.h
 * @brief Computation graph for lazy execution (inspired by ggml)
 *
 * Provides explicit graph building and lazy execution:
 * - Build computation graph without executing
 * - Optimize graph before execution
 * - Execute graph efficiently
 * - Support both forward and backward passes
 */

#ifndef CML_CORE_COMPUTATION_GRAPH_H
#define CML_CORE_COMPUTATION_GRAPH_H

#include "tensor/tensor.h"
#include "backend/device.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Computation graph (opaque pointer)
 */
typedef struct CMLComputationGraph* CMLComputationGraph_t;

/**
 * @brief Graph node (opaque pointer)
 */
typedef struct CMLGraphNode* CMLGraphNode_t;

/**
 * @brief Operation type
 */
typedef enum {
    CML_OP_NONE = 0,
    CML_OP_ADD,
    CML_OP_SUB,
    CML_OP_MUL,
    CML_OP_DIV,
    CML_OP_MATMUL,
    CML_OP_RELU,
    CML_OP_SIGMOID,
    CML_OP_TANH,
    CML_OP_EXP,
    CML_OP_LOG,
    CML_OP_SUM,
    CML_OP_MEAN,
    CML_OP_TRANSPOSE,
    CML_OP_RESHAPE,
    CML_OP_CONV2D,
    CML_OP_POOL2D,
    CML_OP_BATCHNORM,
    CML_OP_DROPOUT,
    CML_OP_LINEAR,
    CML_OP_CUSTOM, // Custom operation
} CMLOpType;

/**
 * @brief Graph execution mode
 */
typedef enum {
    CML_GRAPH_MODE_EAGER = 0, // Execute immediately (current behavior)
    CML_GRAPH_MODE_LAZY  = 1, // Build graph, execute later
} CMLGraphMode;

/**
 * @brief Graph execution parameters
 */
typedef struct {
    int n_threads;     // Number of threads for execution
    bool sync;         // Synchronize after execution
    DeviceType device; // Target device
} CMLGraphExecParams;

/**
 * @brief Create new computation graph
 */
CMLComputationGraph_t cml_graph_new(void);

/**
 * @brief Free computation graph
 */
void cml_graph_free(CMLComputationGraph_t graph);

/**
 * @brief Clear graph (remove all nodes, keep structure)
 */
void cml_graph_clear(CMLComputationGraph_t graph);

/**
 * @brief Get number of nodes in graph
 */
size_t cml_graph_get_node_count(CMLComputationGraph_t graph);

/**
 * @brief Get number of leaf nodes (inputs) in graph
 */
size_t cml_graph_get_leaf_count(CMLComputationGraph_t graph);

/**
 * @brief Get node by index (for iteration)
 */
CMLGraphNode_t cml_graph_get_node_by_index(CMLComputationGraph_t graph, size_t index);

/**
 * @brief Create input node (leaf node)
 */
CMLGraphNode_t cml_graph_node_input(CMLComputationGraph_t graph, Tensor* tensor);

/**
 * @brief Create parameter node (trainable)
 */
CMLGraphNode_t cml_graph_node_param(CMLComputationGraph_t graph, Tensor* tensor);

/**
 * @brief Create operation node
 */
CMLGraphNode_t cml_graph_node_op(CMLComputationGraph_t graph, CMLOpType op_type,
                                 CMLGraphNode_t* inputs, int num_inputs, void* op_params);

/**
 * @brief Create unary operation node
 */
CMLGraphNode_t cml_graph_node_unary(CMLComputationGraph_t graph, CMLOpType op_type,
                                    CMLGraphNode_t input);

/**
 * @brief Create binary operation node
 */
CMLGraphNode_t cml_graph_node_binary(CMLComputationGraph_t graph, CMLOpType op_type,
                                     CMLGraphNode_t a, CMLGraphNode_t b);

/**
 * @brief Create matmul node
 */
CMLGraphNode_t cml_graph_node_matmul(CMLComputationGraph_t graph, CMLGraphNode_t a,
                                     CMLGraphNode_t b);

/**
 * @brief Build forward graph from output node
 *
 * Traverses from output backwards to build complete forward graph
 */
void cml_graph_build_forward(CMLComputationGraph_t graph, CMLGraphNode_t output);

/**
 * @brief Build forward graph and expand (add all dependencies)
 */
void cml_graph_build_forward_expand(CMLComputationGraph_t graph, CMLGraphNode_t output);

/**
 * @brief Build backward graph from output node
 */
CMLComputationGraph_t cml_graph_build_backward(CMLComputationGraph_t forward_graph,
                                               CMLGraphNode_t output);

/**
 * @brief Execute computation graph
 *
 * @param graph Computation graph to execute
 * @param params Execution parameters
 * @return 0 on success, negative on failure
 */
int cml_graph_compute(CMLComputationGraph_t graph, CMLGraphExecParams* params);

/**
 * @brief Execute graph with default parameters
 */
int cml_graph_compute_default(CMLComputationGraph_t graph);

/**
 * @brief Execute graph with context (for memory management)
 */
int cml_graph_compute_with_context(CMLComputationGraph_t graph, void* context,
                                   CMLGraphExecParams* params);

/**
 * @brief Get tensor from graph node
 */
Tensor* cml_graph_node_get_tensor(CMLGraphNode_t node);

/**
 * @brief Get operation type of node
 */
CMLOpType cml_graph_node_get_op_type(CMLGraphNode_t node);

/**
 * @brief Get number of input nodes
 */
int cml_graph_node_get_num_inputs(CMLGraphNode_t node);

/**
 * @brief Get input node at index
 */
CMLGraphNode_t cml_graph_node_get_input(CMLGraphNode_t node, int index);

/**
 * @brief Check if node is leaf (input/parameter)
 */
bool cml_graph_node_is_leaf(CMLGraphNode_t node);

/**
 * @brief Check if node is parameter (trainable)
 */
bool cml_graph_node_is_param(CMLGraphNode_t node);

/**
 * @brief Optimize graph (fuse operations, remove dead nodes, etc.)
 */
int cml_graph_optimize(CMLComputationGraph_t graph);

/**
 * @brief Fuse consecutive operations where possible
 */
int cml_graph_fuse_ops(CMLComputationGraph_t graph);

/**
 * @brief Remove dead nodes (unreachable from outputs)
 */
int cml_graph_remove_dead_nodes(CMLComputationGraph_t graph);

/**
 * @brief Set global graph mode (eager or lazy)
 */
void cml_set_graph_mode(CMLGraphMode mode);

/**
 * @brief Get current graph mode
 */
CMLGraphMode cml_get_graph_mode(void);

/**
 * @brief Enable lazy execution mode
 */
void cml_enable_lazy_mode(void);

/**
 * @brief Disable lazy execution mode (use eager)
 */
void cml_disable_lazy_mode(void);

/**
 * @brief Export graph to DOT format (for visualization)
 */
int cml_graph_export_dot(CMLComputationGraph_t graph, const char* filename);

/**
 * @brief Print graph structure
 */
void cml_graph_print(CMLComputationGraph_t graph);

// Graph Context (Global)
// Moved to core/graph_context.h
#include "core/graph_context.h"

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_COMPUTATION_GRAPH_H
