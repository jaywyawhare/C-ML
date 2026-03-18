#ifndef CML_CORE_COMPUTATION_GRAPH_H
#define CML_CORE_COMPUTATION_GRAPH_H

#include "tensor/tensor.h"
#include "backend/device.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLComputationGraph* CMLComputationGraph_t;
typedef struct CMLGraphNode* CMLGraphNode_t;

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

typedef enum {
    CML_GRAPH_MODE_EAGER = 0, // Execute immediately (current behavior)
    CML_GRAPH_MODE_LAZY  = 1, // Build graph, execute later
} CMLGraphMode;

typedef struct {
    int n_threads;     // Number of threads for execution
    bool sync;         // Synchronize after execution
    DeviceType device; // Target device
} CMLGraphExecParams;

CMLComputationGraph_t cml_graph_new(void);
void cml_graph_free(CMLComputationGraph_t graph);
void cml_graph_clear(CMLComputationGraph_t graph);
size_t cml_graph_get_node_count(CMLComputationGraph_t graph);
size_t cml_graph_get_leaf_count(CMLComputationGraph_t graph);
CMLGraphNode_t cml_graph_get_node_by_index(CMLComputationGraph_t graph, size_t index);

CMLGraphNode_t cml_graph_node_input(CMLComputationGraph_t graph, Tensor* tensor);
CMLGraphNode_t cml_graph_node_param(CMLComputationGraph_t graph, Tensor* tensor);
CMLGraphNode_t cml_graph_node_op(CMLComputationGraph_t graph, CMLOpType op_type,
                                 CMLGraphNode_t* inputs, int num_inputs, void* op_params);
CMLGraphNode_t cml_graph_node_unary(CMLComputationGraph_t graph, CMLOpType op_type,
                                    CMLGraphNode_t input);
CMLGraphNode_t cml_graph_node_binary(CMLComputationGraph_t graph, CMLOpType op_type,
                                     CMLGraphNode_t a, CMLGraphNode_t b);
CMLGraphNode_t cml_graph_node_matmul(CMLComputationGraph_t graph, CMLGraphNode_t a,
                                     CMLGraphNode_t b);

void cml_graph_build_forward(CMLComputationGraph_t graph, CMLGraphNode_t output);
void cml_graph_build_forward_expand(CMLComputationGraph_t graph, CMLGraphNode_t output);
CMLComputationGraph_t cml_graph_build_backward(CMLComputationGraph_t forward_graph,
                                               CMLGraphNode_t output);

int cml_graph_compute(CMLComputationGraph_t graph, CMLGraphExecParams* params);
int cml_graph_compute_default(CMLComputationGraph_t graph);
int cml_graph_compute_with_context(CMLComputationGraph_t graph, void* context,
                                   CMLGraphExecParams* params);

Tensor* cml_graph_node_get_tensor(CMLGraphNode_t node);
CMLOpType cml_graph_node_get_op_type(CMLGraphNode_t node);
int cml_graph_node_get_num_inputs(CMLGraphNode_t node);
CMLGraphNode_t cml_graph_node_get_input(CMLGraphNode_t node, int index);
bool cml_graph_node_is_leaf(CMLGraphNode_t node);
bool cml_graph_node_is_param(CMLGraphNode_t node);

int cml_graph_optimize(CMLComputationGraph_t graph);
int cml_graph_fuse_ops(CMLComputationGraph_t graph);
int cml_graph_remove_dead_nodes(CMLComputationGraph_t graph);

void cml_set_graph_mode(CMLGraphMode mode);
CMLGraphMode cml_get_graph_mode(void);
void cml_enable_lazy_mode(void);
void cml_disable_lazy_mode(void);

int cml_graph_export_dot(CMLComputationGraph_t graph, const char* filename);
void cml_graph_print(CMLComputationGraph_t graph);

#include "core/graph_context.h"

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_COMPUTATION_GRAPH_H
