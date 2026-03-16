/**
 * @file computation_graph.c
 * @brief Computation graph implementation
 */

#include "core/computation_graph.h"
#include "core/graph_context.h"
#include "core/logging.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct CMLGraphNode {
    CMLOpType op_type;
    Tensor* tensor;
    CMLGraphNode_t* inputs;
    int num_inputs;
    void* op_params;
    bool is_leaf;
    bool is_param;
    bool visited;
    int execution_order;
};

struct CMLComputationGraph {
    CMLGraphNode_t* nodes;
    size_t num_nodes;
    size_t capacity;
    CMLGraphNode_t* leaf_nodes;
    size_t num_leaves;
    size_t leaf_capacity;
    bool built;
};

static CMLGraphMode g_graph_mode = CML_GRAPH_MODE_EAGER;

static void graph_node_free(CMLGraphNode_t node);
static int graph_execute_node(CMLGraphNode_t node, CMLGraphExecParams* params);
static void graph_topo_sort(CMLComputationGraph_t graph);
static void graph_mark_reachable(CMLGraphNode_t node);

CMLComputationGraph_t cml_graph_new(void) {
    CMLComputationGraph_t graph = malloc(sizeof(struct CMLComputationGraph));
    if (!graph)
        return NULL;

    graph->nodes         = NULL;
    graph->num_nodes     = 0;
    graph->capacity      = 0;
    graph->leaf_nodes    = NULL;
    graph->num_leaves    = 0;
    graph->leaf_capacity = 0;
    graph->built         = false;

    return graph;
}

void cml_graph_free(CMLComputationGraph_t graph) {
    if (!graph)
        return;

    if (graph->nodes) {
        for (size_t i = 0; i < graph->num_nodes; i++) {
            graph_node_free(graph->nodes[i]);
        }
        free(graph->nodes);
    }

    if (graph->leaf_nodes) {
        free(graph->leaf_nodes);
    }

    free(graph);
}

void cml_graph_clear(CMLComputationGraph_t graph) {
    if (!graph)
        return;

    if (graph->nodes) {
        for (size_t i = 0; i < graph->num_nodes; i++) {
            graph_node_free(graph->nodes[i]);
        }
        free(graph->nodes);
    }

    if (graph->leaf_nodes) {
        free(graph->leaf_nodes);
    }

    graph->nodes         = NULL;
    graph->num_nodes     = 0;
    graph->capacity      = 0;
    graph->leaf_nodes    = NULL;
    graph->num_leaves    = 0;
    graph->leaf_capacity = 0;
    graph->built         = false;
}

size_t cml_graph_get_node_count(CMLComputationGraph_t graph) {
    if (!graph)
        return 0;
    return graph->num_nodes;
}

size_t cml_graph_get_leaf_count(CMLComputationGraph_t graph) {
    if (!graph)
        return 0;
    return graph->num_leaves;
}

CMLGraphNode_t cml_graph_get_node_by_index(CMLComputationGraph_t graph, size_t index) {
    if (!graph || index >= graph->num_nodes)
        return NULL;
    return graph->nodes[index];
}

static CMLGraphNode_t graph_node_create(CMLOpType op_type, Tensor* tensor) {
    CMLGraphNode_t node = malloc(sizeof(struct CMLGraphNode));
    if (!node)
        return NULL;

    node->op_type         = op_type;
    node->tensor          = tensor;
    node->inputs          = NULL;
    node->num_inputs      = 0;
    node->op_params       = NULL;
    node->is_leaf         = false;
    node->is_param        = false;
    node->visited         = false;
    node->execution_order = -1;

    return node;
}

static void graph_node_free(CMLGraphNode_t node) {
    if (!node)
        return;

    if (node->inputs) {
        free(node->inputs);
    }

    if (node->op_params) {
        free(node->op_params);
    }

    // Don't free tensor - it's managed separately
    free(node);
}

static void graph_add_node(CMLComputationGraph_t graph, CMLGraphNode_t node) {
    if (!graph || !node)
        return;

    if (graph->num_nodes >= graph->capacity) {
        size_t new_capacity = graph->capacity == 0 ? 16 : graph->capacity * 2;
        CMLGraphNode_t* new_nodes =
            realloc(graph->nodes, (size_t)new_capacity * sizeof(CMLGraphNode_t));
        if (!new_nodes) {
            LOG_ERROR("Failed to expand graph node array");
            return;
        }
        graph->nodes    = new_nodes;
        graph->capacity = new_capacity;
    }

    graph->nodes[graph->num_nodes++] = node;
}

CMLGraphNode_t cml_graph_node_input(CMLComputationGraph_t graph, Tensor* tensor) {
    if (!graph || !tensor)
        return NULL;

    CMLGraphNode_t node = graph_node_create(CML_OP_NONE, tensor);
    if (!node)
        return NULL;

    node->is_leaf = true;
    graph_add_node(graph, node);

    // Add to leaf nodes
    if (graph->num_leaves >= graph->leaf_capacity) {
        size_t new_capacity = graph->leaf_capacity == 0 ? 8 : graph->leaf_capacity * 2;
        CMLGraphNode_t* new_leaves =
            realloc(graph->leaf_nodes, (size_t)new_capacity * sizeof(CMLGraphNode_t));
        if (!new_leaves) {
            graph_node_free(node);
            return NULL;
        }
        graph->leaf_nodes    = new_leaves;
        graph->leaf_capacity = new_capacity;
    }

    graph->leaf_nodes[graph->num_leaves++] = node;

    return node;
}

CMLGraphNode_t cml_graph_node_param(CMLComputationGraph_t graph, Tensor* tensor) {
    if (!graph || !tensor)
        return NULL;

    CMLGraphNode_t node = cml_graph_node_input(graph, tensor);
    if (node) {
        node->is_param = true;
    }
    return node;
}

CMLGraphNode_t cml_graph_node_op(CMLComputationGraph_t graph, CMLOpType op_type,
                                 CMLGraphNode_t* inputs, int num_inputs, void* op_params) {
    if (!graph || !inputs || num_inputs <= 0)
        return NULL;

    CMLGraphNode_t node = graph_node_create(op_type, NULL);
    if (!node)
        return NULL;

    node->inputs = malloc((size_t)num_inputs * sizeof(CMLGraphNode_t));
    if (!node->inputs) {
        graph_node_free(node);
        return NULL;
    }

    memcpy(node->inputs, inputs, (size_t)num_inputs * sizeof(CMLGraphNode_t));
    node->num_inputs = num_inputs;

    if (op_params) {
        // Store op_params pointer (caller retains ownership)
        // For deep copy, would need operation-specific size information
        node->op_params = op_params;
    }

    graph_add_node(graph, node);

    return node;
}

CMLGraphNode_t cml_graph_node_unary(CMLComputationGraph_t graph, CMLOpType op_type,
                                    CMLGraphNode_t input) {
    if (!graph || !input)
        return NULL;
    return cml_graph_node_op(graph, op_type, &input, 1, NULL);
}

CMLGraphNode_t cml_graph_node_binary(CMLComputationGraph_t graph, CMLOpType op_type,
                                     CMLGraphNode_t a, CMLGraphNode_t b) {
    if (!graph || !a || !b)
        return NULL;
    CMLGraphNode_t inputs[2] = {a, b};
    return cml_graph_node_op(graph, op_type, inputs, 2, NULL);
}

CMLGraphNode_t cml_graph_node_matmul(CMLComputationGraph_t graph, CMLGraphNode_t a,
                                     CMLGraphNode_t b) {
    if (!graph || !a || !b)
        return NULL;
    CMLGraphNode_t inputs[2] = {a, b};
    return cml_graph_node_op(graph, CML_OP_MATMUL, inputs, 2, NULL);
}

void cml_graph_build_forward(CMLComputationGraph_t graph, CMLGraphNode_t output) {
    if (!graph || !output)
        return;

    // Mark all nodes as unvisited
    for (size_t i = 0; i < graph->num_nodes; i++) {
        graph->nodes[i]->visited = false;
    }

    // Mark graph as built
    // In a full implementation, this would:
    // 1. Perform topological sort from output
    // 2. Mark all reachable nodes
    // 3. Remove unreachable nodes (dead code elimination)
    // For now, mark all nodes as part of forward graph
    graph->built = true;
}

void cml_graph_build_forward_expand(CMLComputationGraph_t graph, CMLGraphNode_t output) {
    cml_graph_build_forward(graph, output);
}

CMLComputationGraph_t cml_graph_build_backward(CMLComputationGraph_t forward_graph,
                                               CMLGraphNode_t output) {
    if (!forward_graph || !output)
        return NULL;

    // Build backward graph for autograd
    // Note: This computation graph backward construction is separate from IR-based backward.
    // The IR system (used by default) handles backward pass via cml_ir_build_backward().
    // This function is for graph mode visualization/debugging of backward pass.

    CMLComputationGraph_t backward_graph = cml_graph_new();
    if (!backward_graph)
        return NULL;

    // Build backward graph by traversing forward graph in reverse
    // 1. Start from output node
    // 2. For each forward node, create corresponding backward node
    // 3. Connect backward nodes based on forward dependencies

    // Track visited nodes to avoid cycles
    bool* visited = calloc(forward_graph->num_nodes, sizeof(bool));
    if (!visited) {
        cml_graph_free(backward_graph);
        return NULL;
    }

    // Stack for DFS traversal
    CMLGraphNode_t* stack = malloc(forward_graph->num_nodes * sizeof(CMLGraphNode_t));
    if (!stack) {
        free(visited);
        cml_graph_free(backward_graph);
        return NULL;
    }

    int stack_top      = 0;
    stack[stack_top++] = output;

    // Map forward nodes to backward nodes
    CMLGraphNode_t* forward_to_backward = calloc(forward_graph->num_nodes, sizeof(CMLGraphNode_t));
    if (!forward_to_backward) {
        free(stack);
        free(visited);
        cml_graph_free(backward_graph);
        return NULL;
    }

    // Traverse forward graph in reverse (DFS from output)
    while (stack_top > 0) {
        CMLGraphNode_t forward_node = stack[--stack_top];

        // Find forward node index
        size_t forward_idx = SIZE_MAX;
        for (size_t i = 0; i < forward_graph->num_nodes; i++) {
            if (forward_graph->nodes[i] == forward_node) {
                forward_idx = i;
                break;
            }
        }

        if (forward_idx == SIZE_MAX || visited[forward_idx])
            continue;
        visited[forward_idx] = true;

        // Create backward node for this forward node
        CMLOpType backward_op = CML_OP_NONE; // Default backward op
        // Map forward ops to backward ops (basic mapping for graph visualization)
        switch (forward_node->op_type) {
        case CML_OP_ADD:
        case CML_OP_SUB:
        case CML_OP_MUL:
        case CML_OP_DIV:
            backward_op = forward_node->op_type; // Same op for backward
            break;
        case CML_OP_MATMUL:
            backward_op = CML_OP_MATMUL; // MatMul backward
            break;
        case CML_OP_RELU:
        case CML_OP_SIGMOID:
        case CML_OP_TANH:
        case CML_OP_EXP:
        case CML_OP_LOG:
        case CML_OP_SUM:
        case CML_OP_MEAN:
        case CML_OP_TRANSPOSE:
        case CML_OP_RESHAPE:
        case CML_OP_CONV2D:
        case CML_OP_POOL2D:
        case CML_OP_BATCHNORM:
        case CML_OP_DROPOUT:
        case CML_OP_LINEAR:
        case CML_OP_CUSTOM:
        case CML_OP_NONE:
        default:
            backward_op = CML_OP_NONE;
            break;
        }

        // Create backward node (gradient node)
        CMLGraphNode_t backward_node = graph_node_create(backward_op, NULL);
        if (backward_node) {
            // Link to forward node's inputs (for gradient flow)
            if (forward_node->num_inputs > 0) {
                backward_node->inputs =
                    malloc((size_t)forward_node->num_inputs * sizeof(CMLGraphNode_t));
                if (backward_node->inputs) {
                    // For backward, inputs are the forward node's inputs
                    memcpy(backward_node->inputs, forward_node->inputs,
                           (size_t)forward_node->num_inputs * sizeof(CMLGraphNode_t));
                    backward_node->num_inputs = forward_node->num_inputs;
                }
            }

            graph_add_node(backward_graph, backward_node);
            forward_to_backward[forward_idx] = backward_node;
        }

        // Add forward node's inputs to stack (traverse backward)
        for (int i = 0; i < forward_node->num_inputs; i++) {
            if (forward_node->inputs[i]) {
                // Find input node index
                size_t input_idx = SIZE_MAX;
                for (size_t j = 0; j < forward_graph->num_nodes; j++) {
                    if (forward_graph->nodes[j] == forward_node->inputs[i]) {
                        input_idx = j;
                        break;
                    }
                }
                if (input_idx != SIZE_MAX && !visited[input_idx]) {
                    stack[stack_top++] = forward_node->inputs[i];
                }
            }
        }
    }

    // Link backward nodes based on forward dependencies
    for (size_t i = 0; i < backward_graph->num_nodes; i++) {
        CMLGraphNode_t backward_node = backward_graph->nodes[i];
        if (backward_node && backward_node->inputs) {
            // Update inputs to point to corresponding backward nodes
            for (int j = 0; j < backward_node->num_inputs; j++) {
                // Find corresponding backward node for this forward input
                for (size_t k = 0; k < forward_graph->num_nodes; k++) {
                    if (forward_graph->nodes[k] == backward_node->inputs[j]) {
                        if (forward_to_backward[k]) {
                            backward_node->inputs[j] = forward_to_backward[k];
                        }
                        break;
                    }
                }
            }
        }
    }

    free(forward_to_backward);
    free(stack);
    free(visited);

    backward_graph->built = true;
    return backward_graph;
}

static int graph_execute_node(CMLGraphNode_t node, CMLGraphExecParams* params) {
    if (!node)
        return -1;

    // If already executed, skip
    if (node->tensor && node->tensor->data) {
        return 0;
    }

    // Execute inputs first
    for (int i = 0; i < node->num_inputs; i++) {
        if (graph_execute_node(node->inputs[i], params) != 0) {
            return -1;
        }
    }

    // Execute operation based on type
    // Note: This computation graph execution is separate from IR-based execution.
    // IR-based execution (used by default) handles operations via uops.
    // This graph execution is for graph mode visualization/debugging.
    // For actual execution, the IR system should be used instead.
    if (node->tensor && !node->tensor->data) {
        // If tensor exists but not executed, this is a graph mode node
        // In graph mode, execution would happen here, but we use IR mode by default
        /* graph mode is visualization-only; IR handles actual execution */
    }

    return 0;
}

static void graph_topo_sort(CMLComputationGraph_t graph) {
    if (!graph)
        return;

    // Reset execution order
    for (size_t i = 0; i < graph->num_nodes; i++) {
        graph->nodes[i]->execution_order = -1;
        graph->nodes[i]->visited         = false;
    }

    // Simple topological sort (Kahn's algorithm)
    int order    = 0;
    bool changed = true;

    while (changed) {
        changed = false;
        for (size_t i = 0; i < graph->num_nodes; i++) {
            CMLGraphNode_t node = graph->nodes[i];
            if (node->execution_order >= 0)
                continue;

            // Check if all inputs are executed
            bool ready = true;
            for (int j = 0; j < node->num_inputs; j++) {
                if (node->inputs[j]->execution_order < 0) {
                    ready = false;
                    break;
                }
            }

            if (ready || node->is_leaf) {
                node->execution_order = order++;
                changed               = true;
            }
        }
    }
}

int cml_graph_compute(CMLComputationGraph_t graph, CMLGraphExecParams* params) {
    if (!graph)
        return -1;

    if (!graph->built) {
        LOG_WARNING("Graph not built, building now");
        // Build graph from all nodes
    }

    // Topological sort
    graph_topo_sort(graph);

    // Execute nodes in order
    for (int order = 0; order < (int)graph->num_nodes; order++) {
        for (size_t i = 0; i < graph->num_nodes; i++) {
            CMLGraphNode_t node = graph->nodes[i];
            if (node->execution_order == order) {
                if (graph_execute_node(node, params) != 0) {
                    LOG_ERROR("Failed to execute node at order %d", order);
                    return -1;
                }
            }
        }
    }

    return 0;
}

int cml_graph_compute_default(CMLComputationGraph_t graph) {
    CMLGraphExecParams params = {.n_threads = 4, .sync = true, .device = DEVICE_CPU};
    return cml_graph_compute(graph, &params);
}

int cml_graph_compute_with_context(CMLComputationGraph_t graph, void* context,
                                   CMLGraphExecParams* params) {
    // Note: context parameter reserved for future memory management integration
    // Currently, graph execution doesn't use context (IR system handles memory)
    (void)context; // Reserved for future use (intentional)
    return cml_graph_compute(graph, params);
}

Tensor* cml_graph_node_get_tensor(CMLGraphNode_t node) {
    if (!node)
        return NULL;
    return node->tensor;
}

CMLOpType cml_graph_node_get_op_type(CMLGraphNode_t node) {
    if (!node)
        return CML_OP_NONE;
    return node->op_type;
}

int cml_graph_node_get_num_inputs(CMLGraphNode_t node) {
    if (!node)
        return 0;
    return node->num_inputs;
}

CMLGraphNode_t cml_graph_node_get_input(CMLGraphNode_t node, int index) {
    if (!node || index < 0 || index >= node->num_inputs)
        return NULL;
    return node->inputs[index];
}

bool cml_graph_node_is_leaf(CMLGraphNode_t node) {
    if (!node)
        return false;
    return node->is_leaf;
}

bool cml_graph_node_is_param(CMLGraphNode_t node) {
    if (!node)
        return false;
    return node->is_param;
}

int cml_graph_optimize(CMLComputationGraph_t graph) {
    if (!graph)
        return -1;

    cml_graph_remove_dead_nodes(graph);
    cml_graph_fuse_ops(graph);

    return 0;
}

int cml_graph_fuse_ops(CMLComputationGraph_t graph) {
    if (!graph || graph->num_nodes < 2)
        return 0;

    // Operation fusion: identify consecutive operations that can be fused
    // Common patterns: add+relu, mul+add (FMA), etc.

    int fused_count = 0;
    bool* fused     = calloc(graph->num_nodes, sizeof(bool));
    if (!fused)
        return -1;

    // Look for fusion patterns
    for (size_t i = 0; i < graph->num_nodes; i++) {
        if (fused[i])
            continue;

        CMLGraphNode_t node1 = graph->nodes[i];
        if (!node1 || node1->num_inputs == 0)
            continue;

        // Check if this node's output is used by exactly one other node
        CMLGraphNode_t consumer = NULL;
        int consumer_count      = 0;

        for (size_t j = 0; j < graph->num_nodes; j++) {
            if (i == j || fused[j])
                continue;
            CMLGraphNode_t node2 = graph->nodes[j];
            if (!node2)
                continue;

            for (int k = 0; k < node2->num_inputs; k++) {
                if (node2->inputs[k] == node1) {
                    consumer = node2;
                    consumer_count++;
                    break;
                }
            }
        }

        // Only fuse if node1 has exactly one consumer
        if (consumer_count == 1 && consumer) {
            // Check for fusion patterns
            bool can_fuse = false;

            // Pattern 1: ADD + RELU -> FusedAddReLU
            if (node1->op_type == CML_OP_ADD && consumer->op_type == CML_OP_RELU) {
                can_fuse = true;
            }
            // Pattern 2: MUL + ADD -> FMA (Fused Multiply-Add)
            else if (node1->op_type == CML_OP_MUL && consumer->op_type == CML_OP_ADD) {
                can_fuse = true;
            }
            // Pattern 3: Any unary op followed by another unary op
            else if ((node1->op_type == CML_OP_EXP || node1->op_type == CML_OP_LOG) &&
                     (consumer->op_type == CML_OP_EXP || consumer->op_type == CML_OP_LOG)) {
                can_fuse = true;
            }

            if (can_fuse) {
                // Mark nodes as fused
                fused[i] = true;

                // Find consumer index
                size_t consumer_idx = SIZE_MAX;
                for (size_t j = 0; j < graph->num_nodes; j++) {
                    if (graph->nodes[j] == consumer) {
                        consumer_idx = j;
                        break;
                    }
                }

                if (consumer_idx != SIZE_MAX) {
                    fused[consumer_idx] = true;

                    // Update consumer to use node1's inputs directly
                    // This effectively fuses the operations
                    if (consumer->num_inputs > 0 && consumer->inputs) {
                        // Replace consumer's input (node1) with node1's inputs
                        // Basic fusion for graph visualization (IR system handles actual fusion
                        // optimization)
                        free(consumer->inputs);
                        consumer->inputs =
                            malloc((size_t)node1->num_inputs * sizeof(CMLGraphNode_t));
                        if (consumer->inputs) {
                            memcpy(consumer->inputs, node1->inputs,
                                   (size_t)node1->num_inputs * sizeof(CMLGraphNode_t));
                            consumer->num_inputs = node1->num_inputs;
                        }
                    }

                    fused_count++;
                    /* fused */
                }
            }
        }
    }

    free(fused);

    return 0;
}

int cml_graph_remove_dead_nodes(CMLComputationGraph_t graph) {
    if (!graph)
        return -1;

    for (size_t i = 0; i < graph->num_nodes; i++) {
        graph->nodes[i]->visited = false;
    }

    for (size_t i = 0; i < graph->num_nodes; i++) {
        CMLGraphNode_t node = graph->nodes[i];

        // Check if this node is an output (not used as input by any other node)
        bool is_output = true;
        for (size_t j = 0; j < graph->num_nodes; j++) {
            if (i == j)
                continue;
            CMLGraphNode_t other = graph->nodes[j];
            for (int k = 0; k < other->num_inputs; k++) {
                if (other->inputs[k] == node) {
                    is_output = false;
                    break;
                }
            }
            if (!is_output)
                break;
        }

        // If it's an output or a leaf (parameter/input), mark as reachable
        if (is_output || node->is_leaf) {
            // Mark this node and all its inputs as reachable (DFS)
            graph_mark_reachable(node);
        }
    }

    size_t write_idx = 0;
    for (size_t i = 0; i < graph->num_nodes; i++) {
        if (graph->nodes[i]->visited) {
            // Keep this node
            if (write_idx != i) {
                graph->nodes[write_idx] = graph->nodes[i];
            }
            write_idx++;
        } else {
            // Free unreachable node
            if (graph->nodes[i]) {
                free(graph->nodes[i]);
            }
        }
    }

    graph->num_nodes = write_idx;

    return 0;
}

static void graph_mark_reachable(CMLGraphNode_t node) {
    if (!node || node->visited)
        return;

    node->visited = true;

    for (int i = 0; i < node->num_inputs; i++) {
        graph_mark_reachable(node->inputs[i]);
    }
}

void cml_set_graph_mode(CMLGraphMode mode) { g_graph_mode = mode; }

CMLGraphMode cml_get_graph_mode(void) { return g_graph_mode; }

void cml_enable_lazy_mode(void) { g_graph_mode = CML_GRAPH_MODE_LAZY; }

void cml_disable_lazy_mode(void) { g_graph_mode = CML_GRAPH_MODE_EAGER; }

int cml_graph_export_dot(CMLComputationGraph_t graph, const char* filename) {
    if (!graph || !filename)
        return -1;

    FILE* f = fopen(filename, "w");
    if (!f) {
        LOG_ERROR("Failed to open file for graph export: %s", filename);
        return -1;
    }

    fprintf(f, "digraph G {\n");
    fprintf(f, "  rankdir=LR;\n");

    for (size_t i = 0; i < graph->num_nodes; i++) {
        CMLGraphNode_t node = graph->nodes[i];
        fprintf(f, "  node%zu [label=\"%d\"];\n", i, node->op_type);

        for (int j = 0; j < node->num_inputs; j++) {
            // Find input node index
            for (size_t k = 0; k < graph->num_nodes; k++) {
                if (graph->nodes[k] == node->inputs[j]) {
                    fprintf(f, "  node%zu -> node%zu;\n", k, i);
                    break;
                }
            }
        }
    }

    fprintf(f, "}\n");
    fclose(f);

    return 0;
}

void cml_graph_print(CMLComputationGraph_t graph) {
    if (!graph) {
        printf("Graph: NULL\n");
        return;
    }

    printf("Graph: %zu nodes, %zu leaves\n", graph->num_nodes, graph->num_leaves);
    for (size_t i = 0; i < graph->num_nodes; i++) {
        CMLGraphNode_t node = graph->nodes[i];
        printf("  Node %zu: op=%d, inputs=%d, leaf=%d, param=%d\n", i, node->op_type,
               node->num_inputs, node->is_leaf, node->is_param);
    }
}
