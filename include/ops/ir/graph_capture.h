/*
 * GPU graph capture and replay.
 * Captures a sequence of GPU operations into a replayable graph,
 * enabling CUDA Graph / Metal Command Buffer style optimized replay.
 * Eliminates per-kernel launch overhead for repeated execution patterns.
 */

#ifndef CML_GRAPH_CAPTURE_H
#define CML_GRAPH_CAPTURE_H

#include "ops/ir/ir.h"
#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CML_GRAPH_CAPTURE_MAX_NODES 256

typedef enum {
    CML_CAPTURE_IDLE = 0,
    CML_CAPTURE_RECORDING,
    CML_CAPTURE_READY,
    CML_CAPTURE_ERROR,
} CMLCaptureState;

typedef struct CMLCapturedNode {
    UOpType op;
    void* kernel_handle; /* Backend-specific compiled kernel */
    size_t grid[3];
    size_t block[3];
    void** kernel_args;
    int num_args;
    size_t shared_mem;
} CMLCapturedNode;

typedef struct CMLCapturedGraph {
    CMLCapturedNode* nodes;
    int num_nodes;
    int node_capacity;

    CMLCaptureState state;
    int replay_count;

    Tensor** input_bindings;
    int num_input_bindings;
    Tensor** output_bindings;
    int num_output_bindings;

    void* backend_graph;    /* e.g., cudaGraph_t */
    void* backend_instance; /* e.g., cudaGraphExec_t */

    /* Backend teardown hooks captured alongside the handles so the graph can be
     * destroyed without holding a reference to the backend. Each takes the
     * corresponding handle above; NULL if the backend has nothing to free. */
    int (*backend_destroy_instance)(void* instance); /* e.g. cuGraphExecDestroy */
    int (*backend_destroy_graph)(void* graph);       /* e.g. cuGraphDestroy     */

    /* Per-node dispatch, set by the backend. Replay calls this for each node
     * after input substitution; NULL means replay is a timing-only no-op (the
     * previous behaviour). */
    int (*dispatch_fn)(const struct CMLCapturedNode* node, void* user);
    void* dispatch_user;

    /* Input-argument substitution map: entry k says node[sub_node[k]]'s argument
     * sub_arg[k] should be replaced at replay time with input_bindings[sub_in[k]]'s
     * data buffer. This is what lets one capture replay against new input tensors. */
    int* sub_node;
    int* sub_arg;
    int* sub_in;
    int num_subs;
    int sub_capacity;

    double capture_time_ms;
    double last_replay_time_ms;
    double total_replay_time_ms;
} CMLCapturedGraph;

CMLCapturedGraph* cml_graph_capture_create(void);
void cml_graph_capture_free(CMLCapturedGraph* graph);
int cml_graph_capture_begin(CMLCapturedGraph* graph);
int cml_graph_capture_record(CMLCapturedGraph* graph, UOpType op, void* kernel_handle,
                             const size_t grid[3], const size_t block[3], void** args, int num_args,
                             size_t shared_mem);
int cml_graph_capture_end(CMLCapturedGraph* graph);
int cml_graph_capture_replay(CMLCapturedGraph* graph);
int cml_graph_capture_bind_input(CMLCapturedGraph* graph, int index, Tensor* tensor);
int cml_graph_capture_bind_output(CMLCapturedGraph* graph, int index, Tensor* tensor);

/* Install the backend's per-node dispatch callback used by replay. */
int cml_graph_capture_set_dispatch(CMLCapturedGraph* graph,
                                   int (*dispatch_fn)(const struct CMLCapturedNode*, void*),
                                   void* user);

/* Record that argument @p arg_index of node @p node_index is the input bound at
 * @p input_index, so replay substitutes that input's current data buffer. This
 * is how one captured graph runs against fresh input tensors. */
int cml_graph_capture_map_input_arg(CMLCapturedGraph* graph, int node_index, int arg_index,
                                    int input_index);
int cml_graph_capture_reset(CMLCapturedGraph* graph);
CMLCaptureState cml_graph_capture_state(const CMLCapturedGraph* graph);
int cml_graph_capture_num_nodes(const CMLCapturedGraph* graph);
void cml_graph_capture_stats(const CMLCapturedGraph* graph, int* replay_count,
                             double* avg_replay_ms);
void cml_graph_capture_print(const CMLCapturedGraph* graph);

#ifdef __cplusplus
}
#endif

#endif /* CML_GRAPH_CAPTURE_H */
