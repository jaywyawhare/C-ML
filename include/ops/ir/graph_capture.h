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
    void* kernel_handle;       /* Backend-specific compiled kernel */
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

    void* backend_graph;       /* e.g., cudaGraph_t */
    void* backend_instance;    /* e.g., cudaGraphExec_t */

    double capture_time_ms;
    double last_replay_time_ms;
    double total_replay_time_ms;
} CMLCapturedGraph;

CMLCapturedGraph* cml_graph_capture_create(void);
void cml_graph_capture_free(CMLCapturedGraph* graph);
int cml_graph_capture_begin(CMLCapturedGraph* graph);
int cml_graph_capture_record(CMLCapturedGraph* graph, UOpType op,
                              void* kernel_handle,
                              const size_t grid[3], const size_t block[3],
                              void** args, int num_args, size_t shared_mem);
int cml_graph_capture_end(CMLCapturedGraph* graph);
int cml_graph_capture_replay(CMLCapturedGraph* graph);
int cml_graph_capture_bind_input(CMLCapturedGraph* graph, int index, Tensor* tensor);
int cml_graph_capture_bind_output(CMLCapturedGraph* graph, int index, Tensor* tensor);
int cml_graph_capture_reset(CMLCapturedGraph* graph);
CMLCaptureState cml_graph_capture_state(const CMLCapturedGraph* graph);
int cml_graph_capture_num_nodes(const CMLCapturedGraph* graph);
void cml_graph_capture_stats(const CMLCapturedGraph* graph,
                              int* replay_count, double* avg_replay_ms);
void cml_graph_capture_print(const CMLCapturedGraph* graph);

#ifdef __cplusplus
}
#endif

#endif /* CML_GRAPH_CAPTURE_H */
