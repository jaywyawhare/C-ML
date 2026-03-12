/**
 * @file graph_capture.h
 * @brief GPU graph capture and replay
 *
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

/** Maximum kernels in a captured graph */
#define CML_GRAPH_CAPTURE_MAX_NODES 256

/** Graph capture state */
typedef enum {
    CML_CAPTURE_IDLE = 0,
    CML_CAPTURE_RECORDING,
    CML_CAPTURE_READY,
    CML_CAPTURE_ERROR,
} CMLCaptureState;

/** Captured kernel node */
typedef struct CMLCapturedNode {
    UOpType op;
    void* kernel_handle;       /* Backend-specific compiled kernel */
    size_t grid[3];            /* Grid dimensions */
    size_t block[3];           /* Block dimensions */
    void** kernel_args;        /* Pointer to kernel arguments */
    int num_args;
    size_t shared_mem;         /* Shared memory bytes */
} CMLCapturedNode;

/** Captured graph for replay */
typedef struct CMLCapturedGraph {
    CMLCapturedNode* nodes;
    int num_nodes;
    int node_capacity;

    CMLCaptureState state;
    int replay_count;          /* Number of times replayed */

    /* Memory bindings: map placeholder -> actual tensor */
    Tensor** input_bindings;
    int num_input_bindings;
    Tensor** output_bindings;
    int num_output_bindings;

    /* Backend handle (e.g., cudaGraph_t) */
    void* backend_graph;
    void* backend_instance;    /* e.g., cudaGraphExec_t */

    /* Timing */
    double capture_time_ms;
    double last_replay_time_ms;
    double total_replay_time_ms;
} CMLCapturedGraph;

/** Create a new captured graph */
CMLCapturedGraph* cml_graph_capture_create(void);

/** Free captured graph */
void cml_graph_capture_free(CMLCapturedGraph* graph);

/** Begin recording operations */
int cml_graph_capture_begin(CMLCapturedGraph* graph);

/** Record a kernel launch into the graph */
int cml_graph_capture_record(CMLCapturedGraph* graph, UOpType op,
                              void* kernel_handle,
                              const size_t grid[3], const size_t block[3],
                              void** args, int num_args, size_t shared_mem);

/** End recording and finalize the graph */
int cml_graph_capture_end(CMLCapturedGraph* graph);

/** Replay the captured graph */
int cml_graph_capture_replay(CMLCapturedGraph* graph);

/** Update input bindings for replay with different data */
int cml_graph_capture_bind_input(CMLCapturedGraph* graph, int index, Tensor* tensor);

/** Update output bindings */
int cml_graph_capture_bind_output(CMLCapturedGraph* graph, int index, Tensor* tensor);

/** Reset graph for re-recording */
int cml_graph_capture_reset(CMLCapturedGraph* graph);

/** Get capture state */
CMLCaptureState cml_graph_capture_state(const CMLCapturedGraph* graph);

/** Get number of captured nodes */
int cml_graph_capture_num_nodes(const CMLCapturedGraph* graph);

/** Get replay statistics */
void cml_graph_capture_stats(const CMLCapturedGraph* graph,
                              int* replay_count, double* avg_replay_ms);

/** Print captured graph summary */
void cml_graph_capture_print(const CMLCapturedGraph* graph);

#ifdef __cplusplus
}
#endif

#endif /* CML_GRAPH_CAPTURE_H */
