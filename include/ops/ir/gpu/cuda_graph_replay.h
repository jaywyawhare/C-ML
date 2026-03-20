/*
 * CUDA Graph native capture and replay.
 * Wraps repeated kernel sequences in cudaGraphCreate/cudaGraphLaunch
 * to eliminate per-launch overhead on repeated execution patterns.
 */

#ifndef CML_GPU_CUDA_GRAPH_REPLAY_H
#define CML_GPU_CUDA_GRAPH_REPLAY_H

#include "ops/ir/gpu/cuda_backend.h"
#include "ops/ir/graph_capture.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* CUgraph;
typedef void* CUgraphExec;
typedef void* CUgraphNode;

typedef struct CMLCUDAGraphBackend {
    CMLCUDABackend* backend;
    bool capture_active;

    CUresult (*cuStreamBeginCapture)(CUstream hStream, int mode);
    CUresult (*cuStreamEndCapture)(CUstream hStream, CUgraph* phGraph);
    CUresult (*cuGraphInstantiate)(CUgraphExec* phExec, CUgraph hGraph,
                                   void* logBuffer, size_t bufferSize);
    CUresult (*cuGraphLaunch)(CUgraphExec hExec, CUstream hStream);
    CUresult (*cuGraphExecDestroy)(CUgraphExec hExec);
    CUresult (*cuGraphDestroy)(CUgraph hGraph);

    bool symbols_loaded;
} CMLCUDAGraphBackend;

CMLCUDAGraphBackend* cml_cuda_graph_backend_create(CMLCUDABackend* backend);
void cml_cuda_graph_backend_free(CMLCUDAGraphBackend* gb);

int cml_cuda_graph_begin_capture(CMLCUDAGraphBackend* gb);
int cml_cuda_graph_end_capture(CMLCUDAGraphBackend* gb, CMLCapturedGraph* out);
int cml_cuda_graph_replay(CMLCUDAGraphBackend* gb, CMLCapturedGraph* graph);
void cml_cuda_graph_free(CMLCapturedGraph* graph);

#ifdef __cplusplus
}
#endif

#endif /* CML_GPU_CUDA_GRAPH_REPLAY_H */
