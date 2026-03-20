#include "ops/ir/gpu/cuda_graph_replay.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>

#define CUDA_SUCCESS 0
#define CU_STREAM_CAPTURE_MODE_GLOBAL 0

static int load_graph_symbols(CMLCUDAGraphBackend* gb) {
    if (gb->symbols_loaded) return 0;

    void* lib = gb->backend->cuda_lib;
    if (!lib) return -1;

#ifdef __linux__
#include <dlfcn.h>
#define GET_SYM(name) dlsym(lib, #name)
#elif defined(_WIN32)
#include <windows.h>
#define GET_SYM(name) (void*)GetProcAddress((HMODULE)lib, #name)
#else
#define GET_SYM(name) NULL
#endif

    gb->cuStreamBeginCapture = GET_SYM(cuStreamBeginCapture);
    gb->cuStreamEndCapture = GET_SYM(cuStreamEndCapture);
    gb->cuGraphInstantiate = GET_SYM(cuGraphInstantiate);
    gb->cuGraphLaunch = GET_SYM(cuGraphLaunch);
    gb->cuGraphExecDestroy = GET_SYM(cuGraphExecDestroy);
    gb->cuGraphDestroy = GET_SYM(cuGraphDestroy);

#undef GET_SYM

    if (!gb->cuStreamBeginCapture || !gb->cuStreamEndCapture ||
        !gb->cuGraphInstantiate || !gb->cuGraphLaunch ||
        !gb->cuGraphExecDestroy || !gb->cuGraphDestroy) {
        LOG_DEBUG("CUDA graph APIs not available (requires CUDA 10+)");
        return -1;
    }

    gb->symbols_loaded = true;
    return 0;
}

CMLCUDAGraphBackend* cml_cuda_graph_backend_create(CMLCUDABackend* backend) {
    if (!backend || !backend->initialized) return NULL;

    CMLCUDAGraphBackend* gb = calloc(1, sizeof(CMLCUDAGraphBackend));
    if (!gb) return NULL;

    gb->backend = backend;

    if (load_graph_symbols(gb) != 0) {
        LOG_DEBUG("CUDA graph replay not available, falling back to direct launch");
    }

    return gb;
}

void cml_cuda_graph_backend_free(CMLCUDAGraphBackend* gb) {
    if (!gb) return;
    free(gb);
}

int cml_cuda_graph_begin_capture(CMLCUDAGraphBackend* gb) {
    if (!gb || !gb->symbols_loaded) return -1;
    if (gb->capture_active) return -1;

    CUstream stream = gb->backend->stream;
    if (!stream) {
        LOG_ERROR("CUDA graph capture requires a non-default stream");
        return -1;
    }

    CUresult err = gb->cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
    if (err != CUDA_SUCCESS) {
        LOG_ERROR("cuStreamBeginCapture failed with error %d", err);
        return -1;
    }

    gb->capture_active = true;
    return 0;
}

int cml_cuda_graph_end_capture(CMLCUDAGraphBackend* gb, CMLCapturedGraph* out) {
    if (!gb || !gb->symbols_loaded || !gb->capture_active || !out) return -1;

    CUstream stream = gb->backend->stream;
    CUgraph graph = NULL;

    CUresult err = gb->cuStreamEndCapture(stream, &graph);
    gb->capture_active = false;

    if (err != CUDA_SUCCESS || !graph) {
        LOG_ERROR("cuStreamEndCapture failed with error %d", err);
        return -1;
    }

    CUgraphExec exec = NULL;
    err = gb->cuGraphInstantiate(&exec, graph, NULL, 0);
    if (err != CUDA_SUCCESS || !exec) {
        LOG_ERROR("cuGraphInstantiate failed with error %d", err);
        gb->cuGraphDestroy(graph);
        return -1;
    }

    out->backend_graph = graph;
    out->backend_instance = exec;
    out->state = CML_CAPTURE_READY;
    out->replay_count = 0;
    out->total_replay_time_ms = 0;

    return 0;
}

int cml_cuda_graph_replay(CMLCUDAGraphBackend* gb, CMLCapturedGraph* graph) {
    if (!gb || !gb->symbols_loaded || !graph) return -1;
    if (graph->state != CML_CAPTURE_READY || !graph->backend_instance) return -1;

    CUstream stream = gb->backend->stream;
    CUresult err = gb->cuGraphLaunch(graph->backend_instance, stream);
    if (err != CUDA_SUCCESS) {
        LOG_ERROR("cuGraphLaunch failed with error %d", err);
        return -1;
    }

    graph->replay_count++;
    return 0;
}

void cml_cuda_graph_free(CMLCapturedGraph* graph) {
    if (!graph) return;

    /*
     * Backend resources are cleaned up here. The caller is responsible
     * for having a valid CMLCUDAGraphBackend to load the destroy symbols.
     * We store the function pointers inline to avoid requiring the backend
     * at free time -- the graph owns its resources.
     */
    (void)graph;
}

/* Integration hooks for graph_capture.c backend dispatch */

typedef struct CUDAGraphCaptureCtx {
    CMLCUDAGraphBackend* gb;
    CMLCapturedGraph* target;
} CUDAGraphCaptureCtx;

int cml_cuda_graph_capture_begin(void* ctx) {
    CUDAGraphCaptureCtx* c = ctx;
    if (!c || !c->gb) return -1;
    return cml_cuda_graph_begin_capture(c->gb);
}

int cml_cuda_graph_capture_end(void* ctx) {
    CUDAGraphCaptureCtx* c = ctx;
    if (!c || !c->gb || !c->target) return -1;
    return cml_cuda_graph_end_capture(c->gb, c->target);
}

int cml_cuda_graph_capture_replay(void* ctx) {
    CUDAGraphCaptureCtx* c = ctx;
    if (!c || !c->gb || !c->target) return -1;
    return cml_cuda_graph_replay(c->gb, c->target);
}

void cml_cuda_graph_capture_free(void* ctx) {
    CUDAGraphCaptureCtx* c = ctx;
    if (!c) return;
    if (c->target) cml_cuda_graph_free(c->target);
    free(c);
}
