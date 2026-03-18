#include "ops/ir/graph_capture.h"
#include "ops/ir/ir.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _POSIX_C_SOURCE
#include <time.h>
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#else
static double get_time_ms(void) { return 0.0; }
#endif

CMLCapturedGraph* cml_graph_capture_create(void) {
    CMLCapturedGraph* g = (CMLCapturedGraph*)calloc(1, sizeof(CMLCapturedGraph));
    if (!g) return NULL;

    g->node_capacity = 32;
    g->nodes = (CMLCapturedNode*)calloc((size_t)g->node_capacity, sizeof(CMLCapturedNode));
    if (!g->nodes) { free(g); return NULL; }

    g->state = CML_CAPTURE_IDLE;
    return g;
}

void cml_graph_capture_free(CMLCapturedGraph* graph) {
    if (!graph) return;

    for (int i = 0; i < graph->num_nodes; i++) {
        free(graph->nodes[i].kernel_args);
    }
    free(graph->nodes);
    free(graph->input_bindings);
    free(graph->output_bindings);
    free(graph);
}

int cml_graph_capture_begin(CMLCapturedGraph* graph) {
    if (!graph) return -1;
    if (graph->state == CML_CAPTURE_RECORDING) return -1;

    for (int i = 0; i < graph->num_nodes; i++) {
        free(graph->nodes[i].kernel_args);
    }
    graph->num_nodes = 0;
    graph->state = CML_CAPTURE_RECORDING;
    graph->capture_time_ms = get_time_ms();
    return 0;
}

int cml_graph_capture_record(CMLCapturedGraph* graph, UOpType op,
                              void* kernel_handle,
                              const size_t grid[3], const size_t block[3],
                              void** args, int num_args, size_t shared_mem) {
    if (!graph || graph->state != CML_CAPTURE_RECORDING) return -1;

    if (graph->num_nodes >= graph->node_capacity) {
        int new_cap = graph->node_capacity * 2;
        if (new_cap > CML_GRAPH_CAPTURE_MAX_NODES)
            new_cap = CML_GRAPH_CAPTURE_MAX_NODES;
        if (graph->num_nodes >= new_cap) {
            graph->state = CML_CAPTURE_ERROR;
            return -1;
        }
        CMLCapturedNode* new_nodes = (CMLCapturedNode*)realloc(
            graph->nodes, (size_t)new_cap * sizeof(CMLCapturedNode));
        if (!new_nodes) { graph->state = CML_CAPTURE_ERROR; return -1; }
        graph->nodes = new_nodes;
        graph->node_capacity = new_cap;
    }

    CMLCapturedNode* node = &graph->nodes[graph->num_nodes];
    memset(node, 0, sizeof(*node));
    node->op = op;
    node->kernel_handle = kernel_handle;
    memcpy(node->grid, grid, sizeof(size_t) * 3);
    memcpy(node->block, block, sizeof(size_t) * 3);
    node->shared_mem = shared_mem;
    node->num_args = num_args;

    if (num_args > 0 && args) {
        node->kernel_args = (void**)malloc((size_t)num_args * sizeof(void*));
        if (!node->kernel_args) { graph->state = CML_CAPTURE_ERROR; return -1; }
        memcpy(node->kernel_args, args, (size_t)num_args * sizeof(void*));
    }

    graph->num_nodes++;
    return 0;
}

int cml_graph_capture_end(CMLCapturedGraph* graph) {
    if (!graph || graph->state != CML_CAPTURE_RECORDING) return -1;

    double end = get_time_ms();
    graph->capture_time_ms = end - graph->capture_time_ms;
    graph->state = CML_CAPTURE_READY;
    graph->replay_count = 0;
    graph->total_replay_time_ms = 0;
    return 0;
}

int cml_graph_capture_replay(CMLCapturedGraph* graph) {
    if (!graph || graph->state != CML_CAPTURE_READY) return -1;

    double start = get_time_ms();

    /*
     * In a full GPU implementation, this would call:
     *   - cudaGraphLaunch(graph->backend_instance, stream)
     *   - or MTLCommandBuffer commit + waitUntilCompleted
     *
     * For CPU fallback, we iterate and "execute" each node.
     * The actual kernel dispatch is handled by the backend.
     */
    for (int i = 0; i < graph->num_nodes; i++) {
        CMLCapturedNode* node = &graph->nodes[i];
        (void)node; /* Backend would dispatch node->kernel_handle here */
    }

    double end = get_time_ms();
    graph->last_replay_time_ms = end - start;
    graph->total_replay_time_ms += graph->last_replay_time_ms;
    graph->replay_count++;
    return 0;
}

int cml_graph_capture_bind_input(CMLCapturedGraph* graph, int index, Tensor* tensor) {
    if (!graph || index < 0) return -1;

    if (index >= graph->num_input_bindings) {
        int new_count = index + 1;
        Tensor** new_bindings = (Tensor**)realloc(
            graph->input_bindings, (size_t)new_count * sizeof(Tensor*));
        if (!new_bindings) return -1;
        for (int i = graph->num_input_bindings; i < new_count; i++)
            new_bindings[i] = NULL;
        graph->input_bindings = new_bindings;
        graph->num_input_bindings = new_count;
    }
    graph->input_bindings[index] = tensor;
    return 0;
}

int cml_graph_capture_bind_output(CMLCapturedGraph* graph, int index, Tensor* tensor) {
    if (!graph || index < 0) return -1;

    if (index >= graph->num_output_bindings) {
        int new_count = index + 1;
        Tensor** new_bindings = (Tensor**)realloc(
            graph->output_bindings, (size_t)new_count * sizeof(Tensor*));
        if (!new_bindings) return -1;
        for (int i = graph->num_output_bindings; i < new_count; i++)
            new_bindings[i] = NULL;
        graph->output_bindings = new_bindings;
        graph->num_output_bindings = new_count;
    }
    graph->output_bindings[index] = tensor;
    return 0;
}

int cml_graph_capture_reset(CMLCapturedGraph* graph) {
    if (!graph) return -1;

    for (int i = 0; i < graph->num_nodes; i++)
        free(graph->nodes[i].kernel_args);
    graph->num_nodes = 0;
    graph->state = CML_CAPTURE_IDLE;
    graph->replay_count = 0;
    graph->total_replay_time_ms = 0;
    graph->last_replay_time_ms = 0;
    return 0;
}

CMLCaptureState cml_graph_capture_state(const CMLCapturedGraph* graph) {
    return graph ? graph->state : CML_CAPTURE_IDLE;
}

int cml_graph_capture_num_nodes(const CMLCapturedGraph* graph) {
    return graph ? graph->num_nodes : 0;
}

void cml_graph_capture_stats(const CMLCapturedGraph* graph,
                              int* replay_count, double* avg_replay_ms) {
    if (!graph) return;
    if (replay_count) *replay_count = graph->replay_count;
    if (avg_replay_ms) {
        *avg_replay_ms = graph->replay_count > 0
            ? graph->total_replay_time_ms / graph->replay_count
            : 0.0;
    }
}

void cml_graph_capture_print(const CMLCapturedGraph* graph) {
    if (!graph) {
        printf("CapturedGraph: NULL\n");
        return;
    }

    const char* state_str;
    switch (graph->state) {
    case CML_CAPTURE_IDLE:      state_str = "IDLE"; break;
    case CML_CAPTURE_RECORDING: state_str = "RECORDING"; break;
    case CML_CAPTURE_READY:     state_str = "READY"; break;
    case CML_CAPTURE_ERROR:     state_str = "ERROR"; break;
    default:                    state_str = "UNKNOWN"; break;
    }

    printf("Captured Graph\n");
    printf("State: %s\n", state_str);
    printf("Nodes: %d\n", graph->num_nodes);
    printf("Inputs: %d, Outputs: %d\n",
           graph->num_input_bindings, graph->num_output_bindings);
    printf("Replays: %d\n", graph->replay_count);
    if (graph->replay_count > 0)
        printf("Avg replay: %.3f ms\n",
               graph->total_replay_time_ms / graph->replay_count);
    printf("Capture time: %.3f ms\n", graph->capture_time_ms);
    printf("\n");
}
