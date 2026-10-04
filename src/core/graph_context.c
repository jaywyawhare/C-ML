#include "core/computation_graph.h"
#include "core/graph_context.h"
#include "core/logging.h"
#include <stdlib.h>

static CMLComputationGraph_t g_current_graph = NULL;
static bool g_graph_context_initialized      = false;

/** Free the thread's current computation graph, if any, and clear the slot. */
void cml_free_current_graph(void) {
    if (g_current_graph) {
        cml_graph_free(g_current_graph);
        g_current_graph = NULL;
    }
}

/** Initialize the graph context subsystem; idempotent. */
void cml_graph_context_init(void) {
    if (g_graph_context_initialized)
        return;
    g_graph_context_initialized = true;
}

/** Release the current graph and mark the context uninitialized. */
void cml_graph_context_cleanup(void) {
    cml_free_current_graph();
    g_graph_context_initialized = false;
}
