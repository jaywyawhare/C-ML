/**
 * @file graph_context.c
 * @brief Global graph context for lazy execution
 */

#include "core/computation_graph.h"
#include "core/graph_context.h"
#include "core/logging.h"
#include <stdlib.h>

static CMLComputationGraph_t g_current_graph = NULL;
static bool g_graph_context_initialized      = false;

void cml_free_current_graph(void) {
    if (g_current_graph) {
        cml_graph_free(g_current_graph);
        g_current_graph = NULL;
    }
}

void cml_graph_context_init(void) {
    if (g_graph_context_initialized)
        return;
    g_graph_context_initialized = true;
}

void cml_graph_context_cleanup(void) {
    cml_free_current_graph();
    g_graph_context_initialized = false;
}
