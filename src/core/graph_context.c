/**
 * @file graph_context.c
 * @brief Global graph context for lazy execution
 */

#include "core/computation_graph.h"
#include "core/graph_context.h"
#include "core/logging.h"
#include <stdlib.h>

// Global graph context
static CMLComputationGraph_t g_current_graph = NULL;
static bool g_graph_context_initialized      = false;

/**
 * @brief Get or create current graph
 */
CMLComputationGraph_t cml_get_current_graph(void) {
    if (cml_get_graph_mode() == CML_GRAPH_MODE_EAGER) {
        return NULL; // Eager mode doesn't use graphs
    }

    if (!g_current_graph) {
        g_current_graph = cml_graph_new();
        if (!g_current_graph) {
            LOG_ERROR("Failed to create graph context");
            return NULL;
        }
    }

    return g_current_graph;
}

/**
 * @brief Set current graph
 */
void cml_set_current_graph(CMLComputationGraph_t graph) {
    if (g_current_graph && g_current_graph != graph) {
        cml_graph_free(g_current_graph);
    }
    g_current_graph = graph;
}

/**
 * @brief Clear current graph
 */
void cml_clear_current_graph(void) {
    if (g_current_graph) {
        cml_graph_clear(g_current_graph);
    }
}

/**
 * @brief Free current graph
 */
void cml_free_current_graph(void) {
    if (g_current_graph) {
        cml_graph_free(g_current_graph);
        g_current_graph = NULL;
    }
}

/**
 * @brief Execute current graph
 */
int cml_execute_current_graph(void) {
    if (!g_current_graph) {
        return 0; // No graph to execute
    }

    int result = cml_graph_compute_default(g_current_graph);
    if (result == 0) {
        // Clear graph after execution (optional - could keep for reuse)
        // cml_clear_current_graph();
    }
    return result;
}

/**
 * @brief Initialize graph context
 */
void cml_graph_context_init(void) {
    if (g_graph_context_initialized)
        return;
    g_graph_context_initialized = true;
}

/**
 * @brief Cleanup graph context
 */
void cml_graph_context_cleanup(void) {
    cml_free_current_graph();
    g_graph_context_initialized = false;
}
