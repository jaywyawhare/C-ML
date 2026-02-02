#ifndef CML_CORE_GRAPH_CONTEXT_H
#define CML_CORE_GRAPH_CONTEXT_H

#include "core/computation_graph.h"
#include "core/graph_context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Graph Context (Global)

/**
 * @brief Get or create current graph context
 */
CMLComputationGraph_t cml_get_current_graph(void);

/**
 * @brief Set current graph context
 */
void cml_set_current_graph(CMLComputationGraph_t graph);

/**
 * @brief Clear current graph
 */
void cml_clear_current_graph(void);

/**
 * @brief Free current graph
 */
void cml_free_current_graph(void);

/**
 * @brief Execute current graph
 */
int cml_execute_current_graph(void);

/**
 * @brief Initialize graph context
 */
void cml_graph_context_init(void);

/**
 * @brief Cleanup graph context
 */
void cml_graph_context_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_GRAPH_CONTEXT_H
