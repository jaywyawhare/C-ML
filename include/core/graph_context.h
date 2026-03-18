#ifndef CML_CORE_GRAPH_CONTEXT_H
#define CML_CORE_GRAPH_CONTEXT_H

#include "core/computation_graph.h"
#include "core/graph_context.h"

#ifdef __cplusplus
extern "C" {
#endif

void cml_graph_context_init(void);
void cml_graph_context_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_GRAPH_CONTEXT_H
