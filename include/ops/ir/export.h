#ifndef CML_OPS_IR_EXPORT_H
#define CML_OPS_IR_EXPORT_H

#include "ops/ir/ir.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Exports detailed information about kernels, optimizations, dead code,
   and fusion opportunities for visualization in the Kernel Studio. */
char* cml_ir_export_kernel_analysis(CMLGraph_t ir, bool optimized);

/* Exports the graph structure (nodes and edges) for visualization.
   Uses the same node ordering as cml_ir_export_kernel_analysis. */
char* cml_ir_export_graph_json(CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_EXPORT_H
