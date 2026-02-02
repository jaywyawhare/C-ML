/**
 * @file export.h
 * @brief IR export functionality
 */

#ifndef CML_OPS_IR_EXPORT_H
#define CML_OPS_IR_EXPORT_H

#include "ops/ir/ir.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Export kernel analysis data as JSON
 *
 * Exports detailed information about kernels, optimizations, dead code,
 * and fusion opportunities for visualization in the Kernel Studio.
 *
 * @param ir IR context
 * @param optimized Whether to export optimized or unoptimized view
 * @return JSON string (caller must free) or NULL on failure
 */
char* cml_ir_export_kernel_analysis(CMLIR_t ir, bool optimized);

/**
 * @brief Export graph topology as JSON
 *
 * Exports the graph structure (nodes and edges) for visualization.
 * Uses the same node ordering as cml_ir_export_kernel_analysis to ensure
 * consistency between views.
 *
 * @param ir IR context
 * @return JSON string (caller must free) or NULL on failure
 */
char* cml_ir_export_graph_json(CMLIR_t ir);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_EXPORT_H
