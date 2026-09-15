/*
 * IR Decomposition Pass -- rewrites composite ops into primitive ops.
 * Reduces the number of ops each backend must implement by decomposing
 * composite operations (sigmoid, tanh, abs, comparisons, etc.) into
 * chains of ~28 primitive ALU ops before execution.
 */

#ifndef CML_OPS_IR_DECOMPOSE_H
#define CML_OPS_IR_DECOMPOSE_H

#include "ops/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Idempotent -- guarded by ir->is_decomposed flag.
   Should be called before optimization (fusion can then optimize the
   primitive chains). */
int cml_ir_decompose(CMLGraph_t ir);

/* Decompose only the node range starting at `start` (NULL means head).
 * Used by double-backward to lower ops appended between two grad passes
 * without touching already-emitted backward nodes, which a whole-graph
 * re-lower would corrupt. Marks the context decomposed like the full pass. */
int cml_ir_decompose_from(CMLGraph_t ir, struct IRNode* start);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_DECOMPOSE_H
