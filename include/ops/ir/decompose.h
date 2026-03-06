/**
 * @file decompose.h
 * @brief IR Decomposition Pass — rewrites composite ops into primitive ops
 *
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

/**
 * @brief Decompose composite IR nodes into primitive operations
 *
 * Walks the IR linked list and replaces composite nodes (SIGMOID, TANH,
 * ABS, comparisons, etc.) with chains of primitive nodes (ADD, MUL, NEG,
 * EXP, RECIP, CMPLT, WHERE, SIN, etc.).
 *
 * This pass is idempotent — guarded by ir->is_decomposed flag.
 * Should be called before optimization (fusion can then optimize the
 * primitive chains).
 *
 * @param ir The IR graph to decompose
 * @return 0 on success, -1 on failure
 */
int cml_ir_decompose(CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_DECOMPOSE_H
