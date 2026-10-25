#ifndef CML_OPS_IR_OPTIMIZATION_H
#define CML_OPS_IR_OPTIMIZATION_H

#include "ops/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run DCE, rewrite rules, fusion tagging and reordering on @p ir in place. Rewrites can leave
 * tensors nothing to compute them, so optimize a cml_ir_view to inspect a graph that will run.
 * @param ir (CMLGraph_t)
 * @return (int)
 */
int cml_ir_optimize(CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_OPTIMIZATION_H
