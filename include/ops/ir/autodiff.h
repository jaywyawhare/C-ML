/*
 * Graph-level reverse-mode autodiff.
 * Emits each primitive's VJP as UOP nodes into the same lazy graph, so every
 * parameter's ->grad becomes a lazy IR subgraph (fusable with forward).
 * Run AFTER cml_ir_decompose (VJPs are defined for the primitive set only).
 */
#ifndef CML_OPS_IR_AUTODIFF_H
#define CML_OPS_IR_AUTODIFF_H

#include "ops/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Reverse-mode gradient over the primitive graph rooted at @p loss_node: seed d(loss)/d(loss)
 * with @p seed (ones when NULL), walk the graph in reverse emitting VJP UOPs, and set v->grad (a
 * lazy tensor) for every requires_grad value. @p differentiable_grads makes those grads
 * re-differentiable (double-backward via tensor_backward create_graph=true).
 * @param ir (CMLGraph_t)
 * @param loss_node (struct IRNode*)
 * @param seed (Tensor*)
 * @param differentiable_grads (bool)
 * @return (int)
 */
int cml_ir_grad(CMLGraph_t ir, struct IRNode* loss_node, Tensor* seed, bool differentiable_grads);

/**
 * 1 if backward should use graph-level autodiff (GRAD_MODE=graph), else 0.
 * @return (int)
 */
int cml_autodiff_use_graph(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_AUTODIFF_H */
