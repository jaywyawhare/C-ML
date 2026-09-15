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

/* Seed d(loss)/d(loss)=1 at loss_node and walk the graph in reverse, emitting
 * VJP UOPs and setting v->grad (a lazy tensor) for every requires_grad value. */
/* Reverse-mode gradient over the primitive graph rooted at `loss_node`.
 * `differentiable_grads` controls the requires_grad flag on the published
 * lazy grad tensors: true makes them re-differentiable (double-backward via
 * tensor_backward create_graph=true); false leaves them inert results. */
int cml_ir_grad(CMLGraph_t ir, struct IRNode* loss_node, bool differentiable_grads);

/* 1 if backward should use graph-level autodiff (GRAD_MODE=graph), else 0. */
int cml_autodiff_use_graph(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_AUTODIFF_H */
