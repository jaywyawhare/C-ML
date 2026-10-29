#ifndef CML_AUTOGRAD_FORWARD_MODE_H
#define CML_AUTOGRAD_FORWARD_MODE_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Forward function rebuilt per call: maps an input tensor to an output tensor.
 * @param x (Tensor*)
 * @param ctx (void*) user context
 * @return (Tensor*)
 */
typedef Tensor* (*CMLForwardFn)(Tensor* x, void* ctx);

/**
 * Forward-mode Jacobian-vector product J*v, evaluated by the forward-over-reverse
 * (double-VJP) identity so it reuses the reverse-mode engine. Requires the graph
 * autodiff engine (returns NULL under the eager engine).
 * @param f (CMLForwardFn) forward map y = f(x)
 * @param ctx (void*) context passed to f
 * @param x (Tensor*) evaluation point
 * @param v (Tensor*) tangent, same shape as x
 * @return (Tensor*) directional derivative with y's shape, or NULL
 */
Tensor* cml_jvp(CMLForwardFn f, void* ctx, Tensor* x, Tensor* v);

/**
 * Full Jacobian via forward mode: column j is J*e_j. Requires the graph engine.
 * @param f (CMLForwardFn) forward map y = f(x)
 * @param ctx (void*) context passed to f
 * @param x (Tensor*) evaluation point
 * @return (Tensor*) Jacobian [numel(y), numel(x)], or NULL
 */
Tensor* cml_jacfwd(CMLForwardFn f, void* ctx, Tensor* x);

#ifdef __cplusplus
}
#endif

#endif /* CML_AUTOGRAD_FORWARD_MODE_H */
