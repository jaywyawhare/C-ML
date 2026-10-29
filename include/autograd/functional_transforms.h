#ifndef CML_AUTOGRAD_FUNCTIONAL_TRANSFORMS_H
#define CML_AUTOGRAD_FUNCTIONAL_TRANSFORMS_H

#include "nn.h"
#include "nn/state.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* grad/vjp/jacrev are reverse mode (both engines); hessian (jacrev-of-grad) and
 * jacfwd/jvp (forward_mode.h) need differentiable grads, so graph engine only. */

/**
 * User function under transform: maps n input tensors to a single output.
 * @param inputs (Tensor**)
 * @param n (int)
 * @param ud (void*)
 * @return (Tensor*)
 */
typedef Tensor* (*CMLTransformFn)(Tensor** inputs, int n, void* ud);

/**
 * Gradient of a scalar-output f w.r.t. inputs[argnum] (reverse mode).
 * @param f (CMLTransformFn)
 * @param ud (void*)
 * @param inputs (Tensor**)
 * @param n (int)
 * @param argnum (int)
 * @return (Tensor*)
 */
Tensor* cml_grad(CMLTransformFn f, void* ud, Tensor** inputs, int n, int argnum);

/**
 * Gradient of a scalar-output f w.r.t. several inputs at once.
 * @param f (CMLTransformFn)
 * @param ud (void*)
 * @param inputs (Tensor**)
 * @param n (int)
 * @param argnums (const int*)
 * @param num_argnums (int)
 * @param grads_out (Tensor**)
 * @return (int)
 */
int cml_grad_multi(CMLTransformFn f, void* ud, Tensor** inputs, int n, const int* argnums,
                   int num_argnums, Tensor** grads_out);

typedef struct CMLVjp CMLVjp;

/**
 * Evaluate y = f(inputs) and build a VJP closure that borrows the inputs until freed.
 * @param f (CMLTransformFn)
 * @param ud (void*)
 * @param inputs (Tensor**)
 * @param n (int)
 * @param output (Tensor**)
 * @return (CMLVjp*)
 */
CMLVjp* cml_vjp(CMLTransformFn f, void* ud, Tensor** inputs, int n, Tensor** output);

/**
 * Apply a VJP: propagate cotangent back to the n input gradients (v^T J).
 * @param vjp (CMLVjp*)
 * @param cotangent (Tensor*)
 * @param grads_out (Tensor**)
 * @return (int)
 */
int cml_vjp_fn(CMLVjp* vjp, Tensor* cotangent, Tensor** grads_out);

/**
 * Free a VJP closure (leaves the borrowed inputs alone).
 * @param vjp (CMLVjp*)
 */
void cml_vjp_free(CMLVjp* vjp);

/**
 * Reverse-mode Jacobian of f w.r.t. inputs[argnum], one VJP per output basis.
 * @param f (CMLTransformFn)
 * @param ud (void*)
 * @param inputs (Tensor**)
 * @param n (int)
 * @param argnum (int)
 * @return (Tensor*)
 */
Tensor* cml_jacrev(CMLTransformFn f, void* ud, Tensor** inputs, int n, int argnum);

/**
 * Hessian of a scalar-output f w.r.t. inputs[argnum] as jacrev-of-grad (graph engine only).
 * @param f (CMLTransformFn)
 * @param ud (void*)
 * @param inputs (Tensor**)
 * @param n (int)
 * @param argnum (int)
 * @return (Tensor*)
 */
Tensor* cml_hessian(CMLTransformFn f, void* ud, Tensor** inputs, int n, int argnum);

/**
 * Reference (non-fused) vmap: slice each mapped input, apply f, stack results.
 * @param f (CMLTransformFn)
 * @param ud (void*)
 * @param inputs (Tensor**)
 * @param n (int)
 * @param in_axes (const int*)
 * @param out_axis (int)
 * @return (Tensor*)
 */
Tensor* cml_vmap(CMLTransformFn f, void* ud, Tensor** inputs, int n, const int* in_axes,
                 int out_axis);

/**
 * Run module->forward(input) with params temporarily swapped in, then restore.
 * @param module (Module*)
 * @param params (const StateDict*)
 * @param input (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_functional_call(Module* module, const StateDict* params, Tensor* input);

#ifdef __cplusplus
}
#endif

#endif /* CML_AUTOGRAD_FUNCTIONAL_TRANSFORMS_H */
