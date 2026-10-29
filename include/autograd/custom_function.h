#ifndef CML_AUTOGRAD_CUSTOM_FUNCTION_H
#define CML_AUTOGRAD_CUSTOM_FUNCTION_H

#include "tensor/tensor.h"
#include <stdbool.h>

/* C equivalent of torch.autograd.Function. Works on both the graph and eager
 * engines (the sweep only needs output->grad populated, then accumulates
 * eagerly). Double-backward through a custom function is NOT supported. */

/**
 * Opaque context carried from a custom forward into its backward.
 */
typedef struct CMLAutogradContext CMLAutogradContext;

/**
 * User forward: compute an output tensor from the inputs.
 * @param ctx (CMLAutogradContext*)
 * @param inputs (Tensor**)
 * @param n_inputs (int)
 * @param user_data (void*)
 * @return (Tensor*)
 */
typedef Tensor* (*CMLCustomForward)(CMLAutogradContext* ctx, Tensor** inputs, int n_inputs,
                                    void* user_data);

/**
 * User backward: write the gradient w.r.t. each input into grad_inputs (a NULL
 * entry is treated as a zero gradient for that input).
 * @param ctx (CMLAutogradContext*)
 * @param grad_output (Tensor*)
 * @param grad_inputs (Tensor**)
 * @param n_inputs (int)
 * @param user_data (void*)
 * @return (void)
 */
typedef void (*CMLCustomBackward)(CMLAutogradContext* ctx, Tensor* grad_output,
                                  Tensor** grad_inputs, int n_inputs, void* user_data);

/**
 * Run fwd on the inputs, record a backward node tied to the output, return it.
 * @param fwd (CMLCustomForward)
 * @param bwd (CMLCustomBackward)
 * @param inputs (Tensor**)
 * @param n_inputs (int)
 * @param user_data (void*)
 * @return (Tensor*)
 */
Tensor* cml_autograd_function_apply(CMLCustomForward fwd, CMLCustomBackward bwd, Tensor** inputs,
                                    int n_inputs, void* user_data);

/**
 * Stash tensors on the context for the backward to read.
 * @param ctx (CMLAutogradContext*)
 * @param tensors (Tensor**)
 * @param n (int)
 * @return (void)
 */
void cml_autograd_ctx_save(CMLAutogradContext* ctx, Tensor** tensors, int n);

/**
 * Return the saved tensors and their count (via out_n).
 * @param ctx (CMLAutogradContext*)
 * @param out_n (int*)
 * @return (Tensor**)
 */
Tensor** cml_autograd_ctx_saved(CMLAutogradContext* ctx, int* out_n);

/**
 * Attach an opaque scratch pointer to the context.
 * @param ctx (CMLAutogradContext*)
 * @param p (void*)
 * @return (void)
 */
void cml_autograd_ctx_set_data(CMLAutogradContext* ctx, void* p);

/**
 * Retrieve the scratch pointer attached to the context.
 * @param ctx (CMLAutogradContext*)
 * @return (void*)
 */
void* cml_autograd_ctx_get_data(CMLAutogradContext* ctx);

/**
 * Run any pending custom backward functions; invoked by tensor_backward after
 * the reverse pass. A no-op when no custom nodes are registered.
 * @param create_graph (bool)
 * @return (void)
 */
void cml_autograd_custom_backward_sweep(bool create_graph);

/**
 * Drop every registered custom node (frees node bookkeeping and contexts, not
 * the user's tensors).
 * @return (void)
 */
void cml_autograd_custom_reset(void);

#endif /* CML_AUTOGRAD_CUSTOM_FUNCTION_H */
