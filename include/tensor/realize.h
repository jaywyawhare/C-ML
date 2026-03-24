/*
 * realize — explicit tensor materialization API.
 *
 * Mirrors TinyGrad's Tensor.realize() / Tensor.is_realized().
 * C-ML tensors are lazy: data is NULL until realized.  These helpers
 * give callers explicit control over when execution occurs.
 */

#ifndef CML_TENSOR_REALIZE_H
#define CML_TENSOR_REALIZE_H

#include "tensor/tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Materialize a tensor: traverse its IR graph, schedule kernels, execute,
 * and store the result in t->data.  No-op if already realized.
 * Returns 0 on success.
 */
int tensor_realize(Tensor* t);

/*
 * Realize a batch of tensors together.  Allows the scheduler to share
 * common sub-graphs and avoid duplicate kernel launches.
 * Returns 0 if all succeed.
 */
int tensor_realize_all(Tensor** tensors, int num_tensors);

/*
 * True when t->data != NULL (tensor has been materialized).
 */
bool tensor_is_realized(const Tensor* t);

/*
 * Release t->data without freeing the tensor or its IR graph.
 * The tensor goes back to "lazy" state and will re-execute on next access.
 * Useful to free GPU memory when intermediate results are no longer needed.
 */
void tensor_unrealize(Tensor* t);

/*
 * Recursively realize t and all tensors in its gradient graph.
 * Returns 0 on success.
 */
int tensor_realize_with_grads(Tensor* t);

/*
 * Schedule all pending operations on the tensor's device without blocking.
 * The tensor is NOT immediately usable after this call; use tensor_sync()
 * to wait for completion.  Returns 0 on success.
 */
int tensor_schedule(Tensor* t);

/*
 * Block until all previously scheduled operations for t have completed.
 * Returns 0 on success.
 */
int tensor_sync(Tensor* t);

#ifdef __cplusplus
}
#endif

#endif /* CML_TENSOR_REALIZE_H */
