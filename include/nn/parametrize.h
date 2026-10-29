#ifndef CML_NN_PARAMETRIZE_H
#define CML_NN_PARAMETRIZE_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Weight normalization: reparameterize a weight as w = g * v / ||v||, with the
 * Euclidean norm taken over every dimension except @p dim. Built from autograd
 * ops, so gradients flow to both @p v and @p g.
 * @param v (Tensor*) direction parameter, any shape
 * @param g (Tensor*) magnitude parameter, numel == v->shape[dim]
 * @param dim (int) dimension kept un-reduced (the per-slice norm axis)
 * @return (Tensor*) normalized weight with the same shape as v, or NULL on error
 */
Tensor* cml_nn_weight_norm(Tensor* v, Tensor* g, int dim);

/**
 * Spectral normalization of a 2-D weight: w = W / sigma, where sigma is the
 * largest singular value estimated by @p n_power_iterations of power iteration.
 * The left singular vector @p u is refined in place (detached from autograd);
 * sigma stays differentiable in W, so gradients flow to W.
 * @param W (Tensor*) weight matrix [M, N]
 * @param u (Tensor*) persistent left singular vector estimate [M], updated in place
 * @param n_power_iterations (int) power-iteration steps per call
 * @return (Tensor*) spectrally normalized weight [M, N], or NULL on error
 */
Tensor* cml_nn_spectral_norm(Tensor* W, Tensor* u, int n_power_iterations);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_PARAMETRIZE_H */
