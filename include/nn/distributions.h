#ifndef CML_NN_DISTRIBUTIONS_H
#define CML_NN_DISTRIBUTIONS_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Reparameterized Normal sample: loc + scale * eps, eps ~ N(0,1). Differentiable
 * in loc and scale (the noise is detached).
 * @param loc (Tensor*) mean
 * @param scale (Tensor*) standard deviation, same shape as loc
 * @return (Tensor*) sample with loc's shape, or NULL
 */
Tensor* cml_normal_rsample(Tensor* loc, Tensor* scale);

/**
 * Elementwise Normal log-density of @p value.
 * @param loc (Tensor*) mean
 * @param scale (Tensor*) standard deviation
 * @param value (Tensor*) evaluation point
 * @return (Tensor*) log N(value; loc, scale), or NULL
 */
Tensor* cml_normal_log_prob(Tensor* loc, Tensor* scale, Tensor* value);

/**
 * Elementwise Normal differential entropy 0.5*log(2*pi*e*scale^2).
 * @param scale (Tensor*) standard deviation
 * @return (Tensor*) entropy, or NULL
 */
Tensor* cml_normal_entropy(Tensor* scale);

/**
 * Bernoulli sample (detached): 1 where u < probs, else 0.
 * @param probs (Tensor*) success probabilities in [0,1]
 * @return (Tensor*) 0/1 sample with probs's shape, or NULL
 */
Tensor* cml_bernoulli_sample(Tensor* probs);

/**
 * Elementwise Bernoulli log-probability of @p value.
 * @param probs (Tensor*) success probabilities
 * @param value (Tensor*) 0/1 outcomes
 * @return (Tensor*) log-probability, or NULL
 */
Tensor* cml_bernoulli_log_prob(Tensor* probs, Tensor* value);

/**
 * Elementwise Bernoulli entropy -(p*log p + (1-p)*log(1-p)).
 * @param probs (Tensor*) success probabilities
 * @return (Tensor*) entropy, or NULL
 */
Tensor* cml_bernoulli_entropy(Tensor* probs);

/**
 * Categorical sample (detached) over the last dimension via multinomial.
 * @param logits (Tensor*) unnormalized log-probabilities [N, K]
 * @param num_samples (int) draws per row
 * @return (Tensor*) sampled class indices [N, num_samples], or NULL
 */
Tensor* cml_categorical_sample(Tensor* logits, int num_samples);

/**
 * Categorical log-probability of @p indices over the last dimension.
 * @param logits (Tensor*) unnormalized log-probabilities [N, K]
 * @param indices (Tensor*) class indices [N]
 * @return (Tensor*) log-probability [N], or NULL
 */
Tensor* cml_categorical_log_prob(Tensor* logits, Tensor* indices);

/**
 * Categorical entropy -sum(p*log p) over the last dimension.
 * @param logits (Tensor*) unnormalized log-probabilities [N, K]
 * @return (Tensor*) entropy [N], or NULL
 */
Tensor* cml_categorical_entropy(Tensor* logits);

/**
 * Gumbel-Softmax relaxed categorical sample over the last dimension.
 * Differentiable in @p logits; with @p hard the forward value is one-hot via a
 * straight-through estimator while gradients use the soft sample.
 * @param logits (Tensor*) unnormalized log-probabilities [N, K]
 * @param tau (float) temperature (> 0)
 * @param hard (bool) straight-through one-hot forward if true
 * @return (Tensor*) relaxed sample [N, K], or NULL
 */
Tensor* cml_gumbel_softmax(Tensor* logits, float tau, bool hard);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_DISTRIBUTIONS_H */
