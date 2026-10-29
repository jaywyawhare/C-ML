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

/**
 * Reparameterized Uniform(low, high) sample low + (high-low)*u, u ~ U(0,1).
 * @param low (Tensor*)
 * @param high (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_uniform_rsample(Tensor* low, Tensor* high);

/**
 * Elementwise Uniform log-density of @p value.
 * @param low (Tensor*)
 * @param high (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_uniform_log_prob(Tensor* low, Tensor* high, Tensor* value);

/**
 * Uniform mean (low+high)/2.
 * @param low (Tensor*)
 * @param high (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_uniform_mean(Tensor* low, Tensor* high);

/**
 * Uniform variance (high-low)^2/12.
 * @param low (Tensor*)
 * @param high (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_uniform_variance(Tensor* low, Tensor* high);

/**
 * Uniform entropy log(high-low).
 * @param low (Tensor*)
 * @param high (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_uniform_entropy(Tensor* low, Tensor* high);

/**
 * Reparameterized Exponential(rate) sample via inverse-CDF -log(u)/rate.
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_exponential_rsample(Tensor* rate);

/**
 * Elementwise Exponential log-density of @p value.
 * @param rate (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_exponential_log_prob(Tensor* rate, Tensor* value);

/**
 * Exponential mean 1/rate.
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_exponential_mean(Tensor* rate);

/**
 * Exponential variance 1/rate^2.
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_exponential_variance(Tensor* rate);

/**
 * Exponential entropy 1-log(rate).
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_exponential_entropy(Tensor* rate);

/**
 * Reparameterized Laplace(loc, scale) sample via inverse-CDF with u ~ U(-0.5, 0.5).
 * @param loc (Tensor*)
 * @param scale (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_laplace_rsample(Tensor* loc, Tensor* scale);

/**
 * Elementwise Laplace log-density of @p value.
 * @param loc (Tensor*)
 * @param scale (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_laplace_log_prob(Tensor* loc, Tensor* scale, Tensor* value);

/**
 * Laplace mean (= loc).
 * @param loc (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_laplace_mean(Tensor* loc);

/**
 * Laplace variance 2*scale^2.
 * @param scale (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_laplace_variance(Tensor* scale);

/**
 * Laplace entropy 1+log(2*scale).
 * @param scale (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_laplace_entropy(Tensor* scale);

/**
 * Reparameterized Cauchy(loc, scale) sample loc + scale*tan(pi*(u-0.5)).
 * @param loc (Tensor*)
 * @param scale (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_cauchy_rsample(Tensor* loc, Tensor* scale);

/**
 * Elementwise Cauchy log-density of @p value.
 * @param loc (Tensor*)
 * @param scale (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_cauchy_log_prob(Tensor* loc, Tensor* scale, Tensor* value);

/**
 * Cauchy entropy log(4*pi*scale); mean and variance are undefined.
 * @param scale (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_cauchy_entropy(Tensor* scale);

/**
 * Gamma(concentration, rate) variate via Marsaglia-Tsang.
 * @param concentration (Tensor*)
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_gamma_sample(Tensor* concentration, Tensor* rate);

/**
 * Elementwise Gamma log-density of @p value.
 * @param concentration (Tensor*)
 * @param rate (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_gamma_log_prob(Tensor* concentration, Tensor* rate, Tensor* value);

/**
 * Gamma mean concentration/rate.
 * @param concentration (Tensor*)
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_gamma_mean(Tensor* concentration, Tensor* rate);

/**
 * Gamma variance concentration/rate^2.
 * @param concentration (Tensor*)
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_gamma_variance(Tensor* concentration, Tensor* rate);

/**
 * Gamma entropy.
 * @param concentration (Tensor*)
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_gamma_entropy(Tensor* concentration, Tensor* rate);

/**
 * Beta(c1, c0) variate X/(X+Y) with X~Gamma(c1,1), Y~Gamma(c0,1).
 * @param c1 (Tensor*)
 * @param c0 (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_beta_sample(Tensor* c1, Tensor* c0);

/**
 * Elementwise Beta log-density of @p value.
 * @param c1 (Tensor*)
 * @param c0 (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_beta_log_prob(Tensor* c1, Tensor* c0, Tensor* value);

/**
 * Beta mean c1/(c1+c0).
 * @param c1 (Tensor*)
 * @param c0 (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_beta_mean(Tensor* c1, Tensor* c0);

/**
 * Beta variance.
 * @param c1 (Tensor*)
 * @param c0 (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_beta_variance(Tensor* c1, Tensor* c0);

/**
 * Beta entropy.
 * @param c1 (Tensor*)
 * @param c0 (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_beta_entropy(Tensor* c1, Tensor* c0);

/**
 * Chi2(df) variate (== Gamma(df/2, 1/2)).
 * @param df (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_chi2_sample(Tensor* df);

/**
 * Elementwise Chi2 log-density of @p value.
 * @param df (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_chi2_log_prob(Tensor* df, Tensor* value);

/**
 * Chi2 mean (= df).
 * @param df (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_chi2_mean(Tensor* df);

/**
 * Chi2 variance 2*df.
 * @param df (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_chi2_variance(Tensor* df);

/**
 * Chi2 entropy.
 * @param df (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_chi2_entropy(Tensor* df);

/**
 * StudentT(df, loc, scale) variate loc + scale*Z/sqrt(Chi2(df)/df).
 * @param df (Tensor*)
 * @param loc (Tensor*)
 * @param scale (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_studentt_sample(Tensor* df, Tensor* loc, Tensor* scale);

/**
 * Elementwise StudentT log-density of @p value.
 * @param df (Tensor*)
 * @param loc (Tensor*)
 * @param scale (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_studentt_log_prob(Tensor* df, Tensor* loc, Tensor* scale, Tensor* value);

/**
 * StudentT mean (= loc, for df > 1).
 * @param df (Tensor*)
 * @param loc (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_studentt_mean(Tensor* df, Tensor* loc);

/**
 * StudentT variance scale^2*df/(df-2), for df > 2.
 * @param df (Tensor*)
 * @param scale (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_studentt_variance(Tensor* df, Tensor* scale);

/**
 * StudentT entropy.
 * @param df (Tensor*)
 * @param scale (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_studentt_entropy(Tensor* df, Tensor* scale);

/**
 * Poisson(rate) variate: Knuth below rate 30, normal approximation above.
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_poisson_sample(Tensor* rate);

/**
 * Elementwise Poisson log-probability of @p value.
 * @param rate (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_poisson_log_prob(Tensor* rate, Tensor* value);

/**
 * Poisson mean (= rate).
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_poisson_mean(Tensor* rate);

/**
 * Poisson variance (= rate).
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_poisson_variance(Tensor* rate);

/**
 * Geometric(probs) variate: number of failures before the first success.
 * @param probs (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_geometric_sample(Tensor* probs);

/**
 * Elementwise Geometric log-probability of @p value.
 * @param probs (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_geometric_log_prob(Tensor* probs, Tensor* value);

/**
 * Geometric mean (1-probs)/probs.
 * @param probs (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_geometric_mean(Tensor* probs);

/**
 * Geometric variance (1-probs)/probs^2.
 * @param probs (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_geometric_variance(Tensor* probs);

/**
 * Geometric entropy.
 * @param probs (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_geometric_entropy(Tensor* probs);

/**
 * Binomial(total_count, probs) variate: sum of total_count Bernoulli draws.
 * @param total_count (int)
 * @param probs (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_binomial_sample(int total_count, Tensor* probs);

/**
 * Elementwise Binomial log-probability of @p value.
 * @param total_count (int)
 * @param probs (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_binomial_log_prob(int total_count, Tensor* probs, Tensor* value);

/**
 * Binomial mean total_count*probs.
 * @param total_count (int)
 * @param probs (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_binomial_mean(int total_count, Tensor* probs);

/**
 * Binomial variance total_count*probs*(1-probs).
 * @param total_count (int)
 * @param probs (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_binomial_variance(int total_count, Tensor* probs);

/**
 * Dirichlet(concentration) variate over the last axis of concentration [..., K].
 * @param concentration (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_dirichlet_sample(Tensor* concentration);

/**
 * Dirichlet log-density of @p value over the last axis.
 * @param concentration (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_dirichlet_log_prob(Tensor* concentration, Tensor* value);

/**
 * Dirichlet mean concentration/sum(concentration).
 * @param concentration (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_dirichlet_mean(Tensor* concentration);

/**
 * Dirichlet variance over the last axis.
 * @param concentration (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_dirichlet_variance(Tensor* concentration);

/**
 * Dirichlet entropy.
 * @param concentration (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_dirichlet_entropy(Tensor* concentration);

/**
 * Reparameterized MultivariateNormal sample loc + L*eps via Cholesky factor L.
 * @param loc (Tensor*) mean [D]
 * @param covariance (Tensor*) [D, D]
 * @return (Tensor*)
 */
Tensor* cml_mvn_rsample(Tensor* loc, Tensor* covariance);

/**
 * MultivariateNormal log-density of @p value ([D] point or [N, D] batch).
 * @param loc (Tensor*) mean [D]
 * @param covariance (Tensor*) [D, D]
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_mvn_log_prob(Tensor* loc, Tensor* covariance, Tensor* value);

/**
 * MultivariateNormal entropy from the covariance log-determinant.
 * @param covariance (Tensor*) [D, D]
 * @return (Tensor*)
 */
Tensor* cml_mvn_entropy(Tensor* covariance);

/** Distribution family tag for the KL dispatcher. */
typedef enum {
    CML_DIST_NORMAL,
    CML_DIST_BERNOULLI,
    CML_DIST_CATEGORICAL,
    CML_DIST_GAMMA,
    CML_DIST_BETA,
    CML_DIST_EXPONENTIAL,
    CML_DIST_UNIFORM,
    CML_DIST_DIRICHLET
} CMLDistType;

/** A family tag plus up to two parameter tensors. */
typedef struct {
    CMLDistType type;
    Tensor* p1;
    Tensor* p2;
} CMLDistribution;

/**
 * Wrap a Normal(loc, scale) as a CMLDistribution.
 * @param loc (Tensor*)
 * @param scale (Tensor*)
 * @return (CMLDistribution)
 */
CMLDistribution cml_dist_normal(Tensor* loc, Tensor* scale);

/**
 * Wrap a Bernoulli(probs) as a CMLDistribution.
 * @param probs (Tensor*)
 * @return (CMLDistribution)
 */
CMLDistribution cml_dist_bernoulli(Tensor* probs);

/**
 * Wrap a Categorical(logits) as a CMLDistribution.
 * @param logits (Tensor*)
 * @return (CMLDistribution)
 */
CMLDistribution cml_dist_categorical(Tensor* logits);

/**
 * Wrap a Gamma(concentration, rate) as a CMLDistribution.
 * @param concentration (Tensor*)
 * @param rate (Tensor*)
 * @return (CMLDistribution)
 */
CMLDistribution cml_dist_gamma(Tensor* concentration, Tensor* rate);

/**
 * Wrap a Beta(c1, c0) as a CMLDistribution.
 * @param c1 (Tensor*)
 * @param c0 (Tensor*)
 * @return (CMLDistribution)
 */
CMLDistribution cml_dist_beta(Tensor* c1, Tensor* c0);

/**
 * Wrap an Exponential(rate) as a CMLDistribution.
 * @param rate (Tensor*)
 * @return (CMLDistribution)
 */
CMLDistribution cml_dist_exponential(Tensor* rate);

/**
 * Wrap a Uniform(low, high) as a CMLDistribution.
 * @param low (Tensor*)
 * @param high (Tensor*)
 * @return (CMLDistribution)
 */
CMLDistribution cml_dist_uniform(Tensor* low, Tensor* high);

/**
 * Wrap a Dirichlet(concentration) as a CMLDistribution.
 * @param concentration (Tensor*)
 * @return (CMLDistribution)
 */
CMLDistribution cml_dist_dirichlet(Tensor* concentration);

/**
 * Analytic KL(p || q) for matching-type pairs, else NULL.
 * @param p (CMLDistribution)
 * @param q (CMLDistribution)
 * @return (Tensor*)
 */
Tensor* cml_kl_divergence(CMLDistribution p, CMLDistribution q);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_DISTRIBUTIONS_H */
