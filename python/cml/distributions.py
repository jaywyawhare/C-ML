"""Probability distributions, mirroring ``torch.distributions``.

Each distribution wraps the C-ML distribution kernels: a sampler (``rsample``
for reparameterizable families, else ``sample``), an elementwise ``log_prob``,
and ``mean`` / ``variance`` / ``entropy`` where closed-form. ``kl_divergence``
dispatches to the analytic KL for supported same-type pairs.
"""

from cml._cml_lib import ffi, lib
from cml.core import Tensor, tensor as _tensor


def _as_tensor(x):
    """Accept a Tensor as-is, else build one via the tensor factory."""
    return x if isinstance(x, Tensor) else _tensor(x)


def _wrap(result, name):
    """Wrap a C result tensor, raising on a NULL (invalid shapes/args)."""
    if result == ffi.NULL:
        raise RuntimeError(f"{name}: the C library returned NULL - check argument shapes/dtypes.")
    return Tensor(result)


class Distribution:
    """Base class; subclasses set ``_kl`` (a CMLDistribution builder) when the
    family participates in the analytic KL dispatcher."""

    _kl = None

    def sample(self):
        raise NotImplementedError

    def rsample(self):
        return self.sample()

    def log_prob(self, value):
        raise NotImplementedError

    def _kl_handle(self):
        return self._kl() if self._kl is not None else None


class Normal(Distribution):
    def __init__(self, loc, scale):
        self.loc, self.scale = _as_tensor(loc), _as_tensor(scale)
        self._kl = lambda: lib.cml_dist_normal(self.loc._tensor, self.scale._tensor)

    def rsample(self):
        return _wrap(lib.cml_normal_rsample(self.loc._tensor, self.scale._tensor), "Normal.rsample")

    sample = rsample

    def log_prob(self, value):
        v = _as_tensor(value)
        return _wrap(lib.cml_normal_log_prob(self.loc._tensor, self.scale._tensor, v._tensor),
                     "Normal.log_prob")

    @property
    def mean(self):
        return self.loc

    def entropy(self):
        return _wrap(lib.cml_normal_entropy(self.scale._tensor), "Normal.entropy")


class Bernoulli(Distribution):
    def __init__(self, probs):
        self.probs = _as_tensor(probs)
        self._kl = lambda: lib.cml_dist_bernoulli(self.probs._tensor)

    def sample(self):
        return _wrap(lib.cml_bernoulli_sample(self.probs._tensor), "Bernoulli.sample")

    def log_prob(self, value):
        v = _as_tensor(value)
        return _wrap(lib.cml_bernoulli_log_prob(self.probs._tensor, v._tensor), "Bernoulli.log_prob")

    def entropy(self):
        return _wrap(lib.cml_bernoulli_entropy(self.probs._tensor), "Bernoulli.entropy")


class Categorical(Distribution):
    def __init__(self, logits):
        self.logits = _as_tensor(logits)
        self._kl = lambda: lib.cml_dist_categorical(self.logits._tensor)

    def sample(self, num_samples=1):
        return _wrap(lib.cml_categorical_sample(self.logits._tensor, int(num_samples)),
                     "Categorical.sample")

    def log_prob(self, indices):
        idx = _as_tensor(indices)
        return _wrap(lib.cml_categorical_log_prob(self.logits._tensor, idx._tensor),
                     "Categorical.log_prob")

    def entropy(self):
        return _wrap(lib.cml_categorical_entropy(self.logits._tensor), "Categorical.entropy")


class Uniform(Distribution):
    def __init__(self, low, high):
        self.low, self.high = _as_tensor(low), _as_tensor(high)
        self._kl = lambda: lib.cml_dist_uniform(self.low._tensor, self.high._tensor)

    def rsample(self):
        return _wrap(lib.cml_uniform_rsample(self.low._tensor, self.high._tensor), "Uniform.rsample")

    sample = rsample

    def log_prob(self, value):
        v = _as_tensor(value)
        return _wrap(lib.cml_uniform_log_prob(self.low._tensor, self.high._tensor, v._tensor),
                     "Uniform.log_prob")

    @property
    def mean(self):
        return _wrap(lib.cml_uniform_mean(self.low._tensor, self.high._tensor), "Uniform.mean")

    @property
    def variance(self):
        return _wrap(lib.cml_uniform_variance(self.low._tensor, self.high._tensor),
                     "Uniform.variance")

    def entropy(self):
        return _wrap(lib.cml_uniform_entropy(self.low._tensor, self.high._tensor), "Uniform.entropy")


class Exponential(Distribution):
    def __init__(self, rate):
        self.rate = _as_tensor(rate)
        self._kl = lambda: lib.cml_dist_exponential(self.rate._tensor)

    def rsample(self):
        return _wrap(lib.cml_exponential_rsample(self.rate._tensor), "Exponential.rsample")

    sample = rsample

    def log_prob(self, value):
        v = _as_tensor(value)
        return _wrap(lib.cml_exponential_log_prob(self.rate._tensor, v._tensor),
                     "Exponential.log_prob")

    @property
    def mean(self):
        return _wrap(lib.cml_exponential_mean(self.rate._tensor), "Exponential.mean")

    @property
    def variance(self):
        return _wrap(lib.cml_exponential_variance(self.rate._tensor), "Exponential.variance")

    def entropy(self):
        return _wrap(lib.cml_exponential_entropy(self.rate._tensor), "Exponential.entropy")


class Laplace(Distribution):
    def __init__(self, loc, scale):
        self.loc, self.scale = _as_tensor(loc), _as_tensor(scale)

    def rsample(self):
        return _wrap(lib.cml_laplace_rsample(self.loc._tensor, self.scale._tensor),
                     "Laplace.rsample")

    sample = rsample

    def log_prob(self, value):
        v = _as_tensor(value)
        return _wrap(lib.cml_laplace_log_prob(self.loc._tensor, self.scale._tensor, v._tensor),
                     "Laplace.log_prob")

    @property
    def mean(self):
        return _wrap(lib.cml_laplace_mean(self.loc._tensor), "Laplace.mean")

    @property
    def variance(self):
        return _wrap(lib.cml_laplace_variance(self.scale._tensor), "Laplace.variance")

    def entropy(self):
        return _wrap(lib.cml_laplace_entropy(self.scale._tensor), "Laplace.entropy")


class Cauchy(Distribution):
    def __init__(self, loc, scale):
        self.loc, self.scale = _as_tensor(loc), _as_tensor(scale)

    def rsample(self):
        return _wrap(lib.cml_cauchy_rsample(self.loc._tensor, self.scale._tensor), "Cauchy.rsample")

    sample = rsample

    def log_prob(self, value):
        v = _as_tensor(value)
        return _wrap(lib.cml_cauchy_log_prob(self.loc._tensor, self.scale._tensor, v._tensor),
                     "Cauchy.log_prob")

    def entropy(self):
        return _wrap(lib.cml_cauchy_entropy(self.scale._tensor), "Cauchy.entropy")


class Gamma(Distribution):
    def __init__(self, concentration, rate):
        self.concentration, self.rate = _as_tensor(concentration), _as_tensor(rate)
        self._kl = lambda: lib.cml_dist_gamma(self.concentration._tensor, self.rate._tensor)

    def sample(self):
        return _wrap(lib.cml_gamma_sample(self.concentration._tensor, self.rate._tensor),
                     "Gamma.sample")

    def log_prob(self, value):
        v = _as_tensor(value)
        return _wrap(
            lib.cml_gamma_log_prob(self.concentration._tensor, self.rate._tensor, v._tensor),
            "Gamma.log_prob")

    @property
    def mean(self):
        return _wrap(lib.cml_gamma_mean(self.concentration._tensor, self.rate._tensor), "Gamma.mean")

    @property
    def variance(self):
        return _wrap(lib.cml_gamma_variance(self.concentration._tensor, self.rate._tensor),
                     "Gamma.variance")

    def entropy(self):
        return _wrap(lib.cml_gamma_entropy(self.concentration._tensor, self.rate._tensor),
                     "Gamma.entropy")


class Beta(Distribution):
    def __init__(self, concentration1, concentration0):
        self.c1, self.c0 = _as_tensor(concentration1), _as_tensor(concentration0)
        self._kl = lambda: lib.cml_dist_beta(self.c1._tensor, self.c0._tensor)

    def sample(self):
        return _wrap(lib.cml_beta_sample(self.c1._tensor, self.c0._tensor), "Beta.sample")

    def log_prob(self, value):
        v = _as_tensor(value)
        return _wrap(lib.cml_beta_log_prob(self.c1._tensor, self.c0._tensor, v._tensor),
                     "Beta.log_prob")

    @property
    def mean(self):
        return _wrap(lib.cml_beta_mean(self.c1._tensor, self.c0._tensor), "Beta.mean")

    @property
    def variance(self):
        return _wrap(lib.cml_beta_variance(self.c1._tensor, self.c0._tensor), "Beta.variance")

    def entropy(self):
        return _wrap(lib.cml_beta_entropy(self.c1._tensor, self.c0._tensor), "Beta.entropy")


class Chi2(Distribution):
    def __init__(self, df):
        self.df = _as_tensor(df)

    def sample(self):
        return _wrap(lib.cml_chi2_sample(self.df._tensor), "Chi2.sample")

    def log_prob(self, value):
        v = _as_tensor(value)
        return _wrap(lib.cml_chi2_log_prob(self.df._tensor, v._tensor), "Chi2.log_prob")

    @property
    def mean(self):
        return _wrap(lib.cml_chi2_mean(self.df._tensor), "Chi2.mean")

    @property
    def variance(self):
        return _wrap(lib.cml_chi2_variance(self.df._tensor), "Chi2.variance")

    def entropy(self):
        return _wrap(lib.cml_chi2_entropy(self.df._tensor), "Chi2.entropy")


class StudentT(Distribution):
    def __init__(self, df, loc=0.0, scale=1.0):
        self.df, self.loc, self.scale = _as_tensor(df), _as_tensor(loc), _as_tensor(scale)

    def sample(self):
        return _wrap(lib.cml_studentt_sample(self.df._tensor, self.loc._tensor, self.scale._tensor),
                     "StudentT.sample")

    def log_prob(self, value):
        v = _as_tensor(value)
        return _wrap(
            lib.cml_studentt_log_prob(self.df._tensor, self.loc._tensor, self.scale._tensor,
                                      v._tensor),
            "StudentT.log_prob")

    @property
    def mean(self):
        return _wrap(lib.cml_studentt_mean(self.df._tensor, self.loc._tensor), "StudentT.mean")

    @property
    def variance(self):
        return _wrap(lib.cml_studentt_variance(self.df._tensor, self.scale._tensor),
                     "StudentT.variance")

    def entropy(self):
        return _wrap(lib.cml_studentt_entropy(self.df._tensor, self.scale._tensor),
                     "StudentT.entropy")


class Poisson(Distribution):
    def __init__(self, rate):
        self.rate = _as_tensor(rate)

    def sample(self):
        return _wrap(lib.cml_poisson_sample(self.rate._tensor), "Poisson.sample")

    def log_prob(self, value):
        v = _as_tensor(value)
        return _wrap(lib.cml_poisson_log_prob(self.rate._tensor, v._tensor), "Poisson.log_prob")

    @property
    def mean(self):
        return _wrap(lib.cml_poisson_mean(self.rate._tensor), "Poisson.mean")

    @property
    def variance(self):
        return _wrap(lib.cml_poisson_variance(self.rate._tensor), "Poisson.variance")


class Geometric(Distribution):
    def __init__(self, probs):
        self.probs = _as_tensor(probs)

    def sample(self):
        return _wrap(lib.cml_geometric_sample(self.probs._tensor), "Geometric.sample")

    def log_prob(self, value):
        v = _as_tensor(value)
        return _wrap(lib.cml_geometric_log_prob(self.probs._tensor, v._tensor),
                     "Geometric.log_prob")

    @property
    def mean(self):
        return _wrap(lib.cml_geometric_mean(self.probs._tensor), "Geometric.mean")

    @property
    def variance(self):
        return _wrap(lib.cml_geometric_variance(self.probs._tensor), "Geometric.variance")

    def entropy(self):
        return _wrap(lib.cml_geometric_entropy(self.probs._tensor), "Geometric.entropy")


class Binomial(Distribution):
    def __init__(self, total_count, probs):
        self.total_count = int(total_count)
        self.probs = _as_tensor(probs)

    def sample(self):
        return _wrap(lib.cml_binomial_sample(self.total_count, self.probs._tensor),
                     "Binomial.sample")

    def log_prob(self, value):
        v = _as_tensor(value)
        return _wrap(lib.cml_binomial_log_prob(self.total_count, self.probs._tensor, v._tensor),
                     "Binomial.log_prob")

    @property
    def mean(self):
        return _wrap(lib.cml_binomial_mean(self.total_count, self.probs._tensor), "Binomial.mean")

    @property
    def variance(self):
        return _wrap(lib.cml_binomial_variance(self.total_count, self.probs._tensor),
                     "Binomial.variance")


class Dirichlet(Distribution):
    def __init__(self, concentration):
        self.concentration = _as_tensor(concentration)
        self._kl = lambda: lib.cml_dist_dirichlet(self.concentration._tensor)

    def sample(self):
        return _wrap(lib.cml_dirichlet_sample(self.concentration._tensor), "Dirichlet.sample")

    def log_prob(self, value):
        v = _as_tensor(value)
        return _wrap(lib.cml_dirichlet_log_prob(self.concentration._tensor, v._tensor),
                     "Dirichlet.log_prob")

    @property
    def mean(self):
        return _wrap(lib.cml_dirichlet_mean(self.concentration._tensor), "Dirichlet.mean")

    @property
    def variance(self):
        return _wrap(lib.cml_dirichlet_variance(self.concentration._tensor), "Dirichlet.variance")

    def entropy(self):
        return _wrap(lib.cml_dirichlet_entropy(self.concentration._tensor), "Dirichlet.entropy")


class MultivariateNormal(Distribution):
    def __init__(self, loc, covariance_matrix):
        self.loc = _as_tensor(loc)
        self.covariance_matrix = _as_tensor(covariance_matrix)

    def rsample(self):
        return _wrap(lib.cml_mvn_rsample(self.loc._tensor, self.covariance_matrix._tensor),
                     "MultivariateNormal.rsample")

    sample = rsample

    def log_prob(self, value):
        v = _as_tensor(value)
        return _wrap(lib.cml_mvn_log_prob(self.loc._tensor, self.covariance_matrix._tensor,
                                          v._tensor),
                     "MultivariateNormal.log_prob")

    @property
    def mean(self):
        return self.loc

    def entropy(self):
        return _wrap(lib.cml_mvn_entropy(self.covariance_matrix._tensor),
                     "MultivariateNormal.entropy")


def kl_divergence(p, q):
    """Analytic KL(p || q) for matching-type pairs (Normal, Bernoulli,
    Categorical, Gamma, Beta, Exponential, Uniform, Dirichlet). Raises for
    unsupported or mismatched pairs."""
    hp, hq = p._kl_handle(), q._kl_handle()
    if hp is None or hq is None:
        raise TypeError(
            f"kl_divergence is not defined for ({type(p).__name__}, {type(q).__name__}).")
    result = lib.cml_kl_divergence(hp, hq)
    if result == ffi.NULL:
        raise TypeError(
            f"kl_divergence is not defined for ({type(p).__name__}, {type(q).__name__}).")
    return Tensor(result)


__all__ = [
    "Distribution", "Normal", "Bernoulli", "Categorical", "Uniform", "Exponential",
    "Laplace", "Cauchy", "Gamma", "Beta", "Chi2", "StudentT", "Poisson", "Geometric",
    "Binomial", "Dirichlet", "MultivariateNormal", "kl_divergence",
]
