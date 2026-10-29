/*
 * Extended distributions (Uniform, Exponential, Laplace, Cauchy, Gamma, Beta,
 * Chi2, StudentT, Poisson, Geometric, Binomial, Dirichlet, MultivariateNormal)
 * and the kl_divergence dispatcher. Closed-form log_prob / entropy / variance
 * are checked against scipy oracles; samplers are checked by empirical mean and
 * variance; KL is checked against known analytic values.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cml.h"
#include "core/threefry.h"
#include "nn/distributions.h"
#include "test_harness.h"
#include "test_require.h"

static TensorConfig cfg = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static Tensor* mk(const float* data, int* shape, int ndim) {
    Tensor* t = tensor_zeros(shape, ndim, &cfg);
    memcpy(tensor_data_ptr(t), data, tensor_numel(shape, ndim) * sizeof(float));
    return t;
}

/* Scalar value of a [1]-element result tensor. */
static float scalar(Tensor* t) {
    REQUIRE(t != NULL);
    tensor_ensure_executed(t);
    return ((const float*)tensor_data_ptr(t))[0];
}

static int approx(float got, float want, float tol) { return fabsf(got - want) <= tol; }

/* Build a [M] tensor filled with a constant parameter. */
static Tensor* filled(int m, float v) {
    int sh[1] = {m};
    Tensor* t = tensor_zeros(sh, 1, &cfg);
    float* d  = (float*)tensor_data_ptr(t);
    for (int i = 0; i < m; i++)
        d[i] = v;
    return t;
}

static void moments(Tensor* s, int m, double* mean, double* var) {
    tensor_ensure_executed(s);
    const float* v = (const float*)tensor_data_ptr(s);
    double mu      = 0;
    for (int i = 0; i < m; i++)
        mu += v[i];
    mu /= m;
    double vv = 0;
    for (int i = 0; i < m; i++) {
        double d = v[i] - mu;
        vv += d * d;
    }
    *mean = mu;
    *var  = vv / m;
}

/* ------------------------------------------------------------- log_prob / stats */

static int test_uniform(void) {
    cml_reset_ir_context();
    float lo[1] = {0}, hi[1] = {2}, x[1] = {0.5f};
    int sh[1] = {1};
    REQUIRE(approx(scalar(cml_uniform_log_prob(mk(lo, sh, 1), mk(hi, sh, 1), mk(x, sh, 1))),
                   -0.693147f, 1e-4f));
    REQUIRE(approx(scalar(cml_uniform_mean(mk(lo, sh, 1), mk(hi, sh, 1))), 1.0f, 1e-5f));
    REQUIRE(approx(scalar(cml_uniform_variance(mk(lo, sh, 1), mk(hi, sh, 1))), 1.0f / 3.0f, 1e-5f));
    REQUIRE(approx(scalar(cml_uniform_entropy(mk(lo, sh, 1), mk(hi, sh, 1))), 0.693147f, 1e-4f));
    cml_reset_ir_context();
    return 1;
}

static int test_exponential(void) {
    cml_reset_ir_context();
    float r[1] = {2}, x[1] = {1.0f};
    int sh[1] = {1};
    REQUIRE(approx(scalar(cml_exponential_log_prob(mk(r, sh, 1), mk(x, sh, 1))), -1.306853f, 1e-4f));
    REQUIRE(approx(scalar(cml_exponential_mean(mk(r, sh, 1))), 0.5f, 1e-5f));
    REQUIRE(approx(scalar(cml_exponential_variance(mk(r, sh, 1))), 0.25f, 1e-5f));
    REQUIRE(approx(scalar(cml_exponential_entropy(mk(r, sh, 1))), 0.306853f, 1e-4f));
    cml_reset_ir_context();
    return 1;
}

static int test_laplace(void) {
    cml_reset_ir_context();
    float m[1] = {0}, s[1] = {1}, x[1] = {1.0f};
    int sh[1] = {1};
    REQUIRE(approx(scalar(cml_laplace_log_prob(mk(m, sh, 1), mk(s, sh, 1), mk(x, sh, 1))),
                   -1.693147f, 1e-4f));
    REQUIRE(approx(scalar(cml_laplace_variance(mk(s, sh, 1))), 2.0f, 1e-5f));
    REQUIRE(approx(scalar(cml_laplace_entropy(mk(s, sh, 1))), 1.693147f, 1e-4f));
    cml_reset_ir_context();
    return 1;
}

static int test_cauchy(void) {
    cml_reset_ir_context();
    float m[1] = {0}, s[1] = {1}, x[1] = {1.0f};
    int sh[1] = {1};
    REQUIRE(approx(scalar(cml_cauchy_log_prob(mk(m, sh, 1), mk(s, sh, 1), mk(x, sh, 1))), -1.837877f,
                   1e-4f));
    REQUIRE(approx(scalar(cml_cauchy_entropy(mk(s, sh, 1))), 2.531024f, 1e-4f));
    cml_reset_ir_context();
    return 1;
}

static int test_gamma(void) {
    cml_reset_ir_context();
    float a[1] = {2}, b[1] = {1}, x[1] = {1.0f};
    int sh[1] = {1};
    REQUIRE(approx(scalar(cml_gamma_log_prob(mk(a, sh, 1), mk(b, sh, 1), mk(x, sh, 1))), -1.0f,
                   1e-4f));
    REQUIRE(approx(scalar(cml_gamma_mean(mk(a, sh, 1), mk(b, sh, 1))), 2.0f, 1e-5f));
    REQUIRE(approx(scalar(cml_gamma_variance(mk(a, sh, 1), mk(b, sh, 1))), 2.0f, 1e-5f));
    REQUIRE(approx(scalar(cml_gamma_entropy(mk(a, sh, 1), mk(b, sh, 1))), 1.577216f, 1e-4f));
    cml_reset_ir_context();
    return 1;
}

static int test_beta(void) {
    cml_reset_ir_context();
    float a[1] = {2}, b[1] = {3}, x[1] = {0.5f};
    int sh[1] = {1};
    REQUIRE(
        approx(scalar(cml_beta_log_prob(mk(a, sh, 1), mk(b, sh, 1), mk(x, sh, 1))), 0.405465f, 1e-4f));
    REQUIRE(approx(scalar(cml_beta_mean(mk(a, sh, 1), mk(b, sh, 1))), 0.4f, 1e-5f));
    REQUIRE(approx(scalar(cml_beta_variance(mk(a, sh, 1), mk(b, sh, 1))), 0.04f, 1e-5f));
    REQUIRE(approx(scalar(cml_beta_entropy(mk(a, sh, 1), mk(b, sh, 1))), -0.234907f, 1e-4f));
    cml_reset_ir_context();
    return 1;
}

static int test_chi2(void) {
    cml_reset_ir_context();
    float df[1] = {4}, x[1] = {2.0f};
    int sh[1] = {1};
    REQUIRE(approx(scalar(cml_chi2_log_prob(mk(df, sh, 1), mk(x, sh, 1))), -1.693147f, 1e-4f));
    REQUIRE(approx(scalar(cml_chi2_mean(mk(df, sh, 1))), 4.0f, 1e-5f));
    REQUIRE(approx(scalar(cml_chi2_variance(mk(df, sh, 1))), 8.0f, 1e-5f));
    REQUIRE(approx(scalar(cml_chi2_entropy(mk(df, sh, 1))), 2.270363f, 1e-4f));
    cml_reset_ir_context();
    return 1;
}

static int test_studentt(void) {
    cml_reset_ir_context();
    float df[1] = {3}, loc[1] = {0}, sc[1] = {1}, x[1] = {1.0f};
    int sh[1] = {1};
    REQUIRE(approx(scalar(cml_studentt_log_prob(mk(df, sh, 1), mk(loc, sh, 1), mk(sc, sh, 1),
                                                mk(x, sh, 1))),
                   -1.576253f, 1e-4f));
    REQUIRE(approx(scalar(cml_studentt_variance(mk(df, sh, 1), mk(sc, sh, 1))), 3.0f, 1e-5f));
    REQUIRE(approx(scalar(cml_studentt_entropy(mk(df, sh, 1), mk(sc, sh, 1))), 1.773478f, 1e-4f));
    cml_reset_ir_context();
    return 1;
}

static int test_poisson(void) {
    cml_reset_ir_context();
    float r[1] = {3}, x[1] = {2.0f};
    int sh[1] = {1};
    REQUIRE(approx(scalar(cml_poisson_log_prob(mk(r, sh, 1), mk(x, sh, 1))), -1.495923f, 1e-4f));
    REQUIRE(approx(scalar(cml_poisson_mean(mk(r, sh, 1))), 3.0f, 1e-5f));
    REQUIRE(approx(scalar(cml_poisson_variance(mk(r, sh, 1))), 3.0f, 1e-5f));
    cml_reset_ir_context();
    return 1;
}

static int test_geometric(void) {
    cml_reset_ir_context();
    float p[1] = {0.5f}, x[1] = {2.0f};
    int sh[1] = {1};
    REQUIRE(approx(scalar(cml_geometric_log_prob(mk(p, sh, 1), mk(x, sh, 1))), -2.079442f, 1e-4f));
    REQUIRE(approx(scalar(cml_geometric_mean(mk(p, sh, 1))), 1.0f, 1e-5f));
    REQUIRE(approx(scalar(cml_geometric_variance(mk(p, sh, 1))), 2.0f, 1e-5f));
    REQUIRE(approx(scalar(cml_geometric_entropy(mk(p, sh, 1))), 1.386294f, 1e-4f));
    cml_reset_ir_context();
    return 1;
}

static int test_binomial(void) {
    cml_reset_ir_context();
    float p[1] = {0.5f}, x[1] = {5.0f};
    int sh[1] = {1};
    REQUIRE(approx(scalar(cml_binomial_log_prob(10, mk(p, sh, 1), mk(x, sh, 1))), -1.402043f, 1e-4f));
    REQUIRE(approx(scalar(cml_binomial_mean(10, mk(p, sh, 1))), 5.0f, 1e-5f));
    REQUIRE(approx(scalar(cml_binomial_variance(10, mk(p, sh, 1))), 2.5f, 1e-5f));
    cml_reset_ir_context();
    return 1;
}

static int test_dirichlet(void) {
    cml_reset_ir_context();
    float a[3] = {1, 1, 1}, x[3] = {1.0f / 3, 1.0f / 3, 1.0f / 3};
    int sh[1] = {3};
    REQUIRE(approx(scalar(cml_dirichlet_log_prob(mk(a, sh, 1), mk(x, sh, 1))), 0.693147f, 1e-4f));

    float a2[3] = {2, 3, 4}, x2[3] = {0.2f, 0.3f, 0.5f};
    REQUIRE(approx(scalar(cml_dirichlet_log_prob(mk(a2, sh, 1), mk(x2, sh, 1))), 2.022871f, 1e-3f));
    REQUIRE(approx(scalar(cml_dirichlet_entropy(mk(a2, sh, 1))), -1.312553f, 1e-4f));

    Tensor* mean = cml_dirichlet_mean(mk(a2, sh, 1));
    tensor_ensure_executed(mean);
    const float* md = (const float*)tensor_data_ptr(mean);
    REQUIRE(approx(md[0], 0.222222f, 1e-5f) && approx(md[1], 0.333333f, 1e-5f) &&
            approx(md[2], 0.444444f, 1e-5f));

    Tensor* var = cml_dirichlet_variance(mk(a2, sh, 1));
    tensor_ensure_executed(var);
    const float* vd = (const float*)tensor_data_ptr(var);
    REQUIRE(approx(vd[0], 0.017284f, 1e-5f) && approx(vd[1], 0.022222f, 1e-5f) &&
            approx(vd[2], 0.024691f, 1e-5f));
    cml_reset_ir_context();
    return 1;
}

static int test_mvn(void) {
    cml_reset_ir_context();
    float mu[2]  = {1.0f, -1.0f};
    float cov[4] = {2.0f, 0.5f, 0.5f, 1.0f};
    float x[2]   = {0.5f, 0.0f};
    int msh[1]   = {2};
    int csh[2]   = {2, 2};
    REQUIRE(approx(scalar(cml_mvn_log_prob(mk(mu, msh, 1), mk(cov, csh, 2), mk(x, msh, 1))),
                   -2.903399f, 1e-3f));
    REQUIRE(approx(scalar(cml_mvn_entropy(mk(cov, csh, 2))), 3.117685f, 1e-4f));
    cml_reset_ir_context();
    return 1;
}

/* --------------------------------------------------------------- sampling moments */

static int test_sampling_moments(void) {
    const int M = 40000;
    cml_reset_ir_context();
    cml_rng_set_global_seed(123);

    double mean, var;
    /* Exponential(2): mean 0.5, var 0.25. */
    moments(cml_exponential_rsample(filled(M, 2.0f)), M, &mean, &var);
    REQUIRE(fabs(mean - 0.5) < 0.02 && fabs(var - 0.25) < 0.02);

    /* Uniform(0,2): mean 1, var 1/3. */
    {
        Tensor* lo = filled(M, 0.0f);
        Tensor* hi = filled(M, 2.0f);
        moments(cml_uniform_rsample(lo, hi), M, &mean, &var);
        REQUIRE(fabs(mean - 1.0) < 0.02 && fabs(var - 1.0 / 3.0) < 0.02);
    }

    /* Gamma(2,1): mean 2, var 2. */
    moments(cml_gamma_sample(filled(M, 2.0f), filled(M, 1.0f)), M, &mean, &var);
    REQUIRE(fabs(mean - 2.0) < 0.05 && fabs(var - 2.0) < 0.15);

    /* Gamma with shape < 1 (boosting path), Gamma(0.5,1): mean 0.5, var 0.5. */
    moments(cml_gamma_sample(filled(M, 0.5f), filled(M, 1.0f)), M, &mean, &var);
    REQUIRE(fabs(mean - 0.5) < 0.03 && fabs(var - 0.5) < 0.08);

    /* Beta(2,3): mean 0.4, var 0.04. */
    moments(cml_beta_sample(filled(M, 2.0f), filled(M, 3.0f)), M, &mean, &var);
    REQUIRE(fabs(mean - 0.4) < 0.01 && fabs(var - 0.04) < 0.01);

    /* Poisson(4): mean 4, var 4. */
    moments(cml_poisson_sample(filled(M, 4.0f)), M, &mean, &var);
    REQUIRE(fabs(mean - 4.0) < 0.08 && fabs(var - 4.0) < 0.2);

    /* Geometric(0.3): mean (1-p)/p = 2.333, var (1-p)/p^2 = 7.778. */
    moments(cml_geometric_sample(filled(M, 0.3f)), M, &mean, &var);
    REQUIRE(fabs(mean - 7.0 / 3.0) < 0.1 && fabs(var - 7.0 / 0.9) < 0.6);

    /* Binomial(10,0.5): mean 5, var 2.5. */
    moments(cml_binomial_sample(10, filled(M, 0.5f)), M, &mean, &var);
    REQUIRE(fabs(mean - 5.0) < 0.05 && fabs(var - 2.5) < 0.1);

    /* StudentT(10): mean 0, var df/(df-2) = 1.25. */
    moments(cml_studentt_sample(filled(M, 10.0f), filled(M, 0.0f), filled(M, 1.0f)), M, &mean, &var);
    REQUIRE(fabs(mean - 0.0) < 0.03 && fabs(var - 1.25) < 0.15);

    cml_reset_ir_context();
    return 1;
}

static int test_dirichlet_sample_moments(void) {
    const int M = 40000;
    cml_reset_ir_context();
    cml_rng_set_global_seed(321);
    int sh[2] = {M, 3};
    Tensor* a = tensor_zeros(sh, 2, &cfg);
    float* ad = (float*)tensor_data_ptr(a);
    for (int i = 0; i < M; i++) {
        ad[3 * i]     = 2.0f;
        ad[3 * i + 1] = 3.0f;
        ad[3 * i + 2] = 4.0f;
    }
    Tensor* s = cml_dirichlet_sample(a);
    tensor_ensure_executed(s);
    const float* v = (const float*)tensor_data_ptr(s);
    double col[3]  = {0, 0, 0};
    double rowsum  = 0;
    for (int i = 0; i < M; i++) {
        for (int c = 0; c < 3; c++)
            col[c] += v[3 * i + c];
        rowsum += v[3 * i] + v[3 * i + 1] + v[3 * i + 2];
    }
    for (int c = 0; c < 3; c++)
        col[c] /= M;
    float want[3] = {0.222222f, 0.333333f, 0.444444f};
    int ok        = fabs(rowsum / M - 1.0) < 1e-3; /* rows sum to 1 */
    for (int c = 0; c < 3; c++)
        if (fabs(col[c] - want[c]) > 0.01)
            ok = 0;
    REQUIRE(ok);
    cml_reset_ir_context();
    return 1;
}

/* ------------------------------------------------------------------ kl_divergence */

static int test_kl(void) {
    cml_reset_ir_context();
    int sh[1] = {1};

    float m1[1] = {0}, s1[1] = {1}, m2[1] = {1}, s2[1] = {2};
    REQUIRE(approx(scalar(cml_kl_divergence(cml_dist_normal(mk(m1, sh, 1), mk(s1, sh, 1)),
                                            cml_dist_normal(mk(m2, sh, 1), mk(s2, sh, 1)))),
                   0.443147f, 1e-4f));

    float pb[1] = {0.5f};
    REQUIRE(approx(scalar(cml_kl_divergence(cml_dist_bernoulli(mk(pb, sh, 1)),
                                            cml_dist_bernoulli(mk(pb, sh, 1)))),
                   0.0f, 1e-5f));

    float r1[1] = {2}, r2[1] = {1};
    REQUIRE(approx(scalar(cml_kl_divergence(cml_dist_exponential(mk(r1, sh, 1)),
                                            cml_dist_exponential(mk(r2, sh, 1)))),
                   0.193147f, 1e-4f));

    float ga[1] = {2}, gb[1] = {1}, ga2[1] = {3};
    REQUIRE(approx(scalar(cml_kl_divergence(cml_dist_gamma(mk(ga, sh, 1), mk(gb, sh, 1)),
                                            cml_dist_gamma(mk(ga2, sh, 1), mk(gb, sh, 1)))),
                   0.270363f, 1e-4f));

    float ba1[1] = {2}, bb1[1] = {3}, ba2[1] = {2}, bb2[1] = {2};
    REQUIRE(approx(scalar(cml_kl_divergence(cml_dist_beta(mk(ba1, sh, 1), mk(bb1, sh, 1)),
                                            cml_dist_beta(mk(ba2, sh, 1), mk(bb2, sh, 1)))),
                   0.109814f, 1e-4f));

    float lo1[1] = {0}, hi1[1] = {1}, lo2[1] = {-1}, hi2[1] = {2};
    REQUIRE(approx(scalar(cml_kl_divergence(cml_dist_uniform(mk(lo1, sh, 1), mk(hi1, sh, 1)),
                                            cml_dist_uniform(mk(lo2, sh, 1), mk(hi2, sh, 1)))),
                   1.098612f, 1e-4f));

    float lp[3] = {1, 2, 3}, lq[3] = {0, 0, 0};
    int csh[2] = {1, 3};
    REQUIRE(approx(scalar(cml_kl_divergence(cml_dist_categorical(mk(lp, csh, 2)),
                                            cml_dist_categorical(mk(lq, csh, 2)))),
                   0.266217f, 1e-4f));

    float da[3] = {2, 3, 4}, db[3] = {1, 1, 1};
    int dsh[1] = {3};
    REQUIRE(approx(scalar(cml_kl_divergence(cml_dist_dirichlet(mk(da, dsh, 1)),
                                            cml_dist_dirichlet(mk(db, dsh, 1)))),
                   0.619406f, 1e-4f));

    /* Mismatched types return NULL. */
    REQUIRE(cml_kl_divergence(cml_dist_bernoulli(mk(pb, sh, 1)),
                              cml_dist_exponential(mk(r1, sh, 1))) == NULL);
    cml_reset_ir_context();
    return 1;
}

int main(void) {
    printf("test_distributions_ext\n\n");
    TEST(uniform);
    TEST(exponential);
    TEST(laplace);
    TEST(cauchy);
    TEST(gamma);
    TEST(beta);
    TEST(chi2);
    TEST(studentt);
    TEST(poisson);
    TEST(geometric);
    TEST(binomial);
    TEST(dirichlet);
    TEST(mvn);
    TEST(sampling_moments);
    TEST(dirichlet_sample_moments);
    TEST(kl);
    cml_reset_ir_context();
    return TEST_SUMMARY();
}
