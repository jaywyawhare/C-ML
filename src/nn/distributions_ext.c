#include "nn/distributions.h"

#include <math.h>
#include <stdlib.h>

#include "cml.h"
#include "core/threefry.h"

#define LOG_2PI 1.8378770664093453f
#define LOG_PI 1.1447298858494002f
#define LOG_4PI 2.5310242469692907f
#define CML_PI 3.14159265358979323846f

/**
 * Differentiable t + c.
 * @param t (Tensor*)
 * @param c (float)
 * @return (Tensor*)
 */
static Tensor* add_c(Tensor* t, float c) { return cml_add(t, cml_full_like(t, c)); }

/**
 * Differentiable t * c.
 * @param t (Tensor*)
 * @param c (float)
 * @return (Tensor*)
 */
static Tensor* mul_c(Tensor* t, float c) { return cml_mul(t, cml_full_like(t, c)); }

/**
 * Differentiable 1 - t.
 * @param t (Tensor*)
 * @return (Tensor*)
 */
static Tensor* one_minus(Tensor* t) { return add_c(mul_c(t, -1.0f), 1.0f); }

/**
 * lgamma(a) + lgamma(b) - lgamma(a+b).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
static Tensor* lbeta(Tensor* a, Tensor* b) {
    return cml_sub(cml_add(cml_lgamma(a), cml_lgamma(b)), cml_lgamma(cml_add(a, b)));
}

/**
 * Float32 config on t's device.
 * @param t (Tensor*)
 * @return (TensorConfig)
 */
static TensorConfig cfg_of(Tensor* t) {
    TensorConfig c = {
        .dtype = DTYPE_FLOAT32, .device = t->device, .has_dtype = true, .has_device = true};
    return c;
}

/**
 * Execute t then allocate a detached float output of its shape, exposing the
 * output buffer, element count, and global RNG for elementwise sampling.
 * @param t (Tensor*)
 * @param od (float**)
 * @param n (size_t*)
 * @param rng (CMLRNGState**)
 * @return (Tensor*)
 */
static Tensor* sampler_out(Tensor* t, float** od, size_t* n, CMLRNGState** rng) {
    if (!t || tensor_ensure_executed(t) != 0)
        return NULL;
    TensorConfig cfg = cfg_of(t);
    Tensor* out      = cml_zeros(t->shape, t->ndim, &cfg);
    if (!out)
        return NULL;
    *od  = (float*)tensor_data_ptr(out);
    *n   = tensor_numel(t->shape, t->ndim);
    *rng = cml_rng_get_global();
    return out;
}

/**
 * Draw detached U(0,1) of ref's shape and map each element through fn into a
 * fresh detached noise tensor.
 * @param ref (Tensor*)
 * @param fn (float (*)(float))
 * @return (Tensor*)
 */
static Tensor* map_uniform_noise(Tensor* ref, float (*fn)(float)) {
    TensorConfig cfg = cfg_of(ref);
    Tensor* u        = tensor_rand(ref->shape, ref->ndim, &cfg);
    if (!u || tensor_ensure_executed(u) != 0)
        return NULL;
    const float* ud = (const float*)tensor_data_ptr(u);
    size_t n        = tensor_numel(ref->shape, ref->ndim);
    Tensor* z       = cml_zeros(ref->shape, ref->ndim, &cfg);
    if (!z)
        return NULL;
    float* zd = (float*)tensor_data_ptr(z);
    for (size_t i = 0; i < n; i++)
        zd[i] = fn(ud[i]);
    return z;
}

/**
 * Standard Exp(1) draw from a U(0,1) sample, clamped away from 0 so -log stays finite.
 * @param u (float)
 * @return (float)
 */
static float exp_noise(float u) { return -logf(u < 1e-20f ? 1e-20f : u); }

/**
 * Standard Laplace draw from a U(0,1) sample via inverse-CDF.
 * @param u (float)
 * @return (float)
 */
static float laplace_noise(float u) {
    float w  = u - 0.5f;
    float aw = fabsf(w);
    if (aw > 0.4999999f)
        aw = 0.4999999f;
    float s = (w > 0.0f) ? 1.0f : -1.0f;
    return -s * log1pf(-2.0f * aw);
}

/**
 * Standard Cauchy draw from a U(0,1) sample via inverse-CDF.
 * @param u (float)
 * @return (float)
 */
static float cauchy_noise(float u) { return tanf(CML_PI * (u - 0.5f)); }

/**
 * Reparameterized Uniform(low, high) sample low + (high-low)*u.
 * @param low (Tensor*)
 * @param high (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_uniform_rsample(Tensor* low, Tensor* high) {
    if (!low || !high)
        return NULL;
    TensorConfig cfg = cfg_of(low);
    Tensor* u        = tensor_rand(low->shape, low->ndim, &cfg);
    if (!u)
        return NULL;
    return cml_add(low, cml_mul(cml_sub(high, low), u));
}

/**
 * Elementwise Uniform log-density of value, broadcast to value's shape.
 * @param low (Tensor*)
 * @param high (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_uniform_log_prob(Tensor* low, Tensor* high, Tensor* value) {
    if (!low || !high || !value)
        return NULL;
    Tensor* neg_log = mul_c(cml_log(cml_sub(high, low)), -1.0f);
    return cml_add(neg_log, mul_c(value, 0.0f));
}

/**
 * Uniform mean (low+high)/2.
 * @param low (Tensor*)
 * @param high (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_uniform_mean(Tensor* low, Tensor* high) {
    if (!low || !high)
        return NULL;
    return mul_c(cml_add(low, high), 0.5f);
}

/**
 * Uniform variance (high-low)^2/12.
 * @param low (Tensor*)
 * @param high (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_uniform_variance(Tensor* low, Tensor* high) {
    if (!low || !high)
        return NULL;
    Tensor* r = cml_sub(high, low);
    return mul_c(cml_mul(r, r), 1.0f / 12.0f);
}

/**
 * Uniform entropy log(high-low).
 * @param low (Tensor*)
 * @param high (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_uniform_entropy(Tensor* low, Tensor* high) {
    if (!low || !high)
        return NULL;
    return cml_log(cml_sub(high, low));
}

/**
 * Reparameterized Exponential(rate) sample -log(u)/rate.
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_exponential_rsample(Tensor* rate) {
    if (!rate)
        return NULL;
    Tensor* e = map_uniform_noise(rate, exp_noise);
    return e ? cml_div(e, rate) : NULL;
}

/**
 * Elementwise Exponential log-density of value.
 * @param rate (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_exponential_log_prob(Tensor* rate, Tensor* value) {
    if (!rate || !value)
        return NULL;
    return cml_sub(cml_log(rate), cml_mul(rate, value));
}

/**
 * Exponential mean 1/rate.
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_exponential_mean(Tensor* rate) {
    return rate ? cml_div(cml_ones_like(rate), rate) : NULL;
}

/**
 * Exponential variance 1/rate^2.
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_exponential_variance(Tensor* rate) {
    return rate ? cml_div(cml_ones_like(rate), cml_mul(rate, rate)) : NULL;
}

/**
 * Exponential entropy 1-log(rate).
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_exponential_entropy(Tensor* rate) {
    return rate ? add_c(mul_c(cml_log(rate), -1.0f), 1.0f) : NULL;
}

/**
 * Reparameterized Laplace(loc, scale) sample via inverse-CDF.
 * @param loc (Tensor*)
 * @param scale (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_laplace_rsample(Tensor* loc, Tensor* scale) {
    if (!loc || !scale)
        return NULL;
    Tensor* z = map_uniform_noise(loc, laplace_noise);
    return z ? cml_add(loc, cml_mul(scale, z)) : NULL;
}

/**
 * Elementwise Laplace log-density of value.
 * @param loc (Tensor*)
 * @param scale (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_laplace_log_prob(Tensor* loc, Tensor* scale, Tensor* value) {
    if (!loc || !scale || !value)
        return NULL;
    Tensor* dev = cml_div(cml_sqrt(cml_square(cml_sub(value, loc))), scale);
    return cml_sub(mul_c(cml_log(mul_c(scale, 2.0f)), -1.0f), dev);
}

/**
 * Laplace mean (= loc).
 * @param loc (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_laplace_mean(Tensor* loc) { return loc ? cml_add(loc, cml_zeros_like(loc)) : NULL; }

/**
 * Laplace variance 2*scale^2.
 * @param scale (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_laplace_variance(Tensor* scale) {
    return scale ? mul_c(cml_mul(scale, scale), 2.0f) : NULL;
}

/**
 * Laplace entropy 1+log(2*scale).
 * @param scale (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_laplace_entropy(Tensor* scale) {
    return scale ? add_c(cml_log(mul_c(scale, 2.0f)), 1.0f) : NULL;
}

/**
 * Reparameterized Cauchy(loc, scale) sample loc + scale*tan(pi*(u-0.5)).
 * @param loc (Tensor*)
 * @param scale (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_cauchy_rsample(Tensor* loc, Tensor* scale) {
    if (!loc || !scale)
        return NULL;
    Tensor* t = map_uniform_noise(loc, cauchy_noise);
    return t ? cml_add(loc, cml_mul(scale, t)) : NULL;
}

/**
 * Elementwise Cauchy log-density of value.
 * @param loc (Tensor*)
 * @param scale (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_cauchy_log_prob(Tensor* loc, Tensor* scale, Tensor* value) {
    if (!loc || !scale || !value)
        return NULL;
    Tensor* z    = cml_div(cml_sub(value, loc), scale);
    Tensor* dens = cml_log(add_c(cml_mul(z, z), 1.0f));
    return cml_sub(mul_c(cml_log(scale), -1.0f), add_c(dens, LOG_PI));
}

/**
 * Cauchy entropy log(4*pi*scale).
 * @param scale (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_cauchy_entropy(Tensor* scale) {
    return scale ? add_c(cml_log(scale), LOG_4PI) : NULL;
}

/**
 * Draw a Gamma(shape, 1) variate (Marsaglia-Tsang); shape<1 uses the boosting
 * identity G(a) = G(a+1) * U^(1/a).
 * @param rng (CMLRNGState*)
 * @param shape (float)
 * @return (float)
 */
static float sample_gamma_scalar(CMLRNGState* rng, float shape) {
    if (shape < 1.0f) {
        float u;
        cml_rng_uniform(rng, &u, 1);
        if (u < 1e-20f)
            u = 1e-20f;
        return sample_gamma_scalar(rng, shape + 1.0f) * powf(u, 1.0f / shape);
    }
    float d = shape - 1.0f / 3.0f;
    float c = 1.0f / sqrtf(9.0f * d);
    for (;;) {
        float x, v;
        do {
            cml_rng_normal(rng, &x, 1);
            v = 1.0f + c * x;
        } while (v <= 0.0f);
        v = v * v * v;
        float u;
        cml_rng_uniform(rng, &u, 1);
        float x2 = x * x;
        if (u < 1.0f - 0.0331f * x2 * x2)
            return d * v;
        if (logf(u) < 0.5f * x2 + d * (1.0f - v + logf(v)))
            return d * v;
    }
}

/**
 * Gamma(concentration, rate) variate.
 * @param concentration (Tensor*)
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_gamma_sample(Tensor* concentration, Tensor* rate) {
    if (!rate || tensor_ensure_executed(rate) != 0)
        return NULL;
    float* od;
    size_t n;
    CMLRNGState* rng;
    Tensor* out = sampler_out(concentration, &od, &n, &rng);
    if (!out)
        return NULL;
    const float* a = (const float*)tensor_data_ptr(concentration);
    const float* b = (const float*)tensor_data_ptr(rate);
    for (size_t i = 0; i < n; i++)
        od[i] = sample_gamma_scalar(rng, a[i]) / b[i];
    return out;
}

/**
 * Elementwise Gamma log-density of value.
 * @param concentration (Tensor*)
 * @param rate (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_gamma_log_prob(Tensor* concentration, Tensor* rate, Tensor* value) {
    if (!concentration || !rate || !value)
        return NULL;
    Tensor* t = cml_mul(concentration, cml_log(rate));
    t         = cml_sub(t, cml_lgamma(concentration));
    t         = cml_add(t, cml_mul(add_c(concentration, -1.0f), cml_log(value)));
    return cml_sub(t, cml_mul(rate, value));
}

/**
 * Gamma mean concentration/rate.
 * @param concentration (Tensor*)
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_gamma_mean(Tensor* concentration, Tensor* rate) {
    if (!concentration || !rate)
        return NULL;
    return cml_div(concentration, rate);
}

/**
 * Gamma variance concentration/rate^2.
 * @param concentration (Tensor*)
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_gamma_variance(Tensor* concentration, Tensor* rate) {
    if (!concentration || !rate)
        return NULL;
    return cml_div(concentration, cml_mul(rate, rate));
}

/**
 * Gamma entropy.
 * @param concentration (Tensor*)
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_gamma_entropy(Tensor* concentration, Tensor* rate) {
    if (!concentration || !rate)
        return NULL;
    Tensor* t = cml_sub(concentration, cml_log(rate));
    t         = cml_add(t, cml_lgamma(concentration));
    return cml_add(t, cml_mul(one_minus(concentration), cml_digamma(concentration)));
}

/**
 * Beta(c1, c0) variate X/(X+Y) with X~Gamma(c1,1), Y~Gamma(c0,1).
 * @param c1 (Tensor*)
 * @param c0 (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_beta_sample(Tensor* c1, Tensor* c0) {
    if (!c0 || tensor_ensure_executed(c0) != 0)
        return NULL;
    float* od;
    size_t n;
    CMLRNGState* rng;
    Tensor* out = sampler_out(c1, &od, &n, &rng);
    if (!out)
        return NULL;
    const float* a = (const float*)tensor_data_ptr(c1);
    const float* b = (const float*)tensor_data_ptr(c0);
    for (size_t i = 0; i < n; i++) {
        float x = sample_gamma_scalar(rng, a[i]);
        float y = sample_gamma_scalar(rng, b[i]);
        od[i]   = x / (x + y);
    }
    return out;
}

/**
 * Elementwise Beta log-density of value.
 * @param c1 (Tensor*)
 * @param c0 (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_beta_log_prob(Tensor* c1, Tensor* c0, Tensor* value) {
    if (!c1 || !c0 || !value)
        return NULL;
    Tensor* t = cml_add(cml_mul(add_c(c1, -1.0f), cml_log(value)),
                        cml_mul(add_c(c0, -1.0f), cml_log(one_minus(value))));
    return cml_sub(t, lbeta(c1, c0));
}

/**
 * Beta mean c1/(c1+c0).
 * @param c1 (Tensor*)
 * @param c0 (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_beta_mean(Tensor* c1, Tensor* c0) {
    if (!c1 || !c0)
        return NULL;
    return cml_div(c1, cml_add(c1, c0));
}

/**
 * Beta variance.
 * @param c1 (Tensor*)
 * @param c0 (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_beta_variance(Tensor* c1, Tensor* c0) {
    if (!c1 || !c0)
        return NULL;
    Tensor* s   = cml_add(c1, c0);
    Tensor* num = cml_mul(c1, c0);
    Tensor* den = cml_mul(cml_mul(s, s), add_c(s, 1.0f));
    return cml_div(num, den);
}

/**
 * Beta entropy.
 * @param c1 (Tensor*)
 * @param c0 (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_beta_entropy(Tensor* c1, Tensor* c0) {
    if (!c1 || !c0)
        return NULL;
    Tensor* s = cml_add(c1, c0);
    Tensor* e = lbeta(c1, c0);
    e         = cml_sub(e, cml_mul(add_c(c1, -1.0f), cml_digamma(c1)));
    e         = cml_sub(e, cml_mul(add_c(c0, -1.0f), cml_digamma(c0)));
    return cml_add(e, cml_mul(add_c(s, -2.0f), cml_digamma(s)));
}

/**
 * Chi2(df) variate (== Gamma(df/2, 1/2)).
 * @param df (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_chi2_sample(Tensor* df) {
    if (!df)
        return NULL;
    Tensor* conc = mul_c(df, 0.5f);
    Tensor* rate = cml_full_like(df, 0.5f);
    return cml_gamma_sample(conc, rate);
}

/**
 * Elementwise Chi2 log-density of value.
 * @param df (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_chi2_log_prob(Tensor* df, Tensor* value) {
    if (!df || !value)
        return NULL;
    return cml_gamma_log_prob(mul_c(df, 0.5f), cml_full_like(df, 0.5f), value);
}

/**
 * Chi2 mean (= df).
 * @param df (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_chi2_mean(Tensor* df) { return df ? cml_add(df, cml_zeros_like(df)) : NULL; }

/**
 * Chi2 variance 2*df.
 * @param df (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_chi2_variance(Tensor* df) { return df ? mul_c(df, 2.0f) : NULL; }

/**
 * Chi2 entropy.
 * @param df (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_chi2_entropy(Tensor* df) {
    return df ? cml_gamma_entropy(mul_c(df, 0.5f), cml_full_like(df, 0.5f)) : NULL;
}

/**
 * StudentT(df, loc, scale) variate loc + scale*Z/sqrt(Chi2(df)/df).
 * @param df (Tensor*)
 * @param loc (Tensor*)
 * @param scale (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_studentt_sample(Tensor* df, Tensor* loc, Tensor* scale) {
    if (!loc || !scale)
        return NULL;
    float* zd;
    size_t n;
    CMLRNGState* rng;
    Tensor* z = sampler_out(df, &zd, &n, &rng);
    if (!z)
        return NULL;
    const float* dd = (const float*)tensor_data_ptr(df);
    for (size_t i = 0; i < n; i++) {
        float g;
        cml_rng_normal(rng, &g, 1);
        float v = sample_gamma_scalar(rng, dd[i] * 0.5f) * 2.0f;
        zd[i]   = g / sqrtf(v / dd[i]);
    }
    return cml_add(loc, cml_mul(scale, z));
}

/**
 * Elementwise StudentT log-density of value.
 * @param df (Tensor*)
 * @param loc (Tensor*)
 * @param scale (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_studentt_log_prob(Tensor* df, Tensor* loc, Tensor* scale, Tensor* value) {
    if (!df || !loc || !scale || !value)
        return NULL;
    Tensor* z       = cml_div(cml_sub(value, loc), scale);
    Tensor* dfp1_2  = mul_c(add_c(df, 1.0f), 0.5f);
    Tensor* df_2    = mul_c(df, 0.5f);
    Tensor* t       = cml_sub(cml_lgamma(dfp1_2), cml_lgamma(df_2));
    t               = cml_sub(t, mul_c(cml_log(mul_c(df, CML_PI)), 0.5f));
    t               = cml_sub(t, cml_log(scale));
    Tensor* z2overd = cml_div(cml_mul(z, z), df);
    return cml_sub(t, cml_mul(dfp1_2, cml_log(add_c(z2overd, 1.0f))));
}

/**
 * StudentT mean (= loc).
 * @param df (Tensor*)
 * @param loc (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_studentt_mean(Tensor* df, Tensor* loc) {
    (void)df;
    return loc ? cml_add(loc, cml_zeros_like(loc)) : NULL;
}

/**
 * StudentT variance scale^2*df/(df-2), valid for df > 2.
 * @param df (Tensor*)
 * @param scale (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_studentt_variance(Tensor* df, Tensor* scale) {
    if (!df || !scale)
        return NULL;
    return cml_mul(cml_mul(scale, scale), cml_div(df, add_c(df, -2.0f)));
}

/**
 * StudentT entropy.
 * @param df (Tensor*)
 * @param scale (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_studentt_entropy(Tensor* df, Tensor* scale) {
    if (!df || !scale)
        return NULL;
    Tensor* dfp1_2 = mul_c(add_c(df, 1.0f), 0.5f);
    Tensor* df_2   = mul_c(df, 0.5f);
    Tensor* lb     = lbeta(df_2, cml_full_like(df, 0.5f));
    Tensor* e      = cml_log(scale);
    e = cml_add(e, cml_mul(dfp1_2, cml_sub(cml_digamma(dfp1_2), cml_digamma(df_2))));
    e = cml_add(e, mul_c(cml_log(df), 0.5f));
    return cml_add(e, lb);
}

/**
 * Draw a Poisson(lam) variate: Knuth's multiplication method below lam 30, a
 * clamped normal approximation above.
 * @param rng (CMLRNGState*)
 * @param lam (float)
 * @return (float)
 */
static float sample_poisson_scalar(CMLRNGState* rng, float lam) {
    if (lam <= 0.0f)
        return 0.0f;
    if (lam < 30.0f) {
        float target = expf(-lam);
        float p      = 1.0f;
        int k        = 0;
        do {
            k++;
            float u;
            cml_rng_uniform(rng, &u, 1);
            p *= u;
        } while (p > target);
        return (float)(k - 1);
    }
    float g;
    cml_rng_normal(rng, &g, 1);
    float v = lam + sqrtf(lam) * g;
    v       = floorf(v + 0.5f);
    return v < 0.0f ? 0.0f : v;
}

/**
 * Poisson(rate) variate.
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_poisson_sample(Tensor* rate) {
    float* od;
    size_t n;
    CMLRNGState* rng;
    Tensor* out = sampler_out(rate, &od, &n, &rng);
    if (!out)
        return NULL;
    const float* r = (const float*)tensor_data_ptr(rate);
    for (size_t i = 0; i < n; i++)
        od[i] = sample_poisson_scalar(rng, r[i]);
    return out;
}

/**
 * Elementwise Poisson log-probability of value.
 * @param rate (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_poisson_log_prob(Tensor* rate, Tensor* value) {
    if (!rate || !value)
        return NULL;
    Tensor* t = cml_sub(cml_mul(value, cml_log(rate)), rate);
    return cml_sub(t, cml_lgamma(add_c(value, 1.0f)));
}

/**
 * Poisson mean (= rate).
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_poisson_mean(Tensor* rate) { return rate ? cml_add(rate, cml_zeros_like(rate)) : NULL; }

/**
 * Poisson variance (= rate).
 * @param rate (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_poisson_variance(Tensor* rate) {
    return rate ? cml_add(rate, cml_zeros_like(rate)) : NULL;
}

/**
 * Geometric(probs) variate: number of failures before the first success.
 * @param probs (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_geometric_sample(Tensor* probs) {
    float* od;
    size_t n;
    CMLRNGState* rng;
    Tensor* out = sampler_out(probs, &od, &n, &rng);
    if (!out)
        return NULL;
    const float* p = (const float*)tensor_data_ptr(probs);
    for (size_t i = 0; i < n; i++) {
        float pp = p[i];
        if (pp >= 1.0f) {
            od[i] = 0.0f;
            continue;
        }
        if (pp < 1e-20f)
            pp = 1e-20f;
        float u;
        cml_rng_uniform(rng, &u, 1);
        if (u < 1e-20f)
            u = 1e-20f;
        od[i] = floorf(logf(u) / log1pf(-pp));
    }
    return out;
}

/**
 * Elementwise Geometric log-probability of value.
 * @param probs (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_geometric_log_prob(Tensor* probs, Tensor* value) {
    if (!probs || !value)
        return NULL;
    return cml_add(cml_mul(value, cml_log(one_minus(probs))), cml_log(probs));
}

/**
 * Geometric mean (1-probs)/probs.
 * @param probs (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_geometric_mean(Tensor* probs) {
    return probs ? cml_div(one_minus(probs), probs) : NULL;
}

/**
 * Geometric variance (1-probs)/probs^2.
 * @param probs (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_geometric_variance(Tensor* probs) {
    return probs ? cml_div(one_minus(probs), cml_mul(probs, probs)) : NULL;
}

/**
 * Geometric entropy.
 * @param probs (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_geometric_entropy(Tensor* probs) {
    if (!probs)
        return NULL;
    Tensor* bern = mul_c(cml_add(cml_mul(probs, cml_log(probs)),
                                 cml_mul(one_minus(probs), cml_log(one_minus(probs)))),
                         -1.0f);
    return cml_div(bern, probs);
}

/**
 * Binomial(total_count, probs) variate: sum of total_count Bernoulli draws.
 * @param total_count (int)
 * @param probs (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_binomial_sample(int total_count, Tensor* probs) {
    if (total_count < 0)
        return NULL;
    float* od;
    size_t n;
    CMLRNGState* rng;
    Tensor* out = sampler_out(probs, &od, &n, &rng);
    if (!out)
        return NULL;
    const float* p = (const float*)tensor_data_ptr(probs);
    for (size_t i = 0; i < n; i++) {
        int c = 0;
        for (int t = 0; t < total_count; t++) {
            float u;
            cml_rng_uniform(rng, &u, 1);
            if (u < p[i])
                c++;
        }
        od[i] = (float)c;
    }
    return out;
}

/**
 * Elementwise Binomial log-probability of value.
 * @param total_count (int)
 * @param probs (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_binomial_log_prob(int total_count, Tensor* probs, Tensor* value) {
    if (!probs || !value)
        return NULL;
    float lg_n1   = lgammaf((float)total_count + 1.0f);
    Tensor* comb  = cml_sub(cml_full_like(value, lg_n1), cml_lgamma(add_c(value, 1.0f)));
    comb          = cml_sub(comb, cml_lgamma(add_c(add_c(mul_c(value, -1.0f), (float)total_count),
                                                   1.0f)));
    Tensor* terms = cml_add(cml_mul(value, cml_log(probs)),
                            cml_mul(add_c(mul_c(value, -1.0f), (float)total_count),
                                    cml_log(one_minus(probs))));
    return cml_add(comb, terms);
}

/**
 * Binomial mean total_count*probs.
 * @param total_count (int)
 * @param probs (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_binomial_mean(int total_count, Tensor* probs) {
    return probs ? mul_c(probs, (float)total_count) : NULL;
}

/**
 * Binomial variance total_count*probs*(1-probs).
 * @param total_count (int)
 * @param probs (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_binomial_variance(int total_count, Tensor* probs) {
    if (!probs)
        return NULL;
    return mul_c(cml_mul(probs, one_minus(probs)), (float)total_count);
}

/**
 * Dirichlet(concentration) variate over the last axis of concentration [..., K].
 * @param concentration (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_dirichlet_sample(Tensor* concentration) {
    float* od;
    size_t total;
    CMLRNGState* rng;
    Tensor* out = sampler_out(concentration, &od, &total, &rng);
    if (!out)
        return NULL;
    const float* a = (const float*)tensor_data_ptr(concentration);
    int k          = concentration->shape[concentration->ndim - 1];
    size_t rows    = total / (size_t)k;
    for (size_t r = 0; r < rows; r++) {
        double s = 0.0;
        for (int c = 0; c < k; c++) {
            float g       = sample_gamma_scalar(rng, a[r * k + c]);
            od[r * k + c] = g;
            s += (double)g;
        }
        if (s <= 0.0)
            s = 1e-20;
        for (int c = 0; c < k; c++)
            od[r * k + c] = (float)(od[r * k + c] / s);
    }
    return out;
}

/**
 * Dirichlet log-density of value over the last axis.
 * @param concentration (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_dirichlet_log_prob(Tensor* concentration, Tensor* value) {
    if (!concentration || !value)
        return NULL;
    Tensor* a0    = cml_sum(concentration, -1, false);
    Tensor* term1 = cml_sum(cml_mul(add_c(concentration, -1.0f), cml_log(value)), -1, false);
    Tensor* term2 = cml_lgamma(a0);
    Tensor* term3 = cml_sum(cml_lgamma(concentration), -1, false);
    return cml_sub(cml_add(term1, term2), term3);
}

/**
 * Dirichlet mean concentration/sum(concentration).
 * @param concentration (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_dirichlet_mean(Tensor* concentration) {
    if (!concentration)
        return NULL;
    Tensor* a0 = cml_sum(concentration, -1, true);
    return cml_div(concentration, a0);
}

/**
 * Dirichlet variance over the last axis.
 * @param concentration (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_dirichlet_variance(Tensor* concentration) {
    if (!concentration)
        return NULL;
    Tensor* a0  = cml_sum(concentration, -1, true);
    Tensor* num = cml_mul(concentration, cml_sub(a0, concentration));
    Tensor* den = cml_mul(cml_mul(a0, a0), add_c(a0, 1.0f));
    return cml_div(num, den);
}

/**
 * Dirichlet entropy.
 * @param concentration (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_dirichlet_entropy(Tensor* concentration) {
    if (!concentration)
        return NULL;
    int k            = concentration->shape[concentration->ndim - 1];
    Tensor* a0       = cml_sum(concentration, -1, false);
    Tensor* log_beta = cml_sub(cml_sum(cml_lgamma(concentration), -1, false), cml_lgamma(a0));
    Tensor* t2       = cml_mul(add_c(a0, -(float)k), cml_digamma(a0));
    Tensor* t3 = cml_sum(cml_mul(add_c(concentration, -1.0f), cml_digamma(concentration)), -1, false);
    return cml_sub(cml_add(log_beta, t2), t3);
}

/**
 * log|covariance| from a Cholesky factor: 2*sum(log(diag(L))).
 * @param Ld (const float*)
 * @param d (int)
 * @return (double)
 */
static double chol_logdet(const float* Ld, int d) {
    double logdet = 0.0;
    for (int i = 0; i < d; i++)
        logdet += log((double)Ld[i * d + i]);
    return 2.0 * logdet;
}

/**
 * Reparameterized MultivariateNormal sample loc + L*eps via Cholesky factor L.
 * @param loc (Tensor*)
 * @param covariance (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_mvn_rsample(Tensor* loc, Tensor* covariance) {
    if (!loc || !covariance || covariance->ndim != 2 || loc->ndim != 1)
        return NULL;
    int d = loc->shape[0];
    if (covariance->shape[0] != d || covariance->shape[1] != d)
        return NULL;
    Tensor* L = cml_cholesky(covariance);
    if (!L)
        return NULL;
    TensorConfig cfg = cfg_of(loc);
    int esh[2]       = {d, 1};
    Tensor* eps      = tensor_randn(esh, 2, &cfg);
    if (!eps)
        return NULL;
    Tensor* leps = cml_matmul(L, eps);
    int fsh[1]   = {d};
    Tensor* flat = cml_reshape(leps, fsh, 1);
    return cml_add(loc, flat);
}

/**
 * MultivariateNormal log-density of value ([D] point or [N, D] batch).
 * @param loc (Tensor*)
 * @param covariance (Tensor*)
 * @param value (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_mvn_log_prob(Tensor* loc, Tensor* covariance, Tensor* value) {
    if (!loc || !covariance || !value || loc->ndim != 1 || covariance->ndim != 2)
        return NULL;
    int d = loc->shape[0];
    if (covariance->shape[0] != d || covariance->shape[1] != d)
        return NULL;
    int rows;
    if (value->ndim == 1 && value->shape[0] == d)
        rows = 1;
    else if (value->ndim == 2 && value->shape[1] == d)
        rows = value->shape[0];
    else
        return NULL;

    Tensor* L = cml_cholesky(covariance);
    if (!L || tensor_ensure_executed(L) != 0 || tensor_ensure_executed(loc) != 0 ||
        tensor_ensure_executed(value) != 0)
        return NULL;
    const float* Ld = (const float*)tensor_data_ptr(L);
    const float* mu = (const float*)tensor_data_ptr(loc);
    const float* xv = (const float*)tensor_data_ptr(value);

    double logdet = chol_logdet(Ld, d);

    TensorConfig cfg = cfg_of(loc);
    int osh[1]       = {rows};
    Tensor* out      = cml_zeros(osh, 1, &cfg);
    if (!out)
        return NULL;
    float* od = (float*)tensor_data_ptr(out);

    double* z = (double*)malloc(sizeof(double) * (size_t)d);
    if (!z)
        return NULL;
    for (int r = 0; r < rows; r++) {
        /* Forward substitution L z = (x - mu); maha = ||z||^2. */
        double maha = 0.0;
        for (int i = 0; i < d; i++) {
            double acc = (double)xv[r * d + i] - (double)mu[i];
            for (int j = 0; j < i; j++)
                acc -= (double)Ld[i * d + j] * z[j];
            z[i] = acc / (double)Ld[i * d + i];
            maha += z[i] * z[i];
        }
        od[r] = (float)(-0.5 * ((double)d * (double)LOG_2PI + logdet + maha));
    }
    free(z);
    return out;
}

/**
 * MultivariateNormal entropy from the covariance log-determinant.
 * @param covariance (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_mvn_entropy(Tensor* covariance) {
    if (!covariance || covariance->ndim != 2)
        return NULL;
    int d     = covariance->shape[0];
    Tensor* L = cml_cholesky(covariance);
    if (!L || tensor_ensure_executed(L) != 0)
        return NULL;
    const float* Ld  = (const float*)tensor_data_ptr(L);
    double logdet    = chol_logdet(Ld, d);
    TensorConfig cfg = cfg_of(covariance);
    int osh[1]       = {1};
    Tensor* out      = cml_zeros(osh, 1, &cfg);
    if (!out)
        return NULL;
    ((float*)tensor_data_ptr(out))[0] =
        (float)(0.5 * (double)d * (1.0 + (double)LOG_2PI) + 0.5 * logdet);
    return out;
}

/**
 * Wrap a Normal(loc, scale) as a CMLDistribution.
 * @param loc (Tensor*)
 * @param scale (Tensor*)
 * @return (CMLDistribution)
 */
CMLDistribution cml_dist_normal(Tensor* loc, Tensor* scale) {
    CMLDistribution d = {CML_DIST_NORMAL, loc, scale};
    return d;
}

/**
 * Wrap a Bernoulli(probs) as a CMLDistribution.
 * @param probs (Tensor*)
 * @return (CMLDistribution)
 */
CMLDistribution cml_dist_bernoulli(Tensor* probs) {
    CMLDistribution d = {CML_DIST_BERNOULLI, probs, NULL};
    return d;
}

/**
 * Wrap a Categorical(logits) as a CMLDistribution.
 * @param logits (Tensor*)
 * @return (CMLDistribution)
 */
CMLDistribution cml_dist_categorical(Tensor* logits) {
    CMLDistribution d = {CML_DIST_CATEGORICAL, logits, NULL};
    return d;
}

/**
 * Wrap a Gamma(concentration, rate) as a CMLDistribution.
 * @param concentration (Tensor*)
 * @param rate (Tensor*)
 * @return (CMLDistribution)
 */
CMLDistribution cml_dist_gamma(Tensor* concentration, Tensor* rate) {
    CMLDistribution d = {CML_DIST_GAMMA, concentration, rate};
    return d;
}

/**
 * Wrap a Beta(c1, c0) as a CMLDistribution.
 * @param c1 (Tensor*)
 * @param c0 (Tensor*)
 * @return (CMLDistribution)
 */
CMLDistribution cml_dist_beta(Tensor* c1, Tensor* c0) {
    CMLDistribution d = {CML_DIST_BETA, c1, c0};
    return d;
}

/**
 * Wrap an Exponential(rate) as a CMLDistribution.
 * @param rate (Tensor*)
 * @return (CMLDistribution)
 */
CMLDistribution cml_dist_exponential(Tensor* rate) {
    CMLDistribution d = {CML_DIST_EXPONENTIAL, rate, NULL};
    return d;
}

/**
 * Wrap a Uniform(low, high) as a CMLDistribution.
 * @param low (Tensor*)
 * @param high (Tensor*)
 * @return (CMLDistribution)
 */
CMLDistribution cml_dist_uniform(Tensor* low, Tensor* high) {
    CMLDistribution d = {CML_DIST_UNIFORM, low, high};
    return d;
}

/**
 * Wrap a Dirichlet(concentration) as a CMLDistribution.
 * @param concentration (Tensor*)
 * @return (CMLDistribution)
 */
CMLDistribution cml_dist_dirichlet(Tensor* concentration) {
    CMLDistribution d = {CML_DIST_DIRICHLET, concentration, NULL};
    return d;
}

/**
 * KL(N(m1,s1) || N(m2,s2)).
 * @param m1 (Tensor*)
 * @param s1 (Tensor*)
 * @param m2 (Tensor*)
 * @param s2 (Tensor*)
 * @return (Tensor*)
 */
static Tensor* kl_normal(Tensor* m1, Tensor* s1, Tensor* m2, Tensor* s2) {
    Tensor* dm      = cml_sub(m1, m2);
    Tensor* var2    = cml_mul(s2, s2);
    Tensor* ratio   = cml_div(cml_add(cml_mul(s1, s1), cml_mul(dm, dm)), mul_c(var2, 2.0f));
    Tensor* log_std = cml_sub(cml_log(s2), cml_log(s1));
    return add_c(cml_add(log_std, ratio), -0.5f);
}

/**
 * KL(Bernoulli(p) || Bernoulli(q)).
 * @param p (Tensor*)
 * @param q (Tensor*)
 * @return (Tensor*)
 */
static Tensor* kl_bernoulli(Tensor* p, Tensor* q) {
    Tensor* a = cml_mul(p, cml_sub(cml_log(p), cml_log(q)));
    Tensor* b = cml_mul(one_minus(p), cml_sub(cml_log(one_minus(p)), cml_log(one_minus(q))));
    return cml_add(a, b);
}

/**
 * KL(Categorical(lp) || Categorical(lq)) over the last axis of logits.
 * @param lp (Tensor*)
 * @param lq (Tensor*)
 * @return (Tensor*)
 */
static Tensor* kl_categorical(Tensor* lp, Tensor* lq) {
    Tensor* p    = cml_softmax(lp, -1);
    Tensor* logp = cml_log(p);
    Tensor* logq = cml_log(cml_softmax(lq, -1));
    return cml_sum(cml_mul(p, cml_sub(logp, logq)), -1, false);
}

/**
 * KL(Gamma(a1,b1) || Gamma(a2,b2)).
 * @param a1 (Tensor*)
 * @param b1 (Tensor*)
 * @param a2 (Tensor*)
 * @param b2 (Tensor*)
 * @return (Tensor*)
 */
static Tensor* kl_gamma(Tensor* a1, Tensor* b1, Tensor* a2, Tensor* b2) {
    Tensor* t = cml_mul(cml_sub(a1, a2), cml_digamma(a1));
    t         = cml_sub(t, cml_lgamma(a1));
    t         = cml_add(t, cml_lgamma(a2));
    t         = cml_add(t, cml_mul(a2, cml_sub(cml_log(b1), cml_log(b2))));
    return cml_add(t, cml_mul(a1, add_c(cml_div(b2, b1), -1.0f)));
}

/**
 * KL(Beta(a1,b1) || Beta(a2,b2)).
 * @param a1 (Tensor*)
 * @param b1 (Tensor*)
 * @param a2 (Tensor*)
 * @param b2 (Tensor*)
 * @return (Tensor*)
 */
static Tensor* kl_beta(Tensor* a1, Tensor* b1, Tensor* a2, Tensor* b2) {
    Tensor* s1 = cml_add(a1, b1);
    Tensor* t  = cml_sub(lbeta(a2, b2), lbeta(a1, b1));
    t          = cml_add(t, cml_mul(cml_sub(a1, a2), cml_digamma(a1)));
    t          = cml_add(t, cml_mul(cml_sub(b1, b2), cml_digamma(b1)));
    Tensor* cc = cml_add(cml_sub(a2, a1), cml_sub(b2, b1));
    return cml_add(t, cml_mul(cc, cml_digamma(s1)));
}

/**
 * KL(Exponential(l1) || Exponential(l2)).
 * @param l1 (Tensor*)
 * @param l2 (Tensor*)
 * @return (Tensor*)
 */
static Tensor* kl_exponential(Tensor* l1, Tensor* l2) {
    return add_c(cml_add(cml_sub(cml_log(l1), cml_log(l2)), cml_div(l2, l1)), -1.0f);
}

/**
 * KL(Uniform(lo1,hi1) || Uniform(lo2,hi2)); q's support must contain p's, else inf.
 * @param lo1 (Tensor*)
 * @param hi1 (Tensor*)
 * @param lo2 (Tensor*)
 * @param hi2 (Tensor*)
 * @return (Tensor*)
 */
static Tensor* kl_uniform(Tensor* lo1, Tensor* hi1, Tensor* lo2, Tensor* hi2) {
    return cml_sub(cml_log(cml_sub(hi2, lo2)), cml_log(cml_sub(hi1, lo1)));
}

/**
 * KL(Dirichlet(a) || Dirichlet(b)) over the last axis.
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
static Tensor* kl_dirichlet(Tensor* a, Tensor* b) {
    Tensor* a0k  = cml_sum(a, -1, true);
    Tensor* t1   = cml_sub(cml_lgamma(cml_sum(a, -1, false)), cml_lgamma(cml_sum(b, -1, false)));
    Tensor* t2   = cml_sum(cml_sub(cml_lgamma(a), cml_lgamma(b)), -1, false);
    Tensor* diff = cml_mul(cml_sub(a, b), cml_sub(cml_digamma(a), cml_digamma(a0k)));
    Tensor* t3   = cml_sum(diff, -1, false);
    return cml_add(cml_sub(t1, t2), t3);
}

/**
 * Analytic KL(p || q) for matching-type pairs, else NULL.
 * @param p (CMLDistribution)
 * @param q (CMLDistribution)
 * @return (Tensor*)
 */
Tensor* cml_kl_divergence(CMLDistribution p, CMLDistribution q) {
    if (p.type != q.type)
        return NULL;
    switch (p.type) {
        case CML_DIST_NORMAL:
            return kl_normal(p.p1, p.p2, q.p1, q.p2);
        case CML_DIST_BERNOULLI:
            return kl_bernoulli(p.p1, q.p1);
        case CML_DIST_CATEGORICAL:
            return kl_categorical(p.p1, q.p1);
        case CML_DIST_GAMMA:
            return kl_gamma(p.p1, p.p2, q.p1, q.p2);
        case CML_DIST_BETA:
            return kl_beta(p.p1, p.p2, q.p1, q.p2);
        case CML_DIST_EXPONENTIAL:
            return kl_exponential(p.p1, q.p1);
        case CML_DIST_UNIFORM:
            return kl_uniform(p.p1, p.p2, q.p1, q.p2);
        case CML_DIST_DIRICHLET:
            return kl_dirichlet(p.p1, q.p1);
        default:
            return NULL;
    }
}
