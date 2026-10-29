#include "nn/distributions.h"

#include <math.h>

#include "cml.h"

#define LOG_2PI 1.8378770664093453f       /* log(2*pi) */
#define HALF_LOG_2PIE 1.4189385332046727f /* 0.5*log(2*pi*e) */

/**
 * Elementwise add of a scalar constant (autograd-safe: constant has no grad).
 * @param t (Tensor*)
 * @param c (float)
 * @return (Tensor*)
 */
static Tensor* add_c(Tensor* t, float c) { return cml_add(t, cml_full_like(t, c)); }

/**
 * Elementwise multiply by a scalar constant.
 * @param t (Tensor*)
 * @param c (float)
 * @return (Tensor*)
 */
static Tensor* mul_c(Tensor* t, float c) { return cml_mul(t, cml_full_like(t, c)); }

/** 1 - t, as a differentiable expression. */
static Tensor* one_minus(Tensor* t) { return add_c(mul_c(t, -1.0f), 1.0f); }

static TensorConfig cfg_of(Tensor* t) {
    TensorConfig c = {
        .dtype = DTYPE_FLOAT32, .device = t->device, .has_dtype = true, .has_device = true};
    return c;
}

Tensor* cml_normal_rsample(Tensor* loc, Tensor* scale) {
    if (!loc || !scale)
        return NULL;
    TensorConfig cfg = cfg_of(loc);
    Tensor* eps      = tensor_randn(loc->shape, loc->ndim, &cfg); /* detached noise */
    if (!eps)
        return NULL;
    return cml_add(loc, cml_mul(scale, eps));
}

Tensor* cml_normal_log_prob(Tensor* loc, Tensor* scale, Tensor* value) {
    if (!loc || !scale || !value)
        return NULL;
    Tensor* z    = cml_div(cml_sub(value, loc), scale); /* standardized */
    Tensor* quad = mul_c(cml_mul(z, z), -0.5f);
    return add_c(cml_sub(quad, cml_log(scale)), -0.5f * LOG_2PI);
}

Tensor* cml_normal_entropy(Tensor* scale) {
    if (!scale)
        return NULL;
    return add_c(cml_log(scale), HALF_LOG_2PIE);
}

Tensor* cml_bernoulli_sample(Tensor* probs) {
    if (!probs || tensor_ensure_executed(probs) != 0)
        return NULL;
    const float* p   = (const float*)tensor_data_ptr(probs);
    TensorConfig cfg = cfg_of(probs);
    Tensor* u        = tensor_rand(probs->shape, probs->ndim, &cfg);
    Tensor* out      = cml_zeros(probs->shape, probs->ndim, &cfg);
    if (!u || !out)
        return NULL;
    const float* ud = (const float*)tensor_data_ptr(u);
    float* od       = (float*)tensor_data_ptr(out);
    size_t n        = tensor_numel(probs->shape, probs->ndim);
    for (size_t i = 0; i < n; i++)
        od[i] = (ud[i] < p[i]) ? 1.0f : 0.0f;
    return out;
}

Tensor* cml_bernoulli_log_prob(Tensor* probs, Tensor* value) {
    if (!probs || !value)
        return NULL;
    Tensor* a = cml_mul(value, cml_log(probs));
    Tensor* b = cml_mul(one_minus(value), cml_log(one_minus(probs)));
    return cml_add(a, b);
}

Tensor* cml_bernoulli_entropy(Tensor* probs) {
    if (!probs)
        return NULL;
    Tensor* pm = probs;
    Tensor* a  = cml_mul(pm, cml_log(pm));
    Tensor* b  = cml_mul(one_minus(pm), cml_log(one_minus(pm)));
    return mul_c(cml_add(a, b), -1.0f);
}

Tensor* cml_categorical_sample(Tensor* logits, int num_samples) {
    if (!logits || num_samples <= 0)
        return NULL;
    Tensor* probs = cml_softmax(logits, -1);
    return tensor_multinomial(probs, num_samples, true);
}

Tensor* cml_categorical_log_prob(Tensor* logits, Tensor* indices) {
    if (!logits || !indices || logits->ndim != 2)
        return NULL;
    int k           = logits->shape[1];
    Tensor* logp    = cml_log(cml_softmax(logits, -1)); /* [N,K] */
    Tensor* one_hot = cml_one_hot(indices, k);          /* [N,K], detached */
    if (!one_hot)
        return NULL;
    return cml_sum(cml_mul(one_hot, logp), -1, false); /* [N] */
}

Tensor* cml_categorical_entropy(Tensor* logits) {
    if (!logits)
        return NULL;
    Tensor* p    = cml_softmax(logits, -1);
    Tensor* plog = cml_mul(p, cml_log(p));
    return mul_c(cml_sum(plog, -1, false), -1.0f);
}

Tensor* cml_gumbel_softmax(Tensor* logits, float tau, bool hard) {
    if (!logits || tau <= 0.0f)
        return NULL;
    TensorConfig cfg = cfg_of(logits);
    Tensor* u        = tensor_rand(logits->shape, logits->ndim, &cfg);
    if (!u || tensor_ensure_executed(u) != 0)
        return NULL;

    /* Gumbel noise g = -log(-log(u)) built as a detached constant. */
    Tensor* gnoise = cml_zeros(logits->shape, logits->ndim, &cfg);
    if (!gnoise)
        return NULL;
    const float* ud = (const float*)tensor_data_ptr(u);
    float* gd       = (float*)tensor_data_ptr(gnoise);
    size_t n        = tensor_numel(logits->shape, logits->ndim);
    for (size_t i = 0; i < n; i++) {
        float uu = ud[i];
        if (uu < 1e-20f)
            uu = 1e-20f;
        gd[i] = -logf(-logf(uu));
    }

    Tensor* y = cml_softmax(mul_c(cml_add(logits, gnoise), 1.0f / tau), -1);
    if (!hard)
        return y;

    /* Straight-through: forward value is one-hot(argmax(y)), gradient flows through
     * the soft sample. Build the one-hot and the detached soft term as plain
     * grad-free constants to avoid aliasing the lazy graph node y. */
    if (tensor_ensure_executed(y) != 0)
        return NULL;
    const float* yv = (const float*)tensor_data_ptr(y);
    int k           = logits->shape[logits->ndim - 1];
    size_t rows     = tensor_numel(logits->shape, logits->ndim) / (size_t)k;
    Tensor* yh      = cml_zeros(logits->shape, logits->ndim, &cfg);
    Tensor* soft    = cml_zeros(logits->shape, logits->ndim, &cfg);
    if (!yh || !soft)
        return NULL;
    float* yhd = (float*)tensor_data_ptr(yh);
    float* sd  = (float*)tensor_data_ptr(soft);
    for (size_t r = 0; r < rows; r++) {
        int am = 0;
        for (int c = 1; c < k; c++)
            if (yv[r * k + c] > yv[r * k + am])
                am = c;
        for (int c = 0; c < k; c++)
            sd[r * k + c] = yv[r * k + c];
        yhd[r * k + am] = 1.0f;
    }
    return cml_add(yh, cml_sub(y, soft)); /* forward == yh, grad == d/d logits of y */
}
