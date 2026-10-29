/*
 * Distributions (Normal, Bernoulli, Categorical, Gumbel-Softmax). Closed-form
 * log_prob / entropy are checked against numpy oracles; rsample/sample are
 * checked by empirical mean, variance and class frequency; a deterministic
 * finite difference confirms gradients flow through Normal log_prob.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cml.h"
#include "core/threefry.h"
#include "nn/distributions.h"
#include "test_harness.h"

static TensorConfig cfg = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static Tensor* mk(const float* data, int* shape, int ndim, bool grad) {
    Tensor* t = tensor_zeros(shape, ndim, &cfg);
    memcpy(tensor_data_ptr(t), data, tensor_numel(shape, ndim) * sizeof(float));
    t->requires_grad = grad;
    return t;
}

static int close2(const float* got, const float* want, int n, float tol) {
    for (int i = 0; i < n; i++)
        if (fabsf(got[i] - want[i]) > tol)
            return 0;
    return 1;
}

static int test_normal_log_prob(void) {
    float loc[2] = {0, 1}, sc[2] = {1, 2}, x[2] = {0.5f, 2.0f};
    int sh[1] = {2};
    cml_reset_ir_context();
    Tensor* lp =
        cml_normal_log_prob(mk(loc, sh, 1, false), mk(sc, sh, 1, false), mk(x, sh, 1, false));
    tensor_ensure_executed(lp);
    float want[2] = {-1.043939f, -1.737086f};
    int ok        = close2((float*)tensor_data_ptr(lp), want, 2, 1e-4f);
    cml_reset_ir_context();
    return ok;
}

static int test_normal_entropy(void) {
    float sc[2] = {1, 2};
    int sh[1]   = {2};
    cml_reset_ir_context();
    Tensor* e = cml_normal_entropy(mk(sc, sh, 1, false));
    tensor_ensure_executed(e);
    float want[2] = {1.418939f, 2.112086f};
    int ok        = close2((float*)tensor_data_ptr(e), want, 2, 1e-4f);
    cml_reset_ir_context();
    return ok;
}

static int test_normal_rsample_moments(void) {
    const int M = 20000;
    cml_reset_ir_context();
    cml_rng_set_global_seed(11);
    int sh[2]   = {M, 2};
    Tensor* loc = tensor_zeros(sh, 2, &cfg);
    Tensor* sc  = tensor_zeros(sh, 2, &cfg);
    float* ld   = (float*)tensor_data_ptr(loc);
    float* sd   = (float*)tensor_data_ptr(sc);
    for (int i = 0; i < M; i++) {
        ld[2 * i]     = 0.0f;
        ld[2 * i + 1] = 1.0f;
        sd[2 * i]     = 1.0f;
        sd[2 * i + 1] = 2.0f;
    }
    Tensor* s = cml_normal_rsample(loc, sc);
    tensor_ensure_executed(s);
    float* v       = (float*)tensor_data_ptr(s);
    double mean[2] = {0, 0}, var[2] = {0, 0};
    for (int i = 0; i < M; i++)
        for (int c = 0; c < 2; c++)
            mean[c] += v[2 * i + c];
    mean[0] /= M;
    mean[1] /= M;
    for (int i = 0; i < M; i++)
        for (int c = 0; c < 2; c++) {
            double d = v[2 * i + c] - mean[c];
            var[c] += d * d;
        }
    var[0] /= M;
    var[1] /= M;
    int ok = fabs(mean[0] - 0.0) < 0.05 && fabs(mean[1] - 1.0) < 0.08 && fabs(var[0] - 1.0) < 0.1 &&
             fabs(var[1] - 4.0) < 0.3;
    cml_reset_ir_context();
    return ok;
}

static float normal_lp0(const float* loc, const float* sc, const float* x) {
    int sh[1] = {2};
    cml_reset_ir_context();
    Tensor* lp =
        cml_normal_log_prob(mk(loc, sh, 1, false), mk(sc, sh, 1, false), mk(x, sh, 1, false));
    Tensor* s = cml_sum(lp, -1, false);
    tensor_ensure_executed(s);
    float r = ((float*)tensor_data_ptr(s))[0];
    cml_reset_ir_context();
    return r;
}

static int test_normal_log_prob_grad(void) {
    float loc[2] = {0.2f, 0.8f}, sc[2] = {1.0f, 1.5f}, x[2] = {0.5f, 2.0f};
    int sh[1] = {2};
    cml_reset_ir_context();
    Tensor* tl   = mk(loc, sh, 1, true);
    Tensor* lp   = cml_normal_log_prob(tl, mk(sc, sh, 1, false), mk(x, sh, 1, false));
    Tensor* loss = cml_sum(lp, -1, false);
    tensor_backward(loss, NULL, false, false);
    if (!tl->grad) {
        cml_reset_ir_context();
        return 0;
    }
    tensor_ensure_executed(tl->grad);
    float ana[2];
    memcpy(ana, tensor_data_ptr(tl->grad), sizeof(ana));
    cml_reset_ir_context();

    const float eps = 1e-3f;
    for (int i = 0; i < 2; i++) {
        float sv  = loc[i];
        loc[i]    = sv + eps;
        float fp  = normal_lp0(loc, sc, x);
        loc[i]    = sv - eps;
        float fm  = normal_lp0(loc, sc, x);
        loc[i]    = sv;
        float num = (fp - fm) / (2 * eps);
        if (fabsf(num - ana[i]) > 1e-2f * fmaxf(1.0f, fabsf(num)))
            return 0;
    }
    return 1;
}

static int test_bernoulli_log_prob_entropy(void) {
    float p[2] = {0.3f, 0.8f}, v[2] = {1, 0};
    int sh[1] = {2};
    cml_reset_ir_context();
    Tensor* lp = cml_bernoulli_log_prob(mk(p, sh, 1, false), mk(v, sh, 1, false));
    Tensor* e  = cml_bernoulli_entropy(mk(p, sh, 1, false));
    tensor_ensure_executed(lp);
    tensor_ensure_executed(e);
    float wlp[2] = {-1.203973f, -1.609438f};
    float we[2]  = {0.610864f, 0.500402f};
    int ok       = close2((float*)tensor_data_ptr(lp), wlp, 2, 1e-4f) &&
             close2((float*)tensor_data_ptr(e), we, 2, 1e-4f);
    cml_reset_ir_context();
    return ok;
}

static int test_bernoulli_sample_mean(void) {
    const int M = 20000;
    cml_reset_ir_context();
    cml_rng_set_global_seed(7);
    int sh[2] = {M, 2};
    Tensor* p = tensor_zeros(sh, 2, &cfg);
    float* pd = (float*)tensor_data_ptr(p);
    for (int i = 0; i < M; i++) {
        pd[2 * i]     = 0.3f;
        pd[2 * i + 1] = 0.8f;
    }
    Tensor* s = cml_bernoulli_sample(p);
    tensor_ensure_executed(s);
    float* v    = (float*)tensor_data_ptr(s);
    double m[2] = {0, 0};
    for (int i = 0; i < M; i++) {
        m[0] += v[2 * i];
        m[1] += v[2 * i + 1];
    }
    m[0] /= M;
    m[1] /= M;
    int ok = fabs(m[0] - 0.3) < 0.03 && fabs(m[1] - 0.8) < 0.03;
    cml_reset_ir_context();
    return ok;
}

static int test_categorical_log_prob_entropy(void) {
    float lg[3] = {1, 2, 3};
    float id[1] = {2};
    int lsh[2]  = {1, 3};
    int ish[1]  = {1};
    cml_reset_ir_context();
    Tensor* lp = cml_categorical_log_prob(mk(lg, lsh, 2, false), mk(id, ish, 1, false));
    Tensor* e  = cml_categorical_entropy(mk(lg, lsh, 2, false));
    tensor_ensure_executed(lp);
    tensor_ensure_executed(e);
    float wlp = -0.407606f, we = 0.832396f;
    int ok = fabsf(((float*)tensor_data_ptr(lp))[0] - wlp) < 1e-4f &&
             fabsf(((float*)tensor_data_ptr(e))[0] - we) < 1e-4f;
    cml_reset_ir_context();
    return ok;
}

static int test_categorical_sample_freq(void) {
    const int M = 20000;
    cml_reset_ir_context();
    cml_rng_set_global_seed(3);
    float lg[3] = {1, 2, 3};
    int lsh[2]  = {1, 3};
    Tensor* idx = cml_categorical_sample(mk(lg, lsh, 2, false), M);
    tensor_ensure_executed(idx);
    int32_t* id = (int32_t*)tensor_data_ptr(idx); /* multinomial returns int32 indices */
    int cnt[3]  = {0, 0, 0};
    for (int i = 0; i < M; i++) {
        int c = id[i];
        if (c >= 0 && c < 3)
            cnt[c]++;
    }
    float want[3] = {0.090031f, 0.244728f, 0.665241f};
    int ok        = 1;
    for (int c = 0; c < 3; c++)
        if (fabsf((float)cnt[c] / M - want[c]) > 0.03f)
            ok = 0;
    cml_reset_ir_context();
    return ok;
}

static int test_gumbel_softmax(void) {
    const int M = 20000;
    cml_reset_ir_context();
    cml_rng_set_global_seed(5);
    int lsh[2] = {M, 3};
    Tensor* lg = tensor_zeros(lsh, 2, &cfg);
    float* d   = (float*)tensor_data_ptr(lg);
    for (int i = 0; i < M; i++) {
        d[3 * i]     = 1;
        d[3 * i + 1] = 2;
        d[3 * i + 2] = 3;
    }
    Tensor* y = cml_gumbel_softmax(lg, 0.5f, true);
    tensor_ensure_executed(y);
    float* yv   = (float*)tensor_data_ptr(y);
    int cnt[3]  = {0, 0, 0};
    int rows_ok = 1;
    for (int i = 0; i < M; i++) {
        float s = yv[3 * i] + yv[3 * i + 1] + yv[3 * i + 2];
        if (fabsf(s - 1.0f) > 1e-3f)
            rows_ok = 0;
        int am = 0;
        for (int c = 1; c < 3; c++)
            if (yv[3 * i + c] > yv[3 * i + am])
                am = c;
        cnt[am]++;
    }
    float want[3] = {0.090031f, 0.244728f, 0.665241f};
    int ok        = rows_ok;
    for (int c = 0; c < 3; c++)
        if (fabsf((float)cnt[c] / M - want[c]) > 0.04f)
            ok = 0;
    cml_reset_ir_context();
    return ok;
}

static int test_gumbel_softmax_grad(void) {
    cml_reset_ir_context();
    cml_rng_set_global_seed(9);
    float lg[3]  = {0.5f, 1.0f, -0.3f};
    int lsh[2]   = {1, 3};
    Tensor* t    = mk(lg, lsh, 2, true);
    Tensor* y    = cml_gumbel_softmax(t, 1.0f, false);
    Tensor* loss = cml_sum(cml_mul(y, y), -1, false);
    tensor_backward(loss, NULL, false, false);
    int ok = t->grad != NULL;
    if (ok) {
        tensor_ensure_executed(t->grad);
        float* g = (float*)tensor_data_ptr(t->grad);
        for (int i = 0; i < 3; i++)
            if (!isfinite(g[i]))
                ok = 0;
    }
    cml_reset_ir_context();
    return ok;
}

int main(void) {
    printf("test_distributions\n\n");
    TEST(normal_log_prob);
    TEST(normal_entropy);
    TEST(normal_rsample_moments);
    TEST(normal_log_prob_grad);
    TEST(bernoulli_log_prob_entropy);
    TEST(bernoulli_sample_mean);
    TEST(categorical_log_prob_entropy);
    TEST(categorical_sample_freq);
    TEST(gumbel_softmax);
    TEST(gumbel_softmax_grad);
    cml_reset_ir_context();
    return TEST_SUMMARY();
}
