/* CTC loss via the log-space forward-backward algorithm. Alpha and beta both include the
 * per-frame emission, so the per-frame class posterior used for the gradient is
 * (1 / (p * y)) * sum_s alpha_t(s) beta_t(s) over extended-label positions emitting that class. */
#include "core/ctc.h"
#include <math.h>
#include "alloc/cml_allocator.h"
#include "core/logging.h"

/** Numerically stable log(exp(a) + exp(b)).
 * @param a (float)
 * @param b (float)
 * @return (float)
 */
static float logaddexp(float a, float b) {
    if (a == -INFINITY)
        return b;
    if (b == -INFINITY)
        return a;
    float m = a > b ? a : b;
    return m + log1pf(expf(-fabsf(a - b)));
}

float cml_ctc_loss(const float* log_probs, int T, int C, const int* targets, int target_len,
                   int blank, float* grad) {
    if (!log_probs || T <= 0 || C <= 0 || target_len < 0 || blank < 0 || blank >= C) {
        LOG_ERROR("cml_ctc_loss: bad arguments");
        return INFINITY;
    }
    int S     = 2 * target_len + 1;
    int* l    = cml_malloc((size_t)S * sizeof(int)); /* extended label: blanks between targets */
    float* la = cml_malloc((size_t)T * S * sizeof(float));
    float* lb = cml_malloc((size_t)T * S * sizeof(float));
    if (!l || !la || !lb) {
        cml_free(l);
        cml_free(la);
        cml_free(lb);
        return INFINITY;
    }
    for (int i = 0; i < S; i++)
        l[i] = (i % 2 == 0) ? blank : targets[i / 2];
    for (int i = 0; i < T * S; i++) {
        la[i] = -INFINITY;
        lb[i] = -INFINITY;
    }
#define LP(t, k) log_probs[(size_t)(t) * C + (k)]

    la[0] = LP(0, l[0]);
    if (S > 1)
        la[1] = LP(0, l[1]);
    for (int t = 1; t < T; t++) {
        for (int s = 0; s < S; s++) {
            float a = la[(size_t)(t - 1) * S + s];
            if (s >= 1)
                a = logaddexp(a, la[(size_t)(t - 1) * S + s - 1]);
            if (s >= 2 && l[s] != blank && l[s] != l[s - 2])
                a = logaddexp(a, la[(size_t)(t - 1) * S + s - 2]);
            la[(size_t)t * S + s] = a + LP(t, l[s]);
        }
    }
    float logP = logaddexp(la[(size_t)(T - 1) * S + S - 1],
                           S > 1 ? la[(size_t)(T - 1) * S + S - 2] : -INFINITY);
    if (logP == -INFINITY) {
        cml_free(l);
        cml_free(la);
        cml_free(lb);
        return INFINITY;
    }
    if (grad) {
        lb[(size_t)(T - 1) * S + S - 1] = LP(T - 1, l[S - 1]);
        if (S > 1)
            lb[(size_t)(T - 1) * S + S - 2] = LP(T - 1, l[S - 2]);
        for (int t = T - 2; t >= 0; t--) {
            for (int s = 0; s < S; s++) {
                float b = lb[(size_t)(t + 1) * S + s];
                if (s + 1 < S)
                    b = logaddexp(b, lb[(size_t)(t + 1) * S + s + 1]);
                if (s + 2 < S && l[s] != blank && l[s] != l[s + 2])
                    b = logaddexp(b, lb[(size_t)(t + 1) * S + s + 2]);
                lb[(size_t)t * S + s] = b + LP(t, l[s]);
            }
        }
        for (int i = 0; i < T * C; i++)
            grad[i] = 0.0f;
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < C; k++) {
                float acc = -INFINITY;
                for (int s = 0; s < S; s++)
                    if (l[s] == k)
                        acc = logaddexp(acc, la[(size_t)t * S + s] + lb[(size_t)t * S + s]);
                if (acc != -INFINITY)
                    grad[(size_t)t * C + k] = -expf(acc - logP - LP(t, k));
            }
        }
    }
#undef LP
    cml_free(l);
    cml_free(la);
    cml_free(lb);
    return -logP;
}
