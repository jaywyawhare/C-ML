/* CTC loss: forward loss checked against hand-enumerated path probabilities, and the analytic
 * gradient checked against central finite differences. */
#include <math.h>
#include <stdio.h>
#include "core/ctc.h"
#include "test_harness.h"

static int test_loss_empty_target(void) {
    /* T=1, C=2, blank=0, empty target -> loss = -log P(blank). */
    float lp[2] = {logf(0.6f), logf(0.4f)};
    float loss  = cml_ctc_loss(lp, 1, 2, NULL, 0, 0, NULL);
    return fabsf(loss - (-logf(0.6f))) < 1e-5f;
}

static int test_loss_known(void) {
    /* T=2, C=2, blank=0, target=[1], uniform probs 0.5 -> P = 0.75, loss = -ln 0.75. */
    float lp[4] = {logf(0.5f), logf(0.5f), logf(0.5f), logf(0.5f)};
    int tgt[1]  = {1};
    float loss  = cml_ctc_loss(lp, 2, 2, tgt, 1, 0, NULL);
    return fabsf(loss - (-logf(0.75f))) < 1e-5f;
}

static int grad_matches_fd(int T, int C, const int* tgt, int tlen) {
    float lp[64], grad[64];
    for (int i = 0; i < T * C; i++)
        lp[i] = -0.3f * ((i % 3) + 1); /* arbitrary log-probs */
    cml_ctc_loss(lp, T, C, tgt, tlen, 0, grad);
    float eps = 1e-3f;
    int ok    = 1;
    for (int i = 0; ok && i < T * C; i++) {
        float save     = lp[i];
        lp[i]          = save + eps;
        float lp_plus  = cml_ctc_loss(lp, T, C, tgt, tlen, 0, NULL);
        lp[i]          = save - eps;
        float lp_minus = cml_ctc_loss(lp, T, C, tgt, tlen, 0, NULL);
        lp[i]          = save;
        float fd       = (lp_plus - lp_minus) / (2.0f * eps);
        ok &= fabsf(fd - grad[i]) < 2e-2f;
    }
    return ok;
}

static int test_grad_no_repeat(void) {
    int tgt[2] = {1, 2};
    return grad_matches_fd(4, 3, tgt, 2);
}

static int test_grad_with_repeat(void) {
    int tgt[2] = {1, 1}; /* repeated label exercises the skip condition */
    return grad_matches_fd(4, 3, tgt, 2);
}

static int test_rejects_bad(void) {
    float lp[2] = {0.0f, 0.0f};
    return isinf(cml_ctc_loss(lp, 1, 2, NULL, 0, 5, NULL)); /* blank out of range */
}

int main(void) {
    printf("=== CTC loss ===\n");
    TEST(loss_empty_target);
    TEST(loss_known);
    TEST(grad_no_repeat);
    TEST(grad_with_repeat);
    TEST(rejects_bad);
    return TEST_SUMMARY();
}
