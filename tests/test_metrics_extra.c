/* Correlation and calibration metrics (MCC, Cohen's kappa, Pearson, Spearman, ECE),
 * checked against hand-computed values. */
#include <math.h>
#include <stdio.h>
#include "core/metrics.h"
#include "test_harness.h"

static int test_mcc(void) {
    int t[] = {1, 1, 0, 0};
    int p[] = {1, 0, 0, 0}; /* TP=1,FN=1,TN=2,FP=0 -> 2/sqrt(12)=1/sqrt(3) */
    return fabsf(cml_matthews_corrcoef(t, p, 4) - 1.0f / sqrtf(3.0f)) < 1e-5f;
}

static int test_cohen_kappa(void) {
    int t[]    = {0, 0, 1, 1};
    int p[]    = {0, 1, 1, 1}; /* po=0.75, pe=0.5 -> kappa=0.5 */
    int perf[] = {0, 1, 0, 1};
    int ok     = fabsf(cml_cohen_kappa(t, p, 4, 2) - 0.5f) < 1e-5f;
    ok &= fabsf(cml_cohen_kappa(perf, perf, 4, 2) - 1.0f) < 1e-5f;
    return ok;
}

static int test_pearson(void) {
    float x[] = {1, 2, 3};
    float y[] = {2, 4, 6}; /* perfectly positively correlated */
    float z[] = {3, 2, 1}; /* perfectly negatively correlated */
    int ok    = fabsf(cml_pearson(x, y, 3) - 1.0f) < 1e-5f;
    ok &= fabsf(cml_pearson(x, z, 3) + 1.0f) < 1e-5f;
    return ok;
}

static int test_spearman(void) {
    float x[] = {1, 2, 3, 4};
    float y[] = {1, 4, 9, 16}; /* monotonic but nonlinear -> spearman 1.0 */
    return fabsf(cml_spearman(x, y, 4) - 1.0f) < 1e-5f;
}

static int test_ece(void) {
    float conf[]  = {0.9f, 0.9f};
    int correct[] = {1, 0}; /* bin 9: acc=0.5, conf=0.9 -> gap 0.4 */
    return fabsf(cml_expected_calibration_error(conf, correct, 2, 10) - 0.4f) < 1e-5f;
}

int main(void) {
    printf("=== correlation / calibration metrics ===\n");
    TEST(mcc);
    TEST(cohen_kappa);
    TEST(pearson);
    TEST(spearman);
    TEST(ece);
    return TEST_SUMMARY();
}
