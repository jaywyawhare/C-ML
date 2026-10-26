/* Regression metrics round 2 (max error, median absolute error, MSLE, explained variance),
 * checked against sklearn.metrics. */
#include <math.h>
#include <stdio.h>
#include "core/metrics.h"
#include "test_harness.h"

static int test_max_median(void) {
    float yt[] = {3, 5, 2.5f, 7}, yp[] = {2.5f, 5, 4, 8};
    return fabsf(cml_max_error(yt, yp, 4) - 1.5f) < 1e-6f &&
           fabsf(cml_median_absolute_error(yt, yp, 4) - 0.75f) < 1e-6f;
}

static int test_msle(void) {
    float a[] = {1, 2, 3, 4}, b[] = {1.2f, 1.8f, 3.5f, 3.9f};
    return fabsf(cml_mean_squared_log_error(a, b, 4) - 0.00703126f) < 1e-6f;
}

static int test_evs(void) {
    float yt[] = {3, 5, 2.5f, 7}, yp[] = {2.5f, 5, 4, 8};
    return fabsf(cml_explained_variance_score(yt, yp, 4) - 0.80295567f) < 1e-5f;
}

int main(void) {
    printf("=== regression metrics (round 2) ===\n");
    TEST(max_median);
    TEST(msle);
    TEST(evs);
    return TEST_SUMMARY();
}
