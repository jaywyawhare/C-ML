/* ROC / PR curves and trapezoidal AUC, checked against sklearn.metrics (roc_curve and
 * precision_recall_curve with drop_intermediate=False, auc). */
#include <math.h>
#include <stdio.h>
#include "alloc/cml_allocator.h"
#include "cml.h"
#include "core/metrics.h"
#include "test_harness.h"

static int arr_eq(const float* a, const float* e, int n) {
    for (int i = 0; i < n; i++)
        if (fabsf(a[i] - e[i]) > 1e-5f)
            return 0;
    return 1;
}

static int test_roc_curve(void) {
    int yt[]   = {0, 0, 1, 1};
    float sc[] = {0.1f, 0.4f, 0.35f, 0.8f};
    float *fpr, *tpr, *thr;
    int np       = cml_roc_curve(yt, sc, 4, &fpr, &tpr, &thr);
    float efpr[] = {0.0f, 0.0f, 0.5f, 0.5f, 1.0f};
    float etpr[] = {0.0f, 0.5f, 0.5f, 1.0f, 1.0f};
    int ok       = np == 5 && isinf(thr[0]) && arr_eq(fpr, efpr, 5) && arr_eq(tpr, etpr, 5) &&
             fabsf(cml_auc(fpr, tpr, 5) - 0.75f) < 1e-5f;
    cml_free(fpr);
    cml_free(tpr);
    cml_free(thr);
    return ok;
}

static int test_pr_curve(void) {
    int yt[]   = {0, 0, 1, 1};
    float sc[] = {0.1f, 0.4f, 0.35f, 0.8f};
    float *pr, *rc, *thr;
    int nt      = cml_precision_recall_curve(yt, sc, 4, &pr, &rc, &thr);
    float epr[] = {0.5f, 0.6666667f, 0.5f, 1.0f, 1.0f};
    float erc[] = {1.0f, 1.0f, 0.5f, 0.5f, 0.0f};
    float eth[] = {0.1f, 0.35f, 0.4f, 0.8f};
    int ok      = nt == 4 && arr_eq(pr, epr, 5) && arr_eq(rc, erc, 5) && arr_eq(thr, eth, 4);
    cml_free(pr);
    cml_free(rc);
    cml_free(thr);
    return ok;
}

static int test_auc(void) {
    float x[] = {0, 1, 2}, y[] = {0, 1, 4};
    return fabsf(cml_auc(x, y, 3) - 3.0f) < 1e-6f;
}

int main(void) {
    cml_init();
    printf("=== ROC / PR curves and AUC ===\n");
    TEST(roc_curve);
    TEST(pr_curve);
    TEST(auc);
    cml_cleanup();
    return TEST_SUMMARY();
}
