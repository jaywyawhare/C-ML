/* Classification and ranking metrics (log loss, balanced accuracy, Jaccard, Hamming, NDCG),
 * checked against sklearn.metrics. */
#include <math.h>
#include <stdio.h>
#include "core/metrics.h"
#include "test_harness.h"

static int test_log_loss(void) {
    int yt[]    = {0, 1, 2, 2};
    float p[12] = {0.8f, 0.1f, 0.1f, 0.2f, 0.7f, 0.1f, 0.1f, 0.2f, 0.7f, 0.3f, 0.3f, 0.4f};
    return fabsf(cml_log_loss(yt, p, 4, 3) - 0.46319604f) < 1e-5f; /* sklearn log_loss */
}

static int test_balanced_accuracy(void) {
    int yt[] = {0, 0, 1, 1, 2, 2};
    int yp[] = {0, 0, 1, 2, 2, 2};
    return fabsf(cml_balanced_accuracy(yt, yp, 6, 3) - 0.8333333f) < 1e-5f;
}

static int test_jaccard(void) {
    int yt[] = {0, 0, 1, 1, 2, 2};
    int yp[] = {0, 0, 1, 2, 2, 2};
    return fabsf(cml_jaccard_score(yt, yp, 6, 3) - 0.7222222f) < 1e-5f; /* macro */
}

static int test_hamming(void) {
    int yt[] = {0, 0, 1, 1, 2, 2};
    int yp[] = {0, 0, 1, 2, 2, 2};
    return fabsf(cml_hamming_loss(yt, yp, 6) - 0.16666667f) < 1e-5f;
}

static int test_ndcg(void) {
    float sc[]  = {3, 2, 1, 0};
    float rel[] = {3, 2, 3, 0};
    return fabsf(cml_ndcg(sc, rel, 4, 0) - 0.97778136f) < 1e-5f && /* full */
           fabsf(cml_ndcg(sc, rel, 4, 2) - 0.87104906f) < 1e-5f;   /* @2 */
}

int main(void) {
    printf("=== classification & ranking metrics ===\n");
    TEST(log_loss);
    TEST(balanced_accuracy);
    TEST(jaccard);
    TEST(hamming);
    TEST(ndcg);
    return TEST_SUMMARY();
}
