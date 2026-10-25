/* NLP / detection / top-k metrics, verified against hand-computed values:
 * top-k accuracy, perplexity, IoU, average precision, and sentence BLEU. */
#include <math.h>
#include <stdio.h>
#include "core/metrics.h"
#include "test_harness.h"

#define NEAR(a, b) (fabsf((float)(a) - (float)(b)) < 1e-4f)

static int test_top_k(void) {
    /* 2 samples, 3 classes. */
    float logits[6] = {0.1f, 0.9f, 0.2f,  /* argmax=1, 2nd=2 */
                       0.5f, 0.4f, 0.3f}; /* argmax=0, 2nd=1 */
    int yt[2]       = {2, 1};
    /* top-1: sample0 true=2 (not argmax) miss; sample1 true=1 (not argmax) miss -> 0 */
    int ok = NEAR(cml_top_k_accuracy(logits, 2, 3, yt, 1), 0.0f);
    /* top-2: sample0 true=2 is 2nd -> hit; sample1 true=1 is 2nd -> hit -> 1.0 */
    ok &= NEAR(cml_top_k_accuracy(logits, 2, 3, yt, 2), 1.0f);
    return ok;
}

static int test_perplexity(void) {
    float nll[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
    int ok        = NEAR(cml_perplexity(nll, 4), 1.0f); /* exp(0)=1 */
    float nll2[2] = {1.0f, 3.0f};                       /* mean 2 -> exp(2) */
    ok &= NEAR(cml_perplexity(nll2, 2), expf(2.0f));
    return ok;
}

static int test_iou(void) {
    float a[4] = {0, 0, 2, 2}; /* area 4 */
    float b[4] = {1, 1, 3, 3}; /* area 4; overlap [1,1,2,2]=1 */
    /* IoU = 1 / (4+4-1) = 1/7 */
    int ok = NEAR(cml_iou(a, b), 1.0f / 7.0f);
    /* identical -> 1 */
    ok &= NEAR(cml_iou(a, a), 1.0f);
    /* disjoint -> 0 */
    float c[4] = {5, 5, 6, 6};
    ok &= NEAR(cml_iou(a, c), 0.0f);
    return ok;
}

static int test_average_precision(void) {
    /* Perfect ranking (all positives above negatives) -> AP = 1. */
    int yt[4]   = {0, 0, 1, 1};
    float ys[4] = {0.1f, 0.2f, 0.8f, 0.9f};
    int ok      = NEAR(cml_average_precision(yt, ys, 4), 1.0f);
    /* Known case: scores 0.9,0.8,0.7,0.6 labels 1,0,1,1 ->
     * ranks: tp,fp,prec,recall: (1,0,1,1/3)+(1,1,.5,1/3)+(2,1,2/3,2/3)+(3,1,3/4,1)
     * AP = 1*(1/3) + .5*0 + (2/3)*(1/3) + (3/4)*(1/3) = .3333+.2222+.25 = 0.80556 */
    int yt2[4]   = {1, 0, 1, 1};
    float ys2[4] = {0.9f, 0.8f, 0.7f, 0.6f};
    ok &= NEAR(cml_average_precision(yt2, ys2, 4), 0.8055556f);
    return ok;
}

static int test_bleu(void) {
    /* Identical candidate/reference -> BLEU 1. */
    int ref[5]  = {1, 2, 3, 4, 5};
    int cand[5] = {1, 2, 3, 4, 5};
    int ok      = NEAR(cml_bleu(cand, 5, ref, 5, 4), 1.0f);
    /* No overlap -> 0. */
    int cand2[5] = {6, 7, 8, 9, 10};
    ok &= NEAR(cml_bleu(cand2, 5, ref, 5, 4), 0.0f);
    /* Partial unigram-only (max_n=1): cand {1,2,9} vs ref {1,2,3,4,5}: 2/3 match, BP=1
     * (cand_len 3 < ref_len 5 -> BP=exp(1-5/3)). BLEU = exp(1-5/3) * (2/3). */
    int cand3[3] = {1, 2, 9};
    float bp     = expf(1.0f - 5.0f / 3.0f);
    ok &= NEAR(cml_bleu(cand3, 3, ref, 5, 1), bp * (2.0f / 3.0f));
    return ok;
}

int main(void) {
    printf("=== NLP / detection / top-k metrics ===\n");
    TEST(top_k);
    TEST(perplexity);
    TEST(iou);
    TEST(average_precision);
    TEST(bleu);
    return TEST_SUMMARY();
}
