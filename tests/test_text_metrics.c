/* Edit distance, WER and ROUGE-N over token-id sequences, checked against hand-computed
 * values. */
#include <math.h>
#include <stdio.h>
#include "core/metrics.h"
#include "test_harness.h"

static int test_edit_distance(void) {
    int a[] = {1, 2, 3, 4};    /* "kitten"-style: 3 edits to "sitting" analogue */
    int b[] = {1, 5, 3, 4, 6}; /* substitute pos2, insert at end -> 2 edits */
    int ok  = cml_edit_distance(a, 4, b, 5) == 2;
    ok &= cml_edit_distance(a, 4, a, 4) == 0;
    ok &= cml_edit_distance(a, 4, NULL, 0) == 4;
    return ok;
}

static int test_wer(void) {
    int ref[] = {1, 2, 3, 4}; /* 4 reference words */
    int hyp[] = {1, 2, 9, 4}; /* one substitution */
    int ok    = fabsf(cml_wer(ref, 4, hyp, 4) - 0.25f) < 1e-6f;
    ok &= fabsf(cml_wer(ref, 4, ref, 4)) < 1e-6f;
    return ok;
}

static int test_rouge_n(void) {
    int cand[] = {1, 2, 3, 4};
    int ref[]  = {1, 2, 5, 4};
    /* unigrams: ref has {1,2,5,4}; cand matches 1,2,4 -> 3/4 */
    int ok = fabsf(cml_rouge_n(cand, 4, ref, 4, 1) - 0.75f) < 1e-6f;
    /* bigrams: ref {12,25,54}; cand {12,23,34} matches 12 -> 1/3 */
    ok &= fabsf(cml_rouge_n(cand, 4, ref, 4, 2) - (1.0f / 3.0f)) < 1e-6f;
    return ok;
}

int main(void) {
    printf("=== text / ASR metrics ===\n");
    TEST(edit_distance);
    TEST(wer);
    TEST(rouge_n);
    return TEST_SUMMARY();
}
