#include "nn/sampling.h"
#include <stdio.h>
#include <math.h>
#include "test_assert.h"

/* Oracle values computed with numpy against HuggingFace LogitsProcessor semantics for
 * logits = [2.0, 1.0, 0.5, 0.0, -1.0]:
 *   probs      ~ [0.563, 0.2071, 0.1256, 0.0762, 0.028]
 *   top_k(2)   -> keep idx {0,1}
 *   top_p(0.9) -> keep idx {0,1,2,3}
 *   min_p(0.2) -> keep idx {0,1,2}
 */

static void load(float* l) {
    l[0] = 2.0f;
    l[1] = 1.0f;
    l[2] = 0.5f;
    l[3] = 0.0f;
    l[4] = -1.0f;
}

static void test_temperature(void) {
    printf("test_temperature...\n");
    float l[5];
    load(l);
    cml_logits_temperature(l, 5, 0.5f); /* divide by 0.5 == *2 */
    float want[5] = {4.0f, 2.0f, 1.0f, 0.0f, -2.0f};
    for (int i = 0; i < 5; i++)
        ASSERT(fabsf(l[i] - want[i]) < 1e-6f, "temperature value");
    tests_passed++;
}

static void test_repetition_penalty(void) {
    printf("test_repetition_penalty...\n");
    float l[5];
    load(l);
    int toks[2] = {0, 4};
    cml_logits_repetition_penalty(l, 5, toks, 2, 2.0f);
    ASSERT(fabsf(l[0] - 1.0f) < 1e-6f, "positive logit divided");       /* 2.0/2 */
    ASSERT(fabsf(l[4] - (-2.0f)) < 1e-6f, "negative logit multiplied"); /* -1.0*2 */
    ASSERT(fabsf(l[1] - 1.0f) < 1e-6f, "untouched logit");
    tests_passed++;
}

static void test_top_k(void) {
    printf("test_top_k...\n");
    float l[5];
    load(l);
    int kept = cml_logits_top_k(l, 5, 2);
    ASSERT(kept == 2, "top_k kept count");
    ASSERT(isfinite(l[0]) && isfinite(l[1]), "top_k keeps top 2");
    ASSERT(!isfinite(l[2]) && !isfinite(l[3]) && !isfinite(l[4]), "top_k masks the rest");
    tests_passed++;
}

static void test_top_p(void) {
    printf("test_top_p...\n");
    float l[5];
    load(l);
    int kept = cml_logits_top_p(l, 5, 0.9f);
    ASSERT(kept == 4, "top_p kept count");
    ASSERT(isfinite(l[0]) && isfinite(l[1]) && isfinite(l[2]) && isfinite(l[3]),
           "top_p keeps {0..3}");
    ASSERT(!isfinite(l[4]), "top_p masks the tail");
    tests_passed++;
}

static void test_min_p(void) {
    printf("test_min_p...\n");
    float l[5];
    load(l);
    int kept = cml_logits_min_p(l, 5, 0.2f);
    ASSERT(kept == 3, "min_p kept count");
    ASSERT(isfinite(l[0]) && isfinite(l[1]) && isfinite(l[2]), "min_p keeps {0,1,2}");
    ASSERT(!isfinite(l[3]) && !isfinite(l[4]), "min_p masks below threshold");
    tests_passed++;
}

static void test_bias(void) {
    printf("test_bias...\n");
    float l[5];
    load(l);
    int ids[2]      = {2, 4};
    float biases[2] = {1.0f, -INFINITY};
    cml_logits_bias(l, 5, ids, biases, 2);
    ASSERT(fabsf(l[2] - 1.5f) < 1e-6f, "bias adds to the targeted logit"); /* 0.5 + 1.0 */
    ASSERT(!isfinite(l[4]), "a -inf bias hard-bans a token");
    ASSERT(fabsf(l[0] - 2.0f) < 1e-6f, "unbiased logit untouched");
    tests_passed++;
}

static void test_frequency_presence_penalty(void) {
    printf("test_frequency_presence_penalty...\n");
    float l[5];
    load(l);
    int hist[3] = {0, 0, 4}; /* token 0 appears twice, token 4 once */
    cml_logits_frequency_presence_penalty(l, 5, hist, 3, 0.5f, 1.0f);
    /* token 0: 2.0 - 0.5*2 (frequency) - 1.0 (presence) = 0.0 */
    ASSERT(fabsf(l[0] - 0.0f) < 1e-6f, "frequency+presence on repeated token");
    /* token 4: -1.0 - 0.5*1 - 1.0 = -2.5 */
    ASSERT(fabsf(l[4] - (-2.5f)) < 1e-6f, "frequency+presence on single token");
    ASSERT(fabsf(l[1] - 1.0f) < 1e-6f, "untouched token");
    tests_passed++;
}

static void test_no_repeat_ngram(void) {
    printf("test_no_repeat_ngram...\n");
    float l[6];
    for (int i = 0; i < 6; i++)
        l[i] = 0.0f;
    /* history [5,1,2,1], prefix (last 1) = [1]; earlier '1' at idx 1 is followed by token 2. */
    int hist[4] = {5, 1, 2, 1};
    int kept    = cml_logits_no_repeat_ngram(l, 6, hist, 4, 2);
    ASSERT(kept == 5, "no_repeat_ngram kept count");
    ASSERT(!isfinite(l[2]), "no_repeat_ngram bans the completing token");
    ASSERT(isfinite(l[0]) && isfinite(l[1]) && isfinite(l[3]), "no_repeat_ngram keeps the rest");
    /* ngram_size 3: prefix [2,1] never recurs earlier -> nothing banned. */
    for (int i = 0; i < 6; i++)
        l[i] = 0.0f;
    ASSERT(cml_logits_no_repeat_ngram(l, 6, hist, 4, 3) == 6, "no_repeat_ngram no-op when unseen");
    tests_passed++;
}

static void test_typical(void) {
    printf("test_typical...\n");
    float l[5];
    load(l);
    int kept = cml_logits_typical(l, 5, 0.9f); /* numpy oracle: keep {0,1,2,3} */
    ASSERT(kept == 4, "typical(0.9) kept count");
    ASSERT(isfinite(l[0]) && isfinite(l[1]) && isfinite(l[2]) && isfinite(l[3]),
           "typical(0.9) keeps {0..3}");
    ASSERT(!isfinite(l[4]), "typical(0.9) masks the tail");
    load(l);
    kept = cml_logits_typical(l, 5, 0.6f); /* numpy oracle: keep {0,1} */
    ASSERT(kept == 2, "typical(0.6) kept count");
    ASSERT(isfinite(l[0]) && isfinite(l[1]), "typical(0.6) keeps {0,1}");
    ASSERT(!isfinite(l[2]) && !isfinite(l[3]) && !isfinite(l[4]), "typical(0.6) masks the rest");
    tests_passed++;
}

int main(void) {
    printf("Sampling / logits-processor tests\n\n");
    test_temperature();
    test_bias();
    test_repetition_penalty();
    test_top_k();
    test_top_p();
    test_min_p();
    test_frequency_presence_penalty();
    test_no_repeat_ngram();
    test_typical();
    printf("\nResults: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
