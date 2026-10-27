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

int main(void) {
    printf("Sampling / logits-processor tests\n\n");
    test_temperature();
    test_repetition_penalty();
    test_top_k();
    test_top_p();
    test_min_p();
    printf("\nResults: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
