/* BPE tokenizer training: learns merges from a repetitive corpus (so the vocabulary
 * grows past the distinct characters), encode/decode round-trips the text exactly, and
 * learned merges compress a training string below its character count. */
#include <stdio.h>
#include <string.h>
#include "cml.h"
#include "nn/llm_ops.h"
#include "test_harness.h"

static int test_trains_and_roundtrips(void) {
    const char* corpus[] = {"ababababab", "abcabcabc", "aaaa bbbb"};
    CMLTokenizer* tok    = cml_tokenizer_train(corpus, 3, 40);
    if (!tok)
        return 0;

    /* Distinct chars are a,b,c,space = 4; merges should have added more. */
    int ok = cml_tokenizer_vocab_size(tok) > 4;

    /* Round-trip a training string exactly. */
    const char* text = "ababab";
    int n            = 0;
    int* ids         = cml_tokenizer_encode(tok, text, &n);
    ok &= ids != NULL && n > 0;
    /* merges collapsed "ab" pairs, so fewer tokens than the 6 characters */
    ok &= n < (int)strlen(text);
    char* back = cml_tokenizer_decode(tok, ids, n);
    ok &= back != NULL && strcmp(back, text) == 0;

    cml_free(ids);
    cml_free(back);
    cml_tokenizer_free(tok);
    return ok;
}

static int test_roundtrip_unseen_chars(void) {
    /* Characters present in the corpus round-trip even in new arrangements. */
    const char* corpus[] = {"hello world hello"};
    CMLTokenizer* tok    = cml_tokenizer_train(corpus, 1, 30);
    if (!tok)
        return 0;
    const char* text = "world hello";
    int n            = 0;
    int* ids         = cml_tokenizer_encode(tok, text, &n);
    char* back       = cml_tokenizer_decode(tok, ids, n);
    int ok           = back != NULL && strcmp(back, text) == 0;
    cml_free(ids);
    cml_free(back);
    cml_tokenizer_free(tok);
    return ok;
}

static int test_rejects_bad_args(void) {
    return cml_tokenizer_train(NULL, 1, 10) == NULL &&
           cml_tokenizer_train((const char*[]){"x"}, 0, 10) == NULL;
}

int main(void) {
    cml_init();
    printf("=== BPE tokenizer training ===\n");
    TEST(trains_and_roundtrips);
    TEST(roundtrip_unseen_chars);
    TEST(rejects_bad_args);
    cml_cleanup();
    return TEST_SUMMARY();
}
