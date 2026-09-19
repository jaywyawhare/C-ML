/*
 * AUDIT #18: the serving layer now drives real autoregressive generation.
 *
 * Uses a deterministic mock model (next token = last token + 1, mod vocab) and
 * exercises the full loop: submit -> step (prefill+decode) -> get_tokens ->
 * finish, verifying token sequences, EOS termination, max-token termination,
 * multi-request batching, and stats.
 */
#include <stdio.h>
#include <string.h>

#include "nn/serving.h"
#include "test_harness.h"

static int check(const char* name, int ok) {
    tests_run++;
    if (ok) {
        tests_passed++;
        printf("  PASS: %s\n", name);
    } else {
        printf("  FAIL: %s\n", name);
    }
    return ok;
}

/* Deterministic mock: argmax(logits) = (last_input_token + 1) % vocab. */
static int mock_forward(void* model, const int* tokens, int num_tokens, int position, float* logits,
                        int vocab) {
    (void)model;
    (void)position;
    if (num_tokens <= 0)
        return -1;
    int cur  = tokens[num_tokens - 1];
    int next = (cur + 1) % vocab;
    for (int i = 0; i < vocab; i++)
        logits[i] = 0.0f;
    logits[next] = 10.0f;
    return 0;
}

static void drain(CMLServingContext* ctx, int id) {
    int guard = 0;
    while (cml_serving_get_status(ctx, id) == CML_SEQ_STATUS_DECODING ||
           cml_serving_get_status(ctx, id) == CML_SEQ_STATUS_QUEUED ||
           cml_serving_get_status(ctx, id) == CML_SEQ_STATUS_PREFILL) {
        if (guard++ > 10000)
            break;
        cml_serving_step(ctx);
    }
}

static int test_eos_termination(void) {
    CMLServingConfig cfg    = cml_serving_default_config();
    cfg.temperature_default = 0.0f; /* greedy => deterministic */
    CMLServingContext* ctx  = cml_serving_create(&cfg);
    cml_serving_set_model(ctx, mock_forward, NULL, /*vocab*/ 32, /*eos*/ 10);

    int prompt[] = {1, 2, 3, 4, 5};
    int id       = cml_serving_submit(ctx, prompt, 5, /*max_new*/ 20);
    drain(ctx, id);

    int ok          = cml_serving_get_status(ctx, id) == CML_SEQ_STATUS_FINISHED;
    int n           = 0;
    const int* toks = cml_serving_get_tokens(ctx, id, &n);
    /* last prompt token 5 -> 6,7,8,9,10(eos) */
    int expect[] = {6, 7, 8, 9, 10};
    ok           = ok && n == 5 && toks && memcmp(toks, expect, sizeof(expect)) == 0;

    cml_serving_finish_request(ctx, id);
    CMLServingStats st = cml_serving_get_stats(ctx);
    ok                 = ok && st.completed_requests == 1 && st.total_tokens_generated == 5;

    cml_serving_free(ctx);
    return ok;
}

static int test_max_tokens_termination(void) {
    CMLServingConfig cfg    = cml_serving_default_config();
    cfg.temperature_default = 0.0f;
    CMLServingContext* ctx  = cml_serving_create(&cfg);
    cml_serving_set_model(ctx, mock_forward, NULL, 64, /*eos*/ -1); /* no eos */

    int prompt[] = {0};
    int id       = cml_serving_submit(ctx, prompt, 1, /*max_new*/ 8);
    drain(ctx, id);

    int n           = 0;
    const int* toks = cml_serving_get_tokens(ctx, id, &n);
    int expect[]    = {1, 2, 3, 4, 5, 6, 7, 8};
    int ok = cml_serving_get_status(ctx, id) == CML_SEQ_STATUS_FINISHED && n == 8 && toks &&
             memcmp(toks, expect, sizeof(expect)) == 0;
    cml_serving_finish_request(ctx, id);
    cml_serving_free(ctx);
    return ok;
}

static int test_batched_requests(void) {
    CMLServingConfig cfg    = cml_serving_default_config();
    cfg.temperature_default = 0.0f;
    CMLServingContext* ctx  = cml_serving_create(&cfg);
    cml_serving_set_model(ctx, mock_forward, NULL, 100, -1);

    int p1[] = {10};
    int p2[] = {20};
    int p3[] = {30};
    int a    = cml_serving_submit(ctx, p1, 1, 3);
    int b    = cml_serving_submit(ctx, p2, 1, 3);
    int c    = cml_serving_submit(ctx, p3, 1, 3);
    /* step until all finished */
    int guard = 0;
    while (guard++ < 1000 && (cml_serving_get_status(ctx, a) != CML_SEQ_STATUS_FINISHED ||
                              cml_serving_get_status(ctx, b) != CML_SEQ_STATUS_FINISHED ||
                              cml_serving_get_status(ctx, c) != CML_SEQ_STATUS_FINISHED)) {
        cml_serving_step(ctx);
    }
    int na = 0, nb = 0, nc = 0;
    const int* ta = cml_serving_get_tokens(ctx, a, &na);
    const int* tb = cml_serving_get_tokens(ctx, b, &nb);
    const int* tc = cml_serving_get_tokens(ctx, c, &nc);
    int ok        = na == 3 && nb == 3 && nc == 3 && ta[0] == 11 && tb[0] == 21 && tc[0] == 31 &&
             ta[2] == 13 && tb[2] == 23 && tc[2] == 33;
    cml_serving_finish_request(ctx, a);
    cml_serving_finish_request(ctx, b);
    cml_serving_finish_request(ctx, c);
    cml_serving_free(ctx);
    return ok;
}

int main(void) {
    printf("=== AUDIT #18: serving generation loop ===\n");
    check("eos_termination", test_eos_termination());
    check("max_tokens_termination", test_max_tokens_termination());
    check("batched_requests", test_batched_requests());
    return TEST_SUMMARY();
}
