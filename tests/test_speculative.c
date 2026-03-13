/**
 * @file test_speculative.c
 * @brief Tests for speculative decoding
 *
 * Uses mock draft/target models with predictable forward and sample
 * functions so that acceptance / rejection behaviour is deterministic.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "nn/speculative.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-50s ", #name); \
    fflush(stdout); \
    if (test_##name()) { \
        tests_passed++; \
        printf("[PASS]\n"); \
    } else { \
        printf("[FAIL]\n"); \
    } \
} while(0)

/* ===================================================================
 * Mock models
 *
 * Both mock models ignore the actual token content and simply return
 * logits tensors where a specific token index has the highest value.
 * This lets us control exactly which tokens are "predicted" and thus
 * whether the target agrees or disagrees with the draft.
 * =================================================================== */

#define MOCK_VOCAB_SIZE 32

/* ---------- Mock context ------------------------------------------ */

typedef struct MockModelCtx {
    int predict_token;   /* The token this model always predicts. */
} MockModelCtx;

/* ---------- Forward: returns [seq_len, MOCK_VOCAB_SIZE] logits ----- */

static Tensor* mock_forward(void* model_ctx, const int* token_ids, int seq_len) {
    (void)token_ids;  /* Not used by the mock. */

    MockModelCtx* ctx = (MockModelCtx*)model_ctx;
    int predict = ctx->predict_token;

    int shape[] = {seq_len, MOCK_VOCAB_SIZE};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .has_dtype = true,
        .has_device = true
    };
    Tensor* logits = tensor_zeros(shape, 2, &cfg);
    if (!logits) return NULL;
    tensor_ensure_executed(logits);

    float* data = (float*)tensor_data_ptr(logits);
    if (!data) { tensor_free(logits); return NULL; }

    /* For every row, set the "predict" column to a large value. */
    for (int r = 0; r < seq_len; r++) {
        data[r * MOCK_VOCAB_SIZE + predict] = 10.0f;
    }

    return logits;
}

/* ---------- Sample: argmax of last row ----------------------------- */

static int mock_sample(void* model_ctx, Tensor* logits, float temperature) {
    (void)model_ctx;
    (void)temperature;

    if (!logits) return 0;
    tensor_ensure_executed(logits);
    float* data = (float*)tensor_data_ptr(logits);
    if (!data) return 0;

    /* Find last row. */
    int seq_len = logits->shape[0];
    int vocab   = logits->shape[1];
    int offset  = (seq_len - 1) * vocab;

    int best = 0;
    float best_val = data[offset];
    for (int i = 1; i < vocab; i++) {
        if (data[offset + i] > best_val) {
            best_val = data[offset + i];
            best = i;
        }
    }
    return best;
}

/* ---------- Position-aware forward for partial-agree tests -------- */

/**
 * A forward function whose prediction depends on the sequence length.
 * It agrees with token 7 for the first N positions, then predicts
 * token 13 for the rest (simulating a disagreement after N tokens).
 *
 * The context's predict_token stores the "agree count" N.
 */
typedef struct PartialAgreeCtx {
    int agree_count;     /* Number of draft positions to agree with. */
    int agree_token;     /* Token to agree on. */
    int disagree_token;  /* Token to predict after agree_count mismatches. */
} PartialAgreeCtx;

static Tensor* partial_agree_forward(void* model_ctx, const int* token_ids,
                                     int seq_len) {
    (void)token_ids;

    PartialAgreeCtx* ctx = (PartialAgreeCtx*)model_ctx;

    int shape[] = {seq_len, MOCK_VOCAB_SIZE};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .has_dtype = true,
        .has_device = true
    };
    Tensor* logits = tensor_zeros(shape, 2, &cfg);
    if (!logits) return NULL;
    tensor_ensure_executed(logits);

    float* data = (float*)tensor_data_ptr(logits);
    if (!data) { tensor_free(logits); return NULL; }

    /*
     * For verify, the target model gets prefix_len + K tokens.
     * Row (prefix_len - 1 + i) is used to verify draft position i.
     * We want the first agree_count rows (starting from row 0) to
     * predict agree_token and the rest to predict disagree_token.
     */
    for (int r = 0; r < seq_len; r++) {
        int tok = (r < ctx->agree_count) ? ctx->agree_token
                                         : ctx->disagree_token;
        data[r * MOCK_VOCAB_SIZE + tok] = 10.0f;
    }

    return logits;
}

static int partial_agree_sample(void* model_ctx, Tensor* logits,
                                float temperature) {
    /* Reuse the generic argmax sampler. */
    return mock_sample(model_ctx, logits, temperature);
}

/* ===================================================================
 * Tests
 * =================================================================== */

/* ----- 1. Create / free decoder ----------------------------------- */

static int test_create_free(void) {
    CMLSpeculativeConfig cfg = cml_speculative_default_config();
    CMLSpeculativeDecoder* dec = cml_speculative_create(&cfg, MOCK_VOCAB_SIZE);
    if (!dec) return 0;
    if (dec->vocab_size != MOCK_VOCAB_SIZE) { cml_speculative_free(dec); return 0; }
    if (dec->config.num_draft_tokens != 5) { cml_speculative_free(dec); return 0; }
    cml_speculative_free(dec);
    return 1;
}

static int test_create_null_config(void) {
    CMLSpeculativeDecoder* dec = cml_speculative_create(NULL, MOCK_VOCAB_SIZE);
    if (dec != NULL) { cml_speculative_free(dec); return 0; }
    return 1;
}

static int test_create_bad_vocab(void) {
    CMLSpeculativeConfig cfg = cml_speculative_default_config();
    CMLSpeculativeDecoder* dec = cml_speculative_create(&cfg, 0);
    if (dec != NULL) { cml_speculative_free(dec); return 0; }
    dec = cml_speculative_create(&cfg, -1);
    if (dec != NULL) { cml_speculative_free(dec); return 0; }
    return 1;
}

static int test_create_bad_k(void) {
    CMLSpeculativeConfig cfg = cml_speculative_default_config();
    cfg.num_draft_tokens = 0;
    CMLSpeculativeDecoder* dec = cml_speculative_create(&cfg, MOCK_VOCAB_SIZE);
    if (dec != NULL) { cml_speculative_free(dec); return 0; }

    cfg.num_draft_tokens = CML_SPEC_MAX_DRAFT_TOKENS + 1;
    dec = cml_speculative_create(&cfg, MOCK_VOCAB_SIZE);
    if (dec != NULL) { cml_speculative_free(dec); return 0; }
    return 1;
}

static int test_free_null(void) {
    /* Should not crash. */
    cml_speculative_free(NULL);
    return 1;
}

static int test_result_free_null(void) {
    /* Should not crash. */
    cml_speculative_result_free(NULL);
    return 1;
}

/* ----- 2. Default config values ----------------------------------- */

static int test_default_config(void) {
    CMLSpeculativeConfig cfg = cml_speculative_default_config();
    if (cfg.num_draft_tokens != 5) return 0;
    if (fabsf(cfg.temperature - 0.8f) > 1e-5f) return 0;
    if (fabsf(cfg.top_p - 0.9f) > 1e-5f) return 0;
    if (cfg.top_k != 40) return 0;
    if (!cfg.do_sample) return 0;
    return 1;
}

/* ----- 3. Full agreement: draft and target predict the same token -- */

static int test_full_agreement(void) {
    CMLSpeculativeConfig cfg = cml_speculative_default_config();
    cfg.num_draft_tokens = 5;
    CMLSpeculativeDecoder* dec = cml_speculative_create(&cfg, MOCK_VOCAB_SIZE);
    if (!dec) return 0;

    /* Both models always predict token 7. */
    MockModelCtx draft_ctx = { .predict_token = 7 };
    MockModelCtx target_ctx = { .predict_token = 7 };

    cml_speculative_set_draft_model(dec, &draft_ctx, mock_forward, mock_sample);
    cml_speculative_set_target_model(dec, &target_ctx, mock_forward, mock_sample);

    int prefix[] = {1, 2, 3};
    CMLSpeculativeResult* res = cml_speculative_decode_step(dec, prefix, 3);
    if (!res) { cml_speculative_free(dec); return 0; }

    /* All 5 draft tokens accepted + 1 bonus = 6 total. */
    if (res->num_accepted != 6) {
        printf("(expected 6 accepted, got %d) ", res->num_accepted);
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }

    /* Acceptance rate for the draft portion should be 1.0 (5/5). */
    if (fabsf(res->acceptance_rate - 1.0f) > 1e-5f) {
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }

    /* All tokens should be 7. */
    for (int i = 0; i < res->num_accepted; i++) {
        if (res->accepted_tokens[i] != 7) {
            cml_speculative_result_free(res);
            cml_speculative_free(dec);
            return 0;
        }
    }

    if (res->num_drafted != 5) {
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }

    cml_speculative_result_free(res);
    cml_speculative_free(dec);
    return 1;
}

/* ----- 4. Full disagreement: target always disagrees --------------- */

static int test_full_disagreement(void) {
    CMLSpeculativeConfig cfg = cml_speculative_default_config();
    cfg.num_draft_tokens = 5;
    CMLSpeculativeDecoder* dec = cml_speculative_create(&cfg, MOCK_VOCAB_SIZE);
    if (!dec) return 0;

    /* Draft predicts 7, target predicts 13. */
    MockModelCtx draft_ctx = { .predict_token = 7 };
    MockModelCtx target_ctx = { .predict_token = 13 };

    cml_speculative_set_draft_model(dec, &draft_ctx, mock_forward, mock_sample);
    cml_speculative_set_target_model(dec, &target_ctx, mock_forward, mock_sample);

    int prefix[] = {1, 2, 3};
    CMLSpeculativeResult* res = cml_speculative_decode_step(dec, prefix, 3);
    if (!res) { cml_speculative_free(dec); return 0; }

    /* First draft token rejected immediately; 0 accepted + 1 correction = 1 token. */
    if (res->num_accepted != 1) {
        printf("(expected 1 accepted, got %d) ", res->num_accepted);
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }

    /* The single output token should be the target's correction: 13. */
    if (res->accepted_tokens[0] != 13) {
        printf("(expected correction 13, got %d) ", res->accepted_tokens[0]);
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }

    /* Acceptance rate should be 0.0. */
    if (fabsf(res->acceptance_rate) > 1e-5f) {
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }

    cml_speculative_result_free(res);
    cml_speculative_free(dec);
    return 1;
}

/* ----- 5. Partial agreement: target agrees on first 3, rejects 4th - */

static int test_partial_agreement(void) {
    CMLSpeculativeConfig cfg = cml_speculative_default_config();
    cfg.num_draft_tokens = 5;
    CMLSpeculativeDecoder* dec = cml_speculative_create(&cfg, MOCK_VOCAB_SIZE);
    if (!dec) return 0;

    /* Draft always predicts token 7. */
    MockModelCtx draft_ctx = { .predict_token = 7 };
    cml_speculative_set_draft_model(dec, &draft_ctx, mock_forward, mock_sample);

    /*
     * Target agrees for the first (prefix_len - 1 + 3) = 5 rows,
     * then disagrees.  With prefix_len=3, draft position i uses target
     * row (2 + i).  So rows 0..4 predict 7, rows 5+ predict 13.
     * Draft positions 0,1,2 (rows 2,3,4) => agree.
     * Draft position 3 (row 5) => disagree.
     */
    PartialAgreeCtx target_ctx = {
        .agree_count = 5,    /* rows 0..4 predict agree_token */
        .agree_token = 7,
        .disagree_token = 13
    };
    cml_speculative_set_target_model(dec, &target_ctx,
                                     partial_agree_forward,
                                     partial_agree_sample);

    int prefix[] = {1, 2, 3};
    CMLSpeculativeResult* res = cml_speculative_decode_step(dec, prefix, 3);
    if (!res) { cml_speculative_free(dec); return 0; }

    /* 3 accepted + 1 correction = 4 tokens total. */
    if (res->num_accepted != 4) {
        printf("(expected 4 accepted, got %d) ", res->num_accepted);
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }

    /* First 3 should be 7 (the draft token), last should be 13 (correction). */
    for (int i = 0; i < 3; i++) {
        if (res->accepted_tokens[i] != 7) {
            printf("(token %d expected 7, got %d) ", i, res->accepted_tokens[i]);
            cml_speculative_result_free(res);
            cml_speculative_free(dec);
            return 0;
        }
    }
    if (res->accepted_tokens[3] != 13) {
        printf("(correction expected 13, got %d) ", res->accepted_tokens[3]);
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }

    /* Acceptance rate for draft portion: 3/5 = 0.6. */
    if (fabsf(res->acceptance_rate - 0.6f) > 1e-5f) {
        printf("(expected rate 0.6, got %.4f) ", res->acceptance_rate);
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }

    cml_speculative_result_free(res);
    cml_speculative_free(dec);
    return 1;
}

/* ----- 6. Lifetime acceptance rate statistics --------------------- */

static int test_acceptance_rate_stats(void) {
    CMLSpeculativeConfig cfg = cml_speculative_default_config();
    cfg.num_draft_tokens = 5;
    CMLSpeculativeDecoder* dec = cml_speculative_create(&cfg, MOCK_VOCAB_SIZE);
    if (!dec) return 0;

    /* Initial rate should be 0. */
    if (fabsf(cml_speculative_acceptance_rate(dec)) > 1e-5f) {
        cml_speculative_free(dec);
        return 0;
    }

    /* Run a fully-agreeing step. */
    MockModelCtx agree_ctx = { .predict_token = 7 };
    cml_speculative_set_draft_model(dec, &agree_ctx, mock_forward, mock_sample);
    cml_speculative_set_target_model(dec, &agree_ctx, mock_forward, mock_sample);

    int prefix[] = {1, 2, 3};
    CMLSpeculativeResult* res1 = cml_speculative_decode_step(dec, prefix, 3);
    if (!res1) { cml_speculative_free(dec); return 0; }
    cml_speculative_result_free(res1);

    /* After full agreement: 5/5 = 1.0. */
    float rate = cml_speculative_acceptance_rate(dec);
    if (fabsf(rate - 1.0f) > 1e-5f) {
        printf("(after agree step: expected 1.0, got %.4f) ", rate);
        cml_speculative_free(dec);
        return 0;
    }

    /* Run a fully-disagreeing step. */
    MockModelCtx disagree_target = { .predict_token = 13 };
    cml_speculative_set_target_model(dec, &disagree_target,
                                     mock_forward, mock_sample);

    CMLSpeculativeResult* res2 = cml_speculative_decode_step(dec, prefix, 3);
    if (!res2) { cml_speculative_free(dec); return 0; }
    cml_speculative_result_free(res2);

    /* Lifetime: 5 accepted out of 10 drafted = 0.5. */
    rate = cml_speculative_acceptance_rate(dec);
    if (fabsf(rate - 0.5f) > 1e-5f) {
        printf("(after disagree step: expected 0.5, got %.4f) ", rate);
        cml_speculative_free(dec);
        return 0;
    }

    if (dec->total_steps != 2) {
        cml_speculative_free(dec);
        return 0;
    }

    cml_speculative_free(dec);
    return 1;
}

/* ----- 7. Edge case: K = 1 ---------------------------------------- */

static int test_k_equals_1_agree(void) {
    CMLSpeculativeConfig cfg = cml_speculative_default_config();
    cfg.num_draft_tokens = 1;
    CMLSpeculativeDecoder* dec = cml_speculative_create(&cfg, MOCK_VOCAB_SIZE);
    if (!dec) return 0;

    MockModelCtx ctx = { .predict_token = 5 };
    cml_speculative_set_draft_model(dec, &ctx, mock_forward, mock_sample);
    cml_speculative_set_target_model(dec, &ctx, mock_forward, mock_sample);

    int prefix[] = {1, 2};
    CMLSpeculativeResult* res = cml_speculative_decode_step(dec, prefix, 2);
    if (!res) { cml_speculative_free(dec); return 0; }

    /* 1 draft accepted + 1 bonus = 2 tokens. */
    if (res->num_accepted != 2) {
        printf("(expected 2 accepted, got %d) ", res->num_accepted);
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }
    if (res->accepted_tokens[0] != 5 || res->accepted_tokens[1] != 5) {
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }
    if (fabsf(res->acceptance_rate - 1.0f) > 1e-5f) {
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }

    cml_speculative_result_free(res);
    cml_speculative_free(dec);
    return 1;
}

static int test_k_equals_1_disagree(void) {
    CMLSpeculativeConfig cfg = cml_speculative_default_config();
    cfg.num_draft_tokens = 1;
    CMLSpeculativeDecoder* dec = cml_speculative_create(&cfg, MOCK_VOCAB_SIZE);
    if (!dec) return 0;

    MockModelCtx draft_ctx = { .predict_token = 5 };
    MockModelCtx target_ctx = { .predict_token = 20 };
    cml_speculative_set_draft_model(dec, &draft_ctx, mock_forward, mock_sample);
    cml_speculative_set_target_model(dec, &target_ctx, mock_forward, mock_sample);

    int prefix[] = {1, 2};
    CMLSpeculativeResult* res = cml_speculative_decode_step(dec, prefix, 2);
    if (!res) { cml_speculative_free(dec); return 0; }

    /* 0 accepted + 1 correction = 1 token. */
    if (res->num_accepted != 1) {
        printf("(expected 1 accepted, got %d) ", res->num_accepted);
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }
    if (res->accepted_tokens[0] != 20) {
        printf("(expected correction 20, got %d) ", res->accepted_tokens[0]);
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }
    if (fabsf(res->acceptance_rate) > 1e-5f) {
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }

    cml_speculative_result_free(res);
    cml_speculative_free(dec);
    return 1;
}

/* ----- 8. Decode step with no callbacks set ----------------------- */

static int test_decode_no_callbacks(void) {
    CMLSpeculativeConfig cfg = cml_speculative_default_config();
    CMLSpeculativeDecoder* dec = cml_speculative_create(&cfg, MOCK_VOCAB_SIZE);
    if (!dec) return 0;

    int prefix[] = {1, 2, 3};
    CMLSpeculativeResult* res = cml_speculative_decode_step(dec, prefix, 3);
    /* Should fail gracefully. */
    if (res != NULL) {
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }

    cml_speculative_free(dec);
    return 1;
}

/* ----- 9. Decode step with invalid prefix ------------------------- */

static int test_decode_invalid_prefix(void) {
    CMLSpeculativeConfig cfg = cml_speculative_default_config();
    CMLSpeculativeDecoder* dec = cml_speculative_create(&cfg, MOCK_VOCAB_SIZE);
    if (!dec) return 0;

    MockModelCtx ctx = { .predict_token = 7 };
    cml_speculative_set_draft_model(dec, &ctx, mock_forward, mock_sample);
    cml_speculative_set_target_model(dec, &ctx, mock_forward, mock_sample);

    /* NULL prefix. */
    CMLSpeculativeResult* res = cml_speculative_decode_step(dec, NULL, 3);
    if (res != NULL) { cml_speculative_result_free(res); cml_speculative_free(dec); return 0; }

    /* Zero-length prefix. */
    int prefix[] = {1};
    res = cml_speculative_decode_step(dec, prefix, 0);
    if (res != NULL) { cml_speculative_result_free(res); cml_speculative_free(dec); return 0; }

    /* NULL decoder. */
    res = cml_speculative_decode_step(NULL, prefix, 1);
    if (res != NULL) { cml_speculative_result_free(res); cml_speculative_free(dec); return 0; }

    cml_speculative_free(dec);
    return 1;
}

/* ----- 10. Timing fields are populated ----------------------------- */

static int test_timing_fields(void) {
    CMLSpeculativeConfig cfg = cml_speculative_default_config();
    cfg.num_draft_tokens = 3;
    CMLSpeculativeDecoder* dec = cml_speculative_create(&cfg, MOCK_VOCAB_SIZE);
    if (!dec) return 0;

    MockModelCtx ctx = { .predict_token = 7 };
    cml_speculative_set_draft_model(dec, &ctx, mock_forward, mock_sample);
    cml_speculative_set_target_model(dec, &ctx, mock_forward, mock_sample);

    int prefix[] = {1, 2, 3};
    CMLSpeculativeResult* res = cml_speculative_decode_step(dec, prefix, 3);
    if (!res) { cml_speculative_free(dec); return 0; }

    /* Timing values should be non-negative (they can be 0.0 on fast CPUs). */
    if (res->draft_time_ms < 0.0 || res->verify_time_ms < 0.0 ||
        res->total_time_ms < 0.0) {
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }

    /* total >= draft + verify (approximately). */
    if (res->total_time_ms + 0.01 < res->draft_time_ms) {
        cml_speculative_result_free(res);
        cml_speculative_free(dec);
        return 0;
    }

    cml_speculative_result_free(res);
    cml_speculative_free(dec);
    return 1;
}

/* ----- 11. Set model on NULL decoder ------------------------------ */

static int test_set_model_null_decoder(void) {
    /* Should not crash. */
    cml_speculative_set_draft_model(NULL, NULL, mock_forward, mock_sample);
    cml_speculative_set_target_model(NULL, NULL, mock_forward, mock_sample);
    return 1;
}

/* ----- 12. Acceptance rate on NULL / fresh decoder ----------------- */

static int test_acceptance_rate_null(void) {
    if (fabsf(cml_speculative_acceptance_rate(NULL)) > 1e-5f) return 0;

    CMLSpeculativeConfig cfg = cml_speculative_default_config();
    CMLSpeculativeDecoder* dec = cml_speculative_create(&cfg, MOCK_VOCAB_SIZE);
    if (!dec) return 0;

    /* No steps yet; rate should be 0. */
    if (fabsf(cml_speculative_acceptance_rate(dec)) > 1e-5f) {
        cml_speculative_free(dec);
        return 0;
    }

    cml_speculative_free(dec);
    return 1;
}

/* ===================================================================
 * Main
 * =================================================================== */

int main(void) {
    printf("test_speculative\n\n");

    /* Create / free */
    TEST(create_free);
    TEST(create_null_config);
    TEST(create_bad_vocab);
    TEST(create_bad_k);
    TEST(free_null);
    TEST(result_free_null);

    /* Config */
    TEST(default_config);

    /* Agreement scenarios */
    TEST(full_agreement);
    TEST(full_disagreement);
    TEST(partial_agreement);

    /* Statistics */
    TEST(acceptance_rate_stats);

    /* Edge cases */
    TEST(k_equals_1_agree);
    TEST(k_equals_1_disagree);
    TEST(decode_no_callbacks);
    TEST(decode_invalid_prefix);
    TEST(timing_fields);
    TEST(set_model_null_decoder);
    TEST(acceptance_rate_null);

    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
