/**
 * @file speculative.c
 * @brief Speculative decoding implementation
 *
 * Draft K tokens with a small model, verify them in a single forward pass
 * of the larger target model, accept matching tokens and use the target
 * model's prediction as a correction on the first mismatch.
 */

#include "nn/speculative.h"
#include "core/logging.h"
#include "tensor/tensor.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ===== Helpers ===== */

/** Return wall-clock time in milliseconds. */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/**
 * Given a 2D logits tensor [seq_len, vocab_size], return the argmax token
 * for the row at @p row_index.
 */
static int argmax_at_row(Tensor* logits, int row_index, int vocab_size) {
    if (!logits || !logits->data) {
        tensor_ensure_executed(logits);
    }
    float* data = (float*)tensor_data_ptr(logits);
    if (!data) return 0;

    int offset = row_index * vocab_size;
    int best = 0;
    float best_val = data[offset];
    for (int i = 1; i < vocab_size; i++) {
        if (data[offset + i] > best_val) {
            best_val = data[offset + i];
            best = i;
        }
    }
    return best;
}

/* ===== Public API ===== */

CMLSpeculativeConfig cml_speculative_default_config(void) {
    CMLSpeculativeConfig cfg;
    cfg.num_draft_tokens = 5;
    cfg.temperature      = 0.8f;
    cfg.top_p            = 0.9f;
    cfg.top_k            = 40;
    cfg.do_sample        = true;
    return cfg;
}

CMLSpeculativeDecoder* cml_speculative_create(const CMLSpeculativeConfig* config,
                                               int vocab_size) {
    if (!config) {
        LOG_ERROR("cml_speculative_create: config is NULL");
        return NULL;
    }
    if (vocab_size <= 0) {
        LOG_ERROR("cml_speculative_create: vocab_size must be > 0");
        return NULL;
    }
    if (config->num_draft_tokens <= 0 ||
        config->num_draft_tokens > CML_SPEC_MAX_DRAFT_TOKENS) {
        LOG_ERROR("cml_speculative_create: num_draft_tokens must be in [1, %d]",
                  CML_SPEC_MAX_DRAFT_TOKENS);
        return NULL;
    }

    CMLSpeculativeDecoder* dec = (CMLSpeculativeDecoder*)calloc(1, sizeof(CMLSpeculativeDecoder));
    if (!dec) {
        LOG_ERROR("cml_speculative_create: allocation failed");
        return NULL;
    }

    dec->config     = *config;
    dec->vocab_size = vocab_size;
    return dec;
}

void cml_speculative_free(CMLSpeculativeDecoder* decoder) {
    if (!decoder) return;
    free(decoder);
}

void cml_speculative_set_draft_model(CMLSpeculativeDecoder* dec, void* ctx,
                                     CMLModelForwardFn forward_fn,
                                     CMLSampleTokenFn sample_fn) {
    if (!dec) return;
    dec->draft_model_ctx = ctx;
    dec->draft_forward   = forward_fn;
    dec->draft_sample    = sample_fn;
}

void cml_speculative_set_target_model(CMLSpeculativeDecoder* dec, void* ctx,
                                      CMLModelForwardFn forward_fn,
                                      CMLSampleTokenFn sample_fn) {
    if (!dec) return;
    dec->target_model_ctx = ctx;
    dec->target_forward   = forward_fn;
    dec->target_sample    = sample_fn;
}

CMLSpeculativeResult* cml_speculative_decode_step(CMLSpeculativeDecoder* dec,
                                                   const int* prefix_tokens,
                                                   int prefix_len) {
    if (!dec) {
        LOG_ERROR("cml_speculative_decode_step: decoder is NULL");
        return NULL;
    }
    if (!dec->draft_forward || !dec->draft_sample ||
        !dec->target_forward || !dec->target_sample) {
        LOG_ERROR("cml_speculative_decode_step: model callbacks not set");
        return NULL;
    }
    if (!prefix_tokens || prefix_len <= 0) {
        LOG_ERROR("cml_speculative_decode_step: invalid prefix");
        return NULL;
    }

    int K = dec->config.num_draft_tokens;
    float temperature = dec->config.temperature;

    /* Allocate working buffer for prefix + K draft tokens. */
    int max_seq = prefix_len + K;
    int* full_seq = (int*)malloc((size_t)max_seq * sizeof(int));
    if (!full_seq) {
        LOG_ERROR("cml_speculative_decode_step: allocation failed");
        return NULL;
    }
    memcpy(full_seq, prefix_tokens, (size_t)prefix_len * sizeof(int));

    int* draft_tokens = (int*)malloc((size_t)K * sizeof(int));
    if (!draft_tokens) {
        free(full_seq);
        LOG_ERROR("cml_speculative_decode_step: allocation failed");
        return NULL;
    }

    /* ------------------------------------------------------------------
     * 1. Draft phase: autoregressively generate K tokens with draft model.
     * ------------------------------------------------------------------ */
    double t_draft_start = now_ms();

    int draft_seq_len = prefix_len;
    for (int i = 0; i < K; i++) {
        /* Forward the draft model on the current sequence. */
        Tensor* draft_logits = dec->draft_forward(dec->draft_model_ctx,
                                                   full_seq, draft_seq_len);
        if (!draft_logits) {
            LOG_WARNING("draft forward returned NULL at step %d", i);
            K = i;  /* Truncate to whatever we managed. */
            break;
        }

        /* Sample from last position. */
        int token = dec->draft_sample(dec->draft_model_ctx,
                                      draft_logits, temperature);
        tensor_free(draft_logits);

        draft_tokens[i] = token;
        full_seq[draft_seq_len] = token;
        draft_seq_len++;
    }

    double t_draft_end = now_ms();

    /* ------------------------------------------------------------------
     * 2. Verify phase: single target-model forward on prefix + K drafts.
     * ------------------------------------------------------------------ */
    double t_verify_start = now_ms();

    int verify_seq_len = prefix_len + K;
    Tensor* target_logits = dec->target_forward(dec->target_model_ctx,
                                                 full_seq, verify_seq_len);
    double t_verify_end = now_ms();

    if (!target_logits) {
        LOG_ERROR("target forward returned NULL");
        free(full_seq);
        free(draft_tokens);
        return NULL;
    }

    /* ------------------------------------------------------------------
     * 3. Accept / reject each draft token.
     *
     * For draft position i (0-based), the target model's prediction is at
     * logits row (prefix_len - 1 + i): the target predicts "next token
     * given everything up to position prefix_len + i - 1".
     *
     * We use a simple greedy acceptance criterion: accept if the target
     * model's argmax at that row equals the draft token.
     * ------------------------------------------------------------------ */
    int num_accepted = 0;
    int correction_token = -1;

    for (int i = 0; i < K; i++) {
        int target_row = prefix_len - 1 + i;
        int target_argmax = argmax_at_row(target_logits, target_row,
                                          dec->vocab_size);

        if (target_argmax == draft_tokens[i]) {
            num_accepted++;
        } else {
            /* First mismatch: use target's prediction as correction. */
            correction_token = target_argmax;
            break;
        }
    }

    /* If all K draft tokens accepted, sample one bonus token from the
     * target logits at the last position. */
    int bonus_token = -1;
    if (num_accepted == K) {
        int last_row = prefix_len + K - 1;
        bonus_token = argmax_at_row(target_logits, last_row, dec->vocab_size);
    }

    tensor_free(target_logits);

    /* ------------------------------------------------------------------
     * 4. Build the result.
     * ------------------------------------------------------------------ */
    int total_output = num_accepted + ((num_accepted == K) ? 1 : 1);
    /* accepted draft tokens + either correction or bonus */

    CMLSpeculativeResult* result = (CMLSpeculativeResult*)calloc(1, sizeof(CMLSpeculativeResult));
    if (!result) {
        free(full_seq);
        free(draft_tokens);
        return NULL;
    }

    result->accepted_tokens = (int*)malloc((size_t)total_output * sizeof(int));
    if (!result->accepted_tokens) {
        free(result);
        free(full_seq);
        free(draft_tokens);
        return NULL;
    }

    /* Copy accepted draft tokens. */
    for (int i = 0; i < num_accepted; i++) {
        result->accepted_tokens[i] = draft_tokens[i];
    }

    /* Append correction or bonus token. */
    if (num_accepted == K) {
        result->accepted_tokens[num_accepted] = bonus_token;
        result->num_accepted = num_accepted + 1;  /* K drafts + 1 bonus */
    } else {
        result->accepted_tokens[num_accepted] = correction_token;
        result->num_accepted = num_accepted + 1;  /* accepted drafts + correction */
    }

    result->num_drafted  = K;
    result->num_verified = 1;
    result->acceptance_rate = (K > 0) ? (float)num_accepted / (float)K : 0.0f;
    result->draft_time_ms   = t_draft_end - t_draft_start;
    result->verify_time_ms  = t_verify_end - t_verify_start;
    result->total_time_ms   = t_verify_end - t_draft_start;

    /* Update decoder lifetime statistics. */
    dec->total_drafted  += (size_t)K;
    dec->total_accepted += (size_t)num_accepted;
    dec->total_steps++;

    free(full_seq);
    free(draft_tokens);

    LOG_DEBUG("speculative step: drafted=%d accepted=%d rate=%.2f",
              K, num_accepted, result->acceptance_rate);

    return result;
}

void cml_speculative_result_free(CMLSpeculativeResult* result) {
    if (!result) return;
    free(result->accepted_tokens);
    free(result);
}

float cml_speculative_acceptance_rate(const CMLSpeculativeDecoder* dec) {
    if (!dec || dec->total_drafted == 0) return 0.0f;
    return (float)dec->total_accepted / (float)dec->total_drafted;
}
