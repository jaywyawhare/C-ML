#include "nn/speculative.h"
#include "core/logging.h"
#include "tensor/tensor.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "alloc/cml_allocator.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

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

CMLSpeculativeConfig cml_speculative_default_config(void) {
    CMLSpeculativeConfig cfg;
    cfg.num_draft_tokens  = 5;
    cfg.temperature       = 0.8f;
    cfg.top_p             = 0.9f;
    cfg.top_k             = 40;
    cfg.do_sample         = true;
    cfg.stochastic_accept = false;
    return cfg;
}

/* Temperature softmax of one logits row into probs[vocab]. */
static void softmax_row(const float* row, int vocab, float temperature, float* probs) {
    float t = temperature > 1e-6f ? temperature : 1.0f;
    float maxl = row[0];
    for (int i = 1; i < vocab; i++)
        if (row[i] > maxl) maxl = row[i];
    double denom = 0.0;
    for (int i = 0; i < vocab; i++) {
        probs[i] = expf((row[i] - maxl) / t);
        denom += probs[i];
    }
    float inv = denom > 0.0 ? (float)(1.0 / denom) : 0.0f;
    for (int i = 0; i < vocab; i++)
        probs[i] *= inv;
}

static int sample_from_probs(const float* probs, int vocab) {
    float u = (float)rand() / ((float)RAND_MAX + 1.0f);
    float acc = 0.0f;
    for (int i = 0; i < vocab; i++) {
        acc += probs[i];
        if (u < acc) return i;
    }
    return vocab - 1;
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

    CMLSpeculativeDecoder* dec = (CMLSpeculativeDecoder*)cml_calloc(1, sizeof(CMLSpeculativeDecoder));
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
    cml_free(decoder);
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
    int* full_seq = (int*)cml_malloc((size_t)max_seq * sizeof(int));
    if (!full_seq) {
        LOG_ERROR("cml_speculative_decode_step: allocation failed");
        return NULL;
    }
    memcpy(full_seq, prefix_tokens, (size_t)prefix_len * sizeof(int));

    int* draft_tokens = (int*)cml_malloc((size_t)K * sizeof(int));
    if (!draft_tokens) {
        cml_free(full_seq);
        LOG_ERROR("cml_speculative_decode_step: allocation failed");
        return NULL;
    }

    /* Stochastic verification needs the draft distribution at each drafted
     * position; keep the last-row logits per draft step. */
    bool stochastic = dec->config.stochastic_accept && dec->config.do_sample;
    float* draft_rows = NULL;
    if (stochastic) {
        draft_rows = (float*)cml_malloc((size_t)K * (size_t)dec->vocab_size * sizeof(float));
        if (!draft_rows) stochastic = false; /* degrade to greedy acceptance */
    }

    /* 1. Draft phase: autoregressively generate K tokens with draft model. */
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

        if (stochastic) {
            tensor_ensure_executed(draft_logits);
            const float* d = (const float*)tensor_data_ptr(draft_logits);
            if (d) {
                memcpy(draft_rows + (size_t)i * dec->vocab_size,
                       d + (size_t)(draft_seq_len - 1) * dec->vocab_size,
                       (size_t)dec->vocab_size * sizeof(float));
            } else {
                stochastic = false;
            }
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

    /* 2. Verify phase: single target-model forward on prefix + K drafts. */
    double t_verify_start = now_ms();

    int verify_seq_len = prefix_len + K;
    Tensor* target_logits = dec->target_forward(dec->target_model_ctx,
                                                 full_seq, verify_seq_len);
    double t_verify_end = now_ms();

    if (!target_logits) {
        LOG_ERROR("target forward returned NULL");
        cml_free(full_seq);
        cml_free(draft_tokens);
        cml_free(draft_rows);
        return NULL;
    }

    /* 3. Accept / reject each draft token.
     *
     * For draft position i (0-based), the target model's prediction is at
     * logits row (prefix_len - 1 + i): the target predicts "next token
     * given everything up to position prefix_len + i - 1".
     *
     * Greedy mode: accept iff the target argmax equals the draft token —
     * the output is then exactly the target model's greedy decode.
     * Stochastic mode (config.stochastic_accept): Leviathan et al. rule —
     * accept with prob min(1, p/q), resample rejects from norm(max(0,p-q)). */
    int num_accepted = 0;
    int correction_token = -1;
    int bonus_token = -1;

    if (stochastic) {
        tensor_ensure_executed(target_logits);
        const float* tdata = (const float*)tensor_data_ptr(target_logits);
        int vocab = dec->vocab_size;
        float* p = (float*)cml_malloc((size_t)vocab * sizeof(float));
        float* q = (float*)cml_malloc((size_t)vocab * sizeof(float));
        if (!tdata || !p || !q) {
            cml_free(p); cml_free(q);
            tensor_free(target_logits);
            cml_free(full_seq);
            cml_free(draft_tokens);
            cml_free(draft_rows);
            return NULL;
        }

        for (int i = 0; i < K; i++) {
            int target_row = prefix_len - 1 + i;
            softmax_row(tdata + (size_t)target_row * vocab, vocab, temperature, p);
            softmax_row(draft_rows + (size_t)i * vocab, vocab, temperature, q);

            int tok = draft_tokens[i];
            float qd = q[tok] > 1e-12f ? q[tok] : 1e-12f;
            float ratio = p[tok] / qd;
            float u = (float)rand() / ((float)RAND_MAX + 1.0f);
            if (u < ratio) {
                num_accepted++;
                continue;
            }
            /* Reject: resample from the normalized residual max(0, p - q). */
            double mass = 0.0;
            for (int j = 0; j < vocab; j++) {
                p[j] = p[j] > q[j] ? p[j] - q[j] : 0.0f;
                mass += p[j];
            }
            if (mass > 1e-12) {
                float inv = (float)(1.0 / mass);
                for (int j = 0; j < vocab; j++) p[j] *= inv;
                correction_token = sample_from_probs(p, vocab);
            } else {
                /* p <= q everywhere it matters: fall back to target argmax. */
                correction_token = argmax_at_row(target_logits, target_row, vocab);
            }
            break;
        }

        if (num_accepted == K) {
            int last_row = prefix_len + K - 1;
            softmax_row(tdata + (size_t)last_row * vocab, vocab, temperature, p);
            bonus_token = sample_from_probs(p, vocab);
        }
        cml_free(p);
        cml_free(q);
    } else {
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

        /* If all K draft tokens accepted, take the bonus token from the
         * target logits at the last position. */
        if (num_accepted == K) {
            int last_row = prefix_len + K - 1;
            bonus_token = argmax_at_row(target_logits, last_row, dec->vocab_size);
        }
    }

    cml_free(draft_rows);
    tensor_free(target_logits);

    /* 4. Build the result. */
    int total_output = num_accepted + ((num_accepted == K) ? 1 : 1);
    /* accepted draft tokens + either correction or bonus */

    CMLSpeculativeResult* result = (CMLSpeculativeResult*)cml_calloc(1, sizeof(CMLSpeculativeResult));
    if (!result) {
        cml_free(full_seq);
        cml_free(draft_tokens);
        return NULL;
    }

    result->accepted_tokens = (int*)cml_malloc((size_t)total_output * sizeof(int));
    if (!result->accepted_tokens) {
        cml_free(result);
        cml_free(full_seq);
        cml_free(draft_tokens);
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

    cml_free(full_seq);
    cml_free(draft_tokens);

    LOG_DEBUG("speculative step: drafted=%d accepted=%d rate=%.2f",
              K, num_accepted, result->acceptance_rate);

    return result;
}

void cml_speculative_result_free(CMLSpeculativeResult* result) {
    if (!result) return;
    cml_free(result->accepted_tokens);
    cml_free(result);
}

float cml_speculative_acceptance_rate(const CMLSpeculativeDecoder* dec) {
    if (!dec || dec->total_drafted == 0) return 0.0f;
    return (float)dec->total_accepted / (float)dec->total_drafted;
}
