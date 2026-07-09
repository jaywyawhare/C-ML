#include "nn/serving.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "alloc/cml_allocator.h"

static double serving_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static CMLSequenceRequest* find_request(CMLServingContext* ctx, int request_id) {
    if (!ctx) return NULL;

    /* Search the circular queue */
    for (int i = 0; i < ctx->queue_count; i++) {
        int idx = (ctx->queue_head + i) % ctx->queue_capacity;
        if (ctx->queue[idx] && ctx->queue[idx]->request_id == request_id) {
            return ctx->queue[idx];
        }
    }

    /* Search the active batch */
    for (int i = 0; i < ctx->batch_size; i++) {
        if (ctx->active_batch[i] && ctx->active_batch[i]->request_id == request_id) {
            return ctx->active_batch[i];
        }
    }

    return NULL;
}

static void free_request(CMLSequenceRequest* req) {
    if (!req) return;
    cml_free(req->prompt_tokens);
    cml_free(req->generated_tokens);
    cml_free(req);
}

CMLServingConfig cml_serving_default_config(void) {
    CMLServingConfig config = {
        .max_batch_size       = 8,
        .max_queue_size       = 256,
        .max_seq_len          = 2048,
        .max_new_tokens_default = 256,
        .temperature_default  = 0.8f,
        .top_p_default        = 0.9f,
    };
    return config;
}

CMLServingContext* cml_serving_create(const CMLServingConfig* config) {
    if (!config) {
        LOG_ERROR("cml_serving_create: NULL config");
        return NULL;
    }

    CMLServingContext* ctx = (CMLServingContext*)cml_calloc(1, sizeof(CMLServingContext));
    if (!ctx) {
        LOG_ERROR("cml_serving_create: allocation failed");
        return NULL;
    }

    ctx->config = *config;

    /* Clamp to hard maximums */
    if (ctx->config.max_batch_size <= 0)
        ctx->config.max_batch_size = 8;
    if (ctx->config.max_batch_size > CML_SERVING_MAX_BATCH)
        ctx->config.max_batch_size = CML_SERVING_MAX_BATCH;

    if (ctx->config.max_queue_size <= 0)
        ctx->config.max_queue_size = 256;
    if (ctx->config.max_queue_size > CML_SERVING_MAX_QUEUE)
        ctx->config.max_queue_size = CML_SERVING_MAX_QUEUE;

    /* Allocate circular queue */
    ctx->queue_capacity = ctx->config.max_queue_size;
    ctx->queue = (CMLSequenceRequest**)cml_calloc((size_t)ctx->queue_capacity,
                                              sizeof(CMLSequenceRequest*));
    if (!ctx->queue) {
        LOG_ERROR("cml_serving_create: queue allocation failed");
        cml_free(ctx);
        return NULL;
    }
    ctx->queue_head = 0;
    ctx->queue_tail = 0;
    ctx->queue_count = 0;

    /* Allocate active batch array */
    ctx->active_batch = (CMLSequenceRequest**)cml_calloc((size_t)ctx->config.max_batch_size,
                                                     sizeof(CMLSequenceRequest*));
    if (!ctx->active_batch) {
        LOG_ERROR("cml_serving_create: active batch allocation failed");
        cml_free(ctx->queue);
        cml_free(ctx);
        return NULL;
    }
    ctx->batch_size = 0;

    /* Stats start at zero (calloc) */
    ctx->next_request_id = 1;
    ctx->kv_cache = NULL;

    LOG_INFO("Serving context created: max_batch=%d, max_queue=%d",
             ctx->config.max_batch_size, ctx->config.max_queue_size);
    return ctx;
}

void cml_serving_free(CMLServingContext* ctx) {
    if (!ctx) return;

    /* Free all queued requests */
    for (int i = 0; i < ctx->queue_count; i++) {
        int idx = (ctx->queue_head + i) % ctx->queue_capacity;
        free_request(ctx->queue[idx]);
        ctx->queue[idx] = NULL;
    }
    cml_free(ctx->queue);

    /* Free all active requests */
    for (int i = 0; i < ctx->batch_size; i++) {
        free_request(ctx->active_batch[i]);
        ctx->active_batch[i] = NULL;
    }
    cml_free(ctx->active_batch);

    LOG_INFO("Serving context freed (total_requests=%zu, completed=%zu)",
             ctx->stats.total_requests, ctx->stats.completed_requests);
    cml_free(ctx);
}

void cml_serving_set_kv_cache(CMLServingContext* ctx, CMLPagedKVCache* cache) {
    if (!ctx) return;
    ctx->kv_cache = cache;
    LOG_INFO("Paged KV cache set on serving context");
}

void cml_serving_set_model(CMLServingContext* ctx, CMLServingForwardFn forward_fn,
                           void* model, int vocab_size, int eos_token_id) {
    if (!ctx) return;
    ctx->forward_fn    = forward_fn;
    ctx->model         = model;
    ctx->vocab_size    = vocab_size;
    ctx->eos_token_id  = eos_token_id;
    LOG_INFO("Serving model attached (vocab=%d, eos=%d)", vocab_size, eos_token_id);
}

/* xorshift RNG for stochastic sampling (per-context determinism not required;
 * greedy decoding — temperature <= 0 — is fully deterministic). */
static uint32_t serving_rng_state = 0x2545F491u;
static float serving_rand_uniform(void) {
    uint32_t x = serving_rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    serving_rng_state = x;
    return (float)(x & 0xffffff) / (float)0x1000000;
}

static int serving_argmax(const float* logits, int n) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

/* Sample a token id from logits. temperature <= 0 => greedy argmax; otherwise
 * temperature scaling + nucleus (top-p) sampling. */
static int serving_sample(const float* logits, int vocab, float temperature, float top_p) {
    if (temperature <= 0.0f || vocab <= 1)
        return serving_argmax(logits, vocab);

    /* softmax with temperature (numerically stable) */
    float* probs = (float*)cml_malloc((size_t)vocab * sizeof(float));
    if (!probs) return serving_argmax(logits, vocab);

    float maxl = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > maxl) maxl = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        float p = expf((logits[i] - maxl) / temperature);
        probs[i] = p; sum += p;
    }
    float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    for (int i = 0; i < vocab; i++) probs[i] *= inv;

    /* nucleus: build a simple descending cutoff by cumulative mass. We avoid a
     * full sort by iterating with a running threshold: draw against the top-p
     * renormalised distribution over tokens whose cumulative prob (in descending
     * order) is within top_p. Approximate via a threshold pass. */
    if (top_p > 0.0f && top_p < 1.0f) {
        /* find the probability threshold t such that mass(prob >= t) ~ top_p */
        float lo = 0.0f, hi = 1.0f, thresh = 0.0f;
        for (int it = 0; it < 24; it++) {
            float mid = 0.5f * (lo + hi);
            float mass = 0.0f;
            for (int i = 0; i < vocab; i++) if (probs[i] >= mid) mass += probs[i];
            if (mass > top_p) lo = mid; else hi = mid;
            thresh = mid;
        }
        float kept = 0.0f;
        for (int i = 0; i < vocab; i++) { if (probs[i] < thresh) probs[i] = 0.0f; else kept += probs[i]; }
        if (kept > 0.0f) { float k = 1.0f / kept; for (int i = 0; i < vocab; i++) probs[i] *= k; }
    }

    float r = serving_rand_uniform();
    float cum = 0.0f;
    int chosen = vocab - 1;
    for (int i = 0; i < vocab; i++) { cum += probs[i]; if (r <= cum) { chosen = i; break; } }
    cml_free(probs);
    return chosen;
}

/* Generate one token for an active DECODING request. Returns 0 on success. */
static int serving_generate_one(CMLServingContext* ctx, CMLSequenceRequest* req) {
    float* logits = (float*)cml_malloc((size_t)ctx->vocab_size * sizeof(float));
    if (!logits) { LOG_ERROR("serving_generate_one: logits alloc failed"); return -1; }

    int rc;
    if (req->num_generated == 0) {
        /* prefill: run the whole prompt, logits are for the next token */
        rc = ctx->forward_fn(ctx->model, req->prompt_tokens, req->num_prompt_tokens,
                             0, logits, ctx->vocab_size);
        req->current_pos = req->num_prompt_tokens;
        if (req->first_token_time_ms == 0.0) req->first_token_time_ms = serving_time_ms();
    } else {
        int last = req->generated_tokens[req->num_generated - 1];
        rc = ctx->forward_fn(ctx->model, &last, 1, req->current_pos, logits, ctx->vocab_size);
        req->current_pos++;
    }

    if (rc != 0) {
        cml_free(logits);
        req->status = CML_SEQ_STATUS_ERROR;
        LOG_ERROR("serving_generate_one: model forward failed for request %d", req->request_id);
        return -1;
    }

    int tok = serving_sample(logits, ctx->vocab_size, req->temperature, req->top_p);
    cml_free(logits);

    /* grow the generated-token buffer if needed */
    if (req->num_generated >= req->gen_capacity) {
        int new_cap = req->gen_capacity > 0 ? req->gen_capacity * 2 : 16;
        int* grown = (int*)realloc(req->generated_tokens, (size_t)new_cap * sizeof(int));
        if (!grown) { LOG_ERROR("serving_generate_one: token buffer grow failed"); return -1; }
        req->generated_tokens = grown;
        req->gen_capacity = new_cap;
    }
    req->generated_tokens[req->num_generated++] = tok;

    if ((ctx->eos_token_id >= 0 && tok == ctx->eos_token_id) ||
        req->num_generated >= req->max_new_tokens) {
        req->status = CML_SEQ_STATUS_FINISHED;
        if (req->finish_time_ms == 0.0) req->finish_time_ms = serving_time_ms();
    }
    return 0;
}

int cml_serving_submit(CMLServingContext* ctx, const int* prompt_tokens,
                       int num_tokens, int max_new_tokens) {
    if (!ctx) {
        LOG_ERROR("cml_serving_submit: NULL context");
        return -1;
    }
    if (!prompt_tokens || num_tokens <= 0) {
        LOG_ERROR("cml_serving_submit: invalid prompt (tokens=%p, n=%d)",
                  (const void*)prompt_tokens, num_tokens);
        return -1;
    }

    /* Check queue capacity */
    if (ctx->queue_count >= ctx->queue_capacity) {
        LOG_WARNING("cml_serving_submit: queue full (%d/%d)",
                    ctx->queue_count, ctx->queue_capacity);
        return -1;
    }

    /* Allocate request */
    CMLSequenceRequest* req = (CMLSequenceRequest*)cml_calloc(1, sizeof(CMLSequenceRequest));
    if (!req) {
        LOG_ERROR("cml_serving_submit: request allocation failed");
        return -1;
    }

    req->request_id = ctx->next_request_id++;
    req->num_prompt_tokens = num_tokens;
    req->prompt_tokens = (int*)cml_malloc((size_t)num_tokens * sizeof(int));
    if (!req->prompt_tokens) {
        LOG_ERROR("cml_serving_submit: token copy allocation failed");
        cml_free(req);
        return -1;
    }
    memcpy(req->prompt_tokens, prompt_tokens, (size_t)num_tokens * sizeof(int));

    req->max_new_tokens = (max_new_tokens > 0)
                          ? max_new_tokens
                          : ctx->config.max_new_tokens_default;
    req->temperature = ctx->config.temperature_default;
    req->top_p = ctx->config.top_p_default;
    req->status = CML_SEQ_STATUS_QUEUED;
    req->paged_seq_id = -1;
    req->current_pos = 0;

    /* Pre-allocate generated token buffer */
    req->gen_capacity = req->max_new_tokens;
    req->generated_tokens = (int*)cml_calloc((size_t)req->gen_capacity, sizeof(int));
    if (!req->generated_tokens) {
        LOG_ERROR("cml_serving_submit: generated token buffer allocation failed");
        cml_free(req->prompt_tokens);
        cml_free(req);
        return -1;
    }
    req->num_generated = 0;

    req->submit_time_ms = serving_time_ms();
    req->first_token_time_ms = 0.0;
    req->finish_time_ms = 0.0;

    /* Enqueue (circular buffer) */
    ctx->queue[ctx->queue_tail] = req;
    ctx->queue_tail = (ctx->queue_tail + 1) % ctx->queue_capacity;
    ctx->queue_count++;
    ctx->stats.total_requests++;

    LOG_DEBUG("Request %d submitted (%d prompt tokens, max_new=%d)",
              req->request_id, num_tokens, req->max_new_tokens);
    return req->request_id;
}

int cml_serving_step(CMLServingContext* ctx) {
    if (!ctx) return 0;

    /* Admit queued requests into the active batch */
    while (ctx->queue_count > 0 && ctx->batch_size < ctx->config.max_batch_size) {
        CMLSequenceRequest* req = ctx->queue[ctx->queue_head];
        ctx->queue[ctx->queue_head] = NULL;
        ctx->queue_head = (ctx->queue_head + 1) % ctx->queue_capacity;
        ctx->queue_count--;

        /* Transition: QUEUED -> PREFILL */
        req->status = CML_SEQ_STATUS_PREFILL;

        /* Place in active batch */
        ctx->active_batch[ctx->batch_size] = req;
        ctx->batch_size++;
        ctx->stats.active_sequences++;

        LOG_DEBUG("Request %d admitted to batch (batch_size=%d)",
                  req->request_id, ctx->batch_size);
    }

    /* Transition any PREFILL requests to DECODING. */
    for (int i = 0; i < ctx->batch_size; i++) {
        CMLSequenceRequest* req = ctx->active_batch[i];
        if (req && req->status == CML_SEQ_STATUS_PREFILL)
            req->status = CML_SEQ_STATUS_DECODING;
    }

    /* If a model forward callback is attached, advance every active DECODING
     * request by one token (continuous batching: one step = one token/seq). */
    if (ctx->forward_fn && ctx->vocab_size > 0) {
        for (int i = 0; i < ctx->batch_size; i++) {
            CMLSequenceRequest* req = ctx->active_batch[i];
            if (req && req->status == CML_SEQ_STATUS_DECODING)
                serving_generate_one(ctx, req);
        }
    } else {
        /* No model attached: mark first-token time so bookkeeping stays sane. */
        for (int i = 0; i < ctx->batch_size; i++) {
            CMLSequenceRequest* req = ctx->active_batch[i];
            if (req && req->first_token_time_ms == 0.0)
                req->first_token_time_ms = serving_time_ms();
        }
    }

    return ctx->batch_size;
}

CMLSequenceStatus cml_serving_get_status(CMLServingContext* ctx, int request_id) {
    CMLSequenceRequest* req = find_request(ctx, request_id);
    if (!req) return CML_SEQ_STATUS_ERROR;
    return req->status;
}

const int* cml_serving_get_tokens(CMLServingContext* ctx, int request_id,
                                  int* out_count) {
    if (out_count) *out_count = 0;
    CMLSequenceRequest* req = find_request(ctx, request_id);
    if (!req) return NULL;
    if (out_count) *out_count = req->num_generated;
    return req->generated_tokens;
}

int cml_serving_finish_request(CMLServingContext* ctx, int request_id) {
    if (!ctx) return -1;

    /* Search active batch for this request */
    int found_idx = -1;
    for (int i = 0; i < ctx->batch_size; i++) {
        if (ctx->active_batch[i] &&
            ctx->active_batch[i]->request_id == request_id) {
            found_idx = i;
            break;
        }
    }

    if (found_idx < 0) {
        LOG_WARNING("cml_serving_finish_request: request %d not in active batch",
                    request_id);
        return -1;
    }

    CMLSequenceRequest* req = ctx->active_batch[found_idx];
    req->status = CML_SEQ_STATUS_FINISHED;
    req->finish_time_ms = serving_time_ms();

    /* Update stats */
    ctx->stats.completed_requests++;
    ctx->stats.total_tokens_generated += (size_t)req->num_generated;
    if (ctx->stats.active_sequences > 0)
        ctx->stats.active_sequences--;

    /* Compute timing stats */
    double request_time_ms = req->finish_time_ms - req->submit_time_ms;
    ctx->stats.total_time_ms += request_time_ms;

    if (req->first_token_time_ms > 0.0) {
        double ttft = req->first_token_time_ms - req->submit_time_ms;
        /* Running average of time-to-first-token */
        size_t n = ctx->stats.completed_requests;
        ctx->stats.avg_time_to_first_token_ms =
            ((ctx->stats.avg_time_to_first_token_ms * (double)(n - 1)) + ttft) / (double)n;
    }

    if (ctx->stats.total_time_ms > 0.0) {
        ctx->stats.avg_tokens_per_second =
            (double)ctx->stats.total_tokens_generated /
            (ctx->stats.total_time_ms / 1000.0);
    }

    LOG_DEBUG("Request %d finished (%d tokens, %.1f ms)",
              request_id, req->num_generated, req->finish_time_ms - req->submit_time_ms);

    /* Remove from active batch: swap with last element */
    free_request(req);
    ctx->active_batch[found_idx] = ctx->active_batch[ctx->batch_size - 1];
    ctx->active_batch[ctx->batch_size - 1] = NULL;
    ctx->batch_size--;

    return 0;
}

CMLServingStats cml_serving_get_stats(const CMLServingContext* ctx) {
    if (!ctx) {
        CMLServingStats empty = {0};
        return empty;
    }
    return ctx->stats;
}
