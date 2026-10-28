/*
 * End-to-end tests driving a real (tiny, random-weight) LLaMA model through the full generation,
 * serving, speculative-decoding and quantization paths. Where the unit tests check one op in
 * isolation, these exercise the whole pipeline: forward -> sample -> append -> stop, continuous
 * batching, prompt-lookup drafting, and the sliding-window KV cache inside the model loop.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "nn/llama.h"
#include "nn/serving.h"
#include "nn/speculative.h"
#include "nn/sampling.h"
#include "nn/quantization.h"
#include "nn/llm_ops.h"
#include "core/threefry.h"
#include "alloc/cml_allocator.h"
#include "test_harness.h"
#include "llama_test_model.h"

static int test_pipeline_runs(void) {
    CMLLLaMAModel* m = build_tiny_model();
    if (!m)
        return 0;
    int prompt[]          = {1, 2, 3};
    int out[8]            = {0};
    CMLGenerationConfig c = tiny_greedy_config();
    int n                 = tiny_decode(m, prompt, 3, &c, 8, out, NULL);
    int ok                = (n == 8);
    for (int i = 0; i < n; i++)
        if (out[i] < 0 || out[i] >= m->config.vocab_size)
            ok = 0;
    cml_llama_free(m);
    return ok;
}

static int test_greedy_deterministic(void) {
    CMLLLaMAModel* m = build_tiny_model();
    if (!m)
        return 0;
    int prompt[] = {5, 6};
    int a[6] = {0}, b[6] = {0};
    CMLGenerationConfig c = tiny_greedy_config();
    int na                = tiny_decode(m, prompt, 2, &c, 6, a, NULL);
    int nb                = tiny_decode(m, prompt, 2, &c, 6, b, NULL);
    int ok                = (na == nb && na == 6);
    for (int i = 0; i < na && ok; i++)
        if (a[i] != b[i])
            ok = 0;
    cml_llama_free(m);
    return ok;
}

static int test_seed_reproducible(void) {
    CMLLLaMAModel* m = build_tiny_model();
    if (!m)
        return 0;
    int prompt[] = {2, 4, 6};
    int a[6] = {0}, b[6] = {0};
    CMLGenerationConfig c = cml_generation_default_config();
    c.do_sample           = true;
    c.temperature         = 1.0f;

    /* Realize the lazily-materialized random weights first: they draw from the same global RNG, so
     * leaving it to the first seeded run would desync its sampling draws from the second. */
    warmup_tiny_model(m, prompt, 3);

    cml_rng_set_global_seed(777);
    int na = tiny_decode(m, prompt, 3, &c, 6, a, NULL);
    cml_rng_set_global_seed(777);
    int nb = tiny_decode(m, prompt, 3, &c, 6, b, NULL);

    int ok = (na == nb && na == 6);
    for (int i = 0; i < na && ok; i++)
        if (a[i] != b[i])
            ok = 0;
    cml_llama_free(m);
    return ok;
}

static int test_logprobs_valid(void) {
    CMLLLaMAModel* m = build_tiny_model();
    if (!m)
        return 0;
    int prompt[]          = {1, 1, 2};
    int out[5]            = {0};
    float lp[5]           = {0};
    CMLGenerationConfig c = tiny_greedy_config();
    int n                 = tiny_decode(m, prompt, 3, &c, 5, out, lp);
    int ok                = (n == 5);
    /* A log-probability is non-positive and finite. */
    for (int i = 0; i < n && ok; i++)
        if (!(lp[i] <= 1e-4f) || !isfinite(lp[i]))
            ok = 0;
    cml_llama_free(m);
    return ok;
}

static int test_logit_bias_forces_token(void) {
    CMLLLaMAModel* m = build_tiny_model();
    if (!m)
        return 0;
    int prompt[]          = {3, 1};
    int out[5]            = {0};
    int ids[1]            = {7};
    float bias[1]         = {100.0f}; /* dominate the distribution so 7 is always sampled */
    CMLGenerationConfig c = cml_generation_default_config();
    c.do_sample           = true;
    c.temperature         = 1.0f;
    c.top_k               = 0;
    c.top_p               = 1.0f;
    c.logit_bias_ids      = ids;
    c.logit_bias_values   = bias;
    c.logit_bias_count    = 1;
    int n                 = tiny_decode(m, prompt, 2, &c, 5, out, NULL);
    int ok                = (n == 5);
    for (int i = 0; i < n && ok; i++)
        if (out[i] != 7)
            ok = 0;
    cml_llama_free(m);
    return ok;
}

static int test_kv_window_in_model_loop(void) {
    CMLLLaMAModel* m = build_tiny_model();
    if (!m)
        return 0;
    int window = 8;
    for (int i = 0; i < m->num_layers; i++)
        cml_kv_cache_set_window(m->layers[i]->kv_cache, window, 0);

    int prompt[]          = {1, 2, 3, 4};
    int out[16]           = {0};
    CMLGenerationConfig c = tiny_greedy_config();
    int n                 = tiny_decode(m, prompt, 4, &c, 16, out, NULL);

    /* The cache must stay bounded by the window and the forward must keep succeeding. */
    int ok = (n > 0);
    for (int i = 0; i < m->num_layers && ok; i++)
        if (m->layers[i]->kv_cache->current_len > window)
            ok = 0;
    cml_llama_free(m);
    return ok;
}

/**
 * Serving forward adapter: prefill on position 0 (reset first), otherwise continue, writing the
 * last row's logits out.
 * @param model (void*)
 * @param tokens (const int*)
 * @param num_tokens (int)
 * @param position (int)
 * @param logits_out (float*)
 * @param vocab (int)
 * @return (int) 0 on success, -1 on failure
 */
static int e2e_serving_forward(void* model, const int* tokens, int num_tokens, int position,
                               float* logits_out, int vocab) {
    CMLLLaMAModel* m = (CMLLLaMAModel*)model;
    if (position == 0)
        cml_llama_reset(m);
    Tensor* logits = cml_llama_forward(m, tokens, num_tokens);
    if (!logits)
        return -1;
    tensor_ensure_executed(logits);
    float* data = (float*)tensor_data_ptr(logits);
    if (!data) {
        tensor_free(logits);
        return -1;
    }
    int last = (logits->ndim == 2) ? (logits->shape[0] - 1) : 0;
    memcpy(logits_out, data + (size_t)last * vocab, (size_t)vocab * sizeof(float));
    tensor_free(logits);
    return 0;
}

static int test_serving_e2e(void) {
    CMLLLaMAModel* m = build_tiny_model();
    if (!m)
        return 0;
    CMLServingConfig scfg  = cml_serving_default_config();
    scfg.max_seq_len       = m->config.max_seq_len;
    CMLServingContext* ctx = cml_serving_create(&scfg);
    if (!ctx) {
        cml_llama_free(m);
        return 0;
    }
    cml_serving_set_model(ctx, e2e_serving_forward, m, m->config.vocab_size, -1);

    int prompt[] = {1, 2, 3};
    int id       = cml_serving_submit(ctx, prompt, 3, 6);
    int cancel   = cml_serving_submit(ctx, prompt, 3, 6);
    int ok       = (id >= 0 && cancel >= 0);

    /* Cancel the second request before it runs, then drive the first to completion. */
    ok = ok && (cml_serving_cancel(ctx, cancel) == 0);
    for (int step = 0; step < 32 && cml_serving_get_status(ctx, id) != CML_SEQ_STATUS_FINISHED;
         step++)
        cml_serving_step(ctx);

    int count         = 0;
    const int* tokens = cml_serving_get_tokens(ctx, id, &count);
    ok                = ok && (tokens != NULL) && (count > 0) && (count <= 6);
    for (int i = 0; i < count && ok; i++)
        if (tokens[i] < 0 || tokens[i] >= m->config.vocab_size)
            ok = 0;

    cml_serving_free(ctx);
    cml_llama_free(m);
    return ok;
}

static int test_speculative_matches_greedy(void) {
    CMLLLaMAModel* m = build_tiny_model();
    if (!m)
        return 0;

    int prompt[] = {4, 5, 6, 7};

    /* Plain greedy next token after the prompt. */
    cml_llama_reset(m);
    Tensor* logits         = cml_llama_forward(m, prompt, 4);
    CMLGenerationConfig gc = tiny_greedy_config();
    int greedy_next        = cml_llama_sample_token(logits, &gc, NULL, 0, NULL);
    tensor_free(logits);

    /* Prompt-lookup speculative decode with the model as the (greedy) target. */
    CMLSpeculativeConfig cfg   = cml_speculative_default_config();
    cfg.do_sample              = false;
    CMLSpeculativeDecoder* dec = cml_speculative_create(&cfg, m->config.vocab_size);
    if (!dec) {
        cml_llama_free(m);
        return 0;
    }
    cml_speculative_set_ngram_draft(dec, 2);
    cml_speculative_set_target_model(dec, m, tiny_spec_forward, tiny_spec_argmax);

    CMLSpeculativeResult* res = cml_speculative_decode_step(dec, prompt, 4);
    int ok = (res != NULL) && (res->num_accepted >= 1) && (res->accepted_tokens[0] == greedy_next);

    if (res)
        cml_speculative_result_free(res);
    cml_speculative_free(dec);
    cml_llama_free(m);
    return ok;
}

int main(void) {
    printf("test_e2e\n\n");
    srand(1234);

    TEST(pipeline_runs);
    TEST(greedy_deterministic);
    TEST(seed_reproducible);
    TEST(logprobs_valid);
    TEST(logit_bias_forces_token);
    TEST(kv_window_in_model_loop);
    TEST(serving_e2e);
    TEST(speculative_matches_greedy);

    return TEST_SUMMARY();
}
