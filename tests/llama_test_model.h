/*
 * Shared tiny random LLaMA model scaffolding for the e2e / A/B / profiling harnesses and the
 * unit tests. One source of truth for "spin up a small, runnable model with random weights" so the
 * timing and correctness harnesses all measure the identical architecture.
 */
#ifndef CML_LLAMA_TEST_MODEL_H
#define CML_LLAMA_TEST_MODEL_H

#include "cml.h"
#include "nn/llama.h"
#include "tensor/tensor.h"
#include "alloc/cml_allocator.h"
#include <string.h>

/**
 * A small LLaMA config (vocab 64, hidden 32, 2 layers) cheap enough to run many times per test.
 * @return (CMLLLaMAConfig)
 */
static inline CMLLLaMAConfig tiny_test_config(void) {
    CMLLLaMAConfig config = {.vocab_size        = 64,
                             .hidden_size       = 32,
                             .intermediate_size = 64,
                             .num_layers        = 2,
                             .num_heads         = 4,
                             .num_kv_heads      = 4,
                             .max_seq_len       = 32,
                             .rope_theta        = 10000.0f,
                             .rms_norm_eps      = 1e-5f};
    return config;
}

/**
 * Fill one transformer layer's projections with random weights and norms with ones.
 * @param layer (CMLLLaMALayer*)
 * @param config (const CMLLLaMAConfig*)
 * @return (int) 0 on success, -1 on allocation failure
 */
static inline int init_layer_random(CMLLLaMALayer* layer, const CMLLLaMAConfig* config) {
    int hidden   = config->hidden_size;
    int inter    = config->intermediate_size;
    int head_dim = hidden / config->num_heads;
    int q_dim    = config->num_heads * head_dim;
    int kv_dim   = config->num_kv_heads * head_dim;

    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

    int q_shape[]    = {hidden, q_dim};
    int kv_shape[]   = {hidden, kv_dim};
    int o_shape[]    = {q_dim, hidden};
    int gate_shape[] = {hidden, inter};
    int down_shape[] = {inter, hidden};
    int norm_shape[] = {hidden};

    layer->q_proj              = tensor_rand(q_shape, 2, &cfg);
    layer->k_proj              = tensor_rand(kv_shape, 2, &cfg);
    layer->v_proj              = tensor_rand(kv_shape, 2, &cfg);
    layer->o_proj              = tensor_rand(o_shape, 2, &cfg);
    layer->gate_proj           = tensor_rand(gate_shape, 2, &cfg);
    layer->up_proj             = tensor_rand(gate_shape, 2, &cfg);
    layer->down_proj           = tensor_rand(down_shape, 2, &cfg);
    layer->input_layernorm     = tensor_ones(norm_shape, 1, &cfg);
    layer->post_attn_layernorm = tensor_ones(norm_shape, 1, &cfg);

    if (!layer->q_proj || !layer->k_proj || !layer->v_proj || !layer->o_proj || !layer->gate_proj ||
        !layer->up_proj || !layer->down_proj || !layer->input_layernorm ||
        !layer->post_attn_layernorm) {
        return -1;
    }
    return 0;
}

/**
 * Fill a model's embeddings, norm, lm_head and every layer with random weights.
 * @param model (CMLLLaMAModel*)
 * @return (int) 0 on success, -1 on allocation failure
 */
static inline int init_model_random(CMLLLaMAModel* model) {
    const CMLLLaMAConfig* config = &model->config;
    TensorConfig cfg             = {
                    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

    int embed_shape[]   = {config->vocab_size, config->hidden_size};
    int norm_shape[]    = {config->hidden_size};
    int lm_head_shape[] = {config->hidden_size, config->vocab_size};

    model->embed_tokens = tensor_rand(embed_shape, 2, &cfg);
    model->norm         = tensor_ones(norm_shape, 1, &cfg);
    model->lm_head      = tensor_rand(lm_head_shape, 2, &cfg);

    if (!model->embed_tokens || !model->norm || !model->lm_head)
        return -1;

    for (int i = 0; i < model->num_layers; i++)
        if (init_layer_random(model->layers[i], config) != 0)
            return -1;

    model->weights_loaded = true;
    return 0;
}

/**
 * Create and fully initialise a tiny random model, or NULL on failure.
 * @return (CMLLLaMAModel*)
 */
static inline CMLLLaMAModel* build_tiny_model(void) {
    CMLLLaMAConfig cfg   = tiny_test_config();
    CMLLLaMAModel* model = cml_llama_create(&cfg);
    if (!model)
        return NULL;
    if (init_model_random(model) != 0) {
        cml_llama_free(model);
        return NULL;
    }
    return model;
}

/**
 * Realize a model's lazily-materialized random weights with a throwaway forward, so later
 * RNG-seeded sampling isn't desynced by weight materialization drawing from the same global RNG.
 * @param m (CMLLLaMAModel*)
 * @return (void)
 */
static inline void warmup_tiny_model(CMLLLaMAModel* m, const int* prompt, int plen) {
    cml_llama_reset(m);
    Tensor* w = cml_llama_forward(m, prompt, plen);
    if (w)
        tensor_free(w);
}

/**
 * Run prefill + up to `steps` decode steps at the token level, collecting generated tokens (and
 * their logprobs when out_lp is non-NULL). Returns the token count, or -1 on error.
 * @param m (CMLLLaMAModel*)
 * @param prompt (const int*)
 * @param plen (int)
 * @param cfg (const CMLGenerationConfig*)
 * @param steps (int)
 * @param out (int*)
 * @param out_lp (float*)
 * @return (int)
 */
static inline int tiny_decode(CMLLLaMAModel* m, const int* prompt, int plen,
                              const CMLGenerationConfig* cfg, int steps, int* out, float* out_lp) {
    cml_llama_reset(m);
    Tensor* logits = cml_llama_forward(m, prompt, plen);
    if (!logits)
        return -1;

    int* hist = (int*)cml_malloc((size_t)(plen + steps) * sizeof(int));
    memcpy(hist, prompt, (size_t)plen * sizeof(int));
    int hlen = plen;

    int n = 0;
    for (int s = 0; s < steps; s++) {
        float lp = 0.0f;
        int tok  = cml_llama_sample_token(logits, cfg, hist, hlen, &lp);
        tensor_free(logits);
        logits = NULL;
        if (tok < 0)
            break;
        out[n] = tok;
        if (out_lp)
            out_lp[n] = lp;
        n++;
        hist[hlen++] = tok;
        if (m->current_seq_len >= m->config.max_seq_len)
            break;
        logits = cml_llama_forward(m, &tok, 1);
        if (!logits)
            break;
    }
    if (logits)
        tensor_free(logits);
    cml_free(hist);
    return n;
}

/**
 * Default generation config switched to greedy (argmax) decoding.
 * @return (CMLGenerationConfig)
 */
static inline CMLGenerationConfig tiny_greedy_config(void) {
    CMLGenerationConfig c = cml_generation_default_config();
    c.do_sample           = false;
    return c;
}

/**
 * Speculative target-forward adapter: reset and run a fresh forward over the whole sequence.
 * @param ctx (void*)
 * @param tokens (const int*)
 * @param seq_len (int)
 * @return (Tensor*)
 */
static inline Tensor* tiny_spec_forward(void* ctx, const int* tokens, int seq_len) {
    CMLLLaMAModel* m = (CMLLLaMAModel*)ctx;
    cml_llama_reset(m);
    return cml_llama_forward(m, tokens, seq_len);
}

/**
 * Speculative target-sample adapter: greedy argmax over the last logits row.
 * @param ctx (void*)
 * @param logits (Tensor*)
 * @param temperature (float)
 * @return (int)
 */
static inline int tiny_spec_argmax(void* ctx, Tensor* logits, float temperature) {
    (void)ctx;
    (void)temperature;
    tensor_ensure_executed(logits);
    float* data = (float*)tensor_data_ptr(logits);
    if (!data)
        return 0;
    int vocab  = logits->shape[logits->ndim - 1];
    int last   = (logits->ndim == 2) ? (logits->shape[0] - 1) : 0;
    float* row = data + (size_t)last * vocab;
    int best   = 0;
    for (int i = 1; i < vocab; i++)
        if (row[i] > row[best])
            best = i;
    return best;
}

#endif /* CML_LLAMA_TEST_MODEL_H */
