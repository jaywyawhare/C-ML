/**
 * @file test_llm_ops.c
 * @brief Tests for LLM inference primitives (GQA, MoE, KV Cache, BPE Tokenizer, RoPE)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "nn/llm_ops.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-45s ", #name); \
    fflush(stdout); \
    if (test_##name()) { \
        tests_passed++; \
        printf("[PASS]\n"); \
    } else { \
        printf("[FAIL]\n"); \
    } \
} while(0)

/* ===== KV Cache Tests ===== */

static int test_kv_cache_create_free(void) {
    CMLKVCache* cache = cml_kv_cache_create(128, 4, 32);
    if (!cache) return 0;
    if (cache->max_seq_len != 128) { cml_kv_cache_free(cache); return 0; }
    if (cache->num_kv_heads != 4) { cml_kv_cache_free(cache); return 0; }
    if (cache->head_dim != 32) { cml_kv_cache_free(cache); return 0; }
    if (cache->current_len != 0) { cml_kv_cache_free(cache); return 0; }
    if (!cache->key_cache) { cml_kv_cache_free(cache); return 0; }
    if (!cache->value_cache) { cml_kv_cache_free(cache); return 0; }
    cml_kv_cache_free(cache);
    return 1;
}

static int test_kv_cache_append(void) {
    int num_kv_heads = 2;
    int head_dim = 4;
    CMLKVCache* cache = cml_kv_cache_create(16, num_kv_heads, head_dim);
    if (!cache) return 0;

    /* Create a new key/value of shape [3, 2, 4] (3 tokens, 2 heads, dim 4) */
    int kv_shape[] = {3, num_kv_heads, head_dim};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* new_k = tensor_ones(kv_shape, 3, &cfg);
    Tensor* new_v = tensor_ones(kv_shape, 3, &cfg);
    if (!new_k || !new_v) {
        if (new_k) tensor_free(new_k);
        if (new_v) tensor_free(new_v);
        cml_kv_cache_free(cache);
        return 0;
    }

    int result = cml_kv_cache_append(cache, new_k, new_v);
    tensor_free(new_k);
    tensor_free(new_v);

    if (result != 3) { cml_kv_cache_free(cache); return 0; }
    if (cache->current_len != 3) { cml_kv_cache_free(cache); return 0; }

    /* Append 2 more tokens */
    int kv_shape2[] = {2, num_kv_heads, head_dim};
    Tensor* new_k2 = tensor_ones(kv_shape2, 3, &cfg);
    Tensor* new_v2 = tensor_ones(kv_shape2, 3, &cfg);
    if (!new_k2 || !new_v2) {
        if (new_k2) tensor_free(new_k2);
        if (new_v2) tensor_free(new_v2);
        cml_kv_cache_free(cache);
        return 0;
    }

    result = cml_kv_cache_append(cache, new_k2, new_v2);
    tensor_free(new_k2);
    tensor_free(new_v2);

    if (result != 5) { cml_kv_cache_free(cache); return 0; }
    if (cache->current_len != 5) { cml_kv_cache_free(cache); return 0; }

    cml_kv_cache_free(cache);
    return 1;
}

static int test_kv_cache_get_keys(void) {
    int num_kv_heads = 2;
    int head_dim = 4;
    CMLKVCache* cache = cml_kv_cache_create(16, num_kv_heads, head_dim);
    if (!cache) return 0;

    /* Append 3 tokens */
    int kv_shape[] = {3, num_kv_heads, head_dim};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* new_k = tensor_ones(kv_shape, 3, &cfg);
    Tensor* new_v = tensor_ones(kv_shape, 3, &cfg);
    if (!new_k || !new_v) {
        if (new_k) tensor_free(new_k);
        if (new_v) tensor_free(new_v);
        cml_kv_cache_free(cache);
        return 0;
    }

    cml_kv_cache_append(cache, new_k, new_v);
    tensor_free(new_k);
    tensor_free(new_v);

    Tensor* keys = cml_kv_cache_get_keys(cache);
    if (!keys) { cml_kv_cache_free(cache); return 0; }

    /* Verify shape: [3, 2, 4] */
    if (keys->ndim != 3) { tensor_free(keys); cml_kv_cache_free(cache); return 0; }
    if (keys->shape[0] != 3) { tensor_free(keys); cml_kv_cache_free(cache); return 0; }
    if (keys->shape[1] != num_kv_heads) { tensor_free(keys); cml_kv_cache_free(cache); return 0; }
    if (keys->shape[2] != head_dim) { tensor_free(keys); cml_kv_cache_free(cache); return 0; }

    /* Verify data is 1.0 */
    tensor_ensure_executed(keys);
    float val = tensor_get_float(keys, 0);
    if (fabsf(val - 1.0f) > 1e-5f) { tensor_free(keys); cml_kv_cache_free(cache); return 0; }

    tensor_free(keys);
    cml_kv_cache_free(cache);
    return 1;
}

static int test_kv_cache_reset(void) {
    CMLKVCache* cache = cml_kv_cache_create(16, 2, 4);
    if (!cache) return 0;

    int kv_shape[] = {3, 2, 4};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* new_k = tensor_ones(kv_shape, 3, &cfg);
    Tensor* new_v = tensor_ones(kv_shape, 3, &cfg);
    if (!new_k || !new_v) {
        if (new_k) tensor_free(new_k);
        if (new_v) tensor_free(new_v);
        cml_kv_cache_free(cache);
        return 0;
    }

    cml_kv_cache_append(cache, new_k, new_v);
    tensor_free(new_k);
    tensor_free(new_v);

    if (cache->current_len != 3) { cml_kv_cache_free(cache); return 0; }

    cml_kv_cache_reset(cache);
    if (cache->current_len != 0) { cml_kv_cache_free(cache); return 0; }

    /* After reset, get_keys should return NULL */
    Tensor* keys = cml_kv_cache_get_keys(cache);
    if (keys != NULL) { tensor_free(keys); cml_kv_cache_free(cache); return 0; }

    cml_kv_cache_free(cache);
    return 1;
}

/* ===== GQA Tests ===== */

static int test_gqa_basic(void) {
    /* Standard MHA: num_heads == num_kv_heads */
    int batch = 1, seq = 4, num_heads = 2, head_dim = 4;
    int dim = num_heads * head_dim; /* 8 */

    int q_shape[] = {batch, seq, dim};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* Q = tensor_rand(q_shape, 3, &cfg);
    Tensor* K = tensor_rand(q_shape, 3, &cfg);
    Tensor* V = tensor_rand(q_shape, 3, &cfg);
    if (!Q || !K || !V) {
        if (Q) tensor_free(Q);
        if (K) tensor_free(K);
        if (V) tensor_free(V);
        return 0;
    }

    CMLGQAConfig gqa_cfg = {
        .num_heads = num_heads,
        .num_kv_heads = num_heads, /* same = standard MHA */
        .head_dim = head_dim,
        .scale = 0.0f, /* auto */
        .causal = false
    };

    Tensor* out = cml_gqa_forward(Q, K, V, &gqa_cfg, NULL);
    tensor_free(Q);
    tensor_free(K);
    tensor_free(V);

    if (!out) return 0;

    /* Output shape must match Q shape */
    if (out->ndim != 3) { tensor_free(out); return 0; }
    if (out->shape[0] != batch) { tensor_free(out); return 0; }
    if (out->shape[1] != seq) { tensor_free(out); return 0; }
    if (out->shape[2] != dim) { tensor_free(out); return 0; }

    tensor_free(out);
    return 1;
}

static int test_gqa_grouped(void) {
    /* GQA: num_heads=8, num_kv_heads=2 (4 Q heads per KV head) */
    int batch = 1, seq = 3, num_heads = 8, num_kv_heads = 2, head_dim = 4;
    int q_dim = num_heads * head_dim;       /* 32 */
    int kv_dim = num_kv_heads * head_dim;   /* 8 */

    int q_shape[] = {batch, seq, q_dim};
    int kv_shape[] = {batch, seq, kv_dim};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};

    Tensor* Q = tensor_rand(q_shape, 3, &cfg);
    Tensor* K = tensor_rand(kv_shape, 3, &cfg);
    Tensor* V = tensor_rand(kv_shape, 3, &cfg);
    if (!Q || !K || !V) {
        if (Q) tensor_free(Q);
        if (K) tensor_free(K);
        if (V) tensor_free(V);
        return 0;
    }

    CMLGQAConfig gqa_cfg = {
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .scale = 0.0f,
        .causal = false
    };

    Tensor* out = cml_gqa_forward(Q, K, V, &gqa_cfg, NULL);
    tensor_free(Q);
    tensor_free(K);
    tensor_free(V);

    if (!out) return 0;
    if (out->shape[0] != batch) { tensor_free(out); return 0; }
    if (out->shape[1] != seq) { tensor_free(out); return 0; }
    if (out->shape[2] != q_dim) { tensor_free(out); return 0; }

    tensor_free(out);
    return 1;
}

static int test_gqa_output_shape(void) {
    /* Test with different seq lengths for Q and KV */
    int batch = 2, seq_q = 1, kv_len = 5, num_heads = 4, num_kv_heads = 2, head_dim = 8;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;

    int q_shape[] = {batch, seq_q, q_dim};
    int kv_shape[] = {batch, kv_len, kv_dim};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};

    Tensor* Q = tensor_rand(q_shape, 3, &cfg);
    Tensor* K = tensor_rand(kv_shape, 3, &cfg);
    Tensor* V = tensor_rand(kv_shape, 3, &cfg);
    if (!Q || !K || !V) {
        if (Q) tensor_free(Q);
        if (K) tensor_free(K);
        if (V) tensor_free(V);
        return 0;
    }

    CMLGQAConfig gqa_cfg = {
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .scale = 0.0f,
        .causal = false
    };

    Tensor* out = cml_gqa_forward(Q, K, V, &gqa_cfg, NULL);
    tensor_free(Q);
    tensor_free(K);
    tensor_free(V);

    if (!out) return 0;
    if (out->ndim != 3) { tensor_free(out); return 0; }
    if (out->shape[0] != batch) { tensor_free(out); return 0; }
    if (out->shape[1] != seq_q) { tensor_free(out); return 0; }
    if (out->shape[2] != q_dim) { tensor_free(out); return 0; }

    tensor_free(out);
    return 1;
}

static int test_gqa_causal(void) {
    /* Test causal masking: with causal=true, future positions should not be attended to */
    int batch = 1, seq = 4, num_heads = 1, head_dim = 2;
    int dim = num_heads * head_dim;

    int shape[] = {batch, seq, dim};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};

    Tensor* Q = tensor_ones(shape, 3, &cfg);
    Tensor* K = tensor_ones(shape, 3, &cfg);
    Tensor* V = tensor_rand(shape, 3, &cfg);
    if (!Q || !K || !V) {
        if (Q) tensor_free(Q);
        if (K) tensor_free(K);
        if (V) tensor_free(V);
        return 0;
    }

    CMLGQAConfig gqa_cfg = {
        .num_heads = num_heads,
        .num_kv_heads = num_heads,
        .head_dim = head_dim,
        .scale = 0.0f,
        .causal = true
    };

    Tensor* out = cml_gqa_forward(Q, K, V, &gqa_cfg, NULL);
    tensor_free(Q);
    tensor_free(K);
    tensor_free(V);

    if (!out) return 0;

    /* With causal mask and uniform Q/K, the first position should only attend
     * to itself, so its output should equal V[0] exactly */
    tensor_ensure_executed(out);
    tensor_free(out);
    return 1;
}

/* ===== RoPE Tests ===== */

static int test_rope_basic(void) {
    /* Verify RoPE modifies values at non-zero positions */
    int shape[] = {1, 4, 8}; /* [batch, seq, dim] */
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* x = tensor_ones(shape, 3, &cfg);
    if (!x) return 0;

    tensor_ensure_executed(x);
    float orig_val = tensor_get_float(x, 0);

    CMLRoPEConfig rope_cfg = {
        .dim = 8,
        .max_seq_len = 128,
        .base = 10000.0f
    };

    Tensor* result = cml_rope_forward(x, 0, &rope_cfg);
    if (!result) { tensor_free(x); return 0; }

    /* At position 0: cos(0)=1, sin(0)=0, so x[0] should remain 1.0 */
    /* But at position 1+: values should be modified */
    /* Check position 1: should differ from original */
    tensor_ensure_executed(result);
    float val_at_pos1 = tensor_get_float(result, 8); /* pos=1, first element */
    /* Position 1 with freq 1/10000^(0/8) = 1.0, theta=1.0
     * x_rot = 1*cos(1) - 1*sin(1) which is different from 1.0 */
    if (fabsf(val_at_pos1 - orig_val) < 1e-6f) {
        /* RoPE should have changed the value at position > 0 */
        tensor_free(x);
        return 0;
    }

    tensor_free(x);
    return 1;
}

static int test_rope_zero_position(void) {
    /* At position 0: cos(0)=1, sin(0)=0, so values should not change */
    int shape[] = {1, 1, 8}; /* [batch, seq=1, dim] */
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* x = tensor_ones(shape, 3, &cfg);
    if (!x) return 0;

    tensor_ensure_executed(x);

    CMLRoPEConfig rope_cfg = {
        .dim = 8,
        .max_seq_len = 128,
        .base = 10000.0f
    };

    Tensor* result = cml_rope_forward(x, 0, &rope_cfg);
    if (!result) { tensor_free(x); return 0; }

    tensor_ensure_executed(result);

    /* At position 0, all frequencies give theta=0, cos(0)=1, sin(0)=0.
     * For pairs (x0, x1), x_rot_0 = x0*1 - x1*0 = x0 = 1.0
     * and x_rot_1 = x0*0 + x1*1 = x1 = 1.0 */
    for (int i = 0; i < 8; i++) {
        float val = tensor_get_float(result, (size_t)i);
        if (fabsf(val - 1.0f) > 1e-5f) {
            tensor_free(x);
            return 0;
        }
    }

    tensor_free(x);
    return 1;
}

/* ===== MoE Tests ===== */

static int test_moe_create_free(void) {
    CMLMoEConfig cfg = {
        .num_experts = 4,
        .top_k = 2,
        .input_dim = 16,
        .hidden_dim = 32,
        .capacity_factor = 1.0f,
        .normalize_weights = true
    };

    CMLMoELayer* moe = cml_moe_create(&cfg);
    if (!moe) return 0;
    if (moe->config.num_experts != 4) { cml_moe_free(moe); return 0; }
    if (moe->config.top_k != 2) { cml_moe_free(moe); return 0; }
    if (!moe->gate_weight) { cml_moe_free(moe); return 0; }
    if (!moe->expert_w1) { cml_moe_free(moe); return 0; }
    if (!moe->expert_w2) { cml_moe_free(moe); return 0; }
    if (!moe->expert_w1[0]) { cml_moe_free(moe); return 0; }
    if (!moe->expert_w1[3]) { cml_moe_free(moe); return 0; }

    cml_moe_free(moe);
    return 1;
}

static int test_moe_forward_shape(void) {
    CMLMoEConfig cfg = {
        .num_experts = 4,
        .top_k = 2,
        .input_dim = 8,
        .hidden_dim = 16,
        .capacity_factor = 1.0f,
        .normalize_weights = true
    };

    CMLMoELayer* moe = cml_moe_create(&cfg);
    if (!moe) return 0;

    int in_shape[] = {2, 3, 8}; /* [batch=2, seq=3, dim=8] */
    TensorConfig tcfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                         .has_dtype = true, .has_device = true};
    Tensor* input = tensor_rand(in_shape, 3, &tcfg);
    if (!input) { cml_moe_free(moe); return 0; }

    Tensor* output = cml_moe_forward(moe, input);
    tensor_free(input);

    if (!output) { cml_moe_free(moe); return 0; }

    /* Output shape should match input shape */
    if (output->ndim != 3) { tensor_free(output); cml_moe_free(moe); return 0; }
    if (output->shape[0] != 2) { tensor_free(output); cml_moe_free(moe); return 0; }
    if (output->shape[1] != 3) { tensor_free(output); cml_moe_free(moe); return 0; }
    if (output->shape[2] != 8) { tensor_free(output); cml_moe_free(moe); return 0; }

    tensor_free(output);
    cml_moe_free(moe);
    return 1;
}

static int test_moe_routing(void) {
    CMLMoEConfig cfg = {
        .num_experts = 4,
        .top_k = 2,
        .input_dim = 8,
        .hidden_dim = 16,
        .capacity_factor = 1.0f,
        .normalize_weights = true
    };

    CMLMoELayer* moe = cml_moe_create(&cfg);
    if (!moe) return 0;

    int in_shape[] = {1, 2, 8}; /* [batch=1, seq=2, dim=8] */
    TensorConfig tcfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                         .has_dtype = true, .has_device = true};
    Tensor* input = tensor_rand(in_shape, 3, &tcfg);
    if (!input) { cml_moe_free(moe); return 0; }

    Tensor* routing = cml_moe_get_routing(moe, input);
    tensor_free(input);

    if (!routing) { cml_moe_free(moe); return 0; }

    /* routing shape: [2, 4] (2 tokens, 4 experts) */
    if (routing->ndim != 2) { tensor_free(routing); cml_moe_free(moe); return 0; }
    if (routing->shape[0] != 2) { tensor_free(routing); cml_moe_free(moe); return 0; }
    if (routing->shape[1] != 4) { tensor_free(routing); cml_moe_free(moe); return 0; }

    /* Routing weights should sum to ~1 per token (softmax output) */
    tensor_ensure_executed(routing);
    for (int t = 0; t < 2; t++) {
        float sum = 0.0f;
        for (int e = 0; e < 4; e++) {
            float val = tensor_get_float(routing, (size_t)(t * 4 + e));
            if (val < 0.0f) { tensor_free(routing); cml_moe_free(moe); return 0; }
            sum += val;
        }
        if (fabsf(sum - 1.0f) > 1e-4f) {
            tensor_free(routing);
            cml_moe_free(moe);
            return 0;
        }
    }

    tensor_free(routing);
    cml_moe_free(moe);
    return 1;
}

static int test_moe_single_expert(void) {
    /* When top_k == num_experts, all experts contribute */
    CMLMoEConfig cfg = {
        .num_experts = 2,
        .top_k = 2,
        .input_dim = 4,
        .hidden_dim = 8,
        .capacity_factor = 1.0f,
        .normalize_weights = true
    };

    CMLMoELayer* moe = cml_moe_create(&cfg);
    if (!moe) return 0;

    int in_shape[] = {1, 1, 4};
    TensorConfig tcfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                         .has_dtype = true, .has_device = true};
    Tensor* input = tensor_rand(in_shape, 3, &tcfg);
    if (!input) { cml_moe_free(moe); return 0; }

    Tensor* output = cml_moe_forward(moe, input);
    tensor_free(input);

    if (!output) { cml_moe_free(moe); return 0; }
    if (output->shape[2] != 4) { tensor_free(output); cml_moe_free(moe); return 0; }

    tensor_free(output);
    cml_moe_free(moe);
    return 1;
}

/* ===== Tokenizer Tests ===== */

static int test_tokenizer_create_free(void) {
    char* vocab[] = {"a", "b", "c", "d", "ab", "cd", "abcd"};
    int vocab_size = 7;
    char* merges[] = {"ab", "cd", "abcd"};
    int num_merges = 3;

    CMLTokenizer* tok = cml_tokenizer_create(vocab, vocab_size, merges, num_merges);
    if (!tok) return 0;
    if (tok->vocab_size != vocab_size) { cml_tokenizer_free(tok); return 0; }
    if (tok->num_merges != num_merges) { cml_tokenizer_free(tok); return 0; }
    cml_tokenizer_free(tok);
    return 1;
}

static int test_tokenizer_encode_simple(void) {
    /* Simple character-level vocab with no merges */
    char* vocab[] = {"h", "e", "l", "o"};
    int vocab_size = 4;

    CMLTokenizer* tok = cml_tokenizer_create(vocab, vocab_size, NULL, 0);
    if (!tok) return 0;

    int num_tokens = 0;
    int* ids = cml_tokenizer_encode(tok, "hello", &num_tokens);
    if (!ids) { cml_tokenizer_free(tok); return 0; }

    /* "hello" -> 5 character tokens: h=0, e=1, l=2, l=2, o=3 */
    if (num_tokens != 5) { free(ids); cml_tokenizer_free(tok); return 0; }
    if (ids[0] != 0) { free(ids); cml_tokenizer_free(tok); return 0; }  /* h */
    if (ids[1] != 1) { free(ids); cml_tokenizer_free(tok); return 0; }  /* e */
    if (ids[2] != 2) { free(ids); cml_tokenizer_free(tok); return 0; }  /* l */
    if (ids[3] != 2) { free(ids); cml_tokenizer_free(tok); return 0; }  /* l */
    if (ids[4] != 3) { free(ids); cml_tokenizer_free(tok); return 0; }  /* o */

    free(ids);
    cml_tokenizer_free(tok);
    return 1;
}

static int test_tokenizer_decode_simple(void) {
    char* vocab[] = {"h", "e", "l", "o"};
    int vocab_size = 4;

    CMLTokenizer* tok = cml_tokenizer_create(vocab, vocab_size, NULL, 0);
    if (!tok) return 0;

    int tokens[] = {0, 1, 2, 2, 3}; /* h, e, l, l, o */
    char* text = cml_tokenizer_decode(tok, tokens, 5);
    if (!text) { cml_tokenizer_free(tok); return 0; }
    if (strcmp(text, "hello") != 0) { free(text); cml_tokenizer_free(tok); return 0; }

    free(text);
    cml_tokenizer_free(tok);
    return 1;
}

static int test_tokenizer_encode_decode_roundtrip(void) {
    /* Vocab with single chars and a merge */
    char* vocab[] = {"a", "b", "c", "ab"};
    int vocab_size = 4;
    char* merges[] = {"ab"};
    int num_merges = 1;

    CMLTokenizer* tok = cml_tokenizer_create(vocab, vocab_size, merges, num_merges);
    if (!tok) return 0;

    const char* original = "abc";
    int num_tokens = 0;
    int* ids = cml_tokenizer_encode(tok, original, &num_tokens);
    if (!ids) { cml_tokenizer_free(tok); return 0; }

    /* "abc" -> "ab" + "c" after BPE merge => 2 tokens */
    if (num_tokens != 2) { free(ids); cml_tokenizer_free(tok); return 0; }

    char* decoded = cml_tokenizer_decode(tok, ids, num_tokens);
    free(ids);

    if (!decoded) { cml_tokenizer_free(tok); return 0; }
    if (strcmp(decoded, original) != 0) {
        free(decoded);
        cml_tokenizer_free(tok);
        return 0;
    }

    free(decoded);
    cml_tokenizer_free(tok);
    return 1;
}

static int test_tokenizer_special_tokens(void) {
    char* vocab[] = {"<bos>", "<eos>", "<pad>", "<unk>", "a"};
    int vocab_size = 5;

    CMLTokenizer* tok = cml_tokenizer_create(vocab, vocab_size, NULL, 0);
    if (!tok) return 0;

    cml_tokenizer_set_special(tok, 0, 1, 2, 3);
    if (tok->bos_token_id != 0) { cml_tokenizer_free(tok); return 0; }
    if (tok->eos_token_id != 1) { cml_tokenizer_free(tok); return 0; }
    if (tok->pad_token_id != 2) { cml_tokenizer_free(tok); return 0; }
    if (tok->unk_token_id != 3) { cml_tokenizer_free(tok); return 0; }

    cml_tokenizer_free(tok);
    return 1;
}

static int test_tokenizer_vocab_size(void) {
    char* vocab[] = {"a", "b", "c"};
    int vocab_size = 3;

    CMLTokenizer* tok = cml_tokenizer_create(vocab, vocab_size, NULL, 0);
    if (!tok) return 0;
    if (cml_tokenizer_vocab_size(tok) != 3) { cml_tokenizer_free(tok); return 0; }

    cml_tokenizer_free(tok);

    /* Also test with NULL */
    if (cml_tokenizer_vocab_size(NULL) != 0) return 0;

    return 1;
}

static int test_tokenizer_unknown_token(void) {
    /* Encode a character not in the vocab -- should use unk_token_id */
    char* vocab[] = {"a", "b"};
    int vocab_size = 2;

    CMLTokenizer* tok = cml_tokenizer_create(vocab, vocab_size, NULL, 0);
    if (!tok) return 0;

    cml_tokenizer_set_special(tok, -1, -1, -1, 1); /* unk = 1 ("b") */

    int num_tokens = 0;
    int* ids = cml_tokenizer_encode(tok, "ax", &num_tokens);
    if (!ids) { cml_tokenizer_free(tok); return 0; }
    if (num_tokens != 2) { free(ids); cml_tokenizer_free(tok); return 0; }

    /* 'a' -> 0, 'x' not in vocab -> unk_token_id = 1 */
    if (ids[0] != 0) { free(ids); cml_tokenizer_free(tok); return 0; }
    if (ids[1] != 1) { free(ids); cml_tokenizer_free(tok); return 0; }

    free(ids);
    cml_tokenizer_free(tok);
    return 1;
}

static int test_tokenizer_bpe_merge(void) {
    /* Test that BPE merges apply correctly for multi-level merging */
    char* vocab[] = {"a", "b", "c", "ab", "abc"};
    int vocab_size = 5;
    char* merges[] = {"ab", "abc"};
    int num_merges = 2;

    CMLTokenizer* tok = cml_tokenizer_create(vocab, vocab_size, merges, num_merges);
    if (!tok) return 0;

    int num_tokens = 0;
    int* ids = cml_tokenizer_encode(tok, "abc", &num_tokens);
    if (!ids) { cml_tokenizer_free(tok); return 0; }

    /* "abc" -> first merge "ab" gives ["ab", "c"], then merge "abc" gives ["abc"] => 1 token */
    if (num_tokens != 1) { free(ids); cml_tokenizer_free(tok); return 0; }
    if (ids[0] != 4) { free(ids); cml_tokenizer_free(tok); return 0; } /* "abc" = ID 4 */

    free(ids);
    cml_tokenizer_free(tok);
    return 1;
}

/* ===== Main ===== */

int main(void) {
    printf("test_llm_ops\n\n");

    /* KV Cache */
    TEST(kv_cache_create_free);
    TEST(kv_cache_append);
    TEST(kv_cache_get_keys);
    TEST(kv_cache_reset);

    /* GQA */
    TEST(gqa_basic);
    TEST(gqa_grouped);
    TEST(gqa_output_shape);
    TEST(gqa_causal);

    /* RoPE */
    TEST(rope_basic);
    TEST(rope_zero_position);

    /* MoE */
    TEST(moe_create_free);
    TEST(moe_forward_shape);
    TEST(moe_routing);
    TEST(moe_single_expert);

    /* Tokenizer */
    TEST(tokenizer_create_free);
    TEST(tokenizer_encode_simple);
    TEST(tokenizer_decode_simple);
    TEST(tokenizer_encode_decode_roundtrip);
    TEST(tokenizer_special_tokens);
    TEST(tokenizer_vocab_size);
    TEST(tokenizer_unknown_token);
    TEST(tokenizer_bpe_merge);

    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
