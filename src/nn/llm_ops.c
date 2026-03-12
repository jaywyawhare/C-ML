/**
 * @file llm_ops.c
 * @brief LLM inference primitives implementation
 *
 * Implements Grouped Query Attention, KV Cache, Mixture of Experts,
 * Rotary Position Embeddings, and BPE Tokenizer.
 */

#include "nn/llm_ops.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* =========================================================================
 * Internal helpers
 * ========================================================================= */

/** Softmax over last dimension for a flat buffer treated as [rows, cols] */
static void llm_softmax_inplace(float* data, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float* row = data + (size_t)r * cols;
        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (int c = 0; c < cols; c++) {
                row[c] *= inv;
            }
        }
    }
}

/** Xavier/Glorot uniform initialization */
static void llm_xavier_init(float* data, size_t numel, int fan_in, int fan_out) {
    float scale = sqrtf(2.0f / (float)(fan_in + fan_out));
    for (size_t i = 0; i < numel; i++) {
        data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

/* =========================================================================
 * Hash table helpers for tokenizer
 * ========================================================================= */

#define HASH_EMPTY (-1)

/** FNV-1a hash */
static unsigned int fnv1a_hash(const char* str) {
    unsigned int hash = 2166136261u;
    while (*str) {
        hash ^= (unsigned char)*str++;
        hash *= 16777619u;
    }
    return hash;
}

/** Insert into open-addressing hash table. Returns 0 on success, -1 on full. */
static int hash_insert(int* table, int table_size, char** vocab, const char* key, int value) {
    unsigned int h = fnv1a_hash(key) % (unsigned int)table_size;
    for (int probe = 0; probe < table_size; probe++) {
        unsigned int idx = (h + (unsigned int)probe) % (unsigned int)table_size;
        if (table[idx] == HASH_EMPTY) {
            table[idx] = value;
            return 0;
        }
        /* If the slot already maps to the same string, update */
        if (table[idx] >= 0 && strcmp(vocab[table[idx]], key) == 0) {
            table[idx] = value;
            return 0;
        }
    }
    return -1; /* table full */
}

/** Lookup in open-addressing hash table. Returns token ID or -1 if not found. */
static int hash_lookup(const int* table, int table_size, char** vocab, const char* key) {
    unsigned int h = fnv1a_hash(key) % (unsigned int)table_size;
    for (int probe = 0; probe < table_size; probe++) {
        unsigned int idx = (h + (unsigned int)probe) % (unsigned int)table_size;
        if (table[idx] == HASH_EMPTY) {
            return -1;
        }
        if (table[idx] >= 0 && strcmp(vocab[table[idx]], key) == 0) {
            return table[idx];
        }
    }
    return -1;
}

/* =========================================================================
 * KV Cache
 * ========================================================================= */

CMLKVCache* cml_kv_cache_create(int max_seq_len, int num_kv_heads, int head_dim) {
    if (max_seq_len <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
        LOG_ERROR("cml_kv_cache_create: invalid parameters (max_seq=%d, kv_heads=%d, head_dim=%d)",
                  max_seq_len, num_kv_heads, head_dim);
        return NULL;
    }

    CMLKVCache* cache = (CMLKVCache*)calloc(1, sizeof(CMLKVCache));
    if (!cache) {
        LOG_ERROR("cml_kv_cache_create: allocation failed");
        return NULL;
    }

    cache->max_seq_len = max_seq_len;
    cache->num_kv_heads = num_kv_heads;
    cache->head_dim = head_dim;
    cache->current_len = 0;

    int shape[] = {max_seq_len, num_kv_heads, head_dim};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};

    cache->key_cache = tensor_zeros(shape, 3, &cfg);
    cache->value_cache = tensor_zeros(shape, 3, &cfg);

    if (!cache->key_cache || !cache->value_cache) {
        LOG_ERROR("cml_kv_cache_create: failed to allocate cache tensors");
        if (cache->key_cache) tensor_free(cache->key_cache);
        if (cache->value_cache) tensor_free(cache->value_cache);
        free(cache);
        return NULL;
    }

    LOG_DEBUG("Created KV cache: max_seq=%d, kv_heads=%d, head_dim=%d",
              max_seq_len, num_kv_heads, head_dim);
    return cache;
}

void cml_kv_cache_free(CMLKVCache* cache) {
    if (!cache) return;
    if (cache->key_cache) tensor_free(cache->key_cache);
    if (cache->value_cache) tensor_free(cache->value_cache);
    free(cache);
}

int cml_kv_cache_append(CMLKVCache* cache, Tensor* new_key, Tensor* new_value) {
    if (!cache || !new_key || !new_value) {
        LOG_ERROR("cml_kv_cache_append: NULL argument");
        return -1;
    }

    tensor_ensure_executed(new_key);
    tensor_ensure_executed(new_value);

    /* new_key expected shape: [seq_len, num_kv_heads, head_dim] or
     * [1, num_kv_heads, head_dim] for single-token append */
    int append_len = new_key->shape[0];
    if (cache->current_len + append_len > cache->max_seq_len) {
        LOG_ERROR("cml_kv_cache_append: cache overflow (%d + %d > %d)",
                  cache->current_len, append_len, cache->max_seq_len);
        return -1;
    }

    float* k_cache_data = (float*)tensor_data_ptr(cache->key_cache);
    float* v_cache_data = (float*)tensor_data_ptr(cache->value_cache);
    float* k_new_data = (float*)tensor_data_ptr(new_key);
    float* v_new_data = (float*)tensor_data_ptr(new_value);

    if (!k_cache_data || !v_cache_data || !k_new_data || !v_new_data) {
        LOG_ERROR("cml_kv_cache_append: failed to get data pointers");
        return -1;
    }

    size_t row_size = (size_t)cache->num_kv_heads * cache->head_dim;
    size_t offset = (size_t)cache->current_len * row_size;
    size_t copy_bytes = (size_t)append_len * row_size * sizeof(float);

    memcpy(k_cache_data + offset, k_new_data, copy_bytes);
    memcpy(v_cache_data + offset, v_new_data, copy_bytes);

    cache->current_len += append_len;
    return cache->current_len;
}

void cml_kv_cache_reset(CMLKVCache* cache) {
    if (!cache) return;
    cache->current_len = 0;
}

Tensor* cml_kv_cache_get_keys(CMLKVCache* cache) {
    if (!cache || cache->current_len == 0) return NULL;

    float* k_data = (float*)tensor_data_ptr(cache->key_cache);
    if (!k_data) return NULL;

    int shape[] = {cache->current_len, cache->num_kv_heads, cache->head_dim};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};

    size_t copy_size = (size_t)cache->current_len * cache->num_kv_heads * cache->head_dim;
    Tensor* result = tensor_from_data(k_data, shape, 3, &cfg);
    (void)copy_size;
    return result;
}

Tensor* cml_kv_cache_get_values(CMLKVCache* cache) {
    if (!cache || cache->current_len == 0) return NULL;

    float* v_data = (float*)tensor_data_ptr(cache->value_cache);
    if (!v_data) return NULL;

    int shape[] = {cache->current_len, cache->num_kv_heads, cache->head_dim};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};

    return tensor_from_data(v_data, shape, 3, &cfg);
}

/* =========================================================================
 * Grouped Query Attention (GQA)
 * ========================================================================= */

Tensor* cml_gqa_forward(Tensor* Q, Tensor* K, Tensor* V, const CMLGQAConfig* config,
                         Tensor* mask) {
    if (!Q || !K || !V || !config) {
        LOG_ERROR("cml_gqa_forward: NULL argument");
        return NULL;
    }

    tensor_ensure_executed(Q);
    tensor_ensure_executed(K);
    tensor_ensure_executed(V);

    if (Q->ndim != 3 || K->ndim != 3 || V->ndim != 3) {
        LOG_ERROR("cml_gqa_forward: expected 3D tensors [batch, seq, dim]");
        return NULL;
    }

    int num_heads = config->num_heads;
    int num_kv_heads = config->num_kv_heads;
    int head_dim = config->head_dim;

    if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
        LOG_ERROR("cml_gqa_forward: invalid config (num_heads=%d, kv_heads=%d, head_dim=%d)",
                  num_heads, num_kv_heads, head_dim);
        return NULL;
    }

    if (num_heads % num_kv_heads != 0) {
        LOG_ERROR("cml_gqa_forward: num_heads (%d) must be divisible by num_kv_heads (%d)",
                  num_heads, num_kv_heads);
        return NULL;
    }

    int groups = num_heads / num_kv_heads; /* how many Q heads share each KV head */
    int batch = Q->shape[0];
    int seq_q = Q->shape[1];
    int kv_len = K->shape[1];

    float scale = config->scale;
    if (scale <= 0.0f) {
        scale = 1.0f / sqrtf((float)head_dim);
    }

    float* q_data = (float*)tensor_data_ptr(Q);
    float* k_data = (float*)tensor_data_ptr(K);
    float* v_data = (float*)tensor_data_ptr(V);

    if (!q_data || !k_data || !v_data) {
        LOG_ERROR("cml_gqa_forward: failed to get data pointers");
        return NULL;
    }

    /* Allocate output: [batch, seq_q, num_heads * head_dim] */
    size_t out_size = (size_t)batch * seq_q * num_heads * head_dim;
    float* output = (float*)calloc(out_size, sizeof(float));
    if (!output) {
        LOG_ERROR("cml_gqa_forward: allocation failed");
        return NULL;
    }

    /* Allocate scratch for attention scores: [seq_q, kv_len] per head */
    float* scores = (float*)malloc((size_t)seq_q * kv_len * sizeof(float));
    if (!scores) {
        LOG_ERROR("cml_gqa_forward: scores allocation failed");
        free(output);
        return NULL;
    }

    float* mask_data = NULL;
    if (mask) {
        tensor_ensure_executed(mask);
        mask_data = (float*)tensor_data_ptr(mask);
    }

    /* Process each batch and each query head */
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / groups; /* which KV head this Q head uses */

            /* Compute attention scores: Q[b, :, h, :] @ K[b, :, kv_h, :]^T */
            for (int sq = 0; sq < seq_q; sq++) {
                for (int sk = 0; sk < kv_len; sk++) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        /* Q layout: [batch, seq_q, num_heads*head_dim] */
                        float q_val = q_data[((size_t)b * seq_q + sq) * num_heads * head_dim
                                             + h * head_dim + d];
                        /* K layout: [batch, kv_len, num_kv_heads*head_dim] */
                        float k_val = k_data[((size_t)b * kv_len + sk) * num_kv_heads * head_dim
                                             + kv_h * head_dim + d];
                        dot += q_val * k_val;
                    }
                    scores[sq * kv_len + sk] = dot * scale;
                }
            }

            /* Apply causal mask */
            if (config->causal) {
                for (int sq = 0; sq < seq_q; sq++) {
                    for (int sk = 0; sk < kv_len; sk++) {
                        if (sk > sq) {
                            scores[sq * kv_len + sk] = -1e9f;
                        }
                    }
                }
            }

            /* Apply external mask if provided */
            if (mask_data) {
                for (int sq = 0; sq < seq_q; sq++) {
                    for (int sk = 0; sk < kv_len; sk++) {
                        size_t mask_idx;
                        if (mask->ndim == 2) {
                            mask_idx = (size_t)sq * kv_len + sk;
                        } else {
                            mask_idx = (size_t)b * seq_q * kv_len + sq * kv_len + sk;
                        }
                        if (mask_data[mask_idx] == 0.0f) {
                            scores[sq * kv_len + sk] = -1e9f;
                        }
                    }
                }
            }

            /* Softmax over kv_len dimension */
            llm_softmax_inplace(scores, seq_q, kv_len);

            /* Compute weighted sum: scores @ V[b, :, kv_h, :] */
            for (int sq = 0; sq < seq_q; sq++) {
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int sk = 0; sk < kv_len; sk++) {
                        float v_val = v_data[((size_t)b * kv_len + sk) * num_kv_heads * head_dim
                                             + kv_h * head_dim + d];
                        sum += scores[sq * kv_len + sk] * v_val;
                    }
                    /* output layout: [batch, seq_q, num_heads * head_dim] */
                    output[((size_t)b * seq_q + sq) * num_heads * head_dim
                           + h * head_dim + d] = sum;
                }
            }
        }
    }

    free(scores);

    /* Create output tensor */
    int out_shape[] = {batch, seq_q, num_heads * head_dim};
    TensorConfig out_cfg = {.dtype = Q->dtype, .device = Q->device,
                            .has_dtype = true, .has_device = true};
    Tensor* result = tensor_from_data(output, out_shape, 3, &out_cfg);
    free(output);

    if (!result) {
        LOG_ERROR("cml_gqa_forward: failed to create output tensor");
    }
    return result;
}

Tensor* cml_gqa_forward_cached(Tensor* Q, Tensor* K, Tensor* V,
                                CMLKVCache* kv_cache,
                                const CMLGQAConfig* config) {
    if (!Q || !K || !V || !kv_cache || !config) {
        LOG_ERROR("cml_gqa_forward_cached: NULL argument");
        return NULL;
    }

    /* Append new K, V to cache */
    tensor_ensure_executed(K);
    tensor_ensure_executed(V);

    /* Reshape K from [batch, seq, kv_heads*head_dim] to [seq, kv_heads, head_dim]
     * for cache append (assume batch=1 for autoregressive decoding) */
    int seq_new = K->shape[1];
    int kv_heads = config->num_kv_heads;
    int head_dim = config->head_dim;

    float* k_data = (float*)tensor_data_ptr(K);
    float* v_data = (float*)tensor_data_ptr(V);
    if (!k_data || !v_data) {
        LOG_ERROR("cml_gqa_forward_cached: failed to get K/V data");
        return NULL;
    }

    /* Create temporary tensors for cache append: [seq_new, kv_heads, head_dim] */
    int kv_shape[] = {seq_new, kv_heads, head_dim};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};

    Tensor* k_append = tensor_from_data(k_data, kv_shape, 3, &cfg);
    Tensor* v_append = tensor_from_data(v_data, kv_shape, 3, &cfg);

    if (!k_append || !v_append) {
        if (k_append) tensor_free(k_append);
        if (v_append) tensor_free(v_append);
        return NULL;
    }

    int new_len = cml_kv_cache_append(kv_cache, k_append, v_append);
    tensor_free(k_append);
    tensor_free(v_append);

    if (new_len < 0) {
        return NULL;
    }

    /* Get full cached K, V */
    Tensor* cached_k = cml_kv_cache_get_keys(kv_cache);
    Tensor* cached_v = cml_kv_cache_get_values(kv_cache);

    if (!cached_k || !cached_v) {
        if (cached_k) tensor_free(cached_k);
        if (cached_v) tensor_free(cached_v);
        return NULL;
    }

    /* Reshape cached K, V to [1, cached_len, kv_heads*head_dim] for GQA */
    int cached_len = kv_cache->current_len;
    float* ck_data = (float*)tensor_data_ptr(cached_k);
    float* cv_data = (float*)tensor_data_ptr(cached_v);

    int full_k_shape[] = {1, cached_len, kv_heads * head_dim};
    Tensor* full_k = tensor_from_data(ck_data, full_k_shape, 3, &cfg);
    Tensor* full_v = tensor_from_data(cv_data, full_k_shape, 3, &cfg);

    tensor_free(cached_k);
    tensor_free(cached_v);

    if (!full_k || !full_v) {
        if (full_k) tensor_free(full_k);
        if (full_v) tensor_free(full_v);
        return NULL;
    }

    /* Run GQA with full cached KV (no external mask -- causal is in config) */
    Tensor* result = cml_gqa_forward(Q, full_k, full_v, config, NULL);

    tensor_free(full_k);
    tensor_free(full_v);
    return result;
}

/* =========================================================================
 * Rotary Position Embeddings (RoPE)
 * ========================================================================= */

Tensor* cml_rope_forward(Tensor* x, int start_pos, const CMLRoPEConfig* config) {
    if (!x || !config) {
        LOG_ERROR("cml_rope_forward: NULL argument");
        return NULL;
    }

    tensor_ensure_executed(x);

    if (x->ndim < 2) {
        LOG_ERROR("cml_rope_forward: input must have at least 2 dims");
        return NULL;
    }

    float* data = (float*)tensor_data_ptr(x);
    if (!data) {
        LOG_ERROR("cml_rope_forward: failed to get data pointer");
        return NULL;
    }

    int dim = config->dim;
    float base = config->base;
    if (base <= 0.0f) base = 10000.0f;

    /* Compute the total number of vectors to apply RoPE to.
     * The last dimension is head_dim and we rotate pairs within it.
     * All other dimensions are "batch" dimensions. */
    int last_dim = x->shape[x->ndim - 1];
    int rope_dim = dim < last_dim ? dim : last_dim; /* apply to first 'dim' elements */
    int half_dim = rope_dim / 2;

    /* Determine seq_len from the shape. For [batch, seq_len, num_heads, head_dim],
     * seq_len is shape[1]. For [seq_len, head_dim], seq_len is shape[0]. */
    int seq_len;
    size_t num_outer;    /* number of independent sequences (batch * heads, etc.) */

    if (x->ndim == 4) {
        /* [batch, seq_len, num_heads, head_dim] */
        int batch_size = x->shape[0];
        seq_len = x->shape[1];
        int num_heads = x->shape[2];
        num_outer = (size_t)batch_size * num_heads;

        for (size_t outer = 0; outer < num_outer; outer++) {
            int b = (int)(outer / (size_t)num_heads);
            int h = (int)(outer % (size_t)num_heads);
            for (int s = 0; s < seq_len; s++) {
                int pos = start_pos + s;
                size_t base_idx = ((size_t)b * seq_len * num_heads + (size_t)s * num_heads + h)
                                  * last_dim;
                for (int i = 0; i < half_dim; i++) {
                    float freq = 1.0f / powf(base, (float)(2 * i) / (float)dim);
                    float theta = (float)pos * freq;
                    float cos_t = cosf(theta);
                    float sin_t = sinf(theta);
                    float x0 = data[base_idx + 2 * i];
                    float x1 = data[base_idx + 2 * i + 1];
                    data[base_idx + 2 * i]     = x0 * cos_t - x1 * sin_t;
                    data[base_idx + 2 * i + 1] = x0 * sin_t + x1 * cos_t;
                }
            }
        }
    } else if (x->ndim == 3) {
        /* [batch, seq_len, dim] */
        int batch_size = x->shape[0];
        seq_len = x->shape[1];
        num_outer = (size_t)batch_size;

        for (int b = 0; b < batch_size; b++) {
            for (int s = 0; s < seq_len; s++) {
                int pos = start_pos + s;
                size_t base_idx = ((size_t)b * seq_len + s) * last_dim;
                for (int i = 0; i < half_dim; i++) {
                    float freq = 1.0f / powf(base, (float)(2 * i) / (float)dim);
                    float theta = (float)pos * freq;
                    float cos_t = cosf(theta);
                    float sin_t = sinf(theta);
                    float x0 = data[base_idx + 2 * i];
                    float x1 = data[base_idx + 2 * i + 1];
                    data[base_idx + 2 * i]     = x0 * cos_t - x1 * sin_t;
                    data[base_idx + 2 * i + 1] = x0 * sin_t + x1 * cos_t;
                }
            }
        }
    } else {
        /* [seq_len, dim] -- simplest case */
        seq_len = x->shape[0];
        for (int s = 0; s < seq_len; s++) {
            int pos = start_pos + s;
            size_t base_idx = (size_t)s * last_dim;
            for (int i = 0; i < half_dim; i++) {
                float freq = 1.0f / powf(base, (float)(2 * i) / (float)dim);
                float theta = (float)pos * freq;
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);
                float x0 = data[base_idx + 2 * i];
                float x1 = data[base_idx + 2 * i + 1];
                data[base_idx + 2 * i]     = x0 * cos_t - x1 * sin_t;
                data[base_idx + 2 * i + 1] = x0 * sin_t + x1 * cos_t;
            }
        }
    }

    return x; /* modified in-place, return same tensor for chaining */
}

/* =========================================================================
 * Mixture of Experts (MoE)
 * ========================================================================= */

CMLMoELayer* cml_moe_create(const CMLMoEConfig* config) {
    if (!config) {
        LOG_ERROR("cml_moe_create: NULL config");
        return NULL;
    }

    if (config->num_experts <= 0 || config->top_k <= 0 ||
        config->input_dim <= 0 || config->hidden_dim <= 0) {
        LOG_ERROR("cml_moe_create: invalid config (experts=%d, top_k=%d, in=%d, hid=%d)",
                  config->num_experts, config->top_k, config->input_dim, config->hidden_dim);
        return NULL;
    }

    if (config->top_k > config->num_experts) {
        LOG_ERROR("cml_moe_create: top_k (%d) > num_experts (%d)",
                  config->top_k, config->num_experts);
        return NULL;
    }

    CMLMoELayer* moe = (CMLMoELayer*)calloc(1, sizeof(CMLMoELayer));
    if (!moe) {
        LOG_ERROR("cml_moe_create: allocation failed");
        return NULL;
    }

    moe->config = *config;
    moe->ref_count = 1;

    int input_dim = config->input_dim;
    int hidden_dim = config->hidden_dim;
    int num_experts = config->num_experts;

    TensorConfig tcfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                         .has_dtype = true, .has_device = true};

    /* Gate weight: [input_dim, num_experts] */
    int gate_shape[] = {input_dim, num_experts};
    moe->gate_weight = tensor_empty(gate_shape, 2, &tcfg);
    if (!moe->gate_weight) {
        LOG_ERROR("cml_moe_create: failed to create gate weight");
        free(moe);
        return NULL;
    }
    float* gw = (float*)tensor_data_ptr(moe->gate_weight);
    if (gw) llm_xavier_init(gw, (size_t)input_dim * num_experts, input_dim, num_experts);

    /* Expert weights */
    moe->expert_w1 = (Tensor**)calloc((size_t)num_experts, sizeof(Tensor*));
    moe->expert_w2 = (Tensor**)calloc((size_t)num_experts, sizeof(Tensor*));
    if (!moe->expert_w1 || !moe->expert_w2) {
        LOG_ERROR("cml_moe_create: failed to allocate expert arrays");
        cml_moe_free(moe);
        return NULL;
    }

    int w1_shape[] = {input_dim, hidden_dim};
    int w2_shape[] = {hidden_dim, input_dim};

    for (int e = 0; e < num_experts; e++) {
        moe->expert_w1[e] = tensor_empty(w1_shape, 2, &tcfg);
        moe->expert_w2[e] = tensor_empty(w2_shape, 2, &tcfg);

        if (!moe->expert_w1[e] || !moe->expert_w2[e]) {
            LOG_ERROR("cml_moe_create: failed to create expert %d weights", e);
            cml_moe_free(moe);
            return NULL;
        }

        float* w1 = (float*)tensor_data_ptr(moe->expert_w1[e]);
        float* w2 = (float*)tensor_data_ptr(moe->expert_w2[e]);
        if (w1) llm_xavier_init(w1, (size_t)input_dim * hidden_dim, input_dim, hidden_dim);
        if (w2) llm_xavier_init(w2, (size_t)hidden_dim * input_dim, hidden_dim, input_dim);
    }

    LOG_DEBUG("Created MoE layer: %d experts, top-%d, dim=%d, hidden=%d",
              num_experts, config->top_k, input_dim, hidden_dim);
    return moe;
}

void cml_moe_free(CMLMoELayer* moe) {
    if (!moe) return;

    if (moe->gate_weight) tensor_free(moe->gate_weight);

    int num_experts = moe->config.num_experts;
    if (moe->expert_w1) {
        for (int e = 0; e < num_experts; e++) {
            if (moe->expert_w1[e]) tensor_free(moe->expert_w1[e]);
        }
        free(moe->expert_w1);
    }
    if (moe->expert_w2) {
        for (int e = 0; e < num_experts; e++) {
            if (moe->expert_w2[e]) tensor_free(moe->expert_w2[e]);
        }
        free(moe->expert_w2);
    }
    free(moe);
}

Tensor* cml_moe_forward(CMLMoELayer* moe, Tensor* input) {
    if (!moe || !input) {
        LOG_ERROR("cml_moe_forward: NULL argument");
        return NULL;
    }

    tensor_ensure_executed(input);

    if (input->ndim != 3) {
        LOG_ERROR("cml_moe_forward: expected 3D input [batch, seq_len, input_dim], got %dD",
                  input->ndim);
        return NULL;
    }

    int batch = input->shape[0];
    int seq_len = input->shape[1];
    int input_dim = moe->config.input_dim;
    int hidden_dim = moe->config.hidden_dim;
    int num_experts = moe->config.num_experts;
    int top_k = moe->config.top_k;
    int total_tokens = batch * seq_len;

    float* in_data = (float*)tensor_data_ptr(input);
    float* gate_w = (float*)tensor_data_ptr(moe->gate_weight);

    if (!in_data || !gate_w) {
        LOG_ERROR("cml_moe_forward: failed to get data pointers");
        return NULL;
    }

    /* Step 1: Compute gating scores [total_tokens, num_experts] */
    float* gate_scores = (float*)calloc((size_t)total_tokens * num_experts, sizeof(float));
    if (!gate_scores) {
        LOG_ERROR("cml_moe_forward: allocation failed");
        return NULL;
    }

    /* gate_scores = input @ gate_weight: [total_tokens, input_dim] @ [input_dim, num_experts] */
    for (int t = 0; t < total_tokens; t++) {
        for (int e = 0; e < num_experts; e++) {
            float sum = 0.0f;
            for (int d = 0; d < input_dim; d++) {
                sum += in_data[t * input_dim + d] * gate_w[d * num_experts + e];
            }
            gate_scores[t * num_experts + e] = sum;
        }
    }

    /* Step 2: Softmax to get routing probabilities */
    llm_softmax_inplace(gate_scores, total_tokens, num_experts);

    /* Step 3: Find top-k experts per token */
    int* top_k_indices = (int*)malloc((size_t)total_tokens * top_k * sizeof(int));
    float* top_k_weights = (float*)malloc((size_t)total_tokens * top_k * sizeof(float));
    if (!top_k_indices || !top_k_weights) {
        LOG_ERROR("cml_moe_forward: allocation failed");
        free(gate_scores);
        free(top_k_indices);
        free(top_k_weights);
        return NULL;
    }

    for (int t = 0; t < total_tokens; t++) {
        float* row = gate_scores + t * num_experts;

        /* Simple selection of top-k by repeated argmax */
        bool* selected = (bool*)calloc((size_t)num_experts, sizeof(bool));
        if (!selected) {
            free(gate_scores);
            free(top_k_indices);
            free(top_k_weights);
            return NULL;
        }

        for (int ki = 0; ki < top_k; ki++) {
            int best_idx = -1;
            float best_val = -1e30f;
            for (int e = 0; e < num_experts; e++) {
                if (!selected[e] && row[e] > best_val) {
                    best_val = row[e];
                    best_idx = e;
                }
            }
            top_k_indices[t * top_k + ki] = best_idx;
            top_k_weights[t * top_k + ki] = best_val;
            if (best_idx >= 0) selected[best_idx] = true;
        }
        free(selected);

        /* Optionally re-normalize top-k weights to sum to 1 */
        if (moe->config.normalize_weights) {
            float w_sum = 0.0f;
            for (int ki = 0; ki < top_k; ki++) {
                w_sum += top_k_weights[t * top_k + ki];
            }
            if (w_sum > 0.0f) {
                float inv = 1.0f / w_sum;
                for (int ki = 0; ki < top_k; ki++) {
                    top_k_weights[t * top_k + ki] *= inv;
                }
            }
        }
    }

    free(gate_scores);

    /* Step 4: Compute expert outputs and combine */
    float* output = (float*)calloc((size_t)total_tokens * input_dim, sizeof(float));
    float* expert_hidden = (float*)malloc((size_t)hidden_dim * sizeof(float));
    float* expert_out = (float*)malloc((size_t)input_dim * sizeof(float));

    if (!output || !expert_hidden || !expert_out) {
        LOG_ERROR("cml_moe_forward: allocation failed");
        free(output);
        free(expert_hidden);
        free(expert_out);
        free(top_k_indices);
        free(top_k_weights);
        return NULL;
    }

    for (int t = 0; t < total_tokens; t++) {
        float* token_in = in_data + t * input_dim;
        float* token_out = output + t * input_dim;

        for (int ki = 0; ki < top_k; ki++) {
            int expert_idx = top_k_indices[t * top_k + ki];
            float weight = top_k_weights[t * top_k + ki];

            if (expert_idx < 0 || expert_idx >= num_experts) continue;

            float* w1_data = (float*)tensor_data_ptr(moe->expert_w1[expert_idx]);
            float* w2_data = (float*)tensor_data_ptr(moe->expert_w2[expert_idx]);
            if (!w1_data || !w2_data) continue;

            /* hidden = ReLU(input @ W1): [input_dim] @ [input_dim, hidden_dim] -> [hidden_dim] */
            for (int h = 0; h < hidden_dim; h++) {
                float sum = 0.0f;
                for (int d = 0; d < input_dim; d++) {
                    sum += token_in[d] * w1_data[d * hidden_dim + h];
                }
                /* ReLU */
                expert_hidden[h] = sum > 0.0f ? sum : 0.0f;
            }

            /* out = hidden @ W2: [hidden_dim] @ [hidden_dim, input_dim] -> [input_dim] */
            for (int d = 0; d < input_dim; d++) {
                float sum = 0.0f;
                for (int h = 0; h < hidden_dim; h++) {
                    sum += expert_hidden[h] * w2_data[h * input_dim + d];
                }
                expert_out[d] = sum;
            }

            /* Weighted accumulation */
            for (int d = 0; d < input_dim; d++) {
                token_out[d] += weight * expert_out[d];
            }
        }
    }

    free(expert_hidden);
    free(expert_out);
    free(top_k_indices);
    free(top_k_weights);

    /* Create output tensor [batch, seq_len, input_dim] */
    int out_shape[] = {batch, seq_len, input_dim};
    TensorConfig out_cfg = {.dtype = input->dtype, .device = input->device,
                            .has_dtype = true, .has_device = true};
    Tensor* result = tensor_from_data(output, out_shape, 3, &out_cfg);
    free(output);

    if (!result) {
        LOG_ERROR("cml_moe_forward: failed to create output tensor");
    }
    return result;
}

Tensor* cml_moe_get_routing(CMLMoELayer* moe, Tensor* input) {
    if (!moe || !input) {
        LOG_ERROR("cml_moe_get_routing: NULL argument");
        return NULL;
    }

    tensor_ensure_executed(input);

    int total_tokens;
    if (input->ndim == 3) {
        total_tokens = input->shape[0] * input->shape[1];
    } else if (input->ndim == 2) {
        total_tokens = input->shape[0];
    } else {
        LOG_ERROR("cml_moe_get_routing: expected 2D or 3D input");
        return NULL;
    }

    int input_dim = moe->config.input_dim;
    int num_experts = moe->config.num_experts;

    float* in_data = (float*)tensor_data_ptr(input);
    float* gate_w = (float*)tensor_data_ptr(moe->gate_weight);

    if (!in_data || !gate_w) {
        LOG_ERROR("cml_moe_get_routing: failed to get data pointers");
        return NULL;
    }

    float* routing = (float*)calloc((size_t)total_tokens * num_experts, sizeof(float));
    if (!routing) {
        LOG_ERROR("cml_moe_get_routing: allocation failed");
        return NULL;
    }

    /* routing = input @ gate_weight */
    for (int t = 0; t < total_tokens; t++) {
        for (int e = 0; e < num_experts; e++) {
            float sum = 0.0f;
            for (int d = 0; d < input_dim; d++) {
                sum += in_data[t * input_dim + d] * gate_w[d * num_experts + e];
            }
            routing[t * num_experts + e] = sum;
        }
    }

    /* Softmax */
    llm_softmax_inplace(routing, total_tokens, num_experts);

    int out_shape[] = {total_tokens, num_experts};
    TensorConfig out_cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
    Tensor* result = tensor_from_data(routing, out_shape, 2, &out_cfg);
    free(routing);
    return result;
}

/* =========================================================================
 * BPE Tokenizer
 * ========================================================================= */

CMLTokenizer* cml_tokenizer_create(char** vocab, int vocab_size,
                                     char** merge_pairs, int num_merges) {
    if (!vocab || vocab_size <= 0) {
        LOG_ERROR("cml_tokenizer_create: invalid vocab (ptr=%p, size=%d)",
                  (void*)vocab, vocab_size);
        return NULL;
    }

    CMLTokenizer* tok = (CMLTokenizer*)calloc(1, sizeof(CMLTokenizer));
    if (!tok) {
        LOG_ERROR("cml_tokenizer_create: allocation failed");
        return NULL;
    }

    tok->vocab_size = vocab_size;
    tok->bos_token_id = -1;
    tok->eos_token_id = -1;
    tok->pad_token_id = -1;
    tok->unk_token_id = -1;

    /* Copy vocab strings */
    tok->vocab = (char**)calloc((size_t)vocab_size, sizeof(char*));
    if (!tok->vocab) {
        LOG_ERROR("cml_tokenizer_create: failed to allocate vocab array");
        free(tok);
        return NULL;
    }

    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i]) {
            tok->vocab[i] = strdup(vocab[i]);
            if (!tok->vocab[i]) {
                LOG_ERROR("cml_tokenizer_create: strdup failed for token %d", i);
                cml_tokenizer_free(tok);
                return NULL;
            }
        }
    }

    /* Build hash table: size = 2 * vocab_size for low load factor */
    tok->hash_size = vocab_size * 2;
    if (tok->hash_size < 64) tok->hash_size = 64;
    tok->token_to_id = (int*)malloc((size_t)tok->hash_size * sizeof(int));
    if (!tok->token_to_id) {
        LOG_ERROR("cml_tokenizer_create: failed to allocate hash table");
        cml_tokenizer_free(tok);
        return NULL;
    }

    for (int i = 0; i < tok->hash_size; i++) {
        tok->token_to_id[i] = HASH_EMPTY;
    }

    for (int i = 0; i < vocab_size; i++) {
        if (tok->vocab[i]) {
            hash_insert(tok->token_to_id, tok->hash_size, tok->vocab,
                        tok->vocab[i], i);
        }
    }

    /* Copy merge rules */
    tok->num_merges = num_merges;
    if (num_merges > 0 && merge_pairs) {
        tok->merges = (CMLBPEMerge*)calloc((size_t)num_merges, sizeof(CMLBPEMerge));
        if (!tok->merges) {
            LOG_ERROR("cml_tokenizer_create: failed to allocate merges");
            cml_tokenizer_free(tok);
            return NULL;
        }

        for (int i = 0; i < num_merges; i++) {
            if (merge_pairs[i]) {
                tok->merges[i].pair = strdup(merge_pairs[i]);
                if (!tok->merges[i].pair) {
                    LOG_ERROR("cml_tokenizer_create: strdup failed for merge %d", i);
                    cml_tokenizer_free(tok);
                    return NULL;
                }
                /* The merged token should exist in vocab; look it up */
                int id = hash_lookup(tok->token_to_id, tok->hash_size,
                                     tok->vocab, merge_pairs[i]);
                tok->merges[i].new_token_id = id; /* -1 if not found */
            }
        }
    } else {
        tok->merges = NULL;
    }

    LOG_DEBUG("Created tokenizer: vocab_size=%d, num_merges=%d", vocab_size, num_merges);
    return tok;
}

void cml_tokenizer_free(CMLTokenizer* tok) {
    if (!tok) return;

    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        free(tok->vocab);
    }

    if (tok->merges) {
        for (int i = 0; i < tok->num_merges; i++) {
            free(tok->merges[i].pair);
        }
        free(tok->merges);
    }

    free(tok->token_to_id);
    free(tok);
}

int* cml_tokenizer_encode(CMLTokenizer* tok, const char* text, int* num_tokens) {
    if (!tok || !text || !num_tokens) {
        LOG_ERROR("cml_tokenizer_encode: NULL argument");
        if (num_tokens) *num_tokens = 0;
        return NULL;
    }

    int text_len = (int)strlen(text);
    if (text_len == 0) {
        *num_tokens = 0;
        return NULL;
    }

    /* Step 1: Split text into initial character tokens.
     * Each character becomes a separate token string. */
    int capacity = text_len + 16;
    int count = 0;
    char** tokens = (char**)malloc((size_t)capacity * sizeof(char*));
    if (!tokens) {
        *num_tokens = 0;
        return NULL;
    }

    for (int i = 0; i < text_len; i++) {
        char buf[2] = {text[i], '\0'};
        tokens[count] = strdup(buf);
        if (!tokens[count]) {
            for (int j = 0; j < count; j++) free(tokens[j]);
            free(tokens);
            *num_tokens = 0;
            return NULL;
        }
        count++;
    }

    /* Step 2: Iteratively apply BPE merges.
     * For each merge rule (in priority order), scan the token list for
     * adjacent pairs that concatenate to the merge's pair string,
     * and merge them. */
    for (int m = 0; m < tok->num_merges && count > 1; m++) {
        if (!tok->merges[m].pair) continue;
        const char* merge_str = tok->merges[m].pair;
        size_t merge_len = strlen(merge_str);

        bool found = true;
        while (found && count > 1) {
            found = false;
            for (int i = 0; i < count - 1; i++) {
                /* Check if tokens[i] + tokens[i+1] == merge_str */
                size_t len_a = strlen(tokens[i]);
                size_t len_b = strlen(tokens[i + 1]);
                if (len_a + len_b != merge_len) continue;

                /* Build concatenated string to compare */
                char* concat = (char*)malloc(merge_len + 1);
                if (!concat) continue;
                memcpy(concat, tokens[i], len_a);
                memcpy(concat + len_a, tokens[i + 1], len_b);
                concat[merge_len] = '\0';

                if (strcmp(concat, merge_str) == 0) {
                    /* Merge: replace tokens[i] with the merged string, remove tokens[i+1] */
                    free(tokens[i]);
                    tokens[i] = concat;
                    free(tokens[i + 1]);
                    /* Shift remaining tokens left */
                    for (int j = i + 1; j < count - 1; j++) {
                        tokens[j] = tokens[j + 1];
                    }
                    count--;
                    found = true;
                    break; /* restart scan for this merge rule */
                } else {
                    free(concat);
                }
            }
        }
    }

    /* Step 3: Convert token strings to IDs */
    int* ids = (int*)malloc((size_t)count * sizeof(int));
    if (!ids) {
        for (int i = 0; i < count; i++) free(tokens[i]);
        free(tokens);
        *num_tokens = 0;
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        int id = hash_lookup(tok->token_to_id, tok->hash_size, tok->vocab, tokens[i]);
        if (id >= 0) {
            ids[i] = id;
        } else {
            /* Unknown token */
            ids[i] = tok->unk_token_id >= 0 ? tok->unk_token_id : 0;
        }
        free(tokens[i]);
    }
    free(tokens);

    *num_tokens = count;
    return ids;
}

char* cml_tokenizer_decode(CMLTokenizer* tok, const int* tokens, int num_tokens) {
    if (!tok || !tokens || num_tokens <= 0) {
        LOG_ERROR("cml_tokenizer_decode: invalid arguments");
        return NULL;
    }

    /* First pass: compute total length */
    size_t total_len = 0;
    for (int i = 0; i < num_tokens; i++) {
        int id = tokens[i];
        if (id >= 0 && id < tok->vocab_size && tok->vocab[id]) {
            total_len += strlen(tok->vocab[id]);
        }
    }

    char* result = (char*)malloc(total_len + 1);
    if (!result) {
        LOG_ERROR("cml_tokenizer_decode: allocation failed");
        return NULL;
    }

    /* Second pass: concatenate */
    size_t pos = 0;
    for (int i = 0; i < num_tokens; i++) {
        int id = tokens[i];
        if (id >= 0 && id < tok->vocab_size && tok->vocab[id]) {
            size_t len = strlen(tok->vocab[id]);
            memcpy(result + pos, tok->vocab[id], len);
            pos += len;
        }
    }
    result[pos] = '\0';

    return result;
}

void cml_tokenizer_set_special(CMLTokenizer* tok, int bos, int eos, int pad, int unk) {
    if (!tok) return;
    tok->bos_token_id = bos;
    tok->eos_token_id = eos;
    tok->pad_token_id = pad;
    tok->unk_token_id = unk;
}

int cml_tokenizer_vocab_size(const CMLTokenizer* tok) {
    if (!tok) return 0;
    return tok->vocab_size;
}
