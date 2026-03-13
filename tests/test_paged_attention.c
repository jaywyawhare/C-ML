/**
 * @file test_paged_attention.c
 * @brief Tests for paged KV cache and paged GQA forward
 *
 * Covers block allocation/free, sequence lifecycle, token append,
 * paged GQA output shape, and multi-sequence block pool sharing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "nn/paged_attention.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-55s ", #name); \
    fflush(stdout); \
    if (test_##name()) { \
        tests_passed++; \
        printf("[PASS]\n"); \
    } else { \
        printf("[FAIL]\n"); \
    } \
} while(0)

/* ===== Helpers ===== */

/** Fill a float buffer with a constant value. */
static void fill_float(float* buf, size_t n, float val) {
    for (size_t i = 0; i < n; i++) buf[i] = val;
}

/* ===================================================================
 * 1. Block allocation and freeing
 * =================================================================== */

static int test_alloc_single_block(void) {
    CMLPagedKVCache* cache = cml_paged_kv_cache_create(4, 2, 2, 8);
    if (!cache) return 0;

    int bid = cml_paged_cache_alloc_block(cache);
    if (bid < 0) { cml_paged_kv_cache_free(cache); return 0; }
    if (!cache->blocks[bid].in_use) { cml_paged_kv_cache_free(cache); return 0; }
    if (cache->free_count != 3) { cml_paged_kv_cache_free(cache); return 0; }

    cml_paged_kv_cache_free(cache);
    return 1;
}

static int test_alloc_all_blocks(void) {
    int max_blocks = 8;
    CMLPagedKVCache* cache = cml_paged_kv_cache_create(max_blocks, 2, 2, 8);
    if (!cache) return 0;

    for (int i = 0; i < max_blocks; i++) {
        int bid = cml_paged_cache_alloc_block(cache);
        if (bid < 0) { cml_paged_kv_cache_free(cache); return 0; }
    }
    if (cache->free_count != 0) { cml_paged_kv_cache_free(cache); return 0; }

    /* Next alloc should fail */
    int bid = cml_paged_cache_alloc_block(cache);
    if (bid != -1) { cml_paged_kv_cache_free(cache); return 0; }

    cml_paged_kv_cache_free(cache);
    return 1;
}

static int test_free_block(void) {
    CMLPagedKVCache* cache = cml_paged_kv_cache_create(4, 2, 2, 8);
    if (!cache) return 0;

    int bid = cml_paged_cache_alloc_block(cache);
    if (bid < 0) { cml_paged_kv_cache_free(cache); return 0; }
    if (cache->free_count != 3) { cml_paged_kv_cache_free(cache); return 0; }

    cml_paged_cache_free_block(cache, bid);
    if (cache->free_count != 4) { cml_paged_kv_cache_free(cache); return 0; }
    if (cache->blocks[bid].in_use) { cml_paged_kv_cache_free(cache); return 0; }

    cml_paged_kv_cache_free(cache);
    return 1;
}

static int test_alloc_free_realloc(void) {
    /* Allocate, free, then re-allocate -- the freed block should be reused. */
    CMLPagedKVCache* cache = cml_paged_kv_cache_create(2, 2, 1, 4);
    if (!cache) return 0;

    int b1 = cml_paged_cache_alloc_block(cache);
    int b2 = cml_paged_cache_alloc_block(cache);
    if (b1 < 0 || b2 < 0) { cml_paged_kv_cache_free(cache); return 0; }

    /* Pool exhausted */
    if (cml_paged_cache_alloc_block(cache) != -1) { cml_paged_kv_cache_free(cache); return 0; }

    /* Free one */
    cml_paged_cache_free_block(cache, b1);
    if (cache->free_count != 1) { cml_paged_kv_cache_free(cache); return 0; }

    /* Re-allocate -- should get b1 back */
    int b3 = cml_paged_cache_alloc_block(cache);
    if (b3 != b1) { cml_paged_kv_cache_free(cache); return 0; }

    cml_paged_kv_cache_free(cache);
    return 1;
}

static int test_alloc_stats(void) {
    CMLPagedKVCache* cache = cml_paged_kv_cache_create(4, 2, 1, 4);
    if (!cache) return 0;

    int b1 = cml_paged_cache_alloc_block(cache);
    int b2 = cml_paged_cache_alloc_block(cache);
    (void)b2;
    if (cache->total_allocated != 2) { cml_paged_kv_cache_free(cache); return 0; }

    cml_paged_cache_free_block(cache, b1);
    if (cache->total_freed != 1) { cml_paged_kv_cache_free(cache); return 0; }

    cml_paged_kv_cache_free(cache);
    return 1;
}

/* ===================================================================
 * 2. Sequence init, append, token count
 * =================================================================== */

static int test_init_sequence(void) {
    CMLPagedKVCache* cache = cml_paged_kv_cache_create(16, 4, 2, 8);
    if (!cache) return 0;

    int s0 = cml_paged_cache_init_sequence(cache);
    if (s0 < 0) { cml_paged_kv_cache_free(cache); return 0; }
    if (cache->num_sequences != 1) { cml_paged_kv_cache_free(cache); return 0; }

    int s1 = cml_paged_cache_init_sequence(cache);
    if (s1 < 0 || s1 == s0) { cml_paged_kv_cache_free(cache); return 0; }
    if (cache->num_sequences != 2) { cml_paged_kv_cache_free(cache); return 0; }

    cml_paged_kv_cache_free(cache);
    return 1;
}

static int test_append_tokens(void) {
    int num_kv_heads = 2;
    int head_dim = 4;
    CMLPagedKVCache* cache = cml_paged_kv_cache_create(16, 4, num_kv_heads, head_dim);
    if (!cache) return 0;

    int seq = cml_paged_cache_init_sequence(cache);
    if (seq < 0) { cml_paged_kv_cache_free(cache); return 0; }

    size_t kv_floats = (size_t)num_kv_heads * head_dim;
    float* key = (float*)malloc(kv_floats * sizeof(float));
    float* val = (float*)malloc(kv_floats * sizeof(float));
    if (!key || !val) { free(key); free(val); cml_paged_kv_cache_free(cache); return 0; }

    /* Append 5 tokens */
    for (int t = 0; t < 5; t++) {
        fill_float(key, kv_floats, (float)(t + 1));
        fill_float(val, kv_floats, (float)(t + 1) * 0.1f);
        if (cml_paged_cache_append(cache, seq, key, val) != 0) {
            free(key); free(val);
            cml_paged_kv_cache_free(cache);
            return 0;
        }
    }

    CMLBlockTable* bt = &cache->sequences[seq];
    if (bt->seq_len != 5) { free(key); free(val); cml_paged_kv_cache_free(cache); return 0; }
    /* 5 tokens < block_size (16) => 1 block */
    if (bt->num_blocks != 1) { free(key); free(val); cml_paged_kv_cache_free(cache); return 0; }

    free(key);
    free(val);
    cml_paged_kv_cache_free(cache);
    return 1;
}

static int test_append_triggers_new_block(void) {
    int num_kv_heads = 1;
    int head_dim = 2;
    /* block_size = CML_PAGE_BLOCK_SIZE = 16 */
    CMLPagedKVCache* cache = cml_paged_kv_cache_create(8, 2, num_kv_heads, head_dim);
    if (!cache) return 0;

    int seq = cml_paged_cache_init_sequence(cache);
    if (seq < 0) { cml_paged_kv_cache_free(cache); return 0; }

    size_t kv_floats = (size_t)num_kv_heads * head_dim;
    float key[2], val[2];

    /* Append 17 tokens: should need 2 blocks (16 + 1) */
    for (int t = 0; t < 17; t++) {
        fill_float(key, kv_floats, 1.0f);
        fill_float(val, kv_floats, 1.0f);
        if (cml_paged_cache_append(cache, seq, key, val) != 0) {
            cml_paged_kv_cache_free(cache);
            return 0;
        }
    }

    CMLBlockTable* bt = &cache->sequences[seq];
    if (bt->seq_len != 17) { cml_paged_kv_cache_free(cache); return 0; }
    if (bt->num_blocks != 2) { cml_paged_kv_cache_free(cache); return 0; }

    /* First block should be full, second has 1 token */
    int b0 = bt->block_ids[0];
    int b1 = bt->block_ids[1];
    if (cache->blocks[b0].num_tokens != 16) { cml_paged_kv_cache_free(cache); return 0; }
    if (cache->blocks[b1].num_tokens != 1)  { cml_paged_kv_cache_free(cache); return 0; }

    cml_paged_kv_cache_free(cache);
    return 1;
}

static int test_append_verifies_data(void) {
    /* Append a token with known data and read it back from the block. */
    int num_kv_heads = 1;
    int head_dim = 4;
    CMLPagedKVCache* cache = cml_paged_kv_cache_create(4, 2, num_kv_heads, head_dim);
    if (!cache) return 0;

    int seq = cml_paged_cache_init_sequence(cache);
    if (seq < 0) { cml_paged_kv_cache_free(cache); return 0; }

    float key[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float val[] = {0.1f, 0.2f, 0.3f, 0.4f};

    if (cml_paged_cache_append(cache, seq, key, val) != 0) {
        cml_paged_kv_cache_free(cache);
        return 0;
    }

    CMLBlockTable* bt = &cache->sequences[seq];
    int bid = bt->block_ids[0];
    CMLPageBlock* blk = &cache->blocks[bid];

    /* Verify key data at token slot 0 */
    for (int d = 0; d < head_dim; d++) {
        if (fabsf(blk->key_data[d] - key[d]) > 1e-6f) {
            cml_paged_kv_cache_free(cache);
            return 0;
        }
        if (fabsf(blk->value_data[d] - val[d]) > 1e-6f) {
            cml_paged_kv_cache_free(cache);
            return 0;
        }
    }

    cml_paged_kv_cache_free(cache);
    return 1;
}

/* ===================================================================
 * 3. Free sequence returns blocks to pool
 * =================================================================== */

static int test_free_sequence_returns_blocks(void) {
    CMLPagedKVCache* cache = cml_paged_kv_cache_create(8, 4, 1, 4);
    if (!cache) return 0;

    int initial_free = cache->free_count;

    int seq = cml_paged_cache_init_sequence(cache);
    if (seq < 0) { cml_paged_kv_cache_free(cache); return 0; }

    float key[4] = {1, 2, 3, 4};
    float val[4] = {5, 6, 7, 8};

    /* Append 20 tokens => ceil(20/16) = 2 blocks */
    for (int t = 0; t < 20; t++) {
        if (cml_paged_cache_append(cache, seq, key, val) != 0) {
            cml_paged_kv_cache_free(cache);
            return 0;
        }
    }

    int blocks_used = cache->sequences[seq].num_blocks;
    if (blocks_used != 2) { cml_paged_kv_cache_free(cache); return 0; }

    int free_before = cache->free_count;

    cml_paged_cache_free_sequence(cache, seq);

    /* All blocks should be returned */
    if (cache->free_count != free_before + blocks_used) {
        cml_paged_kv_cache_free(cache);
        return 0;
    }
    /* Total free should equal initial free count */
    if (cache->free_count != initial_free) {
        cml_paged_kv_cache_free(cache);
        return 0;
    }
    /* Sequence slot should be marked unused */
    if (cache->sequences[seq].block_ids != NULL) {
        cml_paged_kv_cache_free(cache);
        return 0;
    }
    if (cache->num_sequences != 0) {
        cml_paged_kv_cache_free(cache);
        return 0;
    }

    cml_paged_kv_cache_free(cache);
    return 1;
}

static int test_free_sequence_reuse_slot(void) {
    CMLPagedKVCache* cache = cml_paged_kv_cache_create(8, 2, 1, 4);
    if (!cache) return 0;

    int s0 = cml_paged_cache_init_sequence(cache);
    int s1 = cml_paged_cache_init_sequence(cache);
    if (s0 < 0 || s1 < 0) { cml_paged_kv_cache_free(cache); return 0; }

    /* Max sequences = 2, so another init should fail */
    if (cml_paged_cache_init_sequence(cache) != -1) {
        cml_paged_kv_cache_free(cache);
        return 0;
    }

    /* Free one sequence, then init again should succeed */
    cml_paged_cache_free_sequence(cache, s0);
    int s2 = cml_paged_cache_init_sequence(cache);
    if (s2 < 0) { cml_paged_kv_cache_free(cache); return 0; }
    /* Should reuse the same slot */
    if (s2 != s0) { cml_paged_kv_cache_free(cache); return 0; }

    cml_paged_kv_cache_free(cache);
    return 1;
}

/* ===================================================================
 * 4. Paged GQA forward produces correct shape
 * =================================================================== */

static int test_paged_gqa_output_shape(void) {
    int num_kv_heads = 2;
    int num_heads = 4;  /* 4 Q heads, 2 KV heads => groups = 2 */
    int head_dim = 8;
    int block_count = 16;

    CMLPagedKVCache* cache = cml_paged_kv_cache_create(block_count, 4,
                                                        num_kv_heads, head_dim);
    if (!cache) return 0;

    int seq = cml_paged_cache_init_sequence(cache);
    if (seq < 0) { cml_paged_kv_cache_free(cache); return 0; }

    /* Populate sequence with 10 tokens */
    size_t kv_floats = (size_t)num_kv_heads * head_dim;
    float* key = (float*)malloc(kv_floats * sizeof(float));
    float* val = (float*)malloc(kv_floats * sizeof(float));
    if (!key || !val) { free(key); free(val); cml_paged_kv_cache_free(cache); return 0; }

    for (int t = 0; t < 10; t++) {
        fill_float(key, kv_floats, 0.1f * (t + 1));
        fill_float(val, kv_floats, 0.01f * (t + 1));
        cml_paged_cache_append(cache, seq, key, val);
    }
    free(key);
    free(val);

    /* Create Q tensor: [1, 1, num_heads * head_dim] (single query token) */
    int q_dim = num_heads * head_dim;
    float* q_data = (float*)malloc((size_t)q_dim * sizeof(float));
    if (!q_data) { cml_paged_kv_cache_free(cache); return 0; }
    fill_float(q_data, (size_t)q_dim, 0.5f);

    int q_shape[] = {1, 1, q_dim};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* Q = tensor_from_data(q_data, q_shape, 3, &cfg);
    free(q_data);
    if (!Q) { cml_paged_kv_cache_free(cache); return 0; }

    CMLGQAConfig gqa_cfg = {
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .scale = 0.0f,   /* auto: 1/sqrt(head_dim) */
        .causal = false,
        .window_size = 0
    };

    Tensor* out = cml_paged_gqa_forward(cache, seq, Q, &gqa_cfg);
    tensor_free(Q);

    if (!out) { cml_paged_kv_cache_free(cache); return 0; }

    /* Output shape: [1, 1, num_heads * head_dim] */
    if (out->ndim != 3) { tensor_free(out); cml_paged_kv_cache_free(cache); return 0; }
    if (out->shape[0] != 1) { tensor_free(out); cml_paged_kv_cache_free(cache); return 0; }
    if (out->shape[1] != 1) { tensor_free(out); cml_paged_kv_cache_free(cache); return 0; }
    if (out->shape[2] != q_dim) { tensor_free(out); cml_paged_kv_cache_free(cache); return 0; }

    tensor_free(out);
    cml_paged_kv_cache_free(cache);
    return 1;
}

static int test_paged_gqa_multi_query_tokens(void) {
    /* Test with seq_q > 1 (prefill-style) */
    int num_kv_heads = 2;
    int num_heads = 2;
    int head_dim = 4;

    CMLPagedKVCache* cache = cml_paged_kv_cache_create(16, 4, num_kv_heads, head_dim);
    if (!cache) return 0;

    int seq = cml_paged_cache_init_sequence(cache);
    if (seq < 0) { cml_paged_kv_cache_free(cache); return 0; }

    size_t kv_floats = (size_t)num_kv_heads * head_dim;
    float kbuf[8], vbuf[8];

    /* 20 cached KV tokens */
    for (int t = 0; t < 20; t++) {
        fill_float(kbuf, kv_floats, 0.05f * (t + 1));
        fill_float(vbuf, kv_floats, 0.02f * (t + 1));
        cml_paged_cache_append(cache, seq, kbuf, vbuf);
    }

    /* Q with 3 query positions */
    int q_dim = num_heads * head_dim;
    int seq_q = 3;
    int total_q = seq_q * q_dim;
    float* q_data = (float*)malloc((size_t)total_q * sizeof(float));
    if (!q_data) { cml_paged_kv_cache_free(cache); return 0; }
    fill_float(q_data, (size_t)total_q, 0.3f);

    int q_shape[] = {1, seq_q, q_dim};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* Q = tensor_from_data(q_data, q_shape, 3, &cfg);
    free(q_data);
    if (!Q) { cml_paged_kv_cache_free(cache); return 0; }

    CMLGQAConfig gqa_cfg = {
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .scale = 0.0f,
        .causal = false,
        .window_size = 0
    };

    Tensor* out = cml_paged_gqa_forward(cache, seq, Q, &gqa_cfg);
    tensor_free(Q);

    if (!out) { cml_paged_kv_cache_free(cache); return 0; }
    if (out->ndim != 3)          { tensor_free(out); cml_paged_kv_cache_free(cache); return 0; }
    if (out->shape[0] != 1)      { tensor_free(out); cml_paged_kv_cache_free(cache); return 0; }
    if (out->shape[1] != seq_q)  { tensor_free(out); cml_paged_kv_cache_free(cache); return 0; }
    if (out->shape[2] != q_dim)  { tensor_free(out); cml_paged_kv_cache_free(cache); return 0; }

    tensor_free(out);
    cml_paged_kv_cache_free(cache);
    return 1;
}

static int test_paged_gqa_output_nonzero(void) {
    /* Verify the output is not all zeros when given non-zero inputs. */
    int num_kv_heads = 1;
    int num_heads = 1;
    int head_dim = 4;

    CMLPagedKVCache* cache = cml_paged_kv_cache_create(8, 2, num_kv_heads, head_dim);
    if (!cache) return 0;

    int seq = cml_paged_cache_init_sequence(cache);
    if (seq < 0) { cml_paged_kv_cache_free(cache); return 0; }

    float key[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float val[] = {0.0f, 0.0f, 0.0f, 1.0f};
    cml_paged_cache_append(cache, seq, key, val);

    float q_raw[] = {1.0f, 0.0f, 0.0f, 0.0f};
    int q_shape[] = {1, 1, 4};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* Q = tensor_from_data(q_raw, q_shape, 3, &cfg);
    if (!Q) { cml_paged_kv_cache_free(cache); return 0; }

    CMLGQAConfig gqa_cfg = {
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .scale = 0.0f,
        .causal = false,
        .window_size = 0
    };

    Tensor* out = cml_paged_gqa_forward(cache, seq, Q, &gqa_cfg);
    tensor_free(Q);

    if (!out) { cml_paged_kv_cache_free(cache); return 0; }

    tensor_ensure_executed(out);
    float* out_data = (float*)tensor_data_ptr(out);
    if (!out_data) { tensor_free(out); cml_paged_kv_cache_free(cache); return 0; }

    /* With a single KV token, softmax gives weight 1.0 to it.
       Output should equal the value vector: [0, 0, 0, 1] */
    if (fabsf(out_data[3] - 1.0f) > 1e-5f) {
        tensor_free(out);
        cml_paged_kv_cache_free(cache);
        return 0;
    }

    tensor_free(out);
    cml_paged_kv_cache_free(cache);
    return 1;
}

/* ===================================================================
 * 5. Multiple sequences sharing the block pool
 * =================================================================== */

static int test_multi_sequence_shared_pool(void) {
    int max_blocks = 16;
    CMLPagedKVCache* cache = cml_paged_kv_cache_create(max_blocks, 4, 1, 4);
    if (!cache) return 0;

    int s0 = cml_paged_cache_init_sequence(cache);
    int s1 = cml_paged_cache_init_sequence(cache);
    if (s0 < 0 || s1 < 0) { cml_paged_kv_cache_free(cache); return 0; }

    float key[4] = {1, 2, 3, 4};
    float val[4] = {5, 6, 7, 8};

    /* Sequence 0: 20 tokens => 2 blocks */
    for (int t = 0; t < 20; t++) {
        if (cml_paged_cache_append(cache, s0, key, val) != 0) {
            cml_paged_kv_cache_free(cache);
            return 0;
        }
    }

    /* Sequence 1: 10 tokens => 1 block */
    for (int t = 0; t < 10; t++) {
        if (cml_paged_cache_append(cache, s1, key, val) != 0) {
            cml_paged_kv_cache_free(cache);
            return 0;
        }
    }

    if (cache->sequences[s0].seq_len != 20) { cml_paged_kv_cache_free(cache); return 0; }
    if (cache->sequences[s1].seq_len != 10) { cml_paged_kv_cache_free(cache); return 0; }
    if (cache->sequences[s0].num_blocks != 2) { cml_paged_kv_cache_free(cache); return 0; }
    if (cache->sequences[s1].num_blocks != 1) { cml_paged_kv_cache_free(cache); return 0; }

    /* Total blocks used: 3 */
    if (cache->free_count != max_blocks - 3) { cml_paged_kv_cache_free(cache); return 0; }

    /* Free sequence 0, pool should recover 2 blocks */
    cml_paged_cache_free_sequence(cache, s0);
    if (cache->free_count != max_blocks - 1) { cml_paged_kv_cache_free(cache); return 0; }

    /* Sequence 1 still intact */
    if (cache->sequences[s1].seq_len != 10) { cml_paged_kv_cache_free(cache); return 0; }

    cml_paged_kv_cache_free(cache);
    return 1;
}

static int test_multi_sequence_independent_data(void) {
    /* Two sequences should not interfere with each other's data. */
    int num_kv_heads = 1;
    int head_dim = 2;
    CMLPagedKVCache* cache = cml_paged_kv_cache_create(8, 4, num_kv_heads, head_dim);
    if (!cache) return 0;

    int s0 = cml_paged_cache_init_sequence(cache);
    int s1 = cml_paged_cache_init_sequence(cache);
    if (s0 < 0 || s1 < 0) { cml_paged_kv_cache_free(cache); return 0; }

    float key0[] = {1.0f, 2.0f};
    float val0[] = {3.0f, 4.0f};
    float key1[] = {10.0f, 20.0f};
    float val1[] = {30.0f, 40.0f};

    cml_paged_cache_append(cache, s0, key0, val0);
    cml_paged_cache_append(cache, s1, key1, val1);

    /* Verify s0's block has key0 */
    int bid0 = cache->sequences[s0].block_ids[0];
    if (fabsf(cache->blocks[bid0].key_data[0] - 1.0f) > 1e-6f) {
        cml_paged_kv_cache_free(cache);
        return 0;
    }
    if (fabsf(cache->blocks[bid0].key_data[1] - 2.0f) > 1e-6f) {
        cml_paged_kv_cache_free(cache);
        return 0;
    }

    /* Verify s1's block has key1 */
    int bid1 = cache->sequences[s1].block_ids[0];
    if (fabsf(cache->blocks[bid1].key_data[0] - 10.0f) > 1e-6f) {
        cml_paged_kv_cache_free(cache);
        return 0;
    }
    if (fabsf(cache->blocks[bid1].key_data[1] - 20.0f) > 1e-6f) {
        cml_paged_kv_cache_free(cache);
        return 0;
    }

    /* Blocks must be different */
    if (bid0 == bid1) { cml_paged_kv_cache_free(cache); return 0; }

    cml_paged_kv_cache_free(cache);
    return 1;
}

/* ===================================================================
 * Edge cases and error handling
 * =================================================================== */

static int test_create_invalid_params(void) {
    if (cml_paged_kv_cache_create(0, 4, 2, 8) != NULL) return 0;
    if (cml_paged_kv_cache_create(4, 0, 2, 8) != NULL) return 0;
    if (cml_paged_kv_cache_create(4, 4, 0, 8) != NULL) return 0;
    if (cml_paged_kv_cache_create(4, 4, 2, 0) != NULL) return 0;
    if (cml_paged_kv_cache_create(-1, 4, 2, 8) != NULL) return 0;
    return 1;
}

static int test_free_null_cache(void) {
    /* Should not crash */
    cml_paged_kv_cache_free(NULL);
    return 1;
}

static int test_append_invalid_seq(void) {
    CMLPagedKVCache* cache = cml_paged_kv_cache_create(4, 2, 1, 4);
    if (!cache) return 0;

    float key[4] = {0};
    float val[4] = {0};

    /* Append to non-existent sequence */
    if (cml_paged_cache_append(cache, 0, key, val) != -1) {
        cml_paged_kv_cache_free(cache);
        return 0;
    }
    /* Negative seq_id */
    if (cml_paged_cache_append(cache, -1, key, val) != -1) {
        cml_paged_kv_cache_free(cache);
        return 0;
    }

    cml_paged_kv_cache_free(cache);
    return 1;
}

static int test_gqa_null_args(void) {
    CMLPagedKVCache* cache = cml_paged_kv_cache_create(4, 2, 1, 4);
    if (!cache) return 0;

    CMLGQAConfig gqa_cfg = {.num_heads = 1, .num_kv_heads = 1, .head_dim = 4};

    if (cml_paged_gqa_forward(NULL, 0, NULL, &gqa_cfg) != NULL) {
        cml_paged_kv_cache_free(cache);
        return 0;
    }
    if (cml_paged_gqa_forward(cache, 0, NULL, &gqa_cfg) != NULL) {
        cml_paged_kv_cache_free(cache);
        return 0;
    }

    cml_paged_kv_cache_free(cache);
    return 1;
}

/* ===================================================================
 * Main
 * =================================================================== */

int main(void) {
    printf("test_paged_attention\n\n");

    /* Block allocation and freeing */
    TEST(alloc_single_block);
    TEST(alloc_all_blocks);
    TEST(free_block);
    TEST(alloc_free_realloc);
    TEST(alloc_stats);

    /* Sequence init, append, token count */
    TEST(init_sequence);
    TEST(append_tokens);
    TEST(append_triggers_new_block);
    TEST(append_verifies_data);

    /* Free sequence returns blocks to pool */
    TEST(free_sequence_returns_blocks);
    TEST(free_sequence_reuse_slot);

    /* Paged GQA forward */
    TEST(paged_gqa_output_shape);
    TEST(paged_gqa_multi_query_tokens);
    TEST(paged_gqa_output_nonzero);

    /* Multiple sequences sharing block pool */
    TEST(multi_sequence_shared_pool);
    TEST(multi_sequence_independent_data);

    /* Edge cases */
    TEST(create_invalid_params);
    TEST(free_null_cache);
    TEST(append_invalid_seq);
    TEST(gqa_null_args);

    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
