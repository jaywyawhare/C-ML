#include "nn/paged_attention.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void paged_softmax_inplace(float* data, int rows, int cols) {
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

static inline size_t token_kv_size(const CMLPagedKVCache* cache) {
    return (size_t)cache->num_kv_heads * cache->head_dim;
}

static inline size_t block_buf_size(const CMLPagedKVCache* cache) {
    return (size_t)cache->block_size * token_kv_size(cache);
}

CMLPagedKVCache* cml_paged_kv_cache_create(int max_blocks, int max_sequences,
                                            int num_kv_heads, int head_dim) {
    if (max_blocks <= 0 || max_sequences <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
        LOG_ERROR("cml_paged_kv_cache_create: invalid parameters "
                  "(max_blocks=%d, max_sequences=%d, num_kv_heads=%d, head_dim=%d)",
                  max_blocks, max_sequences, num_kv_heads, head_dim);
        return NULL;
    }

    CMLPagedKVCache* cache = (CMLPagedKVCache*)calloc(1, sizeof(CMLPagedKVCache));
    if (!cache) {
        LOG_ERROR("cml_paged_kv_cache_create: allocation failed");
        return NULL;
    }

    cache->max_blocks = max_blocks;
    cache->max_sequences = max_sequences;
    cache->num_kv_heads = num_kv_heads;
    cache->head_dim = head_dim;
    cache->block_size = CML_PAGE_BLOCK_SIZE;
    cache->blocks = (CMLPageBlock*)calloc((size_t)max_blocks, sizeof(CMLPageBlock));
    if (!cache->blocks) {
        LOG_ERROR("cml_paged_kv_cache_create: block array allocation failed");
        free(cache);
        return NULL;
    }

    size_t buf_floats = block_buf_size(cache);
    for (int i = 0; i < max_blocks; i++) {
        cache->blocks[i].key_data = (float*)calloc(buf_floats, sizeof(float));
        cache->blocks[i].value_data = (float*)calloc(buf_floats, sizeof(float));
        if (!cache->blocks[i].key_data || !cache->blocks[i].value_data) {
            LOG_ERROR("cml_paged_kv_cache_create: block %d data allocation failed", i);
            /* Clean up already-allocated blocks */
            for (int j = 0; j <= i; j++) {
                free(cache->blocks[j].key_data);
                free(cache->blocks[j].value_data);
            }
            free(cache->blocks);
            free(cache);
            return NULL;
        }
        cache->blocks[i].block_id = i;
        cache->blocks[i].num_tokens = 0;
        cache->blocks[i].in_use = false;
    }
    cache->num_blocks = max_blocks;
    cache->free_list = (int*)malloc((size_t)max_blocks * sizeof(int));
    if (!cache->free_list) {
        LOG_ERROR("cml_paged_kv_cache_create: free list allocation failed");
        for (int i = 0; i < max_blocks; i++) {
            free(cache->blocks[i].key_data);
            free(cache->blocks[i].value_data);
        }
        free(cache->blocks);
        free(cache);
        return NULL;
    }
    /* Push all block IDs onto the free list (top of stack = last element) */
    for (int i = 0; i < max_blocks; i++) {
        cache->free_list[i] = i;
    }
    cache->free_count = max_blocks;
    cache->sequences = (CMLBlockTable*)calloc((size_t)max_sequences, sizeof(CMLBlockTable));
    if (!cache->sequences) {
        LOG_ERROR("cml_paged_kv_cache_create: sequence table allocation failed");
        free(cache->free_list);
        for (int i = 0; i < max_blocks; i++) {
            free(cache->blocks[i].key_data);
            free(cache->blocks[i].value_data);
        }
        free(cache->blocks);
        free(cache);
        return NULL;
    }
    /* Mark all sequence slots as unused (block_ids == NULL) */
    cache->num_sequences = 0;

    cache->total_allocated = 0;
    cache->total_freed = 0;

    LOG_DEBUG("cml_paged_kv_cache_create: created cache with %d blocks "
              "(block_size=%d, kv_heads=%d, head_dim=%d)",
              max_blocks, cache->block_size, num_kv_heads, head_dim);

    return cache;
}

void cml_paged_kv_cache_free(CMLPagedKVCache* cache) {
    if (!cache) return;

    /* Free all sequence block tables */
    if (cache->sequences) {
        for (int i = 0; i < cache->max_sequences; i++) {
            free(cache->sequences[i].block_ids);
        }
        free(cache->sequences);
    }

    /* Free block data */
    if (cache->blocks) {
        for (int i = 0; i < cache->num_blocks; i++) {
            free(cache->blocks[i].key_data);
            free(cache->blocks[i].value_data);
        }
        free(cache->blocks);
    }

    free(cache->free_list);
    free(cache);
}

int cml_paged_cache_alloc_block(CMLPagedKVCache* cache) {
    if (!cache) {
        LOG_ERROR("cml_paged_cache_alloc_block: NULL cache");
        return -1;
    }
    if (cache->free_count <= 0) {
        LOG_ERROR("cml_paged_cache_alloc_block: no free blocks");
        return -1;
    }

    /* Pop from top of stack */
    int block_id = cache->free_list[--cache->free_count];
    CMLPageBlock* blk = &cache->blocks[block_id];
    blk->in_use = true;
    blk->num_tokens = 0;
    /* Zero the buffers for a clean slate */
    memset(blk->key_data, 0, block_buf_size(cache) * sizeof(float));
    memset(blk->value_data, 0, block_buf_size(cache) * sizeof(float));

    cache->total_allocated++;
    return block_id;
}

void cml_paged_cache_free_block(CMLPagedKVCache* cache, int block_id) {
    if (!cache) {
        LOG_ERROR("cml_paged_cache_free_block: NULL cache");
        return;
    }
    if (block_id < 0 || block_id >= cache->num_blocks) {
        LOG_ERROR("cml_paged_cache_free_block: invalid block_id %d", block_id);
        return;
    }
    if (!cache->blocks[block_id].in_use) {
        LOG_WARNING("cml_paged_cache_free_block: block %d already free", block_id);
        return;
    }

    cache->blocks[block_id].in_use = false;
    cache->blocks[block_id].num_tokens = 0;

    /* Push back onto free list */
    cache->free_list[cache->free_count++] = block_id;
    cache->total_freed++;
}

int cml_paged_cache_init_sequence(CMLPagedKVCache* cache) {
    if (!cache) {
        LOG_ERROR("cml_paged_cache_init_sequence: NULL cache");
        return -1;
    }

    /* Find a free slot (block_ids == NULL means unused) */
    int seq_id = -1;
    for (int i = 0; i < cache->max_sequences; i++) {
        if (cache->sequences[i].block_ids == NULL) {
            seq_id = i;
            break;
        }
    }

    if (seq_id < 0) {
        LOG_ERROR("cml_paged_cache_init_sequence: no free sequence slots "
                  "(max=%d)", cache->max_sequences);
        return -1;
    }

    /* Allocate an initial block table with room for a few block IDs */
    int initial_capacity = 8;
    CMLBlockTable* bt = &cache->sequences[seq_id];
    bt->block_ids = (int*)malloc((size_t)initial_capacity * sizeof(int));
    if (!bt->block_ids) {
        LOG_ERROR("cml_paged_cache_init_sequence: allocation failed");
        return -1;
    }
    bt->num_blocks = 0;
    bt->capacity = initial_capacity;
    bt->seq_len = 0;
    cache->num_sequences++;

    LOG_DEBUG("cml_paged_cache_init_sequence: created sequence %d", seq_id);
    return seq_id;
}

void cml_paged_cache_free_sequence(CMLPagedKVCache* cache, int seq_id) {
    if (!cache) {
        LOG_ERROR("cml_paged_cache_free_sequence: NULL cache");
        return;
    }
    if (seq_id < 0 || seq_id >= cache->max_sequences) {
        LOG_ERROR("cml_paged_cache_free_sequence: invalid seq_id %d", seq_id);
        return;
    }

    CMLBlockTable* bt = &cache->sequences[seq_id];
    if (!bt->block_ids) {
        LOG_WARNING("cml_paged_cache_free_sequence: sequence %d not initialised", seq_id);
        return;
    }

    /* Return all blocks to the free pool */
    for (int i = 0; i < bt->num_blocks; i++) {
        cml_paged_cache_free_block(cache, bt->block_ids[i]);
    }

    free(bt->block_ids);
    bt->block_ids = NULL;
    bt->num_blocks = 0;
    bt->capacity = 0;
    bt->seq_len = 0;
    cache->num_sequences--;
}

int cml_paged_cache_append(CMLPagedKVCache* cache, int seq_id,
                            const float* key, const float* value) {
    if (!cache || !key || !value) {
        LOG_ERROR("cml_paged_cache_append: NULL argument");
        return -1;
    }
    if (seq_id < 0 || seq_id >= cache->max_sequences) {
        LOG_ERROR("cml_paged_cache_append: invalid seq_id %d", seq_id);
        return -1;
    }

    CMLBlockTable* bt = &cache->sequences[seq_id];
    if (!bt->block_ids) {
        LOG_ERROR("cml_paged_cache_append: sequence %d not initialised", seq_id);
        return -1;
    }

    /* Determine if we need a new block */
    bool need_new_block = (bt->num_blocks == 0);
    if (!need_new_block) {
        int tail_block_id = bt->block_ids[bt->num_blocks - 1];
        if (cache->blocks[tail_block_id].num_tokens >= cache->block_size) {
            need_new_block = true;
        }
    }

    if (need_new_block) {
        int new_bid = cml_paged_cache_alloc_block(cache);
        if (new_bid < 0) {
            LOG_ERROR("cml_paged_cache_append: failed to allocate new block for seq %d",
                      seq_id);
            return -1;
        }

        /* Grow block table if needed */
        if (bt->num_blocks >= bt->capacity) {
            int new_cap = bt->capacity * 2;
            int* new_ids = (int*)realloc(bt->block_ids, (size_t)new_cap * sizeof(int));
            if (!new_ids) {
                LOG_ERROR("cml_paged_cache_append: block table realloc failed");
                cml_paged_cache_free_block(cache, new_bid);
                return -1;
            }
            bt->block_ids = new_ids;
            bt->capacity = new_cap;
        }

        bt->block_ids[bt->num_blocks++] = new_bid;
    }

    /* Write token KV into the tail block */
    int tail_bid = bt->block_ids[bt->num_blocks - 1];
    CMLPageBlock* blk = &cache->blocks[tail_bid];
    size_t kv_floats = token_kv_size(cache);
    size_t offset = (size_t)blk->num_tokens * kv_floats;

    memcpy(blk->key_data + offset, key, kv_floats * sizeof(float));
    memcpy(blk->value_data + offset, value, kv_floats * sizeof(float));
    blk->num_tokens++;
    bt->seq_len++;

    return 0;
}

Tensor* cml_paged_gqa_forward(CMLPagedKVCache* cache, int seq_id,
                               Tensor* Q, const CMLGQAConfig* config) {
    if (!cache || !Q || !config) {
        LOG_ERROR("cml_paged_gqa_forward: NULL argument");
        return NULL;
    }
    if (seq_id < 0 || seq_id >= cache->max_sequences) {
        LOG_ERROR("cml_paged_gqa_forward: invalid seq_id %d", seq_id);
        return NULL;
    }

    CMLBlockTable* bt = &cache->sequences[seq_id];
    if (!bt->block_ids) {
        LOG_ERROR("cml_paged_gqa_forward: sequence %d not initialised", seq_id);
        return NULL;
    }

    tensor_ensure_executed(Q);

    if (Q->ndim != 3) {
        LOG_ERROR("cml_paged_gqa_forward: Q must be 3D [batch, seq_len, num_heads*head_dim], "
                  "got ndim=%d", Q->ndim);
        return NULL;
    }

    int num_heads = config->num_heads;
    int num_kv_heads = config->num_kv_heads;
    int head_dim = config->head_dim;

    if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
        LOG_ERROR("cml_paged_gqa_forward: invalid config "
                  "(num_heads=%d, kv_heads=%d, head_dim=%d)",
                  num_heads, num_kv_heads, head_dim);
        return NULL;
    }
    if (num_heads % num_kv_heads != 0) {
        LOG_ERROR("cml_paged_gqa_forward: num_heads (%d) must be divisible by num_kv_heads (%d)",
                  num_heads, num_kv_heads);
        return NULL;
    }
    if (num_kv_heads != cache->num_kv_heads || head_dim != cache->head_dim) {
        LOG_ERROR("cml_paged_gqa_forward: config mismatch with cache "
                  "(config kv_heads=%d vs cache %d, config head_dim=%d vs cache %d)",
                  num_kv_heads, cache->num_kv_heads, head_dim, cache->head_dim);
        return NULL;
    }

    int groups = num_heads / num_kv_heads;
    int batch = Q->shape[0];
    int seq_q = Q->shape[1];
    int kv_len = bt->seq_len;

    if (batch != 1) {
        LOG_ERROR("cml_paged_gqa_forward: paged attention only supports batch=1, got %d",
                  batch);
        return NULL;
    }
    if (kv_len == 0) {
        LOG_ERROR("cml_paged_gqa_forward: sequence %d has no cached tokens", seq_id);
        return NULL;
    }

    float scale = config->scale;
    if (scale <= 0.0f) {
        scale = 1.0f / sqrtf((float)head_dim);
    }

    float* q_data = (float*)tensor_data_ptr(Q);
    if (!q_data) {
        LOG_ERROR("cml_paged_gqa_forward: failed to get Q data pointer");
        return NULL;
    }

    /* Allocate output: [1, seq_q, num_heads * head_dim] */
    size_t out_size = (size_t)seq_q * num_heads * head_dim;
    float* output = (float*)calloc(out_size, sizeof(float));
    if (!output) {
        LOG_ERROR("cml_paged_gqa_forward: output allocation failed");
        return NULL;
    }

    /* Allocate scratch for attention scores: [seq_q, kv_len] */
    float* scores = (float*)malloc((size_t)seq_q * kv_len * sizeof(float));
    if (!scores) {
        LOG_ERROR("cml_paged_gqa_forward: scores allocation failed");
        free(output);
        return NULL;
    }

    size_t kv_stride = (size_t)num_kv_heads * head_dim;  /* floats per token in a block */

    /* For each query head */
    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / groups;

        /* Compute attention scores by iterating over paged blocks */
        for (int sq = 0; sq < seq_q; sq++) {
            int token_idx = 0;  /* running KV position across blocks */
            for (int bi = 0; bi < bt->num_blocks; bi++) {
                int bid = bt->block_ids[bi];
                CMLPageBlock* blk = &cache->blocks[bid];
                int tokens_in_blk = blk->num_tokens;

                for (int t = 0; t < tokens_in_blk; t++, token_idx++) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        float q_val = q_data[(size_t)sq * num_heads * head_dim
                                             + h * head_dim + d];
                        float k_val = blk->key_data[(size_t)t * kv_stride
                                                    + kv_h * head_dim + d];
                        dot += q_val * k_val;
                    }
                    scores[sq * kv_len + token_idx] = dot * scale;
                }
            }
        }

        /* Apply causal mask */
        if (config->causal) {
            for (int sq = 0; sq < seq_q; sq++) {
                /* In decode mode the query position is typically at the end of
                   the KV sequence.  Map query position sq to its absolute
                   position: kv_len - seq_q + sq  (last seq_q positions). */
                int abs_pos = kv_len - seq_q + sq;
                for (int sk = 0; sk < kv_len; sk++) {
                    if (sk > abs_pos) {
                        scores[sq * kv_len + sk] = -1e9f;
                    }
                }
            }
        }

        /* Softmax over kv_len */
        paged_softmax_inplace(scores, seq_q, kv_len);

        /* Weighted sum: gather V from paged blocks */
        for (int sq = 0; sq < seq_q; sq++) {
            int token_idx = 0;
            for (int bi = 0; bi < bt->num_blocks; bi++) {
                int bid = bt->block_ids[bi];
                CMLPageBlock* blk = &cache->blocks[bid];
                int tokens_in_blk = blk->num_tokens;

                for (int t = 0; t < tokens_in_blk; t++, token_idx++) {
                    float w = scores[sq * kv_len + token_idx];
                    for (int d = 0; d < head_dim; d++) {
                        float v_val = blk->value_data[(size_t)t * kv_stride
                                                      + kv_h * head_dim + d];
                        output[(size_t)sq * num_heads * head_dim
                               + h * head_dim + d] += w * v_val;
                    }
                }
            }
        }
    }

    free(scores);

    /* Wrap output into a tensor [1, seq_q, num_heads * head_dim] */
    int out_shape[] = {1, seq_q, num_heads * head_dim};
    TensorConfig out_cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
    Tensor* result = tensor_from_data(output, out_shape, 3, &out_cfg);
    free(output);

    if (!result) {
        LOG_ERROR("cml_paged_gqa_forward: failed to create output tensor");
    }
    return result;
}
