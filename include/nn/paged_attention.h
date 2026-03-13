/**
 * @file paged_attention.h
 * @brief Paged KV cache and attention for efficient LLM serving
 *
 * Implements a paged block allocator for KV caches, enabling multiple
 * sequences to share a fixed GPU/CPU memory pool without contiguous
 * allocation.  Pairs with Grouped Query Attention (GQA) to run
 * attention directly over non-contiguous paged blocks.
 */

#ifndef CML_NN_PAGED_ATTENTION_H
#define CML_NN_PAGED_ATTENTION_H

#include "tensor/tensor.h"
#include "nn/llm_ops.h"
#include "core/logging.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===== Constants ===== */

#define CML_PAGE_BLOCK_SIZE 16  /* tokens per block */
#define CML_MAX_BLOCKS      4096
#define CML_MAX_SEQUENCES   256

/* ===== Data Structures ===== */

/**
 * A single page block that holds a fixed number of KV token slots.
 * Data layout per buffer: [block_size, num_kv_heads, head_dim].
 */
typedef struct CMLPageBlock {
    float* key_data;    /* [block_size, num_kv_heads, head_dim] */
    float* value_data;  /* [block_size, num_kv_heads, head_dim] */
    int num_tokens;     /* How many tokens actually stored (0..block_size) */
    int block_id;
    bool in_use;
} CMLPageBlock;

/**
 * Per-sequence block table: maps a logical sequence position to
 * physical block IDs in the cache pool.
 */
typedef struct CMLBlockTable {
    int* block_ids;     /* Array of block IDs for this sequence */
    int num_blocks;
    int capacity;
    int seq_len;        /* Total tokens in this sequence */
} CMLBlockTable;

/**
 * The paged KV cache: a pool of fixed-size blocks shared across
 * multiple sequences, with a stack-based free list allocator.
 */
typedef struct CMLPagedKVCache {
    CMLPageBlock* blocks;
    int num_blocks;
    int max_blocks;

    int* free_list;
    int free_count;

    CMLBlockTable* sequences;
    int num_sequences;
    int max_sequences;

    int num_kv_heads;
    int head_dim;
    int block_size;

    /* Stats */
    size_t total_allocated;
    size_t total_freed;
} CMLPagedKVCache;

/* ===== Cache Lifecycle ===== */

/** Create a paged KV cache with pre-allocated block pool.
 *  @param max_blocks     Maximum number of page blocks in the pool.
 *  @param max_sequences  Maximum concurrent sequences.
 *  @param num_kv_heads   Number of KV attention heads.
 *  @param head_dim       Dimension per attention head.
 *  @return Allocated cache, or NULL on failure.
 */
CMLPagedKVCache* cml_paged_kv_cache_create(int max_blocks, int max_sequences,
                                            int num_kv_heads, int head_dim);

/** Free the entire paged KV cache and all block data. */
void cml_paged_kv_cache_free(CMLPagedKVCache* cache);

/* ===== Block Allocator ===== */

/** Allocate a single block from the free list.
 *  @return Block ID (>= 0) on success, -1 if pool exhausted.
 */
int cml_paged_cache_alloc_block(CMLPagedKVCache* cache);

/** Return a block to the free list.
 *  @param block_id  The block to release.
 */
void cml_paged_cache_free_block(CMLPagedKVCache* cache, int block_id);

/* ===== Sequence Management ===== */

/** Initialise a new sequence in the cache.
 *  @return Sequence ID (>= 0) on success, -1 if max sequences reached.
 */
int cml_paged_cache_init_sequence(CMLPagedKVCache* cache);

/** Free all blocks belonging to a sequence and mark the slot unused. */
void cml_paged_cache_free_sequence(CMLPagedKVCache* cache, int seq_id);

/* ===== Token Append ===== */

/** Append one token's KV data to a sequence.
 *  Allocates a new block automatically when the current tail block is full.
 *  @param key    Pointer to key data [num_kv_heads * head_dim] floats.
 *  @param value  Pointer to value data [num_kv_heads * head_dim] floats.
 *  @return 0 on success, -1 on failure (pool exhausted, bad seq_id, etc.).
 */
int cml_paged_cache_append(CMLPagedKVCache* cache, int seq_id,
                            const float* key, const float* value);

/* ===== Paged GQA Forward ===== */

/** Run Grouped Query Attention gathering K/V from paged blocks.
 *  @param cache   The paged KV cache.
 *  @param seq_id  Which sequence's block table to use for K/V.
 *  @param Q       Query tensor [batch=1, seq_len, num_heads * head_dim].
 *  @param config  GQA configuration (num_heads, num_kv_heads, head_dim, etc.).
 *  @return Output tensor [batch=1, seq_len, num_heads * head_dim], or NULL on error.
 */
Tensor* cml_paged_gqa_forward(CMLPagedKVCache* cache, int seq_id,
                               Tensor* Q, const CMLGQAConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_PAGED_ATTENTION_H */
