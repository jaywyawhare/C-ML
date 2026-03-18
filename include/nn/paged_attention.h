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

#define CML_PAGE_BLOCK_SIZE 16  /* tokens per block */
#define CML_MAX_BLOCKS      4096
#define CML_MAX_SEQUENCES   256

/* Data layout per buffer: [block_size, num_kv_heads, head_dim]. */
typedef struct CMLPageBlock {
    float* key_data;    /* [block_size, num_kv_heads, head_dim] */
    float* value_data;  /* [block_size, num_kv_heads, head_dim] */
    int num_tokens;     /* How many tokens actually stored (0..block_size) */
    int block_id;
    bool in_use;
} CMLPageBlock;

/* Maps logical sequence position to physical block IDs in the cache pool. */
typedef struct CMLBlockTable {
    int* block_ids;     /* Array of block IDs for this sequence */
    int num_blocks;
    int capacity;
    int seq_len;        /* Total tokens in this sequence */
} CMLBlockTable;

/* Pool of fixed-size blocks shared across sequences, with stack-based free list. */
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

CMLPagedKVCache* cml_paged_kv_cache_create(int max_blocks, int max_sequences,
                                            int num_kv_heads, int head_dim);

void cml_paged_kv_cache_free(CMLPagedKVCache* cache);

/* Returns block ID >= 0, or -1 if pool exhausted. */
int cml_paged_cache_alloc_block(CMLPagedKVCache* cache);

void cml_paged_cache_free_block(CMLPagedKVCache* cache, int block_id);

/* Returns sequence ID >= 0, or -1 if max sequences reached. */
int cml_paged_cache_init_sequence(CMLPagedKVCache* cache);

void cml_paged_cache_free_sequence(CMLPagedKVCache* cache, int seq_id);

/* Allocates a new block automatically when the current tail block is full. */
int cml_paged_cache_append(CMLPagedKVCache* cache, int seq_id,
                            const float* key, const float* value);

Tensor* cml_paged_gqa_forward(CMLPagedKVCache* cache, int seq_id,
                               Tensor* Q, const CMLGQAConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_PAGED_ATTENTION_H */
