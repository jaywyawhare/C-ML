/**
 * @file llm_ops.h
 * @brief LLM inference primitives
 *
 * Provides optimized components for large language model inference:
 * - Grouped Query Attention (GQA) with KV cache
 * - Mixture of Experts (MoE) with top-k gating
 * - BPE tokenizer
 * - RoPE (Rotary Position Embeddings)
 */

#ifndef CML_LLM_OPS_H
#define CML_LLM_OPS_H

#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===== KV Cache ===== */

typedef struct CMLKVCache {
    Tensor* key_cache;     /* [max_seq_len, num_kv_heads, head_dim] */
    Tensor* value_cache;   /* [max_seq_len, num_kv_heads, head_dim] */
    int max_seq_len;
    int current_len;       /* Current sequence length in cache */
    int num_kv_heads;
    int head_dim;
} CMLKVCache;

/** Create KV cache */
CMLKVCache* cml_kv_cache_create(int max_seq_len, int num_kv_heads, int head_dim);

/** Free KV cache */
void cml_kv_cache_free(CMLKVCache* cache);

/** Append new key/value to cache, returns updated length */
int cml_kv_cache_append(CMLKVCache* cache, Tensor* new_key, Tensor* new_value);

/** Reset cache (for new sequence) */
void cml_kv_cache_reset(CMLKVCache* cache);

/** Get cached keys up to current length */
Tensor* cml_kv_cache_get_keys(CMLKVCache* cache);

/** Get cached values up to current length */
Tensor* cml_kv_cache_get_values(CMLKVCache* cache);

/* ===== Grouped Query Attention ===== */

typedef struct CMLGQAConfig {
    int num_heads;          /* Total attention heads (Q) */
    int num_kv_heads;       /* Number of KV heads (K, V) - can be < num_heads */
    int head_dim;           /* Dimension per head */
    float scale;            /* Attention scale (default: 1/sqrt(head_dim)) */
    bool causal;            /* Apply causal mask */
} CMLGQAConfig;

/** Grouped Query Attention forward pass
 * Q: [batch, seq_len, num_heads * head_dim]
 * K: [batch, kv_len, num_kv_heads * head_dim]
 * V: [batch, kv_len, num_kv_heads * head_dim]
 * Returns: [batch, seq_len, num_heads * head_dim]
 */
Tensor* cml_gqa_forward(Tensor* Q, Tensor* K, Tensor* V, const CMLGQAConfig* config,
                         Tensor* mask);

/** GQA with KV cache (for autoregressive decoding) */
Tensor* cml_gqa_forward_cached(Tensor* Q, Tensor* K, Tensor* V,
                                CMLKVCache* kv_cache,
                                const CMLGQAConfig* config);

/* ===== Rotary Position Embeddings (RoPE) ===== */

typedef struct CMLRoPEConfig {
    int dim;               /* Embedding dimension */
    int max_seq_len;       /* Maximum sequence length */
    float base;            /* Base frequency (default: 10000.0) */
} CMLRoPEConfig;

/** Apply RoPE to Q and K tensors in-place
 * x: [batch, seq_len, num_heads, head_dim]
 * position_ids: [batch, seq_len] or NULL for 0..seq_len-1
 */
Tensor* cml_rope_forward(Tensor* x, int start_pos, const CMLRoPEConfig* config);

/* ===== Mixture of Experts (MoE) ===== */

typedef struct CMLMoEConfig {
    int num_experts;       /* Total number of experts */
    int top_k;             /* Number of experts to route to */
    int input_dim;         /* Input dimension */
    int hidden_dim;        /* Expert hidden dimension */
    float capacity_factor; /* Load balancing capacity factor */
    bool normalize_weights;/* Normalize gating weights to sum to 1 */
} CMLMoEConfig;

typedef struct CMLMoELayer {
    CMLMoEConfig config;
    Tensor* gate_weight;           /* [input_dim, num_experts] - gating network */
    Tensor** expert_w1;            /* [num_experts] x [input_dim, hidden_dim] */
    Tensor** expert_w2;            /* [num_experts] x [hidden_dim, input_dim] */
    int ref_count;
} CMLMoELayer;

/** Create MoE layer */
CMLMoELayer* cml_moe_create(const CMLMoEConfig* config);

/** Free MoE layer */
void cml_moe_free(CMLMoELayer* moe);

/** MoE forward pass
 * input: [batch, seq_len, input_dim]
 * Returns: [batch, seq_len, input_dim]
 */
Tensor* cml_moe_forward(CMLMoELayer* moe, Tensor* input);

/** Get expert routing (for debugging/analysis)
 * Returns gating weights: [batch * seq_len, num_experts]
 */
Tensor* cml_moe_get_routing(CMLMoELayer* moe, Tensor* input);

/* ===== BPE Tokenizer ===== */

typedef struct CMLBPEMerge {
    char* pair;            /* "ab" merged token */
    int new_token_id;      /* ID of merged token */
} CMLBPEMerge;

typedef struct CMLTokenizer {
    char** vocab;          /* Token strings indexed by ID */
    int vocab_size;
    CMLBPEMerge* merges;   /* BPE merge rules */
    int num_merges;
    int* token_to_id;      /* Hash-based token to ID lookup (internal) */
    int hash_size;
    /* Special tokens */
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;
    int unk_token_id;
} CMLTokenizer;

/** Create tokenizer from vocab and merges arrays */
CMLTokenizer* cml_tokenizer_create(char** vocab, int vocab_size,
                                     char** merge_pairs, int num_merges);

/** Free tokenizer */
void cml_tokenizer_free(CMLTokenizer* tok);

/** Encode text to token IDs
 * Returns array of token IDs, sets num_tokens
 */
int* cml_tokenizer_encode(CMLTokenizer* tok, const char* text, int* num_tokens);

/** Decode token IDs to text
 * Returns allocated string
 */
char* cml_tokenizer_decode(CMLTokenizer* tok, const int* tokens, int num_tokens);

/** Set special token IDs */
void cml_tokenizer_set_special(CMLTokenizer* tok, int bos, int eos, int pad, int unk);

/** Get vocab size */
int cml_tokenizer_vocab_size(const CMLTokenizer* tok);

#ifdef __cplusplus
}
#endif

#endif /* CML_LLM_OPS_H */
