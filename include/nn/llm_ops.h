#ifndef CML_LLM_OPS_H
#define CML_LLM_OPS_H

#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLKVCache {
    Tensor* key_cache;   /* [max_seq_len, num_kv_heads, head_dim] */
    Tensor* value_cache; /* [max_seq_len, num_kv_heads, head_dim] */
    int max_seq_len;
    int current_len; /* Current sequence length in cache */
    int num_kv_heads;
    int head_dim;
} CMLKVCache;

/**
 * In-place row-wise softmax over a [rows, cols] float matrix, max-shifted for
 * numerical stability. Shared by the attention kernels in this family.
 * @param data (float*)
 * @param rows (int)
 * @param cols (int)
 * @return (void)
 */
void cml_softmax_rows_inplace(float* data, int rows, int cols);

/**
 * Temperature softmax of one length-`n` logits row into `out`, max-shifted for
 * numerical stability. `t` must be > 0 (callers apply their own guard).
 * @param in (const float*)
 * @param n (int)
 * @param t (float)
 * @param out (float*)
 * @return (void)
 */
void cml_softmax_row_temp(const float* in, int n, float t, float* out);

CMLKVCache* cml_kv_cache_create(int max_seq_len, int num_kv_heads, int head_dim);

void cml_kv_cache_free(CMLKVCache* cache);

int cml_kv_cache_append(CMLKVCache* cache, Tensor* new_key, Tensor* new_value);

void cml_kv_cache_reset(CMLKVCache* cache);

Tensor* cml_kv_cache_get_keys(CMLKVCache* cache);

Tensor* cml_kv_cache_get_values(CMLKVCache* cache);

typedef struct CMLGQAConfig {
    int num_heads;       /* Total attention heads (Q) */
    int num_kv_heads;    /* Number of KV heads (K, V) - can be < num_heads */
    int head_dim;        /* Dimension per head */
    float scale;         /* Attention scale (default: 1/sqrt(head_dim)) */
    bool causal;         /* Apply causal mask */
    int window_size;     /* Sliding window size (0 = unlimited) */
    int num_sink_tokens; /* StreamingLLM: first N tokens stay attended past the window (0 = off) */
    bool use_alibi;      /* Add ALiBi linear position bias instead of relying on RoPE */
} CMLGQAConfig;

typedef struct CMLFlashAttentionConfig {
    int tile_size_q;  /* Tile size for Q dimension (default: 64) */
    int tile_size_kv; /* Tile size for KV dimension (default: 64) */
    bool enabled;     /* Enable flash attention */
} CMLFlashAttentionConfig;

CMLFlashAttentionConfig cml_flash_attention_default_config(void);

/**
 * Write the num_heads ALiBi per-head slopes into out (geometric sequence 2^(-8i/n)).
 * @param num_heads (int)
 * @param out (float*)
 * @return (void)
 */
void cml_alibi_slopes(int num_heads, float* out);

/**
 * Fold one KV tile into a Q tile's FlashAttention online-softmax state (shared by the local
 * and ring attention kernels). Masking uses global positions q_pos0+qi / kv_pos0+ki. Query
 * rows are q_head[qi*q_stride + d]; KV rows k_head/v_head[ki*kv_stride + d]. State buffers
 * row_max (init -1e30f), row_sum (init 0), acc (init 0) and scratch scores[tq_len*tkv_len]
 * are caller-owned; normalize acc by row_sum after the last tile. The first num_sink keys stay
 * attended past the sliding window; a non-zero alibi_slope adds the ALiBi -slope*(sq-sk) bias.
 * @param row_max (float*)
 * @param row_sum (float*)
 * @param acc (float*)
 * @param scores (float*)
 * @param q_head (const float*)
 * @param q_stride (int)
 * @param tq_len (int)
 * @param q_pos0 (int)
 * @param k_head (const float*)
 * @param v_head (const float*)
 * @param kv_stride (int)
 * @param tkv_len (int)
 * @param kv_pos0 (int)
 * @param head_dim (int)
 * @param scale (float)
 * @param causal (bool)
 * @param window_size (int)
 * @param num_sink (int)
 * @param alibi_slope (float)
 * @return (void)
 */
void cml_flash_attention_fold_tile(float* row_max, float* row_sum, float* acc, float* scores,
                                   const float* q_head, int q_stride, int tq_len, int q_pos0,
                                   const float* k_head, const float* v_head, int kv_stride,
                                   int tkv_len, int kv_pos0, int head_dim, float scale, bool causal,
                                   int window_size, int num_sink, float alibi_slope);

/**
 * Flash GQA forward pass (O(N) memory via online softmax)
 * Q: [batch, seq_len, num_heads * head_dim]
 * K: [batch, kv_len, num_kv_heads * head_dim]
 * V: [batch, kv_len, num_kv_heads * head_dim]
 * Returns: [batch, seq_len, num_heads * head_dim]
 * @param Q (Tensor*)
 * @param K (Tensor*)
 * @param V (Tensor*)
 * @param config (const CMLGQAConfig*)
 * @param flash_config (const CMLFlashAttentionConfig*)
 * @return (Tensor*)
 */
Tensor* cml_gqa_flash_forward(Tensor* Q, Tensor* K, Tensor* V, const CMLGQAConfig* config,
                              const CMLFlashAttentionConfig* flash_config);

Tensor* cml_gqa_flash_forward_cached(Tensor* Q, Tensor* K, Tensor* V, CMLKVCache* kv_cache,
                                     const CMLGQAConfig* config,
                                     const CMLFlashAttentionConfig* flash_config);

/**
 * Grouped Query Attention forward pass
 * Q: [batch, seq_len, num_heads * head_dim]
 * K: [batch, kv_len, num_kv_heads * head_dim]
 * V: [batch, kv_len, num_kv_heads * head_dim]
 * Returns: [batch, seq_len, num_heads * head_dim]
 * @param Q (Tensor*)
 * @param K (Tensor*)
 * @param V (Tensor*)
 * @param config (const CMLGQAConfig*)
 * @param mask (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_gqa_forward(Tensor* Q, Tensor* K, Tensor* V, const CMLGQAConfig* config, Tensor* mask);

Tensor* cml_gqa_forward_cached(Tensor* Q, Tensor* K, Tensor* V, CMLKVCache* kv_cache,
                               const CMLGQAConfig* config);

typedef enum {
    CML_ROPE_SCALING_NONE = 0, /* No context-length scaling */
    CML_ROPE_SCALING_LINEAR,   /* Position Interpolation: positions divided by factor */
    CML_ROPE_SCALING_NTK,      /* NTK-aware: base scaled by factor^(dim/(dim-2)) */
} CMLRoPEScalingType;

typedef struct CMLRoPEConfig {
    int dim;                    /* Embedding dimension */
    int max_seq_len;            /* Maximum sequence length */
    float base;                 /* Base frequency (default: 10000.0) */
    CMLRoPEScalingType scaling; /* Context-extension scaling mode (default: NONE) */
    float scaling_factor;       /* Context-extension factor (>1 extends; 0/1 = no scaling) */
} CMLRoPEConfig;

/**
 * Apply RoPE to Q and K tensors in-place
 * x: [batch, seq_len, num_heads, head_dim]
 * position_ids: [batch, seq_len] or NULL for 0..seq_len-1
 * @param x (Tensor*)
 * @param start_pos (int)
 * @param config (const CMLRoPEConfig*)
 * @return (Tensor*)
 */
Tensor* cml_rope_forward(Tensor* x, int start_pos, const CMLRoPEConfig* config);

typedef struct CMLMoEConfig {
    int num_experts;        /* Total number of experts */
    int top_k;              /* Number of experts to route to */
    int input_dim;          /* Input dimension */
    int hidden_dim;         /* Expert hidden dimension */
    float capacity_factor;  /* Load balancing capacity factor */
    bool normalize_weights; /* Normalize gating weights to sum to 1 */
} CMLMoEConfig;

typedef struct CMLMoELayer {
    CMLMoEConfig config;
    Tensor* gate_weight; /* [input_dim, num_experts] - gating network */
    Tensor** expert_w1;  /* [num_experts] x [input_dim, hidden_dim] */
    Tensor** expert_w2;  /* [num_experts] x [hidden_dim, input_dim] */
    int ref_count;
} CMLMoELayer;

CMLMoELayer* cml_moe_create(const CMLMoEConfig* config);

void cml_moe_free(CMLMoELayer* moe);

/**
 * MoE forward pass
 * input: [batch, seq_len, input_dim]
 * Returns: [batch, seq_len, input_dim]
 * @param moe (CMLMoELayer*)
 * @param input (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_moe_forward(CMLMoELayer* moe, Tensor* input);

/**
 * Get expert routing (for debugging/analysis)
 * Returns gating weights: [batch * seq_len, num_experts]
 * @param moe (CMLMoELayer*)
 * @param input (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_moe_get_routing(CMLMoELayer* moe, Tensor* input);

typedef struct CMLBPEMerge {
    char* pair;       /* "ab" merged token */
    int new_token_id; /* ID of merged token */
} CMLBPEMerge;

typedef struct CMLTokenizer {
    char** vocab; /* Token strings indexed by ID */
    int vocab_size;
    CMLBPEMerge* merges; /* BPE merge rules */
    int num_merges;
    int* token_to_id; /* Hash-based token to ID lookup (internal) */
    int hash_size;
    /* Special tokens */
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;
    int unk_token_id;
} CMLTokenizer;

CMLTokenizer* cml_tokenizer_create(char** vocab, int vocab_size, char** merge_pairs,
                                   int num_merges);

/**
 * Train a byte-level BPE tokenizer on a corpus: start from the distinct characters, then
 * merge the most frequent adjacent pair until the vocabulary reaches target_vocab_size
 * (or no pair repeats). Returns a ready-to-use tokenizer; free with cml_tokenizer_free.
 * @param corpus (const char**)
 * @param num_docs (int)
 * @param target_vocab_size (int)
 * @return (CMLTokenizer*)
 */
CMLTokenizer* cml_tokenizer_train(const char** corpus, int num_docs, int target_vocab_size);

void cml_tokenizer_free(CMLTokenizer* tok);

int* cml_tokenizer_encode(CMLTokenizer* tok, const char* text, int* num_tokens);

char* cml_tokenizer_decode(CMLTokenizer* tok, const int* tokens, int num_tokens);

void cml_tokenizer_set_special(CMLTokenizer* tok, int bos, int eos, int pad, int unk);

int cml_tokenizer_vocab_size(const CMLTokenizer* tok);

#ifdef __cplusplus
}
#endif

#endif /* CML_LLM_OPS_H */
