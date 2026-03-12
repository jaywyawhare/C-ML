#ifndef CML_NN_LAYERS_TRANSFORMER_H
#define CML_NN_LAYERS_TRANSFORMER_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MultiHeadAttention {
    Module base;
    int embed_dim;
    int num_heads;
    int head_dim;
    float dropout;
    Parameter* W_q; // [embed_dim, embed_dim]
    Parameter* W_k; // [embed_dim, embed_dim]
    Parameter* W_v; // [embed_dim, embed_dim]
    Parameter* W_o; // [embed_dim, embed_dim]
    Parameter* b_q; // [embed_dim]
    Parameter* b_k; // [embed_dim]
    Parameter* b_v; // [embed_dim]
    Parameter* b_o; // [embed_dim]
} MultiHeadAttention;

MultiHeadAttention* nn_multihead_attention(int embed_dim, int num_heads, float dropout,
                                            DType dtype, DeviceType device);
Tensor* multihead_attention_forward(MultiHeadAttention* mha, Tensor* query, Tensor* key,
                                     Tensor* value, Tensor* mask);

typedef struct TransformerEncoderLayer {
    Module base;
    int d_model;
    int nhead;
    int dim_feedforward;
    float dropout;
    MultiHeadAttention* self_attn;
    Parameter* linear1_weight; // [dim_feedforward, d_model]
    Parameter* linear1_bias;   // [dim_feedforward]
    Parameter* linear2_weight; // [d_model, dim_feedforward]
    Parameter* linear2_bias;   // [d_model]
    Parameter* norm1_weight;   // [d_model]
    Parameter* norm1_bias;     // [d_model]
    Parameter* norm2_weight;   // [d_model]
    Parameter* norm2_bias;     // [d_model]
    float norm_eps;
} TransformerEncoderLayer;

TransformerEncoderLayer* nn_transformer_encoder_layer(int d_model, int nhead, int dim_feedforward,
                                                       float dropout, DType dtype, DeviceType device);

typedef struct TransformerEncoder {
    Module base;
    int num_layers;
    int d_model;
    TransformerEncoderLayer** layers;
    Parameter* norm_weight;  // Final layer norm [d_model]
    Parameter* norm_bias;    // Final layer norm [d_model]
    float norm_eps;
} TransformerEncoder;

TransformerEncoder* nn_transformer_encoder(int d_model, int nhead, int dim_feedforward,
                                            float dropout, int num_layers,
                                            DType dtype, DeviceType device);

typedef struct TransformerDecoderLayer {
    Module base;
    int d_model;
    int nhead;
    int dim_feedforward;
    float dropout;
    MultiHeadAttention* self_attn;   // Self-attention
    MultiHeadAttention* cross_attn;  // Cross-attention (encoder-decoder)
    Parameter* linear1_weight;  // [dim_feedforward, d_model]
    Parameter* linear1_bias;    // [dim_feedforward]
    Parameter* linear2_weight;  // [d_model, dim_feedforward]
    Parameter* linear2_bias;    // [d_model]
    Parameter* norm1_weight;    // [d_model] (after self-attn)
    Parameter* norm1_bias;
    Parameter* norm2_weight;    // [d_model] (after cross-attn)
    Parameter* norm2_bias;
    Parameter* norm3_weight;    // [d_model] (after FFN)
    Parameter* norm3_bias;
    float norm_eps;
} TransformerDecoderLayer;

TransformerDecoderLayer* nn_transformer_decoder_layer(int d_model, int nhead, int dim_feedforward,
                                                       float dropout, DType dtype, DeviceType device);

Tensor* transformer_decoder_layer_forward(TransformerDecoderLayer* layer, Tensor* tgt,
                                           Tensor* memory, Tensor* tgt_mask, Tensor* memory_mask);

typedef struct TransformerDecoder {
    Module base;
    int num_layers;
    int d_model;
    TransformerDecoderLayer** layers;
    Parameter* norm_weight;  // Final layer norm [d_model]
    Parameter* norm_bias;
    float norm_eps;
} TransformerDecoder;

TransformerDecoder* nn_transformer_decoder(int d_model, int nhead, int dim_feedforward,
                                            float dropout, int num_layers,
                                            DType dtype, DeviceType device);

/**
 * @brief Flash attention configuration
 *
 * When enabled, uses tiled Q/K/V computation with online softmax
 * (FlashAttention-2 algorithm on CPU) for memory-efficient attention.
 */
typedef struct FlashAttentionConfig {
    bool enabled;
    int block_size_q;   // Tile size for queries (default: 64)
    int block_size_kv;  // Tile size for keys/values (default: 64)
    bool causal;        // Whether to apply causal masking
} FlashAttentionConfig;

/**
 * @brief Key-value cache for autoregressive generation
 *
 * Caches key and value projections from previous timesteps to avoid
 * redundant computation during inference.
 */
typedef struct KVCache {
    Tensor* key_cache;    // [batch, num_heads, max_seq_len, head_dim]
    Tensor* value_cache;  // [batch, num_heads, max_seq_len, head_dim]
    int max_seq_len;
    int current_len;
} KVCache;

/**
 * @brief Create a KV cache for autoregressive generation
 *
 * @param batch Batch size
 * @param num_heads Number of attention heads
 * @param max_seq_len Maximum sequence length to cache
 * @param head_dim Dimension per head
 * @param dtype Data type
 * @param device Device type
 * @return New KVCache, or NULL on failure
 */
KVCache* kv_cache_create(int batch, int num_heads, int max_seq_len, int head_dim,
                          DType dtype, DeviceType device);

/**
 * @brief Free a KV cache
 * @param cache KV cache to free
 */
void kv_cache_free(KVCache* cache);

/**
 * @brief Reset a KV cache (set current_len to 0)
 * @param cache KV cache to reset
 */
void kv_cache_reset(KVCache* cache);

/**
 * @brief Forward pass with flash attention
 *
 * @param mha Multi-head attention module
 * @param query Query tensor [batch, seq_len, embed_dim]
 * @param key Key tensor
 * @param value Value tensor
 * @param mask Optional attention mask
 * @param config Flash attention configuration
 * @return Output tensor, or NULL on failure
 */
Tensor* flash_attention_forward(MultiHeadAttention* mha, Tensor* query, Tensor* key,
                                 Tensor* value, Tensor* mask, FlashAttentionConfig* config);

/**
 * @brief Forward pass with KV cache (for autoregressive generation)
 *
 * @param mha Multi-head attention module
 * @param query Query tensor (single token: [batch, 1, embed_dim])
 * @param key Key tensor
 * @param value Value tensor
 * @param mask Optional attention mask
 * @param cache KV cache to update
 * @return Output tensor, or NULL on failure
 */
Tensor* multihead_attention_forward_cached(MultiHeadAttention* mha, Tensor* query,
                                            Tensor* key, Tensor* value,
                                            Tensor* mask, KVCache* cache);

/**
 * @brief Configure flash attention on a multi-head attention module
 *
 * @param mha Multi-head attention module
 * @param enabled Enable flash attention
 * @param causal Use causal masking
 */
void multihead_attention_set_flash(MultiHeadAttention* mha, bool enabled, bool causal);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_TRANSFORMER_H
