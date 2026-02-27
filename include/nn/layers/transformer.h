/**
 * @file transformer.h
 * @brief Transformer layers: MultiHeadAttention and TransformerEncoderLayer
 */

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

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_TRANSFORMER_H
