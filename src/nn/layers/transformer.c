/**
 * @file transformer.c
 * @brief Transformer layers: MultiHeadAttention and TransformerEncoderLayer
 */

#include "nn/layers/transformer.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Batch matrix multiply: [B, M, K] @ [B, K, N] -> [B, M, N]
static void batch_matmul(float* out, const float* a, const float* b,
                          int batch, int M, int K, int N) {
    for (int bi = 0; bi < batch; bi++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += a[bi*M*K + m*K + k] * b[bi*K*N + k*N + n];
                }
                out[bi*M*N + m*N + n] = sum;
            }
        }
    }
}

// Softmax over last dim for [total_rows, dim]
static void softmax_inplace(float* data, int total_rows, int dim) {
    for (int r = 0; r < total_rows; r++) {
        float max_val = data[r * dim];
        for (int d = 1; d < dim; d++) {
            if (data[r * dim + d] > max_val) max_val = data[r * dim + d];
        }
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            data[r * dim + d] = expf(data[r * dim + d] - max_val);
            sum += data[r * dim + d];
        }
        for (int d = 0; d < dim; d++) {
            data[r * dim + d] /= sum;
        }
    }
}

// Linear projection: y = x @ W^T + b for [total, in_dim] -> [total, out_dim]
static void linear_project(float* out, const float* in, const float* weight, const float* bias,
                            int total, int in_dim, int out_dim) {
    for (int t = 0; t < total; t++) {
        for (int o = 0; o < out_dim; o++) {
            float sum = bias ? bias[o] : 0.0f;
            for (int i = 0; i < in_dim; i++) {
                sum += in[t * in_dim + i] * weight[o * in_dim + i];
            }
            out[t * out_dim + o] = sum;
        }
    }
}

// Layer normalization in-place over last dim
static void layer_norm_inplace(float* data, int total, int dim, float* weight, float* bias, float eps) {
    for (int t = 0; t < total; t++) {
        float mean = 0.0f;
        for (int d = 0; d < dim; d++) mean += data[t * dim + d];
        mean /= dim;
        float var = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = data[t * dim + d] - mean;
            var += diff * diff;
        }
        var /= dim;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int d = 0; d < dim; d++) {
            data[t * dim + d] = (data[t * dim + d] - mean) * inv_std;
            if (weight) data[t * dim + d] = data[t * dim + d] * weight[d] + bias[d];
        }
    }
}

// Xavier initialization
static void xavier_init(float* data, size_t numel, int fan_in, int fan_out) {
    float scale = sqrtf(2.0f / (float)(fan_in + fan_out));
    for (size_t i = 0; i < numel; i++) {
        data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

static Tensor* mha_module_forward(Module* module, Tensor* input) {
    // Module forward wraps multihead_attention_forward with query=key=value=input, no mask
    MultiHeadAttention* mha = (MultiHeadAttention*)module;
    return multihead_attention_forward(mha, input, input, input, NULL);
}

static void mha_free(Module* module) {
    MultiHeadAttention* mha = (MultiHeadAttention*)module;
    if (!mha) return;
    free(mha);
}

MultiHeadAttention* nn_multihead_attention(int embed_dim, int num_heads, float dropout,
                                            DType dtype, DeviceType device) {
    if (embed_dim % num_heads != 0) {
        LOG_ERROR("embed_dim (%d) must be divisible by num_heads (%d)", embed_dim, num_heads);
        return NULL;
    }

    MultiHeadAttention* mha = malloc(sizeof(MultiHeadAttention));
    if (!mha) {
        LOG_ERROR("Failed to allocate MultiHeadAttention");
        return NULL;
    }

    if (module_init((Module*)mha, "MultiHeadAttention", mha_module_forward, mha_free) != 0) {
        free(mha);
        return NULL;
    }

    mha->embed_dim = embed_dim;
    mha->num_heads = num_heads;
    mha->head_dim = embed_dim / num_heads;
    mha->dropout = dropout;

    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};

    // Create projection weight tensors [embed_dim, embed_dim]
    int weight_shape[] = {embed_dim, embed_dim};
    int bias_shape[] = {embed_dim};

    // Helper struct for parameter creation
    struct { const char* w_name; const char* b_name; Parameter** w_ptr; Parameter** b_ptr; } params[] = {
        {"W_q", "b_q", &mha->W_q, &mha->b_q},
        {"W_k", "b_k", &mha->W_k, &mha->b_k},
        {"W_v", "b_v", &mha->W_v, &mha->b_v},
        {"W_o", "b_o", &mha->W_o, &mha->b_o},
    };

    for (int i = 0; i < 4; i++) {
        // Weight
        Tensor* w = tensor_empty(weight_shape, 2, &config);
        if (!w) {
            module_free((Module*)mha);
            return NULL;
        }
        float* w_data = (float*)tensor_data_ptr(w);
        if (w_data) xavier_init(w_data, (size_t)embed_dim * embed_dim, embed_dim, embed_dim);

        if (module_add_parameter((Module*)mha, w, params[i].w_name, true) != 0) {
            tensor_free(w);
            module_free((Module*)mha);
            return NULL;
        }
        *params[i].w_ptr = module_get_parameter((Module*)mha, params[i].w_name);

        // Bias
        Tensor* b = tensor_zeros(bias_shape, 1, &config);
        if (!b) {
            module_free((Module*)mha);
            return NULL;
        }

        if (module_add_parameter((Module*)mha, b, params[i].b_name, true) != 0) {
            tensor_free(b);
            module_free((Module*)mha);
            return NULL;
        }
        *params[i].b_ptr = module_get_parameter((Module*)mha, params[i].b_name);
    }

    LOG_DEBUG("Created MultiHeadAttention: embed_dim=%d, num_heads=%d, head_dim=%d",
              embed_dim, num_heads, mha->head_dim);

    return mha;
}

Tensor* multihead_attention_forward(MultiHeadAttention* mha, Tensor* query, Tensor* key,
                                     Tensor* value, Tensor* mask) {
    if (!mha || !query || !key || !value) {
        LOG_ERROR("MultiHeadAttention forward: NULL input");
        return NULL;
    }

    // Ensure inputs are executed
    tensor_ensure_executed(query);
    tensor_ensure_executed(key);
    tensor_ensure_executed(value);

    if (query->ndim != 3 || key->ndim != 3 || value->ndim != 3) {
        LOG_ERROR("MultiHeadAttention forward: expected 3D inputs [batch, seq, embed_dim]");
        return NULL;
    }

    int batch = query->shape[0];
    int seq_q = query->shape[1];
    int seq_k = key->shape[1];
    int embed_dim = mha->embed_dim;
    int num_heads = mha->num_heads;
    int head_dim = mha->head_dim;
    int total_q = batch * seq_q;
    int total_k = batch * seq_k;

    // Get parameter data
    float* wq_data = (float*)tensor_data_ptr(mha->W_q->tensor);
    float* wk_data = (float*)tensor_data_ptr(mha->W_k->tensor);
    float* wv_data = (float*)tensor_data_ptr(mha->W_v->tensor);
    float* wo_data = (float*)tensor_data_ptr(mha->W_o->tensor);
    float* bq_data = (float*)tensor_data_ptr(mha->b_q->tensor);
    float* bk_data = (float*)tensor_data_ptr(mha->b_k->tensor);
    float* bv_data = (float*)tensor_data_ptr(mha->b_v->tensor);
    float* bo_data = (float*)tensor_data_ptr(mha->b_o->tensor);
    float* q_in = (float*)tensor_data_ptr(query);
    float* k_in = (float*)tensor_data_ptr(key);
    float* v_in = (float*)tensor_data_ptr(value);

    if (!wq_data || !wk_data || !wv_data || !wo_data || !q_in || !k_in || !v_in) {
        LOG_ERROR("MultiHeadAttention forward: failed to get data pointers");
        return NULL;
    }

    // Step 1: Linear projections
    // Q: [total_q, embed_dim] -> [total_q, embed_dim]
    float* Q = malloc((size_t)total_q * embed_dim * sizeof(float));
    float* K = malloc((size_t)total_k * embed_dim * sizeof(float));
    float* V = malloc((size_t)total_k * embed_dim * sizeof(float));
    if (!Q || !K || !V) {
        LOG_ERROR("MultiHeadAttention forward: allocation failed");
        free(Q); free(K); free(V);
        return NULL;
    }

    linear_project(Q, q_in, wq_data, bq_data, total_q, embed_dim, embed_dim);
    linear_project(K, k_in, wk_data, bk_data, total_k, embed_dim, embed_dim);
    linear_project(V, v_in, wv_data, bv_data, total_k, embed_dim, embed_dim);

    // Step 2: Reshape to multi-head: [B, S, H, D] -> [B, H, S, D]
    // Q: [batch, seq_q, num_heads, head_dim] -> [batch, num_heads, seq_q, head_dim]
    float* Q_mh = malloc((size_t)batch * num_heads * seq_q * head_dim * sizeof(float));
    float* K_mh = malloc((size_t)batch * num_heads * seq_k * head_dim * sizeof(float));
    float* V_mh = malloc((size_t)batch * num_heads * seq_k * head_dim * sizeof(float));
    if (!Q_mh || !K_mh || !V_mh) {
        LOG_ERROR("MultiHeadAttention forward: allocation failed");
        free(Q); free(K); free(V); free(Q_mh); free(K_mh); free(V_mh);
        return NULL;
    }

    // Transpose [B, S, H, D] -> [B, H, S, D]
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_q; s++) {
            for (int h = 0; h < num_heads; h++) {
                for (int d = 0; d < head_dim; d++) {
                    Q_mh[b*num_heads*seq_q*head_dim + h*seq_q*head_dim + s*head_dim + d] =
                        Q[b*seq_q*embed_dim + s*embed_dim + h*head_dim + d];
                }
            }
        }
        for (int s = 0; s < seq_k; s++) {
            for (int h = 0; h < num_heads; h++) {
                for (int d = 0; d < head_dim; d++) {
                    K_mh[b*num_heads*seq_k*head_dim + h*seq_k*head_dim + s*head_dim + d] =
                        K[b*seq_k*embed_dim + s*embed_dim + h*head_dim + d];
                    V_mh[b*num_heads*seq_k*head_dim + h*seq_k*head_dim + s*head_dim + d] =
                        V[b*seq_k*embed_dim + s*embed_dim + h*head_dim + d];
                }
            }
        }
    }

    free(Q); free(K); free(V);

    // Step 3: Attention scores: [B*H, seq_q, head_dim] @ [B*H, head_dim, seq_k] -> [B*H, seq_q, seq_k]
    // First transpose K: [B*H, seq_k, head_dim] -> [B*H, head_dim, seq_k]
    int BH = batch * num_heads;
    float* K_t = malloc((size_t)BH * head_dim * seq_k * sizeof(float));
    if (!K_t) {
        LOG_ERROR("MultiHeadAttention forward: allocation failed");
        free(Q_mh); free(K_mh); free(V_mh);
        return NULL;
    }

    for (int bh = 0; bh < BH; bh++) {
        for (int s = 0; s < seq_k; s++) {
            for (int d = 0; d < head_dim; d++) {
                K_t[bh*head_dim*seq_k + d*seq_k + s] =
                    K_mh[bh*seq_k*head_dim + s*head_dim + d];
            }
        }
    }

    float* scores = malloc((size_t)BH * seq_q * seq_k * sizeof(float));
    if (!scores) {
        LOG_ERROR("MultiHeadAttention forward: allocation failed");
        free(Q_mh); free(K_mh); free(V_mh); free(K_t);
        return NULL;
    }

    batch_matmul(scores, Q_mh, K_t, BH, seq_q, head_dim, seq_k);
    free(K_t);
    free(Q_mh);
    free(K_mh);

    // Scale by 1/sqrt(head_dim)
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t scores_size = (size_t)BH * seq_q * seq_k;
    for (size_t i = 0; i < scores_size; i++) {
        scores[i] *= scale;
    }

    // Step 4: Apply mask if provided
    if (mask) {
        tensor_ensure_executed(mask);
        float* mask_data = (float*)tensor_data_ptr(mask);
        if (mask_data) {
            // Mask shape: [seq_q, seq_k] or [batch, seq_q, seq_k]
            // Broadcast across batch*heads
            for (int bh = 0; bh < BH; bh++) {
                for (int sq = 0; sq < seq_q; sq++) {
                    for (int sk = 0; sk < seq_k; sk++) {
                        size_t mask_idx;
                        if (mask->ndim == 2) {
                            mask_idx = (size_t)sq * seq_k + sk;
                        } else {
                            int b_idx = bh / num_heads;
                            mask_idx = (size_t)b_idx * seq_q * seq_k + sq * seq_k + sk;
                        }
                        if (mask_data[mask_idx] == 0.0f) {
                            scores[bh*seq_q*seq_k + sq*seq_k + sk] = -1e9f;
                        }
                    }
                }
            }
        }
    }

    // Step 5: Softmax over last dim
    softmax_inplace(scores, BH * seq_q, seq_k);

    // Step 6: scores @ V -> [B*H, seq_q, head_dim]
    float* attn_out = malloc((size_t)BH * seq_q * head_dim * sizeof(float));
    if (!attn_out) {
        LOG_ERROR("MultiHeadAttention forward: allocation failed");
        free(scores); free(V_mh);
        return NULL;
    }

    batch_matmul(attn_out, scores, V_mh, BH, seq_q, seq_k, head_dim);
    free(scores);
    free(V_mh);

    // Step 7: Transpose back [B, H, seq_q, head_dim] -> [B, seq_q, H, head_dim] -> [B, seq_q, embed_dim]
    float* concat = malloc((size_t)batch * seq_q * embed_dim * sizeof(float));
    if (!concat) {
        LOG_ERROR("MultiHeadAttention forward: allocation failed");
        free(attn_out);
        return NULL;
    }

    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_q; s++) {
            for (int h = 0; h < num_heads; h++) {
                for (int d = 0; d < head_dim; d++) {
                    concat[b*seq_q*embed_dim + s*embed_dim + h*head_dim + d] =
                        attn_out[b*num_heads*seq_q*head_dim + h*seq_q*head_dim + s*head_dim + d];
                }
            }
        }
    }
    free(attn_out);

    // Step 8: Output projection: [total_q, embed_dim] -> [total_q, embed_dim]
    float* output_data = malloc((size_t)total_q * embed_dim * sizeof(float));
    if (!output_data) {
        LOG_ERROR("MultiHeadAttention forward: allocation failed");
        free(concat);
        return NULL;
    }

    linear_project(output_data, concat, wo_data, bo_data, total_q, embed_dim, embed_dim);
    free(concat);

    // Create output tensor [batch, seq_q, embed_dim]
    int out_shape[] = {batch, seq_q, embed_dim};
    TensorConfig out_config = {.dtype = query->dtype, .device = query->device,
                                .has_dtype = true, .has_device = true};
    Tensor* output = tensor_from_data(output_data, out_shape, 3, &out_config);
    free(output_data);

    if (!output) {
        LOG_ERROR("MultiHeadAttention forward: failed to create output tensor");
        return NULL;
    }

    return output;
}

static Tensor* encoder_layer_forward(Module* module, Tensor* input) {
    TransformerEncoderLayer* layer = (TransformerEncoderLayer*)module;

    if (!layer || !input) return NULL;

    tensor_ensure_executed(input);

    if (input->ndim != 3) {
        LOG_ERROR("TransformerEncoderLayer forward: expected 3D input [batch, seq, d_model]");
        return NULL;
    }

    int batch = input->shape[0];
    int seq_len = input->shape[1];
    int d_model = layer->d_model;
    int dim_ff = layer->dim_feedforward;
    int total = batch * seq_len;

    float* x_data = (float*)tensor_data_ptr(input);
    if (!x_data) return NULL;

    // Working buffer: copy input
    size_t buf_size = (size_t)total * d_model;
    float* x = malloc(buf_size * sizeof(float));
    if (!x) return NULL;
    memcpy(x, x_data, buf_size * sizeof(float));

    // Step 1: Self-attention
    Tensor* attn_out = multihead_attention_forward(layer->self_attn, input, input, input, NULL);
    if (!attn_out) {
        free(x);
        return NULL;
    }
    tensor_ensure_executed(attn_out);
    float* attn_data = (float*)tensor_data_ptr(attn_out);

    // Step 2: Residual connection: x = x + self_attention
    for (size_t i = 0; i < buf_size; i++) {
        x[i] += attn_data[i];
    }
    tensor_free(attn_out);

    // Step 3: Layer norm 1
    float* norm1_w = (float*)tensor_data_ptr(layer->norm1_weight->tensor);
    float* norm1_b = (float*)tensor_data_ptr(layer->norm1_bias->tensor);
    layer_norm_inplace(x, total, d_model, norm1_w, norm1_b, layer->norm_eps);

    // Step 4: Feedforward network
    // ff_hidden = relu(x @ linear1_weight^T + linear1_bias)
    float* l1_w = (float*)tensor_data_ptr(layer->linear1_weight->tensor);
    float* l1_b = (float*)tensor_data_ptr(layer->linear1_bias->tensor);
    float* l2_w = (float*)tensor_data_ptr(layer->linear2_weight->tensor);
    float* l2_b = (float*)tensor_data_ptr(layer->linear2_bias->tensor);

    float* ff_hidden = malloc((size_t)total * dim_ff * sizeof(float));
    if (!ff_hidden) {
        free(x);
        return NULL;
    }

    linear_project(ff_hidden, x, l1_w, l1_b, total, d_model, dim_ff);

    // ReLU
    for (size_t i = 0; i < (size_t)total * dim_ff; i++) {
        if (ff_hidden[i] < 0.0f) ff_hidden[i] = 0.0f;
    }

    // ff_out = ff_hidden @ linear2_weight^T + linear2_bias
    float* ff_out = malloc(buf_size * sizeof(float));
    if (!ff_out) {
        free(x); free(ff_hidden);
        return NULL;
    }

    linear_project(ff_out, ff_hidden, l2_w, l2_b, total, dim_ff, d_model);
    free(ff_hidden);

    // Step 5: Residual connection: x = x + ff_out
    for (size_t i = 0; i < buf_size; i++) {
        x[i] += ff_out[i];
    }
    free(ff_out);

    // Step 6: Layer norm 2
    float* norm2_w = (float*)tensor_data_ptr(layer->norm2_weight->tensor);
    float* norm2_b = (float*)tensor_data_ptr(layer->norm2_bias->tensor);
    layer_norm_inplace(x, total, d_model, norm2_w, norm2_b, layer->norm_eps);

    // Create output tensor
    int out_shape[] = {batch, seq_len, d_model};
    TensorConfig out_config = {.dtype = input->dtype, .device = input->device,
                                .has_dtype = true, .has_device = true};
    Tensor* output = tensor_from_data(x, out_shape, 3, &out_config);
    free(x);

    return output;
}

static void encoder_layer_free(Module* module) {
    TransformerEncoderLayer* layer = (TransformerEncoderLayer*)module;
    if (!layer) return;

    if (layer->self_attn) {
        module_free((Module*)layer->self_attn);
    }

    free(layer);
}

TransformerEncoderLayer* nn_transformer_encoder_layer(int d_model, int nhead, int dim_feedforward,
                                                       float dropout, DType dtype, DeviceType device) {
    TransformerEncoderLayer* layer = malloc(sizeof(TransformerEncoderLayer));
    if (!layer) {
        LOG_ERROR("Failed to allocate TransformerEncoderLayer");
        return NULL;
    }

    if (module_init((Module*)layer, "TransformerEncoderLayer", encoder_layer_forward, encoder_layer_free) != 0) {
        free(layer);
        return NULL;
    }

    layer->d_model = d_model;
    layer->nhead = nhead;
    layer->dim_feedforward = dim_feedforward;
    layer->dropout = dropout;
    layer->norm_eps = 1e-5f;

    // Create multi-head attention sub-module
    layer->self_attn = nn_multihead_attention(d_model, nhead, dropout, dtype, device);
    if (!layer->self_attn) {
        module_free((Module*)layer);
        return NULL;
    }

    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};

    // Feedforward weights
    int l1_w_shape[] = {dim_feedforward, d_model};
    int l1_b_shape[] = {dim_feedforward};
    int l2_w_shape[] = {d_model, dim_feedforward};
    int l2_b_shape[] = {d_model};
    int norm_shape[] = {d_model};

    // Linear1 weight [dim_feedforward, d_model]
    Tensor* l1w = tensor_empty(l1_w_shape, 2, &config);
    if (!l1w) { module_free((Module*)layer); return NULL; }
    float* l1w_data = (float*)tensor_data_ptr(l1w);
    if (l1w_data) xavier_init(l1w_data, (size_t)dim_feedforward * d_model, d_model, dim_feedforward);
    if (module_add_parameter((Module*)layer, l1w, "linear1_weight", true) != 0) {
        tensor_free(l1w); module_free((Module*)layer); return NULL;
    }
    layer->linear1_weight = module_get_parameter((Module*)layer, "linear1_weight");

    // Linear1 bias [dim_feedforward]
    Tensor* l1b = tensor_zeros(l1_b_shape, 1, &config);
    if (!l1b) { module_free((Module*)layer); return NULL; }
    if (module_add_parameter((Module*)layer, l1b, "linear1_bias", true) != 0) {
        tensor_free(l1b); module_free((Module*)layer); return NULL;
    }
    layer->linear1_bias = module_get_parameter((Module*)layer, "linear1_bias");

    // Linear2 weight [d_model, dim_feedforward]
    Tensor* l2w = tensor_empty(l2_w_shape, 2, &config);
    if (!l2w) { module_free((Module*)layer); return NULL; }
    float* l2w_data = (float*)tensor_data_ptr(l2w);
    if (l2w_data) xavier_init(l2w_data, (size_t)d_model * dim_feedforward, dim_feedforward, d_model);
    if (module_add_parameter((Module*)layer, l2w, "linear2_weight", true) != 0) {
        tensor_free(l2w); module_free((Module*)layer); return NULL;
    }
    layer->linear2_weight = module_get_parameter((Module*)layer, "linear2_weight");

    // Linear2 bias [d_model]
    Tensor* l2b = tensor_zeros(l2_b_shape, 1, &config);
    if (!l2b) { module_free((Module*)layer); return NULL; }
    if (module_add_parameter((Module*)layer, l2b, "linear2_bias", true) != 0) {
        tensor_free(l2b); module_free((Module*)layer); return NULL;
    }
    layer->linear2_bias = module_get_parameter((Module*)layer, "linear2_bias");

    // Norm1 weight and bias [d_model]
    Tensor* n1w = tensor_ones(norm_shape, 1, &config);
    if (!n1w) { module_free((Module*)layer); return NULL; }
    if (module_add_parameter((Module*)layer, n1w, "norm1_weight", true) != 0) {
        tensor_free(n1w); module_free((Module*)layer); return NULL;
    }
    layer->norm1_weight = module_get_parameter((Module*)layer, "norm1_weight");

    Tensor* n1b = tensor_zeros(norm_shape, 1, &config);
    if (!n1b) { module_free((Module*)layer); return NULL; }
    if (module_add_parameter((Module*)layer, n1b, "norm1_bias", true) != 0) {
        tensor_free(n1b); module_free((Module*)layer); return NULL;
    }
    layer->norm1_bias = module_get_parameter((Module*)layer, "norm1_bias");

    // Norm2 weight and bias [d_model]
    Tensor* n2w = tensor_ones(norm_shape, 1, &config);
    if (!n2w) { module_free((Module*)layer); return NULL; }
    if (module_add_parameter((Module*)layer, n2w, "norm2_weight", true) != 0) {
        tensor_free(n2w); module_free((Module*)layer); return NULL;
    }
    layer->norm2_weight = module_get_parameter((Module*)layer, "norm2_weight");

    Tensor* n2b = tensor_zeros(norm_shape, 1, &config);
    if (!n2b) { module_free((Module*)layer); return NULL; }
    if (module_add_parameter((Module*)layer, n2b, "norm2_bias", true) != 0) {
        tensor_free(n2b); module_free((Module*)layer); return NULL;
    }
    layer->norm2_bias = module_get_parameter((Module*)layer, "norm2_bias");

    LOG_DEBUG("Created TransformerEncoderLayer: d_model=%d, nhead=%d, dim_ff=%d",
              d_model, nhead, dim_feedforward);

    return layer;
}

static Tensor* transformer_encoder_forward(Module* module, Tensor* input) {
    TransformerEncoder* enc = (TransformerEncoder*)module;
    if (!enc || !input) return NULL;

    Tensor* x = input;
    bool owns_x = false;

    for (int i = 0; i < enc->num_layers; i++) {
        Tensor* out = module_forward((Module*)enc->layers[i], x);
        if (owns_x) tensor_free(x);
        if (!out) return NULL;
        x = out;
        owns_x = true;
    }

    // Final layer norm
    tensor_ensure_executed(x);
    float* x_data = (float*)tensor_data_ptr(x);
    if (!x_data) { if (owns_x) tensor_free(x); return NULL; }

    int total = 1;
    for (int i = 0; i < x->ndim - 1; i++) total *= x->shape[i];
    int d_model = enc->d_model;

    float* norm_w = (float*)tensor_data_ptr(enc->norm_weight->tensor);
    float* norm_b = (float*)tensor_data_ptr(enc->norm_bias->tensor);

    // Clone data for in-place normalization
    size_t buf_size = (size_t)total * d_model;
    float* buf = malloc(buf_size * sizeof(float));
    if (!buf) { if (owns_x) tensor_free(x); return NULL; }
    memcpy(buf, x_data, buf_size * sizeof(float));

    layer_norm_inplace(buf, total, d_model, norm_w, norm_b, enc->norm_eps);

    int* out_shape = malloc(x->ndim * sizeof(int));
    if (!out_shape) { free(buf); if (owns_x) tensor_free(x); return NULL; }
    memcpy(out_shape, x->shape, x->ndim * sizeof(int));
    int out_ndim = x->ndim;

    TensorConfig cfg = {.dtype = x->dtype, .device = x->device, .has_dtype = true, .has_device = true};
    if (owns_x) tensor_free(x);

    Tensor* output = tensor_from_data(buf, out_shape, out_ndim, &cfg);
    free(buf);
    free(out_shape);
    return output;
}

static void transformer_encoder_free(Module* module) {
    TransformerEncoder* enc = (TransformerEncoder*)module;
    if (!enc) return;
    if (enc->layers) {
        for (int i = 0; i < enc->num_layers; i++) {
            if (enc->layers[i]) module_free((Module*)enc->layers[i]);
        }
        free(enc->layers);
    }
    free(enc);
}

TransformerEncoder* nn_transformer_encoder(int d_model, int nhead, int dim_feedforward,
                                            float dropout, int num_layers,
                                            DType dtype, DeviceType device) {
    TransformerEncoder* enc = malloc(sizeof(TransformerEncoder));
    if (!enc) return NULL;

    if (module_init((Module*)enc, "TransformerEncoder", transformer_encoder_forward, transformer_encoder_free) != 0) {
        free(enc); return NULL;
    }

    enc->d_model = d_model;
    enc->num_layers = num_layers;
    enc->norm_eps = 1e-5f;

    enc->layers = malloc(num_layers * sizeof(TransformerEncoderLayer*));
    if (!enc->layers) { module_free((Module*)enc); return NULL; }

    for (int i = 0; i < num_layers; i++) {
        enc->layers[i] = nn_transformer_encoder_layer(d_model, nhead, dim_feedforward, dropout, dtype, device);
        if (!enc->layers[i]) {
            enc->num_layers = i;
            module_free((Module*)enc);
            return NULL;
        }
    }

    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    int norm_shape[] = {d_model};

    Tensor* nw = tensor_ones(norm_shape, 1, &config);
    if (!nw) { module_free((Module*)enc); return NULL; }
    if (module_add_parameter((Module*)enc, nw, "norm_weight", true) != 0) {
        tensor_free(nw); module_free((Module*)enc); return NULL;
    }
    enc->norm_weight = module_get_parameter((Module*)enc, "norm_weight");

    Tensor* nb = tensor_zeros(norm_shape, 1, &config);
    if (!nb) { module_free((Module*)enc); return NULL; }
    if (module_add_parameter((Module*)enc, nb, "norm_bias", true) != 0) {
        tensor_free(nb); module_free((Module*)enc); return NULL;
    }
    enc->norm_bias = module_get_parameter((Module*)enc, "norm_bias");

    return enc;
}

static Tensor* decoder_layer_forward_wrapper(Module* module, Tensor* input) {
    // Module forward wraps with tgt=input, memory=NULL (self-attention only mode)
    TransformerDecoderLayer* layer = (TransformerDecoderLayer*)module;
    return transformer_decoder_layer_forward(layer, input, NULL, NULL, NULL);
}

Tensor* transformer_decoder_layer_forward(TransformerDecoderLayer* layer, Tensor* tgt,
                                           Tensor* memory, Tensor* tgt_mask, Tensor* memory_mask) {
    if (!layer || !tgt) return NULL;

    tensor_ensure_executed(tgt);
    if (tgt->ndim != 3) return NULL;

    int batch = tgt->shape[0];
    int seq_len = tgt->shape[1];
    int d_model = layer->d_model;
    int dim_ff = layer->dim_feedforward;
    int total = batch * seq_len;

    float* x_data = (float*)tensor_data_ptr(tgt);
    if (!x_data) return NULL;

    size_t buf_size = (size_t)total * d_model;
    float* x = malloc(buf_size * sizeof(float));
    if (!x) return NULL;
    memcpy(x, x_data, buf_size * sizeof(float));

    // Step 1: Self-attention
    Tensor* self_attn_out = multihead_attention_forward(layer->self_attn, tgt, tgt, tgt, tgt_mask);
    if (!self_attn_out) { free(x); return NULL; }
    tensor_ensure_executed(self_attn_out);
    float* attn_data = (float*)tensor_data_ptr(self_attn_out);

    // Residual + LayerNorm1
    for (size_t i = 0; i < buf_size; i++) x[i] += attn_data[i];
    tensor_free(self_attn_out);

    float* n1w = (float*)tensor_data_ptr(layer->norm1_weight->tensor);
    float* n1b = (float*)tensor_data_ptr(layer->norm1_bias->tensor);
    layer_norm_inplace(x, total, d_model, n1w, n1b, layer->norm_eps);

    // Step 2: Cross-attention (if memory provided)
    if (memory) {
        tensor_ensure_executed(memory);
        // Create temporary tensor from x for cross-attention query
        int x_shape[] = {batch, seq_len, d_model};
        TensorConfig cfg = {.dtype = tgt->dtype, .device = tgt->device, .has_dtype = true, .has_device = true};
        Tensor* x_tensor = tensor_from_data(x, x_shape, 3, &cfg);
        if (!x_tensor) { free(x); return NULL; }

        Tensor* cross_out = multihead_attention_forward(layer->cross_attn, x_tensor, memory, memory, memory_mask);
        tensor_free(x_tensor);
        if (!cross_out) { free(x); return NULL; }
        tensor_ensure_executed(cross_out);
        float* cross_data = (float*)tensor_data_ptr(cross_out);

        for (size_t i = 0; i < buf_size; i++) x[i] += cross_data[i];
        tensor_free(cross_out);

        float* n2w = (float*)tensor_data_ptr(layer->norm2_weight->tensor);
        float* n2b = (float*)tensor_data_ptr(layer->norm2_bias->tensor);
        layer_norm_inplace(x, total, d_model, n2w, n2b, layer->norm_eps);
    }

    // Step 3: Feedforward
    float* l1w = (float*)tensor_data_ptr(layer->linear1_weight->tensor);
    float* l1b = (float*)tensor_data_ptr(layer->linear1_bias->tensor);
    float* l2w = (float*)tensor_data_ptr(layer->linear2_weight->tensor);
    float* l2b = (float*)tensor_data_ptr(layer->linear2_bias->tensor);

    float* ff_hidden = malloc((size_t)total * dim_ff * sizeof(float));
    if (!ff_hidden) { free(x); return NULL; }
    linear_project(ff_hidden, x, l1w, l1b, total, d_model, dim_ff);
    for (size_t i = 0; i < (size_t)total * dim_ff; i++)
        if (ff_hidden[i] < 0.0f) ff_hidden[i] = 0.0f;

    float* ff_out = malloc(buf_size * sizeof(float));
    if (!ff_out) { free(x); free(ff_hidden); return NULL; }
    linear_project(ff_out, ff_hidden, l2w, l2b, total, dim_ff, d_model);
    free(ff_hidden);

    for (size_t i = 0; i < buf_size; i++) x[i] += ff_out[i];
    free(ff_out);

    float* n3w = (float*)tensor_data_ptr(layer->norm3_weight->tensor);
    float* n3b = (float*)tensor_data_ptr(layer->norm3_bias->tensor);
    layer_norm_inplace(x, total, d_model, n3w, n3b, layer->norm_eps);

    // Output
    int out_shape[] = {batch, seq_len, d_model};
    TensorConfig out_cfg = {.dtype = tgt->dtype, .device = tgt->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_from_data(x, out_shape, 3, &out_cfg);
    free(x);
    return output;
}

static void decoder_layer_free(Module* module) {
    TransformerDecoderLayer* layer = (TransformerDecoderLayer*)module;
    if (!layer) return;
    if (layer->self_attn) module_free((Module*)layer->self_attn);
    if (layer->cross_attn) module_free((Module*)layer->cross_attn);
    free(layer);
}

TransformerDecoderLayer* nn_transformer_decoder_layer(int d_model, int nhead, int dim_feedforward,
                                                       float dropout, DType dtype, DeviceType device) {
    TransformerDecoderLayer* layer = malloc(sizeof(TransformerDecoderLayer));
    if (!layer) return NULL;

    if (module_init((Module*)layer, "TransformerDecoderLayer", decoder_layer_forward_wrapper, decoder_layer_free) != 0) {
        free(layer); return NULL;
    }

    layer->d_model = d_model;
    layer->nhead = nhead;
    layer->dim_feedforward = dim_feedforward;
    layer->dropout = dropout;
    layer->norm_eps = 1e-5f;

    layer->self_attn = nn_multihead_attention(d_model, nhead, dropout, dtype, device);
    if (!layer->self_attn) { module_free((Module*)layer); return NULL; }

    layer->cross_attn = nn_multihead_attention(d_model, nhead, dropout, dtype, device);
    if (!layer->cross_attn) { module_free((Module*)layer); return NULL; }

    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};

    int l1_w_shape[] = {dim_feedforward, d_model};
    int l1_b_shape[] = {dim_feedforward};
    int l2_w_shape[] = {d_model, dim_feedforward};
    int l2_b_shape[] = {d_model};
    int norm_shape[] = {d_model};

    // Linear1
    Tensor* l1w = tensor_empty(l1_w_shape, 2, &config);
    if (!l1w) { module_free((Module*)layer); return NULL; }
    float* l1w_data = (float*)tensor_data_ptr(l1w);
    if (l1w_data) xavier_init(l1w_data, (size_t)dim_feedforward * d_model, d_model, dim_feedforward);
    if (module_add_parameter((Module*)layer, l1w, "linear1_weight", true) != 0) {
        tensor_free(l1w); module_free((Module*)layer); return NULL;
    }
    layer->linear1_weight = module_get_parameter((Module*)layer, "linear1_weight");

    Tensor* l1b = tensor_zeros(l1_b_shape, 1, &config);
    if (!l1b) { module_free((Module*)layer); return NULL; }
    if (module_add_parameter((Module*)layer, l1b, "linear1_bias", true) != 0) {
        tensor_free(l1b); module_free((Module*)layer); return NULL;
    }
    layer->linear1_bias = module_get_parameter((Module*)layer, "linear1_bias");

    // Linear2
    Tensor* l2w = tensor_empty(l2_w_shape, 2, &config);
    if (!l2w) { module_free((Module*)layer); return NULL; }
    float* l2w_data = (float*)tensor_data_ptr(l2w);
    if (l2w_data) xavier_init(l2w_data, (size_t)d_model * dim_feedforward, dim_feedforward, d_model);
    if (module_add_parameter((Module*)layer, l2w, "linear2_weight", true) != 0) {
        tensor_free(l2w); module_free((Module*)layer); return NULL;
    }
    layer->linear2_weight = module_get_parameter((Module*)layer, "linear2_weight");

    Tensor* l2b = tensor_zeros(l2_b_shape, 1, &config);
    if (!l2b) { module_free((Module*)layer); return NULL; }
    if (module_add_parameter((Module*)layer, l2b, "linear2_bias", true) != 0) {
        tensor_free(l2b); module_free((Module*)layer); return NULL;
    }
    layer->linear2_bias = module_get_parameter((Module*)layer, "linear2_bias");

    // 3 layer norms
    struct { const char* wn; const char* bn; Parameter** wp; Parameter** bp; } norms[] = {
        {"norm1_weight", "norm1_bias", &layer->norm1_weight, &layer->norm1_bias},
        {"norm2_weight", "norm2_bias", &layer->norm2_weight, &layer->norm2_bias},
        {"norm3_weight", "norm3_bias", &layer->norm3_weight, &layer->norm3_bias},
    };
    for (int i = 0; i < 3; i++) {
        Tensor* nw = tensor_ones(norm_shape, 1, &config);
        if (!nw) { module_free((Module*)layer); return NULL; }
        if (module_add_parameter((Module*)layer, nw, norms[i].wn, true) != 0) {
            tensor_free(nw); module_free((Module*)layer); return NULL;
        }
        *norms[i].wp = module_get_parameter((Module*)layer, norms[i].wn);

        Tensor* nb = tensor_zeros(norm_shape, 1, &config);
        if (!nb) { module_free((Module*)layer); return NULL; }
        if (module_add_parameter((Module*)layer, nb, norms[i].bn, true) != 0) {
            tensor_free(nb); module_free((Module*)layer); return NULL;
        }
        *norms[i].bp = module_get_parameter((Module*)layer, norms[i].bn);
    }

    return layer;
}

static Tensor* transformer_decoder_forward(Module* module, Tensor* input) {
    // When called via module_forward, only self-attention (no cross-attention)
    TransformerDecoder* dec = (TransformerDecoder*)module;
    if (!dec || !input) return NULL;

    Tensor* x = input;
    bool owns_x = false;

    for (int i = 0; i < dec->num_layers; i++) {
        Tensor* out = transformer_decoder_layer_forward(dec->layers[i], x, NULL, NULL, NULL);
        if (owns_x) tensor_free(x);
        if (!out) return NULL;
        x = out;
        owns_x = true;
    }

    // Final layer norm
    tensor_ensure_executed(x);
    float* x_data = (float*)tensor_data_ptr(x);
    if (!x_data) { if (owns_x) tensor_free(x); return NULL; }

    int total = 1;
    for (int i = 0; i < x->ndim - 1; i++) total *= x->shape[i];
    int d_model = dec->d_model;

    float* norm_w = (float*)tensor_data_ptr(dec->norm_weight->tensor);
    float* norm_b = (float*)tensor_data_ptr(dec->norm_bias->tensor);

    size_t buf_size = (size_t)total * d_model;
    float* buf = malloc(buf_size * sizeof(float));
    if (!buf) { if (owns_x) tensor_free(x); return NULL; }
    memcpy(buf, x_data, buf_size * sizeof(float));

    layer_norm_inplace(buf, total, d_model, norm_w, norm_b, dec->norm_eps);

    int* out_shape = malloc(x->ndim * sizeof(int));
    if (!out_shape) { free(buf); if (owns_x) tensor_free(x); return NULL; }
    memcpy(out_shape, x->shape, x->ndim * sizeof(int));
    int out_ndim = x->ndim;

    TensorConfig cfg = {.dtype = x->dtype, .device = x->device, .has_dtype = true, .has_device = true};
    if (owns_x) tensor_free(x);

    Tensor* output = tensor_from_data(buf, out_shape, out_ndim, &cfg);
    free(buf);
    free(out_shape);
    return output;
}

static void transformer_decoder_free(Module* module) {
    TransformerDecoder* dec = (TransformerDecoder*)module;
    if (!dec) return;
    if (dec->layers) {
        for (int i = 0; i < dec->num_layers; i++) {
            if (dec->layers[i]) module_free((Module*)dec->layers[i]);
        }
        free(dec->layers);
    }
    free(dec);
}

TransformerDecoder* nn_transformer_decoder(int d_model, int nhead, int dim_feedforward,
                                            float dropout, int num_layers,
                                            DType dtype, DeviceType device) {
    TransformerDecoder* dec = malloc(sizeof(TransformerDecoder));
    if (!dec) return NULL;

    if (module_init((Module*)dec, "TransformerDecoder", transformer_decoder_forward, transformer_decoder_free) != 0) {
        free(dec); return NULL;
    }

    dec->d_model = d_model;
    dec->num_layers = num_layers;
    dec->norm_eps = 1e-5f;

    dec->layers = malloc(num_layers * sizeof(TransformerDecoderLayer*));
    if (!dec->layers) { module_free((Module*)dec); return NULL; }

    for (int i = 0; i < num_layers; i++) {
        dec->layers[i] = nn_transformer_decoder_layer(d_model, nhead, dim_feedforward, dropout, dtype, device);
        if (!dec->layers[i]) {
            dec->num_layers = i;
            module_free((Module*)dec);
            return NULL;
        }
    }

    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    int norm_shape[] = {d_model};

    Tensor* nw = tensor_ones(norm_shape, 1, &config);
    if (!nw) { module_free((Module*)dec); return NULL; }
    if (module_add_parameter((Module*)dec, nw, "norm_weight", true) != 0) {
        tensor_free(nw); module_free((Module*)dec); return NULL;
    }
    dec->norm_weight = module_get_parameter((Module*)dec, "norm_weight");

    Tensor* nb = tensor_zeros(norm_shape, 1, &config);
    if (!nb) { module_free((Module*)dec); return NULL; }
    if (module_add_parameter((Module*)dec, nb, "norm_bias", true) != 0) {
        tensor_free(nb); module_free((Module*)dec); return NULL;
    }
    dec->norm_bias = module_get_parameter((Module*)dec, "norm_bias");

    return dec;
}
