#include "distributed/ring_attention.h"
#include "distributed/distributed.h"
#include "distributed/ring_allreduce.h"
#include "nn/llm_ops.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include "alloc/cml_allocator.h"
#include <math.h>
#include <string.h>

/**
 * Fold one KV shard (all batch rows and heads) into the running online-softmax state of the
 * local Q, reusing the shared FlashAttention tile kernel. @p kv_pos0 is the shard's global
 * first-token position for causal/window masking.
 * @return (void)
 */
static void ring_fold_shard(float* row_max, float* row_sum, float* acc, float* scores,
                            const float* q, int seq_q, int q_pos0, const float* k, const float* v,
                            int kv_len, int kv_pos0, int batch, int num_heads, int num_kv_heads,
                            int head_dim, float scale, bool causal, int window) {
    int groups    = num_heads / num_kv_heads;
    int q_stride  = num_heads * head_dim;
    int kv_stride = num_kv_heads * head_dim;
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < num_heads; h++) {
            int kv_h            = h / groups;
            const float* q_head = q + ((size_t)b * seq_q * num_heads + h) * head_dim;
            const float* k_head = k + ((size_t)b * kv_len * num_kv_heads + kv_h) * head_dim;
            const float* v_head = v + ((size_t)b * kv_len * num_kv_heads + kv_h) * head_dim;
            size_t st           = ((size_t)b * num_heads + h) * seq_q;
            cml_flash_attention_fold_tile(row_max + st, row_sum + st, acc + st * head_dim, scores,
                                          q_head, q_stride, seq_q, q_pos0, k_head, v_head,
                                          kv_stride, kv_len, kv_pos0, head_dim, scale, causal,
                                          window);
        }
    }
}

/**
 * Core ring attention: fold the local KV shard, then rotate KV around the ring folding each
 * received shard, and normalize. @p causal and @p q_pos0 differ between prefill and decode.
 * @return (Tensor*)
 */
static Tensor* ring_attention_core(Tensor* Q, Tensor* K, Tensor* V, const CMLGQAConfig* config,
                                   bool causal) {
    if (!Q || !K || !V || !config) {
        LOG_ERROR("cml_ring_attention: NULL argument");
        return NULL;
    }

    int num_heads    = config->num_heads;
    int num_kv_heads = config->num_kv_heads;
    int head_dim     = config->head_dim;
    if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || num_heads % num_kv_heads != 0) {
        LOG_ERROR("cml_ring_attention: invalid config");
        return NULL;
    }

    int world = cml_dist_get_world_size();
    int rank  = cml_dist_get_rank();

    tensor_ensure_executed(Q);
    tensor_ensure_executed(K);
    tensor_ensure_executed(V);
    if (Q->ndim != 3 || K->ndim != 3 || V->ndim != 3) {
        LOG_ERROR("cml_ring_attention: expected 3D tensors");
        return NULL;
    }

    /* Single rank: the full sequence is local, so defer to the local flash kernel. */
    if (world <= 1) {
        CMLGQAConfig cfg              = *config;
        cfg.causal                    = causal;
        CMLFlashAttentionConfig flash = cml_flash_attention_default_config();
        return cml_gqa_flash_forward(Q, K, V, &cfg, &flash);
    }

    DistProcessGroup* g = cml_dist_get_default_group();
    if (!g || !g->ops) {
        LOG_ERROR("cml_ring_attention: process group not initialized");
        return NULL;
    }

    int batch     = Q->shape[0];
    int seq_q     = Q->shape[1];
    int shard_len = K->shape[1]; /* equal across ranks (send/recv counts must match) */
    float scale   = config->scale;
    if (scale <= 0.0f)
        scale = 1.0f / sqrtf((float)head_dim);
    int window = config->window_size;
    int q_pos0 = rank * shard_len; /* global position of this rank's first query row */

    float* q_data = (float*)tensor_data_ptr(Q);
    float* k_data = (float*)tensor_data_ptr(K);
    float* v_data = (float*)tensor_data_ptr(V);
    if (!q_data || !k_data || !v_data)
        return NULL;

    size_t kv_elems = (size_t)batch * shard_len * num_kv_heads * head_dim;
    size_t nq       = (size_t)batch * num_heads * seq_q;

    float* cur_k   = (float*)cml_malloc(kv_elems * sizeof(float));
    float* cur_v   = (float*)cml_malloc(kv_elems * sizeof(float));
    float* nxt_k   = (float*)cml_malloc(kv_elems * sizeof(float));
    float* nxt_v   = (float*)cml_malloc(kv_elems * sizeof(float));
    float* row_max = (float*)cml_malloc(nq * sizeof(float));
    float* row_sum = (float*)cml_malloc(nq * sizeof(float));
    float* acc     = (float*)cml_calloc(nq * head_dim, sizeof(float));
    float* scores  = (float*)cml_malloc((size_t)seq_q * shard_len * sizeof(float));
    if (!cur_k || !cur_v || !nxt_k || !nxt_v || !row_max || !row_sum || !acc || !scores) {
        cml_free(cur_k);
        cml_free(cur_v);
        cml_free(nxt_k);
        cml_free(nxt_v);
        cml_free(row_max);
        cml_free(row_sum);
        cml_free(acc);
        cml_free(scores);
        return NULL;
    }

    memcpy(cur_k, k_data, kv_elems * sizeof(float));
    memcpy(cur_v, v_data, kv_elems * sizeof(float));
    for (size_t i = 0; i < nq; i++) {
        row_max[i] = -1e30f;
        row_sum[i] = 0.0f;
    }

    /* Fold the local shard first so the running max is real before any future (masked) shard. */
    ring_fold_shard(row_max, row_sum, acc, scores, q_data, seq_q, q_pos0, cur_k, cur_v, shard_len,
                    rank * shard_len, batch, num_heads, num_kv_heads, head_dim, scale, causal,
                    window);

    int left          = (rank - 1 + world) % world;
    int right         = (rank + 1) % world;
    int shape[3]      = {batch, shard_len, num_kv_heads * head_dim};
    TensorConfig vcfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

    int rc = 0;
    for (int step = 1; step < world && rc == 0; step++) {
        Tensor* sk = tensor_from_blob(cur_k, shape, 3, &vcfg);
        Tensor* rk = tensor_from_blob(nxt_k, shape, 3, &vcfg);
        Tensor* sv = tensor_from_blob(cur_v, shape, 3, &vcfg);
        Tensor* rv = tensor_from_blob(nxt_v, shape, 3, &vcfg);
        if (!sk || !rk || !sv || !rv ||
            cml_ring_sendrecv(g->ops, sk, right, rk, left, step, g->backend_ctx) != 0 ||
            cml_ring_sendrecv(g->ops, sv, right, rv, left, world + step, g->backend_ctx) != 0)
            rc = -1;
        tensor_free(sk);
        tensor_free(rk);
        tensor_free(sv);
        tensor_free(rv);
        if (rc != 0)
            break;

        /* The shard received at this step originated at rank (rank - step). */
        int origin = (rank - step + world) % world;
        ring_fold_shard(row_max, row_sum, acc, scores, q_data, seq_q, q_pos0, nxt_k, nxt_v,
                        shard_len, origin * shard_len, batch, num_heads, num_kv_heads, head_dim,
                        scale, causal, window);

        /* Forward what we just received on the next step. */
        float* tk = cur_k;
        cur_k     = nxt_k;
        nxt_k     = tk;
        float* tv = cur_v;
        cur_v     = nxt_v;
        nxt_v     = tv;
    }

    Tensor* result = NULL;
    if (rc == 0) {
        size_t out_size = (size_t)batch * seq_q * num_heads * head_dim;
        float* output   = (float*)cml_calloc(out_size, sizeof(float));
        if (output) {
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < num_heads; h++) {
                    size_t st = ((size_t)b * num_heads + h) * seq_q;
                    for (int qi = 0; qi < seq_q; qi++) {
                        float sum   = row_sum[st + qi];
                        float inv   = (sum > 0.0f) ? 1.0f / sum : 0.0f;
                        float* arow = acc + (st + qi) * head_dim;
                        float* orow = output + ((size_t)b * seq_q + qi) * num_heads * head_dim +
                                      (size_t)h * head_dim;
                        for (int d = 0; d < head_dim; d++)
                            orow[d] = arow[d] * inv;
                    }
                }
            }
            int out_shape[]      = {batch, seq_q, num_heads * head_dim};
            TensorConfig out_cfg = {.dtype      = DTYPE_FLOAT32,
                                    .device     = DEVICE_CPU,
                                    .has_dtype  = true,
                                    .has_device = true};
            result               = tensor_from_data(output, out_shape, 3, &out_cfg);
            cml_free(output);
        }
    }

    cml_free(cur_k);
    cml_free(cur_v);
    cml_free(nxt_k);
    cml_free(nxt_v);
    cml_free(row_max);
    cml_free(row_sum);
    cml_free(acc);
    cml_free(scores);
    return result;
}

Tensor* cml_ring_attention_forward(Tensor* Q, Tensor* K, Tensor* V, const CMLGQAConfig* config) {
    return ring_attention_core(Q, K, V, config, config ? config->causal : false);
}

Tensor* cml_ring_attention_decode(Tensor* Q, Tensor* K, Tensor* V, const CMLGQAConfig* config) {
    return ring_attention_core(Q, K, V, config, false);
}
