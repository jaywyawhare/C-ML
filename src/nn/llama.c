#include "nn/llama.h"
#include "nn/llm_ops.h"
#include "core/gguf.h"
#include "ops/uops.h"
#include "core/logging.h"
#include "tensor/tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

CMLLLaMAConfig cml_llama_config_7b(void) {
    CMLLLaMAConfig config = {
        .vocab_size        = 32000,
        .hidden_size       = 4096,
        .intermediate_size = 11008,
        .num_layers        = 32,
        .num_heads         = 32,
        .num_kv_heads      = 32,
        .max_seq_len       = 2048,
        .rope_theta        = 10000.0f,
        .rms_norm_eps      = 1e-5f
    };
    return config;
}

CMLLLaMAConfig cml_llama_config_13b(void) {
    CMLLLaMAConfig config = {
        .vocab_size        = 32000,
        .hidden_size       = 5120,
        .intermediate_size = 13824,
        .num_layers        = 40,
        .num_heads         = 40,
        .num_kv_heads      = 40,
        .max_seq_len       = 2048,
        .rope_theta        = 10000.0f,
        .rms_norm_eps      = 1e-5f
    };
    return config;
}

CMLLLaMAConfig cml_llama_config_70b(void) {
    CMLLLaMAConfig config = {
        .vocab_size        = 32000,
        .hidden_size       = 8192,
        .intermediate_size = 28672,
        .num_layers        = 80,
        .num_heads         = 64,
        .num_kv_heads      = 8,
        .max_seq_len       = 4096,
        .rope_theta        = 10000.0f,
        .rms_norm_eps      = 1e-5f
    };
    return config;
}

CMLGenerationConfig cml_generation_default_config(void) {
    CMLGenerationConfig config = {
        .temperature   = 0.8f,
        .top_p         = 0.9f,
        .top_k         = 40,
        .max_new_tokens = 256,
        .eos_token_id  = 2,
        .do_sample     = true
    };
    return config;
}

static Tensor* rms_norm(Tensor* x, Tensor* weight, float eps) {
    if (!x || !weight) return NULL;

    /* x_sq = x * x */
    Tensor* x_sq = uop_mul(x, x);
    if (!x_sq) return NULL;

    /* mean_sq = mean(x_sq, dim=-1, keepdim=true) */
    int last_dim = x->ndim - 1;
    ReduceParams rp = {
        .dims     = &last_dim,
        .num_dims = 1,
        .keepdim  = true
    };
    Tensor* mean_sq = uop_mean(x_sq, &rp);
    if (mean_sq)
        tensor_ensure_executed(mean_sq);
    tensor_free(x_sq);
    if (!mean_sq) return NULL;

    /* Create eps tensor for addition */
    Tensor* eps_t = tensor_full(mean_sq->shape, mean_sq->ndim, NULL, eps);
    if (!eps_t) { tensor_free(mean_sq); return NULL; }

    /* mean_sq_eps = mean_sq + eps */
    Tensor* mean_sq_eps = uop_add(mean_sq, eps_t);
    if (mean_sq_eps)
        tensor_ensure_executed(mean_sq_eps);
    tensor_free(mean_sq);
    tensor_free(eps_t);
    if (!mean_sq_eps) return NULL;

    /* rsqrt_val = rsqrt(mean_sq_eps) */
    Tensor* rsqrt_val = uop_rsqrt(mean_sq_eps);
    if (rsqrt_val)
        tensor_ensure_executed(rsqrt_val);
    tensor_free(mean_sq_eps);
    if (!rsqrt_val) return NULL;

    /* normed = x * rsqrt_val */
    Tensor* normed = uop_mul(x, rsqrt_val);
    if (normed)
        tensor_ensure_executed(normed);
    tensor_free(rsqrt_val);
    if (!normed) return NULL;

    /* result = normed * weight (broadcast weight across seq dim) */
    Tensor* result = uop_mul(normed, weight);
    if (result)
        tensor_ensure_executed(result);
    tensor_free(normed);

    return result;
}

static Tensor* swiglu_ffn(Tensor* x, Tensor* gate_proj, Tensor* up_proj, Tensor* down_proj) {
    if (!x || !gate_proj || !up_proj || !down_proj) return NULL;

    /* gate_out = x @ gate_proj -> [seq_len, intermediate_size] */
    Tensor* gate_out = uop_matmul(x, gate_proj);
    if (!gate_out) return NULL;

    /* gate_activated = silu(gate_out) */
    Tensor* gate_activated = uop_silu(gate_out);
    if (gate_activated)
        tensor_ensure_executed(gate_activated);
    tensor_free(gate_out);
    if (!gate_activated) return NULL;

    /* up_out = x @ up_proj -> [seq_len, intermediate_size] */
    Tensor* up_out = uop_matmul(x, up_proj);
    if (!up_out) { tensor_free(gate_activated); return NULL; }

    /* combined = gate_activated * up_out */
    Tensor* combined = uop_mul(gate_activated, up_out);
    if (combined)
        tensor_ensure_executed(combined);
    tensor_free(gate_activated);
    tensor_free(up_out);
    if (!combined) return NULL;

    /* output = combined @ down_proj -> [seq_len, hidden_size] */
    Tensor* output = uop_matmul(combined, down_proj);
    if (output)
        tensor_ensure_executed(output);
    tensor_free(combined);

    return output;
}

static CMLLLaMALayer* llama_layer_create(const CMLLLaMAConfig* config) {
    CMLLLaMALayer* layer = (CMLLLaMALayer*)calloc(1, sizeof(CMLLLaMALayer));
    if (!layer) return NULL;

    int head_dim = config->hidden_size / config->num_heads;

    /* Create KV cache for this layer */
    layer->kv_cache = cml_kv_cache_create(config->max_seq_len,
                                           config->num_kv_heads, head_dim);
    if (!layer->kv_cache) {
        free(layer);
        return NULL;
    }

    /* Weight tensors are set to NULL here; they get populated during weight loading
     * or can be initialized with random values for testing */

    return layer;
}

static void llama_layer_free(CMLLLaMALayer* layer) {
    if (!layer) return;

    if (layer->q_proj) tensor_free(layer->q_proj);
    if (layer->k_proj) tensor_free(layer->k_proj);
    if (layer->v_proj) tensor_free(layer->v_proj);
    if (layer->o_proj) tensor_free(layer->o_proj);
    if (layer->gate_proj) tensor_free(layer->gate_proj);
    if (layer->up_proj) tensor_free(layer->up_proj);
    if (layer->down_proj) tensor_free(layer->down_proj);
    if (layer->input_layernorm) tensor_free(layer->input_layernorm);
    if (layer->post_attn_layernorm) tensor_free(layer->post_attn_layernorm);
    if (layer->kv_cache) cml_kv_cache_free(layer->kv_cache);

    free(layer);
}

CMLLLaMAModel* cml_llama_create(const CMLLLaMAConfig* config) {
    if (!config) {
        LOG_ERROR("cml_llama_create: NULL config");
        return NULL;
    }

    CMLLLaMAModel* model = (CMLLLaMAModel*)calloc(1, sizeof(CMLLLaMAModel));
    if (!model) {
        LOG_ERROR("cml_llama_create: allocation failed");
        return NULL;
    }

    model->config = *config;
    model->num_layers = config->num_layers;
    model->weights_loaded = false;
    model->current_seq_len = 0;

    /* Allocate layer array */
    model->layers = (CMLLLaMALayer**)calloc((size_t)config->num_layers,
                                             sizeof(CMLLLaMALayer*));
    if (!model->layers) {
        LOG_ERROR("cml_llama_create: layer array allocation failed");
        free(model);
        return NULL;
    }

    /* Create each layer */
    for (int i = 0; i < config->num_layers; i++) {
        model->layers[i] = llama_layer_create(config);
        if (!model->layers[i]) {
            LOG_ERROR("cml_llama_create: layer %d creation failed", i);
            /* Free previously created layers */
            for (int j = 0; j < i; j++) {
                llama_layer_free(model->layers[j]);
            }
            free(model->layers);
            free(model);
            return NULL;
        }
    }

    LOG_INFO("LLaMA model created: %d layers, hidden=%d, heads=%d, kv_heads=%d",
             config->num_layers, config->hidden_size,
             config->num_heads, config->num_kv_heads);

    return model;
}

void cml_llama_free(CMLLLaMAModel* model) {
    if (!model) return;

    /* Free layers */
    if (model->layers) {
        for (int i = 0; i < model->num_layers; i++) {
            llama_layer_free(model->layers[i]);
        }
        free(model->layers);
    }

    /* Free embeddings and output weights */
    if (model->embed_tokens) tensor_free(model->embed_tokens);
    if (model->norm) tensor_free(model->norm);
    if (model->lm_head) tensor_free(model->lm_head);

    /* Free tokenizer */
    if (model->tokenizer) cml_tokenizer_free(model->tokenizer);

    free(model);
}

static int load_tensor_by_name(CMLLLaMAModel* model, GGUFContext* ctx, const char* name) {
    Tensor* t = gguf_read_tensor(ctx, name);
    if (!t) {
        LOG_WARNING("GGUF: tensor '%s' not found, skipping", name);
        return -1;
    }

    /* Match model.embed_tokens.weight */
    if (strcmp(name, "model.embed_tokens.weight") == 0 ||
        strcmp(name, "token_embd.weight") == 0) {
        if (model->embed_tokens) tensor_free(model->embed_tokens);
        model->embed_tokens = t;
        return 0;
    }

    /* Match model.norm.weight */
    if (strcmp(name, "model.norm.weight") == 0 ||
        strcmp(name, "output_norm.weight") == 0) {
        if (model->norm) tensor_free(model->norm);
        model->norm = t;
        return 0;
    }

    /* Match lm_head.weight */
    if (strcmp(name, "lm_head.weight") == 0 ||
        strcmp(name, "output.weight") == 0) {
        if (model->lm_head) tensor_free(model->lm_head);
        model->lm_head = t;
        return 0;
    }

    /* Try to parse layer index: model.layers.N.xxx or blk.N.xxx */
    int layer_idx = -1;
    const char* suffix = NULL;

    /* Try "model.layers.N." format */
    if (strncmp(name, "model.layers.", 13) == 0) {
        const char* p = name + 13;
        layer_idx = (int)strtol(p, (char**)&suffix, 10);
        if (suffix && *suffix == '.') {
            suffix++; /* skip the dot */
        } else {
            layer_idx = -1;
        }
    }
    /* Try "blk.N." format (llama.cpp style) */
    else if (strncmp(name, "blk.", 4) == 0) {
        const char* p = name + 4;
        layer_idx = (int)strtol(p, (char**)&suffix, 10);
        if (suffix && *suffix == '.') {
            suffix++;
        } else {
            layer_idx = -1;
        }
    }

    if (layer_idx < 0 || layer_idx >= model->num_layers || !suffix) {
        LOG_WARNING("GGUF: unrecognized tensor name '%s'", name);
        tensor_free(t);
        return -1;
    }

    CMLLLaMALayer* layer = model->layers[layer_idx];

    /* Self-attention projections */
    if (strcmp(suffix, "self_attn.q_proj.weight") == 0 ||
        strcmp(suffix, "attn_q.weight") == 0) {
        if (layer->q_proj) tensor_free(layer->q_proj);
        layer->q_proj = t;
    } else if (strcmp(suffix, "self_attn.k_proj.weight") == 0 ||
               strcmp(suffix, "attn_k.weight") == 0) {
        if (layer->k_proj) tensor_free(layer->k_proj);
        layer->k_proj = t;
    } else if (strcmp(suffix, "self_attn.v_proj.weight") == 0 ||
               strcmp(suffix, "attn_v.weight") == 0) {
        if (layer->v_proj) tensor_free(layer->v_proj);
        layer->v_proj = t;
    } else if (strcmp(suffix, "self_attn.o_proj.weight") == 0 ||
               strcmp(suffix, "attn_output.weight") == 0) {
        if (layer->o_proj) tensor_free(layer->o_proj);
        layer->o_proj = t;
    }
    /* FFN projections */
    else if (strcmp(suffix, "mlp.gate_proj.weight") == 0 ||
             strcmp(suffix, "ffn_gate.weight") == 0) {
        if (layer->gate_proj) tensor_free(layer->gate_proj);
        layer->gate_proj = t;
    } else if (strcmp(suffix, "mlp.up_proj.weight") == 0 ||
               strcmp(suffix, "ffn_up.weight") == 0) {
        if (layer->up_proj) tensor_free(layer->up_proj);
        layer->up_proj = t;
    } else if (strcmp(suffix, "mlp.down_proj.weight") == 0 ||
               strcmp(suffix, "ffn_down.weight") == 0) {
        if (layer->down_proj) tensor_free(layer->down_proj);
        layer->down_proj = t;
    }
    /* Layer norms */
    else if (strcmp(suffix, "input_layernorm.weight") == 0 ||
             strcmp(suffix, "attn_norm.weight") == 0) {
        if (layer->input_layernorm) tensor_free(layer->input_layernorm);
        layer->input_layernorm = t;
    } else if (strcmp(suffix, "post_attention_layernorm.weight") == 0 ||
               strcmp(suffix, "ffn_norm.weight") == 0) {
        if (layer->post_attn_layernorm) tensor_free(layer->post_attn_layernorm);
        layer->post_attn_layernorm = t;
    } else {
        LOG_WARNING("GGUF: unrecognized layer suffix '%s' for layer %d", suffix, layer_idx);
        tensor_free(t);
        return -1;
    }

    return 0;
}

int cml_llama_load_gguf(CMLLLaMAModel* model, const char* filepath) {
    if (!model || !filepath) {
        LOG_ERROR("cml_llama_load_gguf: NULL argument");
        return -1;
    }

    GGUFContext* ctx = gguf_open_read(filepath);
    if (!ctx) {
        LOG_ERROR("cml_llama_load_gguf: failed to open '%s'", filepath);
        return -1;
    }

    int num_tensors = gguf_get_num_tensors(ctx);
    LOG_INFO("Loading GGUF: %s (%d tensors)", filepath, num_tensors);

    int loaded = 0;
    int skipped = 0;

    for (int i = 0; i < num_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx, i);
        if (!name) continue;

        if (load_tensor_by_name(model, ctx, name) == 0) {
            loaded++;
        } else {
            skipped++;
        }
    }

    gguf_close(ctx);

    /* If lm_head is not provided, share embed_tokens (weight tying) */
    if (!model->lm_head && model->embed_tokens) {
        LOG_INFO("Weight tying: lm_head shares embed_tokens");
        model->lm_head = model->embed_tokens;
    }

    model->weights_loaded = true;
    LOG_INFO("GGUF loading complete: %d loaded, %d skipped", loaded, skipped);

    return 0;
}

static Tensor* embed_tokens_lookup(Tensor* embed, const int* token_ids, int seq_len) {
    if (!embed || !token_ids || seq_len <= 0) return NULL;

    int hidden_size = embed->shape[1];
    int shape[] = {seq_len, hidden_size};
    Tensor* output = tensor_zeros(shape, 2, NULL);
    if (!output) return NULL;

    tensor_ensure_executed(embed);
    tensor_ensure_executed(output);

    float* embed_data = (float*)tensor_data_ptr(embed);
    float* out_data = (float*)tensor_data_ptr(output);
    if (!embed_data || !out_data) {
        tensor_free(output);
        return NULL;
    }

    int vocab_size = embed->shape[0];
    for (int i = 0; i < seq_len; i++) {
        int tid = token_ids[i];
        if (tid < 0 || tid >= vocab_size) {
            LOG_WARNING("Token ID %d out of range [0, %d)", tid, vocab_size);
            tid = 0;
        }
        memcpy(out_data + i * hidden_size,
               embed_data + tid * hidden_size,
               (size_t)hidden_size * sizeof(float));
    }

    return output;
}

Tensor* cml_llama_layer_forward(CMLLLaMAModel* model, CMLLLaMALayer* layer,
                                 Tensor* hidden, int start_pos) {
    if (!model || !layer || !hidden) return NULL;

    const CMLLLaMAConfig* cfg = &model->config;
    int head_dim = cfg->hidden_size / cfg->num_heads;

    Tensor* normed = rms_norm(hidden, layer->input_layernorm, cfg->rms_norm_eps);
    if (!normed) return NULL;

    Tensor* Q = uop_matmul(normed, layer->q_proj);
    Tensor* K = uop_matmul(normed, layer->k_proj);
    Tensor* V = uop_matmul(normed, layer->v_proj);
    if (Q)
        tensor_ensure_executed(Q);
    if (K)
        tensor_ensure_executed(K);
    if (V)
        tensor_ensure_executed(V);
    tensor_free(normed);

    if (!Q || !K || !V) {
        if (Q) tensor_free(Q);
        if (K) tensor_free(K);
        if (V) tensor_free(V);
        return NULL;
    }

    CMLRoPEConfig rope_cfg = {
        .dim         = head_dim,
        .max_seq_len = cfg->max_seq_len,
        .base        = cfg->rope_theta
    };

    Tensor* Q_rope = cml_rope_forward(Q, start_pos, &rope_cfg);
    Tensor* K_rope = cml_rope_forward(K, start_pos, &rope_cfg);
    /* cml_rope_forward modifies in-place and returns the same pointer,
     * so only free Q/K if rope returned a different (new) tensor */
    if (Q_rope != Q) tensor_free(Q);
    if (K_rope != K) tensor_free(K);

    if (!Q_rope || !K_rope) {
        if (Q_rope) tensor_free(Q_rope);
        if (K_rope) tensor_free(K_rope);
        tensor_free(V);
        return NULL;
    }

    CMLGQAConfig gqa_cfg = {
        .num_heads    = cfg->num_heads,
        .num_kv_heads = cfg->num_kv_heads,
        .head_dim     = head_dim,
        .scale        = 0.0f, /* auto: 1/sqrt(head_dim) */
        .causal       = true
    };

    /* GQA expects 3D [batch, seq, dim] -- add batch=1 dimension */
    int seq_len = Q_rope->shape[0];
    int q_3d[] = {1, seq_len, Q_rope->shape[Q_rope->ndim - 1]};
    int k_3d[] = {1, seq_len, K_rope->shape[K_rope->ndim - 1]};
    int v_3d[] = {1, seq_len, V->shape[V->ndim - 1]};

    Tensor* Q3 = tensor_reshape(Q_rope, q_3d, 3);
    Tensor* K3 = tensor_reshape(K_rope, k_3d, 3);
    Tensor* V3 = tensor_reshape(V, v_3d, 3);

    Tensor* attn_out = cml_gqa_forward_cached(Q3, K3, V3,
                                               layer->kv_cache, &gqa_cfg);
    tensor_free(Q3);
    tensor_free(K3);
    tensor_free(V3);
    tensor_free(Q_rope);
    tensor_free(K_rope);
    tensor_free(V);

    if (!attn_out) return NULL;

    /* Squeeze batch dimension: [1, seq, dim] -> [seq, dim] */
    int out_2d[] = {attn_out->shape[1], attn_out->shape[2]};
    Tensor* attn_2d = tensor_reshape(attn_out, out_2d, 2);

    Tensor* attn_proj = uop_matmul(attn_2d, layer->o_proj);
    if (attn_proj)
        tensor_ensure_executed(attn_proj);
    tensor_free(attn_2d);
    tensor_free(attn_out);
    if (!attn_proj) return NULL;

    Tensor* residual1 = uop_add(hidden, attn_proj);
    if (residual1)
        tensor_ensure_executed(residual1);
    tensor_free(attn_proj);
    if (!residual1) return NULL;

    Tensor* normed2 = rms_norm(residual1, layer->post_attn_layernorm, cfg->rms_norm_eps);
    if (!normed2) { tensor_free(residual1); return NULL; }

    Tensor* ffn_out = swiglu_ffn(normed2, layer->gate_proj, layer->up_proj, layer->down_proj);
    if (ffn_out)
        tensor_ensure_executed(ffn_out);
    tensor_free(normed2);
    if (!ffn_out) { tensor_free(residual1); return NULL; }

    Tensor* output = uop_add(residual1, ffn_out);
    if (output)
        tensor_ensure_executed(output);
    tensor_free(residual1);
    tensor_free(ffn_out);

    return output;
}

Tensor* cml_llama_forward(CMLLLaMAModel* model, const int* token_ids, int seq_len) {
    if (!model || !token_ids || seq_len <= 0) {
        LOG_ERROR("cml_llama_forward: invalid arguments");
        return NULL;
    }

    if (!model->embed_tokens) {
        LOG_ERROR("cml_llama_forward: embed_tokens not loaded");
        return NULL;
    }

    int start_pos = model->current_seq_len;

    Tensor* hidden = embed_tokens_lookup(model->embed_tokens, token_ids, seq_len);
    if (!hidden) {
        LOG_ERROR("cml_llama_forward: embedding lookup failed");
        return NULL;
    }

    for (int i = 0; i < model->num_layers; i++) {
        Tensor* next_hidden = cml_llama_layer_forward(model, model->layers[i],
                                                       hidden, start_pos);
        tensor_free(hidden);
        if (!next_hidden) {
            LOG_ERROR("cml_llama_forward: layer %d failed", i);
            return NULL;
        }
        hidden = next_hidden;
    }

    if (model->norm) {
        Tensor* normed = rms_norm(hidden, model->norm, model->config.rms_norm_eps);
        tensor_free(hidden);
        if (!normed) {
            LOG_ERROR("cml_llama_forward: final norm failed");
            return NULL;
        }
        hidden = normed;
    }

    Tensor* logits = NULL;
    if (model->lm_head && model->lm_head != model->embed_tokens) {
        logits = uop_matmul(hidden, model->lm_head);
    } else if (model->embed_tokens) {
        /* Weight tying: use embed_tokens transposed.
         * For simplicity, use matmul with embed_tokens directly;
         * the embedding shape is [vocab_size, hidden_size], and hidden is
         * [seq_len, hidden_size]. We need hidden @ embed^T = [seq_len, vocab_size].
         * We compute this via the matmul of hidden [seq, hidden] x embed^T [hidden, vocab].
         * We'll explicitly use permute to transpose embed. */
        PermuteParams pp = { .perm = (int[]){1, 0}, .num_dims = 2 };
        Tensor* embed_t = uop_permute(model->embed_tokens, &pp);
        if (embed_t) {
            logits = uop_matmul(hidden, embed_t);
            tensor_free(embed_t);
        }
    }
    tensor_free(hidden);

    if (!logits) {
        LOG_ERROR("cml_llama_forward: lm_head projection failed");
        return NULL;
    }

    model->current_seq_len = start_pos + seq_len;
    return logits;
}

typedef struct {
    float value;
    int index;
} LogitEntry;

static int logit_entry_cmp_desc(const void* a, const void* b) {
    float va = ((const LogitEntry*)a)->value;
    float vb = ((const LogitEntry*)b)->value;
    if (va > vb) return -1;
    if (va < vb) return 1;
    return 0;
}

int cml_llama_sample_token(Tensor* logits, const CMLGenerationConfig* config) {
    if (!logits || !config) return -1;

    tensor_ensure_executed(logits);
    float* data = (float*)tensor_data_ptr(logits);
    if (!data) return -1;

    /* Get the last token's logits if logits is 2D [seq_len, vocab_size] */
    int vocab_size;
    float* token_logits;
    if (logits->ndim == 2) {
        vocab_size = logits->shape[1];
        int last_pos = logits->shape[0] - 1;
        token_logits = data + last_pos * vocab_size;
    } else if (logits->ndim == 1) {
        vocab_size = logits->shape[0];
        token_logits = data;
    } else {
        LOG_ERROR("cml_llama_sample_token: unexpected logits ndim=%d", logits->ndim);
        return -1;
    }

    /* Greedy decoding */
    if (!config->do_sample || config->temperature <= 0.0f) {
        int best_id = 0;
        float best_val = token_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (token_logits[i] > best_val) {
                best_val = token_logits[i];
                best_id = i;
            }
        }
        return best_id;
    }

    /* Apply temperature */
    float* scaled = (float*)malloc((size_t)vocab_size * sizeof(float));
    if (!scaled) return -1;

    float inv_temp = 1.0f / config->temperature;
    for (int i = 0; i < vocab_size; i++) {
        scaled[i] = token_logits[i] * inv_temp;
    }

    /* Build sorted entries for top-k / top-p filtering */
    LogitEntry* entries = (LogitEntry*)malloc((size_t)vocab_size * sizeof(LogitEntry));
    if (!entries) { free(scaled); return -1; }

    for (int i = 0; i < vocab_size; i++) {
        entries[i].value = scaled[i];
        entries[i].index = i;
    }
    qsort(entries, (size_t)vocab_size, sizeof(LogitEntry), logit_entry_cmp_desc);

    /* Top-k filtering: keep only top_k entries */
    int k = config->top_k;
    if (k <= 0 || k > vocab_size) k = vocab_size;

    /* Softmax over top-k entries */
    float max_val = entries[0].value;
    float sum_exp = 0.0f;
    for (int i = 0; i < k; i++) {
        entries[i].value = expf(entries[i].value - max_val);
        sum_exp += entries[i].value;
    }
    for (int i = 0; i < k; i++) {
        entries[i].value /= sum_exp;
    }

    /* Top-p (nucleus) filtering */
    float cumulative = 0.0f;
    int cutoff = k;
    if (config->top_p > 0.0f && config->top_p < 1.0f) {
        for (int i = 0; i < k; i++) {
            cumulative += entries[i].value;
            if (cumulative >= config->top_p) {
                cutoff = i + 1;
                break;
            }
        }
    }

    /* Re-normalize after top-p cutoff */
    if (cutoff < k) {
        float re_sum = 0.0f;
        for (int i = 0; i < cutoff; i++) {
            re_sum += entries[i].value;
        }
        if (re_sum > 0.0f) {
            for (int i = 0; i < cutoff; i++) {
                entries[i].value /= re_sum;
            }
        }
    }

    /* Sample from the distribution */
    float r = (float)rand() / (float)RAND_MAX;
    float acc = 0.0f;
    int sampled_id = entries[0].index;
    for (int i = 0; i < cutoff; i++) {
        acc += entries[i].value;
        if (r <= acc) {
            sampled_id = entries[i].index;
            break;
        }
    }

    free(entries);
    free(scaled);
    return sampled_id;
}

CMLGenerationResult* cml_llama_generate(CMLLLaMAModel* model, const char* prompt,
                                          const CMLGenerationConfig* config) {
    if (!model || !prompt || !config) {
        LOG_ERROR("cml_llama_generate: NULL argument");
        return NULL;
    }

    if (!model->weights_loaded) {
        LOG_ERROR("cml_llama_generate: weights not loaded");
        return NULL;
    }

    int num_prompt_tokens = 0;
    int* prompt_tokens = NULL;

    if (model->tokenizer) {
        prompt_tokens = cml_tokenizer_encode(model->tokenizer, prompt, &num_prompt_tokens);
    }

    if (!prompt_tokens || num_prompt_tokens <= 0) {
        LOG_ERROR("cml_llama_generate: tokenization failed or empty prompt");
        if (prompt_tokens) free(prompt_tokens);
        return NULL;
    }

    int max_total = num_prompt_tokens + config->max_new_tokens;
    CMLGenerationResult* result = (CMLGenerationResult*)calloc(1, sizeof(CMLGenerationResult));
    if (!result) { free(prompt_tokens); return NULL; }

    result->token_ids = (int*)malloc((size_t)max_total * sizeof(int));
    if (!result->token_ids) {
        free(prompt_tokens);
        free(result);
        return NULL;
    }

    /* Copy prompt tokens into result */
    memcpy(result->token_ids, prompt_tokens, (size_t)num_prompt_tokens * sizeof(int));
    result->num_tokens = num_prompt_tokens;

    /* Reset KV caches */
    cml_llama_reset(model);

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    Tensor* logits = cml_llama_forward(model, prompt_tokens, num_prompt_tokens);
    free(prompt_tokens);

    if (!logits) {
        LOG_ERROR("cml_llama_generate: prefill forward failed");
        free(result->token_ids);
        free(result);
        return NULL;
    }

    for (int step = 0; step < config->max_new_tokens; step++) {
        int next_token = cml_llama_sample_token(logits, config);
        tensor_free(logits);
        logits = NULL;

        if (next_token < 0) {
            LOG_ERROR("cml_llama_generate: sampling failed at step %d", step);
            break;
        }

        /* Append token */
        result->token_ids[result->num_tokens] = next_token;
        result->num_tokens++;

        /* Check EOS */
        if (next_token == config->eos_token_id) {
            break;
        }

        /* Check sequence length limit */
        if (model->current_seq_len >= model->config.max_seq_len) {
            LOG_WARNING("cml_llama_generate: max sequence length reached");
            break;
        }

        /* Forward with single new token */
        logits = cml_llama_forward(model, &next_token, 1);
        if (!logits) {
            LOG_ERROR("cml_llama_generate: decode forward failed at step %d", step);
            break;
        }
    }

    if (logits) tensor_free(logits);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed_ms = (ts_end.tv_sec - ts_start.tv_sec) * 1000.0 +
                        (ts_end.tv_nsec - ts_start.tv_nsec) / 1e6;
    result->total_time_ms = (float)elapsed_ms;

    int generated_tokens = result->num_tokens - num_prompt_tokens;
    if (elapsed_ms > 0.0) {
        result->tokens_per_second = (float)(generated_tokens * 1000.0 / elapsed_ms);
    }

    if (model->tokenizer) {
        result->text = cml_tokenizer_decode(model->tokenizer,
                                             result->token_ids, result->num_tokens);
    } else {
        result->text = NULL;
    }

    LOG_INFO("Generation complete: %d tokens in %.1f ms (%.1f tok/s)",
             generated_tokens, result->total_time_ms, result->tokens_per_second);

    return result;
}

void cml_generation_result_free(CMLGenerationResult* result) {
    if (!result) return;
    if (result->token_ids) free(result->token_ids);
    if (result->text) free(result->text);
    free(result);
}

void cml_llama_reset(CMLLLaMAModel* model) {
    if (!model) return;

    /* Reset all KV caches */
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i] && model->layers[i]->kv_cache) {
            cml_kv_cache_reset(model->layers[i]->kv_cache);
        }
    }
    model->current_seq_len = 0;
}

void cml_llama_print_config(const CMLLLaMAConfig* config) {
    if (!config) return;

    int head_dim = config->hidden_size / config->num_heads;
    int q_params = config->hidden_size * config->num_heads * head_dim;
    int kv_params = 2 * config->hidden_size * config->num_kv_heads * head_dim;
    int ffn_params = 3 * config->hidden_size * config->intermediate_size;
    int layer_params = q_params + kv_params + ffn_params;
    long long total_params = (long long)config->num_layers * layer_params +
                             (long long)config->vocab_size * config->hidden_size * 2;

    printf("LLaMA Configuration:\n");
    printf("  vocab_size:        %d\n", config->vocab_size);
    printf("  hidden_size:       %d\n", config->hidden_size);
    printf("  intermediate_size: %d\n", config->intermediate_size);
    printf("  num_layers:        %d\n", config->num_layers);
    printf("  num_heads:         %d\n", config->num_heads);
    printf("  num_kv_heads:      %d\n", config->num_kv_heads);
    printf("  head_dim:          %d\n", head_dim);
    printf("  max_seq_len:       %d\n", config->max_seq_len);
    printf("  rope_theta:        %.1f\n", config->rope_theta);
    printf("  rms_norm_eps:      %.1e\n", config->rms_norm_eps);
    printf("  est. parameters:   %.1fB\n", total_params / 1e9);
}
