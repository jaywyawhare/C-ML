#include "nn/llama.h"
#include "core/threefry.h"
#include "nn/llm_ops.h"
#include "nn/sampling.h"
#include "core/gguf.h"
#include "ops/uops.h"
#include "core/logging.h"
#include "tensor/tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "alloc/cml_allocator.h"

/**
 * Standard hyperparameters for LLaMA-7B (32 layers, hidden 4096, MHA).
 * @return (CMLLLaMAConfig)
 */
CMLLLaMAConfig cml_llama_config_7b(void) {
    CMLLLaMAConfig config = {.vocab_size        = 32000,
                             .hidden_size       = 4096,
                             .intermediate_size = 11008,
                             .num_layers        = 32,
                             .num_heads         = 32,
                             .num_kv_heads      = 32,
                             .max_seq_len       = 2048,
                             .rope_theta        = 10000.0f,
                             .rms_norm_eps      = 1e-5f};
    return config;
}

/**
 * Standard hyperparameters for LLaMA-13B (40 layers, hidden 5120, MHA).
 * @return (CMLLLaMAConfig)
 */
CMLLLaMAConfig cml_llama_config_13b(void) {
    CMLLLaMAConfig config = {.vocab_size        = 32000,
                             .hidden_size       = 5120,
                             .intermediate_size = 13824,
                             .num_layers        = 40,
                             .num_heads         = 40,
                             .num_kv_heads      = 40,
                             .max_seq_len       = 2048,
                             .rope_theta        = 10000.0f,
                             .rms_norm_eps      = 1e-5f};
    return config;
}

/**
 * Standard hyperparameters for LLaMA-70B (80 layers, hidden 8192, GQA with 8 KV heads).
 * @return (CMLLLaMAConfig)
 */
CMLLLaMAConfig cml_llama_config_70b(void) {
    CMLLLaMAConfig config = {.vocab_size        = 32000,
                             .hidden_size       = 8192,
                             .intermediate_size = 28672,
                             .num_layers        = 80,
                             .num_heads         = 64,
                             .num_kv_heads      = 8,
                             .max_seq_len       = 4096,
                             .rope_theta        = 10000.0f,
                             .rms_norm_eps      = 1e-5f};
    return config;
}

/**
 * Default sampling config for generation (temp 0.8, top-p 0.9, top-k 40).
 * @return (CMLGenerationConfig)
 */
CMLGenerationConfig cml_generation_default_config(void) {
    CMLGenerationConfig config = {.temperature          = 0.8f,
                                  .top_p                = 0.9f,
                                  .top_k                = 40,
                                  .min_p                = 0.0f,
                                  .typical_p            = 0.0f,
                                  .repetition_penalty   = 1.0f,
                                  .frequency_penalty    = 0.0f,
                                  .presence_penalty     = 0.0f,
                                  .no_repeat_ngram_size = 0,
                                  .logit_bias_ids       = NULL,
                                  .logit_bias_values    = NULL,
                                  .logit_bias_count     = 0,
                                  .stop_sequences       = NULL,
                                  .num_stop_sequences   = 0,
                                  .seed                 = 0,
                                  .max_new_tokens       = 256,
                                  .eos_token_id         = 2,
                                  .do_sample            = true};
    return config;
}

/**
 * RMSNorm over the last dim: x * rsqrt(mean(x^2) + eps), scaled by `weight`.
 * One fused kernel (UOP_RMSNORM) instead of a materialized mul/mean/rsqrt chain,
 * so the lazy graph fuses it and realizes a single output.
 * @param x (Tensor*)
 * @param weight (Tensor*)
 * @param eps (float)
 * @return (Tensor*)
 */
static Tensor* rms_norm(Tensor* x, Tensor* weight, float eps) {
    if (!x || !weight)
        return NULL;
    return uop_rmsnorm(x, weight, eps);
}

/**
 * SwiGLU feed-forward block: down_proj(silu(x·gate_proj) * (x·up_proj)).
 * @param x (Tensor*)
 * @param gate_proj (Tensor*)
 * @param up_proj (Tensor*)
 * @param down_proj (Tensor*)
 * @return (Tensor*)
 */
static Tensor* swiglu_ffn(Tensor* x, Tensor* gate_proj, Tensor* up_proj, Tensor* down_proj) {
    if (!x || !gate_proj || !up_proj || !down_proj)
        return NULL;

    /* gate_out = x @ gate_proj -> [seq_len, intermediate_size] */
    Tensor* gate_out = uop_matmul(x, gate_proj);
    if (!gate_out)
        return NULL;

    /* gate_activated = silu(gate_out). No forced realize here or on `combined`:
     * leaving them lazy lets the scheduler fuse silu(gate)*up and the gate/up
     * matmul epilogues into one pass. Only the block output is realized. */
    Tensor* gate_activated = uop_silu(gate_out);
    if (!gate_activated)
        return NULL;

    /* up_out = x @ up_proj -> [seq_len, intermediate_size] */
    Tensor* up_out = uop_matmul(x, up_proj);
    if (!up_out)
        return NULL;

    /* combined = gate_activated * up_out */
    Tensor* combined = uop_mul(gate_activated, up_out);
    if (!combined)
        return NULL;

    /* output = combined @ down_proj -> [seq_len, hidden_size] */
    Tensor* output = uop_matmul(combined, down_proj);
    if (output)
        tensor_ensure_executed(output);

    return output;
}

/**
 * Allocate one transformer layer with its own KV cache; weight tensors start NULL.
 * @param config (const CMLLLaMAConfig*)
 * @return (CMLLLaMALayer*)
 */
static CMLLLaMALayer* llama_layer_create(const CMLLLaMAConfig* config) {
    CMLLLaMALayer* layer = (CMLLLaMALayer*)cml_calloc(1, sizeof(CMLLLaMALayer));
    if (!layer)
        return NULL;

    int head_dim = config->hidden_size / config->num_heads;

    /* Create KV cache for this layer */
    layer->kv_cache = cml_kv_cache_create(config->max_seq_len, config->num_kv_heads, head_dim);
    if (!layer->kv_cache) {
        cml_free(layer);
        return NULL;
    }

    /* Weight tensors are set to NULL here; they get populated during weight loading
     * or can be initialized with random values for testing */

    return layer;
}

/**
 * Free a layer's weight tensors and KV cache.
 * @param layer (CMLLLaMALayer*)
 * @return (void)
 */
static void llama_layer_free(CMLLLaMALayer* layer) {
    if (!layer)
        return;

    if (layer->q_proj)
        tensor_free(layer->q_proj);
    if (layer->k_proj)
        tensor_free(layer->k_proj);
    if (layer->v_proj)
        tensor_free(layer->v_proj);
    if (layer->o_proj)
        tensor_free(layer->o_proj);
    if (layer->gate_proj)
        tensor_free(layer->gate_proj);
    if (layer->up_proj)
        tensor_free(layer->up_proj);
    if (layer->down_proj)
        tensor_free(layer->down_proj);
    if (layer->input_layernorm)
        tensor_free(layer->input_layernorm);
    if (layer->post_attn_layernorm)
        tensor_free(layer->post_attn_layernorm);
    if (layer->kv_cache)
        cml_kv_cache_free(layer->kv_cache);

    cml_free(layer);
}

/**
 * Allocate a LLaMA model and all its layers from `config`; weights loaded separately.
 * @param config (const CMLLLaMAConfig*)
 * @return (CMLLLaMAModel*)
 */
CMLLLaMAModel* cml_llama_create(const CMLLLaMAConfig* config) {
    if (!config) {
        LOG_ERROR("cml_llama_create: NULL config");
        return NULL;
    }

    CMLLLaMAModel* model = (CMLLLaMAModel*)cml_calloc(1, sizeof(CMLLLaMAModel));
    if (!model) {
        LOG_ERROR("cml_llama_create: allocation failed");
        return NULL;
    }

    model->config          = *config;
    model->num_layers      = config->num_layers;
    model->weights_loaded  = false;
    model->current_seq_len = 0;

    /* Allocate layer array */
    model->layers = (CMLLLaMALayer**)cml_calloc((size_t)config->num_layers, sizeof(CMLLLaMALayer*));
    if (!model->layers) {
        LOG_ERROR("cml_llama_create: layer array allocation failed");
        cml_free(model);
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
            cml_free(model->layers);
            cml_free(model);
            return NULL;
        }
    }

    LOG_INFO("LLaMA model created: %d layers, hidden=%d, heads=%d, kv_heads=%d", config->num_layers,
             config->hidden_size, config->num_heads, config->num_kv_heads);

    return model;
}

/**
 * Free the model: all layers, embedding/norm/head weights and the tokenizer.
 * @param model (CMLLLaMAModel*)
 * @return (void)
 */
void cml_llama_free(CMLLLaMAModel* model) {
    if (!model)
        return;

    /* Free layers */
    if (model->layers) {
        for (int i = 0; i < model->num_layers; i++) {
            llama_layer_free(model->layers[i]);
        }
        cml_free(model->layers);
    }

    /* Free embeddings and output weights */
    if (model->embed_tokens)
        tensor_free(model->embed_tokens);
    if (model->norm)
        tensor_free(model->norm);
    if (model->lm_head)
        tensor_free(model->lm_head);

    /* Free tokenizer */
    if (model->tokenizer)
        cml_tokenizer_free(model->tokenizer);

    cml_free(model);
}

/**
 * Read one GGUF tensor and bind it to the matching model/layer weight slot,
 * accepting both HF (model.layers.N.*) and llama.cpp (blk.N.*) naming.
 * @param model (CMLLLaMAModel*)
 * @param ctx (GGUFContext*)
 * @param name (const char*)
 * @return (int)
 */
static int load_tensor_by_name(CMLLLaMAModel* model, GGUFContext* ctx, const char* name) {
    Tensor* t = gguf_read_tensor(ctx, name);
    if (!t) {
        LOG_WARNING("GGUF: tensor '%s' not found, skipping", name);
        return -1;
    }

    /* Match model.embed_tokens.weight */
    if (strcmp(name, "model.embed_tokens.weight") == 0 || strcmp(name, "token_embd.weight") == 0) {
        if (model->embed_tokens)
            tensor_free(model->embed_tokens);
        model->embed_tokens = t;
        return 0;
    }

    /* Match model.norm.weight */
    if (strcmp(name, "model.norm.weight") == 0 || strcmp(name, "output_norm.weight") == 0) {
        if (model->norm)
            tensor_free(model->norm);
        model->norm = t;
        return 0;
    }

    /* Match lm_head.weight */
    if (strcmp(name, "lm_head.weight") == 0 || strcmp(name, "output.weight") == 0) {
        if (model->lm_head)
            tensor_free(model->lm_head);
        model->lm_head = t;
        return 0;
    }

    /* Try to parse layer index: model.layers.N.xxx or blk.N.xxx */
    int layer_idx      = -1;
    const char* suffix = NULL;

    /* Try "model.layers.N." format */
    if (strncmp(name, "model.layers.", 13) == 0) {
        const char* p = name + 13;
        layer_idx     = (int)strtol(p, (char**)&suffix, 10);
        if (suffix && *suffix == '.') {
            suffix++; /* skip the dot */
        } else {
            layer_idx = -1;
        }
    }
    /* Try "blk.N." format (llama.cpp style) */
    else if (strncmp(name, "blk.", 4) == 0) {
        const char* p = name + 4;
        layer_idx     = (int)strtol(p, (char**)&suffix, 10);
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
    if (strcmp(suffix, "self_attn.q_proj.weight") == 0 || strcmp(suffix, "attn_q.weight") == 0) {
        if (layer->q_proj)
            tensor_free(layer->q_proj);
        layer->q_proj = t;
    } else if (strcmp(suffix, "self_attn.k_proj.weight") == 0 ||
               strcmp(suffix, "attn_k.weight") == 0) {
        if (layer->k_proj)
            tensor_free(layer->k_proj);
        layer->k_proj = t;
    } else if (strcmp(suffix, "self_attn.v_proj.weight") == 0 ||
               strcmp(suffix, "attn_v.weight") == 0) {
        if (layer->v_proj)
            tensor_free(layer->v_proj);
        layer->v_proj = t;
    } else if (strcmp(suffix, "self_attn.o_proj.weight") == 0 ||
               strcmp(suffix, "attn_output.weight") == 0) {
        if (layer->o_proj)
            tensor_free(layer->o_proj);
        layer->o_proj = t;
    }
    /* FFN projections */
    else if (strcmp(suffix, "mlp.gate_proj.weight") == 0 ||
             strcmp(suffix, "ffn_gate.weight") == 0) {
        if (layer->gate_proj)
            tensor_free(layer->gate_proj);
        layer->gate_proj = t;
    } else if (strcmp(suffix, "mlp.up_proj.weight") == 0 || strcmp(suffix, "ffn_up.weight") == 0) {
        if (layer->up_proj)
            tensor_free(layer->up_proj);
        layer->up_proj = t;
    } else if (strcmp(suffix, "mlp.down_proj.weight") == 0 ||
               strcmp(suffix, "ffn_down.weight") == 0) {
        if (layer->down_proj)
            tensor_free(layer->down_proj);
        layer->down_proj = t;
    }
    /* Layer norms */
    else if (strcmp(suffix, "input_layernorm.weight") == 0 ||
             strcmp(suffix, "attn_norm.weight") == 0) {
        if (layer->input_layernorm)
            tensor_free(layer->input_layernorm);
        layer->input_layernorm = t;
    } else if (strcmp(suffix, "post_attention_layernorm.weight") == 0 ||
               strcmp(suffix, "ffn_norm.weight") == 0) {
        if (layer->post_attn_layernorm)
            tensor_free(layer->post_attn_layernorm);
        layer->post_attn_layernorm = t;
    } else {
        LOG_WARNING("GGUF: unrecognized layer suffix '%s' for layer %d", suffix, layer_idx);
        tensor_free(t);
        return -1;
    }

    return 0;
}

/**
 * Load all weights and tokenizer metadata from a GGUF file into `model`, tying
 * lm_head to the embeddings when the file omits an output projection.
 * @param model (CMLLLaMAModel*)
 * @param filepath (const char*)
 * @return (int)
 */
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

    int loaded  = 0;
    int skipped = 0;

    for (int i = 0; i < num_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx, i);
        if (!name)
            continue;

        if (load_tensor_by_name(model, ctx, name) == 0) {
            loaded++;
        } else {
            skipped++;
        }
    }

    /* Wire the GGUF tokenizer metadata into a BPE tokenizer so generate()
     * can encode prompts and decode output text. cml_tokenizer_create copies
     * the strings, so this is safe across gguf_close. */
    if (!model->tokenizer) {
        char** tokens  = NULL;
        char** merges  = NULL;
        int num_tokens = 0, num_merges = 0;
        if (gguf_get_tokenizer(ctx, &tokens, &num_tokens, &merges, &num_merges) == 0) {
            model->tokenizer = cml_tokenizer_create(tokens, num_tokens, merges, num_merges);
            if (model->tokenizer)
                LOG_INFO("Loaded GGUF tokenizer: %d tokens, %d merges", num_tokens, num_merges);
            else
                LOG_WARNING("GGUF tokenizer metadata present but tokenizer creation failed");
        } else {
            LOG_WARNING("GGUF file has no tokenizer.ggml.tokens metadata; "
                        "generate() will return token IDs without text");
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

/**
 * Gather embedding rows for `token_ids` into a [seq_len, hidden_size] tensor.
 * @param embed (Tensor*)
 * @param token_ids (const int*)
 * @param seq_len (int)
 * @return (Tensor*)
 */
static Tensor* embed_tokens_lookup(Tensor* embed, const int* token_ids, int seq_len) {
    if (!embed || !token_ids || seq_len <= 0)
        return NULL;

    int hidden_size = embed->shape[1];
    int shape[]     = {seq_len, hidden_size};
    Tensor* output  = tensor_zeros(shape, 2, NULL);
    if (!output)
        return NULL;

    tensor_ensure_executed(embed);
    tensor_ensure_executed(output);

    float* embed_data = (float*)tensor_data_ptr(embed);
    float* out_data   = (float*)tensor_data_ptr(output);
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
        memcpy(out_data + i * hidden_size, embed_data + tid * hidden_size,
               (size_t)hidden_size * sizeof(float));
    }

    return output;
}

/**
 * One decoder layer: RMSNorm -> RoPE'd QKV -> cached causal GQA -> output proj with
 * residual, then RMSNorm -> SwiGLU FFN with residual. `start_pos` indexes the KV cache.
 * @param model (CMLLLaMAModel*)
 * @param layer (CMLLLaMALayer*)
 * @param hidden (Tensor*)
 * @param start_pos (int)
 * @return (Tensor*)
 */
Tensor* cml_llama_layer_forward(CMLLLaMAModel* model, CMLLLaMALayer* layer, Tensor* hidden,
                                int start_pos) {
    if (!model || !layer || !hidden)
        return NULL;

    const CMLLLaMAConfig* cfg = &model->config;
    int head_dim              = cfg->hidden_size / cfg->num_heads;

    Tensor* normed = rms_norm(hidden, layer->input_layernorm, cfg->rms_norm_eps);
    if (!normed)
        return NULL;

    Tensor* Q = uop_matmul(normed, layer->q_proj);
    Tensor* K = uop_matmul(normed, layer->k_proj);
    Tensor* V = uop_matmul(normed, layer->v_proj);
    if (Q)
        tensor_ensure_executed(Q);
    if (K)
        tensor_ensure_executed(K);
    if (V)
        tensor_ensure_executed(V);

    if (!Q || !K || !V)
        return NULL;

    CMLRoPEConfig rope_cfg = {
        .dim = head_dim, .max_seq_len = cfg->max_seq_len, .base = cfg->rope_theta};

    Tensor* Q_rope = cml_rope_forward(Q, start_pos, &rope_cfg);
    Tensor* K_rope = cml_rope_forward(K, start_pos, &rope_cfg);
    /* cml_rope_forward modifies in-place and returns the same pointer,
     * so only free Q/K if rope returned a different (new) tensor */
    if (Q_rope != Q)
        tensor_free(Q);
    if (K_rope != K)
        tensor_free(K);

    if (!Q_rope || !K_rope)
        return NULL;

    CMLGQAConfig gqa_cfg = {.num_heads    = cfg->num_heads,
                            .num_kv_heads = cfg->num_kv_heads,
                            .head_dim     = head_dim,
                            .scale        = 0.0f, /* auto: 1/sqrt(head_dim) */
                            .causal       = true};

    /* GQA expects 3D [batch, seq, dim] -- add batch=1 dimension */
    int seq_len = Q_rope->shape[0];
    int q_3d[]  = {1, seq_len, Q_rope->shape[Q_rope->ndim - 1]};
    int k_3d[]  = {1, seq_len, K_rope->shape[K_rope->ndim - 1]};
    int v_3d[]  = {1, seq_len, V->shape[V->ndim - 1]};

    Tensor* Q3 = tensor_reshape(Q_rope, q_3d, 3);
    Tensor* K3 = tensor_reshape(K_rope, k_3d, 3);
    Tensor* V3 = tensor_reshape(V, v_3d, 3);

    Tensor* attn_out = cml_gqa_forward_cached(Q3, K3, V3, layer->kv_cache, &gqa_cfg);
    tensor_free(Q3);
    tensor_free(K3);
    tensor_free(V3);

    if (!attn_out)
        return NULL;

    /* Squeeze batch dimension: [1, seq, dim] -> [seq, dim] */
    int out_2d[]    = {attn_out->shape[1], attn_out->shape[2]};
    Tensor* attn_2d = tensor_reshape(attn_out, out_2d, 2);

    Tensor* attn_proj = uop_matmul(attn_2d, layer->o_proj);
    if (attn_proj)
        tensor_ensure_executed(attn_proj);
    tensor_free(attn_2d);
    tensor_free(attn_out);
    if (!attn_proj)
        return NULL;

    Tensor* residual1 = uop_add(hidden, attn_proj);
    if (residual1)
        tensor_ensure_executed(residual1);
    if (!residual1)
        return NULL;

    Tensor* normed2 = rms_norm(residual1, layer->post_attn_layernorm, cfg->rms_norm_eps);
    if (!normed2)
        return NULL;

    Tensor* ffn_out = swiglu_ffn(normed2, layer->gate_proj, layer->up_proj, layer->down_proj);
    if (ffn_out)
        tensor_ensure_executed(ffn_out);
    if (!ffn_out)
        return NULL;

    Tensor* output = uop_add(residual1, ffn_out);
    if (output)
        tensor_ensure_executed(output);

    return output;
}

/**
 * Full forward pass over `token_ids`: embed, run every layer, final norm and
 * lm_head projection to logits. Advances current_seq_len for KV-cached decoding.
 * @param model (CMLLLaMAModel*)
 * @param token_ids (const int*)
 * @param seq_len (int)
 * @return (Tensor*)
 */
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
        Tensor* next_hidden = cml_llama_layer_forward(model, model->layers[i], hidden, start_pos);
        if (!next_hidden) {
            LOG_ERROR("cml_llama_forward: layer %d failed", i);
            return NULL;
        }
        hidden = next_hidden;
    }

    if (model->norm) {
        Tensor* normed = rms_norm(hidden, model->norm, model->config.rms_norm_eps);
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
        PermuteParams pp = {.perm = (int[]){1, 0}, .num_dims = 2};
        Tensor* embed_t  = uop_permute(model->embed_tokens, &pp);
        if (embed_t) {
            logits = uop_matmul(hidden, embed_t);
        }
    }

    if (!logits) {
        LOG_ERROR("cml_llama_forward: lm_head projection failed");
        return NULL;
    }

    model->current_seq_len = start_pos + seq_len;
    return logits;
}

/**
 * Pick the next token from the last position's logits: greedy argmax, or logit bias and the
 * history-dependent penalties followed by temperature and the top-k / typical / top-p / min-p
 * warpers. Writes the chosen token's log-probability to out_logprob when non-NULL.
 * @param logits (Tensor*)
 * @param config (const CMLGenerationConfig*)
 * @param history (const int*)
 * @param history_len (int)
 * @param out_logprob (float*)
 * @return (int)
 */
int cml_llama_sample_token(Tensor* logits, const CMLGenerationConfig* config, const int* history,
                           int history_len, float* out_logprob) {
    if (!logits || !config)
        return -1;

    tensor_ensure_executed(logits);
    float* data = (float*)tensor_data_ptr(logits);
    if (!data)
        return -1;

    /* Get the last token's logits if logits is 2D [seq_len, vocab_size] */
    int vocab_size;
    float* token_logits;
    if (logits->ndim == 2) {
        vocab_size   = logits->shape[1];
        int last_pos = logits->shape[0] - 1;
        token_logits = data + last_pos * vocab_size;
    } else if (logits->ndim == 1) {
        vocab_size   = logits->shape[0];
        token_logits = data;
    } else {
        LOG_ERROR("cml_llama_sample_token: unexpected logits ndim=%d", logits->ndim);
        return -1;
    }

    /* Greedy decoding */
    if (!config->do_sample || config->temperature <= 0.0f) {
        int best_id    = 0;
        float best_val = token_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (token_logits[i] > best_val) {
                best_val = token_logits[i];
                best_id  = i;
            }
        }
        if (out_logprob) {
            double se = 0.0;
            for (int i = 0; i < vocab_size; i++)
                se += exp((double)(token_logits[i] - best_val));
            *out_logprob = (float)((double)(token_logits[best_id] - best_val) - log(se));
        }
        return best_id;
    }

    float* work  = (float*)cml_malloc((size_t)vocab_size * sizeof(float));
    float* probs = (float*)cml_malloc((size_t)vocab_size * sizeof(float));
    if (!work || !probs) {
        cml_free(work);
        cml_free(probs);
        return -1;
    }
    memcpy(work, token_logits, (size_t)vocab_size * sizeof(float));

    /* Logit bias and history-dependent penalties act on raw logits, then temperature and warpers.
     */
    cml_logits_bias(work, vocab_size, config->logit_bias_ids, config->logit_bias_values,
                    config->logit_bias_count);
    cml_logits_repetition_penalty(work, vocab_size, history, history_len,
                                  config->repetition_penalty);
    cml_logits_frequency_presence_penalty(work, vocab_size, history, history_len,
                                          config->frequency_penalty, config->presence_penalty);
    cml_logits_no_repeat_ngram(work, vocab_size, history, history_len,
                               config->no_repeat_ngram_size);
    cml_logits_temperature(work, vocab_size, config->temperature);
    cml_logits_top_k(work, vocab_size, config->top_k);
    cml_logits_typical(work, vocab_size, config->typical_p);
    cml_logits_top_p(work, vocab_size, config->top_p);
    cml_logits_min_p(work, vocab_size, config->min_p);
    cml_softmax_row_temp(work, vocab_size, 1.0f, probs); /* masked logits -> 0 probability */

    float r;
    cml_rng_uniform(cml_rng_get_global(), &r, 1);
    float acc      = 0.0f;
    float best     = -1.0f;
    int sampled_id = 0; /* falls back to the argmax if rounding leaves r past the total */
    for (int i = 0; i < vocab_size; i++) {
        if (probs[i] > best) {
            best       = probs[i];
            sampled_id = i;
        }
        acc += probs[i];
        if (r <= acc) {
            sampled_id = i;
            break;
        }
    }

    if (out_logprob) {
        float p      = probs[sampled_id];
        *out_logprob = (p > 0.0f) ? logf(p) : -INFINITY;
    }

    cml_free(work);
    cml_free(probs);
    return sampled_id;
}

/**
 * True when the decoded generated tokens contain any non-empty stop sequence.
 * @param tok (CMLTokenizer*)
 * @param gen_ids (const int*)
 * @param gen_len (int)
 * @param stops (const char**)
 * @param num_stops (int)
 * @return (bool)
 */
static bool generated_hits_stop(CMLTokenizer* tok, const int* gen_ids, int gen_len,
                                const char** stops, int num_stops) {
    if (!tok || num_stops <= 0 || gen_len <= 0)
        return false;
    char* text = cml_tokenizer_decode(tok, gen_ids, gen_len);
    if (!text)
        return false;
    bool hit = false;
    for (int s = 0; s < num_stops; s++) {
        if (stops[s] && stops[s][0] && strstr(text, stops[s])) {
            hit = true;
            break;
        }
    }
    cml_free(text);
    return hit;
}

/**
 * Tokenize `prompt`, prefill, then autoregressively decode up to max_new_tokens
 * (stopping on EOS, a stop sequence or the context limit), returning the token ids, decoded text,
 * per-token logprobs and timing.
 * @param model (CMLLLaMAModel*)
 * @param prompt (const char*)
 * @param config (const CMLGenerationConfig*)
 * @return (CMLGenerationResult*)
 */
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
    int* prompt_tokens    = NULL;

    if (model->tokenizer) {
        prompt_tokens = cml_tokenizer_encode(model->tokenizer, prompt, &num_prompt_tokens);
    }

    if (!prompt_tokens || num_prompt_tokens <= 0) {
        LOG_ERROR("cml_llama_generate: tokenization failed or empty prompt");
        if (prompt_tokens)
            cml_free(prompt_tokens);
        return NULL;
    }

    int max_total               = num_prompt_tokens + config->max_new_tokens;
    CMLGenerationResult* result = (CMLGenerationResult*)cml_calloc(1, sizeof(CMLGenerationResult));
    if (!result) {
        cml_free(prompt_tokens);
        return NULL;
    }

    result->token_ids = (int*)cml_malloc((size_t)max_total * sizeof(int));
    if (!result->token_ids) {
        cml_free(prompt_tokens);
        cml_free(result);
        return NULL;
    }

    result->token_logprobs = (float*)cml_malloc((size_t)config->max_new_tokens * sizeof(float));
    result->num_logprobs   = 0;

    /* Copy prompt tokens into result */
    memcpy(result->token_ids, prompt_tokens, (size_t)num_prompt_tokens * sizeof(int));
    result->num_tokens = num_prompt_tokens;

    /* A non-zero seed makes sampling reproducible across identical calls. */
    if (config->seed != 0)
        cml_rng_set_global_seed(config->seed);

    /* Reset KV caches */
    cml_llama_reset(model);

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    Tensor* logits = cml_llama_forward(model, prompt_tokens, num_prompt_tokens);
    cml_free(prompt_tokens);

    if (!logits) {
        LOG_ERROR("cml_llama_generate: prefill forward failed");
        cml_free(result->token_ids);
        cml_free(result);
        return NULL;
    }

    for (int step = 0; step < config->max_new_tokens; step++) {
        float lp = 0.0f;
        int next_token =
            cml_llama_sample_token(logits, config, result->token_ids, result->num_tokens,
                                   result->token_logprobs ? &lp : NULL);
        tensor_free(logits);
        logits = NULL;

        if (next_token < 0) {
            LOG_ERROR("cml_llama_generate: sampling failed at step %d", step);
            break;
        }

        /* Append token */
        result->token_ids[result->num_tokens] = next_token;
        result->num_tokens++;
        if (result->token_logprobs)
            result->token_logprobs[result->num_logprobs++] = lp;

        /* Check EOS */
        if (next_token == config->eos_token_id) {
            break;
        }

        if (generated_hits_stop(model->tokenizer, result->token_ids + num_prompt_tokens,
                                result->num_tokens - num_prompt_tokens, config->stop_sequences,
                                config->num_stop_sequences)) {
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

    if (logits)
        tensor_free(logits);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed_ms =
        (ts_end.tv_sec - ts_start.tv_sec) * 1000.0 + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e6;
    result->total_time_ms = (float)elapsed_ms;

    int generated_tokens = result->num_tokens - num_prompt_tokens;
    if (elapsed_ms > 0.0) {
        result->tokens_per_second = (float)(generated_tokens * 1000.0 / elapsed_ms);
    }

    if (model->tokenizer) {
        result->text =
            cml_tokenizer_decode(model->tokenizer, result->token_ids, result->num_tokens);
    } else {
        result->text = NULL;
    }

    LOG_INFO("Generation complete: %d tokens in %.1f ms (%.1f tok/s)", generated_tokens,
             result->total_time_ms, result->tokens_per_second);

    return result;
}

/**
 * Free a generation result's token array and decoded text.
 * @param result (CMLGenerationResult*)
 * @return (void)
 */
void cml_generation_result_free(CMLGenerationResult* result) {
    if (!result)
        return;
    if (result->token_ids)
        cml_free(result->token_ids);
    if (result->text)
        cml_free(result->text);
    if (result->token_logprobs)
        cml_free(result->token_logprobs);
    cml_free(result);
}

/**
 * Clear every layer's KV cache and reset the current sequence position to 0.
 * @param model (CMLLLaMAModel*)
 * @return (void)
 */
void cml_llama_reset(CMLLLaMAModel* model) {
    if (!model)
        return;

    /* Reset all KV caches */
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i] && model->layers[i]->kv_cache) {
            cml_kv_cache_reset(model->layers[i]->kv_cache);
        }
    }
    model->current_seq_len = 0;
}

/**
 * Print the config fields and an estimated parameter count to stdout.
 * @param config (const CMLLLaMAConfig*)
 * @return (void)
 */
void cml_llama_print_config(const CMLLLaMAConfig* config) {
    if (!config)
        return;

    int head_dim           = config->hidden_size / config->num_heads;
    int q_params           = config->hidden_size * config->num_heads * head_dim;
    int kv_params          = 2 * config->hidden_size * config->num_kv_heads * head_dim;
    int ffn_params         = 3 * config->hidden_size * config->intermediate_size;
    int layer_params       = q_params + kv_params + ffn_params;
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
