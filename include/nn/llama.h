/**
 * @file llama.h
 * @brief LLaMA model inference
 */

#ifndef CML_NN_LLAMA_H
#define CML_NN_LLAMA_H

#include "tensor/tensor.h"
#include "nn/llm_ops.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLLLaMAConfig {
    int vocab_size;
    int hidden_size;      /* d_model */
    int intermediate_size; /* FFN hidden dim (usually ~2.7x hidden) */
    int num_layers;
    int num_heads;
    int num_kv_heads;     /* For GQA (< num_heads) */
    int max_seq_len;
    float rope_theta;     /* RoPE base frequency (default: 10000.0) */
    float rms_norm_eps;   /* RMSNorm epsilon (default: 1e-5) */
    int tensor_parallel_size;  /* Number of TP ranks (default: 1) */
    int tensor_parallel_rank;  /* This rank's index (default: 0) */
} CMLLLaMAConfig;

/* Pre-defined configs */
CMLLLaMAConfig cml_llama_config_7b(void);
CMLLLaMAConfig cml_llama_config_13b(void);
CMLLLaMAConfig cml_llama_config_70b(void);

typedef struct CMLLLaMALayer {
    /* Self-attention projections */
    Tensor* q_proj;       /* [hidden_size, num_heads * head_dim] */
    Tensor* k_proj;       /* [hidden_size, num_kv_heads * head_dim] */
    Tensor* v_proj;       /* [hidden_size, num_kv_heads * head_dim] */
    Tensor* o_proj;       /* [num_heads * head_dim, hidden_size] */

    /* FFN (SwiGLU) */
    Tensor* gate_proj;    /* [hidden_size, intermediate_size] */
    Tensor* up_proj;      /* [hidden_size, intermediate_size] */
    Tensor* down_proj;    /* [intermediate_size, hidden_size] */

    /* Layer norms */
    Tensor* input_layernorm;   /* RMSNorm weight [hidden_size] */
    Tensor* post_attn_layernorm; /* RMSNorm weight [hidden_size] */

    /* KV cache for this layer */
    CMLKVCache* kv_cache;
} CMLLLaMALayer;

typedef struct CMLLLaMAModel {
    CMLLLaMAConfig config;

    /* Embeddings and output */
    Tensor* embed_tokens;  /* [vocab_size, hidden_size] */
    Tensor* norm;          /* Final RMSNorm weight [hidden_size] */
    Tensor* lm_head;       /* [hidden_size, vocab_size] (may share embed_tokens) */

    /* Layers */
    CMLLLaMALayer** layers;
    int num_layers;

    /* Tokenizer */
    CMLTokenizer* tokenizer;

    /* State */
    bool weights_loaded;
    int current_seq_len;   /* Current position in generation */
} CMLLLaMAModel;

typedef struct CMLGenerationConfig {
    float temperature;     /* Sampling temperature (default: 0.8) */
    float top_p;           /* Nucleus sampling (default: 0.9) */
    int top_k;             /* Top-k sampling (default: 40) */
    int max_new_tokens;    /* Max tokens to generate (default: 256) */
    int eos_token_id;      /* Stop token */
    bool do_sample;        /* Use sampling (vs greedy) */
} CMLGenerationConfig;

typedef struct CMLGenerationResult {
    int* token_ids;
    int num_tokens;
    char* text;            /* Decoded text (heap-allocated) */
    float total_time_ms;
    float tokens_per_second;
} CMLGenerationResult;

CMLGenerationConfig cml_generation_default_config(void);

/* Model lifecycle */
CMLLLaMAModel* cml_llama_create(const CMLLLaMAConfig* config);
void           cml_llama_free(CMLLLaMAModel* model);

/* Weight loading */
int cml_llama_load_gguf(CMLLLaMAModel* model, const char* filepath);

/* Forward pass */
Tensor* cml_llama_forward(CMLLLaMAModel* model, const int* token_ids, int seq_len);
Tensor* cml_llama_layer_forward(CMLLLaMAModel* model, CMLLLaMALayer* layer,
                                 Tensor* hidden, int start_pos);

/* Generation */
CMLGenerationResult* cml_llama_generate(CMLLLaMAModel* model, const char* prompt,
                                          const CMLGenerationConfig* config);
void cml_generation_result_free(CMLGenerationResult* result);

/* Sampling */
int cml_llama_sample_token(Tensor* logits, const CMLGenerationConfig* config);

/* Utility */
void cml_llama_reset(CMLLLaMAModel* model);
void cml_llama_print_config(const CMLLLaMAConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_LLAMA_H */
