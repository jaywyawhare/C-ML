# Advanced Neural Network Features

This document covers C-ML's advanced neural network modules for LLM fine-tuning, inference, and serving.

---

## Table of Contents

1. [LoRA Adapters](#lora-adapters)
2. [QLoRA (Quantized LoRA)](#qlora-quantized-lora)
3. [LLM Operations](#llm-operations)
4. [Paged Attention](#paged-attention)
5. [LLaMA Model](#llama-model)
6. [Model Serving](#model-serving)
7. [Speculative Decoding](#speculative-decoding)

---

## LoRA Adapters

**File:** `include/nn/lora.h`

LoRA (Low-Rank Adaptation) freezes pre-trained weights and injects small trainable rank-decomposition matrices, drastically reducing trainable parameter count. For a linear layer `y = xW^T`, LoRA modifies it to:

```
y = xW^T + (alpha / rank) * x @ A^T @ B^T
```

where A is `[rank, in_features]` and B is `[out_features, rank]`. B is initialized to zero so training starts from the pre-trained checkpoint.

### Key API

```c
CMLLoRALinear* cml_lora_linear_create(Tensor* base_weight, int rank, float alpha);
void           cml_lora_linear_free(CMLLoRALinear* lora);
Tensor*        cml_lora_linear_forward(CMLLoRALinear* lora, Tensor* input);
int            cml_lora_linear_merge(CMLLoRALinear* lora);    // Fold into base weight
int            cml_lora_linear_unmerge(CMLLoRALinear* lora);  // Restore original weight

CMLLoRAAdapter* cml_lora_adapter_create(const char* name, int rank, float alpha);
void            cml_lora_adapter_free(CMLLoRAAdapter* adapter);
int             cml_lora_adapter_add_layer(CMLLoRAAdapter* adapter, CMLLoRALinear* layer);
int             cml_lora_adapter_merge_all(CMLLoRAAdapter* adapter);
int             cml_lora_adapter_unmerge_all(CMLLoRAAdapter* adapter);
```

### Usage Example

```c
#include "nn/lora.h"

CMLLoRALinear* lora = cml_lora_linear_create(base_weight, /*rank=*/8, /*alpha=*/8.0f);

Tensor* out = cml_lora_linear_forward(lora, input);

CMLLoRAAdapter* adapter = cml_lora_adapter_create("task_adapter", 8, 8.0f);
cml_lora_adapter_add_layer(adapter, lora);
cml_lora_adapter_merge_all(adapter);   // Fold LoRA into base weights for fast inference
cml_lora_adapter_unmerge_all(adapter); // Undo merge to switch adapters

cml_lora_adapter_free(adapter);
```

### Notes

- `base_weight` is **not owned** by the LoRA layer; it must outlive it.
- Merging modifies `base_weight` in-place and saves a frozen copy for unmerge.
- Typical rank values: 4, 8, or 16. Alpha is usually set equal to rank.

---

## QLoRA (Quantized LoRA)

**File:** `include/nn/qlora.h`

QLoRA stores frozen base weights in NF4 (4-bit Normal Float) format, reducing memory by ~8x compared to float32, while keeping LoRA matrices in full precision for training.

```
y = x @ dequant(W_nf4)^T + (alpha / rank) * x @ A^T @ B^T
```

### Key API

```c
CMLNF4Tensor* cml_nf4_tensor_create(Tensor* float_tensor, int block_size);
void          cml_nf4_tensor_free(CMLNF4Tensor* nf4);
Tensor*       cml_nf4_tensor_dequantize(const CMLNF4Tensor* nf4);

CMLQLoRALinear* cml_qlora_linear_create(Tensor* base_weight, int rank, float alpha, int block_size);
void            cml_qlora_linear_free(CMLQLoRALinear* qlora);
Tensor*         cml_qlora_linear_forward(CMLQLoRALinear* qlora, Tensor* input);

size_t cml_qlora_memory_usage(const CMLQLoRALinear* qlora);
size_t cml_qlora_full_memory_usage(int in_features, int out_features);
```

### Usage Example

```c
#include "nn/qlora.h"

CMLQLoRALinear* qlora = cml_qlora_linear_create(base_weight, /*rank=*/8,
                                                  /*alpha=*/8.0f, /*block_size=*/64);

Tensor* out = cml_qlora_linear_forward(qlora, input);

size_t qlora_mem = cml_qlora_memory_usage(qlora);
size_t full_mem  = cml_qlora_full_memory_usage(4096, 4096);
printf("Memory: %zu bytes vs %zu bytes (%.1fx reduction)\n",
       qlora_mem, full_mem, (float)full_mem / qlora_mem);

cml_qlora_linear_free(qlora);
```

### Notes

- `base_weight` is consumed (quantized) during creation.
- The forward pass dequantizes on-the-fly to a float32 temporary for matmul.
- `block_size` of 64 is the typical choice. `enable_double_quant` quantizes scale factors too for additional savings.

---

## LLM Operations

**File:** `include/nn/llm_ops.h`

Core primitives for LLM inference: KV cache, Grouped Query Attention, Flash Attention, RoPE, Mixture of Experts, and BPE tokenization.

### KV Cache

Contiguous key/value storage for autoregressive decoding.

```c
CMLKVCache* cml_kv_cache_create(int max_seq_len, int num_kv_heads, int head_dim);
void        cml_kv_cache_free(CMLKVCache* cache);
int         cml_kv_cache_append(CMLKVCache* cache, Tensor* new_key, Tensor* new_value);
void        cml_kv_cache_reset(CMLKVCache* cache);
Tensor*     cml_kv_cache_get_keys(CMLKVCache* cache);
Tensor*     cml_kv_cache_get_values(CMLKVCache* cache);
```

### Grouped Query Attention (GQA)

Supports fewer KV heads than query heads (e.g., LLaMA 2 70B uses 8 KV heads for 64 query heads). Configurable with causal masking and sliding window.

```c
typedef struct CMLGQAConfig {
    int num_heads;       // Query heads
    int num_kv_heads;    // KV heads (can be < num_heads)
    int head_dim;
    float scale;         // 1/sqrt(head_dim)
    bool causal;
    int window_size;     // 0 = unlimited
} CMLGQAConfig;

Tensor* cml_gqa_forward(Tensor* Q, Tensor* K, Tensor* V, const CMLGQAConfig* config, Tensor* mask);
Tensor* cml_gqa_forward_cached(Tensor* Q, Tensor* K, Tensor* V, CMLKVCache* kv_cache, const CMLGQAConfig* config);
```

### Flash Attention

O(N) memory attention using online softmax with tiled computation. Drop-in replacement for standard GQA.

```c
CMLFlashAttentionConfig cml_flash_attention_default_config(void);

Tensor* cml_gqa_flash_forward(Tensor* Q, Tensor* K, Tensor* V,
                               const CMLGQAConfig* config,
                               const CMLFlashAttentionConfig* flash_config);

Tensor* cml_gqa_flash_forward_cached(Tensor* Q, Tensor* K, Tensor* V,
                                      CMLKVCache* kv_cache,
                                      const CMLGQAConfig* config,
                                      const CMLFlashAttentionConfig* flash_config);
```

### RoPE (Rotary Position Embeddings)

Applies rotary position encoding to query/key tensors in-place.

```c
typedef struct CMLRoPEConfig {
    int dim;            // Embedding dimension
    int max_seq_len;
    float base;         // Base frequency (default: 10000.0)
} CMLRoPEConfig;

Tensor* cml_rope_forward(Tensor* x, int start_pos, const CMLRoPEConfig* config);
```

### Mixture of Experts (MoE)

Top-k expert routing with load-balancing capacity factor.

```c
typedef struct CMLMoEConfig {
    int num_experts;
    int top_k;              // Experts routed per token
    int input_dim;
    int hidden_dim;
    float capacity_factor;  // Load balancing
    bool normalize_weights; // Normalize gating weights to sum to 1
} CMLMoEConfig;

CMLMoELayer* cml_moe_create(const CMLMoEConfig* config);
void         cml_moe_free(CMLMoELayer* moe);
Tensor*      cml_moe_forward(CMLMoELayer* moe, Tensor* input);
Tensor*      cml_moe_get_routing(CMLMoELayer* moe, Tensor* input);  // Debug/analysis
```

### BPE Tokenizer

```c
CMLTokenizer* cml_tokenizer_create(char** vocab, int vocab_size, char** merge_pairs, int num_merges);
void          cml_tokenizer_free(CMLTokenizer* tok);
int*          cml_tokenizer_encode(CMLTokenizer* tok, const char* text, int* num_tokens);
char*         cml_tokenizer_decode(CMLTokenizer* tok, const int* tokens, int num_tokens);
void          cml_tokenizer_set_special(CMLTokenizer* tok, int bos, int eos, int pad, int unk);
```

---

## Paged Attention

**File:** `include/nn/paged_attention.h`

Implements a paged block allocator for KV caches, allowing multiple sequences to share a fixed memory pool without contiguous allocation. This eliminates memory fragmentation during serving and enables efficient memory utilization across concurrent requests.

Each page block holds `CML_PAGE_BLOCK_SIZE` (16) tokens. Blocks are allocated from a free list and mapped per-sequence via block tables.

### Key API

```c
CMLPagedKVCache* cml_paged_kv_cache_create(int max_blocks, int max_sequences,
                                            int num_kv_heads, int head_dim);
void cml_paged_kv_cache_free(CMLPagedKVCache* cache);

int  cml_paged_cache_alloc_block(CMLPagedKVCache* cache);
void cml_paged_cache_free_block(CMLPagedKVCache* cache, int block_id);

int  cml_paged_cache_init_sequence(CMLPagedKVCache* cache);
void cml_paged_cache_free_sequence(CMLPagedKVCache* cache, int seq_id);

int cml_paged_cache_append(CMLPagedKVCache* cache, int seq_id,
                            const float* key, const float* value);

Tensor* cml_paged_gqa_forward(CMLPagedKVCache* cache, int seq_id,
                               Tensor* Q, const CMLGQAConfig* config);
```

### Usage Example

```c
#include "nn/paged_attention.h"

CMLPagedKVCache* cache = cml_paged_kv_cache_create(
    /*max_blocks=*/1024, /*max_sequences=*/64,
    /*num_kv_heads=*/8, /*head_dim=*/128);

int seq = cml_paged_cache_init_sequence(cache);

for (int i = 0; i < seq_len; i++) {
    cml_paged_cache_append(cache, seq, key_data[i], value_data[i]);
}

Tensor* out = cml_paged_gqa_forward(cache, seq, Q, &gqa_config);

cml_paged_cache_free_sequence(cache, seq);
cml_paged_kv_cache_free(cache);
```

### Notes

- Max blocks: `CML_MAX_BLOCKS` (4096). Max sequences: `CML_MAX_SEQUENCES` (256).
- Block data layout: `[block_size, num_kv_heads, head_dim]`.
- Pairs naturally with the serving scheduler (see below).

---

## LLaMA Model

**File:** `include/nn/llama.h`

Complete LLaMA model implementation with GQA, SwiGLU FFN, RMSNorm, RoPE, and KV caching. Supports loading weights from GGUF files and text generation with configurable sampling.

### Pre-defined Configs

```c
CMLLLaMAConfig cml_llama_config_7b(void);   // 7B parameters
CMLLLaMAConfig cml_llama_config_13b(void);  // 13B parameters
CMLLLaMAConfig cml_llama_config_70b(void);  // 70B parameters
```

Key config fields: `vocab_size`, `hidden_size`, `intermediate_size`, `num_layers`, `num_heads`, `num_kv_heads`, `max_seq_len`, `rope_theta`, `rms_norm_eps`, `tensor_parallel_size`, `tensor_parallel_rank`.

### Key API

```c
CMLLLaMAModel* cml_llama_create(const CMLLLaMAConfig* config);
void           cml_llama_free(CMLLLaMAModel* model);

int cml_llama_load_gguf(CMLLLaMAModel* model, const char* filepath);

Tensor* cml_llama_forward(CMLLLaMAModel* model, const int* token_ids, int seq_len);

CMLGenerationConfig  cml_generation_default_config(void);
CMLGenerationResult* cml_llama_generate(CMLLLaMAModel* model, const char* prompt,
                                         const CMLGenerationConfig* config);
void cml_generation_result_free(CMLGenerationResult* result);

int  cml_llama_sample_token(Tensor* logits, const CMLGenerationConfig* config);
void cml_llama_reset(CMLLLaMAModel* model);
```

### Generation Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.8 | Sampling temperature |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `top_k` | 40 | Top-k sampling limit |
| `max_new_tokens` | 256 | Maximum tokens to generate |
| `do_sample` | true | Sampling vs greedy decoding |

### Usage Example

```c
#include "nn/llama.h"

CMLLLaMAConfig cfg = cml_llama_config_7b();
CMLLLaMAModel* model = cml_llama_create(&cfg);
cml_llama_load_gguf(model, "llama-7b.gguf");

CMLGenerationConfig gen_cfg = cml_generation_default_config();
gen_cfg.max_new_tokens = 128;
gen_cfg.temperature = 0.7f;

CMLGenerationResult* result = cml_llama_generate(model, "Hello, world!", &gen_cfg);
if (result) {
    printf("%s\n", result->text);
    printf("%.1f tokens/sec\n", result->tokens_per_second);
    cml_generation_result_free(result);
}

cml_llama_free(model);
```

### Notes

- Tensor parallelism is supported via `tensor_parallel_size` and `tensor_parallel_rank` in the config.
- The model owns a `CMLTokenizer*` for encode/decode; set it after creation or let `cml_llama_load_gguf` populate it.
- Call `cml_llama_reset` between unrelated sequences to clear KV caches.

---

## Model Serving

**File:** `include/nn/serving.h`

Scheduling infrastructure for serving multiple concurrent LLM inference requests with continuous batching. Handles request queuing, batch admission, and lifecycle tracking. This is the bookkeeping layer -- the actual forward pass is performed externally.

### Request Lifecycle

```
QUEUED -> PREFILL -> DECODING -> FINISHED
                              -> ERROR
```

### Key API

```c
CMLServingConfig   cml_serving_default_config(void);
CMLServingContext*  cml_serving_create(const CMLServingConfig* config);
void               cml_serving_free(CMLServingContext* ctx);
void               cml_serving_set_kv_cache(CMLServingContext* ctx, CMLPagedKVCache* cache);

int                cml_serving_submit(CMLServingContext* ctx, const int* prompt_tokens,
                                      int num_tokens, int max_new_tokens);
int                cml_serving_step(CMLServingContext* ctx);  // One scheduling iteration
CMLSequenceStatus  cml_serving_get_status(CMLServingContext* ctx, int request_id);
const int*         cml_serving_get_tokens(CMLServingContext* ctx, int request_id, int* out_count);
int                cml_serving_finish_request(CMLServingContext* ctx, int request_id);

CMLServingStats    cml_serving_get_stats(const CMLServingContext* ctx);
```

### Serving Config

| Parameter | Description |
|-----------|-------------|
| `max_batch_size` | Max concurrent sequences (up to 64) |
| `max_queue_size` | Max pending requests (up to 1024) |
| `max_seq_len` | Maximum sequence length |
| `max_new_tokens_default` | Default generation length |
| `temperature_default` | Default sampling temperature |
| `top_p_default` | Default nucleus sampling threshold |

### Usage Example

```c
#include "nn/serving.h"
#include "nn/paged_attention.h"

CMLServingConfig cfg = cml_serving_default_config();
cfg.max_batch_size = 32;
CMLServingContext* ctx = cml_serving_create(&cfg);

CMLPagedKVCache* kv = cml_paged_kv_cache_create(2048, 256, 8, 128);
cml_serving_set_kv_cache(ctx, kv);

int req1 = cml_serving_submit(ctx, prompt_tokens, num_tokens, /*max_new=*/128);

while (cml_serving_step(ctx) > 0) {
    // Run model forward pass on ctx->active_batch here
    // Call cml_serving_finish_request() when a sequence hits EOS
}

CMLServingStats stats = cml_serving_get_stats(ctx);
printf("Completed: %zu, Avg tok/s: %.1f\n",
       stats.completed_requests, stats.avg_tokens_per_second);

cml_serving_free(ctx);
cml_paged_kv_cache_free(kv);
```

### Notes

- The paged KV cache is **not owned** by the serving context; the caller manages its lifetime.
- `cml_serving_step` admits queued requests into the active batch up to `max_batch_size`.
- Stats track time-to-first-token, tokens/second, and total throughput.

---

## Speculative Decoding

**File:** `include/nn/speculative.h`

Speculative decoding accelerates LLM inference by using a small draft model to propose K tokens, verified in parallel by the target model. This preserves target model quality while reducing expensive forward passes.

For full documentation including algorithm details, configuration tuning, and performance analysis, see [docs/speculative_decoding.md](speculative_decoding.md).

### Quick Reference

```c
CMLSpeculativeConfig   cml_speculative_default_config(void);
CMLSpeculativeDecoder* cml_speculative_create(const CMLSpeculativeConfig* config, int vocab_size);
void                   cml_speculative_free(CMLSpeculativeDecoder* decoder);

void cml_speculative_set_draft_model(CMLSpeculativeDecoder* dec, void* ctx,
                                     CMLModelForwardFn forward_fn, CMLSampleTokenFn sample_fn);
void cml_speculative_set_target_model(CMLSpeculativeDecoder* dec, void* ctx,
                                      CMLModelForwardFn forward_fn, CMLSampleTokenFn sample_fn);

CMLSpeculativeResult* cml_speculative_decode_step(CMLSpeculativeDecoder* dec,
                                                   const int* prefix_tokens, int prefix_len);
void  cml_speculative_result_free(CMLSpeculativeResult* result);
float cml_speculative_acceptance_rate(const CMLSpeculativeDecoder* dec);
```

Default config: K=5 draft tokens, temperature=0.8, top_p=0.9, top_k=40, sampling enabled. Max draft tokens per step: 16 (`CML_SPEC_MAX_DRAFT_TOKENS`).
