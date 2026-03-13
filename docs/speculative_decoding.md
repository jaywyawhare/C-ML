# Speculative Decoding

Speculative decoding is an LLM inference acceleration technique (based on Leviathan et al., 2023) that uses a small "draft" model to propose multiple tokens, which are then verified in parallel by a larger "target" model in a single forward pass. This amortizes the cost of autoregressive generation while preserving the target model's output quality.

______________________________________________________________________

## Table of Contents

1. [Overview](#overview)
1. [How It Works](#how-it-works)
1. [API Reference](#api-reference)
1. [Configuration](#configuration)
1. [Usage Example](#usage-example)
1. [Performance Metrics](#performance-metrics)

______________________________________________________________________

## Overview

Traditional autoregressive decoding generates one token per forward pass of a large model, making it slow. Speculative decoding speeds this up by:

1. Using a small **draft model** to quickly generate K candidate tokens
2. Verifying all K tokens in a **single forward pass** of the large target model
3. Accepting matching tokens and correcting at the first disagreement

This produces the same quality as the target model while reducing the number of expensive forward passes.

**Files:** `include/nn/speculative.h`, `src/nn/speculative.c`

______________________________________________________________________

## How It Works

Each speculative decode step follows three phases:

```
Draft Phase                 Verify Phase               Accept/Reject
┌─────────────┐            ┌──────────────┐           ┌──────────────┐
│ Draft model  │            │ Target model  │           │ Compare      │
│ generates K  │──────────▶│ verifies all  │─────────▶│ draft vs     │
│ tokens       │            │ K tokens in   │           │ target       │
│ sequentially │            │ one pass      │           │ predictions  │
└─────────────┘            └──────────────┘           └──────────────┘
```

**Phase 1 — Draft:** Autoregressively generate K tokens using the draft model. Each step feeds the growing sequence back to get the next draft token.

**Phase 2 — Verify:** Concatenate the prefix with all K draft tokens and run a single forward pass of the target model to produce logits for every position.

**Phase 3 — Accept/Reject:**

- Compare each draft token against the target model's argmax prediction at the corresponding position
- Accept tokens sequentially until the first mismatch
- On mismatch, use the target model's prediction as a correction token
- If all K tokens are accepted, sample one bonus token from the target's final logits

______________________________________________________________________

## API Reference

### Structs

```c
typedef struct CMLSpeculativeConfig {
    int num_draft_tokens;       /* K: number of draft tokens per step (default: 5) */
    float temperature;          /* Sampling temperature */
    float top_p;                /* Nucleus sampling */
    int top_k;                  /* Top-k sampling */
    bool do_sample;             /* Use sampling vs greedy */
} CMLSpeculativeConfig;

typedef struct CMLSpeculativeResult {
    int* accepted_tokens;       /* Final accepted token sequence */
    int num_accepted;           /* Number of accepted tokens */
    int num_drafted;            /* Total tokens drafted */
    int num_verified;           /* Total verification passes */
    float acceptance_rate;      /* accepted / drafted */
    double draft_time_ms;
    double verify_time_ms;
    double total_time_ms;
} CMLSpeculativeResult;
```

### Callback Types

```c
/* Forward pass: given token IDs and sequence length, returns logits [seq_len, vocab_size] */
typedef Tensor* (*CMLModelForwardFn)(void* model_ctx, const int* token_ids, int seq_len);

/* Sampling: given logits and temperature, returns a token ID */
typedef int (*CMLSampleTokenFn)(void* model_ctx, Tensor* logits, float temperature);
```

### Functions

| Function | Description |
|----------|-------------|
| `cml_speculative_default_config()` | Returns default config (K=5, temp=0.8, top_p=0.9, top_k=40) |
| `cml_speculative_create(config, vocab_size)` | Create a speculative decoder |
| `cml_speculative_free(decoder)` | Free a speculative decoder |
| `cml_speculative_set_draft_model(dec, ctx, fwd, sample)` | Set draft model callbacks |
| `cml_speculative_set_target_model(dec, ctx, fwd, sample)` | Set target model callbacks |
| `cml_speculative_decode_step(dec, prefix, prefix_len)` | Run one speculative decode step |
| `cml_speculative_result_free(result)` | Free a decode result |
| `cml_speculative_acceptance_rate(decoder)` | Get overall acceptance rate across all steps |

______________________________________________________________________

## Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `num_draft_tokens` | 5 | 1–16 | Number of tokens to draft per step (K) |
| `temperature` | 0.8 | > 0 | Sampling temperature (lower = more deterministic) |
| `top_p` | 0.9 | (0, 1] | Nucleus sampling probability |
| `top_k` | 40 | >= 1 | Top-k sampling limit |
| `do_sample` | true | — | Sampling (true) vs greedy (false) |

Maximum draft tokens per step is capped at `CML_SPEC_MAX_DRAFT_TOKENS` (16).

______________________________________________________________________

## Usage Example

```c
#include "nn/speculative.h"

/* Create decoder with default config */
CMLSpeculativeConfig cfg = cml_speculative_default_config();
cfg.num_draft_tokens = 5;

CMLSpeculativeDecoder* dec = cml_speculative_create(&cfg, 32000);

/* Register model callbacks */
cml_speculative_set_draft_model(dec, &draft_ctx, draft_forward, draft_sample);
cml_speculative_set_target_model(dec, &target_ctx, target_forward, target_sample);

/* Run decode steps */
int prefix[] = {1, 2, 3};
CMLSpeculativeResult* res = cml_speculative_decode_step(dec, prefix, 3);

if (res) {
    printf("Accepted: %d tokens (rate: %.0f%%)\n",
           res->num_accepted, res->acceptance_rate * 100);
    printf("Draft: %.2f ms, Verify: %.2f ms\n",
           res->draft_time_ms, res->verify_time_ms);
    cml_speculative_result_free(res);
}

/* Lifetime stats */
printf("Overall acceptance: %.0f%%\n",
       cml_speculative_acceptance_rate(dec) * 100);

cml_speculative_free(dec);
```

______________________________________________________________________

## Performance Metrics

Each `CMLSpeculativeResult` includes detailed timing:

- **`draft_time_ms`** — Time spent in draft model forward passes
- **`verify_time_ms`** — Time spent in target model verification
- **`total_time_ms`** — Wall-clock time for the full step
- **`acceptance_rate`** — Fraction of draft tokens accepted (0.0–1.0)

The decoder also tracks lifetime statistics via `cml_speculative_acceptance_rate()`, which returns the cumulative acceptance rate across all decode steps. A higher acceptance rate means the draft model closely matches the target, yielding greater speedups.

| Scenario | Accepted | Output | Rate |
|----------|----------|--------|------|
| Full agreement (K=5) | 5 draft + 1 bonus | 6 tokens | 100% |
| Partial match (3 of 5) | 3 draft + 1 correction | 4 tokens | 60% |
| Immediate mismatch | 0 draft + 1 correction | 1 token | 0% |
