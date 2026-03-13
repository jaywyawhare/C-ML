/**
 * @file speculative.h
 * @brief Speculative decoding for LLM inference acceleration
 *
 * Implements speculative decoding (Leviathan et al., 2023) where a small
 * draft model proposes K tokens that are verified in parallel by a larger
 * target model.  Accepted tokens are produced at the target model's quality
 * while amortising the cost of autoregressive steps.
 */

#ifndef CML_NN_SPECULATIVE_H
#define CML_NN_SPECULATIVE_H

#include "tensor/tensor.h"
#include "nn/llama.h"
#include "nn/llm_ops.h"
#include "core/logging.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CML_SPEC_MAX_DRAFT_TOKENS 16

/* ===== Configuration ===== */

typedef struct CMLSpeculativeConfig {
    int num_draft_tokens;       /* K: number of draft tokens per step (default: 5) */
    float temperature;          /* Sampling temperature */
    float top_p;                /* Nucleus sampling */
    int top_k;                  /* Top-k sampling */
    bool do_sample;             /* Use sampling vs greedy */
} CMLSpeculativeConfig;

/* ===== Result ===== */

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

/* ===== Model callbacks ===== */

/**
 * Callback type for model forward pass.
 * Given token_ids and seq_len, returns logits tensor [seq_len, vocab_size].
 * The caller is responsible for freeing the returned tensor.
 */
typedef Tensor* (*CMLModelForwardFn)(void* model_ctx, const int* token_ids, int seq_len);

/** Callback type for sampling a token from logits. */
typedef int (*CMLSampleTokenFn)(void* model_ctx, Tensor* logits, float temperature);

/* ===== Decoder ===== */

typedef struct CMLSpeculativeDecoder {
    CMLSpeculativeConfig config;

    /* Draft model callbacks */
    void* draft_model_ctx;
    CMLModelForwardFn draft_forward;
    CMLSampleTokenFn draft_sample;

    /* Target model callbacks */
    void* target_model_ctx;
    CMLModelForwardFn target_forward;
    CMLSampleTokenFn target_sample;

    int vocab_size;

    /* Statistics */
    size_t total_drafted;
    size_t total_accepted;
    size_t total_steps;
} CMLSpeculativeDecoder;

/* ===== API ===== */

/** Return a sensible default config: K=5, temp=0.8, top_p=0.9, top_k=40, do_sample=true. */
CMLSpeculativeConfig cml_speculative_default_config(void);

/** Create a speculative decoder.  Caller must free with cml_speculative_free(). */
CMLSpeculativeDecoder* cml_speculative_create(const CMLSpeculativeConfig* config, int vocab_size);

/** Free a speculative decoder. */
void cml_speculative_free(CMLSpeculativeDecoder* decoder);

/** Set the draft model callbacks. */
void cml_speculative_set_draft_model(CMLSpeculativeDecoder* dec, void* ctx,
                                     CMLModelForwardFn forward_fn,
                                     CMLSampleTokenFn sample_fn);

/** Set the target model callbacks. */
void cml_speculative_set_target_model(CMLSpeculativeDecoder* dec, void* ctx,
                                      CMLModelForwardFn forward_fn,
                                      CMLSampleTokenFn sample_fn);

/**
 * Perform one speculative decode step.
 *
 * 1. Draft K tokens with the draft model (autoregressive).
 * 2. Verify all K tokens with a single target-model forward pass.
 * 3. Accept matching tokens; on first mismatch use the target model's
 *    sample as the correction token.
 * 4. If all K accepted, sample one bonus token from the target model.
 *
 * @param dec       The speculative decoder.
 * @param prefix_tokens  Token IDs preceding this step.
 * @param prefix_len     Length of prefix_tokens.
 * @return A heap-allocated result (free with cml_speculative_result_free).
 */
CMLSpeculativeResult* cml_speculative_decode_step(CMLSpeculativeDecoder* dec,
                                                   const int* prefix_tokens,
                                                   int prefix_len);

/** Free a speculative result. */
void cml_speculative_result_free(CMLSpeculativeResult* result);

/** Overall acceptance rate across all steps so far. */
float cml_speculative_acceptance_rate(const CMLSpeculativeDecoder* dec);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_SPECULATIVE_H */
