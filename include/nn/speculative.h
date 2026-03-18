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

/* Returns logits [seq_len, vocab_size]. Caller must free the returned tensor. */
typedef Tensor* (*CMLModelForwardFn)(void* model_ctx, const int* token_ids, int seq_len);

typedef int (*CMLSampleTokenFn)(void* model_ctx, Tensor* logits, float temperature);

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

CMLSpeculativeConfig cml_speculative_default_config(void);

CMLSpeculativeDecoder* cml_speculative_create(const CMLSpeculativeConfig* config, int vocab_size);

void cml_speculative_free(CMLSpeculativeDecoder* decoder);

void cml_speculative_set_draft_model(CMLSpeculativeDecoder* dec, void* ctx,
                                     CMLModelForwardFn forward_fn,
                                     CMLSampleTokenFn sample_fn);

void cml_speculative_set_target_model(CMLSpeculativeDecoder* dec, void* ctx,
                                      CMLModelForwardFn forward_fn,
                                      CMLSampleTokenFn sample_fn);

/* Drafts K tokens, verifies with target model, accepts matching tokens.
 * Returns heap-allocated result (free with cml_speculative_result_free). */
CMLSpeculativeResult* cml_speculative_decode_step(CMLSpeculativeDecoder* dec,
                                                   const int* prefix_tokens,
                                                   int prefix_len);

void cml_speculative_result_free(CMLSpeculativeResult* result);

float cml_speculative_acceptance_rate(const CMLSpeculativeDecoder* dec);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_SPECULATIVE_H */
