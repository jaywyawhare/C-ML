/*
 * Continuous batching / serving scheduler for LLM inference.
 * Handles request queuing, batch admission, lifecycle tracking, and — when a
 * model forward callback is attached via cml_serving_set_model — the actual
 * autoregressive token generation (forward -> sample -> append -> stop on EOS
 * or max_new_tokens). The model callback owns the forward pass and KV cache;
 * the scheduler owns batching, sampling, and bookkeeping.
 */

#ifndef CML_NN_SERVING_H
#define CML_NN_SERVING_H

#include "tensor/tensor.h"
#include "nn/llm_ops.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLPagedKVCache CMLPagedKVCache;

#define CML_SERVING_MAX_BATCH 64
#define CML_SERVING_MAX_QUEUE 1024

typedef enum {
    CML_SEQ_STATUS_QUEUED = 0,
    CML_SEQ_STATUS_PREFILL,
    CML_SEQ_STATUS_DECODING,
    CML_SEQ_STATUS_FINISHED,
    CML_SEQ_STATUS_ERROR,
} CMLSequenceStatus;

/*
 * Model forward callback. Given input tokens at a sequence position, writes
 * `vocab_size` next-token logits into logits_out. For the prefill of a request
 * it receives the whole prompt at position 0; for each decode step it receives
 * the single previous token at its position. The callback owns the KV cache.
 * Returns 0 on success, non-zero on failure.
 */
typedef int (*CMLServingForwardFn)(void* model, const int* tokens, int num_tokens, int position,
                                   float* logits_out, int vocab_size);

typedef struct CMLSequenceRequest {
    int request_id;
    int* prompt_tokens;
    int num_prompt_tokens;
    int max_new_tokens;
    float temperature;
    float top_p;

    /* State */
    CMLSequenceStatus status;
    int paged_seq_id; /* ID in paged KV cache */
    int* generated_tokens;
    int num_generated;
    int gen_capacity;
    int current_pos; /* Position in generation */

    /* Timing */
    double submit_time_ms;
    double first_token_time_ms;
    double finish_time_ms;
} CMLSequenceRequest;

typedef struct CMLServingStats {
    size_t total_requests;
    size_t completed_requests;
    size_t active_sequences;
    size_t total_tokens_generated;
    double avg_time_to_first_token_ms;
    double avg_tokens_per_second;
    double total_time_ms;
} CMLServingStats;

typedef struct CMLServingConfig {
    int max_batch_size;
    int max_queue_size;
    int max_seq_len;
    int max_new_tokens_default;
    float temperature_default;
    float top_p_default;
} CMLServingConfig;

typedef struct CMLServingContext {
    CMLServingConfig config;

    /* Request queue */
    CMLSequenceRequest** queue;
    int queue_head;
    int queue_tail;
    int queue_count;
    int queue_capacity;

    /* Active batch */
    CMLSequenceRequest** active_batch;
    int batch_size;

    /* Paged KV cache (not owned, set externally) */
    CMLPagedKVCache* kv_cache;

    /* Stats */
    CMLServingStats stats;
    int next_request_id;

    /* Model forward hook (set via cml_serving_set_model). When non-NULL, the
     * scheduler drives autoregressive generation; when NULL it only schedules. */
    CMLServingForwardFn forward_fn;
    void* model;
    int vocab_size;
    int eos_token_id; /* generation stops when this token is produced (<0 = none) */
} CMLServingContext;

CMLServingConfig cml_serving_default_config(void);

CMLServingContext* cml_serving_create(const CMLServingConfig* config);

void cml_serving_free(CMLServingContext* ctx);

/* KV cache is not owned; caller must keep it alive */
void cml_serving_set_kv_cache(CMLServingContext* ctx, CMLPagedKVCache* cache);

/*
 * Attach a model forward callback so cml_serving_step actually generates tokens.
 * `vocab_size` is the logits width the callback produces; `eos_token_id` stops a
 * sequence early when produced (pass a negative value to disable).
 */
void cml_serving_set_model(CMLServingContext* ctx, CMLServingForwardFn forward_fn, void* model,
                           int vocab_size, int eos_token_id);

/* Returns request_id on success, -1 on failure (e.g. queue full).
 * prompt_tokens is copied internally. max_new_tokens 0 = use config default. */
int cml_serving_submit(CMLServingContext* ctx, const int* prompt_tokens, int num_tokens,
                       int max_new_tokens);

/* Run one scheduling iteration. Returns number of active sequences. */
int cml_serving_step(CMLServingContext* ctx);

CMLSequenceStatus cml_serving_get_status(CMLServingContext* ctx, int request_id);

/* Returns pointer to internal token array (do not free), or NULL if not found. */
const int* cml_serving_get_tokens(CMLServingContext* ctx, int request_id, int* out_count);

int cml_serving_finish_request(CMLServingContext* ctx, int request_id);

CMLServingStats cml_serving_get_stats(const CMLServingContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_SERVING_H */
