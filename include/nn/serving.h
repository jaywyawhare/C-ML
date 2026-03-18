/*
 * Continuous batching / serving scheduler for LLM inference.
 * Handles request queuing, batch admission, and lifecycle tracking.
 * The actual forward pass / token generation is NOT done here; this is
 * purely the scheduling and bookkeeping layer.
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

typedef struct CMLSequenceRequest {
    int request_id;
    int* prompt_tokens;
    int num_prompt_tokens;
    int max_new_tokens;
    float temperature;
    float top_p;

    /* State */
    CMLSequenceStatus status;
    int paged_seq_id;        /* ID in paged KV cache */
    int* generated_tokens;
    int num_generated;
    int gen_capacity;
    int current_pos;         /* Position in generation */

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
} CMLServingContext;

CMLServingConfig cml_serving_default_config(void);

CMLServingContext* cml_serving_create(const CMLServingConfig* config);

void cml_serving_free(CMLServingContext* ctx);

/* KV cache is not owned; caller must keep it alive */
void cml_serving_set_kv_cache(CMLServingContext* ctx, CMLPagedKVCache* cache);

/* Returns request_id on success, -1 on failure (e.g. queue full).
 * prompt_tokens is copied internally. max_new_tokens 0 = use config default. */
int cml_serving_submit(CMLServingContext* ctx, const int* prompt_tokens,
                       int num_tokens, int max_new_tokens);

/* Run one scheduling iteration. Returns number of active sequences. */
int cml_serving_step(CMLServingContext* ctx);

CMLSequenceStatus cml_serving_get_status(CMLServingContext* ctx, int request_id);

/* Returns pointer to internal token array (do not free), or NULL if not found. */
const int* cml_serving_get_tokens(CMLServingContext* ctx, int request_id,
                                  int* out_count);

int cml_serving_finish_request(CMLServingContext* ctx, int request_id);

CMLServingStats cml_serving_get_stats(const CMLServingContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_SERVING_H */
