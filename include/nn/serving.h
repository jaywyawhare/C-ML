/**
 * @file serving.h
 * @brief Continuous batching / serving scheduler for LLM inference
 *
 * Provides the scheduling infrastructure for serving multiple concurrent
 * LLM inference requests with continuous batching. This module handles:
 * - Request queuing (circular buffer)
 * - Batch admission (moving queued requests into the active batch)
 * - Request lifecycle tracking (QUEUED -> PREFILL -> DECODING -> FINISHED)
 * - Serving statistics
 *
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

/* Forward declaration: paged KV cache (defined in nn/paged_attention.h) */
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

/** Return a default serving configuration with sensible defaults */
CMLServingConfig cml_serving_default_config(void);

/** Create a serving context from config. Returns NULL on failure. */
CMLServingContext* cml_serving_create(const CMLServingConfig* config);

/** Free serving context and all pending/active requests */
void cml_serving_free(CMLServingContext* ctx);

/** Set the paged KV cache (not owned; caller must keep it alive) */
void cml_serving_set_kv_cache(CMLServingContext* ctx, CMLPagedKVCache* cache);

/**
 * Submit a new inference request.
 * @param prompt_tokens  Array of token IDs (copied internally)
 * @param num_tokens     Length of prompt_tokens
 * @param max_new_tokens Maximum tokens to generate (0 = use config default)
 * @return request_id on success, -1 on failure (e.g. queue full)
 */
int cml_serving_submit(CMLServingContext* ctx, const int* prompt_tokens,
                       int num_tokens, int max_new_tokens);

/**
 * Run one scheduling iteration:
 * - Admit queued requests into the active batch (up to max_batch_size)
 * - Update stats
 * @return number of active sequences after this step
 */
int cml_serving_step(CMLServingContext* ctx);

/** Get the status of a request by ID. Returns CML_SEQ_STATUS_ERROR if not found. */
CMLSequenceStatus cml_serving_get_status(CMLServingContext* ctx, int request_id);

/**
 * Get the generated tokens for a request so far.
 * @param out_count Set to the number of generated tokens
 * @return Pointer to internal token array (do not free), or NULL if not found
 */
const int* cml_serving_get_tokens(CMLServingContext* ctx, int request_id,
                                  int* out_count);

/**
 * Mark a request as finished, remove from active batch, update stats.
 * @return 0 on success, -1 if not found
 */
int cml_serving_finish_request(CMLServingContext* ctx, int request_id);

/** Return a snapshot of the current serving stats */
CMLServingStats cml_serving_get_stats(const CMLServingContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_SERVING_H */
