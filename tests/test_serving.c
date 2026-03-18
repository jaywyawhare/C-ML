#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cml.h"
#include "nn/serving.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-50s ", #name); \
    fflush(stdout); \
    if (test_##name()) { \
        tests_passed++; \
        printf("[PASS]\n"); \
    } else { \
        printf("[FAIL]\n"); \
    } \
} while(0)


static int test_default_config(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    if (cfg.max_batch_size <= 0) return 0;
    if (cfg.max_queue_size <= 0) return 0;
    if (cfg.max_seq_len <= 0) return 0;
    if (cfg.max_new_tokens_default <= 0) return 0;
    if (cfg.temperature_default <= 0.0f) return 0;
    if (cfg.top_p_default <= 0.0f || cfg.top_p_default > 1.0f) return 0;
    return 1;
}


static int test_create_free(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;
    if (ctx->batch_size != 0) { cml_serving_free(ctx); return 0; }
    if (ctx->queue_count != 0) { cml_serving_free(ctx); return 0; }
    if (ctx->next_request_id < 1) { cml_serving_free(ctx); return 0; }
    cml_serving_free(ctx);
    return 1;
}

static int test_create_null_config(void) {
    CMLServingContext* ctx = cml_serving_create(NULL);
    if (ctx != NULL) { cml_serving_free(ctx); return 0; }
    return 1;
}

static int test_free_null(void) {
    /* Should not crash */
    cml_serving_free(NULL);
    return 1;
}


static int test_set_kv_cache(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    /* Use a dummy non-NULL pointer to simulate setting a cache */
    int dummy;
    cml_serving_set_kv_cache(ctx, (CMLPagedKVCache*)&dummy);
    if (ctx->kv_cache != (CMLPagedKVCache*)&dummy) {
        cml_serving_free(ctx);
        return 0;
    }

    /* Clear it */
    cml_serving_set_kv_cache(ctx, NULL);
    if (ctx->kv_cache != NULL) { cml_serving_free(ctx); return 0; }

    /* Should not crash on NULL ctx */
    cml_serving_set_kv_cache(NULL, NULL);

    cml_serving_free(ctx);
    return 1;
}


static int test_submit_single(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    int tokens[] = {1, 2, 3, 4};
    int id = cml_serving_submit(ctx, tokens, 4, 100);
    if (id < 0) { cml_serving_free(ctx); return 0; }
    if (ctx->queue_count != 1) { cml_serving_free(ctx); return 0; }

    CMLSequenceStatus st = cml_serving_get_status(ctx, id);
    if (st != CML_SEQ_STATUS_QUEUED) { cml_serving_free(ctx); return 0; }

    cml_serving_free(ctx);
    return 1;
}

static int test_submit_multiple(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    int tokens[] = {10, 20, 30};
    int id1 = cml_serving_submit(ctx, tokens, 3, 50);
    int id2 = cml_serving_submit(ctx, tokens, 3, 50);
    int id3 = cml_serving_submit(ctx, tokens, 3, 50);

    if (id1 < 0 || id2 < 0 || id3 < 0) { cml_serving_free(ctx); return 0; }
    if (id1 == id2 || id2 == id3) { cml_serving_free(ctx); return 0; }
    if (ctx->queue_count != 3) { cml_serving_free(ctx); return 0; }
    if (ctx->stats.total_requests != 3) { cml_serving_free(ctx); return 0; }

    cml_serving_free(ctx);
    return 1;
}

static int test_submit_invalid_args(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    /* NULL tokens */
    if (cml_serving_submit(ctx, NULL, 4, 100) != -1) { cml_serving_free(ctx); return 0; }

    /* Zero length */
    int tokens[] = {1};
    if (cml_serving_submit(ctx, tokens, 0, 100) != -1) { cml_serving_free(ctx); return 0; }

    /* NULL context */
    if (cml_serving_submit(NULL, tokens, 1, 100) != -1) { cml_serving_free(ctx); return 0; }

    /* Queue should still be empty */
    if (ctx->queue_count != 0) { cml_serving_free(ctx); return 0; }

    cml_serving_free(ctx);
    return 1;
}

static int test_submit_default_max_new_tokens(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    cfg.max_new_tokens_default = 128;
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    int tokens[] = {1, 2};
    int id = cml_serving_submit(ctx, tokens, 2, 0);  /* 0 = use default */
    if (id < 0) { cml_serving_free(ctx); return 0; }

    /* The request should have picked up the config default.
       We can't directly check the request struct from outside,
       but we verify it was accepted successfully. */
    if (ctx->queue_count != 1) { cml_serving_free(ctx); return 0; }

    cml_serving_free(ctx);
    return 1;
}


static int test_step_admits_to_batch(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    cfg.max_batch_size = 4;
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    int tokens[] = {1, 2, 3};
    int id1 = cml_serving_submit(ctx, tokens, 3, 50);
    int id2 = cml_serving_submit(ctx, tokens, 3, 50);
    if (id1 < 0 || id2 < 0) { cml_serving_free(ctx); return 0; }

    /* Before step: both queued, batch empty */
    if (ctx->batch_size != 0) { cml_serving_free(ctx); return 0; }
    if (ctx->queue_count != 2) { cml_serving_free(ctx); return 0; }

    /* Step */
    int active = cml_serving_step(ctx);
    if (active != 2) { cml_serving_free(ctx); return 0; }
    if (ctx->batch_size != 2) { cml_serving_free(ctx); return 0; }
    if (ctx->queue_count != 0) { cml_serving_free(ctx); return 0; }

    cml_serving_free(ctx);
    return 1;
}

static int test_step_respects_batch_limit(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    cfg.max_batch_size = 2;
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    int tokens[] = {1};
    cml_serving_submit(ctx, tokens, 1, 10);
    cml_serving_submit(ctx, tokens, 1, 10);
    cml_serving_submit(ctx, tokens, 1, 10);  /* This one stays queued */

    int active = cml_serving_step(ctx);
    if (active != 2) { cml_serving_free(ctx); return 0; }
    if (ctx->queue_count != 1) { cml_serving_free(ctx); return 0; }

    cml_serving_free(ctx);
    return 1;
}

static int test_step_null_context(void) {
    /* Should not crash, should return 0 */
    int active = cml_serving_step(NULL);
    if (active != 0) return 0;
    return 1;
}


static int test_status_transitions(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    cfg.max_batch_size = 4;
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    int tokens[] = {5, 10, 15};
    int id = cml_serving_submit(ctx, tokens, 3, 50);
    if (id < 0) { cml_serving_free(ctx); return 0; }

    /* Initially queued */
    CMLSequenceStatus st = cml_serving_get_status(ctx, id);
    if (st != CML_SEQ_STATUS_QUEUED) { cml_serving_free(ctx); return 0; }

    /* After step: should be DECODING (PREFILL is transient within step) */
    cml_serving_step(ctx);
    st = cml_serving_get_status(ctx, id);
    if (st != CML_SEQ_STATUS_DECODING) { cml_serving_free(ctx); return 0; }

    /* After finish: should be ERROR (not found) since request is freed */
    cml_serving_finish_request(ctx, id);
    st = cml_serving_get_status(ctx, id);
    if (st != CML_SEQ_STATUS_ERROR) { cml_serving_free(ctx); return 0; }

    cml_serving_free(ctx);
    return 1;
}

static int test_get_status_not_found(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    CMLSequenceStatus st = cml_serving_get_status(ctx, 9999);
    if (st != CML_SEQ_STATUS_ERROR) { cml_serving_free(ctx); return 0; }

    cml_serving_free(ctx);
    return 1;
}


static int test_get_tokens_empty(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    int tokens[] = {1, 2};
    int id = cml_serving_submit(ctx, tokens, 2, 50);
    cml_serving_step(ctx);

    int count = -1;
    const int* gen = cml_serving_get_tokens(ctx, id, &count);
    if (!gen) { cml_serving_free(ctx); return 0; }
    if (count != 0) { cml_serving_free(ctx); return 0; }

    cml_serving_free(ctx);
    return 1;
}

static int test_get_tokens_not_found(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    int count = -1;
    const int* gen = cml_serving_get_tokens(ctx, 9999, &count);
    if (gen != NULL) { cml_serving_free(ctx); return 0; }
    if (count != 0) { cml_serving_free(ctx); return 0; }

    cml_serving_free(ctx);
    return 1;
}


static int test_finish_request(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    cfg.max_batch_size = 4;
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    int tokens[] = {1, 2, 3};
    int id = cml_serving_submit(ctx, tokens, 3, 50);
    if (id < 0) { cml_serving_free(ctx); return 0; }

    /* Admit to batch */
    cml_serving_step(ctx);
    if (ctx->batch_size != 1) { cml_serving_free(ctx); return 0; }

    /* Finish */
    int rc = cml_serving_finish_request(ctx, id);
    if (rc != 0) { cml_serving_free(ctx); return 0; }
    if (ctx->batch_size != 0) { cml_serving_free(ctx); return 0; }

    /* Stats */
    CMLServingStats stats = cml_serving_get_stats(ctx);
    if (stats.total_requests != 1) { cml_serving_free(ctx); return 0; }
    if (stats.completed_requests != 1) { cml_serving_free(ctx); return 0; }

    cml_serving_free(ctx);
    return 1;
}

static int test_finish_request_not_found(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    int rc = cml_serving_finish_request(ctx, 9999);
    if (rc != -1) { cml_serving_free(ctx); return 0; }

    cml_serving_free(ctx);
    return 1;
}

static int test_finish_request_null_context(void) {
    int rc = cml_serving_finish_request(NULL, 1);
    if (rc != -1) return 0;
    return 1;
}

static int test_finish_multiple_requests(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    cfg.max_batch_size = 4;
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    int tokens[] = {1, 2};
    int id1 = cml_serving_submit(ctx, tokens, 2, 20);
    int id2 = cml_serving_submit(ctx, tokens, 2, 20);
    int id3 = cml_serving_submit(ctx, tokens, 2, 20);

    cml_serving_step(ctx);
    if (ctx->batch_size != 3) { cml_serving_free(ctx); return 0; }

    /* Finish in non-sequential order */
    if (cml_serving_finish_request(ctx, id2) != 0) { cml_serving_free(ctx); return 0; }
    if (ctx->batch_size != 2) { cml_serving_free(ctx); return 0; }

    if (cml_serving_finish_request(ctx, id1) != 0) { cml_serving_free(ctx); return 0; }
    if (ctx->batch_size != 1) { cml_serving_free(ctx); return 0; }

    if (cml_serving_finish_request(ctx, id3) != 0) { cml_serving_free(ctx); return 0; }
    if (ctx->batch_size != 0) { cml_serving_free(ctx); return 0; }

    CMLServingStats stats = cml_serving_get_stats(ctx);
    if (stats.completed_requests != 3) { cml_serving_free(ctx); return 0; }

    cml_serving_free(ctx);
    return 1;
}


static int test_queue_overflow(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    cfg.max_queue_size = 4;
    cfg.max_batch_size = 2;
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    int tokens[] = {1};
    int submitted = 0;
    int last_id = -1;

    /* Fill the queue */
    for (int i = 0; i < 4; i++) {
        int id = cml_serving_submit(ctx, tokens, 1, 10);
        if (id >= 0) {
            submitted++;
            last_id = id;
        }
    }
    if (submitted != 4) { cml_serving_free(ctx); return 0; }
    if (ctx->queue_count != 4) { cml_serving_free(ctx); return 0; }

    /* This should fail: queue is full */
    int overflow_id = cml_serving_submit(ctx, tokens, 1, 10);
    if (overflow_id != -1) { cml_serving_free(ctx); return 0; }

    /* Queue count should not have changed */
    if (ctx->queue_count != 4) { cml_serving_free(ctx); return 0; }

    /* Admit 2 into batch, freeing 2 queue slots */
    cml_serving_step(ctx);
    if (ctx->batch_size != 2) { cml_serving_free(ctx); return 0; }
    if (ctx->queue_count != 2) { cml_serving_free(ctx); return 0; }

    /* Now we can submit again */
    int new_id = cml_serving_submit(ctx, tokens, 1, 10);
    if (new_id < 0) { cml_serving_free(ctx); return 0; }
    if (ctx->queue_count != 3) { cml_serving_free(ctx); return 0; }

    /* Verify last_id is still valid */
    (void)last_id;

    cml_serving_free(ctx);
    return 1;
}


static int test_get_stats(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    CMLServingStats stats = cml_serving_get_stats(ctx);
    if (stats.total_requests != 0) { cml_serving_free(ctx); return 0; }
    if (stats.completed_requests != 0) { cml_serving_free(ctx); return 0; }
    if (stats.active_sequences != 0) { cml_serving_free(ctx); return 0; }

    cml_serving_free(ctx);
    return 1;
}

static int test_get_stats_null(void) {
    CMLServingStats stats = cml_serving_get_stats(NULL);
    if (stats.total_requests != 0) return 0;
    if (stats.completed_requests != 0) return 0;
    return 1;
}


static int test_step_finish_step(void) {
    CMLServingConfig cfg = cml_serving_default_config();
    cfg.max_batch_size = 2;
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    int tokens[] = {1, 2};

    /* Submit 4 requests; batch can hold 2 */
    int id1 = cml_serving_submit(ctx, tokens, 2, 10);
    int id2 = cml_serving_submit(ctx, tokens, 2, 10);
    int id3 = cml_serving_submit(ctx, tokens, 2, 10);
    int id4 = cml_serving_submit(ctx, tokens, 2, 10);
    if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0) {
        cml_serving_free(ctx);
        return 0;
    }

    /* Step 1: admit id1, id2 */
    cml_serving_step(ctx);
    if (ctx->batch_size != 2) { cml_serving_free(ctx); return 0; }
    if (ctx->queue_count != 2) { cml_serving_free(ctx); return 0; }

    /* Finish id1: frees a batch slot */
    cml_serving_finish_request(ctx, id1);
    if (ctx->batch_size != 1) { cml_serving_free(ctx); return 0; }

    /* Step 2: should admit id3 from queue */
    cml_serving_step(ctx);
    if (ctx->batch_size != 2) { cml_serving_free(ctx); return 0; }
    if (ctx->queue_count != 1) { cml_serving_free(ctx); return 0; }

    /* Finish remaining */
    cml_serving_finish_request(ctx, id2);
    cml_serving_finish_request(ctx, id3);
    cml_serving_step(ctx); /* Admit id4 */
    cml_serving_finish_request(ctx, id4);

    CMLServingStats stats = cml_serving_get_stats(ctx);
    if (stats.total_requests != 4) { cml_serving_free(ctx); return 0; }
    if (stats.completed_requests != 4) { cml_serving_free(ctx); return 0; }

    cml_serving_free(ctx);
    return 1;
}


static int test_config_clamping(void) {
    CMLServingConfig cfg = cml_serving_default_config();

    /* Exceed hard limits */
    cfg.max_batch_size = CML_SERVING_MAX_BATCH + 100;
    cfg.max_queue_size = CML_SERVING_MAX_QUEUE + 100;
    CMLServingContext* ctx = cml_serving_create(&cfg);
    if (!ctx) return 0;

    if (ctx->config.max_batch_size > CML_SERVING_MAX_BATCH) {
        cml_serving_free(ctx);
        return 0;
    }
    if (ctx->config.max_queue_size > CML_SERVING_MAX_QUEUE) {
        cml_serving_free(ctx);
        return 0;
    }

    cml_serving_free(ctx);
    return 1;
}


int main(void) {
    printf("test_serving\n\n");

    /* Config */
    TEST(default_config);
    TEST(config_clamping);

    /* Create / free */
    TEST(create_free);
    TEST(create_null_config);
    TEST(free_null);

    /* KV cache */
    TEST(set_kv_cache);

    /* Submit */
    TEST(submit_single);
    TEST(submit_multiple);
    TEST(submit_invalid_args);
    TEST(submit_default_max_new_tokens);

    /* Step / batch admission */
    TEST(step_admits_to_batch);
    TEST(step_respects_batch_limit);
    TEST(step_null_context);

    /* Status transitions */
    TEST(status_transitions);
    TEST(get_status_not_found);

    /* Get tokens */
    TEST(get_tokens_empty);
    TEST(get_tokens_not_found);

    /* Finish / stats */
    TEST(finish_request);
    TEST(finish_request_not_found);
    TEST(finish_request_null_context);
    TEST(finish_multiple_requests);

    /* Queue overflow */
    TEST(queue_overflow);

    /* Stats */
    TEST(get_stats);
    TEST(get_stats_null);

    /* Multi-step lifecycle */
    TEST(step_finish_step);

    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
