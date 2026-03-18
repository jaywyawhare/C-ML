#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ops/ir/beam_search.h"

static void test_create_free(void) {
    printf("  test_create_free...");

    CMLBeamSearchCtx* ctx = cml_beam_search_create();
    assert(ctx != NULL);
    assert(ctx->beam_width > 0);
    assert(ctx->warmup_runs >= 0);
    assert(ctx->timing_runs >= 0);
    assert(ctx->cache_count == 0);
    assert(ctx->num_candidates == 0);

    cml_beam_search_free(ctx);
    printf(" PASS\n");
}

static void test_cache_miss(void) {
    printf("  test_cache_miss...");

    CMLBeamSearchCtx* ctx = cml_beam_search_create();
    assert(ctx != NULL);

    /* Lookup on empty cache should return -1 (not found) */
    CMLBeamConfig config;
    memset(&config, 0, sizeof(config));
    int ret = cml_beam_search_lookup(ctx, 0xDEADBEEF, &config);
    assert(ret == -1);

    /* Try another hash */
    ret = cml_beam_search_lookup(ctx, 12345, &config);
    assert(ret == -1);

    cml_beam_search_free(ctx);
    printf(" PASS\n");
}

static void test_store_and_lookup(void) {
    printf("  test_store_and_lookup...");

    CMLBeamSearchCtx* ctx = cml_beam_search_create();
    assert(ctx != NULL);

    /* Store a config */
    CMLBeamConfig stored_config;
    memset(&stored_config, 0, sizeof(stored_config));
    stored_config.block_size_x = 128;
    stored_config.block_size_y = 1;
    stored_config.block_size_z = 1;
    stored_config.unroll_factor = 4;
    stored_config.vec_width = 4;

    uint64_t hash = 0xCAFEBABE;
    int ret = cml_beam_search_store(ctx, hash, &stored_config, 42.5);
    assert(ret == 0);

    /* Lookup the stored config */
    CMLBeamConfig found_config;
    memset(&found_config, 0, sizeof(found_config));
    ret = cml_beam_search_lookup(ctx, hash, &found_config);
    assert(ret == 0);
    assert(found_config.block_size_x == 128);
    assert(found_config.unroll_factor == 4);
    assert(found_config.vec_width == 4);

    /* Lookup a different hash should still miss */
    ret = cml_beam_search_lookup(ctx, 0xDEAD, &found_config);
    assert(ret == -1);

    cml_beam_search_free(ctx);
    printf(" PASS\n");
}

static void test_store_multiple(void) {
    printf("  test_store_multiple...");

    CMLBeamSearchCtx* ctx = cml_beam_search_create();
    assert(ctx != NULL);

    /* Store several configs with different hashes */
    for (int i = 0; i < 10; i++) {
        CMLBeamConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.block_size_x = 32 * (i + 1);
        cfg.vec_width = i + 1;

        int ret = cml_beam_search_store(ctx, (uint64_t)(100 + i), &cfg, (double)i * 10.0);
        assert(ret == 0);
    }

    /* Verify all can be looked up */
    for (int i = 0; i < 10; i++) {
        CMLBeamConfig found;
        memset(&found, 0, sizeof(found));
        int ret = cml_beam_search_lookup(ctx, (uint64_t)(100 + i), &found);
        assert(ret == 0);
        assert(found.block_size_x == 32 * (i + 1));
        assert(found.vec_width == i + 1);
    }

    cml_beam_search_free(ctx);
    printf(" PASS\n");
}

static void test_tune(void) {
    printf("  test_tune...");

    CMLBeamSearchCtx* ctx = cml_beam_search_create();
    assert(ctx != NULL);

    int shape[] = { 256, 256 };
    size_t total = 256 * 256;
    CMLBeamConfig best;
    memset(&best, 0, sizeof(best));

    int ret = cml_beam_search_tune(ctx, 0xABCD, total, 2, shape, &best);
    assert(ret == 0);

    /* The best config should have reasonable block sizes (> 0) */
    assert(best.block_size_x > 0);

    /* After tuning, the result should be cached */
    CMLBeamConfig cached;
    memset(&cached, 0, sizeof(cached));
    ret = cml_beam_search_lookup(ctx, 0xABCD, &cached);
    assert(ret == 0);
    assert(cached.block_size_x == best.block_size_x);

    cml_beam_search_free(ctx);
    printf(" PASS\n");
}

int main(void) {
    printf("=== BEAM Search Tests ===\n");

    test_create_free();
    test_cache_miss();
    test_store_and_lookup();
    test_store_multiple();
    test_tune();

    printf("All BEAM search tests passed.\n");
    return 0;
}
