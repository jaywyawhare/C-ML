/*
 * A/B comparison harness: run several named generation variants (greedy, nucleus, penalised,
 * min-p, prompt-lookup speculative) against the same tiny model and workload, then report a
 * side-by-side table of latency, throughput and token-agreement against the greedy baseline. It
 * doubles as a regression test: the variants that must match the baseline exactly are asserted, as
 * are reproducibility and valid-token-range invariants, so it runs green in CI while still printing
 * a human-readable comparison.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "nn/llama.h"
#include "nn/speculative.h"
#include "backend/profiling.h"
#include "core/threefry.h"
#include "alloc/cml_allocator.h"
#include "test_harness.h"
#include "llama_test_model.h"

#define AB_STEPS 12
#define AB_REPS 5

static const int AB_PROMPT[] = {1, 2, 3, 4};
#define AB_PLEN ((int)(sizeof(AB_PROMPT) / sizeof(AB_PROMPT[0])))

typedef struct {
    const char* name;
    CMLGenerationConfig cfg;
    int tokens[AB_STEPS];
    int count;
    double median_ms;
    double tokens_per_sec;
    float agreement; /* fraction of positions matching the greedy baseline */
} ABVariant;

/** Median of n doubles (sorts the buffer in place). */
static double ab_median(double* v, int n) {
    for (int i = 1; i < n; i++) {
        double key = v[i];
        int j      = i - 1;
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = key;
    }
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/** Fraction of positions (over the shorter length) where two token runs agree. */
static float ab_agreement(const int* a, int na, const int* b, int nb) {
    int n = na < nb ? na : nb;
    if (n == 0)
        return 0.0f;
    int same = 0;
    for (int i = 0; i < n; i++)
        same += (a[i] == b[i]);
    return (float)same / (float)n;
}

/**
 * Run one variant on its own fresh model (identical architecture, so cost is comparable and the
 * per-process IR-graph growth doesn't bias one variant against the next): time AB_REPS seeded
 * decodes for the median latency and measure the variant's token agreement with greedy decode on
 * the same model.
 * @param v (ABVariant*)
 * @param vocab_out (int*)
 * @return (int)
 */
static int ab_run_variant(ABVariant* v, int* vocab_out) {
    /* Clear the process-global IR graph before building this variant's model so accumulated nodes
     * from earlier variants don't bias its timing. Safe here: no live model references the graph.
     */
    cml_reset_ir_context();
    CMLLLaMAModel* m = build_tiny_model();
    if (!m)
        return 0;
    warmup_tiny_model(m, AB_PROMPT, AB_PLEN);
    *vocab_out = m->config.vocab_size;

    CMLGenerationConfig gc = tiny_greedy_config();
    int base[AB_STEPS];
    int base_n = tiny_decode(m, AB_PROMPT, AB_PLEN, &gc, AB_STEPS, base, NULL);

    double times[AB_REPS];
    for (int r = 0; r < AB_REPS; r++) {
        cml_rng_set_global_seed(4242);
        double t0 = cml_monotonic_ms();
        v->count  = tiny_decode(m, AB_PROMPT, AB_PLEN, &v->cfg, AB_STEPS, v->tokens, NULL);
        times[r]  = cml_monotonic_ms() - t0;
    }
    v->median_ms      = ab_median(times, AB_REPS);
    v->tokens_per_sec = v->count > 0 ? (v->count * 1000.0) / v->median_ms : 0.0;
    v->agreement      = ab_agreement(v->tokens, v->count, base, base_n);

    cml_llama_free(m);
    return 1;
}

static int test_ab_compare(void) {
    ABVariant variants[5];
    int nv = 0;

    variants[nv++] = (ABVariant){.name = "greedy", .cfg = tiny_greedy_config()};

    CMLGenerationConfig nucleus = cml_generation_default_config();
    nucleus.do_sample           = true;
    nucleus.temperature         = 0.8f;
    variants[nv++]              = (ABVariant){.name = "nucleus(t=0.8,p=0.9,k=40)", .cfg = nucleus};

    CMLGenerationConfig penal  = cml_generation_default_config();
    penal.do_sample            = true;
    penal.temperature          = 1.0f;
    penal.repetition_penalty   = 1.3f;
    penal.no_repeat_ngram_size = 2;
    variants[nv++]             = (ABVariant){.name = "penalised(rep=1.3,nrng=2)", .cfg = penal};

    CMLGenerationConfig minp = cml_generation_default_config();
    minp.do_sample           = true;
    minp.temperature         = 1.0f;
    minp.top_k               = 0;
    minp.top_p               = 1.0f;
    minp.min_p               = 0.1f;
    variants[nv++]           = (ABVariant){.name = "min_p(0.1)", .cfg = minp};

    CMLGenerationConfig greedy2 = tiny_greedy_config();
    variants[nv++]              = (ABVariant){.name = "greedy(self)", .cfg = greedy2};

    int vocab = 0;
    for (int i = 0; i < nv; i++)
        if (!ab_run_variant(&variants[i], &vocab))
            return 0;

    printf("\n  %-28s %8s %10s %12s\n", "variant", "ms", "tok/s", "vs-greedy%");
    printf("  %-28s %8s %10s %12s\n", "----------------------------", "--------", "----------",
           "------------");
    for (int i = 0; i < nv; i++)
        printf("  %-28s %8.3f %10.1f %11.0f%%\n", variants[i].name, variants[i].median_ms,
               variants[i].tokens_per_sec, 100.0 * variants[i].agreement);

    /* Invariants: every variant produced the full run of in-range tokens; greedy compared against
     * its own greedy baseline agrees everywhere. */
    int ok = 1;
    for (int i = 0; i < nv && ok; i++) {
        if (variants[i].count != AB_STEPS)
            ok = 0;
        for (int j = 0; j < variants[i].count && ok; j++)
            if (variants[i].tokens[j] < 0 || variants[i].tokens[j] >= vocab)
                ok = 0;
    }
    if (variants[4].agreement < 0.999f) /* greedy vs greedy must be identical */
        ok = 0;

    return ok;
}

/* Speculative prompt-lookup must reproduce the greedy baseline token-for-token (its correctness
 * guarantee), while reporting its own timing against plain greedy decode. */
static int test_ab_speculative_vs_greedy(void) {
    CMLLLaMAModel* m = build_tiny_model();
    if (!m)
        return 0;
    warmup_tiny_model(m, AB_PROMPT, AB_PLEN);

    /* Baseline greedy run. */
    CMLGenerationConfig gc = tiny_greedy_config();
    int base[AB_STEPS];
    double t0      = cml_monotonic_ms();
    int base_n     = tiny_decode(m, AB_PROMPT, AB_PLEN, &gc, AB_STEPS, base, NULL);
    double base_ms = cml_monotonic_ms() - t0;

    /* Prompt-lookup speculative decode producing tokens one accepted-block at a time. */
    CMLSpeculativeConfig scfg  = cml_speculative_default_config();
    scfg.do_sample             = false;
    CMLSpeculativeDecoder* dec = cml_speculative_create(&scfg, m->config.vocab_size);
    if (!dec) {
        cml_llama_free(m);
        return 0;
    }
    cml_speculative_set_ngram_draft(dec, 2);
    cml_speculative_set_target_model(dec, m, tiny_spec_forward, tiny_spec_argmax);

    int spec[AB_STEPS];
    int spec_n = 0;
    int ctx_buf[AB_PLEN + AB_STEPS];
    memcpy(ctx_buf, AB_PROMPT, sizeof(AB_PROMPT));
    int ctx_len = AB_PLEN;

    double s0 = cml_monotonic_ms();
    while (spec_n < AB_STEPS) {
        CMLSpeculativeResult* res = cml_speculative_decode_step(dec, ctx_buf, ctx_len);
        if (!res)
            break;
        for (int i = 0; i < res->num_accepted && spec_n < AB_STEPS; i++) {
            spec[spec_n++]     = res->accepted_tokens[i];
            ctx_buf[ctx_len++] = res->accepted_tokens[i];
        }
        cml_speculative_result_free(res);
    }
    double spec_ms = cml_monotonic_ms() - s0;

    float agree = ab_agreement(spec, spec_n, base, base_n);
    printf("\n  %-28s %8s %10s\n", "decode", "ms", "agree%");
    printf("  %-28s %8.3f %9s\n", "greedy", base_ms, "-");
    printf("  %-28s %8.3f %8.0f%%\n", "prompt-lookup speculative", spec_ms, 100.0 * agree);

    int ok = (spec_n == base_n) && (agree > 0.999f); /* must match greedy exactly */

    cml_speculative_free(dec);
    cml_llama_free(m);
    return ok;
}

int main(void) {
    printf("test_ab (A/B generation comparison)\n");
    srand(2024);

    TEST(ab_compare);
    TEST(ab_speculative_vs_greedy);

    return TEST_SUMMARY();
}
