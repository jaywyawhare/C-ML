/*
 * LLM inference profiler: drives a decode run through the real model and attributes wall time to
 * the phases a serving stack actually cares about (prefill, per-step forward, and sampling)
 * using the project's own Profiler scopes. Reports time-to-first-token, steady-state decode
 * throughput and the forward/sample split, then prints the aggregated scope report and writes it
 * to JSON. A timing harness, not an assertion test.
 */
#include <stdio.h>
#include <stdlib.h>

#include "cml.h"
#include "nn/llama.h"
#include "nn/sampling.h"
#include "backend/profiling.h"
#include "alloc/cml_allocator.h"
#include "core/threefry.h"

#include "../tests/llama_test_model.h"

/* A model a few steps up from the unit-test tiny one, so the phase numbers are representative. */
static CMLLLaMAModel* build_profile_model(void) {
    CMLLLaMAConfig cfg = {.vocab_size        = 256,
                          .hidden_size       = 256,
                          .intermediate_size = 688,
                          .num_layers        = 4,
                          .num_heads         = 8,
                          .num_kv_heads      = 8,
                          .max_seq_len       = 256,
                          .rope_theta        = 10000.0f,
                          .rms_norm_eps      = 1e-5f};
    CMLLLaMAModel* m   = cml_llama_create(&cfg);
    if (!m)
        return NULL;
    if (init_model_random(m) != 0) {
        cml_llama_free(m);
        return NULL;
    }
    return m;
}

int main(void) {
    srand(7);
    CMLLLaMAModel* m = build_profile_model();
    if (!m) {
        fprintf(stderr, "profile_llm: model build failed\n");
        return 1;
    }

    const int prompt[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int plen     = (int)(sizeof(prompt) / sizeof(prompt[0]));
    const int steps    = 64;

    /* Realize weights before timing so lazy materialization isn't charged to prefill. */
    warmup_tiny_model(m, prompt, plen);

    Profiler* prof          = profiler_create();
    CMLGenerationConfig cfg = cml_generation_default_config();
    cfg.do_sample           = true;
    cfg.temperature         = 0.8f;
    cml_rng_set_global_seed(1234);

    int* hist = (int*)cml_malloc((size_t)(plen + steps) * sizeof(int));
    for (int i = 0; i < plen; i++)
        hist[i] = prompt[i];
    int hlen = plen;

    double t_start = cml_monotonic_ms();
    double ttft    = 0.0;

    cml_llama_reset(m);
    profiler_scope_begin(prof, "prefill");
    Tensor* logits = cml_llama_forward(m, prompt, plen);
    profiler_scope_end(prof);
    if (!logits) {
        fprintf(stderr, "profile_llm: prefill failed\n");
        cml_free(hist);
        profiler_free(prof);
        cml_llama_free(m);
        return 1;
    }

    int produced = 0;
    for (int s = 0; s < steps; s++) {
        profiler_scope_begin(prof, "sample");
        int tok = cml_llama_sample_token(logits, &cfg, hist, hlen, NULL);
        profiler_scope_end(prof);
        tensor_free(logits);
        logits = NULL;
        if (tok < 0)
            break;
        if (produced == 0)
            ttft = cml_monotonic_ms() - t_start;
        hist[hlen++] = tok;
        produced++;
        if (m->current_seq_len >= m->config.max_seq_len)
            break;
        profiler_scope_begin(prof, "decode_forward");
        logits = cml_llama_forward(m, &tok, 1);
        profiler_scope_end(prof);
        if (!logits)
            break;
    }
    if (logits)
        tensor_free(logits);
    double total_ms = cml_monotonic_ms() - t_start;

    const ProfileStat* prefill = profiler_find_stat(prof, "prefill");
    const ProfileStat* fwd     = profiler_find_stat(prof, "decode_forward");
    const ProfileStat* smp     = profiler_find_stat(prof, "sample");
    double decode_ms           = total_ms - (prefill ? prefill->total_ms : 0.0);

    printf("\n=== LLM inference profile ===\n");
    printf("model: %d layers, hidden %d, %d heads, vocab %d\n", m->config.num_layers,
           m->config.hidden_size, m->config.num_heads, m->config.vocab_size);
    printf("prompt %d tokens, generated %d tokens\n\n", plen, produced);

    printf("time-to-first-token : %8.3f ms\n", ttft);
    printf("prefill             : %8.3f ms\n", prefill ? prefill->total_ms : 0.0);
    printf("decode (total)      : %8.3f ms  (%.1f tok/s)\n", decode_ms,
           decode_ms > 0.0 ? produced * 1000.0 / decode_ms : 0.0);
    if (fwd && fwd->count > 0)
        printf("  mean fwd step     : %8.3f ms  (x%d)\n", fwd->total_ms / (double)fwd->count,
               (int)fwd->count);
    if (smp && smp->count > 0)
        printf("  mean sample step  : %8.3f ms  (x%d)\n", smp->total_ms / (double)smp->count,
               (int)smp->count);
    if (fwd && smp && (fwd->total_ms + smp->total_ms) > 0.0)
        printf("  forward / sample  : %.1f%% / %.1f%% of decode compute\n",
               100.0 * fwd->total_ms / (fwd->total_ms + smp->total_ms),
               100.0 * smp->total_ms / (fwd->total_ms + smp->total_ms));

    printf("\n");
    profiler_print_report(prof);

    const char* json = "/tmp/profile_llm.json";
    if (profiler_write_json(prof, json) == 0)
        printf("\nscope report written to %s\n", json);

    cml_free(hist);
    profiler_free(prof);
    cml_llama_free(m);
    return 0;
}
