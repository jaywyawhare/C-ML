/**
 * @file beam_search.c
 * @brief BEAM search kernel optimization
 *
 * Parametric kernel tuning: generate candidate launch configurations
 * (block size, unroll factor, vectorization width), score them with a
 * heuristic and pick the best.  Results are cached so repeated calls
 * with the same kernel hash are cheap.
 *
 * Enable via the CML_BEAM environment variable:
 *   CML_BEAM=4   -- beam width 4 (default)
 *   CML_BEAM=8   -- wider search
 *   CML_BEAM=0   -- disabled
 */

#include "ops/ir/beam_search.h"
#include "core/logging.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Tuning parameter space ──────────────────────────────────────────────── */

static const int BLOCK_SIZES[]   = {32, 64, 128, 256, 512, 1024};
static const int UNROLL_FACTORS[] = {1, 2, 4};
static const int VEC_WIDTHS[]    = {1, 2, 4};

#define NUM_BLOCK_SIZES   (int)(sizeof(BLOCK_SIZES)   / sizeof(BLOCK_SIZES[0]))
#define NUM_UNROLL_FACTORS (int)(sizeof(UNROLL_FACTORS) / sizeof(UNROLL_FACTORS[0]))
#define NUM_VEC_WIDTHS    (int)(sizeof(VEC_WIDTHS)    / sizeof(VEC_WIDTHS[0]))

/* ── Cache hash helpers ──────────────────────────────────────────────────── */

/**
 * Map a kernel hash to a cache slot index (simple modular hash).
 */
static int cache_slot(uint64_t hash) {
    return (int)(hash % 256);
}

/* ══════════════════════════════════════════════════════════════════════════
 * Public API
 * ══════════════════════════════════════════════════════════════════════════ */

bool cml_beam_search_enabled(void) {
    const char* env = getenv("CML_BEAM");
    if (!env) return false;
    int val = atoi(env);
    return val > 0;
}

CMLBeamSearchCtx* cml_beam_search_create(void) {
    CMLBeamSearchCtx* ctx =
        (CMLBeamSearchCtx*)calloc(1, sizeof(CMLBeamSearchCtx));
    if (!ctx) {
        LOG_ERROR("Failed to allocate CMLBeamSearchCtx");
        return NULL;
    }

    /* Read beam width from environment; fall back to default. */
    const char* env = getenv("CML_BEAM");
    if (env) {
        int w = atoi(env);
        ctx->beam_width = (w > 0) ? w : CML_BEAM_DEFAULT_WIDTH;
    } else {
        ctx->beam_width = CML_BEAM_DEFAULT_WIDTH;
    }

    ctx->warmup_runs    = CML_BEAM_DEFAULT_WARMUP;
    ctx->timing_runs    = CML_BEAM_DEFAULT_TIMING;
    ctx->cache_count    = 0;
    ctx->num_candidates = 0;

    /* Ensure the cache is empty. */
    for (int i = 0; i < 256; i++) {
        ctx->cache[i].occupied = false;
    }

    LOG_INFO("BEAM search context created (width=%d, warmup=%d, timing=%d)",
             ctx->beam_width, ctx->warmup_runs, ctx->timing_runs);
    return ctx;
}

void cml_beam_search_free(CMLBeamSearchCtx* ctx) {
    if (!ctx) return;
    LOG_DEBUG("Freeing BEAM search context %p", (void*)ctx);
    free(ctx);
}

/* ── Cache lookup ────────────────────────────────────────────────────────── */

int cml_beam_search_lookup(CMLBeamSearchCtx* ctx, uint64_t kernel_hash,
                           CMLBeamConfig* best_out) {
    if (!ctx || !best_out) return -1;

    int slot = cache_slot(kernel_hash);

    /*
     * Linear probe starting from the hashed slot.  The cache is small (256
     * entries) so a full scan in the worst case is still fast.
     */
    for (int i = 0; i < 256; i++) {
        int idx = (slot + i) % 256;
        if (!ctx->cache[idx].occupied) {
            /* Empty slot -- not found. */
            return -1;
        }
        if (ctx->cache[idx].hash == kernel_hash) {
            *best_out = ctx->cache[idx].config;
            LOG_DEBUG("BEAM cache hit for hash 0x%016llx (slot %d, time=%.2f us)",
                      (unsigned long long)kernel_hash, idx,
                      ctx->cache[idx].time_us);
            return 0;
        }
    }

    return -1; /* Cache full and key not found. */
}

/* ── Cache store ─────────────────────────────────────────────────────────── */

int cml_beam_search_store(CMLBeamSearchCtx* ctx, uint64_t kernel_hash,
                          const CMLBeamConfig* config, double time_us) {
    if (!ctx || !config) return -1;

    int slot = cache_slot(kernel_hash);

    /*
     * Linear probe: find an empty slot or the existing entry with the same
     * hash (overwrite with potentially better result).
     */
    for (int i = 0; i < 256; i++) {
        int idx = (slot + i) % 256;

        if (!ctx->cache[idx].occupied) {
            /* Empty slot -- insert. */
            ctx->cache[idx].hash     = kernel_hash;
            ctx->cache[idx].config   = *config;
            ctx->cache[idx].time_us  = time_us;
            ctx->cache[idx].occupied = true;
            ctx->cache_count++;
            LOG_DEBUG("BEAM cache store: hash 0x%016llx -> slot %d "
                      "(time=%.2f us, count=%d)",
                      (unsigned long long)kernel_hash, idx, time_us,
                      ctx->cache_count);
            return 0;
        }

        if (ctx->cache[idx].hash == kernel_hash) {
            /* Existing entry -- update if the new time is better. */
            if (time_us < ctx->cache[idx].time_us) {
                ctx->cache[idx].config  = *config;
                ctx->cache[idx].time_us = time_us;
                LOG_DEBUG("BEAM cache update: hash 0x%016llx slot %d "
                          "(new time=%.2f us)",
                          (unsigned long long)kernel_hash, idx, time_us);
            }
            return 0;
        }
    }

    LOG_WARNING("BEAM cache is full -- cannot store hash 0x%016llx",
                (unsigned long long)kernel_hash);
    return -1;
}

/* ── Heuristic scoring ───────────────────────────────────────────────────── */

/**
 * Compute a heuristic score for a candidate configuration.
 *
 * Lower is better.  The heuristic prefers block sizes close to
 * sqrt(total_elements) (a reasonable initial approximation for occupancy),
 * multiplied by penalties for excessively large unroll or vec_width
 * relative to the problem size.
 */
static double heuristic_score(const CMLBeamConfig* cfg, size_t total_elements) {
    double ideal_block = sqrt((double)total_elements);
    double block_dist  = fabs((double)cfg->block_size_x - ideal_block);

    /*
     * Normalize block_dist so that it contributes roughly equally regardless
     * of problem size.
     */
    double norm_dist = block_dist / (ideal_block + 1.0);

    /*
     * Penalize configurations whose effective parallelism (block * vec * unroll)
     * overshoots total_elements by a lot -- they waste work on guard checks.
     */
    double effective = (double)cfg->block_size_x *
                       (double)cfg->vec_width *
                       (double)cfg->unroll_factor;
    double overshoot_penalty = 0.0;
    if (effective > (double)total_elements) {
        overshoot_penalty = (effective - (double)total_elements) /
                            ((double)total_elements + 1.0);
    }

    return norm_dist + 0.5 * overshoot_penalty;
}

/**
 * Estimate execution time for a candidate on the CPU.
 *
 * Since there is no actual GPU to time, we use a simple throughput model:
 *   time ~ total_elements / (block_size * vec_width * unroll_factor)
 *
 * The result is in arbitrary units -- only the *relative* ordering matters.
 */
static double estimate_time_cpu(const CMLBeamConfig* cfg,
                                size_t total_elements) {
    double throughput = (double)cfg->block_size_x *
                        (double)cfg->vec_width *
                        (double)cfg->unroll_factor;
    if (throughput < 1.0) throughput = 1.0;
    return (double)total_elements / throughput;
}

/* ── Comparison function for qsort ───────────────────────────────────────── */

/**
 * Sort helper -- used to rank candidates by their heuristic score.
 * We stash the score in time_us temporarily during candidate generation.
 */
static int cmp_beam_result(const void* a, const void* b) {
    const CMLBeamResult* ra = (const CMLBeamResult*)a;
    const CMLBeamResult* rb = (const CMLBeamResult*)b;
    if (ra->time_us < rb->time_us) return -1;
    if (ra->time_us > rb->time_us) return  1;
    return 0;
}

/* ── Main tune entry point ───────────────────────────────────────────────── */

int cml_beam_search_tune(CMLBeamSearchCtx* ctx, uint64_t kernel_hash,
                         size_t total_elements, int ndim, const int* shape,
                         CMLBeamConfig* best_out) {
    if (!ctx || !best_out) {
        LOG_ERROR("NULL context or output in beam_search_tune");
        return -1;
    }

    (void)ndim;
    (void)shape;

    /* 1. Check cache first. */
    if (cml_beam_search_lookup(ctx, kernel_hash, best_out) == 0) {
        LOG_DEBUG("BEAM tune: using cached result for hash 0x%016llx",
                  (unsigned long long)kernel_hash);
        return 0;
    }

    LOG_INFO("BEAM tune: searching for hash 0x%016llx (total_elements=%zu)",
             (unsigned long long)kernel_hash, total_elements);

    /* 2. Generate all candidate configurations. */
    CMLBeamResult all_candidates[CML_BEAM_MAX_CANDIDATES];
    int num_all = 0;

    for (int bi = 0; bi < NUM_BLOCK_SIZES && num_all < CML_BEAM_MAX_CANDIDATES; bi++) {
        for (int ui = 0; ui < NUM_UNROLL_FACTORS && num_all < CML_BEAM_MAX_CANDIDATES; ui++) {
            for (int vi = 0; vi < NUM_VEC_WIDTHS && num_all < CML_BEAM_MAX_CANDIDATES; vi++) {
                CMLBeamResult* r = &all_candidates[num_all];
                memset(r, 0, sizeof(*r));

                r->config.block_size_x  = BLOCK_SIZES[bi];
                r->config.block_size_y  = 1;
                r->config.block_size_z  = 1;
                r->config.unroll_factor = UNROLL_FACTORS[ui];
                r->config.vec_width     = VEC_WIDTHS[vi];
                r->config.shared_mem    = 0;

                /* Derive grid from total_elements and block_size. */
                size_t threads = (size_t)r->config.block_size_x;
                r->config.block[0] = threads;
                r->config.block[1] = 1;
                r->config.block[2] = 1;
                r->config.grid[0]  = (total_elements + threads - 1) / threads;
                r->config.grid[1]  = 1;
                r->config.grid[2]  = 1;

                r->valid   = true;
                /* Use heuristic score as an initial "time" for sorting. */
                r->time_us = heuristic_score(&r->config, total_elements);
                num_all++;
            }
        }
    }

    if (num_all == 0) {
        LOG_ERROR("BEAM tune: no candidates generated");
        return -1;
    }

    /* 3. Quick filter: sort by heuristic score and keep top beam_width. */
    qsort(all_candidates, (size_t)num_all, sizeof(CMLBeamResult),
          cmp_beam_result);

    int keep = ctx->beam_width;
    if (keep > num_all) keep = num_all;
    if (keep > CML_BEAM_MAX_CANDIDATES) keep = CML_BEAM_MAX_CANDIDATES;

    LOG_DEBUG("BEAM tune: generated %d candidates, keeping top %d",
              num_all, keep);

    /* Copy the survivors into the context's candidate array. */
    memcpy(ctx->candidates, all_candidates,
           (size_t)keep * sizeof(CMLBeamResult));
    ctx->num_candidates = keep;

    /* 4. Estimate execution time for each surviving candidate (CPU model). */
    for (int i = 0; i < keep; i++) {
        ctx->candidates[i].time_us =
            estimate_time_cpu(&ctx->candidates[i].config, total_elements);
        LOG_DEBUG("  candidate %d: block=%d unroll=%d vec=%d -> est %.2f us",
                  i, ctx->candidates[i].config.block_size_x,
                  ctx->candidates[i].config.unroll_factor,
                  ctx->candidates[i].config.vec_width,
                  ctx->candidates[i].time_us);
    }

    /* 5. Pick the best (lowest estimated time). */
    int best_idx = 0;
    for (int i = 1; i < keep; i++) {
        if (ctx->candidates[i].time_us < ctx->candidates[best_idx].time_us) {
            best_idx = i;
        }
    }

    *best_out = ctx->candidates[best_idx].config;

    LOG_INFO("BEAM tune: best config for hash 0x%016llx: "
             "block=%d unroll=%d vec=%d (est %.2f us)",
             (unsigned long long)kernel_hash,
             best_out->block_size_x, best_out->unroll_factor,
             best_out->vec_width,
             ctx->candidates[best_idx].time_us);

    /* 6. Store in cache. */
    cml_beam_search_store(ctx, kernel_hash, best_out,
                          ctx->candidates[best_idx].time_us);
    return 0;
}
