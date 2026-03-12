/**
 * @file beam_search.h
 * @brief BEAM search kernel optimization
 *
 * Parametric kernel tuning: try N launch configs (block size, unroll,
 * vectorization width) and pick fastest. Enable via CML_BEAM=N env var.
 */

#ifndef CML_OPS_IR_BEAM_SEARCH_H
#define CML_OPS_IR_BEAM_SEARCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CML_BEAM_MAX_CANDIDATES 64
#define CML_BEAM_DEFAULT_WIDTH  4
#define CML_BEAM_DEFAULT_WARMUP 2
#define CML_BEAM_DEFAULT_TIMING 5

/* ── Kernel configuration candidate ── */

typedef struct {
    int block_size_x;
    int block_size_y;
    int block_size_z;
    int unroll_factor;
    int vec_width;
    size_t grid[3];
    size_t block[3];
    size_t shared_mem;
} CMLBeamConfig;

/* ── Timing result ── */

typedef struct {
    CMLBeamConfig config;
    double time_us;     /* execution time in microseconds */
    bool valid;
} CMLBeamResult;

/* ── Search context ── */

typedef struct {
    int beam_width;       /* number of top candidates to keep */
    int warmup_runs;
    int timing_runs;

    /* Results cache (hash -> best config) */
    struct {
        uint64_t hash;
        CMLBeamConfig config;
        double time_us;
        bool occupied;
    } cache[256];
    int cache_count;

    /* Current search state */
    CMLBeamResult candidates[CML_BEAM_MAX_CANDIDATES];
    int num_candidates;
} CMLBeamSearchCtx;

/* ── API ── */

/**
 * @brief Create a BEAM search context (reads CML_BEAM env var for width)
 */
CMLBeamSearchCtx* cml_beam_search_create(void);

/**
 * @brief Free a BEAM search context
 */
void cml_beam_search_free(CMLBeamSearchCtx* ctx);

/**
 * @brief Check if BEAM search is enabled (CML_BEAM env var set)
 */
bool cml_beam_search_enabled(void);

/**
 * @brief Tune a kernel: generate candidates, time, and return best config
 *
 * @param ctx           Search context
 * @param kernel_hash   Hash identifying the kernel
 * @param total_elements Total elements to process
 * @param ndim          Number of dimensions
 * @param shape         Shape array [ndim]
 * @param best_out      Output: best configuration found
 * @return 0 on success, -1 on error
 */
int cml_beam_search_tune(CMLBeamSearchCtx* ctx, uint64_t kernel_hash,
                         size_t total_elements, int ndim, const int* shape,
                         CMLBeamConfig* best_out);

/**
 * @brief Look up cached best config for a kernel hash
 * @return 0 if found (best_out populated), -1 if not cached
 */
int cml_beam_search_lookup(CMLBeamSearchCtx* ctx, uint64_t kernel_hash,
                           CMLBeamConfig* best_out);

/**
 * @brief Store a tuning result in cache
 */
int cml_beam_search_store(CMLBeamSearchCtx* ctx, uint64_t kernel_hash,
                          const CMLBeamConfig* config, double time_us);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_BEAM_SEARCH_H */
