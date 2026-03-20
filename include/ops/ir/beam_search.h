/*
 * BEAM search kernel optimization.
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

typedef struct {
    CMLBeamConfig config;
    double time_us;     /* execution time in microseconds */
    bool valid;
} CMLBeamResult;

typedef struct {
    int beam_width;       /* number of top candidates to keep */
    int warmup_runs;
    int timing_runs;

    struct {
        uint64_t hash;
        CMLBeamConfig config;
        double time_us;
        bool occupied;
    } cache[256];
    int cache_count;

    CMLBeamResult candidates[CML_BEAM_MAX_CANDIDATES];
    int num_candidates;
} CMLBeamSearchCtx;

CMLBeamSearchCtx* cml_beam_search_create(void);
void cml_beam_search_free(CMLBeamSearchCtx* ctx);
bool cml_beam_search_enabled(void);

int cml_beam_search_tune(CMLBeamSearchCtx* ctx, uint64_t kernel_hash,
                         size_t total_elements, int ndim, const int* shape,
                         CMLBeamConfig* best_out);

/* Returns 0 if found (best_out populated), -1 if not cached */
int cml_beam_search_lookup(CMLBeamSearchCtx* ctx, uint64_t kernel_hash,
                           CMLBeamConfig* best_out);

int cml_beam_search_store(CMLBeamSearchCtx* ctx, uint64_t kernel_hash,
                          const CMLBeamConfig* config, double time_us);

typedef struct {
    CMLBeamConfig config;
    void* compiled_kernel;   /* backend-specific compiled kernel */
    char* source_code;       /* generated source (PTX/CUDA C/MSL/WGSL) */
} CMLBeamVariant;

/* Returns execution time in microseconds, or -1.0 on error */
typedef double (*CMLBeamTimingFn)(const CMLBeamVariant* variant, void* user_data);

int cml_beam_search_tune_hw(CMLBeamSearchCtx* ctx, uint64_t kernel_hash,
                             size_t total_elements,
                             CMLBeamTimingFn timing_fn, void* user_data,
                             CMLBeamConfig* best_out);

int cml_beam_cache_save(CMLBeamSearchCtx* ctx, const char* path);
int cml_beam_cache_load(CMLBeamSearchCtx* ctx, const char* path);

double cml_beam_cuda_timing_fn(const CMLBeamVariant* variant, void* user_data);

struct LinearProgram;

int cml_beam_search_tune_opt(CMLBeamSearchCtx* ctx, uint64_t kernel_hash,
                              struct LinearProgram* prog,
                              CMLBeamTimingFn timing_fn, void* user_data,
                              CMLBeamConfig* best_out);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_BEAM_SEARCH_H */
