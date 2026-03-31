# BEAM Search Auto-Tuning

C-ML has two related BEAM autotuning systems:

1. **Generic BEAM search** (`beam_search.h`) — tunes any kernel's block size, unroll factor, and vector width. Supports optional CUDA hardware timing.
2. **OpenCL GEMM BEAM** (built into `opencl_ir_backend`) — tunes tile parameters for GEMM kernels on OpenCL GPUs using OpenCL profiling events.


## Table of Contents

1. [Generic BEAM Search](#generic-beam-search)
   - [Overview](#overview)
   - [How It Works](#how-it-works)
   - [API Reference](#api-reference)
   - [Configuration](#configuration)
   - [CUDA Hardware Timing](#cuda-hardware-timing)
   - [Cache Persistence](#cache-persistence)
   - [Usage Example](#usage-example)
2. [OpenCL GEMM BEAM Autotuner](#opencl-gemm-beam-autotuner)


## Generic BEAM Search

### Overview

When launching GPU kernels, performance depends heavily on the block size, unroll factor, and vector width. BEAM search automates this tuning by:

1. Generating candidate configurations across the parameter space
2. Filtering candidates using heuristic scoring (occupancy-oriented)
3. Optionally timing the top candidates on actual hardware via CUDA events
4. Caching the best configuration to avoid re-tuning

**Files:** `include/ops/ir/beam_search.h`, `src/ops/ir/beam_search.c`, `src/ops/ir/beam_cuda_timing.c`


### How It Works

```
Parameter Space (54 configs)
     |
     v
Heuristic Filtering        score by distance from sqrt(total_elements)
  keep top beam_width      default: top 4 candidates
     |
     +--------+--------+
     |                  |
     v                  v
Heuristic           Hardware
Estimation          Timing (CUDA)
     |                  |
     +--------+---------+
              |
              v
Cache Best Config          store by kernel_hash
  256-entry table          persist to disk (BMCH format)
```

### Tuning Parameter Space

| Dimension | Values | Count |
|-----------|--------|-------|
| Block size (x) | 32, 64, 128, 256, 512, 1024 | 6 |
| Unroll factor | 1, 2, 4 | 3 |
| Vector width | 1, 2, 4 | 3 |
| **Total** | | **54** |

### Heuristic Scoring

Candidates are scored based on:
- **Occupancy**: Block sizes closer to `sqrt(total_elements)` score higher
- **Overshoot penalty**: Configurations that exceed total elements waste work
- The top `beam_width` candidates survive for evaluation


### API Reference

### Core Structs

```c
typedef struct {
    int block_size_x, block_size_y, block_size_z;
    int unroll_factor;
    int vec_width;
    size_t grid[3];
    size_t block[3];
    size_t shared_mem;
} CMLBeamConfig;

typedef struct {
    CMLBeamConfig config;
    double time_us;         /* execution time in microseconds */
    bool valid;
} CMLBeamResult;
```

### Functions

| Function | Description |
|----------|-------------|
| `cml_beam_search_create()` | Create a BEAM search context |
| `cml_beam_search_free(ctx)` | Free a BEAM search context |
| `cml_beam_search_enabled()` | Check if BEAM search is enabled (via `CML_BEAM` env var) |
| `cml_beam_search_tune(ctx, hash, total, ndim, shape, best)` | Tune using heuristic estimation |
| `cml_beam_search_tune_hw(ctx, hash, total, timing_fn, data, best)` | Tune using hardware timing callback |
| `cml_beam_search_lookup(ctx, hash, best)` | Look up cached config |
| `cml_beam_search_store(ctx, hash, config, time)` | Store a config in cache |
| `cml_beam_cache_save(ctx, path)` | Save cache to disk |
| `cml_beam_cache_load(ctx, path)` | Load cache from disk |
| `cml_beam_cuda_timing_fn(variant, user_data)` | CUDA hardware timing callback |


### Configuration

### Environment Variable

```bash
export CML_BEAM=0    # Disable BEAM search
export CML_BEAM=4    # Enable with beam width 4 (default)
export CML_BEAM=8    # Wider search (more candidates evaluated)
export CML_BEAM=16   # Exhaustive search
```

### Context Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beam_width` | 4 | Number of top candidates to keep |
| `warmup_runs` | 2 | GPU warmup iterations before timing |
| `timing_runs` | 5 | Actual measurement iterations |


### CUDA Hardware Timing

The `cml_beam_cuda_timing_fn()` callback provides precise GPU measurements using CUDA events. It replaces CPU-side heuristic estimation with actual kernel execution timing.

**Process:**

1. Compile variant source code via NVRTC (runtime compilation)
2. Set grid/block dimensions from the candidate config
3. Execute warmup runs (2 iterations, untimed)
4. Execute timed runs (5 iterations) between CUDA events
5. Synchronize and compute average time in microseconds

```c
CMLBeamConfig best;
cml_beam_search_tune_hw(
    ctx,
    kernel_hash,
    total_elements,
    cml_beam_cuda_timing_fn,    /* CUDA timing callback */
    NULL,
    &best
);
```

The hardware path pre-filters to `beam_width * 2` candidates using heuristics before running the expensive GPU measurements, balancing search quality with tuning time.

**Requires:** `CML_HAS_CUDA` compile flag. Without it, `cml_beam_cuda_timing_fn()` returns -1.0 (error).


### Cache Persistence

Tuning results are cached in a 256-entry hash table with linear probing. Caches can be saved to and loaded from disk in a binary format with magic number `BMCH` (0x424D4348).

```c
cml_beam_cache_save(ctx, "beam_cache.bin");
cml_beam_cache_load(ctx, "beam_cache.bin");
```


### Usage Example

```c
#include "ops/ir/beam_search.h"

CMLBeamSearchCtx* ctx = cml_beam_search_create();

if (!cml_beam_search_enabled()) {
    printf("BEAM search disabled (set CML_BEAM env var)\n");
    return;
}

CMLBeamConfig best;
int shape[] = {1024, 1024};
int rc = cml_beam_search_tune(ctx, kernel_hash, 1024*1024, 2, shape, &best);
if (rc == 0) {
    printf("Best config: block=%d, unroll=%d, vec=%d\n",
           best.block_size_x, best.unroll_factor, best.vec_width);
}

rc = cml_beam_search_tune_hw(ctx, kernel_hash, 1024*1024,
                              cml_beam_cuda_timing_fn, NULL, &best);

cml_beam_cache_save(ctx, "beam_cache.bin");

cml_beam_search_free(ctx);
```


## OpenCL GEMM BEAM Autotuner

**File:** `src/ops/ir/gpu/opencl_ir_backend.c`

The OpenCL IR backend has a built-in GEMM-specific autotuner that generates and benchmarks parameterized tile kernels. Unlike the generic BEAM search above, it is specialized for GEMM and uses OpenCL profiling events for timing.

### How It Works

At backend init, a set of parameterized `beam_gemm_N` kernels is compiled — each with different tile sizes and register blocks. On the first GEMM call for a given (M, N, K) shape, the autotuner benchmarks all applicable variants and caches the winner. Subsequent calls for the same shape use the cached variant with zero tuning overhead.

```
Backend init:
  compile beam_gemm_0 .. beam_gemm_N  (all valid tile variants)

First GEMM for shape (M, N, K):
  for each applicable variant:
    1 warmup run
    probe run → skip if >4× slower than current best
    3 timed runs via CL_PROFILING_COMMAND_START/END
    take median
  store winner in 64-entry cache keyed by (M<<40)|(N<<20)|K

Subsequent calls:
  cache hit → dispatch cached variant directly
```

### Parameter Space

| Parameter | Values | Notes |
|-----------|--------|-------|
| Tile M × N | 128×128, 128×64, 64×128, 64×64 | Larger tiles preferred first |
| K tile (TSK) | 16, 8 | |
| Register block M × N | 8×8, 8×4, 4×8, 4×4 | Higher compute density preferred first |
| SLM row padding | 0, 1 | Avoids bank conflicts |
| A layout | always transposed | Proven faster on Intel Xe |

Variants are rejected before compilation if:
- WG size > 512
- Tile loading is not evenly divisible across threads
- Register pressure (acc + a + b) > 80 floats
- SLM usage > 64 KB

A variant is skipped during autotuning if M < TSM or N < TSN or M/N/K not divisible by the tile sizes.

### Environment Variable

```bash
export CML_BEAM=0    # Disable: use static matmul / matmul_naive only
# unset or nonzero  # Enable with all compiled variants (default)
export CML_BEAM=4    # Limit search to the first 4 applicable variants
```

When disabled, the backend falls back to:
- `matmul` — for M,N multiples of 128 and K multiple of 16
- `matmul_naive` — for all other shapes

### Logging

Set log level to INFO to see autotuning progress:

```
BEAM: compiled 32 GEMM variants
BEAM: autotuning GEMM M=1024 N=1024 K=1024 (width=32, 32 variants)...
BEAM:   V0: TSM=128 TSN=128 TSK=16 reg=8x8 pad=0 -> 142.3 GFLOPS (7.55 ms)
BEAM:   V1: TSM=128 TSN=128 TSK=16 reg=8x8 pad=1 -> 139.1 GFLOPS (7.72 ms)
BEAM:   V2: TSM=128 TSN=64  TSK=16 reg=8x8 pad=0 -> SKIP (probe 31.2ms, best 7.5ms)
BEAM: WINNER V0 (TSM=128 TSN=128 TSK=16 reg=8x8 pad=0) -> 142.3 GFLOPS
```
