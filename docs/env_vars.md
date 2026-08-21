# Environment Variables

C-ML's runtime behavior is controlled by environment variables, read once at
`cml_init()` into a central flag registry (`include/core/cml_flags.h`), inspired
by tinygrad's `ContextVar`. Any flag can be dumped, queried, or overridden for a
scope from C.

Example:

```bash
DEBUG=4 NOOPT=1 ./my_program
```

Set `DEBUG=1` (or higher) to print every non-default flag at startup.

## Global flags

| Variable | Values | Default | Description |
|---|---|---|---|
| `DEBUG` | `0`–`7` | `0` | Leveled debug output (see breakdown below) |
| `NO_COLOR` | `0`/`1` | `0` | Disable ANSI color in log output |
| `NOOPT` | `0`/`1` | `0` | Disable IR/kernel optimization passes — rewrites, reordering, and operator fusion (keeps structural passes: decompose, DCE, dependency graph) |
| `PROFILE` | `0`/`1` | `0` | Enable kernel/op profiling |
| `NO_MEMORY_PLANNER` | `0`/`1` | `0` | Disable the graph memory planner |
| `WINO` | `-1`/`0`/`1` | `-1` | Winograd conv override: `-1` auto (per-op), `0` force off, `1` force on |
| `CHECK_OOB` | `0`/`1` | `0` | Enable out-of-bounds index checking in kernels |
| `VALIDATE_WITH_CPU` | `0`/`1` | `0` | Force the CPU reference execution path (disables JIT) for a trustworthy baseline |
| `CACHELEVEL` | `0`–`2` | `2` | Kernel cache level: `<2` disables the on-disk cache |
| `JIT` | `0`–`2` | `1` | JIT control: `0` off, `1`/`2` on |
| `TC` | `0`/`1` | `1` | Use tensor cores when available |
| `TC_SELECT` | `-1`+ | `-1` | Select a specific tensor-core config (`-1` auto) |
| `TC_OPT` | `0`–`2` | `0` | Tensor-core optimization aggressiveness |
| `TRANSCENDENTAL` | `0`–`2` | `1` | Transcendental ops: `2` uses a fast polynomial `exp` on the CPU reference path; `0`/`1` keep libm |
| `NOLOCALS` | `0`/`1` | `0` | Disable use of local/shared memory in kernels |
| `SPLIT_REDUCEOP` | `0`/`1` | `1` | Split large reduce ops for parallelism |
| `IGNORE_BEAM_CACHE` | `0`/`1` | `0` | Ignore the on-disk BEAM search cache |
| `FUSE_OPTIM` | `0`/`1` | `0` | Request fusing the optimizer update into the backward graph (recognized; graph-level fusion is a follow-up) |
| `MAX_BUFFER_SIZE` | bytes | `0` | Cap single buffer allocation size (`0` = unlimited) |
| `BEAM` | `#` | `0` | Number of beams in kernel beam search (`0` = disabled) |
| `DISABLE_FUSION` | `0`/`1` | `0` | Disable operator fusion in the scheduler |
| `DISABLE_JIT` | `0`/`1` | `0` | Disable JIT compilation (force interpreter/BLAS path) |
| `VIZ` | `0`/`1` | `0` | Launch the graph/kernel visualizer. Also records per-epoch weight/gradient distributions, tags IR nodes with their module scope, and enables `FLAMEGRAPH` capture. Mutually exclusive with `NO_EXPORT` |
| `NO_EXPORT` | `0`/`1` | `0` | Emit no dashboard/metrics files at all and skip the work behind them. **Mutually exclusive with `VIZ`** — setting both fails `cml_init()`. Use for benchmarking and production training |
| `FLAMEGRAPH` | `0`/`1` | `0` | Time each fused kernel and write `flamegraph.json` at exit. Implied by `VIZ` and by the `PROFILE` flag |
| `DEFAULT_FLOAT` | `FLOAT32`, `HALF`/`FLOAT16`, `BFLOAT16`, `FLOAT64` | `FLOAT32` | Default float dtype |

### `DEBUG` breakdown

| Level | Effect |
|---|---|
| `>= 1` | Raise log level to INFO; dump non-default flags; list devices |
| `>= 2` | Per-operation timing / memory / bandwidth |
| `>= 3` | Applied kernel-level optimizations (raises log level to DEBUG) |
| `>= 4` | Generated kernel code |
| `>= 5` | Intermediate representation (UOps) |
| `>= 6` | Linearized UOps |
| `>= 7` | Target assembly |

### `NOOPT` as a reference mode

Optimizations must not change results, so `NOOPT=1` is the baseline to diff
against when a number looks wrong:

```bash
./my_program > opt.txt
NOOPT=1 ./my_program > ref.txt
diff opt.txt ref.txt      # any difference is a bug in an optimization
```

The whole test suite can be run this way (`NOOPT=1 ctest --test-dir build`),
which is the cheapest way to catch a miscompiled fused kernel. To narrow a
difference further, `DISABLE_FUSION=1` turns off only operator fusion and
`JIT=0` only the JIT, so the three flags together isolate which stage is at
fault.

## Using flags from C

```c
#include "core/cml_flags.h"

if (cml_flag_enabled(CML_FLAG_NOOPT)) { /* ... */ }

int beams = cml_flag(CML_FLAG_BEAM);

/* Scoped override, mirroring tinygrad's `with Context(DEBUG=4):` */
int prev = cml_flag_push(CML_FLAG_DEBUG, 4);
/* ... verbose region ... */
cml_flag_pop(CML_FLAG_DEBUG, prev);
```

`cml_flag_was_set()` reports whether a flag was set explicitly in the environment
(vs. left at its default) — used for tri-state overrides such as `WINO`.

## Other environment variables

These are read directly by their subsystems rather than through the registry:

- **Device selection:** `BACKEND` (e.g. `interp`), `AMD_ARCH`, `USE_VULKAN`, `NAK_LIB`, `BLAS_LIB`
- **Threads:** `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `BLIS_NUM_THREADS`
- **Caching / paths:** `CACHE_DIR`, `DISK_CACHE`, `CML_EXP_DIR`
- **Fusion scheduler:** `FUSION_SCHEDULER`
- **Visualizer:** `VIZ_SCRIPT`, `VIZ_LAUNCHED`
- **Profiling:** `FLAMEGRAPH`
- **Process replay:** `PROCESS_REPLAY`, `PROCESS_REPLAY_DIR`
- **Distributed:** `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `NCCL_PORT`, `GLOO_PORT`
