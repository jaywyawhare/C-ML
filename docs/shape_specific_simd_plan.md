# Migration Plan: Remove Hand-Rolled SIMD → Shape-Specialized JIT Emission

Goal: delete the hand-written SIMD library (`simd_math.c` / `simd_utils.c` /
`simd_views.c`) and make the LLVM JIT backend emit **shape-specialized**
vector code — kernels with tensor sizes and broadcast patterns baked in as
compile-time constants, so LLVM produces optimal SIMD (fixed width, full
unroll, no remainder branch, no runtime broadcast logic) instead of relying
on the hand-rolled intrinsics as the CPU fast path.

Status: **implemented** on branch `shape-specific-simd`. See "Implementation
status" below for what shipped and where it diverged from the original plan.

---

## Implementation status

**Done:**
- **Shape-specialized JIT emission** (`src/ops/ir/llvm/llvm_backend.c`): kernels
  are now keyed by `(op type + concrete shape)` via an open-addressed,
  linear-probed cache (`shape_key` / `cache_lookup`), replacing the
  direct-mapped-by-op-type cache.  Every builder bakes its sizes in as
  `LLVMConstInt` and resolves broadcasting at codegen time (`bcast_index`): the
  runtime `urem`/`select` broadcast logic is gone, and constant trip counts let
  LLVM fully vectorize/unroll with no runtime remainder branch.  Covers binary,
  unary, reduction, where, fill, gather, permute-2d, reshape, expand, matmul.
- **Numerical parity gate** (`tests/test_simd_parity.c`, wired into CTest):
  3-way check (scalar ref / executed path / hand-rolled simd) over a
  width-boundary size matrix and all broadcast modes, with an explicit tight
  jit-vs-simd parity assertion.  Passes 20/20.
- **Hand-rolled SIMD removed**: all SSE/AVX/AVX-512/NEON intrinsics, the SLEEF
  `dlopen`, and CPUID dispatch were stripped from `src/ops/simd_math.c`,
  `simd_utils.c`, `simd_views.c`, and the inline intrinsics in
  `src/backend/backend.c`.  What remains is portable scalar C (compiler
  auto-vectorized at -O3).

**Divergence from the original plan (§2–§3):** rather than *deleting* the three
`simd_*` translation units and rerouting ~37 call sites, the implementations
were **replaced with portable scalar C while preserving the public API**
(including `CMLSimdCaps`).  Rationale: the caps API and `simd_*` symbols are
referenced in ~40+ sites (interpreter, graph cache, backward, sequential, torch
eager, backends, benchmarks); preserving the API removed the hand-rolled SIMD
with zero call-site churn and far lower regression risk.  The net effect is the
same — no hand-rolled SIMD remains; the interpreter fallback is compiler-
vectorized scalar and the JIT emits the shape-specialized SIMD.

**Note on the execution path:** the default eager path is
`cpu_execute_ir → cpu_execute_node` (the interpreter), *not* the LLVM per-node
backend, which is a separate opt-in emitter (AOT / dispatch / `CML_BACKEND`).
So in the default configuration the shape-specialized SIMD is available through
the JIT backend, while the interpreter relies on compiler auto-vectorization of
the now-scalar `simd_*`.  Wiring the interpreter to the shape-specialized JIT by
default is a possible follow-up (see §6.5).

**Not done (optional, §2.3 / §6):** explicit `<W x float>` vector-IR emission
(the constant-bound auto-vectorized kernels suffice); a dedicated perf
benchmark asserting no regression (functional parity + the existing
`benchmarks/` and `test_convergence`/`test_zoo_models` cover correctness).

---

Original plan follows.

---

## 1. Current state (facts, with references)

Two parallel CPU execution paths coexist:

### Path A — hand-rolled SIMD (the current real fast path)
- `src/ops/simd_math.c` (~2.4k lines): ~30 `simd_*_f32` ops, each with
  AVX-512 / AVX2+FMA / SSE / NEON variants + SLEEF transcendentals, selected
  by runtime CPUID dispatch (`cml_get_simd_caps()`). SLEEF is `dlopen`'d at
  runtime, **not** a build dependency.
- `src/ops/simd_utils.c`: horizontal `simd_sum_float` / `simd_max_float` /
  `simd_min_float` reductions.
- `src/ops/simd_views.c`: `transpose_2d`, `gather`, `scatter`, `permute_nd`.
- Call sites (all dispatch into Path A):
  - `src/ops/ir/execution.c` `cpu_execute_node()` (`:657+`) — ~18 sites
    (ADD/SUB/MUL/DIV/NEG/EXP/LOG/SQRT/ABS/SIGMOID/TANH/SIN/COS/TAN/RECIP/POW/CMPLT/MAX/WHERE).
    Each guards `if (numel matches) simd_*() else scalar broadcast loop`.
  - `src/ops/ir/graph_cache.c` — ~11 sites (fewer guards, more aggressive).
  - `src/ops/ir/backward.c` — 2 sites (add/sub grad accumulation).
  - `src/nn/layers/sequential.c` — 2 sites (sigmoid/tanh fast path).
  - `src/torch/torch_eager.c` — 6 sites (legacy eager path).
  - `src/backend/backend.c` — its **own** duplicate inline AVX/SSE intrinsics
    (`simd_add/mul/relu/sigmoid/sum/mean/matmul`), used only when a
    BACKEND_AVX/BACKEND_SSE backend is explicitly selected.
- Build wiring: `CMakeLists.txt:304-306`.

### Path B — LLVM JIT (`src/ops/ir/llvm/llvm_backend.c`)
- Builds **one generic kernel per `UOpType`**, cached direct-mapped by
  `type % OP_CACHE_SIZE` (`:1102`, `OP_CACHE_SIZE 256` at `:27`,
  `op_cache[]` field at `:40`). **Shapes are ignored at codegen.**
- Loop trip counts + input sizes are **runtime `i64` params**:
  `build_binary_op` signature `void(ptr,ptr,ptr, i64 out_n, i64 in0_n, i64 in1_n)`
  (`:202`, params fetched `:220-222`).
- Broadcasting is a **per-element runtime** `urem`+`select` (`:233-239` binary,
  `:318-320` unary).
- Loop bound comes from the runtime param via `emit_loop(...)` (`:135`, called
  `:228`).
- Emits **scalar** IR; vectorization is delegated wholesale to LLVM
  `default<O3>` (`:997-1000`). No `LLVMVectorType` anywhere. Host CPU features
  are already enabled (`LLVMGetHostCPUName/Features`, `:71-73`), so whatever
  width LLVM chooses matches the machine.
- Transcendentals already lower to LLVM intrinsics — `llvm.exp`, `llvm.sin`,
  `llvm.sqrt`, `llvm.fabs`, `llvm.pow` (`build_unary_op` `:334-359+`).
- Falls back to `cpu_execute_node()` (Path A) when: op unsupported, a BLAS
  context is initialized (`:1097`), inputs missing, or JIT compile fails
  (`:1114, 1146, 1149, 1153`).

### The core blocker
`llvm_backend_execute(backend, node)` receives the `node` — which carries full
static shapes — but the kernel builders never read them. Kernels are keyed by
op type alone, so every shape shares one generic, runtime-bounded, scalar-loop
kernel. That is exactly why "shape-specific SIMD" does not happen today and why
the hand-rolled library is still load-bearing.

---

## 2. Target design

Make the JIT the specialized fast path; retire Path A.

1. **Shape-keyed kernel cache.** Replace the `type % OP_CACHE_SIZE` direct map
   with a small hash keyed on
   `(type, out_numel, in0_numel, in1_numel, broadcast_class, ndim, dtype)`.
   `broadcast_class` collapses the interesting cases: `equal`, `scalar (n==1)`,
   `general`. Keep a bounded LRU (e.g. 256–512 entries) so a program that sweeps
   many shapes cannot grow the cache unbounded — on eviction, drop the entry
   (JIT'd code stays owned by ORC; just stop referencing it) and recompile on
   the next miss. Log evictions so silent thrash is visible.

2. **Bake sizes as constants.** Thread the concrete sizes from `node` into the
   builders and emit `LLVMConstInt` instead of `LLVMGetParam` for
   `out_n` / `in0_n` / `in1_n` / reduction `n`. Effects, all free:
   - Constant trip count ⇒ LLVM fully vectorizes **and** unrolls, and drops the
     scalar-remainder branch when the size is a multiple of the vector width
     (and cheaply handles it when not).
   - Broadcast resolves at **compile time**: `in_n==1` becomes a hoisted
     splat, `in_n==out_n` becomes a straight indexed load — the `urem`/`select`
     at `:233-239` / `:318-320` disappears entirely.
   - No runtime CPUID dispatch: host-feature codegen (`:71-73`) already picks
     AVX2/AVX-512/NEON width.

3. **(Optional, hot ops only) emit vector IR directly.** For
   ADD/SUB/MUL/DIV/MAX/CMPLT and the cheap unaries, emit an explicit
   `<W x float>` loop (W = host width) + a constant-sized scalar tail instead of
   trusting the auto-vectorizer. Stronger guarantee; do this only if §5 shows
   the auto-vectorized constant-bound kernels miss on a target. Step 2 alone is
   ~90% of the win.

4. **Transcendentals.** Already covered by `llvm.exp`/`llvm.sin`/`llvm.tanh`
   etc. in `build_unary_op`; these vectorize under `-O3` and replace SLEEF.
   Verify accuracy in §5 (LLVM intrinsic vs SLEEF `_u10` — expect ~1 ULP diff;
   assert a tolerance, don't require bit-exactness).

5. **Views/reductions.** `build_permute_2d`, `build_gather_op`,
   `build_reduction` already exist; give them the same constant-baked
   treatment so `simd_views.c` / `simd_utils.c` have JIT replacements before
   deletion.

---

## 3. File-by-file change list

### `src/ops/ir/llvm/llvm_backend.c` (primary)
- Struct `CMLLLVMBackend` (`:32-42`): replace `op_cache[OP_CACHE_SIZE]` with a
  shape-keyed cache (open-addressed hash of `{key, kernel_fn_t}` + LRU stamp).
- Add `static uint64_t kernel_key(UOpType, const IRNode*)`.
- `build_binary_op` (`:204`), `build_unary_op` (`:292`), `build_reduction`
  (`:~530`), `build_where_op`, `build_fill_op`, `build_permute_2d`,
  `build_gather_op`, `build_reshape_op`, `build_expand_op`: add size params,
  emit `LLVMConstInt(i64, <baked>)` for bounds, and specialize the broadcast
  select at build time (branch in C on `in_n==1` / `in_n==out_n`, emit only the
  needed IR).
- Dispatch block (`:1101-1157`): look up by `kernel_key`; on miss build the
  specialized module. The call-through (`:1160-1254`) can drop the now-constant
  size args from the function signatures (kept internal), or keep passing them
  (ignored) to minimize churn — prefer dropping for clarity.

### Callers to reroute (after JIT parity proven)
- `src/ops/ir/execution.c` `cpu_execute_node()`: this becomes the scalar
  *correctness* fallback only (still needed when LLVM is compiled out via
  `CML_HAS_LLVM_BACKEND`). Replace `simd_*` calls with plain scalar loops (LLVM
  isn't guaranteed present here). It stays as the portable reference path.
- `src/ops/ir/graph_cache.c`: route through the JIT dispatch; delete `simd_*`.
- `src/ops/ir/backward.c`: replace the 2 `simd_add/sub` with scalar loops (or a
  JIT accumulate kernel).
- `src/nn/layers/sequential.c`: drop the sigmoid/tanh `simd_*` fast path; route
  through the JIT (or scalar).
- `src/torch/torch_eager.c`: replace 6 `simd_*` calls with scalar / JIT.
- `src/backend/backend.c`: remove the duplicate inline AVX/SSE intrinsics and
  the `simd_ops` table entries, OR leave BACKEND_AVX/SSE as an explicitly
  selected legacy backend (decide in §6). Its `simd_sum_float` use goes away
  with `simd_utils.c`.

### Deletions (last step)
- `src/ops/simd_math.c` + `include/ops/simd_math.h`
- `src/ops/simd_utils.c` + `include/ops/simd_utils.h`
- `src/ops/simd_views.c` + `include/ops/simd_views.h`
- `CMakeLists.txt:304-306` (source list).
- `#include "ops/simd_*.h"` in: `execution.c:14-15`, `backward.c:12`,
  `sequential.c:14`, `graph_cache.c:7-8`, `backend.c:5`, `torch_eager.c:20`,
  `cml.h:30` (`simd_views.h` — public header, check for external users first).
- Any `cml_get_simd_caps()` / `cml_print_simd_caps()` public API references.

---

## 4. Sequencing (each step independently testable)

1. Add shape-keyed cache + constant-baked bounds to `build_binary_op` /
   `build_unary_op` / `build_reduction` in `llvm_backend.c`. Keep Path A intact.
2. Build the parity + bench harness (§5). Prove JIT-specialized == Path A within
   tolerance and measure speed. **Gate.**
3. Extend baking to where/fill/permute/gather/reshape/expand.
4. Reroute `graph_cache.c`, `backward.c`, `sequential.c`, `torch_eager.c` off
   `simd_*`; convert `cpu_execute_node` `simd_*` sites to scalar reference loops.
5. Handle `backend.c` per §6 decision.
6. Delete the three `simd_*` translation units + headers + CMake wiring +
   includes. Re-run full suite + bench.

---

## 5. Correctness gate: numerical parity + benchmark harness

Build **before** deleting anything (add as `tests/test_simd_parity.c`, wire into
CTest). For each op × a shape matrix that stresses vector width boundaries:

- Sizes: `{1, 3, 7, 8, 15, 16, 17, 63, 64, 255, 256, 1024, 4096}` (hits
  aligned, misaligned, sub-width, and remainder cases for SSE/AVX2/AVX-512).
- Broadcast cases per binary op: `equal`, `lhs scalar`, `rhs scalar`.
- Inputs: fixed-seed PRNG (reuse the planned `cml_manual_seed` or a local
  xorshift), include edge values (0, ±inf-adjacent, denormals, large |x| for
  exp/log domain).
- Compare three producers of the same op: **(a)** current `simd_*`, **(b)** new
  JIT-specialized kernel, **(c)** a naive scalar reference.
- Assertions:
  - Exact ops (add/sub/mul/neg/abs/max/cmplt/where): `b == c` bitwise, and
    `a`≈`c` (already true).
  - Transcendental/inexact (exp/log/sqrt/rsqrt/sigmoid/tanh/sin/cos/tan/pow/div):
    `|b - c| <= atol + rtol*|c|` with per-op tolerances (document them). Compare
    `b` (LLVM intrinsic) vs `a` (SLEEF) and record max ULP to catch surprises.
- Benchmark: reuse `benchmarks/bench_torch_c.c` style or extend
  `tests/bench_backends.c` to time (a) vs (b) across the size matrix; assert no
  regression on the sizes that matter (large contiguous), report the small-size
  numbers (where per-call JIT dispatch overhead could show up — that's why the
  cache + a scalar threshold for tiny `numel` matter).

Existing coverage to lean on: `tests/test_numerical_ops.c`,
`tests/test_elementwise.py`, `tests/test_unary.py`, `tests/test_reductions.py`,
`tests/opcheck.c`, `tests/test_kernel_cache.c` (verify the cache-key change
doesn't break its assumptions).

---

## 6. Open decisions

1. **`backend.c` AVX/SSE backends** — delete the duplicate intrinsics entirely,
   or keep BACKEND_AVX/BACKEND_SSE as an explicitly-selected legacy path? They
   aren't the default execution path; recommend delete unless something selects
   them (grep `BACKEND_AVX`/`BACKEND_SSE` users first).
2. **`cml.h:30` exposes `simd_views.h` publicly** — removing it is an API break.
   Check examples/ and python bindings for `simd_transpose`/`gather` users; if
   any, provide a thin shim or a JIT-backed replacement before removing.
3. **Small-`numel` path** — JIT dispatch + a specialized kernel per tiny shape
   can thrash the cache and cost more than a scalar loop. Keep a
   `numel < THRESHOLD → scalar` short-circuit (the current code already has a
   `BIG_THRESHOLD` guard in `cpu_execute_node`), and quantize the cache key for
   large sizes (e.g. bucket by power-of-two ≥ some floor) so unique large shapes
   don't each compile a fresh kernel.
4. **Vector IR (step 2 auto-vec) vs explicit `<W x float>` (step 3)** — decide
   from §5 numbers per target; default to auto-vec with constant bounds.
5. **`CML_HAS_LLVM_BACKEND` off** — the scalar `cpu_execute_node` must remain a
   complete, correct fallback since the JIT can be compiled out. Don't let the
   deletion strand a no-SIMD build with no working CPU path.

---

## 7. TL;DR

The hand-rolled SIMD is still the fast path only because the JIT keys kernels by
op type and passes shapes as runtime args. **Bake the shapes in (constant bounds
+ compile-time broadcast) and key the cache by shape** — LLVM then emits the
shape-specific SIMD for free, host-width and remainder-free. Prove parity with a
3-way (simd / jit / scalar) harness across a width-boundary size matrix, then
reroute callers and delete `simd_math.c` / `simd_utils.c` / `simd_views.c`.
