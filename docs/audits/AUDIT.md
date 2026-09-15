# C-ML Codebase Audit — Status

~110K LOC C ML library. This file is the single tracking document for audit
findings against the **current** tree.

The ten raw audit reports that used to sit beside this file (codex/opencode
reports, roadmap drafts) were removed on 2026-08-22: every finding in them is
either implemented and recorded below, or tracked under *Open work*. The
originals are recoverable from git history (`docs/audits/*.md` before this
date). A second full sweep against git history ran on 2026-08-23 — see its
section below for the items the first pass missed.

Legend: ✅ fixed · 🟡 partial · ⭕ open

---

## Original audit — 21 findings, all resolved

### Critical bugs

1. ✅ **Non-atomic global ID counter** — `src/symbolic/symbolic.c` uses
   `_Atomic int` with `atomic_fetch_add`.
2. ✅ **`pthread_once_t` manual reset UB** — `src/autograd/autograd.c` init
   uses a mutex + bool; shutdown→init cycles are well-defined.
3. ✅ **`tensor_retain_grad` semantics** — sets `t->retains_grad`, not
   `requires_grad`.
4. ✅ **Hook storage aliasing** — hooks live in dedicated `backward_hooks`
   fields on Tensor and Module, no longer stomping `user_data`.
5. ✅ **Fixed hook capacity** — hook lists grow via realloc; no silent drops.
6. ✅ **Module hook overwrite** — module hooks append to a growable list.
7. ✅ **`tensor_detach_inplace` dropping lazy compute** — materializes via
   `tensor_ensure_executed` before clearing `ir_node`.

### Architecture

8. ✅ **Single global IR context** — thread-local `tls_ir_context`
   (`src/ops/ir/context.c`).
9. ✅ **Constant folding duplicated** — centralized in `sym_fold_binop`.
10. ✅ **GPU backend fragmentation / HCQ unification** — every backend,
    including CPU, dispatches uniformly through `cml_hcq_backend_ops()`.
    ROCm (`hcq_rocm.c`, HIP-event signals) and WebGPU (`hcq_webgpu.c`)
    adapters completed the registry; pinned by `tests/test_hcq_registry.c`.
    Remaining: driver-call bodies need real hardware to validate.
11. ✅ **`rand()` in layer init** — seeded PRNG via `cml_manual_seed`.
12. ✅ **Inconsistent error propagation** — every `LOG_ERROR` lands in the
    thread-local error stack, queryable via `cml_get_last_error()`. Test:
    `tests/test_error_propagation.c`.
13. ✅ **`sym_eval` unknown-variable fallback** — returns -1.

### Performance

14. ✅ **Buffer cache fragmentation** — falls back to TLSF-backed pool.
15. ✅ **Flash attention** — chunked online-softmax path for long sequences.
16. ✅ **Kernel cache on hot path** — wired through dispatch and LLVM backend.
17. ✅ **Reduction SIMD vectorization** — portable auto-vectorizable loops +
    shape-specialized JIT kernels.

### Incomplete features

18. ✅ **Serving generation loop** — autoregressive decoding with sampling,
    EOS/max-token termination, continuous batching.
19. ✅ **Quantized matmul dispatch** — weight-only int8, plus packed affine
    int4 (8× smaller weights) and block-wise NF4. All dispatch in UOP_MATMUL
    without a dequant pass; GGUF payload convention. Tests:
    `tests/test_quant_matmul.c` (12/12), `tests/test_dtype_conformance.c`.
20. ✅ **Python boundary validation** — shape/dtype enforced at the CFFI
    boundary.
21. ✅ **Reproducibility** — `cml_manual_seed` end to end.

---

## Sweep of the full audit folder (2026-08-21)

Cross-checked all ten reports against the tree; every verdict verified by
reading the code, not taken from the reports. Items already fixed in interim
work are recorded in git history of this file.

Fixed in this pass:

- **Device surface** — `DEVICE_OPENCL` named and rejected loudly in
  `device_alloc` (OpenCL buffers live in the OpenCL IR backend); OpenCL added
  to `cml_dispatch_get_best_backend`; legacy `backend_init(BACKEND_CUDA/
  METAL/ROCM)` logs an error instead of silently running host ops;
  `BACKEND=` env routing unified through `cml_dispatch_set_from_env` in both
  `cml_ir_execute` and `cml_ir_execute_up_to`; README gained a per-backend
  maturity table.
- **IR/autograd** — backward restricted to the loss-rooted dependency
  subgraph; loss-grad seed allocated eagerly like `ensure_grad` (the lazy
  seed wrote through NULL data); DCE also roots at nodes whose output escaped
  the graph (`external_refs > 0`); node teardown consolidated into
  `cml_ir_release_node_storage` (the DCE path leaked params/scope/build_stack/
  input_shapes); `cml_dispatch_execute_on` warns on ignored I/O args.
- **Distributed/I-O** — MPI init validates launcher rank/world against
  `MPI_Comm_rank/size`; disk backend save/load is dtype-aware with short-read
  rejection and honest io_uring warning; paged attention supports batch > 1
  via `cml_paged_gqa_forward_batch(cache, seq_ids, Q, cfg)`; inert DDP and
  Pipeline config knobs labeled NOT YET HONORED; dead `bucket_ready` removed.
- **Training loop** — validation runs in eval mode with per-batch graph reset;
  `test_zoo.py` skips cleanly when the CFFI module is missing.
- Also fixed en route: half-precision matmul/conv2d silently returned zero-
  filled buffers (missing FLOAT16/BFLOAT16 arms in the generic kernels);
  CMake double-compiled nir/fusion/linearize sources.

---

## Branch-coverage campaign (2026-08-22)

Methodology: gcov instrumentation, serial ctest (parallel runs corrupt the
shared `.gcda` counters), merged with union semantics — a branch counts when
any test binary takes it — via `tools/coverage_report.sh` or
`cmake --build build-coverage --target coverage`.

New suites (159 C tests total):

- `test_backward_eager.c` — the eager backward engine (`GRAD_MODE=eager`) had
  no coverage at all. Now 48 VJPs checked against central finite differences,
  structural VJPs for conv/pool/cat/scatter/pad/roll/repeat_interleave,
  gradient accumulation, custom seeds, detached-branch pruning.
- `test_op_matrix.c` — UOP families across shape classes, dtypes, broadcast
  classes, and seven executor arms (fusion+JIT, interpreter, polynomial
  transcendentals, Winograd on/off, CHECK_OOB, NOOPT).
- `test_api_contract.c` — bad arguments across public surfaces.
- `test_file_formats.c` — corrupt/truncated CSV, IDX, tensor and optimizer
  checkpoint files; dataset lifecycle.
- `test_training_infra.c` — eight scheduler types across step boundaries,
  metrics record/export, six optimizer variants.

Real bugs these tests caught, all fixed:

1. Backward rooted at `ir->tail` instead of the requested tensor — added
   `cml_ir_execute_backward_from(ir, loss_node)`.
2. `tensor_numel_checked` divided by a zero dim (SIGFPE) and dereferenced
   NULL shapes.
3. `uop_slice` divided by a zero step (SIGFPE) and accepted negative steps,
   producing invalid strided views.
4. GATHER had no typed-executor kernel; int32-index gather failed outright.

Coverage: **59.7% line, 39.6% branch outcomes** (69,592 sites). Core `src/`
stands at 37.1% branch (~29K untaken); hardware-gated code contributes ~5K
untaken branches unreachable without physical devices.

---

## Second sweep of the audit folder (2026-08-23)

Re-extracted every finding from the nine deleted reports in git history and
diffed against this file and the tree. What the first pass missed:

Stale findings — the reports predate later work, no action needed:

- `tiny_jit.c` was flagged as completely dead; it is now **integrated** —
  capture-and-replay JIT on the execution hot path
  (`src/ops/ir/execution.c`, guarded by `g_in_jit` re-entrancy).
- `autograd_no_grad_exit()` was flagged unused; it is the public pair of
  `autograd_no_grad_enter()` and is exposed through the CFFI boundary.
- Z3 verifier duplicate definitions are resolved via `CML_HAS_Z3` ifdefs.
  The verifier itself remains an optional dev tool, never called in
  production (see Open work).

Fixed in this pass:

- **`src/nn/model_io.c` hardened** — every `fread`/`fwrite` checked (short
  I/O now fails with a logged error instead of silently corrupting), file
  version validated on load, and all file-supplied counts bounded before they
  drive allocations or loops (`name_len ≤ 4096`, `ndim ≤ 16`,
  `num_params/groups ≤ 2^20`); skip-past-corrupt-size refuses to seek beyond
  end of file.
- **`einsum` / `where` bound for Python** — `cml_where`/`cml_einsum` C
  wrappers added to `src/cml.c`; exposed as `cml.einsum`, `cml.where`, and
  `Tensor.where(cond, other)`.
- **`CML_LINK_GC_SECTIONS` CMake option** — opt-in `-ffunction-sections/
  -fdata-sections -Wl,--gc-sections` to drop unreferenced code at link time
  (the report-opencode recommendation).
- **`export_graph` demo** — root cause: when execution runs inside backward,
  the fusion pass can rewire the loss graph and detach the export root's
  own IR node, so `autograd_export_json` rejected the tensor as a "leaf".
  The exporter now falls back to the graph tail; the demo passes (rc=0).
  (The binary check_all had been sweeping was also a stale Aug-21 leftover
  from before `BUILD_EXAMPLES=ON` — examples are now built and swept live.)
- **HEVC frame decode fails loudly** — `cml_hevc_decode_iframe` used to
  return a fake blank 64×64 frame (silent corruption for any consumer); it
  now returns NULL with `CM_NOT_IMPLEMENTED`. NAL/SPS parsing remains fully
  implemented; pinned by an updated `test_hevc.c`.
- **DDP `broadcast_buffers` honored** — modules gained a generic
  non-trainable-buffer registry (`module_add_buffer`); BatchNorm registers
  its running stats, and `cml_ddp_forward` broadcasts them from rank 0 when
  world_size > 1.
- **Python long-tail bindings** — `roll`, `copysign`, `logaddexp`,
  `one_hot` bound over their existing fuzz-tested C kernels; all pinned as
  exact cases in the numpy matrix (53 cases, 0 broken).
- **Lifecycle stress suite** — new `tests/test_lifecycle_stress.c`: model
  churn with per-step resets, view-of-temporary chains over freed bases,
  and plan-cache reset pressure; green under ASAN.
- Also fixed en route: `module_init` did not zero the new buffer-registry
  fields on recycled allocator memory (shutdown segfault caught by the full
  pytest run) — and `module_create` now callocs like the tensor
  constructors.

Fixed earlier but never recorded here (verified against the tree today):

- Ring allreduce deadlock — rank-parity send/recv ordering breaks the
  blocking-send cycle once a chunk exceeds the socket buffer
  (`ring_allreduce.c`).
- NCCL bootstrap — rank 0 creates the unique id and socket-broadcasts it;
  ranks no longer each generate their own (`nccl_backend.c`).
- `cml_dist_wait` returns the work item's `error_code`; failed collectives
  no longer report success.
- `cml_dispatch_synchronize` covers every initialized HCQ backend, not just
  CUDA/ROCm/Vulkan.
- Batched matmul preserves batch dimensions in the output shape
  (`tensor_matmul`, forward_ops.c).
- `test_label_smoothing` passes (8/8); `dead_code_example` and
  `training_loop_example` run clean.

---

## Roadmap execution (2026-08-24)

First round against the "what would it take" list:

- **Fail-loudly autograd** — `CML_STRICT_GRAD=1` makes both backward engines
  (graph autodiff and eager) error out instead of silently producing no
  gradient when an op has no VJP. Default remains silent-zero for backwards
  compatibility. `test_convergence` and the parity harness pass under the
  strict flag (every op they use has a VJP).
- **Thread pool wired into execution** — `cml_init()` now starts the global
  worker pool (`CML_THREADS` to size/disable); elementwise kernels route
  through threshold-gated parallel variants (add/sub/mul/div/max, exp, neg,
  log, sqrt, abs, sigmoid, tanh, sin, cos, tan). En route, fixed two real
  bugs in the pool itself:
  - *cross-batch accounting race* — chunk completions were counted in an
    unpacked counter, so a straggler finishing batch N could release batch
    N+1's submitter before all chunks ran (UAF/hang). The done word is now
    generation-tagged like the claim word.
  - *double destroy* — teardown called destroy twice on the global pool.
  Measured: 1.7× on 16M-element exp (12 workers). Note: the default LLVM-JIT
  executor arm generates its own (serial) kernels; the parallel routing
  applies to the interpreter/fallback arms — threading the JIT codegen is
  open work.
- **CNN training parity** — new `test_cnn_training_loss_curve_parity`:
  Conv→ReLU→Flatten→Linear trained 10 steps against real torch; loss curves
  match to ~2.4e-06. Conv gradients are now pinned per-step, not just
  structurally.
- **Wheel CI** — `.github/workflows/wheels.yml`: sdist + Linux x86_64/aarch64,
  macOS arm64 and Windows wheels, each smoke-tested after install.
- **Quickstart tutorial** — `docs/tutorials/quickstart.md`.

Progress on the same list (2026-08-24):

- **Transformer training parity** — implemented as a manual single-head
  attention block (Linear projections + scaled softmax, identical weights on
  both sides). Initially xfail: it exposed a real engine bug below, since
  fixed — the test now runs strict.
- **Batched matmul** — cpu_execute_node's UOP_MATMUL fed whole buffers to
  one 2-D sgemm (correct only for batch 1); now per-slice BLAS with numpy
  broadcast rules. The JIT arm had the same flaw plus a batched-unsafe
  permute; both now fall back to the interpreter for unsupported shapes.

Done from the roadmap: thread the JIT-generated kernels (emit_loop now takes
(start, end); binary/unary/where pass runtime start/end params; bcast_index
uses global i so broadcast is correct when chunked; same-shape guards removed);
full VJP coverage for the high-value tail (decompose_gelu + UOP_LEAKY_RELU
VJPs); GPU backward pass with per-node CPU fallback in vulkan_backend.c.
Still open: GPU codegen coverage gaps (~13 unary, ~5 binary ops).

Ecosystem round (2026-08-24, later session):

- **ONNX export** — `src/core/onnx_export.c` + `python/cml/onnx.py`
  (`cml.onnx.export`). Hand-written protobuf writer (opset 11) over the
  importer's CMLONNXModel structures: elementwise binary/unary, MatMul,
  Gemm (UOP_LINEAR with transB=1), Conv/MaxPool/AveragePool, activations,
  Transpose/Reshape/Slice/Expand/Flatten/Concat/Gather/Where/Clip, and the
  Reduce* family. Ops with no ONNX equivalent fail loudly naming the UOp.
  Validated by numeric round-trip through the *independent* importer
  (`tests/test_onnx_export.c`, 17 checks): export → reparse →
  cml_onnx_run agrees with eager execution. En route, fixed the importer's
  protobuf field numbers to match the real ONNX spec (AttributeProto
  floats=7/ints=8 — was 6/7; TensorProto raw_data=9 — was 4; added
  int64_data=7), so ONNX *import* now reads spec-conformant models too.
- **Safetensors + HF weight loading** — `python/cml/safetensors.py`:
  dependency-free save/load of the safetensors format (spec-compliant
  header alignment), and `load_pretrained(model, ckpt, mapping)` that maps
  HF checkpoint key names onto C-ML parameter names with longest-prefix
  rename rules and strict-mode validation.
- **Fuzz/sanitizer CI** — `.github/workflows/fuzz.yml`: nightly ASan+UBSan
  runs of the five fuzz suites plus a release-build pytest job.
- **Benchmark harness** — `python/benchmarks/bench_vs_frameworks.py` +
  `docs/benchmarks.md`: matmul at three sizes vs numpy (BLAS-backed GEMM is
  competitive at 4096²; small sizes are interpreter-bound, honestly noted),
  elementwise chains, and a full MLP training step. Torch path auto-enables
  when torch is installed.
- **Activation checkpointing** — `cml.checkpoint(fn, *inputs)`:
  no-grad segment evaluation with an opt-in determinism verifier for
  recompute-based backward patterns.

VJP tail closure round (2026-08-24, evening):

- **Graph autodiff (`cml_ir_grad`) now covers every reachable node.** The
  post-decompose primitive set was audited op-by-op against the switch:
  explicit rules already existed for everything differentiable that survives
  `cml_ir_decompose` (which lowers SUB/DIV/SIGMOID/TANH/RELU/pools/convs
  into primitives before backward runs). The remaining reachable ops were
  classified with documented no-gradient semantics: creation ops
  (FILL/CONST/ALLOC/ARANGE/EYE/RAND_*/MESHGRID/ONE_HOT), discrete ops
  (ARGMAX/ARGMIN/ARGSORT/NONZERO), boolean/predicate ops (ALL/ANY/
  ISINF/ISNAN/ISFINITE/LOGICAL_*), integer/bitwise ops (BITWISE_*/LSHIFT/
  RSHIFT/IDIV), and in-graph optimizer steps (SGD_STEP/ADAM_STEP).
- **Real VJP for UOP_FUSED_ELEMENTWISE.** The fuser legitimately fuses
  requires_grad chains during training (tensor_backward sets allow_grad),
  so fused nodes DO reach the backward walk. The chain is differentiated
  step-by-step: forward values are recomputed as primitive uops, then local
  partials are accumulated backwards over the recorded ops (ADD/SUB/MUL/
  DIV/MAX/MINIMUM/POW/unaries/comparisons/WHERE/FILL — exactly the closed
  set fused_eval_block supports). XOR-MLP convergence now passes under
  CML_STRICT_GRAD=1 (it previously errored).
- **Eager engine (cpu_backward_node) tail classified + adjoints added.**
  Same no-gradient classification; real VJPs added for MASKED_FILL,
  SCATTER_ADD (gather adjoint), IM2COL (col2im), COL2IM (im2col), FOLD
  (unfold adjoint).
- **Dtype-aware index reading** — GATHER/SCATTER_ADD read index tensors via
  `cml_index_read()` instead of casting every buffer through `float*`.
  INT32 indices (embedding!) were reinterpreted as denormal floats and cast
  to 0, silently gathering row 0 and scattering gradients to row 0.
  The reading path is correct for all dtypes.

Embedding gradient — **FIXED (2026-08-25, root cause below)**:

- `grad_check` embedding reported analytical grad = 0 where numerical ≈ 1.
  The previous session's note ("scatter_add node built but never dispatched")
  was a red herring: the typed executor DOES claim the node
  (`cml_exec_needs_typed` routes it to `execution_typed.c`, not the f32
  interpreter switch), runs the correct kernel with correct params and inputs,
  and marks it executed. The kernel's writes were landing in an INT32 buffer.
  **Root cause:** `tensor_from_ir_node` inherits the output tensor's dtype from
  `inputs[0]`. For UOP_SCATTER_ADD, inputs[0] is the *index* tensor (INT32),
  so the gradient itself was allocated as INT32; float gradients stored into it
  read back as denormal garbage ≈ 0. Fix: role-based dtype — SCATTER_ADD takes
  the dtype of its data operand (`inputs[1]`). The only op in the uop set whose
  first input is not the data operand (SCATTER is {a,index,src}, GATHER is
  {input,idx}, WHERE decomposes to arithmetic). Note this also means the
  cross-entropy path only ever passed by accident: its label indices are
  float32. Regression-pinned by
  `test_autodiff_ops.c::test_gather_grad_dtype_and_value` (verified to fail
  when the fix is disabled) and `grad_check` embedding now passes 18/18.

Threaded JIT dispatch (2026-08-24, evening):

- **Same-shape elementwise JIT kernels now run on the threadpool.** Binary,
  unary, FILL and WHERE JIT kernels are chunked through
  `threadpool_parallel_for` above 65536 elements. Correctness constraint,
  documented in llvm_backend.c: broadcast operands are indexed with
  `j % in_numel` over the GLOBAL element index, so chunked invocation is
  only exact when every operand numel equals the output numel — broadcast /
  reduction / matmul / fused-chain kernels stay single-threaded until
  codegen learns a global-offset parameter. Verified: test_jit_dtype 5/5,
  test_multi_dtype 15/15, test_simd_parity 20/20, JIT=1 path-equivalence
  0 mismatched, convergence + autodiff suites green under JIT=1.

Known flake (noted 2026-08-24): `test_bindings.py::test_cross_entropy_gradient_matches_analytic`
fails intermittently in full-suite runs (shared global IR state across test
files) but passes standalone — same class as the pre-2026-08-23 teardown
flakes; re-open only if it reproduces reliably. Seen once more on 2026-08-25
(garbage row of magnitude ~1e19 — uninitialized buffer read, not zero grad);
4/4 clean full-suite runs after rebuilding the cffi bindings.

---

## Capability push (2026-08-26)

Gap analysis vs PyTorch/TF/tinygrad drove five work items; all landed with
new pinned test suites. Full suite: 165/165 ctest.

1. ✅ **Double-backward (`create_graph`)** — the graph engine's VJPs are
   already lazy UOP nodes, so `tensor_backward(..., create_graph=true)` now
   publishes them differentiable and a second pass re-differentiates them.
   Enabling this exposed three latent engine bugs, all fixed:
   - `cml_ir_new` left the new `has_backward_nodes`/`decomposed_frontier`
     fields uninitialized on recycled allocator blocks — a fresh context
     inherited the previous context's "backward present" state and skipped
     its forward decompose (null grads for any composite op after any
     backward in the same process).
   - Whole-graph re-decompose on a second grad pass corrupts backward nodes
     (known hazard); decompose now tracks a per-context frontier and
     `cml_ir_decompose_from` lowers only ops appended between passes.
   - Pass-1's execute fused the backward chain into single
     UOP_FUSED_ELEMENTWISE nodes, rewriting node identities out from under
     pass 2; `create_graph` now executes without fusion.
   Also: composites that occur *inside* backward subgraphs got direct VJP
   rules (SUB/DIV/SQUARE/COS/TAN/ABS/RELU/SIGMOID/TANH/RSQRT/MINIMUM), and
   a stale-grad sweep replaces differentiable grads that a later pass never
   reached with explicit zeros (previously they silently kept the previous
   root's gradient). Without `create_graph`, published grads are inert
   (`requires_grad=false`), matching torch's contract.
   Tests: `tests/test_double_backward.c` (exact + finite-difference through
   quad/matmul/quick-gelu chains, runs clean under CML_STRICT_GRAD=1).

2. ✅ **bf16 autocast** — `autocast_set_dtype` / `CML_AMP_DTYPE=bf16`;
   GradScaler no-ops under bf16 (fp32 exponent range). Two real bugs fixed
   en route: the graph-engine loss seed was hardcoded fp32 (zeroed grads of
   half/bf16 graphs) and `decompose_mean` promoted every downstream VJP to
   fp32 via an fp32 `1/n` constant. Test: `tests/test_amp_bf16.c`.

3. ✅ **ONNX import breadth** — 36 → 64 supported ops (Where, Expand, Tile,
   Range, CumSum, ScatterND, Resize nearest/linear, Reduce{Min,Max,Prod,Sum,
   Mean}, ArgMin/Max, Erf, LeakyRelu, PReLU, Softplus, Split, variadic
   Min/Max/Sum/Mean, Pow, Reciprocal, Floor, Ceil, Round, Sign + Gemm fix).
   RNN/LSTM/GRU still rejected loudly. Tests: `tests/test_onnx_import_ops.c`
   (55 checks, hand-assembled protobufs).

4. ✅ **Chrome Trace profiler export** — `cml_flame_export_chrome_trace(path)`
   writes the recorded timeline as Trace Event JSON for chrome://tracing /
   Perfetto. Test: `tests/test_chrome_trace.c`.

5. ✅ **QAT fake-quant** — min-max + EMA observers, symmetric-int8 params,
   fake-quant composed from existing uops; STE via detach-based surrogate
   (out = t + stop_grad(fake_quant(t) − t)) rather than changing ROUND's
   zero-VJP globally. Test: `tests/test_qat.c`.

6. ✅ **Sparse spMM autograd** — spMM was pure eager with NO gradient path;
   now records a pre-executed UOP_SPMM node when either operand requires
   grad, with dense-grad semantics for values in BOTH engines (eager kernel
   + primitive-composed graph VJP). Tests: `tests/test_sparse_autograd.c`.

7. ✅ **tinygrad parity benchmark** — `bench_tinygrad.py` had no CML half;
   added `bench_cml.py` and recorded honest mixed results in
   `docs/benchmarks.md` (BLAS wins ≤1024², tinygrad's tuned GEMM wins ≥2048²,
   cml 17× on small-model throughput). torch unavailable on this machine.

Still open: real-GPU validation of the dlopen/mock driver paths (no
hardware here), ONNX LSTM/GRU import, complex dtypes, sparse autograd
beyond COO-spMM.

---

## Open work

Engine bug from 2026-08-24 — **FIXED same day** (root cause below):

- **Batched matmul in multi-node graphs loses values/gradients.** A lone
  batched mm computed correctly; in a larger graph (attention-style
  linear→reshape→bmm→div chains) earlier nodes' outputs came back zeroed,
  so attention blocks produced flat loss. It failed identically with JIT
  and fusion disabled. Root cause: NOT the plan cache or batched mm at
  all — `cml_ir_execute_up_to` pre-marked `target_node->is_used = true`
  (a leftover from when the function always ran the full graph), and the
  partial-execution DCE walk seeds its stack with the target only if it
  is not yet marked. Result: nothing upstream was ever marked or
  executed; the target node executed alone on unwritten inputs, failed,
  and `tensor_data_ptr` handed back its zero-filled allocation as a
  valid result. Any mid-graph lazy realization whose target was not the
  graph tail silently produced zeros — which is also why gradients
  through multi-node attention subgraphs vanished. Fix: don't pre-mark;
  let the DFS seed itself (`execution.c`). Minimal repro (two
  `[B,S,d]@[B,d,S]` chains sharing inputs, realized mid-graph) returns
  exact values for both; transformer block training loss now decreases
  step-over-step instead of flat. Regression-pinned by
  `test_lazy_eval.c::test_midgraph_realization_executes_upstream` (verified
  to fail when the bug is re-introduced).

- `grad_check` embedding (analytical grad 0 where numerical ≈ 1, 17/18) —
  **FIXED 2026-08-25**, see "Embedding gradient" above: the gradient buffer
  was INT32 because SCATTER_ADD inherited its dtype from the index tensor.

## Global-offset JIT threading (2026-08-27)

Two changes landed together:

- **`emit_loop` now takes (start, end) instead of just `n`** in
  `llvm_backend.c`.  Binary, unary and where kernel builders accept
  `start, end` as runtime i64 parameters; the loop iterates `i ∈ [start, end)`
  and writes to `out[i - start]`.  `bcast_index(i, in_numel, out_numel)`
  operates on the global `i`, so broadcast operands are correct even when
  the loop is chunked across threads.

- **Same-shape guards removed from `jb_run`/`ju_run`/`jw_run`.**  All three
  now unconditionally thread when `n ≥ JIT_PARALLEL_MIN_ELEMS`; the old
  `na == n && nb == n` (binary) / `na == n` (unary) / `nc == n && na == n && nb == n`
  (where) constraints are no longer needed.

Non-threaded kernel builders (fill, gather, expand, reshape, fused_elementwise)
are unchanged — they pass `start=0, end=n` as constants.

- `decompose_gelu()` added to `decompose.c`: `GELU → x * sigmoid(1.702 * x)`,
  giving free VJPs via MUL+SIGMOID.
- `UOP_LEAKY_RELU` VPI added in `autodiff.c` (symbolic) and `backward.c`
  (eager): `d/dx = x > 0 ? 1 : slope` via `ClampParams->min_val`.
- Tests for both added in `test_autodiff_ops.c`.

Full suite: 167/167 ctest.

## GPU backward pass per-node CPU fallback (2026-08-27)

`cml_vulkan_execute_graph()` in `vulkan_backend.c:1121` now falls back to
`cpu_execute_node()` on a per-node basis instead of failing the entire graph.
When `cml_vulkan_execute_node()` returns non-zero (unsupported op), the node
executes on CPU and the graph continues.

```c
if (cml_vulkan_execute_node(backend, node) != 0) {
    LOG_DEBUG("Vulkan: node op=%d unsupported, falling back to CPU", node->type);
    if (cpu_execute_node(node) != 0)
        return -1;
}
```

This unblocks mixed CPU/GPU execution for graphs containing ops not yet
covered by SPIR-V shaders (e.g., GELU, LEAKY_RELU, HARD_SIGMOID, etc.).

Hardware-gated (cannot be validated here):

- On-GPU validation of HCQ adapter driver bodies: async ordering, buffer
  mapping, peer copies.
- GPU residency across chained ops, GPU backward pass, broader GPU op
  coverage.
- Shape-constant CUDA kernels via NVRTC with cached cubins; schedule-v2 GPU
  dispatch.

Mock hardware drivers (added 2026-08-23; no physical device needed):

- **HIP mock** (`src/ops/ir/gpu/hip_mock.c`) — implements the entire HIP
  function-pointer surface the ROCm backend consumes (per the public ROCm
  API signatures in `gpu/rocm_backend.h`): host-backed device memory with
  allocation tracking, direction-correct memcpy, streams/events, and an
  ordered operation journal. `CML_HIP_MOCK=1` routes
  `cml_rocm_backend_init()` to it; `tests/test_hip_mock.c` validates the
  HCQ adapter end to end (H2D -> launch -> D2H ordering, payload integrity,
  signal record/wait lifecycle, leak-free teardown) under ASAN. Kernel
 *launches are journaled with geometry rather than executed — code objects
  cannot run on a CPU.
- **FastRPC mock** (`tools/fastrpc_mock.c`, built as `libfastrpc_mock.so`) —
  implements Qualcomm's three-symbol FastRPC transport
  (`remote_handle_open/invoke/close`) that `hexagon_backend.c` dlsym's;
  selected via the new `CML_DSP_RPC_LIB` override.
  `tests/test_hexagon_rpc.c` drives the full DSP session lifecycle
  (open -> per-node invoke -> close, no leaked handles). The HVX ISA itself
  was already covered by the existing hexagon_sim interpreter.

Still hardware-only: numerical correctness of vendor kernels, peer-to-peer
copies, multi-GPU topologies.

Robustness at scale (partially addressed 2026-08-23):

- `tests/test_lifecycle_stress.c` now covers model churn with per-step
  resets, view-of-temporary chains, and plan-cache reset pressure under
  ASAN. Still open: long-running training soak, larger models/batches.

Python long-tail API:

- Bound as of 2026-08-23: `einsum`, `where`, `roll`, `copysign`,
  `logaddexp`, `one_hot`. Still open: fancy indexing, `pad` binding,
  advanced reductions, dtype-promotion rules.

Optional tooling (deliberate status quo):

- Z3 verifier compiles only when `CML_HAS_Z3` is defined and is never wired
  into production paths; keep as a dev tool or remove if it rots.

Feature-scale status (implemented 2026-08-23 unless noted):

- **future-path roadmap disposition** — Phase 1 zero-IR inference exists via
  the Sequential static-graph fast path + trace cache; Phase 2 fusion pass
  landed (`fuse_elementwise.c`, 11+ patterns); Phase 3 AOT codegen landed
  (`src/ops/ir/aot.c`, dlopen cache with hash-keyed artifacts); Phase 4
  auto-tuning is the `CML_AUTOTUNE` entry below; Phase 5 CUDA/NVRTC
  remains hardware-gated open work.

- **Multi-dtype BLAS** — f64 2-D matmul routes through cblas_dgemm when the
  BLAS library provides it; other dtypes keep the native generic kernels.
- **GEMM tile auto-tuning** — with `CML_AUTOTUNE=1`, the packed kernel's
  MC/KC/NC blocking is measured over a candidate set per (M,N,K) and cached
  in `~/.cml/autotune.json`; later runs load it. Off by default.- **numpy-API breadth matrix** — `python/tests/test_numpy_matrix.py` checks
  49 op/constructor cases against numpy: exact everywhere except 7
  shape-convention cases, 0 wrong, 0 known-issues (the view-dangling
  exclusions were removed after the view-lifetime fix below; `einsum` and
  `where` joined on 2026-08-23).
- **PyTorch parity harness** — `python/tests/test_torch_parity.py`. Forward
  and 20-step training parity both PASS against real torch (max loss diff
  ~2e-07); the skip marker was removed.
- **pip packaging** — `pip install ./python` works from a clean venv:
  build_cffi now auto-builds the C library via CMake when missing, imports
  its spec regardless of cwd, and wheel package-data includes the compiled
  `_cml_lib.so`; version unified on pyproject.toml.
- **Viz exporter** — total_params was 0 for every container model because
  parameter collection walked Module->next while Sequential keeps children
  in its own array; collection now counts containers. Op names were already
  fully mapped.

Formerly open bugs (both FIXED 2026-08-23):

1. **View lifetime** (was: reshape/flatten/squeeze views are non-owning and
   dangled once a temporary base hit the buffer cache) — fixed with shared
   storage ownership. `CMLTensorStorage` is attached lazily to the owning
   tensor the first time a view is taken; every view holds a reference, and
   the data block is freed only when the last referencing tensor dies
   (`tensor_storage_share` / `tensor_storage_release` in tensor.c, wired into
   tensor_reshape / tensor_as_strided / tensor_free / realize paths). Tensors
   whose root does not own an allocation (borrowed plan buffers) keep the old
   documented non-owning contract. Also fixed en route: `tensor_from_ir_node`
   and `tensor_create` used `malloc` on recycled allocator memory, leaving
   `saved_ir_node`/`storage` poisoned; `tensor_free` now nulls `data` after
   freeing so a second free cannot return the same block to the buffer cache
   twice.
2. **Multi-step IR lifecycle** (was: reset after forward/backward/step
   corrupted the heap) — two causes, both fixed:
   - Python wrappers pinned module-owned tensors on *every* access
     (`Parameter.tensor`, `.grad`), but release dropped one ref_count per
     wrap/unwrap cycle — repeatedly reading a parameter progressively freed
     it mid-training. Borrowed handles now wrap without pinning
     (`Tensor.borrow`, used by nn.py Parameter views).
   - `cml_ir_free` phases 1/1b force-freed node outputs even when an external
     owner still held them; they now detach-and-keep (`tensor_detach_keep`)
     any output with `external_refs > 0`.

Both fixes verified under ASAN (C repro of Sequential+SGD multi-step training,
and the torch parity harness run against an ASAN-instrumented binding).

Also noted (2026-08-22): an interpreter-shutdown double-free appeared after
some pytest runs. **No longer reproduces** as of the 2026-08-23 teardown/pin
fixes — full pytest runs exit 0 cleanly; re-open only if it resurfaces.

Coverage campaign (ordered by untaken branches; figures dated 2026-08-22,
before the view-lifetime and lifecycle fixes — run
`tools/coverage_report.sh build-coverage` for the live ranking): backward.c
(1,932), execution.c (1,957), uops.c (976), decompose.c (863),
execution_typed.c (860), optim.c (447), dataset.c (481), dispatch.c (611).
Each focused round adds roughly one point of overall branch coverage and has
surfaced one or two real bugs per round.

Test suite at last count: 167 ctest suites (222 binaries incl. examples,
check_all 177 PASS / 0 FAIL), 25 passed Python tests (0 skipped with torch
installed; the parity harness skips cleanly without it), zero compiler
warnings.
