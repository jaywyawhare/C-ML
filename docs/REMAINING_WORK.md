# Remaining Work Items

## 1. GPU Backward Pass with Per-Node CPU Fallback ✅ DONE

`cml_vulkan_execute_graph()` falls back to `cpu_execute_node()` per node instead
of failing the whole graph. Implemented and tested.

---

## 2. GPU Codegen Elementwise Op Coverage ✅ DONE

The unary/binary op gaps in the GPU codegen are closed in **both** backends,
with the two op tables re-synced (PTX previously lacked TAN):

- `src/ops/ir/gpu/gpu_codegen.c` (LLVM IR): added unary GELU, QUICK_GELU,
  LEAKY_RELU, HARD_SIGMOID, HARD_TANH, RELU6, SQUARE, RSQRT, EXP2, LOG2, FLOOR,
  CEIL, SIGN; binary MINIMUM, CMPGT, CMPGE, CMPLE, CMPEQ, CMPNE.
- `src/ops/ir/gpu/ptx_codegen.c` (PTX text): same set plus TAN, using
  `float_to_ptx_hex` for the irrational activation constants so the emitted
  bit patterns match the CPU path.

Semantics (NaN handling, LEAKY_RELU/ELU fixed default slopes, tanh-approx GELU)
follow `src/ops/ir/execution_typed.c`. Coverage added in
`tests/test_ptx_codegen.c` (`test_new_unary_ops`, `test_new_binary_ops`). The
numeric `tests/test_gpu_codegen.c` cases remain hardware-gated (skip without a
CUDA/ROCm device); PTX text emission is the portable validation.

---

## 3. Other closed gaps (this pass)

- **ONNX importer** (`src/core/onnx_ops.c`): CumSum `exclusive`/`reverse`
  (composed from the inclusive scan) and Resize `align_corners` /
  `pytorch_half_pixel` coordinate modes. Tests in `test_onnx_import_ops.c`.
- **AOT** (`src/ops/ir/aot.c`): general (non-trivial) EXPAND broadcast and
  multi-dim partial reductions. Tests in `test_aot_roundtrip.c`.
- **DDP** (`src/distributed/data_parallel.c`): `find_unused_parameters` now
  reserves a zero-filled bucket slot for gradient-less params so every rank's
  bucket layout matches (the all-reduce would otherwise sum mismatched slots).
- **`FUSE_OPTIM`** (`src/optim.c`): the SGD step now emits every parameter's
  `uop_sgd_step` into the graph and realizes them in a single co-scheduled pass
  (idempotent executor => no double-applied momentum), instead of realizing each
  update on its own. `test_fuse_optim_matches_default` in `test_optim.c` trains
  identical models with and without the flag and requires the weights to match;
  the convergence suite passes under `FUSE_OPTIM=1`. Other optimizers (Adam,
  etc.) still realize per parameter.
- **Async disk I/O** (`src/backend/disk_backend.c`): real io_uring behind
  `CML_HAS_IO_URING` (CMake-detected liburing), synchronous fallback otherwise.
  Test in `test_disk_backend.c`.
- **OpenCL batched matmul** (`src/ops/ir/gpu/opencl_ir_backend.c`): GPU path via
  per-batch sub-buffer views over the reused 2D GEMM; falls back to CPU when a
  slice offset isn't device-aligned. Test in `test_opencl_ir.c`.

---

## 4. Double-backward / `create_graph` ✅ ALREADY DONE (graph mode)

Second-order gradients work under the graph autodiff engine: `tensor_backward`
threads `create_graph` into `cml_ir_grad` and skips backward-subgraph fusion so a
second pass can re-differentiate (`autodiff.c:1067`). Verified end-to-end in
`tests/test_double_backward.c` (exact `d²/dx²`, finite-diff for quadratic and
matmul, composite-VJP, and the no-create_graph-inert case). The *eager* engine
cannot support it — gradients are plain data with nothing to differentiate — and
it says so loudly (`autograd.c:319`); that is a fundamental engine property, not a
gap.

## 5. Closed in this pass

- **`gradient_as_bucket_view`** (`src/distributed/data_parallel.c`) — honored.
  Each gradient's storage is repointed at its slot in the flat bucket
  (`ddp_alias_grad_to_slot`), so the pack/unpack memcpys around the all-reduce
  and the separate per-gradient allocations both go away. The ownership hazard
  the option carries is handled explicitly rather than avoided:
  - `tensor_realize()` runs first, which both materialises a lazy gradient and
    detaches it from the IR graph — a still-attached gradient could be
    re-executed into a fresh buffer later and silently drop the alias.
  - The aliased tensor gets `owns_data = false`, so `tensor_free` leaves the
    bucket's memory alone (no double free).
  - The option implies the reserved bucket layout that `find_unused_parameters`
    requests: an alias cannot survive offsets that shift between steps.
  - `cml_ddp_free()` copies every aliased gradient back into its own allocation
    before releasing the buckets, so gradients stay valid after the wrapper dies
    (no use-after-free). It only un-aliases pointers that really fall inside its
    own buckets, so a gradient replaced since the last sync is left alone.
  - A gradient that cannot be aliased (device memory, non-float32, size
    mismatch) falls back to copying, tracked per parameter in `grad_is_view`.

  Contrary to the previous note here, this **is** validatable without MPI: the
  alias is a storage layout, not a collective, so `world_size == 1` exercises all
  of it. `tests/test_ddp_bucket_view.c` checks that each gradient's data pointer
  *is* its bucket slot, that values are bit-identical to the copying path, that
  re-syncing is stable, and that gradients survive `cml_ddp_free`. Run under
  `-DENABLE_SANITIZERS=ON` for the double-free half.

- **Pipeline `interleaved` schedule** (`src/distributed/pipeline_parallel.c`) —
  honored. `cml_pipeline_build_schedule()` emits the execution order as data and
  both `cml_pipeline_forward` and `cml_pipeline_backward` now run their units in
  it. GPipe is every forward then every backward; `interleaved` is 1F1B, built by
  replaying each stage's warmup/steady/drain program and emitting whichever
  stage's next unit has its dependencies met (backwards preferred, deepest stage
  first). Under 1F1B a micro-batch's cached activations are released the moment
  its backward reaches stage 0, which bounds live activations by the pipeline
  depth instead of the micro-batch count — so an interleaved backward consumes
  the forward's cache and cannot be re-run against it.

  Weight gradients accumulate over micro-batches, so the order cannot change the
  sum: `tests/test_pipeline_schedule.c` checks both schedules are valid
  topological orders across a range of (stages, micro-batches), that 1F1B's peak
  live-activation count is strictly lower and bounded by the depth, and that
  training through the two lands on **bitwise identical** gradients.

## 6. Deliberately still open (engine-scale or hardware-gated)

These are honest, loudly-failing stubs / fallbacks rather than silent fakes.
Each is a cross-cutting engine change with real correctness risk, or needs
hardware not present, so forcing code here would be worse than the honest
fallback:

- **JIT kernel recording** (`src/ops/ir/tiny_jit.c`) — the trace/replay design
  targets *compiled* kernels (`cml_kernel_fn_t(args, n, grid, block)`); CPU
  execution has no compiled kernel to record, so faithful recording would mean a
  second, replay-shaped CPU engine. Current behavior re-executes correctly.
- **HEVC intra-frame decode** (`src/core/hevc.c`) — NAL parsing only; a conformant
  intra decoder (CABAC, 4×4–32×32 transforms, 35 prediction modes, deblocking,
  SAO) is decoder-scale work, out of scope for this library's focus.
- **Real CUDA/NVRTC and ROCm execution** — codegen exists but numeric validation
  needs NVIDIA/AMD hardware (none in CI); the CPU fallback covers correctness.
- **Non-blocking CI legs** — the macOS matrix leg (`ci.yml`) and the Windows
  wheel build (`wheels.yml`) are `continue-on-error: true`. Promoting either to a
  hard gate needs a run on that platform to confirm it is actually green first,
  so it cannot be done from a Linux checkout.

---

## Build & Test

```bash
cd build && cmake .. && make -j$(nproc) && ctest --output-on-failure
```

All 171 ctest suites pass.
