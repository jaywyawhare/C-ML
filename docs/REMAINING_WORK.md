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

## 5. Deliberately still open (engine-scale or hardware-gated)

These are honest, loudly-failing stubs / fallbacks rather than silent fakes.
Each is a cross-cutting engine change with real correctness risk, or needs
hardware not present, so forcing code here would be worse than the honest
fallback:

- **`FUSE_OPTIM` graph fusion** (`src/cml.c`) — the SGD update is already emitted
  as IR (`uop_sgd_step`), but `adopt_param_data` realizes each parameter
  immediately; co-scheduling the whole backward+update graph is a fusion-
  scheduler change touching optim.c + realize + the step-reset path. Honored
  conservatively (per-step path).
- **JIT kernel recording** (`src/ops/ir/tiny_jit.c`) — the trace/replay design
  targets *compiled* kernels (`cml_kernel_fn_t(args, n, grid, block)`); CPU
  execution has no compiled kernel to record, so faithful recording would mean a
  second, replay-shaped CPU engine. Current behavior re-executes correctly.
- **`gradient_as_bucket_view`** (DDP) — aliasing grads into the bucket changes
  tensor data ownership (double-free risk) and is only exercised with
  world_size>1 (MPI); not validatable in CI. Grads are copied into buckets.
- **HEVC intra-frame decode** (`src/core/hevc.c`) — NAL parsing only; a conformant
  intra decoder (CABAC, 4×4–32×32 transforms, 35 prediction modes, deblocking,
  SAO) is decoder-scale work, out of scope for this library's focus.
- **Pipeline `interleaved` schedule** (`src/distributed/pipeline_parallel.c`) —
  no numerical effect in the synchronous single-process implementation.
- **Real CUDA/NVRTC and ROCm execution** — codegen exists but numeric validation
  needs NVIDIA/AMD hardware (none in CI); the CPU fallback covers correctness.

---

## Build & Test

```bash
cd build && cmake .. && make -j$(nproc) && ctest --output-on-failure
```

All 167 ctest suites pass.
