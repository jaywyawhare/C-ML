# C-ML — Status & Roadmap (`plan-middle`)

Snapshot of work completed and what remains to credibly claim "tinygrad / numpy / torch killer".
Branch: `shape-specific-simd`. Nothing pushed to remote.

**Current verification (all green):**
- `ctest` (regular build): **113 / 113**
- `ctest` (ASan build): **111 / 111** — whole suite memory-clean
- `grad_check`: **18 / 18 hard tests** (no more XFAIL), ASan-clean, stable across runs
- GPU exec test (`test_gpu_exec`): **6 / 6** on Intel Iris Xe (Vulkan)
- Heavy tests with `CML_USE_VULKAN=1`: grad_check 18/18, test_new_features 40/40, test_multi_dtype 15/15
- Python binding: builds + `numpy ↔ C-ML` round-trip verified

---

## ✅ DONE

### 1. Fixed the original crash cluster (segfault → clean)
The starting point was a `grad_check` segfault. Root-caused and fixed **five** distinct memory bugs, all verified with ASan:

| Fix | File | What it was |
|-----|------|-------------|
| `uop_expand` trailing-dim stride | `src/ops/uops.c` | Last broadcast dim got stride `1` instead of `0` → strided readers indexed past the source buffer |
| JIT `UOP_EXPAND` → interpreter | `src/ops/ir/llvm/llvm_backend.c` | JIT expand kernel wrote `numel` elems into a smaller aliased buffer → heap overflow corrupting adjacent memory (JIT code isn't sanitizer-instrumented) — **the actual crash** |
| Backward non-contiguous gather | `src/ops/ir/backward.c` | MUL backward read broadcast views linearly by `numel`, over-reading stride-0 storage |
| Correct free routing | `src/ops/ir/llvm/llvm_backend.c` | JIT output buffers lacked `from_buffer_cache=true` → freed via wrong allocator (header over-read) |
| Plan-cache dtype/aliasing | `src/ops/ir/graph_cache.c`, `src/ops/ir/execution.c` | Cache only owned buffers; `buffer_sizes` now in **bytes** so it's correct for every dtype (`sizeof(float)` over-read int16/f16, under-copied f64/i64) |

### 2. Fixed the intermittent conv2d hang (Heisenbug)
`conv2d_forward` built `Conv2DParams` on the stack and **never initialised `groups`**. Garbage-large `groups` slipped past the `groups < 1` guard → `channels / groups = 0` → a `M=0` degenerate GEMM in an infinite loop. Intermittent (stack garbage varies); every instrumentation suppressed it. Found deterministically with **valgrind**. One-line fix: `conv_params.groups = conv2d->groups;` (`src/nn/layers/conv2d.c`).

### 3. Autograd correctness — all 11 previously-broken gradients now correct
Promoted from `RUN_TEST_XFAIL` → hard `RUN_TEST` (regressions now fail loudly):
- **conv1d / conv2d / conv3d** — added the missing **bias gradient** (was NULL).
- **batchnorm2d / layernorm / groupnorm** — fixed the `UOP_EXPAND` backward: it used `i % numel` (wrong for `[1,C,1,1]→[N,C,H,W]`), scattering affine-param gradients into wrong positions. Now proper stride-based broadcast reduction.
- **lstm_cell** — test loss (`sum(h)+sum(c)`) was inconsistent with its `backward(h)` seed; made consistent (framework gradient was correct).
- rnn_cell / gru_cell / embedding / softmax — already cleared by the above fixes.

### 4. Whole-suite memory safety under ASan
- **Un-skipped** the numerical grad-check under ASan (the blind spot that hid the conv2d bug) — grad_check now runs the full finite-difference loop under ASan and is clean.
- Fixed **3 allocator-mismatch bugs** (`cml_free` over-reading foreign pointers):
  - nv/am GPU-mock `munmap` freed `aligned_alloc`/`posix_memalign` memory via `cml_free` (`src/ops/ir/gpu/{nv,am}_mock.c`).
  - imagenet/SQuAD/LibriSpeech loaders mixed system `malloc`/`strndup` with `cml_free` (`src/datasets/loaders.c`).
- Result: **entire suite 111/111 ASan-clean**.

### 5. GPU execution path (Vulkan) — working end-to-end
The existing Vulkan backend had solid device/buffer/dispatch infra but its **hand-rolled SPIR-V was invalid** (kernel creation failed). Replaced with real GLSL compiled by `glslc`.
- **Shaders** (`src/ops/ir/gpu/shaders/`): `binary.comp` (add/sub/mul/div/max + broadcast), `unary.comp` (relu/neg/exp/sqrt/square/sigmoid/tanh/abs/log/recip/rsqrt), `matmul.comp` (2D) → compiled to embedded SPIR-V (`include/ops/ir/gpu/vk_shaders.h`, regen via `tools/gen_vk_shaders.sh`).
- **`cml_vulkan_execute_node()`** — classifies a node, uploads inputs, dispatches the shader on GPU, downloads result; returns −1 for unsupported → CPU fallback.
- **`cml_vulkan_get_backend()`** — lazy process-wide backend.
- **Wired into `cpu_execute_ir`**: `CML_USE_VULKAN=1` routes supported f32 nodes to Vulkan with per-node CPU/JIT fallback.
- **Verified** on Iris Xe: add/mul/sub/relu/matmul/chained all match CPU exactly; default path unchanged (opt-in).

### 6. Python binding (CFFI) — built & working
Was broken (`cml._cml_lib` missing). Fixed: duplicate `tensor_data_ptr` cdef, absolute include/lib paths, missing `distributed.h`, non-PIC static-link failure (made `cml_static` `POSITION_INDEPENDENT_CODE ON` + shared LLVM via `llvm-config --link-shared`), missing `_get_lib`/`_get_ffi` accessors, stale `struct Tensor` layout (→ CFFI `...;` flexible). **Verified**: `numpy → C-ML → add/mul/matmul → numpy`.

### 7. Benchmarks (honest numbers)
- **CPU fusion** (`bench_gemm`, fixed its gated-out C-ML rows): fused matmul+bias+relu beats naive BLAS+separate-ops **3.7× → 1.2×** (N 512→2048).
- **numpy head-to-head** (`benchmarks/bench_vs_numpy.py`, via the binding): matmul **5.4×** at N=256, ~0.65× at N≥512; fused **1.06–1.29×**.

### Files touched (source, excluding build artifacts)
```
CMakeLists.txt                       (cml_static PIC)
examples/benchmarks/bench_gemm.c     (fusion bench sizes/guards)
include/ops/ir/gpu/vulkan_backend.h  (exec_node + get_backend decls)
include/ops/ir/gpu/vk_shaders.h      (NEW — embedded SPIR-V)
python/cml/_cml_cffi.py              (binding build config)
python/cml/core.py                   (_get_lib/_get_ffi)
src/datasets/loaders.c               (allocator-match frees)
src/nn/layers/conv2d.c               (groups init)
src/ops/ir/backward.c                (conv bias, expand reduce, mul gather)
src/ops/ir/execution.c               (dtype plan cache + Vulkan wiring)
src/ops/ir/gpu/am_mock.c, nv_mock.c  (free routing)
src/ops/ir/gpu/vulkan_backend.c      (exec_node, shaders, singleton)
src/ops/ir/gpu/shaders/*.comp        (NEW — GLSL)
src/ops/ir/graph_cache.c             (owns_data gate + byte sizing)
src/ops/ir/llvm/llvm_backend.c       (expand defer + free routing)
src/ops/uops.c                       (expand stride)
tests/grad_check.c                   (hard tests, ASan loop enabled, lstm loss)
tests/test_gpu_exec.c                (NEW — GPU e2e test)
tools/gen_vk_shaders.sh              (NEW — shader regen)
benchmarks/bench_vs_numpy.py         (NEW — numpy head-to-head)
```

---

## 🚧 REMAINING (the "killer" frontier)

Ordered by leverage. Each is a focused, multi-session project — not a quick patch.

### 1. GPU on-device residency  ⭐ highest leverage
Today the GPU path does an **upload → dispatch → download per node** (correctness-first). To be competitive, keep tensors resident on-device across chained ops and only sync to host on read.
- Add a device-buffer association per tensor (the `Tensor.buffer_handle` field already exists).
- Lazy host sync hook in `tensor_data_ptr` (download only when host reads).
- Boundary handling when a node falls back to CPU (download its GPU-resident inputs).
- Risk: touches the data-ownership model (already fragile) — needs careful, incremental work + ASan.

### 2. GPU backward pass
Only forward runs on GPU. Training on GPU needs the backward ops (matmul-grad, elementwise-grad, reductions) as compute shaders + wiring into `cpu_execute_backward`. Prereq for "train on GPU".

### 3. GPU op coverage
Currently: f32 elementwise (~13 ops) + 2D matmul. Missing on GPU: **conv, softmax, layernorm, reductions (sum/mean/max), gather/scatter, batched matmul**, non-f32 dtypes. Needed before real models run mostly-on-GPU.

### 4. CUDA path
This box is **Vulkan/iGPU only**. torch's home turf is CUDA. A real "torch killer" needs a working CUDA backend (the `hcq_cuda.c` / `cuda_backend.c` stubs exist but aren't wired). Requires NVIDIA hardware to validate.

### 5. Convergence + speed parity vs torch (the credibility proof)
Autograd is now correct and small models train (`test_convergence`: XOR MLP to loss<0.01, linear regression learns y=2x+1). Remaining: train a **real model (ResNet / GPT-2)** and show the **loss curve + final accuracy match torch step-for-step**, plus a **wall-clock comparison** on the same workload. torch isn't installed here — needs a proper bench harness + reference numbers.

### 6. numpy-API breadth & semantics
numpy is thousands of functions + full broadcasting/indexing/ufunc/dtype-promotion semantics. Need an honest **op-coverage matrix** and the long tail of ops real code hits (fancy indexing, `einsum`, `where`, `pad`, `roll`, advanced reductions, etc.).

### 7. Ecosystem & packaging
- Python binding builds but the **high-level `Tensor` wrapper has rough edges** (numpy auto-convert in `__init__`, `__del__` truthiness) — polish the pythonic API, add `pip install`able wheel, tests.
- Serialization/IO breadth (GGUF/SafeTensors/ONNX/.pth are claimed — verify + round-trip test).
- Optimizers/schedulers/data-loaders coverage vs torch.

### 8. Robustness at scale
The bug cluster fixed this session (uninit `groups`, buffer-cache Heisenbugs, allocator mismatches) shows the memory model is fragile under stress. Needed: **long-running training stability, larger models/batches, fuzzing**, and stress of the buffer-cache/graph-cache lifecycle.

---

## One-line status
Went from *"CPU engine with wrong gradients that segfaults"* → *"correct, memory-clean CPU trainer with a working (but slow, forward-only) GPU path and a working Python binding."*
The critical path to "killer" is now **GPU training** (residency + backward + coverage, ideally CUDA) and a **published convergence + speed head-to-head vs torch**.
