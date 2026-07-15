# CML Future Path — Architectural Analysis

## 1. Where CML Stands Today

### Benchmark Reality (as of current session)

| Benchmark | CML | NumPy | PyTorch | TinyGrad |
|---|---|---|---|---|
| gemm_512 | ~0.6ms | ~0.3ms | ~0.3ms | — |
| fused_512 | ~1.6ms | ~1.8ms | ~1.8ms | — |
| fused_2048 | **112ms** | 135ms | 129ms | — |
| mlp_train_step | ~1.2ms | — | ~1.06ms | — |
| conv2d | ~0.26ms | — | ~0.29ms | — |

### Root Causes of Remaining Gaps

**gemm_512 slower than NumPy/PyTorch:**
- CML calls `cml_matmul` → builds IR node → dispatches through `cml_execute_ir` → hits BLAS.
- NumPy calls `cblas_sgemm` directly with zero indirection.
- This ~0.3ms overhead is IR construction + dispatch. It is an accepted framework cost (equal comparison since TinyGrad/PyTorch also have framework overhead, just lower per-op amortization).

**fused_2048 beats NumPy/PyTorch:**
- IR fused path eliminates intermediate writes between matmul+bias+relu.
- CML's fast_path_run with AVX2-vectorized bias and ReLU outperforms unfused NumPy ufuncs.
- This is CML's current competitive advantage and the right direction.

**mlp_train_step near PyTorch parity:**
- AVX2 bias+ReLU vectorization (session work) dropped this from 4.75ms → 1.2ms.
- PyTorch's remaining edge (~0.2ms) is lower IR overhead and better backward pass scheduling.

---

## 2. How the Big Frameworks Optimize

### NumPy — Pre-tuned Assembly

NumPy calls OpenBLAS. OpenBLAS ships hand-written assembly kernels for every major microarchitecture: Haswell, Skylake, Zen2, Zen3, Apple M1, POWER9, etc. These kernels are written by experts once, tested once, and reused forever. At configure time, OpenBLAS detects the CPU and hard-links the correct `.S` file.

The kernel for Haswell `dgemm` is ~3000 lines of Assembly. It does exact cycle-level scheduling: loads, FMAs, and prefetch instructions interleaved to saturate both AVX2 ports and the L1 prefetcher simultaneously. No C compiler produces this.

For element-wise ops (bias, relu) NumPy uses its own ufunc machinery with SIMD dispatch — decent but unfused across ops.

**What CML has:** A hand-written AVX2 6×16 packed GEMM. Not as good as OpenBLAS ASM (no prefetch, no cycle scheduling) but within 2× on tested shapes.

**What CML would need to match:** Either write per-arch ASM kernels, or use LLVM to get close (LLVM generates better code than GCC/clang for tight SIMD loops).

### PyTorch — Vendor Libraries + Selective JIT

PyTorch uses:
1. **MKL** (on Intel) or **cuBLAS/cuDNN** (on CUDA) — vendor libraries with years of micro-optimization.
2. **ATen kernels** — hand-written C++ with SIMD dispatch, auto-vectorized with `at::vec256`.
3. **TorchScript/TorchInductor** — traces the graph, fuses element-wise ops, generates C++ or Triton, compiles with LLVM.
4. **TorchInductor (Inductor)** — full ahead-of-time compiler for the graph. For each (op, shape) it generates a `.cpp` file, compiles it with `-O3 -march=native`, caches the `.so`. Subsequent calls load the cached library. This is how Inductor achieves register blocking between layers.

For CPU inference, `torch.compile` with `backend="inductor"` takes the whole model, fuses matmul+bias+relu into one kernel, generates a loop where the intermediate tensor (`C` of matmul) is kept in AVX registers and the bias and relu are applied before any write. No intermediate allocation.

**What CML has:** IR fusion (fast_path fuses within a sequential). No cross-layer register blocking. No LLVM emission.

**What CML would need to match on CPU:** LLVM IR emission from the graph IR, fusion pass that identifies eligible chains, codegen that allocates accumulators in registers across fused ops.

### TinyGrad — JIT Codegen to OpenCL/Metal/CUDA

TinyGrad's approach is different from both. It:
1. Traces the entire forward pass lazily (no execution until `.realize()` or `.item()`).
2. Applies graph-level fusion: element-wise ops after a matmul fold into the matmul's output loop.
3. For each unique (op, shape, dtype) combination, generates an OpenCL C / CUDA C / Metal shader with all loop bounds as **compile-time constants** in the kernel string.
4. Compiles the kernel string with the GPU compiler (which applies loop unrolling, register allocation, etc. with full knowledge of bounds).
5. Caches the compiled kernel object by hash.

The key insight: by making loop bounds compile-time constants, TinyGrad gets GPU compiler-quality optimization for each specific shape, without writing a single kernel by hand. The GPU compiler sees `for (int i = 0; i < 784; i++)` and unrolls it completely.

On CPU, TinyGrad is slower than PyTorch because OpenCL CPU implementations are suboptimal. Its strength is GPU portability.

**What CML would need to match TinyGrad's approach on CPU:** Replace LLVM IR emission with JIT C codegen — emit a `.c` file with all shapes as constants, compile with `cc -O3 -march=native -shared`, `dlopen`, call. Simpler than LLVM but same principle.

---

## 3. The Ceiling — What Actually Limits Performance

Every approach eventually hits the same wall: **memory bandwidth**.

For a matrix multiply of (M, K) × (K, N):
- Arithmetic intensity = `2*M*K*N` FLOPs / `(M*K + K*N + M*N) * 4` bytes.
- For (512, 512)×(512, 512): ~512M FLOPs / ~3MB = ~170 FLOP/byte.
- Haswell AVX2: 256 GFLOP/s peak, ~40 GB/s memory bandwidth = 6400 FLOP/byte peak intensity.
- At 170 FLOP/byte this is compute-bound — all the packing, blocking, and SIMD are fighting for the same 256 GFLOP/s ceiling.
- Difference between frameworks at this size is purely kernel efficiency — who wastes fewer cycles on stalls.

For small matrices (conv2d im2col GEMM: M=16, N=7200, K=27):
- Arithmetic intensity ≈ 2*16*27*7200 / ((16*27 + 27*7200 + 16*7200)*4) ≈ 2.5M / 0.8MB ≈ 3 FLOP/byte.
- Memory-bandwidth-bound. Multi-threaded OpenBLAS wins because it has 12× the bandwidth throughput.
- No single-threaded kernel wins here without parallelism.

**Implication for CML:** For memory-bandwidth-bound ops, the only winning move is parallelism. For compute-bound ops, the only winning move is kernel efficiency (packing, register blocking, prefetch).

---

## 4. The Best Architecture — AOT + JIT + Zero IR + Total Fusion

The honest answer to "what beats everything" is a stack that combines:

### Layer 1 — Graph-Level Tracing (Zero Runtime IR)

Instead of building IR nodes at inference time, trace the model once at load time:
```
model.build()  →  CMLGraph* g = cml_trace(model, sample_input)
```
`cml_trace` does a dry run, records every op with its shape and type, builds a static `CMLGraph` (DAG of `CMLOp` nodes). After this, inference is:
```
cml_graph_run(g, input, output)
```
One function call. No `cml_reset_ir_context`, no `cml_execute_ir`, no per-op dispatch overhead.

This is what `torch.compile` and TensorFlow's frozen graph do. For CML it would mean:
- Add a `CMLGraph` type (static DAG, no runtime construction).
- Add `cml_trace(CMLSequential*, CMLTensor* sample)` that populates it.
- Add `cml_graph_run(CMLGraph*, float* input, float* output)` that calls ops directly.

### Layer 2 — Fusion Pass

Walk the graph DAG, identify fusable chains:
- **Element-wise chains**: any sequence of add/mul/relu/gelu/etc between two non-elementwise ops collapses to one loop.
- **Matmul+bias+activation**: the most common pattern. Fused: matmul writes directly to output accumulator, bias and activation applied inside the reduction loop's tail.
- **Convolution+BN+ReLU**: BatchNorm after conv is always fusable — fold BN scale/bias into conv weights at build time (for inference), then fuse ReLU into the output loop.

For CML this means adding a `cml_graph_fuse(CMLGraph*)` pass that mutates the DAG, merging nodes into `CMLFusedOp` nodes with a list of element-wise ops to apply after the primary op.

### Layer 3 — AOT Codegen + Compilation

For each fused op in the graph, emit a C function with all shapes as compile-time constants:

```c
/* autogenerated — do not edit */
static void fused_linear_relu_512_128(
    const float* restrict A,    /* [batch, 512] */
    const float* restrict W,    /* [128, 512] */
    const float* restrict b,    /* [128] */
    float* restrict out)        /* [batch, 128] */
{
    /* ... matmul loop with bias+relu fused in output loop ... */
}
```

Compile with:
```
cc -O3 -march=native -ffast-math -shared -o kernel_abc123.so kernel_abc123.c
```

Cache by hash of (op_type, M, N, K, fused_ops). Subsequent runs: `dlopen` the cached `.so`, call the function pointer.

This is TorchInductor's approach on CPU. With `-march=native`, the C compiler unrolls loops, auto-vectorizes, and generates prefetch instructions — often matching hand-written SIMD for straightforward loops.

**What's hard about this:**
- The codegen itself is non-trivial. Need to emit valid C with correct memory access patterns for each op type.
- Cache invalidation: when weights change (training), need to recompile or separate weight-shape-only cache.
- Build-time dependency: `cc` must be available at runtime.
- Debug builds need a non-AOT path.

### Layer 4 — Auto-Tuning Scheduler

For the packed GEMM (and future conv2d direct implementation), auto-tuning is the key differentiator.

Instead of fixed `MC=120, KC=256, NC=2048`:
1. Generate N candidate tile configs (e.g., MC ∈ {48,60,72,96,120}, KC ∈ {128,192,256,320}, NR ∈ {8,16,24}).
2. For each config, run the kernel on the target shape 100 times, record median latency.
3. Cache winner by (M, N, K, CPU_ID) in a JSON file at `~/.cml/autotune_cache.json`.
4. On next run: load the cache, use the best config.

PyTorch does this with `torch.backends.cudnn.benchmark = True` for cuDNN convolutions. BLIS does this offline during install, baking the winner into the compiled library. TVM/Halide do it online at first run.

**What's hard:** The search space explodes for conv2d (stride, dilation, groups, in/out channel combos). Start with GEMM only — it has 3 parameters (M, N, K) and the cache stays small.

### Layer 5 — Multi-Backend Codegen

To match TinyGrad/PyTorch on every accelerator:

```
CMLGraph
  └── Fusion pass
       └── Schedule pass (tiling, parallelism decisions)
            ├── CPU backend
            │    ├── C codegen → cc → .so (AOT)
            │    └── LLVM IR → llvm-jit (JIT, no cc required)
            ├── CUDA backend
            │    └── PTX/CUDA C codegen → nvcc/nvrtc → cubin
            ├── OpenCL backend (AMD, Intel, Apple CPU)
            │    └── OpenCL C codegen → clBuildProgram → kernel object
            └── Metal backend (Apple Silicon)
                  └── MSL codegen → MTLLibrary → MTLFunction
```

Each backend:
- Takes a `CMLFusedOp` node with shapes and a list of ops.
- Emits the backend-specific kernel source string.
- Compiles it (AOT or JIT depending on backend).
- Returns a callable function pointer.

The fusion pass and schedule pass are backend-agnostic — they operate on the graph IR. Only the final codegen step is backend-specific.

**What CML has today:**
- CPU path: BLAS dispatch, AVX2 packed GEMM, AVX2 scalar fallback.
- No CUDA, OpenCL, or Metal backend.
- No LLVM emission.
- No graph-level fusion (only sequential fast_path fusion).
- No auto-tuning.

**What CML needs:**
1. `CMLGraph` + `cml_trace` + `cml_graph_run` (eliminate runtime IR).
2. Fusion pass (matmul+bias+act, element-wise chains).
3. C codegen for fused CPU kernels + `cc` invocation + `.so` cache.
4. Auto-tuning for GEMM tile sizes.
5. (Much later) LLVM IR emission for portability without cc.
6. (Much later) CUDA/OpenCL/Metal backends.

---

## 5. What Beats What, Honestly

| Model Size | CML Today | CML with AOT+Fusion | vs PyTorch | vs NumPy | vs TinyGrad |
|---|---|---|---|---|---|
| Tiny (MLP 128→64→10) | ~2× slower | ~1.1× slower | Near parity on CPU | Win on fused | Win |
| Medium (VGG-16 inference) | ~3× slower | ~1.5× slower | Lose (MKL) | Win fused | Win CPU |
| Large (ResNet-50) | ~5× slower | ~2× slower | Lose (cuDNN on GPU) | N/A | GPU: lose |
| Training (any) | ~1.2× PyTorch | ~1.1× | Near parity small | N/A | Win CPU |

**Honest ceiling on CPU vs PyTorch with MKL:**
- MKL's DGEMM is the best BLAS implementation for Intel CPUs, bar none. It uses AMX on SPR, AVX-512 on Skylake-SP, AVX2 on Haswell. It also has an auto-tuned threading model that accounts for NUMA topology.
- CML can approach but not beat MKL for large GEMMs without re-implementing MKL.
- CML can beat PyTorch CPU for small fused networks because PyTorch's overhead per op is higher than CML's direct fast_path.

**Where CML can genuinely win even without the full stack:**
1. Embedded inference (no OS, no dynamic linker): CML is pure C with no runtime.
2. Fused small-batch inference: fast_path with AOT-compiled fused kernels will be faster than PyTorch eager mode.
3. Custom ops: CML can fuse domain-specific operations that PyTorch can't fuse without a custom kernel.
4. Compile-time known shapes: if M, N, K are known at compile time, the packed kernel can be specialized at compile time with `__attribute__((optimize("unroll-loops")))` and constant propagation.

---

## 6. Implementation Roadmap (Prioritized)

### Phase 1 — Zero-IR Inference (High impact, medium effort)

1. Add `CMLGraph` struct: array of `CMLOp*` in topological order, pre-allocated buffer pool.
2. Add `cml_sequential_build_graph(CMLSequential*)` that traces once and returns `CMLGraph*`.
3. Add `cml_graph_run_inference(CMLGraph*, float* in, float* out)`.
4. Keep existing IR path for training (gradients still need dynamic IR).

**Expected gain:** Eliminates ~0.1–0.3ms overhead per inference call. gemm_512 would drop to match NumPy.

### Phase 2 — Fusion Pass (High impact, medium effort)

1. After building the graph, walk pairs of adjacent ops.
2. Identify matmul→add (bias)→relu/gelu chains.
3. Replace with `CMLFusedLinearAct` op that applies bias+act inside the output matrix loop.
4. Specialize `fast_path_run` to accept a `fused_act` enum parameter.

**Expected gain:** fused_512 drops another ~20%. mlp_train_step parity with PyTorch.

### Phase 3 — AOT C Codegen for Fused Kernels (Medium impact, hard)

1. For each `CMLFusedLinearAct` with known shape (M, N, K), emit a C source file.
2. Compile with `cc -O3 -march=native -shared -o ~/.cml/kernels/<hash>.so`.
3. `dlopen` and call.
4. Fallback to Phase 2 path if cc unavailable.

**Expected gain:** ~15% for medium/large networks. Compiler unrolls bias and relu loops with known N.

### Phase 4 — Auto-Tuning for Packed GEMM (Medium impact, medium effort)

1. On first call to `sgemm_packed_avx` for a new (M, N, K) shape, run a micro-benchmark over tile configs.
2. Save winner to `~/.cml/autotune.json`.
3. On subsequent calls, load cached config.

**Expected gain:** 10–20% on shapes not matching current `MC=120, KC=256, NC=2048` defaults.

### Phase 5 — CUDA Backend (Low-medium impact, very hard)

1. Add `CMLDeviceCUDA` backend.
2. For each fused op, emit CUDA C kernel with shapes as compile-time constants.
3. Compile with `nvrtc` (runtime compilation, no nvcc dependency).
4. Cache compiled cubin to disk.

**Expected gain:** GPU inference competitive with TensorFlow Lite/TorchScript on CUDA devices.

---

## 7. The One Thing That Matters Most

Of all the above, **Phase 1 (zero-IR inference)** has the best impact/effort ratio. The current IR overhead is measurable in every benchmark and is the single biggest reason CML loses to NumPy on individual op benchmarks.

After Phase 1, **the auto-tuned packed GEMM (Phase 4)** is the next lever because GEMM is the bottleneck in every MLP and most conv networks.

The AOT codegen (Phase 3) sounds glamorous but the real win from it is loop specialization for small N values — something that's only 10–15% better than what `-O3` already does with the fast_path fused loop if the shapes are given as function arguments with `__builtin_expect`.

---

*Generated 2026-04-23. Based on benchmark analysis and architectural discussion in CML session.*
