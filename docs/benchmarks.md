# Benchmarks: C-ML vs numpy / torch / tinygrad

Head-to-head timing on a single x86-64 Linux machine (f32). Numbers are
indicative only - absolute values vary between machines and runs; relative
comparisons under identical conditions are what matter. **CPU only** - this
environment has no GPU, so GPU numbers (the ones that matter most for a
framework claim) are still owed. These are a baseline, not a victory lap.

## Honesty policy

Any published claim must come from one of the committed harnesses below, on
named hardware, with warmup/repetition methodology stated. Cherry-picked
single-run numbers are worse than no numbers.

## Harnesses

- `benchmarks/bench_vs_torch_cpu.py` - C-ML vs PyTorch, each workload realized
  every iteration (C-ML is lazy - `.numpy()` forces execution; torch eager
  realizes on `.numpy()` too), 5 warmup runs, then the median of 30.
- `python/benchmarks/bench_vs_frameworks.py` - C-ML vs numpy (and torch when
  installed), best-of-5 wall clock. Reports the torch path automatically when
  present.
- `bench_tinygrad.py` + `bench_cml.py` (repo root) - C-ML vs tinygrad on
  identical workloads, sizes and iteration counts in the same environment.

Run them yourself (numbers are host-specific; re-run locally to reproduce):

```bash
cd python && ../benchmarks/.venv/bin/python cml/build_cffi.py   # build binding
PYTHONPATH=python benchmarks/.venv/bin/python benchmarks/bench_vs_torch_cpu.py
python benchmarks/bench_vs_frameworks.py
```

## C-ML vs PyTorch (PyTorch 2.11.0+cpu, 10 threads)

Ratio < 1.0 means C-ML is faster.

| workload            | C-ML ms | torch ms | ratio  | note            |
| ------------------- | ------- | -------- | ------ | --------------- |
| matmul 64x64        |   0.057 |    0.053 | 1.07x  | torch faster    |
| matmul 256x256      |   0.585 |    1.936 | 0.30x  | **C-ML faster** |
| matmul 512x512      |   3.843 |    3.897 | 0.99x  | ~tie            |
| matmul 1024x1024    |  25.566 |   21.352 | 1.20x  | torch faster    |
| elementwise 4096    |   0.083 |    0.019 | 4.37x  | torch faster    |
| elementwise 65536   |   0.110 |    0.083 | 1.32x  | torch faster    |
| elementwise 1048576 |   2.232 |    3.030 | 0.74x  | **C-ML faster** |

- **Large GEMM is a BLAS wash** - both call optimized BLAS, so 512²-1024² land
  within ~20% of each other. Expected.
- **C-ML wins mid-size matmul and large elementwise**, where its fusion + lower
  per-op overhead pay off against torch's dispatch.
- **C-ML loses on tiny tensors** (elementwise 4096): fixed per-op Python/binding
  overhead dominates when there's almost no arithmetic. This is the clearest
  CPU-side optimization target.

## C-ML vs numpy (2026-08-24, idle machine)

| workload | framework | best time | metric |
|---|---|---|---|
| matmul 256x256 f32 | cml | 3.94 ms | 8.5 GFLOP/s |
| matmul 256x256 f32 | numpy | 0.15 ms | 223.6 GFLOP/s |
| matmul 1024x1024 f32 | cml | 8.74 ms | 245.7 GFLOP/s |
| matmul 1024x1024 f32 | numpy | 8.01 ms | 268.0 GFLOP/s |
| matmul 4096x4096 f32 | cml | 668 ms | 205.6 GFLOP/s |
| matmul 4096x4096 f32 | numpy | 1105 ms | 124.3 GFLOP/s |
| elementwise chain x3 @ 16M | cml | 266 ms | |
| elementwise chain x3 @ 16M | numpy | 151 ms | |
| MLP train step (batch 128, hidden 512) | cml | 27.1 ms | |

- **Large matmul is competitive**: C-ML routes GEMM through BLAS; at 4096² it
  measured faster than this machine's numpy/OpenBLAS in the same process.
- **Small matmul is interpreter-bound**: at 256² the per-op graph dispatch
  overhead dominates (~4 ms vs numpy's 0.15 ms). The fix is the planned
  fused-schedule execution of whole subgraphs; until then don't benchmark
  tiny ops.
- **Elementwise chains** carry per-node allocation + dispatch costs; fusion
  (`UOP_FUSED_ELEMENTWISE`) recovers much of this when enabled.
- **A full training step** (forward + backward + SGD over a 3-layer MLP)
  runs at ~37 steps/s on CPU - respectable for an interpreter-driven engine,
  not yet a torch replacement.

## C-ML vs tinygrad 0.12.0 (2026-08-26, CPU, same machine)

tinygrad ran on its CPU backend with JIT; C-ML on its default CPU path
(dlopen'd OpenBLAS ILP64 via numpy's copy). Identical workloads/sizes, f32.

| workload | tinygrad | cml | verdict |
|---|---|---|---|
| GEMM 512² | 15.0 ms (17.9 GF) | **2.7 ms (100.4 GF)** | cml 5.6× |
| GEMM 1024² | 20.7 ms (103.5 GF) | 17.6 ms (122.0 GF) | cml 1.2× |
| GEMM 2048² | **71.9 ms (238.8 GF)** | 130.9 ms (131.3 GF) | tinygrad 1.8× |
| GEMM 4096² | **744.5 ms (184.6 GF)** | 1014.0 ms (135.5 GF) | tinygrad 1.4× |
| fused mm+bias+relu 512² | 17.4 ms | 12.5 ms | - (both slower than raw GEMM) |
| fused mm+bias+relu 4096² | **733.2 ms** | 1168.5 ms | tinygrad 1.6× |
| MLP fwd 784-128-10, b=64 | 12.7 ms/fwd (5.0k samp/s) | **0.75 ms/fwd (85.6k samp/s)** | cml 17× |

- **GEMM crossover**: BLAS wins through ~1024²; tinygrad's beam-tuned CPU GEMM
  pulls ahead at 2048²+. The fusion path currently *hurts* large-GEMM workloads
  vs raw matmul (extra pass-through overhead around the BLAS call) - that gap is
  the concrete target for the fused-schedule execution.
- **Small-model throughput is a clear win**: at MLP scale the per-op overhead
  dominates and C-ML's lazy graph + kernel cache is an order of magnitude faster
  than tinygrad's per-call JIT dispatch.

## Bottom line

Being roughly on par with PyTorch and ahead of tinygrad at small/mid scale on
CPU, from a from-scratch C library, is a real result. The open question a
framework claim hinges on is GPU throughput, which cannot be measured in this
environment.
