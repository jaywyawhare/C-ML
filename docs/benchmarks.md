# Benchmarks: C-ML vs numpy / torch

Measured with `python/benchmarks/bench_vs_frameworks.py` on a single x86-64
Linux machine (f32, best-of-5 wall clock). Numbers are indicative only —
absolute values vary between machines and runs; relative comparisons under
identical conditions are what matter.

Run it yourself:

```bash
cd python
python benchmarks/bench_vs_frameworks.py
```

## Results (2026-08-24, idle machine)

| workload | framework | best time | metric |
|---|---|---|---|
| matmul 256x256 f32 | cml | 3.94 ms | 8.5 GFLOP/s |
| matmul 256x256 f32 | numpy | 0.15 ms | 223.6 GFLOP/s |
| matmul 256x256 f32 | torch ¹ | — | — |
| matmul 1024x1024 f32 | cml | 8.74 ms | 245.7 GFLOP/s |
| matmul 1024x1024 f32 | numpy | 8.01 ms | 268.0 GFLOP/s |
| matmul 4096x4096 f32 | cml | 668 ms | 205.6 GFLOP/s |
| matmul 4096x4096 f32 | numpy | 1105 ms | 124.3 GFLOP/s |
| elementwise chain x3 @ 16M | cml | 266 ms | |
| elementwise chain x3 @ 16M | numpy | 151 ms | |
| MLP train step (batch 128, hidden 512) | cml | 27.1 ms | |

¹ torch was not installed on this machine; the harness includes a torch path
and reports it automatically when present.

## Results (2026-08-26, C-ML vs tinygrad 0.12.0, CPU, same machine)

Via `bench_tinygrad.py` + `bench_cml.py` (repo root) — identical workloads,
sizes and iteration counts, both frameworks in the same environment, f32.
tinygrad ran on its CPU backend with JIT; C-ML on its default CPU path
(dlopen'd OpenBLAS ILP64 via numpy's copy).

| workload | tinygrad | cml | verdict |
|---|---|---|---|
| GEMM 512² | 15.0 ms (17.9 GF) | **2.7 ms (100.4 GF)** | cml 5.6× |
| GEMM 1024² | 20.7 ms (103.5 GF) | 17.6 ms (122.0 GF) | cml 1.2× |
| GEMM 2048² | **71.9 ms (238.8 GF)** | 130.9 ms (131.3 GF) | tinygrad 1.8× |
| GEMM 4096² | **744.5 ms (184.6 GF)** | 1014.0 ms (135.5 GF) | tinygrad 1.4× |
| fused mm+bias+relu 512² | 17.4 ms | 12.5 ms | — (both slower than raw GEMM) |
| fused mm+bias+relu 4096² | **733.2 ms** | 1168.5 ms | tinygrad 1.6× |
| MLP fwd 784-128-10, b=64 | 12.7 ms/fwd (5.0k samp/s) | **0.75 ms/fwd (85.6k samp/s)** | cml 17× |

## Reading (2026-08-26)

- **GEMM crossover**: BLAS wins through ~1024²; tinygrad's beam-tuned CPU
  GEMM pulls ahead at 2048²+. The fusion path currently *hurts* large-GEMM
  workloads vs raw matmul (extra pass-through overhead around the BLAS call)
  — that gap is the concrete target for the fused-schedule execution.
- **Small-model throughput is a clear win**: at MLP scale the per-op overhead
  dominates and cml's lazy graph + kernel cache is an order of magnitude
  faster than tinygrad's per-call JIT dispatch.

## Reading

- **Large matmul is competitive**: C-ML routes GEMM through BLAS; at 4096² it
  measured faster than this machine's numpy/OpenBLAS in the same process.
- **Small matmul is interpreter-bound**: at 256² the per-op graph dispatch
  overhead dominates (~4 ms vs numpy's 0.15 ms). The fix is the planned
  fused-schedule execution of whole subgraphs; until then don't benchmark
  tiny ops.
- **Elementwise chains** carry per-node allocation + dispatch costs; fusion
  (`UOP_FUSED_ELEMENTWISE`) recovers much of this when enabled.
- **A full training step** (forward + backward + SGD over a 3-layer MLP)
  runs at ~37 steps/s on CPU — respectable for an interpreter-driven engine,
  not yet a torch replacement.

## Honesty policy

Any published claim must come from this script (or an equivalent committed
harness), on named hardware, with warmup/repetition methodology stated.
Cherry-picked single-run numbers are worse than no numbers.
