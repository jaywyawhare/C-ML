#!/usr/bin/env python3
"""
Fair benchmark: tinygrad vs CML
Same workloads, same sizes, same iteration counts.

Tests:
  1. GEMM throughput  — square NxN matmul (N=512, 1024, 2048, 4096)
  2. Fused op         — matmul + bias_add + relu (same sizes)
  3. MLP forward pass — batch=64, 784->128->ReLU->10, 100 iters
"""

import time, os, sys
import numpy as np
os.environ["CC"] = "gcc"         # tinygrad CPU JIT uses clang by default; gcc is available

from tinygrad import Tensor
from tinygrad import nn as tnn

# ── helpers ──────────────────────────────────────────────────────────────────

def now() -> float:
    return time.perf_counter()

def realize(*tensors):
    """Force execution of all tensors, then sync to GPU completion via numpy readback."""
    for t in tensors:
        t.realize()
    # .numpy() forces GPU-side completion (clFinish equivalent).
    # Without this, realize() returns after GPU submission, not completion,
    # causing wildly overstated throughput for large kernels.
    for t in tensors:
        t.numpy()

def gemm_gflops(N: int, elapsed: float, iters: int) -> float:
    return 2.0 * N**3 * iters / elapsed / 1e9

def fused_gflops(N: int, elapsed: float, iters: int) -> float:
    return (2.0 * N**3 + 2.0 * N**2) * iters / elapsed / 1e9

def row(label: str, ms: float, gflops: float):
    print(f"  {label:<40s} {ms:8.3f} ms  {gflops:8.2f} GFLOPS")

# ── 1. GEMM benchmark ─────────────────────────────────────────────────────────

def bench_gemm(N: int, iters: int = 5):
    import numpy as np
    a_np = np.random.randn(N, N).astype(np.float32)
    b_np = np.random.randn(N, N).astype(np.float32)

    # warmup — several passes to ensure JIT is compiled and stable
    for _ in range(5):
        realize(Tensor(a_np + np.float32(1e-9 * _)).matmul(Tensor(b_np)))

    t0 = now()
    for i in range(iters):
        # add tiny unique offset so tinygrad doesn't cache-hit the realized tensor
        realize(Tensor(a_np + np.float32(1e-9 * i)).matmul(Tensor(b_np)))
    elapsed = now() - t0

    ms     = elapsed / iters * 1e3
    gflops = gemm_gflops(N, elapsed, iters)
    return ms, gflops

# ── 2. Fused matmul + bias + relu ─────────────────────────────────────────────

def bench_fused(N: int, iters: int = 5):
    import numpy as np
    a_np    = np.random.randn(N, N).astype(np.float32)
    b_np    = np.random.randn(N, N).astype(np.float32)
    bias_np = np.random.randn(1, N).astype(np.float32)

    for _ in range(5):
        realize((Tensor(a_np + np.float32(1e-9 * _)).matmul(Tensor(b_np)) + Tensor(bias_np)).relu())

    t0 = now()
    for i in range(iters):
        realize((Tensor(a_np + np.float32(1e-9 * i)).matmul(Tensor(b_np)) + Tensor(bias_np)).relu())
    elapsed = now() - t0

    ms     = elapsed / iters * 1e3
    gflops = fused_gflops(N, elapsed, iters)
    return ms, gflops

# ── 3. MLP forward pass ───────────────────────────────────────────────────────

class MLP:
    def __init__(self, in_f: int, hid: int, out_f: int):
        self.l1 = tnn.Linear(in_f, hid)
        self.l2 = tnn.Linear(hid, out_f)

    def __call__(self, x: Tensor) -> Tensor:
        return self.l2(self.l1(x).relu())

def bench_mlp(batch: int = 64, in_f: int = 784, hid: int = 128,
              out_f: int = 10, iters: int = 100):
    import numpy as np
    model = MLP(in_f, hid, out_f)

    # Materialize weights once (they are fixed)
    for p in tnn.state.get_parameters(model):
        realize(p)

    x_np = np.ones((batch, in_f), dtype=np.float32)

    # warmup — fresh input tensor each call to avoid cache hits
    for _ in range(5):
        realize(model(Tensor(x_np)))

    t0 = now()
    for _ in range(iters):
        realize(model(Tensor(x_np)))
    elapsed = now() - t0

    ms         = elapsed / iters * 1e3
    throughput = iters * batch / elapsed
    return ms, throughput

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import tinygrad
    print(f"tinygrad {tinygrad.__version__ if hasattr(tinygrad,'__version__') else '0.12.0'}")
    print(f"numpy    {np.__version__}")
    print(f"Python   {sys.version.split()[0]}")
    print()

    # ── GEMM ──
    print("=" * 60)
    print("GEMM Throughput  (float32, C = A @ B, square NxN)")
    print("=" * 60)
    for N, iters in [(512, 5), (1024, 3), (2048, 2), (4096, 2)]:
        ms, gflops = bench_gemm(N, iters)
        print(f"N={N:5d}  iters={iters}  ", end="")
        row("tinygrad matmul", ms, gflops)

    print()

    # ── Fused ──
    print("=" * 60)
    print("Fused  matmul + bias_add + relu  (float32, square NxN)")
    print("=" * 60)
    for N, iters in [(512, 5), (1024, 3), (2048, 2), (4096, 2)]:
        ms, gflops = bench_fused(N, iters)
        print(f"N={N:5d}  iters={iters}  ", end="")
        row("tinygrad fused", ms, gflops)

    print()

    # ── MLP ──
    batch, in_f, hid, out_f, iters = 64, 784, 128, 10, 100
    print("=" * 60)
    print(f"MLP Forward Pass  batch={batch}  {in_f}->{hid}->ReLU->{out_f}")
    print(f"iterations={iters}")
    print("=" * 60)
    ms, throughput = bench_mlp(batch, in_f, hid, out_f, iters)
    print(f"  Time per forward:  {ms:.3f} ms")
    print(f"  Throughput:        {throughput:.1f} samples/sec")

if __name__ == "__main__":
    main()
