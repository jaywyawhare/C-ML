#!/usr/bin/env python3
"""CML half of the fair benchmark (same workloads as bench_tinygrad.py).

Tests:
  1. GEMM throughput  — square NxN matmul (N=512, 1024, 2048, 4096)
  2. Fused op         — matmul + bias_add + relu (same sizes)
  3. MLP forward pass — batch=64, 784->128->ReLU->10, 100 iters

Run:  python3 bench_cml.py   (from the repo root)
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
os.environ.setdefault("CML_LOG_LEVEL", "ERROR")

import cml

cml.init()


def now() -> float:
    return time.perf_counter()


def row(label, ms, gflops=None):
    if gflops is not None:
        print(f"{label:28s} {ms:10.3f} ms {gflops:8.2f} GFLOPS")
    else:
        print(f"{label:28s} {ms:10.3f} ms")


def bench_gemm(N, iters):
    a = cml.Tensor(np.random.rand(N, N).astype(np.float32))
    b = cml.Tensor(np.random.rand(N, N).astype(np.float32))
    for _ in range(2):
        c = a @ b
        _ = c.numpy()
    t0 = now()
    for _ in range(iters):
        c = a @ b
        _ = c.numpy()
    elapsed = now() - t0
    ms = elapsed / iters * 1e3
    gflops = 2.0 * N * N * N / (elapsed / iters) / 1e9
    return ms, gflops


def bench_fused(N, iters):
    a = cml.Tensor(np.random.rand(N, N).astype(np.float32))
    b = cml.Tensor(np.random.rand(N, N).astype(np.float32))
    bias = cml.Tensor(np.full((N,), 0.5, dtype=np.float32))
    # lazy chain: the fuser sees matmul -> add -> relu as one group
    def fused():
        return ((a @ b) + bias).relu()
    for _ in range(2):
        _ = fused().numpy()
    t0 = now()
    for _ in range(iters):
        _ = fused().numpy()
    elapsed = now() - t0
    ms = elapsed / iters * 1e3
    gflops = 2.0 * N * N * N / (elapsed / iters) / 1e9
    return ms, gflops


def bench_mlp(batch, in_f, hid, out_f, iters):
    w1 = cml.Tensor(np.random.rand(in_f, hid).astype(np.float32) * 0.05)
    w2 = cml.Tensor(np.random.rand(hid, out_f).astype(np.float32) * 0.05)
    x_np = np.ones((batch, in_f), dtype=np.float32)

    def forward(x):
        return ((x @ w1).relu() @ w2).numpy()

    for _ in range(5):
        _ = forward(cml.Tensor(x_np))
    t0 = now()
    for _ in range(iters):
        _ = forward(cml.Tensor(x_np))
    elapsed = now() - t0
    ms = elapsed / iters * 1e3
    throughput = iters * batch / elapsed
    return ms, throughput


def main():
    print(f"cml      (local build)")
    print(f"numpy    {np.__version__}")
    print(f"Python   {sys.version.split()[0]}")
    print()

    print("=" * 60)
    print("GEMM Throughput  (float32, C = A @ B, square NxN)")
    print("=" * 60)
    for n, iters in [(512, 5), (1024, 3), (2048, 2), (4096, 2)]:
        ms, gflops = bench_gemm(n, iters)
        print(f"N={n:5d}  iters={iters}  ", end="")
        row("cml matmul", ms, gflops)

    print()

    print("=" * 60)
    print("Fused  matmul + bias_add + relu  (float32, square NxN)")
    print("=" * 60)
    for n, iters in [(512, 5), (1024, 3), (2048, 2), (4096, 2)]:
        ms, gflops = bench_fused(n, iters)
        print(f"N={n:5d}  iters={iters}  ", end="")
        row("cml fused", ms, gflops)

    print()

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
