#!/usr/bin/env python3
"""Benchmark C-ML against numpy (and torch / tinygrad when installed).

Measures wall-clock throughput for the workloads people actually compare
frameworks on:

  * matmul f32 at 256/1024/4096 square sizes
  * elementwise chain over a large tensor
  * full training step (forward + backward + SGD) on an MLP

Writes results to stdout as a markdown table fragment. Run from python/:
    python benchmarks/bench_vs_frameworks.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cml  # noqa: E402


def bench(fn, warmup=2, reps=5):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def gflops_matmul(n, seconds):
    return 2.0 * n ** 3 / seconds / 1e9


def cml_matmul(n):
    a_np = np.random.randn(n, n).astype(np.float32)
    b_np = np.random.randn(n, n).astype(np.float32)
    a = cml.Tensor(a_np)
    b = cml.Tensor(b_np)
    return lambda: float(np.asarray((a @ b).numpy()).sum())


def numpy_matmul(n):
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)
    return lambda: float((a @ b).sum())


def cml_elementwise(count):
    x = cml.Tensor(np.random.randn(count).astype(np.float32))
    def run():
        # compose explicitly to avoid API drift
        y = x * 2.0
        y = y + 1.0
        y = y * x
        z = cml.Tensor(y.numpy())
        return float(z.numpy().sum())
    return run


def numpy_elementwise(count):
    x = np.random.randn(count).astype(np.float32)
    def run():
        y = x * 2.0
        y = y + 1.0
        y = y * x
        return float(y.sum())
    return run


def mlp_step(lib, hidden=512, batch=128):
    """One training step of a [784 -> h -> h -> 10] MLP; returns a thunk."""
    W1 = np.random.randn(784, hidden).astype(np.float32) * 0.05
    W2 = np.random.randn(hidden, hidden).astype(np.float32) * 0.05
    W3 = np.random.randn(hidden, 10).astype(np.float32) * 0.05
    X = np.random.randn(batch, 784).astype(np.float32)
    Y = np.random.randn(batch, 10).astype(np.float32)

    if lib == "cml":
        import cml.nn as cnn
        import cml.optim as coptim
        import cml.losses as closses

        l1, l2, l3 = cnn.Linear(784, hidden), cnn.Linear(hidden, hidden), cnn.Linear(hidden, 10)
        model = cnn.Sequential(l1, l2, l3)
        opt = coptim.SGD(model, lr=0.01)

        def run():
            pred = model(cml.Tensor(X.copy()))
            loss = closses.mse_loss(pred, cml.Tensor(Y.copy()))
            loss.backward()
            opt.step()
            opt.zero_grad()
            cml.reset_graph()
            return float(np.asarray(loss.numpy()).ravel()[0])
        return run

    import torch
    import torch.nn as tnn

    class MLP(tnn.Module):
        def __init__(self):
            super().__init__()
            self.f = tnn.Sequential(
                tnn.Linear(784, hidden), tnn.ReLU(),
                tnn.Linear(hidden, hidden), tnn.ReLU(),
                tnn.Linear(hidden, 10),
            )
            with torch.no_grad():
                self.f[0].weight.copy_(torch.from_numpy(W3.T) * 0)  # shape only
                self.f[0].weight.copy_(torch.from_numpy(W1.T))
                self.f[2].weight.copy_(torch.from_numpy(W2.T))
                self.f[4].weight.copy_(torch.from_numpy(W3.T))

        def forward(self, x):
            return self.f(x)

    m = MLP()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(Y)
    crit = tnn.MSELoss()

    def run():
        opt.zero_grad()
        loss = crit(m(xt), yt)
        loss.backward()
        opt.step()
        return float(loss.item())
    return run


def main():
    print("| workload | framework | best time (s) | metric |")
    print("|---|---|---|---|")

    try:
        import torch  # noqa: F401
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False

    rows = []
    for n in (256, 1024, 4096):
        t = bench(cml_matmul(n), warmup=1, reps=3)
        rows.append((f"matmul {n}x{n} f32", "cml", t, f"{gflops_matmul(n, t):.1f} GFLOP/s"))
        t = bench(numpy_matmul(n), warmup=1, reps=3)
        rows.append((f"matmul {n}x{n} f32", "numpy", t, f"{gflops_matmul(n, t):.1f} GFLOP/s"))
        if HAS_TORCH:
            import torch
            a = torch.randn(n, n)
            b = torch.randn(n, n)
            t = bench(lambda: float((a @ b).sum()), warmup=1, reps=3)
            rows.append((f"matmul {n}x{n} f32", "torch", t, f"{gflops_matmul(n, t):.1f} GFLOP/s"))

    count = 16_000_000
    t = bench(cml_elementwise(count))
    rows.append((f"elementwise x3 @ {count//10**6}M", "cml", t, ""))
    t = bench(numpy_elementwise(count))
    rows.append((f"elementwise x3 @ {count//10**6}M", "numpy", t, ""))

    step_cml = bench(mlp_step("cml"))
    rows.append(("MLP train step (b128,h512)", "cml", step_cml, ""))
    if HAS_TORCH:
        t = bench(mlp_step("torch"))
        rows.append(("MLP train step (b128,h512)", "torch", t, ""))

    for r in rows:
        name, fw, t, metric = r
        ms = f"{t*1000:.2f} ms" if t < 10 else f"{t:.2f} s"
        print(f"| {name} | {fw} | {ms} | {metric} |")


if __name__ == "__main__":
    main()
