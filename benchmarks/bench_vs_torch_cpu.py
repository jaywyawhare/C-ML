#!/usr/bin/env python3
"""C-ML vs PyTorch, CPU, single-thread-ish fair timing.

Forces materialization each iteration (C-ML is lazy; .numpy() realizes it, and
torch eager is realized on .numpy() too), warms up, and reports the median of N
runs. GPU numbers are out of scope here (no device in this environment) — this is
the CPU credibility baseline (a Tier-1 item).

Run:  cd python && ../benchmarks/.venv/bin/python ../benchmarks/bench_vs_torch_cpu.py
"""
import time
import statistics
import numpy as np
import cml
import torch

cml.init()
torch.set_num_threads(max(1, torch.get_num_threads()))  # use torch's default


def med(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3  # ms


def bench_matmul(n):
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    ca, cb = cml.Tensor(a), cml.Tensor(b)
    ta, tb = torch.from_numpy(a), torch.from_numpy(b)
    cml_ms = med(lambda: (ca @ cb).numpy())
    torch_ms = med(lambda: (ta @ tb).numpy())
    return cml_ms, torch_ms


def bench_elementwise(n):
    # ((a*b)+a).relu() — a fusible chain where C-ML's fuser should shine.
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    ca, cb = cml.Tensor(a), cml.Tensor(b)
    ta, tb = torch.from_numpy(a), torch.from_numpy(b)
    cml_ms = med(lambda: ((ca * cb) + ca).relu().numpy())
    torch_ms = med(lambda: ((ta * tb) + ta).relu().numpy())
    return cml_ms, torch_ms


def row(label, cml_ms, torch_ms):
    ratio = cml_ms / torch_ms if torch_ms > 0 else float("inf")
    verdict = "C-ML faster" if ratio < 1 else "torch faster"
    print(f"| {label:<22} | {cml_ms:8.3f} | {torch_ms:8.3f} | {ratio:6.2f}x | {verdict} |")


print(f"# C-ML vs PyTorch {torch.__version__} (CPU)\n")
print(f"threads: torch={torch.get_num_threads()}\n")
print("| workload               | C-ML ms  | torch ms | ratio   | note        |")
print("| ---------------------- | -------- | -------- | ------- | ----------- |")
for n in (64, 256, 512, 1024):
    c, t = bench_matmul(n)
    row(f"matmul {n}x{n}", c, t)
for n in (1 << 12, 1 << 16, 1 << 20):
    c, t = bench_elementwise(n)
    row(f"elementwise {n}", c, t)
