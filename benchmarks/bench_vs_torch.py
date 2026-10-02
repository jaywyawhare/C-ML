#!/usr/bin/env python3
"""C-ML vs PyTorch across available devices (CPU always; CUDA when present).

GPU is the number that decides a framework claim (docs/ROADMAP.md). This box has
no device, so the CUDA section skips here — but the code runs unchanged on a
machine with a GPU, which is exactly the artifact the roadmap asks for.

Run:  cd python && PYTHONPATH=. ../benchmarks/.venv/bin/python \
          ../benchmarks/bench_vs_torch.py
"""
import statistics
import time

import numpy as np
import torch

import cml

cml.init()


def med(fn, sync, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3  # ms


def run_device(dev_name, cml_dev, torch_dev):
    print(f"\n## device: {dev_name}\n")
    print("| workload            | C-ML ms | torch ms | ratio  |")
    print("| ------------------- | ------- | -------- | ------ |")

    def to_cml(x):
        t = cml.Tensor(x)
        return t.to(cml_dev) if cml_dev is not None else t

    def to_torch(x):
        return torch.from_numpy(x).to(torch_dev)

    cml_sync = (lambda: None)  # .numpy() already forces realization + d2h
    torch_sync = (lambda: torch.cuda.synchronize()) if torch_dev == "cuda" else (lambda: None)

    def bench(label, cml_fn, torch_fn):
        c = med(cml_fn, cml_sync)
        t = med(torch_fn, torch_sync)
        print(f"| {label:<19} | {c:7.3f} | {t:8.3f} | {c / t:5.2f}x |")

    for n in (256, 512, 1024):
        a = np.random.rand(n, n).astype(np.float32)
        b = np.random.rand(n, n).astype(np.float32)
        ca, cb, ta, tb = to_cml(a), to_cml(b), to_torch(a), to_torch(b)
        bench(f"matmul {n}x{n}", lambda: (ca @ cb).numpy(), lambda: (ta @ tb).cpu().numpy())
    for n in (1 << 16, 1 << 20):
        a = np.random.rand(n).astype(np.float32)
        b = np.random.rand(n).astype(np.float32)
        ca, cb, ta, tb = to_cml(a), to_cml(b), to_torch(a), to_torch(b)
        bench(f"elementwise {n}", lambda: ((ca * cb) + ca).relu().numpy(),
              lambda: ((ta * tb) + ta).relu().cpu().numpy())


print(f"# C-ML vs PyTorch {torch.__version__}")
run_device("cpu", None, "cpu")

cml_cuda = cml.is_device_available(cml.DEVICE_CUDA)
torch_cuda = torch.cuda.is_available()
if cml_cuda and torch_cuda:
    run_device("cuda", cml.DEVICE_CUDA, "cuda")
else:
    print(f"\n[cuda skipped: cml_cuda={cml_cuda} torch_cuda={torch_cuda} — no device here]")
