#!/usr/bin/env python3
"""Head-to-head throughput: C-ML (via the CFFI binding) vs NumPy.

Both use an optimized BLAS for matmul, so large GEMM is expected to be a wash;
the interesting signal is small/medium sizes and elementwise where per-op
overhead and (for C-ML) graph fusion matter.

Run:  cd python && python3 ../benchmarks/bench_vs_numpy.py
      (requires the CFFI binding built: python3 cml/build_cffi.py)
"""
import time
import numpy as np

from cml.core import Tensor


def timeit(fn, iters):
    fn()  # warmup (JIT / first-touch)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def bench_matmul(n, iters):
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)
    ta, tb = Tensor.from_numpy(a), Tensor.from_numpy(b)
    flops = 2.0 * n * n * n

    np_t = timeit(lambda: a @ b, iters)
    cml_t = timeit(lambda: (ta @ tb).numpy(), iters)  # .numpy() forces execution
    return np_t, cml_t, flops


def bench_fused(n, iters):
    """relu(A@B + bias) — C-ML fuses the epilogue; NumPy does 3 passes."""
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)
    bias = np.random.randn(n, n).astype(np.float32)
    ta, tb, tbias = Tensor.from_numpy(a), Tensor.from_numpy(b), Tensor.from_numpy(bias)
    flops = 2.0 * n * n * n + 2.0 * n * n

    np_t = timeit(lambda: np.maximum(a @ b + bias, 0.0), iters)
    cml_t = timeit(lambda: ((ta @ tb) + tbias).relu().numpy()
                   if hasattr(ta, "relu") else ((ta @ tb) + tbias).numpy(), iters)
    return np_t, cml_t, flops


def gflops(t, f):
    return f / t / 1e9


def main():
    print(f"NumPy {np.__version__}\n")
    print(f"{'op':<16}{'N':>6}{'numpy GFLOPS':>15}{'C-ML GFLOPS':>14}{'ratio':>9}")
    print("-" * 60)
    for n in (256, 512, 1024):
        npt, cmt, f = bench_matmul(n, 5)
        print(f"{'matmul':<16}{n:>6}{gflops(npt,f):>15.1f}{gflops(cmt,f):>14.1f}"
              f"{gflops(cmt,f)/gflops(npt,f):>8.2f}x")
    print()
    for n in (256, 512, 1024):
        npt, cmt, f = bench_fused(n, 5)
        print(f"{'matmul+add+relu':<16}{n:>6}{gflops(npt,f):>15.1f}{gflops(cmt,f):>14.1f}"
              f"{gflops(cmt,f)/gflops(npt,f):>8.2f}x")


if __name__ == "__main__":
    main()
