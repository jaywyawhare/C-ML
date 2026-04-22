#!/usr/bin/env python3
"""Quick benchmark for all frameworks - GEMM only"""

import os
import sys
import time
import subprocess
import statistics

NTHREADS = str(os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", NTHREADS)
os.environ.setdefault("MKL_NUM_THREADS", NTHREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", NTHREADS)


def median(times):
    return sorted(times)[len(times) // 2]


def gf(N, ms):
    return 2 * N**3 / (ms / 1000) / 1e9


def run_numpy(N):
    import numpy as np

    a = np.random.randn(N, N).astype(np.float32)
    b = np.random.randn(N, N).astype(np.float32)
    for _ in range(3):
        np.dot(a, b)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(5):
            np.dot(a, b)
        times.append((time.perf_counter() - t0) / 5 * 1000)
    return median(times)


def run_torch_cpu(N):
    import torch

    torch.set_num_threads(int(NTHREADS))
    a = torch.randn(N, N)
    b = torch.randn(N, N)
    for _ in range(3):
        torch.mm(a, b)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(5):
            torch.mm(a, b)
        times.append((time.perf_counter() - t0) / 5 * 1000)
    return median(times)


def run_torch_cuda(N):
    import torch

    if not torch.cuda.is_available():
        return None
    a = torch.randn(N, N, device="cuda")
    b = torch.randn(N, N, device="cuda")
    for _ in range(3):
        torch.mm(a, b)
    torch.cuda.synchronize()

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(5):
            torch.mm(a, b)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / 5 * 1000)
    return median(times)


def run_tinygrad(N, device):
    from tinygrad import Tensor as T
    from tinygrad.engine.jit import TinyJit
    import numpy as np

    @TinyJit
    def gemm(a, b):
        return (a @ b).realize()

    a = T(np.random.randn(N, N).astype(np.float32))
    b = T(np.random.randn(N, N).astype(np.float32))
    a.realize()
    b.realize()

    for _ in range(3):
        gemm(a, b)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(5):
            out = gemm(a, b)
        out.numpy()
        times.append((time.perf_counter() - t0) / 5 * 1000)
    return median(times)


def run_cml_single(N, backend="cpu", blas_lib=None):
    env = os.environ.copy()
    if backend == "opencl":
        env["CML_BACKEND"] = "opencl"
    if blas_lib:
        env["CML_BLAS_LIB"] = blas_lib

    result = subprocess.run(
        ["./build/bin/bench_cross_framework"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )

    if result.returncode != 0:
        return None

    import json

    stdout = result.stdout
    start = stdout.find("{")
    end = stdout.rfind("}")
    try:
        data = json.loads(stdout[start : end + 1])
        return data.get(f"gemm_{N}")
    except:
        return None


def run_cml(N, backend="cpu", blas_lib=None):
    try:
        return run_cml_single(N, backend, blas_lib)
    except:
        return None


def main():
    print(f"CPU cores: {os.cpu_count()}")
    print(f"Threads: {NTHREADS}")
    print()

    results = {}

    # NumPy
    print("Running NumPy (CPU)...")
    results["NumPy(CPU)"] = {N: run_numpy(N) for N in [512, 1024, 2048]}

    # PyTorch CPU
    print("Running PyTorch (CPU)...")
    results["PyTorch(CPU)"] = {N: run_torch_cpu(N) for N in [512, 1024, 2048]}

    # PyTorch CUDA
    print("Running PyTorch (CUDA)...")
    results["PyTorch(CUDA)"] = {}
    for N in [512, 1024, 2048]:
        ms = run_torch_cuda(N)
        if ms:
            results["PyTorch(CUDA)"][N] = ms

    # TinyGrad CPU
    print("Running TinyGrad (CPU)...")
    results["TinyGrad(CPU)"] = {N: run_tinygrad(N, "CPU") for N in [512, 1024, 2048]}

    # TinyGrad GPU (OpenCL)
    print("Running TinyGrad (GPU)...")
    try:
        results["TinyGrad(GPU)"] = {
            N: run_tinygrad(N, "GPU") for N in [512, 1024, 2048]
        }
    except Exception as e:
        print(f"  TinyGrad GPU failed: {e}")

    # CML CPU - need to set BLAS lib path
    print("Running CML (CPU)...")
    venv_path = "/home/arrry/dev/personal/C-ML/benchmarks/.venv/lib/python3.14/site-packages/scipy_openblas64/lib/libscipy_openblas64_.so"
    blas_lib = os.environ.get("CML_BLAS_LIB") or venv_path
    results["CML(CPU)"] = {}
    for N in [512, 1024, 2048]:
        results["CML(CPU)"][N] = run_cml(N, "cpu", blas_lib)

    # CML OpenCL
    print("Running CML (OpenCL)...")
    results["CML(OpenCL)"] = {}
    for N in [512, 1024, 2048]:
        results["CML(OpenCL)"][N] = run_cml(N, "opencl")

    # Print table
    print("\n" + "=" * 90)
    print(f"{'Framework':<20} {'512':>15} {'1024':>15} {'2048':>15}")
    print("-" * 90)

    for fw, data in results.items():
        vals = []
        for N in [512, 1024, 2048]:
            ms = data.get(N)
            if ms is None:
                vals.append("       --   ")
            else:
                g = gf(N, ms)
                vals.append(f"{g:>6.0f} GF/s")
        print(f"{fw:<20} {vals[0]} {vals[1]} {vals[2]}")

    print("=" * 90)


if __name__ == "__main__":
    main()
