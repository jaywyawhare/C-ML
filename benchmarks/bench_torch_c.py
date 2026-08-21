#!/usr/bin/env python3
"""Benchmark torch_c (PyTorch-like C API) against PyTorch."""

import os
import sys

from bench_common import (
    BENCH_GROUPS,
    bench_torch,
    fmt_ms,
    fmt_tp,
    print_host_info,
    print_results_table,
    run_c_binary,
)


def ratio(torch_ms, c_ms):
    if torch_ms is None or c_ms is None or c_ms == 0:
        return "--"
    r = c_ms / torch_ms
    if r < 1.0:
        return f"{1/r:.2f}x faster"
    return f"{r:.2f}x slower"


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    binary = os.environ.get(
        "TORCH_C_BENCH_BINARY",
        os.path.join(project_dir, "build", "bin", "bench_torch_c"),
    )

    print_host_info("torch_c vs PyTorch Benchmark")

    all_results = {}

    print("\nRunning torch_c (CPU) benchmarks...")
    all_results["torch_c"] = run_c_binary(binary, "torch_c")

    gpu = run_c_binary(binary, "torch_c", {"BACKEND": "opencl"})
    if gpu:
        print("Running torch_c (OpenCL) benchmarks...")
        all_results["torch_c(OpenCL)"] = gpu

    try:
        import torch

        print(f"\nRunning PyTorch {torch.__version__} benchmarks...")
        all_results["pytorch"] = bench_torch()
    except ImportError:
        print("\nPyTorch not available — install torch to compare.")
        all_results["pytorch"] = {}

    frameworks = [k for k in ("torch_c", "pytorch", "torch_c(OpenCL)") if k in all_results and all_results[k]]
    if not frameworks:
        print("\nNo results to display. Build with: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j")
        return 1

    pt = all_results.get("pytorch", {})
    tc = all_results.get("torch_c", {})

    print_results_table(all_results, frameworks)

    if pt and tc:
        print("\n  torch_c vs PyTorch (CPU)\n")
        for _, rows in BENCH_GROUPS:
            for label, key in rows:
                t_ms = pt.get(key)
                c_ms = tc.get(key)
                if t_ms is not None and c_ms is not None:
                    print(f"  {label:20s}  torch_c {fmt_ms(c_ms):>10s}  pytorch {fmt_ms(t_ms):>10s}  ({ratio(t_ms, c_ms)})")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
