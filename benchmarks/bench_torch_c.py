#!/usr/bin/env python3
"""Benchmark torch_c (PyTorch-like C API) against PyTorch."""

import json
import os
import statistics
import subprocess
import sys
import time

_NTHREADS = str(os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", _NTHREADS)
os.environ.setdefault("MKL_NUM_THREADS", _NTHREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NTHREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _NTHREADS)


def now():
    return time.perf_counter()


def median_of(fn, runs=5):
    return statistics.median(fn() for _ in range(runs))


def fmt_ms(ms):
    if ms is None:
        return "--"
    if ms <= 0:
        return "0ms"
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    if ms >= 100:
        return f"{ms:.1f}ms"
    return f"{ms:.2f}ms"


def fmt_tp(key, ms):
    if ms is None or ms <= 0:
        return "--"
    s = ms / 1000.0
    if key.startswith(("gemm_", "fused_")):
        n = int(key.rsplit("_", 1)[1])
        gf = 2.0 * n**3 / s / 1e9
        return f"{gf:.0f} GF/s"
    if key in ("mlp_forward", "mlp_train_step"):
        sps = 64.0 / s
        return f"{sps/1000:.1f}K/s" if sps >= 1000 else f"{sps:.0f}/s"
    if key == "conv2d_forward":
        sps = 8.0 / s
        return f"{sps/1000:.1f}K/s" if sps >= 1000 else f"{sps:.0f}/s"
    return ""


def bench_torch():
    import torch

    n = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
    torch.set_num_threads(n)
    torch.set_num_interop_threads(n)
    results = {}

    for N in [512, 1024, 2048]:
        a = torch.randn(N, N)
        b = torch.randn(N, N)
        for _ in range(3):
            torch.mm(a, b)

        def run(a=a, b=b, N=N):
            iters = 3 if N >= 2048 else 5
            t0 = now()
            for _ in range(iters):
                torch.mm(a, b)
            return (now() - t0) / iters * 1e3

        results[f"gemm_{N}"] = median_of(run)

    for N in [512, 1024, 2048]:
        a = torch.randn(N, N)
        b = torch.randn(N, N)
        bias = torch.randn(1, N)
        for _ in range(3):
            torch.relu(torch.mm(a, b) + bias)

        def run(a=a, b=b, bias=bias, N=N):
            iters = 3 if N >= 2048 else 5
            t0 = now()
            for _ in range(iters):
                torch.relu(torch.mm(a, b) + bias)
            return (now() - t0) / iters * 1e3

        results[f"fused_{N}"] = median_of(run)

    model = torch.nn.Sequential(
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    )
    model.eval()
    x = torch.randn(64, 784)
    with torch.no_grad():
        for _ in range(5):
            model(x)

        def run():
            t0 = now()
            for _ in range(100):
                model(x)
            return (now() - t0) / 100 * 1e3

        results["mlp_forward"] = median_of(run)

    model_train = torch.nn.Sequential(
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    )
    model_train.train()
    opt = torch.optim.SGD(model_train.parameters(), lr=0.01)
    x = torch.randn(64, 784)
    target = torch.randn(64, 10)
    loss_fn = torch.nn.MSELoss()

    for _ in range(5):
        opt.zero_grad()
        out = model_train(x)
        loss = loss_fn(out, target)
        loss.backward()
        opt.step()

    def run():
        t0 = now()
        for _ in range(50):
            opt.zero_grad()
            out = model_train(x)
            loss = loss_fn(out, target)
            loss.backward()
            opt.step()
        return (now() - t0) / 50 * 1e3

    results["mlp_train_step"] = median_of(run)

    conv = torch.nn.Conv2d(3, 16, 3)
    conv.eval()
    x_conv = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        for _ in range(5):
            conv(x_conv)

        def run():
            t0 = now()
            for _ in range(100):
                conv(x_conv)
            return (now() - t0) / 100 * 1e3

        results["conv2d_forward"] = median_of(run)

    return results


def bench_torch_c(binary_path, env_extra=None):
    try:
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        result = subprocess.run(
            [binary_path],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        if result.returncode != 0:
            print(f"  torch_c binary failed: {result.stderr[:300]}")
            return {}
        stdout = result.stdout
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start == -1 or end == -1:
            print("  torch_c: no JSON found in output")
            return {}
        return json.loads(stdout[start : end + 1])
    except json.JSONDecodeError:
        print("  torch_c: invalid JSON in output")
        return {}
    except FileNotFoundError:
        print(f"  torch_c binary not found: {binary_path}")
        return {}
    except subprocess.TimeoutExpired:
        print("  torch_c binary timed out")
        return {}


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

    print(f"torch_c vs PyTorch Benchmark  (float32, {os.environ.get('OMP_NUM_THREADS', '?')} threads)")
    print(f"Python {sys.version.split()[0]}")

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    print(f"CPU: {line.split(':', 1)[1].strip()}  ({os.cpu_count()} cores)")
                    break
    except OSError:
        pass

    all_results = {}

    print("\nRunning torch_c (CPU) benchmarks...")
    all_results["torch_c"] = bench_torch_c(binary)

    gpu = bench_torch_c(binary, env_extra={"CML_BACKEND": "opencl"})
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

    groups = [
        ("GEMM", [
            ("512×512", "gemm_512"),
            ("1024×1024", "gemm_1024"),
            ("2048×2048", "gemm_2048"),
        ]),
        ("Fused  (mm + bias + relu)", [
            ("512", "fused_512"),
            ("1024", "fused_1024"),
            ("2048", "fused_2048"),
        ]),
        ("MLP", [
            ("forward", "mlp_forward"),
            ("train step", "mlp_train_step"),
        ]),
        ("Conv2d", [
            ("forward", "conv2d_forward"),
        ]),
    ]

    lw = 30
    cw = max(max(len(fw) for fw in frameworks) + 2, 10)
    nf = len(frameworks)

    def hline(l, m, r):
        return l + "─" * (lw + 2) + (m + "─" * cw) * nf + r

    def data_row(label, vals, indent=3):
        row = "│ " + (" " * indent + label).ljust(lw) + " "
        for v in vals:
            row += "│" + v.rjust(cw - 1) + " "
        return row + "│"

    print(f"\n  Results  (median · lower ms = faster · higher GF/s or K/s = better)\n")
    print(hline("┌", "┬", "┐"))
    hdr = "│ " + "Benchmark".ljust(lw) + " "
    for fw in frameworks:
        hdr += "│" + fw.center(cw)
    print(hdr + "│")
    print(hline("├", "┼", "┤"))

    pt = all_results.get("pytorch", {})
    tc = all_results.get("torch_c", {})

    for gi, (gname, rows) in enumerate(groups):
        print("│ " + gname.ljust(lw) + " " + ("│" + " " * cw) * nf + "│")
        for label, key in rows:
            ms_vals = [fmt_ms(all_results[fw].get(key)) for fw in frameworks]
            tp_vals = [fmt_tp(key, all_results[fw].get(key)) for fw in frameworks]
            print(data_row(label, ms_vals))
            print(data_row("", tp_vals, indent=0))
        if gi < len(groups) - 1:
            print(hline("├", "┼", "┤"))

    print(hline("└", "┴", "┘"))

    if pt and tc:
        print("\n  torch_c vs PyTorch (CPU)\n")
        for _, rows in groups:
            for label, key in rows:
                t_ms = pt.get(key)
                c_ms = tc.get(key)
                if t_ms is not None and c_ms is not None:
                    print(f"  {label:20s}  torch_c {fmt_ms(c_ms):>10s}  pytorch {fmt_ms(t_ms):>10s}  ({ratio(t_ms, c_ms)})")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
