"""Shared measurement harness for the benchmark drivers.

bench_all.py compares C-ML against the Python frameworks; bench_torch_c.py
compares the torch_c C API against PyTorch. The questions differ, but the
numbers are only comparable if both drivers measure the same way -- so timing,
formatting, the PyTorch reference implementation and the subprocess plumbing
live here rather than in each driver.

Importing this module pins the BLAS thread count, which must happen before
numpy/torch are imported anywhere.
"""

import json
import os
import statistics
import subprocess
import time

_NTHREADS = str(os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", _NTHREADS)
os.environ.setdefault("MKL_NUM_THREADS", _NTHREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NTHREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _NTHREADS)


def now():
    return time.perf_counter()


def median_of(fn, runs=5):
    """Run fn() `runs` times, return the median time in ms."""
    return statistics.median(fn() for _ in range(runs))


def fmt_ms(ms):
    if ms is None:
        return "--"
    if ms <= 0:
        return "0ms"
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    if ms >= 100:
        return f"{ms:.1f}ms"
    return f"{ms:.2f}ms"


def fmt_tp(key, ms):
    """Throughput: GFLOPS for the matrix ops, samples/s for MLP/Conv2d."""
    if ms is None or ms <= 0:
        return "--"
    s = ms / 1000.0
    if key.startswith(("gemm_", "fused_")):
        N = int(key.rsplit("_", 1)[1])
        gf = 2.0 * N**3 / s / 1e9
        return f"{gf:.0f} GF/s"
    if key in ("mlp_forward", "mlp_train_step"):
        sps = 64.0 / s
        return f"{sps / 1000:.1f}K/s" if sps >= 1000 else f"{sps:.0f}/s"
    if key == "conv2d_forward":
        sps = 8.0 / s
        return f"{sps / 1000:.1f}K/s" if sps >= 1000 else f"{sps:.0f}/s"
    return ""


def print_host_info(title):
    """Header identifying what was measured and on what."""
    print(f"{title}  (float32, {os.environ.get('OMP_NUM_THREADS', '?')} threads)")
    import sys
    print(f"Python {sys.version.split()[0]}")
    cpu = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        import platform
        cpu = platform.processor()
    print(f"CPU: {cpu or 'unknown'}  ({os.cpu_count()} cores)")


# The benchmark rows, in report order. Both drivers show the same set because
# both compare against the same C binaries, which emit exactly these keys
# (see bench_print_json in bench_timing.h).
BENCH_GROUPS = [
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


def print_results_table(all_results, frameworks, groups=BENCH_GROUPS):
    """Render the results as a box table, one column per framework.

    Each benchmark prints two lines: the time, and the same figure expressed as
    throughput underneath it.
    """
    LW = 30  # label column text width (between borders)
    CW = max(max(len(fw) for fw in frameworks) + 2, 10)
    NF = len(frameworks)

    def hline(l, m, r):
        return l + "─" * (LW + 2) + (m + "─" * CW) * NF + r

    def data_row(label, vals, indent=3):
        row = "│ " + (" " * indent + label).ljust(LW) + " "
        for v in vals:
            row += "│" + v.rjust(CW - 1) + " "
        return row + "│"

    print("\n  Results  (median · lower ms = faster · higher GF/s or K/s = better)\n")
    print(hline("┌", "┬", "┐"))
    hdr = "│ " + "Benchmark".ljust(LW) + " "
    for fw in frameworks:
        hdr += "│" + fw.center(CW)
    print(hdr + "│")
    print(hline("├", "┼", "┤"))

    for gi, (gname, rows) in enumerate(groups):
        print("│ " + gname.ljust(LW) + " " + ("│" + " " * CW) * NF + "│")
        for label, key in rows:
            ms_vals = [fmt_ms(all_results[fw].get(key)) for fw in frameworks]
            tp_vals = [fmt_tp(key, all_results[fw].get(key)) for fw in frameworks]
            print(data_row(label, ms_vals))
            print(data_row("", tp_vals, indent=0))
        if gi < len(groups) - 1:
            print(hline("├", "┼", "┤"))

    print(hline("└", "┴", "┘"))


def parse_json_results(stdout, who):
    """Pull the results object out of a subprocess's stdout.

    The C bench binaries and the out-of-process TensorFlow run all print a JSON
    object among other chatter, so locate it by its outermost braces.
    """
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start == -1 or end == -1:
        print(f"  {who}: no JSON found in output")
        return {}
    try:
        return json.loads(stdout[start:end + 1])
    except json.JSONDecodeError:
        print(f"  {who}: output parse error")
        print(f"  stdout: {stdout[:500]}")
        return {}


def run_c_binary(binary_path, label, env_extra=None, timeout=300):
    """Run a C bench binary and return the results it printed as JSON."""
    try:
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        result = subprocess.run(
            [binary_path], capture_output=True, text=True, timeout=timeout, env=env
        )
        if result.returncode != 0:
            print(f"  {label} binary failed: {result.stderr[:300]}")
            return {}
        return parse_json_results(result.stdout, label)
    except FileNotFoundError:
        print(f"  {label} binary not found: {binary_path}")
        return {}
    except subprocess.TimeoutExpired:
        print(f"  {label} binary timed out")
        return {}


def bench_torch(device="cpu"):
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        return None

    # On an accelerator the host threads only queue work, so giving them the
    # whole machine measures scheduling noise rather than the device.
    n = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)) if device == "cpu" else 1
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        pass  # settable only once per process; a repeat pass already did it

    # CUDA dispatch is async, so a timed region has to close on a real
    # completion point or it measures enqueue speed. A no-op on CPU.
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    results = {}

    for N in [512, 1024, 2048]:
        a = torch.randn(N, N, device=device)
        b = torch.randn(N, N, device=device)
        for _ in range(3):
            torch.mm(a, b)
        sync()

        def run(a=a, b=b):
            t0 = now()
            for _ in range(5):
                torch.mm(a, b)
            sync()
            return (now() - t0) / 5 * 1e3

        ms = median_of(run, runs=5)
        results[f"gemm_{N}"] = ms

    for N in [512, 1024, 2048]:
        a = torch.randn(N, N, device=device)
        b = torch.randn(N, N, device=device)
        bias = torch.randn(1, N, device=device)
        for _ in range(3):
            torch.relu(torch.mm(a, b) + bias)
        sync()

        def run(a=a, b=b, bias=bias):
            t0 = now()
            for _ in range(5):
                torch.relu(torch.mm(a, b) + bias)
            sync()
            return (now() - t0) / 5 * 1e3

        ms = median_of(run, runs=5)
        results[f"fused_{N}"] = ms

    model = torch.nn.Sequential(
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    ).to(device)
    model.eval()
    x = torch.randn(64, 784, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        sync()

        def run(model=model, x=x):
            t0 = now()
            for _ in range(100):
                model(x)
            sync()
            return (now() - t0) / 100 * 1e3

        ms = median_of(run, runs=5)
        results["mlp_forward"] = ms

    model_train = torch.nn.Sequential(
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    ).to(device)
    model_train.train()
    opt = torch.optim.SGD(model_train.parameters(), lr=0.01)
    x = torch.randn(64, 784, device=device)
    target = torch.randn(64, 10, device=device)
    loss_fn = torch.nn.MSELoss()

    for _ in range(5):
        opt.zero_grad()
        out = model_train(x)
        loss = loss_fn(out, target)
        loss.backward()
        opt.step()
    sync()

    def run():
        t0 = now()
        for _ in range(50):
            opt.zero_grad()
            out = model_train(x)
            loss = loss_fn(out, target)
            loss.backward()
            opt.step()
        sync()
        return (now() - t0) / 50 * 1e3

    ms = median_of(run, runs=5)
    results["mlp_train_step"] = ms

    conv = torch.nn.Conv2d(3, 16, 3).to(device)
    conv.eval()
    x_conv = torch.randn(8, 3, 32, 32, device=device)
    with torch.no_grad():
        for _ in range(5):
            conv(x_conv)
        sync()

        def run(conv=conv, x_conv=x_conv):
            t0 = now()
            for _ in range(100):
                conv(x_conv)
            sync()
            return (now() - t0) / 100 * 1e3

        ms = median_of(run, runs=5)
        results["conv2d_forward"] = ms

    return results
