import time
import sys
import os
import json
import subprocess
import statistics

_NTHREADS = str(os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", _NTHREADS)
os.environ.setdefault("MKL_NUM_THREADS", _NTHREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NTHREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _NTHREADS)

def now():
    return time.perf_counter()

def median_of(fn, runs=5):
    """Run fn() `runs` times, return median time in ms."""
    times = []
    for _ in range(runs):
        t = fn()
        times.append(t)
    return statistics.median(times)

def fmt_ms(ms):
    if ms is None: return "--"
    if ms >= 1000: return f"{ms/1000:.2f}s"
    if ms >= 100:  return f"{ms:.1f}ms"
    return f"{ms:.2f}ms"

def fmt_tp(key, ms):
    """Throughput: GFLOPS for matrix ops, samples/s for MLP/Conv2d."""
    if ms is None: return "--"
    s = ms / 1000.0
    if key.startswith(("gemm_", "fused_")):
        N = int(key.rsplit("_", 1)[1])
        gf = 2.0 * N**3 / s / 1e9
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

        def run():
            t0 = now()
            for _ in range(5):
                torch.mm(a, b)
            return (now() - t0) / 5 * 1e3

        ms = median_of(run, runs=5)
        results[f"gemm_{N}"] = ms

    for N in [512, 1024, 2048]:
        a = torch.randn(N, N)
        b = torch.randn(N, N)
        bias = torch.randn(1, N)
        for _ in range(3):
            torch.relu(torch.mm(a, b) + bias)

        def run(a=a, b=b, bias=bias):
            t0 = now()
            for _ in range(5):
                torch.relu(torch.mm(a, b) + bias)
            return (now() - t0) / 5 * 1e3

        ms = median_of(run, runs=5)
        results[f"fused_{N}"] = ms

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

        ms = median_of(run, runs=5)
        results["mlp_forward"] = ms

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

    ms = median_of(run, runs=5)
    results["mlp_train_step"] = ms

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

        ms = median_of(run, runs=5)
        results["conv2d_forward"] = ms

    return results

def bench_tinygrad():
    # Benchmarks whatever device tinygrad picks (OpenCL GPU if available, else CPU).
    # Uses TinyJit so each benchmark times repeated dispatch of a compiled kernel,
    # not graph construction or numpy wrapping.
    import numpy as np
    from tinygrad import Tensor as TgTensor
    from tinygrad import nn as tg_nn
    from tinygrad.engine.jit import TinyJit
    results = {}

    RUNS = 5
    ITERS = 5  # iters per run() call
    # Each pair used exactly once across all runs*iters calls — tinygrad never revisits
    # a realized output, so every call dispatches the compiled kernel for real.
    POOL = RUNS * ITERS

    for N in [512, 1024, 2048]:
        pool = [(TgTensor(np.random.randn(N, N).astype(np.float32)),
                 TgTensor(np.random.randn(N, N).astype(np.float32))) for _ in range(POOL + 3)]
        for a, b in pool:
            a.realize(); b.realize()

        @TinyJit
        def gemm_jit(a, b): return (a @ b).realize()

        it = iter(pool)
        for a, b in [next(it) for _ in range(3)]:  # warmup + JIT compile
            gemm_jit(a, b)

        def run(it=it):
            pairs = [next(it) for _ in range(ITERS)]
            t0 = now()
            for a, b in pairs:
                out = gemm_jit(a, b)
            out.numpy()  # force GPU sync — OpenCL dispatch is async
            return (now() - t0) / ITERS * 1e3

        results[f"gemm_{N}"] = median_of(run, runs=RUNS)

    for N in [512, 1024, 2048]:
        pool = [(TgTensor(np.random.randn(N, N).astype(np.float32)),
                 TgTensor(np.random.randn(N, N).astype(np.float32)),
                 TgTensor(np.random.randn(1, N).astype(np.float32))) for _ in range(POOL + 3)]
        for a, b, bias in pool:
            a.realize(); b.realize(); bias.realize()

        @TinyJit
        def fused_jit(a, b, bias): return (a @ b + bias).relu().realize()

        it = iter(pool)
        for a, b, bias in [next(it) for _ in range(3)]:
            fused_jit(a, b, bias)

        def run(it=it):
            pairs = [next(it) for _ in range(ITERS)]
            t0 = now()
            for a, b, bias in pairs:
                out = fused_jit(a, b, bias)
            out.numpy()
            return (now() - t0) / ITERS * 1e3

        results[f"fused_{N}"] = median_of(run, runs=RUNS)

    class TgMLP:
        def __init__(self):
            self.l1 = tg_nn.Linear(784, 128)
            self.l2 = tg_nn.Linear(128, 10)
        def __call__(self, x):
            return self.l2(self.l1(x).relu())

    MLP_ITERS = 100
    MLP_POOL = RUNS * MLP_ITERS

    model = TgMLP()
    for p in tg_nn.state.get_parameters(model):
        p.realize()
    x_pool = [TgTensor(np.random.randn(64, 784).astype(np.float32)) for _ in range(MLP_POOL + 5)]
    for x in x_pool:
        x.realize()

    @TinyJit
    def mlp_jit(x): return model(x).realize()

    it = iter(x_pool)
    for x in [next(it) for _ in range(5)]:
        mlp_jit(x)

    def run(it=it):
        xs = [next(it) for _ in range(MLP_ITERS)]
        t0 = now()
        for x in xs:
            out = mlp_jit(x)
        out.numpy()
        return (now() - t0) / MLP_ITERS * 1e3

    results["mlp_forward"] = median_of(run, runs=RUNS)
    results["mlp_train_step"] = None

    CONV_ITERS = 100
    CONV_POOL = RUNS * CONV_ITERS

    conv = tg_nn.Conv2d(3, 16, 3)
    for p in tg_nn.state.get_parameters(conv):
        p.realize()
    xc_pool = [TgTensor(np.random.randn(8, 3, 32, 32).astype(np.float32)) for _ in range(CONV_POOL + 5)]
    for x in xc_pool:
        x.realize()

    @TinyJit
    def conv_jit(x): return conv(x).realize()

    it = iter(xc_pool)
    for x in [next(it) for _ in range(5)]:
        conv_jit(x)

    def run(it=it):
        xs = [next(it) for _ in range(CONV_ITERS)]
        t0 = now()
        for x in xs:
            out = conv_jit(x)
        out.numpy()
        return (now() - t0) / CONV_ITERS * 1e3

    results["conv2d_forward"] = median_of(run, runs=RUNS)
    return results

def bench_cml(binary_path, env_extra=None):
    try:
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        result = subprocess.run(
            [binary_path],
            capture_output=True, text=True, timeout=300,
            env=env
        )
        if result.returncode != 0:
            print(f"  CML binary failed: {result.stderr[:200]}")
            return {}
        stdout = result.stdout
        start = stdout.find('{')
        end = stdout.rfind('}')
        if start == -1 or end == -1:
            print(f"  CML: no JSON found in output")
            return {}
        return json.loads(stdout[start:end+1])
    except FileNotFoundError:
        print(f"  CML binary not found: {binary_path}")
        return {}
    except json.JSONDecodeError:
        print(f"  CML output parse error")
        print(f"  stdout: {result.stdout[:500]}")
        return {}
    except subprocess.TimeoutExpired:
        print(f"  CML binary timed out")
        return {}

def bench_numpy():
    import numpy as np
    results = {}

    for N in [512, 1024, 2048]:
        a = np.random.randn(N, N).astype(np.float32)
        b = np.random.randn(N, N).astype(np.float32)
        for _ in range(3):
            np.dot(a, b)

        def run(a=a, b=b):
            t0 = now()
            for _ in range(5):
                np.dot(a, b)
            return (now() - t0) / 5 * 1e3

        ms = median_of(run, runs=5)
        results[f"gemm_{N}"] = ms

    for N in [512, 1024, 2048]:
        a = np.random.randn(N, N).astype(np.float32)
        b = np.random.randn(N, N).astype(np.float32)
        bias = np.random.randn(1, N).astype(np.float32)

        def run(a=a, b=b, bias=bias):
            t0 = now()
            for _ in range(5):
                np.maximum(np.dot(a, b) + bias, 0)
            return (now() - t0) / 5 * 1e3

        ms = median_of(run, runs=5)
        results[f"fused_{N}"] = ms

    results["mlp_forward"] = None
    results["mlp_train_step"] = None
    results["conv2d_forward"] = None

    return results

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    cml_binary = os.path.join(project_dir, "build", "bin", "bench_cross_framework")

    print(f"Cross-Framework Benchmark  (float32, {os.environ.get('OMP_NUM_THREADS', '?')} threads)")
    print(f"Python {sys.version.split()[0]}")

    try:
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
    except Exception:
        pass

    all_results = {}

    try:
        import numpy as np
        print(f"\nRunning NumPy {np.__version__} benchmarks...")
        all_results["numpy"] = bench_numpy()
    except ImportError:
        print("\nNumPy not available, skipping")

    print(f"\nRunning CML (CPU) benchmarks...")
    all_results["cml"] = bench_cml(cml_binary)

    print(f"\nRunning CML (GPU) benchmarks...")
    cml_gpu = bench_cml(cml_binary, env_extra={"CML_BACKEND": "opencl"})
    if cml_gpu:
        all_results["cml(GPU)"] = cml_gpu

    try:
        import torch
        print(f"\nRunning PyTorch {torch.__version__} benchmarks...")
        all_results["pytorch"] = bench_torch()
    except ImportError:
        print("\nPyTorch not available, skipping")

    try:
        import tinygrad
        from tinygrad.device import Device as TgDevice
        try:
            from importlib.metadata import version as pkg_version
            ver = pkg_version('tinygrad')
        except Exception:
            ver = getattr(tinygrad, '__version__', 'unknown')
        dev = TgDevice.DEFAULT
        label = f"tinygrad({dev})"
        print(f"\nRunning TinyGrad {ver} benchmarks (device: {dev})...")
        all_results[label] = bench_tinygrad()
    except ImportError:
        print("\nTinyGrad not available, skipping")
    except Exception as e:
        print(f"\nTinyGrad failed: {e}")

    ordered = ["numpy", "cml", "pytorch"]
    frameworks = [k for k in ordered if k in all_results]
    frameworks += [k for k in all_results if k not in ordered]

    if not frameworks:
        print("\nNo results to display.")
        return

    groups = [
        ("GEMM", [
            ("512×512",   "gemm_512"),
            ("1024×1024", "gemm_1024"),
            ("2048×2048", "gemm_2048"),
        ]),
        ("Fused  (mm + bias + relu)", [
            ("512",  "fused_512"),
            ("1024", "fused_1024"),
            ("2048", "fused_2048"),
        ]),
        ("MLP", [
            ("forward",    "mlp_forward"),
            ("train step", "mlp_train_step"),
        ]),
        ("Conv2d", [
            ("forward", "conv2d_forward"),
        ]),
    ]

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

    print(f"\n  Results  (median · lower ms = faster · higher GF/s or K/s = better)\n")
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
    print()

if __name__ == "__main__":
    main()
