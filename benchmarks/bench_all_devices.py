#!/usr/bin/env python3
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
    times = []
    for _ in range(runs):
        t = fn()
        times.append(t)
    return statistics.median(times)


def fmt_ms(ms):
    if ms is None:
        return "--"
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    if ms >= 100:
        return f"{ms:.1f}ms"
    return f"{ms:.2f}ms"


def fmt_tp(key, ms):
    if ms is None:
        return ""
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


def bench_torch(device):
    import torch

    results = {}

    if device == "cuda" and not torch.cuda.is_available():
        return None

    if device == "cpu":
        n = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
        torch.set_num_threads(n)
        torch.set_num_interop_threads(n)
    else:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    for N in [512, 1024, 2048]:
        a = torch.randn(N, N, device=device)
        b = torch.randn(N, N, device=device)
        for _ in range(3):
            torch.mm(a, b)
        if device == "cuda":
            torch.cuda.synchronize()

        def run():
            t0 = now()
            for _ in range(5):
                torch.mm(a, b)
            if device == "cuda":
                torch.cuda.synchronize()
            return (now() - t0) / 5 * 1e3

        ms = median_of(run, runs=5)
        results[f"gemm_{N}"] = ms

    for N in [512, 1024, 2048]:
        a = torch.randn(N, N, device=device)
        b = torch.randn(N, N, device=device)
        bias = torch.randn(1, N, device=device)
        for _ in range(3):
            torch.relu(torch.mm(a, b) + bias)
        if device == "cuda":
            torch.cuda.synchronize()

        def run(a=a, b=b, bias=bias):
            t0 = now()
            for _ in range(5):
                torch.relu(torch.mm(a, b) + bias)
            if device == "cuda":
                torch.cuda.synchronize()
            return (now() - t0) / 5 * 1e3

        ms = median_of(run, runs=5)
        results[f"fused_{N}"] = ms

    model = torch.nn.Sequential(
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    )
    model = model.to(device)
    model.eval()
    x = torch.randn(64, 784, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        def run():
            t0 = now()
            for _ in range(100):
                model(x)
            if device == "cuda":
                torch.cuda.synchronize()
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
    if device == "cuda":
        torch.cuda.synchronize()

    def run():
        t0 = now()
        for _ in range(50):
            opt.zero_grad()
            out = model_train(x)
            loss = loss_fn(out, target)
            loss.backward()
            opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        return (now() - t0) / 50 * 1e3

    ms = median_of(run, runs=5)
    results["mlp_train_step"] = ms

    conv = torch.nn.Conv2d(3, 16, 3).to(device)
    conv.eval()
    x_conv = torch.randn(8, 3, 32, 32, device=device)
    with torch.no_grad():
        for _ in range(5):
            conv(x_conv)
        if device == "cuda":
            torch.cuda.synchronize()

        def run():
            t0 = now()
            for _ in range(100):
                conv(x_conv)
            if device == "cuda":
                torch.cuda.synchronize()
            return (now() - t0) / 100 * 1e3

        ms = median_of(run, runs=5)
        results["conv2d_forward"] = ms

    return results


def bench_tinygrad(device):
    import numpy as np
    from tinygrad import Tensor as TgTensor
    from tinygrad import nn as tg_nn
    from tinygrad.engine.jit import TinyJit
    from tinygrad.device import Device

    results = {}

    original_device = Device.DEFAULT
    try:
        Device.DEFAULT = device
    except Exception:
        return None

    # Pool sizes: pre-allocate small/medium sizes, but create on-the-fly for 2048
    # to avoid ~2GB peak memory when running CPU and CL back-to-back.
    RUNS = 5
    ITERS = 5
    POOL = RUNS * ITERS

    def _iters_for(N):
        return ITERS if N <= 1024 else 2

    def _pool_size(N):
        ni = _iters_for(N)
        return RUNS * ni + 3  # 3 warmup + runs*iters

    for N in [512, 1024, 2048]:
        n_pool = _pool_size(N)
        n_iters = _iters_for(N)
        n_runs = RUNS

        pool = [
            (
                TgTensor(np.random.randn(N, N).astype(np.float32)),
                TgTensor(np.random.randn(N, N).astype(np.float32)),
            )
            for _ in range(n_pool + 3)
        ]
        for a, b in pool:
            a.realize()
            b.realize()

        @TinyJit
        def gemm_jit(a, b):
            return (a @ b).realize()

        it = iter(pool)
        for a, b in [next(it) for _ in range(3)]:
            gemm_jit(a, b)

        def run(it=it, ni=n_iters):
            pairs = [next(it) for _ in range(ni)]
            t0 = now()
            for a, b in pairs:
                out = gemm_jit(a, b)
            out.numpy()
            return (now() - t0) / ni * 1e3

        results[f"gemm_{N}"] = median_of(run, runs=n_runs)
        del pool

    for N in [512, 1024, 2048]:
        n_pool = _pool_size(N)
        n_iters = _iters_for(N)
        n_runs = RUNS

        pool = [
            (
                TgTensor(np.random.randn(N, N).astype(np.float32)),
                TgTensor(np.random.randn(N, N).astype(np.float32)),
                TgTensor(np.random.randn(1, N).astype(np.float32)),
            )
            for _ in range(n_pool + 3)
        ]
        for a, b, bias in pool:
            a.realize()
            b.realize()
            bias.realize()

        @TinyJit
        def fused_jit(a, b, bias):
            return (a @ b + bias).relu().realize()

        it = iter(pool)
        for a, b, bias in [next(it) for _ in range(3)]:
            fused_jit(a, b, bias)

        def run(it=it, ni=n_iters):
            pairs = [next(it) for _ in range(ni)]
            t0 = now()
            for a, b, bias in pairs:
                out = fused_jit(a, b, bias)
            out.numpy()
            return (now() - t0) / ni * 1e3

        results[f"fused_{N}"] = median_of(run, runs=n_runs)
        del pool

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
    x_pool = [
        TgTensor(np.random.randn(64, 784).astype(np.float32))
        for _ in range(MLP_POOL + 5)
    ]
    for x in x_pool:
        x.realize()

    @TinyJit
    def mlp_jit(x):
        return model(x).realize()

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

    # MLP train step with SGD — create tensors fresh each step to avoid gradient accumulation OOM
    from tinygrad.nn import optim as tg_optim

    train_model = TgMLP()
    optimizer = tg_optim.SGD(tg_nn.state.get_parameters(train_model), lr=0.01)

    import gc
    TgTensor.training = True

    def tg_train_step_once():
        x_t = TgTensor(np.random.randn(64, 784).astype(np.float32))
        tgt_t = TgTensor(np.random.randn(64, 10).astype(np.float32))
        optimizer.zero_grad()
        out = train_model(x_t)
        loss = ((out - tgt_t) * (out - tgt_t)).mean()
        loss.backward()
        optimizer.step()
        loss.numpy()
        del x_t, tgt_t, out, loss

    for _ in range(3):
        tg_train_step_once()
        gc.collect()

    TRAIN_ITERS = 10

    def run():
        t0 = now()
        for _ in range(TRAIN_ITERS):
            tg_train_step_once()
        gc.collect()
        return (now() - t0) / TRAIN_ITERS * 1e3

    results["mlp_train_step"] = median_of(run, runs=3)
    gc.collect()

    CONV_ITERS = 100
    CONV_POOL = RUNS * CONV_ITERS

    conv = tg_nn.Conv2d(3, 16, 3)
    for p in tg_nn.state.get_parameters(conv):
        p.realize()
    xc_pool = [
        TgTensor(np.random.randn(8, 3, 32, 32).astype(np.float32))
        for _ in range(CONV_POOL + 5)
    ]
    for x in xc_pool:
        x.realize()

    @TinyJit
    def conv_jit(x):
        return conv(x).realize()

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

    Device.DEFAULT = original_device
    return results


def bench_cml(binary_path, env_extra=None):
    try:
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        result = subprocess.run(
            [binary_path], capture_output=True, text=True, timeout=300, env=env
        )
        if result.returncode != 0:
            print(f"  CML binary failed: {result.stderr[:200]}")
            return {}
        stdout = result.stdout
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start == -1 or end == -1:
            print(f"  CML: no JSON found in output")
            return {}
        return json.loads(stdout[start : end + 1])
    except Exception as e:
        print(f"  CML error: {e}")
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

    # MLP forward: 784->128->ReLU->10, batch=64
    W1 = np.random.randn(128, 784).astype(np.float32)
    b1 = np.random.randn(128).astype(np.float32)
    W2 = np.random.randn(10, 128).astype(np.float32)
    b2 = np.random.randn(10).astype(np.float32)
    x_mlp = np.random.randn(64, 784).astype(np.float32)
    for _ in range(5):
        np.maximum(x_mlp @ W1.T + b1, 0) @ W2.T + b2

    def run():
        t0 = now()
        for _ in range(100):
            h = np.maximum(x_mlp @ W1.T + b1, 0)
            np.dot(h, W2.T) + b2
        return (now() - t0) / 100 * 1e3

    results["mlp_forward"] = median_of(run, runs=5)

    # MLP train step (manual SGD, no autograd)
    W1t = np.random.randn(128, 784).astype(np.float32) * 0.01
    b1t = np.zeros(128, dtype=np.float32)
    W2t = np.random.randn(10, 128).astype(np.float32) * 0.01
    b2t = np.zeros(10, dtype=np.float32)
    lr = 0.01

    def numpy_train_step():
        x = np.random.randn(64, 784).astype(np.float32)
        tgt = np.random.randn(64, 10).astype(np.float32)
        h_pre = x @ W1t.T + b1t
        h = np.maximum(h_pre, 0)
        out = h @ W2t.T + b2t
        d_out = 2.0 * (out - tgt) / tgt.size
        W2t[:] -= lr * (d_out.T @ h)
        b2t[:] -= lr * d_out.sum(0)
        d_h = d_out @ W2t
        d_h_pre = d_h * (h_pre > 0)
        W1t[:] -= lr * (d_h_pre.T @ x)
        b1t[:] -= lr * d_h_pre.sum(0)

    for _ in range(5):
        numpy_train_step()

    def run():
        t0 = now()
        for _ in range(50):
            numpy_train_step()
        return (now() - t0) / 50 * 1e3

    results["mlp_train_step"] = median_of(run, runs=5)

    # Conv2d forward: (8,3,32,32) -> 16 filters of 3x3 (no padding)
    weight_c = np.random.randn(16, 3, 3, 3).astype(np.float32)
    x_conv = np.random.randn(8, 3, 32, 32).astype(np.float32)

    def numpy_conv2d(x, w):
        N, C, H, W_in = x.shape
        F, C, KH, KW = w.shape
        OH, OW = H - KH + 1, W_in - KW + 1
        # im2col via stride tricks
        cols = np.lib.stride_tricks.as_strided(
            x,
            shape=(N, C, KH, KW, OH, OW),
            strides=x.strides[:2] + x.strides[2:] + x.strides[2:],
        ).reshape(N, C * KH * KW, OH * OW)
        w_col = w.reshape(F, -1)
        return np.einsum("fi,nio->nfo", w_col, cols).reshape(N, F, OH, OW)

    for _ in range(5):
        numpy_conv2d(x_conv, weight_c)

    def run():
        t0 = now()
        for _ in range(100):
            numpy_conv2d(x_conv, weight_c)
        return (now() - t0) / 100 * 1e3

    results["conv2d_forward"] = median_of(run, runs=5)

    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    cml_binary = os.path.join(project_dir, "build", "bin", "bench_cross_framework")

    print(f"Cross-Framework Benchmark (float32)")
    print(f"Threads: {os.environ.get('OMP_NUM_THREADS', '?')}")

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
        print(f"CPU: {cpu or 'unknown'} ({os.cpu_count()} cores)")
    except Exception:
        pass

    all_results = {}

    # NumPy (CPU)
    try:
        import numpy as np

        print(f"\nRunning NumPy {np.__version__} (CPU)...")
        all_results["NumPy(CPU)"] = bench_numpy()
    except ImportError:
        print("NumPy not available, skipping")

    # CML CPU
    print(f"\nRunning CML (CPU)...")
    all_results["CML(CPU)"] = bench_cml(cml_binary)

    # CML OpenCL
    print(f"\nRunning CML (OpenCL)...")
    cml_ocl = bench_cml(cml_binary, env_extra={"CML_BACKEND": "opencl"})
    if cml_ocl:
        all_results["CML(OpenCL)"] = cml_ocl

    # PyTorch CPU
    try:
        import torch

        print(f"\nRunning PyTorch {torch.__version__} (CPU)...")
        all_results["PyTorch(CPU)"] = bench_torch("cpu")
    except ImportError:
        print("PyTorch not available")

    # PyTorch CUDA
    try:
        import torch

        if torch.cuda.is_available():
            print(f"\nRunning PyTorch {torch.__version__} (CUDA)...")
            all_results["PyTorch(CUDA)"] = bench_torch("cuda")
    except ImportError:
        pass

    # TinyGrad CPU
    try:
        print(f"\nRunning TinyGrad (CPU)...")
        all_results["TinyGrad(CPU)"] = bench_tinygrad("CPU")
    except Exception as e:
        print(f"TinyGrad CPU failed: {e}")

    # TinyGrad OpenCL
    try:
        print(f"\nRunning TinyGrad (OpenCL)...")
        tg_gpu = bench_tinygrad("CL")
        if tg_gpu:
            all_results["TinyGrad(OpenCL)"] = tg_gpu
    except Exception as e:
        print(f"TinyGrad OpenCL failed: {e}")

    # Print results
    frameworks = list(all_results.keys())

    groups = [
        (
            "GEMM",
            [
                ("512×512", "gemm_512"),
                ("1024×1024", "gemm_1024"),
                ("2048×2048", "gemm_2048"),
            ],
        ),
        (
            "Fused (mm+bias+relu)",
            [
                ("512", "fused_512"),
                ("1024", "fused_1024"),
                ("2048", "fused_2048"),
            ],
        ),
        (
            "MLP",
            [
                ("forward", "mlp_forward"),
                ("train step", "mlp_train_step"),
            ],
        ),
        (
            "Conv2d",
            [
                ("forward", "conv2d_forward"),
            ],
        ),
    ]

    LW = 25
    CW = max(12, max(len(fw) for fw in frameworks) + 2)
    NF = len(frameworks)

    def hline(l, m, r):
        return l + "─" * (LW + 2) + (m + "─" * CW) * NF + r

    def data_row(label, vals, indent=3):
        row = "│ " + (" " * indent + label).ljust(LW) + " "
        for v in vals:
            row += "│" + v.rjust(CW - 1) + " "
        return row + "│"

    print(f"\n  Results (median · lower ms = faster · higher GF/s = better)\n")
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
