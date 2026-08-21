import json
import os
import subprocess
import sys

from bench_common import (
    bench_torch,
    median_of,
    now,
    parse_json_results,
    print_host_info,
    print_results_table,
    run_c_binary,
)

def bench_tinygrad(device=None):
    # `device` names a tinygrad backend ("CPU", "CL", ...); None benchmarks
    # whatever tinygrad picks for itself (OpenCL GPU if available, else CPU).
    # Uses TinyJit so each benchmark times repeated dispatch of a compiled kernel,
    # not graph construction or numpy wrapping.
    import numpy as np
    from tinygrad import Tensor as TgTensor
    from tinygrad import nn as tg_nn
    from tinygrad.engine.jit import TinyJit
    from tinygrad.device import Device

    original_device = Device.DEFAULT
    if device is not None:
        try:
            Device.DEFAULT = device
        except Exception:
            return None
    try:
        return _bench_tinygrad_body(np, TgTensor, tg_nn, TinyJit)
    finally:
        Device.DEFAULT = original_device

def _bench_tinygrad_body(np, TgTensor, tg_nn, TinyJit):
    results = {}

    RUNS = 5
    ITERS = 5  # iters per run() call

    # Each pair is used exactly once across all runs*iters calls — tinygrad never
    # revisits a realized output, so every call dispatches the compiled kernel for
    # real. That makes the pool, not the loop, the memory cost: at N=2048 a pair is
    # 32MB, so full ITERS would hold ~1.8GB live. Fewer iters at 2048 keeps the
    # one-use property while halving the pool.
    def _iters_for(N):
        return ITERS if N <= 1024 else 2

    def _pool_size(N):
        return RUNS * _iters_for(N) + 3  # 3 warmup + runs*iters

    for N in [512, 1024, 2048]:
        iters = _iters_for(N)
        pool = [(TgTensor(np.random.randn(N, N).astype(np.float32)),
                 TgTensor(np.random.randn(N, N).astype(np.float32))) for _ in range(_pool_size(N))]
        for a, b in pool:
            a.realize(); b.realize()

        @TinyJit
        def gemm_jit(a, b): return (a @ b).realize()

        it = iter(pool)
        for a, b in [next(it) for _ in range(3)]:  # warmup + JIT compile
            gemm_jit(a, b)

        def run(it=it, iters=iters):
            pairs = [next(it) for _ in range(iters)]
            t0 = now()
            for a, b in pairs:
                out = gemm_jit(a, b)
            out.numpy()  # force GPU sync — OpenCL dispatch is async
            return (now() - t0) / iters * 1e3

        results[f"gemm_{N}"] = median_of(run, runs=RUNS)
        del pool, it  # release before building the next, larger pool

    for N in [512, 1024, 2048]:
        iters = _iters_for(N)
        pool = [(TgTensor(np.random.randn(N, N).astype(np.float32)),
                 TgTensor(np.random.randn(N, N).astype(np.float32)),
                 TgTensor(np.random.randn(1, N).astype(np.float32))) for _ in range(_pool_size(N))]
        for a, b, bias in pool:
            a.realize(); b.realize(); bias.realize()

        @TinyJit
        def fused_jit(a, b, bias): return (a @ b + bias).relu().realize()

        it = iter(pool)
        for a, b, bias in [next(it) for _ in range(3)]:
            fused_jit(a, b, bias)

        def run(it=it, iters=iters):
            pairs = [next(it) for _ in range(iters)]
            t0 = now()
            for a, b, bias in pairs:
                out = fused_jit(a, b, bias)
            out.numpy()
            return (now() - t0) / iters * 1e3

        results[f"fused_{N}"] = median_of(run, runs=RUNS)
        del pool, it

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

    # Not JIT'd and not pooled: a training step mutates parameters, so replaying
    # it from a pool would measure a different computation each time. Tensors are
    # built fresh per step and dropped, or the retained graphs exhaust memory.
    import gc
    from tinygrad.nn import optim as tg_optim

    train_model = TgMLP()
    optimizer = tg_optim.SGD(tg_nn.state.get_parameters(train_model), lr=0.01)
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
    TgTensor.training = False
    gc.collect()

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

def bench_tensorflow():
    """TensorFlow, benchmarked in graph mode (tf.function).

    Graph mode is TF's normal performance path — eager TF pays a large per-op
    Python dispatch cost that says more about the binding than the kernels. Note
    the asymmetry when reading the table: PyTorch here is eager, TF and tinygrad
    are compiled. Conv2d uses NHWC, TF's native CPU layout (torch/cml use NCHW);
    same FLOPs, and forcing NCHW on TF CPU would be an artificial handicap.
    """
    import numpy as np
    import tensorflow as tf

    n = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
    try:
        tf.config.threading.set_intra_op_parallelism_threads(n)
        tf.config.threading.set_inter_op_parallelism_threads(n)
    except RuntimeError:
        pass  # already initialised; keep whatever TF picked

    results = {}

    matmul = tf.function(lambda a, b: tf.matmul(a, b))
    for N in [512, 1024, 2048]:
        a = tf.constant(np.random.randn(N, N).astype(np.float32))
        b = tf.constant(np.random.randn(N, N).astype(np.float32))
        for _ in range(3):
            matmul(a, b)

        def run(a=a, b=b):
            t0 = now()
            for _ in range(5):
                matmul(a, b)
            return (now() - t0) / 5 * 1e3

        results[f"gemm_{N}"] = median_of(run, runs=5)

    fused = tf.function(lambda a, b, bias: tf.nn.relu(tf.matmul(a, b) + bias))
    for N in [512, 1024, 2048]:
        a = tf.constant(np.random.randn(N, N).astype(np.float32))
        b = tf.constant(np.random.randn(N, N).astype(np.float32))
        bias = tf.constant(np.random.randn(1, N).astype(np.float32))
        for _ in range(3):
            fused(a, b, bias)

        def run(a=a, b=b, bias=bias):
            t0 = now()
            for _ in range(5):
                fused(a, b, bias)
            return (now() - t0) / 5 * 1e3

        results[f"fused_{N}"] = median_of(run, runs=5)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(784,)),
        tf.keras.layers.Dense(10),
    ])
    x = tf.constant(np.random.randn(64, 784).astype(np.float32))

    infer = tf.function(lambda inp: model(inp, training=False))
    for _ in range(5):
        infer(x)

    def run_fwd():
        t0 = now()
        for _ in range(100):
            infer(x)
        return (now() - t0) / 100 * 1e3

    results["mlp_forward"] = median_of(run_fwd, runs=5)

    target = tf.constant(np.random.randn(64, 10).astype(np.float32))
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)

    @tf.function
    def train_step(inp, tgt):
        with tf.GradientTape() as tape:
            out = model(inp, training=True)
            loss = tf.reduce_mean(tf.square(out - tgt))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for _ in range(5):
        train_step(x, target)

    def run_train():
        t0 = now()
        for _ in range(50):
            train_step(x, target)
        return (now() - t0) / 50 * 1e3

    results["mlp_train_step"] = median_of(run_train, runs=5)

    conv = tf.keras.layers.Conv2D(16, 3, input_shape=(32, 32, 3))
    x_conv = tf.constant(np.random.randn(8, 32, 32, 3).astype(np.float32))
    conv_fn = tf.function(lambda inp: conv(inp, training=False))
    for _ in range(5):
        conv_fn(x_conv)

    def run_conv():
        t0 = now()
        for _ in range(100):
            conv_fn(x_conv)
        return (now() - t0) / 100 * 1e3

    results["conv2d_forward"] = median_of(run_conv, runs=5)

    return results

def best_of(fn, repeat):
    """Run an engine `repeat` times and keep the fastest time per benchmark.

    Between-run contention on a shared machine inflates times by 2x or more,
    and it inflates them asymmetrically across engines because the engines run
    sequentially. Medians do not remove that -- the minimum does, because the
    fastest observed run is the one least disturbed by other load. Set
    CML_BENCH_REPEAT=1 for a single pass.
    """
    best = {}
    for i in range(repeat):
        res = fn() or {}
        for k, v in res.items():
            if v is None:
                best.setdefault(k, None)
            elif best.get(k) is None:
                best[k] = v
            else:
                best[k] = min(best[k], v)
        if repeat > 1:
            print(f"    pass {i + 1}/{repeat} done")
    return best

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

    # NumPy has no layers, so the MLP and conv rows are written out by hand.
    # They are still the honest NumPy baseline: the same arithmetic the
    # frameworks do, expressed the fastest way NumPy can express it.
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

    # Train step with hand-written backward and SGD -- no autograd tape, so
    # this is the floor the autograd frameworks are paying overhead against.
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

    weight_c = np.random.randn(16, 3, 3, 3).astype(np.float32)
    x_conv = np.random.randn(8, 3, 32, 32).astype(np.float32)

    def numpy_conv2d(x, w):
        N, C, H, W_in = x.shape
        F, C, KH, KW = w.shape
        OH, OW = H - KH + 1, W_in - KW + 1
        # im2col via stride tricks -- no copy, so the einsum below is the cost.
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

def bench_engine_subprocess(python_exe, engine, timeout=1800):
    """Run one engine under a different interpreter and collect its JSON.

    TensorFlow ships no wheels for every Python the other frameworks run on, so
    it lives in its own venv; this re-invokes this same script there with
    --engine and reads the results back.
    """
    if not os.path.isfile(python_exe):
        return {}
    try:
        result = subprocess.run(
            [python_exe, os.path.abspath(__file__), "--engine", engine],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"},
        )
        if result.returncode != 0:
            print(f"  {engine} run failed: {(result.stderr or '').strip()[-300:]}")
            return {}
        return parse_json_results(result.stdout, engine)
    except subprocess.TimeoutExpired:
        print(f"  {engine} run timed out")
        return {}

def ensure_fresh_binary(binary_path):
    """Rebuild the bench binary from its build tree before running.

    Benchmarking a stale binary is a silent correctness trap: a months-old build
    once reported conv2d at 0.002ms (a no-op that had since been fixed) simply
    because the driver never rebuilt it. If the binary lives under a CMake build
    tree (…/<build>/bin/bench_cross_framework), rebuild that target; on any
    failure fall back to whatever binary already exists. Set CML_BENCH_NO_BUILD=1
    to skip (e.g. when benchmarking a deliberately-pinned binary).
    """
    if os.environ.get("CML_BENCH_NO_BUILD"):
        return
    bin_dir = os.path.dirname(binary_path)                       # …/<build>/bin
    build_dir = os.path.dirname(bin_dir)                         # …/<build>
    if not os.path.isfile(os.path.join(build_dir, "CMakeCache.txt")):
        if not os.path.isfile(binary_path):
            print(f"  ! No build tree at {build_dir} and no binary at {binary_path}")
        return
    print(f"Building bench_cross_framework in {build_dir} ...")
    try:
        r = subprocess.run(
            ["cmake", "--build", build_dir, "--target", "bench_cross_framework",
             "-j", str(os.cpu_count() or 1)],
            capture_output=True, text=True, timeout=1800,
        )
        if r.returncode != 0:
            print("  ! Build failed; using existing binary if present:")
            print("   " + (r.stderr.strip().splitlines() or ["<no stderr>"])[-1])
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  ! Could not build ({e}); using existing binary if present.")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    cml_binary = os.environ.get(
        "CML_BENCH_BINARY",
        os.path.join(project_dir, "build_release", "bin", "bench_cross_framework")
    )
    ensure_fresh_binary(cml_binary)

    print_host_info("Cross-Framework Benchmark")

    REPEAT = int(os.environ.get("CML_BENCH_REPEAT", "3"))
    if REPEAT > 1:
        print(f"Best-of-{REPEAT} per engine (guards against machine load skewing a single pass)")

    all_results = {}

    try:
        import numpy as np
        print(f"\nRunning NumPy {np.__version__} benchmarks...")
        all_results["numpy"] = best_of(bench_numpy, REPEAT)
    except ImportError:
        print("\nNumPy not available, skipping")

    print(f"\nRunning CML (CPU) benchmarks...")
    all_results["cml"] = best_of(lambda: run_c_binary(cml_binary, "CML"), REPEAT)

    print(f"\nRunning CML (GPU) benchmarks...")
    cml_gpu = best_of(lambda: run_c_binary(cml_binary, "CML", {"BACKEND": "opencl"}), REPEAT)
    if cml_gpu:
        all_results["cml(OpenCL)"] = cml_gpu

    # Only meaningful on Apple: elsewhere DEVICE_METAL falls back to plain host
    # allocation, so the column would report CPU numbers under a GPU label.
    if sys.platform == "darwin":
        print(f"\nRunning CML (Metal) benchmarks...")
        cml_metal = best_of(lambda: run_c_binary(cml_binary, "CML", {"BACKEND": "metal"}), REPEAT)
        if cml_metal:
            all_results["cml(Metal)"] = cml_metal

    try:
        import torch
        print(f"\nRunning PyTorch {torch.__version__} benchmarks...")
        all_results["pytorch"] = best_of(bench_torch, REPEAT)
        if torch.cuda.is_available():
            print(f"\nRunning PyTorch (CUDA) benchmarks...")
            cuda_res = best_of(lambda: bench_torch("cuda"), REPEAT)
            if cuda_res:
                all_results["pytorch(CUDA)"] = cuda_res
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
        print(f"\nRunning TinyGrad {ver} benchmarks (device: {dev})...")
        all_results[f"tinygrad({dev})"] = best_of(bench_tinygrad, REPEAT)
        # Its default is the GPU where there is one, which makes the column
        # incomparable to the CPU engines beside it; run CPU too so both axes
        # are on the table.
        if dev != "CPU":
            print(f"\nRunning TinyGrad (CPU) benchmarks...")
            tg_cpu = best_of(lambda: bench_tinygrad("CPU"), REPEAT)
            if tg_cpu:
                all_results["tinygrad(CPU)"] = tg_cpu
    except ImportError:
        print("\nTinyGrad not available, skipping")
    except Exception as e:
        print(f"\nTinyGrad failed: {e}")

    try:
        import tensorflow as tf
        print(f"\nRunning TensorFlow {tf.__version__} benchmarks...")
        all_results["tensorflow"] = best_of(bench_tensorflow, REPEAT)
    except ImportError:
        # TF often can't share an interpreter with the rest (it lags on new
        # Python releases), so fall back to a dedicated venv.
        tf_python = os.environ.get("CML_BENCH_TF_PYTHON", "/tmp/tfbench-venv/bin/python")
        if os.path.isfile(tf_python):
            print(f"\nRunning TensorFlow benchmarks (out-of-process: {tf_python})...")
            tf_res = best_of(lambda: bench_engine_subprocess(tf_python, "tensorflow"), REPEAT)
            if tf_res:
                all_results["tensorflow"] = tf_res
        else:
            print("\nTensorFlow not available, skipping"
                  " (set CML_BENCH_TF_PYTHON to a python that has it)")

    ordered = ["numpy", "cml", "pytorch", "tensorflow"]
    frameworks = [k for k in ordered if k in all_results]
    frameworks += [k for k in all_results if k not in ordered]

    if not frameworks:
        print("\nNo results to display.")
        return

    print_results_table(all_results, frameworks)
    print()

ENGINE_FNS = {
    "numpy": bench_numpy,
    "pytorch": bench_torch,
    "tinygrad": bench_tinygrad,
    "tensorflow": bench_tensorflow,
}

def run_single_engine(engine):
    """Run one engine and print its results as JSON on stdout.

    Accepts "<engine>" or "<engine>:<device>" (e.g. pytorch:cuda), so the
    device axis survives the trip through a subprocess.
    """
    name, _, device = engine.partition(":")
    fn = ENGINE_FNS.get(name)
    if not fn:
        print(f"unknown engine: {name}", file=sys.stderr)
        return 2
    print(json.dumps(fn(device) if device else fn()))
    return 0

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--engine":
        sys.exit(run_single_engine(sys.argv[2]))
    main()
