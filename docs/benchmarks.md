# Benchmarks

C-ML includes benchmark programs to measure performance of key operations.

## Running Benchmarks

Build the project with examples enabled (default), then run:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Forward pass benchmark
./build/bin/bench_forward

# GEMM (matrix multiplication) benchmark
./build/bin/bench_gemm

# Backend comparison benchmark
./build/bin/bench_backends

# Cross-framework benchmark (CML vs PyTorch vs TinyGrad vs NumPy)
python3 benchmarks/bench_all.py
```

## Available Benchmarks

### bench_forward

Measures end-to-end forward pass latency for a simple neural network:

- **Model**: Linear(784, 128) -> ReLU -> Linear(128, 10)
- **Batch size**: 64
- **Iterations**: 100 (with 5 warmup iterations)
- **Reports**: Total time, per-forward latency, throughput (samples/sec)
- Also prints SIMD capabilities, BLAS library info, graph cache stats, and execution stats

### bench_gemm

Measures matrix multiplication (GEMM) throughput, comparing multiple implementations:

- **Naive**: Triple-loop implementation (only for small sizes)
- **Raw BLAS**: Direct `cblas_sgemm` call
- **CML tensor_matmul**: CML's tensor matmul path (sizes up to 1024)
- **Fused**: matmul + bias + relu chain, comparing BLAS unfused vs CML fused (sizes up to 1024)
- **Matrix sizes**: 2048x2048 and 4096x4096 (square only)
- **Reports**: GFLOPS (billions of floating-point operations per second)

### bench_backends

Comprehensive backend comparison benchmark (located in `tests/`):

- **Operations**: Matrix multiplication, kernel cache, element-wise ops
- **Backends**: CPU fallback, BLAS, CPU LLVM JIT, CUDA
- **Also measures**: Dispatch overhead

### bench_cross_framework (C binary + Python driver)

A rigorous cross-framework comparison covering five workloads on float32:

| Benchmark | Details |
|-----------|---------|
| GEMM | Square NxN matmul, N ∈ {512, 1024, 2048} |
| Fused | matmul + bias_add + relu, same sizes |
| MLP forward | batch=64, 784→128→ReLU→10, 100 iters |
| MLP training step | forward + MSE loss + backward + SGD, 50 iters |
| Conv2d forward | batch=8, 3×32×32 → 16×30×30, 100 iters |

**C binary** (`benchmarks/bench_cross_framework.c`):
- Compiles to `build/bin/bench_cross_framework`
- Outputs JSON so the Python driver can parse it
- Respects `CML_BACKEND=opencl` to benchmark the GPU path
- Uses median of 5 rounds; 3 iterations/round for N=2048 to reduce thermal throttle
- Includes cooldown sleeps between heavy 2048-size runs

```bash
# CPU benchmark (JSON output)
./build/bin/bench_cross_framework

# GPU benchmark via OpenCL
CML_BACKEND=opencl ./build/bin/bench_cross_framework
```

**Python driver** (`benchmarks/bench_all.py`):
- Runs CML (CPU), CML (GPU/OpenCL), PyTorch, TinyGrad, and NumPy in sequence
- All frameworks use all available cores by default; override with `OMP_NUM_THREADS=N`
- TinyGrad benchmarks use `TinyJit` for compiled-kernel dispatch and `.numpy()` to force GPU sync
- Results grouped by operation type (GEMM, Fused, MLP, Conv2d), median ms, lower is better

```bash
# Run all frameworks (all cores)
python3 benchmarks/bench_all.py

# Single-threaded run for controlled comparison
OMP_NUM_THREADS=1 python3 benchmarks/bench_all.py

# Example output:
#   Cross-Framework Benchmark  (float32, 12 threads)
#
#   Results  (median ms, lower is better)
#
#   Benchmark                      numpy    cml  pytorch  cml(GPU)  tinygrad(CL)
#
#   GEMM
#     512x512                      2.0ms   2.2ms    2.4ms    0.5ms        1.4ms
#     1024x1024                   16.5ms  18.3ms   20.2ms    3.3ms        6.7ms
#     2048x2048                  131.3ms 148.8ms  174.0ms   34.7ms       79.2ms
#   ...
#
#   GFLOPS  (GEMM)
#     512                         134GF   122GF    112GF    351GF        191GF
#   ...
```

PyTorch and TinyGrad are optional — the script skips frameworks that aren't installed.

## Methodology

- Each benchmark includes a warm-up phase before timed iterations
- Timing uses `clock_gettime(CLOCK_MONOTONIC)` for high-resolution wall-clock measurements
- Results report average latency (total time / iterations)

## Reference Numbers

Performance varies between runs and across hardware. These numbers were measured
on an AVX-512 capable CPU with OpenBLAS:

### Forward Pass (bench_forward)

| Model | Batch | Per-forward (ms) | Throughput (samples/sec) |
|-------|-------|-------------------|--------------------------|
| Linear(784,128)->ReLU->Linear(128,10) | 64 | ~5.5 | ~11,627 |

### GEMM Throughput (bench_gemm)

| Implementation | N=2048 (ms) | GFLOPS | N=4096 (ms) | GFLOPS |
|----------------|-------------|--------|-------------|--------|
| Raw BLAS (cblas_sgemm) | ~276 | ~62 | ~1,694 | ~81 |
| BLAS + manual bias+relu | ~435 | ~40 | ~2,139 | ~64 |

*Note: CML tensor paths (tensor_matmul, fused) are benchmarked only for sizes
up to 1024 due to IR/graph overhead at larger sizes. Run the benchmarks on your
own hardware for complete results.*

### Backend Comparison (bench_backends)

**Dispatch Overhead** (1,000 iterations):

| Operation | Latency (us/op) |
|-----------|-----------------|
| Context create/free | ~0.4 |
| Context init | ~165 |
| Backend detection | ~86 |

**Kernel Cache** (10,000 iterations):

| Operation | Latency (us/op) |
|-----------|-----------------|
| Insert | ~6.1 |
| Lookup hit | ~0.03 |
| Lookup miss | ~0.12 |

**Matrix Multiply** (256x256, 10 iterations):

| Backend | Avg Latency (ms) |
|---------|-------------------|
| CPU Fallback | ~1.4 |
| BLAS (OpenBLAS) | ~4.4 |

*Note: At small sizes (256x256), CPU fallback can be faster than BLAS due to
library call overhead. BLAS outperforms CPU at larger sizes (2048+).*

**Element-wise Ops** (100,000 elements, 10 iterations):

| Operation | Avg Latency (ms) |
|-----------|-------------------|
| Add | ~0.95 |
| Mul | ~0.67 |
| Exp | ~2.0 |

## Adding New Benchmarks

Create a new file in `examples/benchmarks/` following the existing pattern:

```c
#include "cml.h"
#include <time.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    cml_init();

    // ... warmup: run operation a few times ...

    double start = get_time_sec();
    int iters = 100;
    for (int i = 0; i < iters; i++) {
        // ... operation under test ...
    }
    double elapsed = get_time_sec() - start;

    printf("Operation: %.3f ms\n", (elapsed / iters) * 1000);

    cml_cleanup();
    return 0;
}
```

Then add it to `CMakeLists.txt`:

```cmake
cml_add_example(bench_my_op examples/benchmarks/bench_my_op.c)
```
