# Benchmarks

C-ML includes several benchmark programs to measure performance of key operations.

## Running Benchmarks

Build the project with examples enabled (default), then run:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Forward pass benchmark
./build/bin/bench_forward

# GEMM (matrix multiplication) benchmark
./build/bin/bench_gemm
```

## Available Benchmarks

### bench_forward

Measures forward pass latency for common layer configurations:

- **Linear layers**: Various input/output sizes (128x256, 512x1024, etc.)
- **Conv2d layers**: Different kernel sizes and channel counts
- **Activation functions**: ReLU, Sigmoid, Tanh
- **Normalization**: BatchNorm2d, LayerNorm

### bench_gemm

Measures matrix multiplication (GEMM) performance:

- **Square matrices**: 128x128 through 2048x2048
- **Rectangular matrices**: Common ML shapes (batch x features)
- **Backend comparison**: CPU vs LLVM JIT (when available)

Reports GFLOPS (billions of floating-point operations per second).

## Methodology

- Each operation is run multiple times with a warm-up phase
- Timing uses `clock_gettime(CLOCK_MONOTONIC)` for high-resolution measurements
- Results report median latency to reduce variance from system noise
- Memory allocation is excluded from timing where possible

## Operations Benchmarked

| Category | Operations |
|----------|-----------|
| Element-wise | add, mul, exp, log, sqrt, relu, sigmoid, tanh |
| Reduction | sum, mean, max, min |
| Linear algebra | matmul (GEMM), transpose |
| Convolution | conv2d forward, conv2d backward |
| Normalization | batchnorm2d, layernorm |

## Reference Numbers

Performance varies significantly by hardware. Below are placeholder numbers
from a typical development machine (update with your own results):

| Operation | Size | CPU (ms) | LLVM JIT (ms) |
|-----------|------|----------|----------------|
| matmul | 512x512 | ~5 | ~2 |
| matmul | 1024x1024 | ~35 | ~12 |
| conv2d | 32x3x224x224, k=3 | ~15 | ~8 |
| relu | 1M elements | ~0.5 | ~0.3 |
| batchnorm2d | 32x64x56x56 | ~2 | ~1 |

*Note: These are approximate and should be updated with actual measurements
on your target hardware.*

## Adding New Benchmarks

Create a new file in `examples/benchmarks/` following the existing pattern:

```c
#include "cml.h"
#include <time.h>

int main(void) {
    cml_init();

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // ... operation to benchmark ...

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0
                      + (end.tv_nsec - start.tv_nsec) / 1e6;

    printf("Operation: %.3f ms\n", elapsed_ms);
    return 0;
}
```

Then add it to `CMakeLists.txt`:

```cmake
cml_add_example(bench_my_op examples/benchmarks/bench_my_op.c)
```
