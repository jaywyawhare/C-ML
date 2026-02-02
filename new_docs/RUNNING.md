# How to Run CML Programs

This guide explains how to run the compiled CML library, examples, and your own programs.

## Table of Contents

1. [Running the Main Executable](#running-the-main-executable)
2. [Running Examples](#running-examples)
3. [Running Tests](#running-tests)
4. [Using the Library](#using-the-library)
5. [Environment Variables](#environment-variables)
6. [Visualization](#visualization)
7. [Benchmarking](#benchmarking)
8. [Troubleshooting](#troubleshooting)

## Running the Main Executable

The main executable is built during compilation:

```bash
./build/main
```

The main executable demonstrates the core CML functionality with example operations.

### Output

You should see output similar to:

```
C-ML Library Initialized
Device: CPU
Data type: FLOAT32
[Running demonstrations...]
```

## Running Examples

CML includes comprehensive examples demonstrating different features.

### List Available Examples

```bash
ls -la build/examples/
```

Common examples:

| Example | Description |
|---------|-------------|
| `autograd_example` | Automatic differentiation demonstrations |
| `training_loop_example` | Training a simple neural network |
| `test_example` | Comprehensive feature test |
| `mnist_example` | MNIST digit recognition |
| `hello_cml` | Minimal "Hello CML" example |
| `simple_xor` | XOR problem training example |
| `benchmark` | Performance benchmarks |

### Running an Example

```bash
# Basic example
./build/examples/hello_cml

# More complex example
./build/examples/training_loop_example

# MNIST example (requires data)
./build/examples/mnist_example
```

### Running All Examples

```bash
for example in build/examples/*; do
    if [[ -x "$example" ]]; then
        echo "=== Running $example ==="
        "$example"
        echo ""
    fi
done
```

### Example: Autograd Demonstration

```bash
./build/examples/autograd_example
```

This example shows:
- Tensor creation and initialization
- Forward pass computation
- Automatic gradient computation
- Backward pass (backpropagation)

Expected output:
```
Testing tensor creation...
Forward pass: result = [...]
Backward pass: gradients computed
```

### Example: Training Loop

```bash
./build/examples/training_loop_example
```

This example demonstrates:
- Building a neural network
- Creating an optimizer
- Training loop with loss computation
- Gradient updates

Expected output:
```
Epoch 1: Loss = 0.523456
Epoch 2: Loss = 0.487123
...
Training complete
```

### Example: MNIST Recognition

```bash
./build/examples/mnist_example
```

Prerequisites:
- MNIST data must be in `data/` directory
- Training takes several minutes

Output:
```
Loading MNIST data...
Building model...
Epoch 1: Loss = 2.301, Accuracy = 0.15
Epoch 2: Loss = 1.923, Accuracy = 0.42
...
Final accuracy: 0.97
```

## Running Tests

CML includes comprehensive tests to verify functionality.

### Run All Tests

```bash
# Using Makefile
make test

# Or directly
cd build && ctest --output-on-failure
```

### Run Specific Test

```bash
./build/test_dispatch
./build/test_backends
./build/test_kernel_cache
```

### Test Output

Successful tests produce:
```
Test 1: PASSED
Test 2: PASSED
...
All tests passed!
```

Failed tests show details:
```
Test 1: FAILED
Error: Expected X, got Y
File: src/test.c, line 42
```

### Backend Tests

To test different backends (CPU, CUDA, etc.):

```bash
./build/test_backends

# Output shows which backends are available:
Backend Test Results:
  CPU:    PASSED
  CUDA:   SKIPPED (not available)
  Metal:  PASSED (macOS only)
  ROCm:   SKIPPED (not available)
```

## Using the Library

### Linking Against Static Library

Create your own program and link against the static library:

```c
// myprogram.c
#include "cml.h"
#include <stdio.h>

int main(void) {
    cml_init();

    // Your code here
    Tensor* x = tensor_randn((int[]){10, 10}, 2, NULL);
    printf("Created tensor of shape [10, 10]\n");

    tensor_free(x);
    cml_cleanup();
    return 0;
}
```

Compile and run:

```bash
# Compile
gcc -std=c11 -O2 myprogram.c \
    -I./include \
    -L./build/lib \
    -lcml \
    -lm \
    -o myprogram

# Run
./myprogram
```

### Linking Against Shared Library

```bash
# Compile with shared library
gcc -std=c11 -O2 myprogram.c \
    -I./include \
    -L./build/lib \
    -lcml \
    -lm \
    -o myprogram

# Set library path and run
export LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH
./myprogram
```

### With CMake

Create `CMakeLists.txt` for your project:

```cmake
cmake_minimum_required(VERSION 3.16)
project(MyMLApp)

find_package(cml REQUIRED)

add_executable(myapp myprogram.c)
target_link_libraries(myapp cml)
target_include_directories(myapp PRIVATE ${CML_INCLUDE_DIR})
```

Then:

```bash
cmake -DCML_DIR=/path/to/cml/build ..
make
./myapp
```

## Environment Variables

Control CML behavior with environment variables:

### Logging

```bash
# Set logging level (0=SILENT, 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG)
CML_LOG_LEVEL=4 ./build/examples/training_loop_example

# Verbose output
CML_LOG_LEVEL=4 ./build/examples/training_loop_example
```

### Device Selection

```bash
# Force CPU execution
CML_DEVICE=cpu ./build/examples/training_loop_example

# Use CUDA GPU (if available)
CML_DEVICE=cuda ./build/examples/training_loop_example

# Use Metal (macOS only)
CML_DEVICE=metal ./build/examples/training_loop_example

# Use ROCm (AMD GPU)
CML_DEVICE=rocm ./build/examples/training_loop_example
```

### Memory

```bash
# Set memory pool size (in MB)
CML_MEMORY=1024 ./build/examples/training_loop_example

# Enable memory debugging
CML_MEMORY_DEBUG=1 ./build/examples/training_loop_example
```

### MLIR Backend

```bash
# Enable MLIR debugging
CML_MLIR_DEBUG=1 ./build/examples/training_loop_example

# Dump generated kernels
CML_DUMP_KERNELS=1 ./build/examples/training_loop_example
```

### Library Path (Linux/macOS)

```bash
# Add custom library paths
export LD_LIBRARY_PATH=/path/to/libs:$LD_LIBRARY_PATH

# Check loaded libraries
ldd ./build/main
```

## Visualization

CML includes a visualization dashboard for training monitoring.

### Starting Visualization

```bash
# Enable visualization with VIZ environment variable
VIZ=1 ./build/examples/training_loop_example
```

The server starts on `http://localhost:8001`

### Custom Port

```bash
# Change visualization port
VIZ_PORT=9000 VIZ=1 ./build/examples/training_loop_example

# Access at http://localhost:9000
```

### Manual Server Start

```bash
# Start FastAPI server separately
python3 scripts/fastapi_server.py

# In another terminal, run your program
export VIZ=1
./build/examples/training_loop_example
```

### Visualization Features

The dashboard displays:
- Training loss curve
- Metrics tracking
- Computation graph
- Memory usage
- Device information
- Generated kernel code

## Benchmarking

### Running Benchmarks

```bash
# Build benchmark programs
make all

# Run GEMM benchmark
./build/examples/bench_gemm

# Output:
# Matrix multiplication benchmark:
# Size 128x128: 1.23ms
# Size 256x256: 2.45ms
# Size 512x512: 9.87ms
# Size 1024x1024: 78.9ms
```

### Performance Comparison

Create a benchmark program:

```c
#include "cml.h"
#include <time.h>
#include <stdio.h>

double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(void) {
    cml_init();

    int sizes[] = {128, 256, 512, 1024};

    for (int i = 0; i < 4; i++) {
        int size = sizes[i];

        Tensor* a = tensor_randn((int[]){size, size}, 2, NULL);
        Tensor* b = tensor_randn((int[]){size, size}, 2, NULL);

        double start = get_time();
        Tensor* c = tensor_matmul(a, b);
        double elapsed = get_time() - start;

        printf("%dx%d: %.3fms\n", size, size, elapsed * 1000);

        tensor_free(a);
        tensor_free(b);
        tensor_free(c);
    }

    cml_cleanup();
    return 0;
}
```

Compile and run:

```bash
gcc -std=c11 -O3 benchmark.c -I./include -L./build/lib -lcml -lm -o benchmark
./benchmark
```

### Profiling with perf (Linux)

```bash
# Record performance data
perf record -g ./build/examples/training_loop_example

# View results
perf report

# Generate flame graph
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
```

### Memory Profiling

```bash
# Using valgrind
valgrind --leak-check=full ./build/examples/training_loop_example

# Using asan (if built with debug)
./build/examples/training_loop_example
```

## Troubleshooting

### Program Won't Start

```
./build/examples/training_loop_example: command not found
```

Solution:
```bash
# Ensure build directory exists and is up to date
make clean
make

# Try with explicit path
./build/examples/training_loop_example
```

### Missing MLIR Library

```
./build/examples/training_loop_example: error while loading shared libraries: libMLIR.so.18: cannot open shared object file
```

Solution:
```bash
# Add MLIR library to path
export LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:$LD_LIBRARY_PATH
./build/examples/training_loop_example

# Or build statically
make clean && make release
```

### Program Crashes

```bash
# Enable debug logging
CML_LOG_LEVEL=4 ./build/examples/training_loop_example

# Run with address sanitizer (if built with debug)
./build/examples/training_loop_example
# Shows memory errors if any

# Or use valgrind
valgrind ./build/examples/training_loop_example
```

### Out of Memory

```bash
# Reduce batch size in your code
# Or limit memory:
CML_MEMORY=512 ./build/examples/training_loop_example
```

### GPU Not Detected

```bash
# Check GPU availability
CML_LOG_LEVEL=4 ./build/examples/training_loop_example

# Falls back to CPU if GPU unavailable
```

### Visualization Not Working

```bash
# Check if Python FastAPI is installed
python3 -m pip install fastapi uvicorn

# Test server manually
python3 scripts/fastapi_server.py

# Access http://localhost:8001 in browser
```

### Performance Issues

```bash
# Use release build for better performance
make release
./build/examples/training_loop_example

# Or fast build with CPU-specific optimizations
make fast
./build/examples/training_loop_example

# Enable MLIR debugging to check generated kernels
CML_DUMP_KERNELS=1 ./build/examples/training_loop_example
```

## Next Steps

- [QUICK_START.md](QUICK_START.md) - Get started with your first program
- [API_GUIDE.md](API_GUIDE.md) - Complete API reference
- [ARCHITECTURE.md](ARCHITECTURE.md) - Understand library architecture
