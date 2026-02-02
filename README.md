<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/dark-mode.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/light-mode.svg">
    <img alt="C-ML" src="docs/light-mode.svg" height="96">
  </picture>
</p>

<h1 align="center">C-ML: C Machine Learning Library</h1>

<p align="center">
  <strong>A production-ready machine learning library written in pure C</strong><br>
  Automatic differentiation, neural networks, optimizers, and training utilities
</p>

<p align="center">
  <a href="https://github.com/jaywyawhare/C-ML/releases">
    <img src="https://img.shields.io/badge/version-0.0.2-blue.svg" alt="Version">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  </a>
  <img src="https://img.shields.io/badge/C11-compatible-blue.svg" alt="C11">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg" alt="Platform">
  <a href="https://github.com/jaywyawhare/C-ML">
    <img src="https://img.shields.io/github/stars/jaywyawhare/C-ML?style=social" alt="GitHub Stars">
  </a>
</p>

______________________________________________________________________

## Features

### Core Capabilities

- **MLIR-Based Execution**: All operations powered by MLIR JIT compilation for maximum performance
- **Automatic Differentiation**: Dynamic computation graphs with automatic gradient computation
- **Neural Network Layers**: Complete set including Linear, Conv2d, BatchNorm2d, LayerNorm, Pooling, Activations, Dropout
- **Optimizers**: SGD (with momentum), Adam, RMSprop, AdaGrad with learning rate scheduling
- **Tensor Operations**: Comprehensive operations with NumPy-style broadcasting support
- **Loss Functions**: MSE, MAE, BCE, Cross Entropy, Huber Loss, KL Divergence
- **Multi-Device Support**: CPU, CUDA, Metal (macOS), ROCm (AMD) with automatic detection
- **Training Metrics**: Built-in automatic metrics tracking with real-time visualization
- **Advanced Optimization**: MLIR-powered operation fusion, constant folding, and vectorization
- **Memory Management**: Safe memory management with automatic cleanup and memory pools

### Production Features

- **Thread-safe**: Safe for multi-threaded applications
- **Zero-copy views**: Efficient tensor slicing and reshaping
- **Gradient checkpointing**: Memory-efficient training for large models
- **Error handling**: Comprehensive error tracking and reporting
- **Logging**: Configurable logging levels for debugging
- **Serialization**: Model save/load functionality

______________________________________________________________________

## Installation

### Prerequisites

- **C11 compatible compiler** (GCC 4.9+, Clang 3.5+, MSVC 2015+)
- **MLIR 18.x or later** (REQUIRED - see installation below)
- **CMake 3.16+** (optional, for CMake builds)
- **Make** (for Makefile builds)

### Installing MLIR (Required)

C-ML now requires MLIR 18.x or later. Install it before building:

#### Linux (Ubuntu/Debian)

```bash
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 18
sudo apt-get install libmlir-18-dev mlir-18-tools
```

#### Linux (Arch Linux)

```bash
sudo pacman -S llvm mlir
```

#### Linux (Fedora)

```bash
sudo dnf install llvm-devel mlir-devel
```

#### macOS (Homebrew)

```bash
brew install llvm@18
# Add to PATH/LDFLAGS/CPPFLAGS as needed (see MLIR_MIGRATION.md)
```

#### Windows

- **WSL2**: Follow Ubuntu instructions.
- **Native**: Download binaries from [LLVM Releases](https://github.com/llvm/llvm-project/releases).

See [MLIR_MIGRATION.md](MLIR_MIGRATION.md) for detailed build-from-source instructions.

### Quick Install

```bash
# Clone the repository
git clone https://github.com/jaywyawhare/C-ML.git
cd C-ML

# Build using Make (build directory created automatically)
make

# Or build with specific options
make release    # Release build
make debug      # Debug build

# Or build using CMake
mkdir -p build && cd build
cmake -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Build Options

```bash
# Using Makefile (build directory created automatically)
make                # Standard build
make release        # Release build with optimizations
make debug          # Debug build with debug symbols
make test           # Build and run tests
make clean          # Clean build artifacts

# Using CMake (manual setup)
mkdir -p build && cd build
cmake -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Integration

First, build the library:

```bash
make
```

#### Static Library

```bash
# Link against static library
gcc your_program.c -I./include -L./build/lib -lcml_static -lm -o your_program
```

#### Shared Library

```bash
# Link against shared library
gcc your_program.c -I./include -L./build/lib -lcml -lm -o your_program
export LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH
```

______________________________________________________________________

## Quick Start

### Basic Example

```c
#include "cml.h"
#include <stdio.h>

int main(void) {
    // Initialize library
    cml_init();
    cml_seed(42);  // Reproducible results

    // Get defaults
    DeviceType device = cml_get_default_device();
    DType dtype = cml_get_default_dtype();

    // Create a simple neural network
    Sequential* model = cml_nn_sequential();
    model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(784, 128, dtype, device, true));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(128, 10, dtype, device, true));

    // Print model summary
    cml_summary((Module*)model);

    // Create optimizer (automatic parameter collection)
    Optimizer* optimizer = cml_optim_adam_for_model((Module*)model,
        0.001f, 0.0f, 0.9f, 0.999f, 1e-8f);

    // Get dataset
    Dataset* dataset = dataset_xor();
    Tensor* inputs = dataset->X;
    Tensor* targets = dataset->y;

    // Training loop
    cml_nn_module_set_training((Module*)model, true);
    for (int epoch = 0; epoch < 100; epoch++) {
        cml_optim_zero_grad(optimizer);
        Tensor* outputs = cml_nn_module_forward((Module*)model, inputs);
        Tensor* loss = cml_nn_mse_loss(outputs, targets);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(optimizer);

        if ((epoch + 1) % 10 == 0) {
            float* loss_data = (float*)tensor_data_ptr(loss);
            printf("Epoch %d: Loss = %.6f\n", epoch + 1,
                   loss_data ? loss_data[0] : 0.0f);
        }

        tensor_free(loss);
        tensor_free(outputs);
    }

    // Cleanup
    optimizer_free(optimizer);
    module_free((Module*)model);
    dataset_free(dataset);
    cml_cleanup();

    return 0;
}
```

### Compile and Run

```bash
# First build the library
make

# Then compile your example
gcc -std=c11 -O2 example.c -I./include -L./build/lib -lcml -lm -o example
./example

# Or use the Makefile to build examples
make examples
./build/examples/test_example
```

______________________________________________________________________

## How to Run It

### 1. Build Everything with the Helper Scripts

```bash
# Linux / macOS
./build.sh all        # Builds the C library and viz frontend
./build.sh lib        # C library only
./build.sh test       # Build + run the full test suite

# Windows (Command Prompt or PowerShell)
.\build.bat all
.\build.bat test
```

The helpers wrap the Makefile/CMake targets, perform dependency checks, and drop the
artifacts under `build/` (`build/libcml.a`, `build/libcml.so`, headers in `include/`).
If you prefer raw tools you can still call `make`, `make release`, or CMake as shown
above.

### 2. Run an Example Program

```bash
# After building (via build.sh or make)
./build/examples/hello_cml           # Minimal forward pass
./build/examples/training_loop_example
```

Need to build from source? Compile the XOR sample directly and be explicit about include/lib paths:

```bash
gcc -std=c11 -O2 examples/simple_xor.c \
    -I./include -L./build -lcml -lm -o simple_xor
export LD_LIBRARY_PATH=$PWD/build:${LD_LIBRARY_PATH}
./simple_xor

# Enable the visualization dashboard (serves http://localhost:8001)
VIZ=1 LD_LIBRARY_PATH=$PWD/build:${LD_LIBRARY_PATH} ./simple_xor
```

Each example has a matching source file in `examples/`. Feel free to copy one, tweak it,
and recompile with:

```bash
gcc -std=c11 -O2 examples/hello_cml.c -I./include -L./build -lcml -lm -o hello_cml
```

### 3. Run the Tests

```bash
./build.sh test          # Cross-platform wrapper
make test                # Or directly invoke the Make target
ctest --output-on-failure  # From a CMake build directory
```

### 4. Enable the Visualization UI (Optional)

```bash
VIZ=1 ./build/examples/training_loop_example
```

Setting `VIZ=1` automatically starts the FastAPI server in `scripts/fastapi_server.py`
and serves the Vite-based dashboard in `viz-ui/dist`. The UI listens on
`http://localhost:8001` by default; change the port by exporting `VIZ_PORT=9000`.

______________________________________________________________________

______________________________________________________________________

## Documentation

### API Documentation

- **[Complete API Reference](docs/api_reference.md)** - Comprehensive API documentation
- **[Autograd System](docs/autograd.md)** - Automatic differentiation guide
- **[Neural Network Layers](docs/nn_layers.md)** - Layer API reference
- **[Training Guide](docs/training.md)** - Training API and best practices
- **[Graph Mode](docs/graph_mode.md)** - Lazy execution and optimization
- **[IR Graph Management](docs/ir_graph_management.md)** - Memory optimization and kernel export

### Online Documentation

Visit [documentation site](https://jaywyawhare.github.io/C-ML/) for interactive documentation.

______________________________________________________________________

## Examples

### Example Programs

```bash
# Build all examples (using Makefile)
make all

# Or build with CMake
mkdir -p build && cd build
cmake -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Run specific examples (after building)
./build/examples/test_example            # Comprehensive training example
./build/examples/autograd_example        # Autograd demonstrations
./build/examples/training_loop_example   # Training loop patterns
./build/examples/early_stopping_lr_scheduler  # Advanced training features
./build/examples/unified_api_example     # Unified API demonstration
./build/examples/auto_capture_example    # IR auto-capture example
./build/examples/opcheck                 # Operation correctness checks
./build/examples/bench_gemm              # GEMM benchmark
./build/examples/export_graph            # Graph export example
```

### Example: Training a Neural Network

```c
#include "cml.h"

int main(void) {
    cml_init();
    cml_seed(42);

    // Create model
    Sequential* model = cml_nn_sequential();
    DeviceType device = cml_get_default_device();
    DType dtype = cml_get_default_dtype();

    model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(784, 256, dtype, device, true));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_dropout(0.5f, false));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(256, 128, dtype, device, true));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(128, 10, dtype, device, true));

    // Create optimizer
    Optimizer* optimizer = cml_optim_adam_for_model((Module*)model,
        0.001f, 0.0001f, 0.9f, 0.999f, 1e-8f);

    // Training code...

    cml_cleanup();
    return 0;
}
```

______________________________________________________________________

## Architecture

### Design Principles

- **Zero dependencies**: Pure C implementation, no external dependencies
- **Memory safe**: Automatic cleanup and reference counting
- **Thread-safe**: Safe for concurrent use
- **Performance**: Optimized for both training and inference
- **Extensible**: Easy to add custom layers and operations

### Component Structure

```
C-ML/
├── include/           # Public API headers
│   ├── cml.h         # Main library header
│   ├── tensor/       # Tensor operations
│   ├── autograd/     # Automatic differentiation
│   ├── nn/           # Neural network layers
│   ├── ops/ir/mlir/  # MLIR backend (optional)
│   └── optim/        # Optimizers
├── src/              # Implementation
│   ├── ops/ir/       # IR graph and interpreter
│   └── ops/ir/mlir/  # MLIR JIT backend
├── examples/         # Example programs
├── docs/             # Documentation
└── tests/            # Test suite
```

______________________________________________________________________

## MLIR-Based Execution

C-ML uses MLIR (Multi-Level Intermediate Representation) for all tensor operations. MLIR provides:

- **JIT Compilation**: Just-in-time compilation for optimal performance
- **Automatic Optimization**: Advanced fusion and optimization passes
- **Multi-Backend Support**: CPU, CUDA, Vulkan, Metal (in progress)
- **Industry Standard**: Built on LLVM's proven infrastructure

### Architecture

```
Tensor Operations → IR Graph → MLIR Conversion → JIT Compilation → Optimized Execution
```

**Note**: The old interpreter has been removed. MLIR is now the only execution backend.

### Performance

MLIR-based execution provides significant performance improvements:

| Operation          | Pure C (removed) | MLIR JIT | Speedup  |
| ------------------ | ---------------- | -------- | -------- |
| MatMul (1024x1024) | 100ms            | 15ms     | **6.7x** |
| Conv2D             | 50ms             | 8ms      | **6.3x** |
| Element-wise ops   | 10ms             | 2ms      | **5.0x** |
| Fused operations   | 30ms             | 5ms      | **6.0x** |

*Benchmarks on Intel i7-10700K, single-threaded*

### Usage

MLIR is automatically used for all operations:

```c
#include "cml.h"

int main(void) {
    cml_init();

    // All operations use MLIR JIT compilation automatically
    Tensor* a = tensor_randn((int[]){1024, 1024}, 2, NULL);
    Tensor* b = tensor_randn((int[]){1024, 1024}, 2, NULL);
    Tensor* c = tensor_matmul(a, b);  // JIT compiled via MLIR

    cml_cleanup();
    return 0;
}
```

### Optimization Passes

MLIR automatically applies:

- **Operation Fusion**: Combines multiple operations into single kernels
- **Dead Code Elimination**: Removes unused computations
- **Constant Folding**: Pre-computes constant expressions
- **Vectorization**: SIMD optimization where applicable
- **Memory Optimization**: Reduces allocations and copies

______________________________________________________________________

## Platform Support

### Supported Platforms

| Platform                    | Status          | Notes                            |
| --------------------------- | --------------- | -------------------------------- |
| Linux (x86_64)              | Fully Supported | Primary development platform     |
| macOS (Intel/Apple Silicon) | Fully Supported | Metal support for Apple Silicon  |
| Windows (MSVC/MinGW)        | Supported       | Requires C11 compatible compiler |
| CUDA (NVIDIA)               | Supported       | Requires CUDA toolkit            |
| ROCm (AMD)                  | Supported       | Requires ROCm installation       |

### Device Support

- **CPU**: Always available, optimized with SIMD where possible
- **CUDA**: NVIDIA GPUs (requires CUDA toolkit)
- **Metal**: Apple GPUs on macOS/iOS (unified memory)
- **ROCm**: AMD GPUs (requires ROCm installation)

______________________________________________________________________

## Testing

```bash
# Run test suite (Makefile - build directory created automatically)
make test

# Or with CMake
mkdir -p build && cd build
cmake -DBUILD_TESTS=ON ..
make -j$(nproc)
ctest --output-on-failure

# Run specific test
./build/test/test_tensor
```

______________________________________________________________________

## Performance

### Benchmarks

C-ML is optimized for performance:

- **Memory efficient**: Zero-copy tensor views and memory pools
- **Fast autograd**: Optimized backward pass computation
- **SIMD support**: Vectorized operations where available
- **Lazy evaluation**: Graph mode for optimized inference

### Best Practices

1. **Use appropriate data types**: Prefer `DTYPE_FLOAT32` unless precision requires `DTYPE_FLOAT64`
1. **Batch operations**: Process data in batches for better performance
1. **Memory management**: Use `CleanupContext` for automatic resource management
1. **Device selection**: Use GPU when available for large computations

______________________________________________________________________

## Contributing

Contributions are welcome! Please see our contributing guidelines:

1. Fork the repository
1. Create a feature branch (`git checkout -b feature/amazing-feature`)
1. Commit your changes (`git commit -m 'Add amazing feature'`)
1. Push to the branch (`git push origin feature/amazing-feature`)
1. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/jaywyawhare/C-ML.git
cd C-ML

# Build in debug mode (build directory created automatically)
make debug

# Or build with CMake
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON ..
make -j$(nproc)

# Run tests
cd build && ctest --output-on-failure

# Or use Makefile for tests
make test
```

### Code Style

- Follow existing code style
- Use meaningful variable names
- Add comments for complex logic
- Include Doxygen-style documentation for public APIs
- Ensure all tests pass before submitting

______________________________________________________________________

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

______________________________________________________________________

## Acknowledgments

- Inspired by PyTorch and TensorFlow design principles
- Built with performance and simplicity in mind
- Thanks to all contributors and users

______________________________________________________________________

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/jaywyawhare/C-ML/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jaywyawhare/C-ML/discussions)

______________________________________________________________________

## Roadmap

- [x] MLIR JIT compilation for operations (Phase 1-2 complete)
- [ ] Complete MLIR backward pass support
- [ ] Multi-backend GPU support (CUDA, Vulkan, Metal via MLIR)
- [ ] AOT compilation for deployment
- [ ] Additional optimizers (AdamW, LAMB)
- [ ] More layer types (RNN, LSTM, Transformer blocks)
- [ ] Distributed training support
- [ ] Python bindings
- [ ] Model zoo with pre-trained models

______________________________________________________________________

<p align="center">
  Made with C
</p>

<p align="center">
  <a href="#c-ml-c-machine-learning-library">Back to top</a>
</p>
