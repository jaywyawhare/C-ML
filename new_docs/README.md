# CML Documentation

Welcome to the C-ML (C Machine Learning Library) documentation. This folder contains comprehensive guides for building, running, and using CML.

## 📚 Documentation Structure

### Getting Started

1. **[OVERVIEW.md](OVERVIEW.md)** - What is CML?
   - Library features and capabilities
   - Use cases and advantages
   - Core features overview
   - Performance characteristics
   - Platform support

2. **[QUICK_START.md](QUICK_START.md)** - Get running in 5 minutes
   - Prerequisites and setup
   - Build instructions
   - First example program
   - Common next steps
   - Troubleshooting tips

### Building & Running

3. **[COMPILATION.md](COMPILATION.md)** - How to compile
   - Prerequisites
   - Installing MLIR (required)
   - Building with Make
   - Building with CMake
   - Build variants (Debug, Release, Fast)
   - Compiler flags and options
   - Cross-compilation
   - Troubleshooting

4. **[RUNNING.md](RUNNING.md)** - How to run programs
   - Running the main executable
   - Running examples
   - Running tests
   - Using the library in your programs
   - Environment variables
   - Visualization
   - Benchmarking
   - Troubleshooting

### Learning & Reference

5. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Understanding CML internals
   - High-level architecture
   - Core components
   - Data structures
   - Execution flow
   - Memory management
   - MLIR integration
   - Computation graph
   - Device management
   - Performance optimizations

6. **[API_GUIDE.md](API_GUIDE.md)** - Complete API reference
   - Initialization & cleanup
   - Tensor API
   - Autograd API
   - Neural network API
   - Optimizer API
   - Loss functions
   - Device & memory
   - Utilities

## 🚀 Quick Navigation

### I want to...

- **Get started quickly** → [QUICK_START.md](QUICK_START.md)
- **Compile the library** → [COMPILATION.md](COMPILATION.md)
- **Run examples** → [RUNNING.md](RUNNING.md)
- **Learn the API** → [API_GUIDE.md](API_GUIDE.md)
- **Understand how it works** → [ARCHITECTURE.md](ARCHITECTURE.md)
- **Know what CML is** → [OVERVIEW.md](OVERVIEW.md)

## 📋 Document Overview

| Document | Purpose | For Whom |
|----------|---------|----------|
| [OVERVIEW.md](OVERVIEW.md) | High-level introduction and features | Everyone |
| [QUICK_START.md](QUICK_START.md) | Get up and running fast | New users |
| [COMPILATION.md](COMPILATION.md) | Build instructions and options | Developers |
| [RUNNING.md](RUNNING.md) | Execute programs and examples | Users |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Internal design and components | Developers |
| [API_GUIDE.md](API_GUIDE.md) | Complete function reference | Developers |

## 🎯 Common Tasks

### Setting up a new machine

1. Read: [OVERVIEW.md](OVERVIEW.md) - Understand what CML is
2. Follow: [COMPILATION.md](COMPILATION.md) - Build the library
3. Try: [RUNNING.md](RUNNING.md) - Run examples
4. Start: [QUICK_START.md](QUICK_START.md) - Write first program

### Building your first application

1. Start: [QUICK_START.md](QUICK_START.md) - Follow the example
2. Reference: [API_GUIDE.md](API_GUIDE.md) - Look up API functions
3. Run: [RUNNING.md](RUNNING.md) - Compile and execute
4. Debug: [RUNNING.md#troubleshooting](RUNNING.md#troubleshooting) - Fix issues

### Optimizing performance

1. Learn: [ARCHITECTURE.md](ARCHITECTURE.md#performance-optimizations) - Understand optimizations
2. Build: [COMPILATION.md](COMPILATION.md#build-variants) - Use fast build variant
3. Benchmark: [RUNNING.md#benchmarking](RUNNING.md#benchmarking) - Measure performance
4. Tune: Set environment variables in [RUNNING.md](RUNNING.md#environment-variables)

### Contributing to CML

1. Understand: [ARCHITECTURE.md](ARCHITECTURE.md) - Internal architecture
2. Reference: [API_GUIDE.md](API_GUIDE.md) - Public API
3. Build: [COMPILATION.md](COMPILATION.md) - Development setup
4. Test: [RUNNING.md](RUNNING.md#running-tests) - Run test suite

## 📦 Prerequisites

Before using CML, ensure you have:

1. **C11 Compiler**: GCC 4.9+, Clang 3.5+, MSVC 2015+
2. **MLIR 18.x**: Required for all operations
3. **CMake 3.16+** or **Make**: For building
4. **Optional**: CUDA, ROCm, Metal, Vulkan for GPU support

See [COMPILATION.md](COMPILATION.md#prerequisites) for detailed setup.

## 🔧 Building CML

### Quick Build

```bash
cd C-ML
make
```

See [COMPILATION.md](COMPILATION.md#building-with-make) for options.

### CMake Build

```bash
mkdir build && cd build
cmake ..
make
```

See [COMPILATION.md](COMPILATION.md#building-with-cmake) for options.

## ✨ Features at a Glance

- **Pure C**: Written in standard C11 for portability
- **MLIR-Powered**: JIT compilation for optimal performance
- **Auto Differentiation**: Automatic gradient computation
- **Neural Networks**: Complete set of layers and activations
- **Multiple Optimizers**: SGD, Adam, RMSprop, AdaGrad
- **Loss Functions**: MSE, MAE, Cross Entropy, BCE, Huber, KL
- **Multi-Device**: CPU, CUDA, Metal, ROCm, Vulkan support
- **Memory Efficient**: Reference counting and memory pools
- **Thread-Safe**: Safe for concurrent use

See [OVERVIEW.md](OVERVIEW.md) for complete feature list.

## 💡 Example Usage

```c
#include "cml.h"

int main(void) {
    cml_init();

    // Build model
    Sequential* model = cml_nn_sequential();
    model = cml_nn_sequential_add(model,
        (Module*)cml_nn_linear(10, 20, DTYPE_FLOAT32, DEVICE_CPU, true));
    model = cml_nn_sequential_add(model,
        (Module*)cml_nn_relu(false));

    // Create optimizer
    Optimizer* opt = cml_optim_adam_for_model((Module*)model, 0.001f, 0, 0.9f, 0.999f, 1e-8f);

    // Training data
    Tensor* X = tensor_randn((int[]){32, 10}, 2, NULL);
    Tensor* y = tensor_randn((int[]){32, 1}, 2, NULL);

    // Training loop
    for (int i = 0; i < 100; i++) {
        cml_optim_zero_grad(opt);
        Tensor* pred = cml_nn_module_forward((Module*)model, X);
        Tensor* loss = cml_nn_mse_loss(pred, y);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(opt);
        tensor_free(loss);
        tensor_free(pred);
    }

    cml_cleanup();
    return 0;
}
```

See [QUICK_START.md](QUICK_START.md) for detailed walkthrough.

## 🆘 Getting Help

### Common Issues

For build issues: → [COMPILATION.md#troubleshooting](COMPILATION.md#troubleshooting)

For runtime issues: → [RUNNING.md#troubleshooting](RUNNING.md#troubleshooting)

For API questions: → [API_GUIDE.md](API_GUIDE.md)

### Need More Help?

1. Check the relevant troubleshooting section
2. Review example code in `examples/` directory
3. Check GitHub Issues for similar problems
4. Enable debug logging: `CML_LOG_LEVEL=4`

## 📖 Reading Guide

### For Complete Beginners

1. [OVERVIEW.md](OVERVIEW.md) - Understand what CML is
2. [QUICK_START.md](QUICK_START.md) - Build and run your first program
3. [API_GUIDE.md](API_GUIDE.md) - Learn the basic API
4. Explore examples in `examples/` directory

### For Software Developers

1. [COMPILATION.md](COMPILATION.md) - Build the library
2. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand the design
3. [API_GUIDE.md](API_GUIDE.md) - Complete API reference
4. [RUNNING.md](RUNNING.md) - Testing and benchmarking

### For ML Researchers

1. [OVERVIEW.md](OVERVIEW.md) - Features and capabilities
2. [QUICK_START.md](QUICK_START.md) - Get running
3. [API_GUIDE.md](API_GUIDE.md) - Learn available operations
4. [RUNNING.md#visualization](RUNNING.md#visualization) - Training monitoring

## 🔗 Related Resources

- **Main Repository**: https://github.com/jaywyawhare/C-ML
- **Project Examples**: See `examples/` directory in repository
- **MLIR Documentation**: https://mlir.llvm.org/
- **LLVM Documentation**: https://llvm.org/docs/

## 📝 Document Maintenance

These documents are maintained alongside the CML codebase. If you find issues or have suggestions:

1. Check if the issue is already documented
2. Create an issue on GitHub
3. Provide both the problem and a potential solution
4. Include your environment details

## ✅ Verification Checklist

After building and installing, verify your setup:

- [ ] Build completes without errors (`make`)
- [ ] Examples run successfully (`./build/examples/autograd_example`)
- [ ] Tests pass (`make test`)
- [ ] Your own program compiles and runs
- [ ] Visualizations work (optional): `VIZ=1 ./build/examples/training_loop_example`

## 📚 Full Table of Contents

### Overview Documents
- [Overview](OVERVIEW.md) - Features, use cases, capabilities

### Getting Started
- [Quick Start](QUICK_START.md) - First program in 5 minutes
- [Compilation](COMPILATION.md) - Building from source
- [Running](RUNNING.md) - Executing programs

### Reference
- [API Guide](API_GUIDE.md) - Complete function reference
- [Architecture](ARCHITECTURE.md) - Internal design details

---

**Last Updated**: January 2026
**CML Version**: 0.0.2
**Documentation Version**: 1.0
