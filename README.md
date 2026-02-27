<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/dark-mode.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/light-mode.svg">
    <img alt="C-ML" src="docs/light-mode.svg" height="96">
  </picture>
</p>

<h1 align="center">C-ML: C Machine Learning Library</h1>

<p align="center">
  <strong>A machine learning library written in pure C</strong><br>
  Automatic differentiation, neural networks, optimizers, datasets, and training utilities
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

- **Automatic Differentiation** -- Dynamic computation graphs with automatic gradient computation
- **Neural Network Layers** -- Linear, Conv1d/2d/3d, RNN/LSTM/GRU, Transformer, Embedding, BatchNorm, LayerNorm, GroupNorm, Pooling, Dropout, 12 activation functions
- **Containers** -- Sequential, ModuleList, ModuleDict
- **Optimizers** -- SGD, Adam, AdamW, RMSprop, Adagrad, AdaDelta + LR schedulers (Step, Exponential, Cosine, ReduceOnPlateau, MultiStep)
- **Loss Functions** -- MSE, MAE, BCE, CrossEntropy, Huber, KL Divergence, Hinge, Focal, SmoothL1
- **Dataset Hub** -- One-liner loading: `cml_dataset_load("iris")` with auto-download and caching
- **Model Zoo** -- Pre-built architectures: MLP, ResNet, VGG, GPT-2, BERT
- **Model I/O** -- Save/load models and training checkpoints
- **Tensor Operations** -- Comprehensive ops with NumPy-style broadcasting
- **SIMD Vectorization** -- SSE/AVX/AVX-512/NEON with runtime detection
- **BLAS Integration** -- Dynamic loading of MKL/OpenBLAS/Accelerate
- **Memory Management** -- Automatic cleanup, memory pools, graph allocator
- **Python Bindings** -- CFFI-based Python interface

______________________________________________________________________

## Quick Start

### Build

```bash
git clone https://github.com/jaywyawhare/C-ML.git
cd C-ML

# CMake build
mkdir -p build && cd build
cmake -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON ..
make -j$(nproc)
```

### Hello World

```c
#include "cml.h"
#include <stdio.h>

int main(void) {
    cml_init();

    // Load a dataset
    Dataset* ds = cml_dataset_load("iris");
    dataset_normalize(ds, "minmax");
    Dataset *train, *test;
    dataset_split(ds, 0.8f, &train, &test);

    // Build a model
    Sequential* model = cml_nn_sequential();
    DeviceType dev = cml_get_default_device();
    DType dt = cml_get_default_dtype();
    model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(4, 16, dt, dev, true));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(16, 3, dt, dev, true));

    // Train
    Optimizer* opt = cml_optim_adam_for_model((Module*)model, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    cml_nn_module_set_training((Module*)model, true);

    for (int epoch = 0; epoch < 100; epoch++) {
        cml_optim_zero_grad(opt);
        Tensor* out = cml_nn_module_forward((Module*)model, train->X);
        Tensor* loss = cml_nn_mse_loss(out, train->y);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(opt);
        tensor_free(loss);
        tensor_free(out);
    }

    cml_cleanup();
    return 0;
}
```

```bash
gcc -std=c11 -O2 example.c -I./include -L./build/lib -lcml_static -lm -ldl -o example
./example
```

______________________________________________________________________

## Examples

C-ML includes 17 example programs covering common ML tasks:

| Example                    | Description                                   | Dataset        |
| -------------------------- | --------------------------------------------- | -------------- |
| `ex01_tensor_ops`          | Basic tensor creation, arithmetic, reductions | None           |
| `ex02_linear_regression`   | Linear regression with SGD                    | Boston Housing |
| `ex03_logistic_regression` | Binary classification with BCE loss           | Breast Cancer  |
| `ex04_mlp_classifier`      | Multi-class MLP classifier                    | Iris           |
| `ex05_autoencoder`         | Autoencoder with bottleneck                   | Digits 8x8     |
| `ex06_conv_net`            | Image classification MLP                      | Digits 8x8     |
| `ex07_rnn_sequence`        | RNN time series prediction                    | Airline        |
| `ex08_lstm_timeseries`     | LSTM time series forecasting                  | Airline        |
| `ex09_gru_classifier`      | GRU sequence classifier                       | Iris           |
| `ex10_embedding`           | Embedding lookup table demo                   | None           |
| `ex11_gan`                 | Generative Adversarial Network                | Digits 8x8     |
| `ex12_multi_task`          | Multi-task learning                           | Wine           |
| `ex13_transformer`         | Transformer encoder with self-attention       | None           |
| `ex14_lr_scheduler`        | LR scheduler comparison                       | Boston Housing |
| `ex15_activations`         | Activation function showcase                  | Breast Cancer  |
| `hello_cml`                | Minimal forward pass                          | None           |
| `simple_xor`               | XOR problem with training loop                | XOR            |

```bash
# Run after building
./build/bin/ex04_mlp_classifier
```

See [examples/README.md](examples/README.md) for details.

______________________________________________________________________

## Dataset Hub

Load datasets with a single function call:

```c
Dataset* ds = cml_dataset_load("iris");        // 150 samples, 4 features, 3 classes
Dataset* ds = cml_dataset_load("mnist");       // 70k samples, 784 features, 10 classes
Dataset* ds = cml_dataset_load("cifar10");     // 60k samples, 3072 features, 10 classes
Dataset* ds = cml_dataset_from_csv("data.csv", -1);  // Custom CSV
```

Supported datasets: `iris`, `wine`, `breast_cancer`, `boston`, `mnist`, `fashion_mnist`, `cifar10`, `airline`, `digits`

Datasets are automatically downloaded and cached in `~/.cml/datasets/`. See [docs/datasets.md](docs/datasets.md).

______________________________________________________________________

## Documentation

| Guide                                                  | Description                             |
| ------------------------------------------------------ | --------------------------------------- |
| [Getting Started](docs/getting_started.md)             | Build, install, first program           |
| [API Reference](docs/api_reference.md)                 | Complete API documentation              |
| [Neural Network Layers](docs/nn_layers.md)             | All layers with signatures and examples |
| [Training Guide](docs/training.md)                     | Optimizers, schedulers, training loops  |
| [Datasets](docs/datasets.md)                           | Dataset hub and custom data loading     |
| [Autograd](docs/autograd.md)                           | Automatic differentiation guide         |
| [Graph Mode](docs/graph_mode.md)                       | Lazy execution and IR optimization      |
| [Optimizations](docs/optimizations.md)                 | SIMD, BLAS, fusion, caching internals   |
| [IR Graph Management](docs/ir_graph_management.md)     | Memory optimization and kernel export   |
| [Kernel Studio](docs/kernel_studio.md)                 | Optimization pass visualization         |
| [External Dependencies](docs/EXTERNAL_DEPENDENCIES.md) | Optional library integration            |
| [Python Bindings](python/INSTALLATION.md)              | Python CFFI setup                       |

______________________________________________________________________

## Architecture

```
C-ML/
├── include/            # Public API headers
│   ├── cml.h           # Main header (include this)
│   ├── tensor/         # Tensor operations
│   ├── autograd/       # Automatic differentiation
│   ├── nn/             # Neural network layers, containers, model I/O
│   ├── ops/ir/         # IR graph, optimization, execution, LLVM backend
│   ├── optim/          # Optimizers and LR schedulers
│   ├── datasets/       # Dataset hub
│   └── zoo/            # Model zoo
├── src/                # Implementation
├── examples/           # 17 example programs
├── tests/              # Test suite
├── python/             # Python CFFI bindings
└── docs/               # Documentation
```

______________________________________________________________________

## Build Options

```bash
# CMake (recommended)
mkdir -p build && cd build
cmake -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Makefile shortcuts
make              # Standard build
make release      # Release build with optimizations
make debug        # Debug build with sanitizers
make test         # Build and run tests
```

### Integration

```bash
# Static library
gcc your_program.c -I./include -L./build/lib -lcml_static -lm -ldl -o your_program

# Shared library
gcc your_program.c -I./include -L./build/lib -lcml -lm -ldl -o your_program
export LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH
```

______________________________________________________________________

## Testing

```bash
cd build && ctest --output-on-failure
# Or: make test
```

______________________________________________________________________

## Contributing

1. Fork the repository
1. Create a feature branch (`git checkout -b feature/amazing-feature`)
1. Commit your changes (`git commit -m 'Add amazing feature'`)
1. Push to the branch (`git push origin feature/amazing-feature`)
1. Open a Pull Request

______________________________________________________________________

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

______________________________________________________________________

<p align="center">
  Made with C
</p>

<p align="center">
  <a href="#c-ml-c-machine-learning-library">Back to top</a>
</p>
