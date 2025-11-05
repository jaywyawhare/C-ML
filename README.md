<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/dark-mode.svg">
    <img alt="C-ML" src="docs/img.svg" height="96">
  </picture>
</p>

# C-ML: C Machine Learning Library

A comprehensive machine learning library written in pure C, providing automatic differentiation, neural network layers, optimizers, and training utilities.

## Features

- **Automatic Differentiation (Autograd)**: Dynamic computation graphs with automatic gradient computation
- **Neural Network Layers**: Complete set of layers including Linear, Conv2d, BatchNorm2d, Pooling, Activations, and more
- **Optimizers**: SGD and Adam optimizers with momentum, weight decay, and learning rate scheduling
- **Tensor Operations**: Comprehensive tensor operations with broadcasting support
- **Loss Functions**: MSE, MAE, BCE, Cross Entropy, and more
- **Memory Management**: Safe memory management with automatic cleanup
- **Logging System**: Configurable logging levels for debugging

## Quick Start

### Requirements

- C11 compatible compiler (GCC, Clang)
- CMake 3.16+ (for CMake build)
- Make (for Makefile build)

### Building the Library

```bash
# Using Make (builds main and examples)
make clean && make

# Using CMake (enable examples)
mkdir build && cd build
cmake -DBUILD_EXAMPLES=ON ..
make -j$(nproc)
```

The build will create:

- `build/main` - Example executable (Makefile)
- `build/examples/*` - Example binaries (Makefile)
- `build/bin/*` - Binaries when using CMake
- `build/libcml.a` - Static library
- `build/libcml.so` - Shared library (if enabled)

### Basic Usage

```c
#include "cml.h"
#include <stdio.h>

int main(void) {
    cml_init();

    Sequential *model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_linear(4, 1, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_sigmoid());

    summary((Module*)model);

    Parameter **params;
    int num_params;
    module_collect_parameters((Module*)model, &params, &num_params, true);

    Optimizer *optimizer = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);

    for (int epoch = 0; epoch < 100; epoch++) {
        optimizer_zero_grad(optimizer);

        Tensor *outputs = module_forward((Module*)model, inputs);

        Tensor *loss = tensor_mse_loss(outputs, targets);

        tensor_backward(loss, NULL, false, false);

        optimizer_step(optimizer);

        tensor_free(loss);
        tensor_free(outputs);
    }

    optimizer_free(optimizer);
    CM_FREE(params);
    module_free((Module*)model);
    cml_cleanup();

    return 0;
}
```

## Documentation

- **[Autograd System](docs/AUTOGRAD.md)** - Complete guide to automatic differentiation
- **[Neural Network Layers](docs/NN_LAYERS.md)** - Available layers and their usage
- **[Training Guide](docs/TRAINING.md)** - Comprehensive guide to training neural networks (manual LR scheduling and early stopping)
- **[Autograd Implementation](docs/AUTOGRAD_IMPLEMENTATION.md)** - Technical implementation details
- **[Layers Implementation](docs/LAYERS_COMPLETE.md)** - Layer implementation details
- **[Integration Summary](docs/INTEGRATION_SUMMARY.md)** - Library integration overview

## License

See [LICENCE.md](LICENCE.md) for license information.

## Contributing

Contributions are welcome! Please ensure your code:

- Follows the existing code style
- Includes appropriate documentation
- Passes all existing tests
- Updates relevant documentation

## Documentation

For comprehensive documentation, see:

- **[Documentation Index](docs/INDEX.md)** - Complete documentation index
- **[Training Guide](docs/TRAINING.md)** - Start here for training examples
- **[Autograd System](docs/AUTOGRAD.md)** - Automatic differentiation guide
- **[Neural Network Layers](docs/NN_LAYERS.md)** - Available layers reference

## Examples

Example programs are available in the `examples/` directory and in `main.c`. These demonstrate:

- Complete training loops
- Model creation and parameter management
- Optimizer usage
- Loss function usage
- Autograd system usage

## Support

For issues, questions, or contributions, please refer to the project repository.
