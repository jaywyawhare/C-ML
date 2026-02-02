<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="dark-mode.svg">
    <source media="(prefers-color-scheme: light)" srcset="light-mode.svg">
    <img alt="C-ML" src="light-mode.svg" height="96">
  </picture>
</p>

# C-ML Documentation

**Version 0.0.2** | [GitHub](https://github.com/jaywyawhare/C-ML) | [Issues](https://github.com/jaywyawhare/C-ML/issues)

______________________________________________________________________

## Overview

C-ML is a production-ready machine learning library written in pure C, providing automatic differentiation, neural network layers, optimizers, and training utilities. This documentation covers all aspects of the library from basic usage to advanced features.

## Quick Navigation

### Getting Started

- **[Getting Started](getting_started.md)** - Quick start guide and installation

### Core Documentation

- **[API Reference](api_reference.md)** - Complete API documentation with examples
- **[Autograd](autograd.md)** - Automatic differentiation system
- **[Neural Network Layers](nn_layers.md)** - Layer API reference
- **[Training](training.md)** - Training API and best practices
- **[Graph Mode](graph_mode.md)** - Lazy execution and optimization

## Quick Start

### Installation

```bash
git clone https://github.com/jaywyawhare/C-ML.git
cd C-ML
make clean && make
```

### Basic Usage

```c
#include "cml.h"

int main() {
    // Initialize library
    cml_init();
    cml_seed(42);

    // Get defaults
    DeviceType device = cml_get_default_device();
    DType dtype = cml_get_default_dtype();

    // Create model
    Sequential* model = cml_nn_sequential();
    model = sequential_add_chain(model,
        (Module*)cml_nn_linear(784, 128, dtype, device, true),
        (Module*)cml_nn_relu(false),
        (Module*)cml_nn_linear(128, 10, dtype, device, true),
        NULL
    );

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

## Key Features

### High-Level API

The `cml_*` prefix functions provide a PyTorch-like interface:

```c
Tensor* x = cml_zeros(shape, ndim, NULL);
Tensor* y = cml_add(a, b);
Tensor* z = cml_relu(y);
```

### Low-Level API

Direct control with `tensor_*`, `nn_*`, `optim_*` functions:

```c
Tensor* x = tensor_zeros(shape, ndim, NULL);
Tensor* y = tensor_add(a, b);
Linear* layer = nn_linear(10, 20, dtype, device, true);
```

### Automatic Differentiation

Full autograd support with dynamic computation graphs:

```c
x->requires_grad = true;
Tensor* y = cml_mul(x, x);
cml_backward(y, NULL, false, false);
// x->grad now contains gradients
```

### Multi-Device Support

Automatic device detection and memory allocation:

```c
DeviceType device = cml_get_default_device();  // Auto-detects best device
cml_set_default_device(DEVICE_CUDA);           // Use CUDA
cml_set_default_device(DEVICE_METAL);          // Use Metal (macOS)
```

## API Overview

### Tensor Operations

- **Creation**: `cml_zeros()`, `cml_ones()`, `cml_empty()`, `cml_tensor()`
- **Elementwise**: `cml_add()`, `cml_mul()`, `cml_exp()`, `cml_log()`, etc.
- **Reductions**: `cml_sum()`, `cml_mean()`, `cml_max()`, `cml_min()`
- **Views**: `cml_reshape()`, `cml_transpose()`, `cml_clone()`, `cml_detach()`

### Neural Network Layers

- **Linear**: `cml_nn_linear()`
- **Convolutional**: `cml_nn_conv2d()`
- **Normalization**: `cml_nn_batchnorm2d()`, `cml_nn_layernorm()`
- **Pooling**: `cml_nn_maxpool2d()`, `cml_nn_avgpool2d()`
- **Activations**: `cml_nn_relu()`, `cml_nn_sigmoid()`, `cml_nn_tanh()`, etc.
- **Dropout**: `cml_nn_dropout()`

### Optimizers

- **Adam**: `cml_optim_adam()`, `cml_optim_adam_for_model()`
- **SGD**: `cml_optim_sgd()`, `cml_optim_sgd_for_model()`
- **RMSprop**: `cml_optim_rmsprop()`
- **AdaGrad**: `cml_optim_adagrad()`

### Loss Functions

- **MSE**: `cml_nn_mse_loss()`
- **MAE**: `cml_nn_mae_loss()`
- **BCE**: `cml_nn_bce_loss()`
- **Cross Entropy**: `cml_nn_cross_entropy_loss()`
- **Huber**: `cml_nn_huber_loss()`
- **KL Divergence**: `cml_nn_kl_div_loss()`

## Examples

See the `examples/` directory for complete working examples:

- **Basic Training**: `main.c` - Simple XOR classification
- **Comprehensive Training**: `examples/test.c` - Full training pipeline with metrics
- **Autograd**: `examples/autograd_example.c` - Gradient computation examples
- **Advanced Features**: `examples/early_stopping_lr_scheduler.c` - Early stopping and LR scheduling

## Platform Support

| Platform                    | Status          | Device Support  |
| --------------------------- | --------------- | --------------- |
| Linux (x86_64)              | Fully Supported | CPU, CUDA, ROCm |
| macOS (Intel/Apple Silicon) | Fully Supported | CPU, Metal      |
| Windows (MSVC/MinGW)        | Supported       | CPU             |

## Performance Tips

1. **Use appropriate data types**: Prefer `DTYPE_FLOAT32` unless precision requires `DTYPE_FLOAT64`
1. **Batch operations**: Process data in batches for better performance
1. **Memory management**: Use `CleanupContext` for automatic resource management
1. **Device selection**: Use GPU when available for large computations
1. **Graph mode**: Use lazy execution for inference workloads

## Error Handling

```c
Tensor* t = tensor_empty(shape, ndim, NULL);
if (!t && CML_HAS_ERRORS()) {
    const char* msg = CML_LAST_ERROR();
    int code = CML_LAST_ERROR_CODE();
    error_stack_print_all();
}

// Or use checked constructors
Tensor* t = CML_CHECK(tensor_empty(shape, ndim, NULL), "Failed to create tensor");
```

## Contributing

We welcome contributions! Please see the main [README](../README.md) for contributing guidelines.

## License

MIT License - see [LICENSE](../LICENSE) file for details.

______________________________________________________________________

**Need help?** Open an [issue](https://github.com/jaywyawhare/C-ML/issues) or check the [API Reference](api_reference.md).
