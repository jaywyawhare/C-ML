# Neural Network Layers

This document describes the neural network layers available in C-ML.

## Table of Contents

1. [Overview](#overview)
1. [Layer Architecture](#layer-architecture)
1. [Available Layers](#available-layers)
1. [Usage Examples](#usage-examples)
1. [API Summary](#api-summary)
1. [Implementation Status](#implementation-status)

## Overview

C-ML provides a comprehensive set of neural network layers following a consistent module pattern. All layers:

- Inherit from the `Module` base class
- Support automatic differentiation via autograd
- Manage their own parameters (weights, biases)
- Support training/evaluation mode switching
- Integrate seamlessly with the autograd system

## Layer Architecture

All layers follow a consistent architecture:

```c
typedef struct {
    Module base;        // Base module structure
    // Layer-specific fields
    // Parameters (weights, biases)
} LayerType;
```

Each layer:

- Implements `forward_fn` for forward pass
- Can be added to `Sequential` containers
- Supports parameter management
- Can be used in custom models

## Available Layers

### 1. Linear (Fully Connected) Layer

**Usage:** `nn_linear(in_features, out_features, dtype, device, bias)`

```c
#include "nn/layers.h"

// Create a linear layer
Linear *fc = nn_linear(10, 20, DTYPE_FLOAT32, DEVICE_CPU, true);

// Forward pass
Tensor *output = module_forward((Module*)fc, input);

// Access parameters
Parameter *weight = linear_get_weight(fc);
Parameter *bias = linear_get_bias(fc);
```

**Features:**

- Xavier/Glorot weight initialization
- Optional bias
- Supports custom weight initialization

### 2. Activation Layers

**Available:** `nn_relu()`, `nn_sigmoid()`, `nn_tanh()`, `nn_leaky_relu()`, `nn_gelu()`

```c
// ReLU activation
ReLU *relu = nn_relu(false);

// LeakyReLU
LeakyReLU *leaky_relu = nn_leaky_relu(0.01, false);

// Sigmoid
Sigmoid *sigmoid = nn_sigmoid();

// Tanh
Tanh *tanh = nn_tanh();

// GELU
GELU *gelu = nn_gelu(false);

// Forward pass
Tensor *output = module_forward((Module*)relu, input);
```

**Features:**

- All activation functions support autograd
- In-place operations (where applicable)
- No parameters (stateless)

### 3. Dropout Layer

**Usage:** `nn_dropout(p, inplace)`

```c
// Create dropout layer with 50% dropout probability
Dropout *dropout = nn_dropout(0.5, false);

// Forward pass
Tensor *output = module_forward((Module*)dropout, input);
```

**Features:**

- Only active in training mode
- Scales output by `1/(1-p)` during training
- No-op in evaluation mode

### 4. Conv2d Layer

**Usage:** `nn_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, dtype, device)`

```c
// Create 2D convolution layer
Conv2d *conv = nn_conv2d(3, 16, 3,  // 3 input channels, 16 output channels, 3x3 kernel
                         1,          // stride
                         1,          // padding
                         1,          // dilation
                         true,       // use bias
                         DTYPE_FLOAT32, DEVICE_CPU);

// Forward pass
Tensor *output = module_forward((Module*)conv, input);
```

**Features:**

- Kaiming/He weight initialization
- Supports stride, padding, dilation
- Optional bias

**Note:** Full convolution implementation is a placeholder. Full implementation requires im2col transformation or direct convolution loops.

### 5. BatchNorm2d Layer

**Usage:** `nn_batchnorm2d(num_features, eps, momentum, affine, track_running_stats, dtype, device)`

```c
// Create BatchNorm2d layer
BatchNorm2d *bn = nn_batchnorm2d(16,    // num_features
                                 1e-5,  // eps
                                 0.1,   // momentum
                                 true,  // affine
                                 true,  // track_running_stats
                                 DTYPE_FLOAT32, DEVICE_CPU);

// Forward pass
Tensor *output = module_forward((Module*)bn, input);
```

**Features:**

- Learnable scale (gamma) and shift (beta) parameters
- Running mean and variance statistics
- Different behavior in training vs evaluation mode

**Note:** Full BatchNorm2d forward pass is a placeholder. Requires mean/variance computation over spatial dimensions.

### 6. Pooling Layers

**Available:** `nn_maxpool2d()`, `nn_avgpool2d()`

```c
// MaxPool2d
MaxPool2d *maxpool = nn_maxpool2d(2,    // kernel_size
                                  2,    // stride
                                  0,    // padding
                                  1,    // dilation
                                  false); // ceil_mode

// AvgPool2d
AvgPool2d *avgpool = nn_avgpool2d(2,    // kernel_size
                                  2,    // stride
                                  0,    // padding
                                  false, // ceil_mode
                                  true); // count_include_pad

// Forward pass
Tensor *output = module_forward((Module*)maxpool, input);
```

**Features:**

- Supports stride, padding, dilation
- Ceil mode for output size calculation
- Count include padding for AvgPool2d

**Note:** Full pooling implementation is a placeholder. Requires window-based max/average computation.

### 7. LayerNorm Layer

**Usage:** `nn_layernorm(normalized_shape, eps, elementwise_affine, dtype, device)`

```c
// Create LayerNorm layer
LayerNorm *ln = nn_layernorm(64,      // normalized_shape (last dimension size)
                             1e-5,   // eps
                             true,   // elementwise_affine (learnable scale/shift)
                             DTYPE_FLOAT32, DEVICE_CPU);

// Forward pass
Tensor *output = module_forward((Module*)ln, input);
```

**Features:**

- Normalizes across features (last dimension)
- Learnable scale (gamma) and shift (beta) parameters
- No running statistics (stateless normalization)
- Different behavior from BatchNorm2d (normalizes across different dimensions)

**Note:** Essential for transformer architectures, BERT, GPT models, and modern NLP architectures.

### 8. Sequential Container

**Usage:** `nn_sequential()` with `sequential_add()`

```c
// Create Sequential container
Sequential *seq = nn_sequential();

// Add layers
sequential_add(seq, (Module*)nn_linear(10, 20, DTYPE_FLOAT32, DEVICE_CPU, true));
sequential_add(seq, (Module*)nn_relu(false));
sequential_add(seq, (Module*)nn_linear(20, 1, DTYPE_FLOAT32, DEVICE_CPU, true));

// Forward pass through all layers
Tensor *output = module_forward((Module*)seq, input);

// Access layers
Module *layer = sequential_get(seq, 0);
int length = sequential_get_length(seq);
```

**Features:**

- Automatically chains layers
- Collects parameters from all submodules
- Forward pass goes through all layers sequentially

## Example: Building a Complete Model

```c
#include "nn/layers.h"
#include "nn/module.h"

// Create a simple feedforward network
Sequential *model = nn_sequential();

// Add layers
sequential_add(model, (Module*)nn_linear(784, 128, DTYPE_FLOAT32, DEVICE_CPU, true));
sequential_add(model, (Module*)nn_relu(false));
sequential_add(model, (Module*)nn_dropout(0.5, false));
sequential_add(model, (Module*)nn_linear(128, 64, DTYPE_FLOAT32, DEVICE_CPU, true));
sequential_add(model, (Module*)nn_relu(false));
sequential_add(model, (Module*)nn_linear(64, 10, DTYPE_FLOAT32, DEVICE_CPU, true));

// Set training mode
module_set_training((Module*)model, true);

// Forward pass
Tensor *output = module_forward((Module*)model, input);

// Collect all parameters for optimizer
Parameter **params = NULL;
int num_params = 0;
module_collect_parameters((Module*)model, &params, &num_params, true);

// Create optimizer
Optimizer *optimizer = optim_adam(params, num_params, 0.001f, 0.0f, 0.9f, 0.999f, 1e-8f);

// Training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    optimizer_zero_grad(optimizer);
    Tensor *output = module_forward((Module*)model, input);
    Tensor *loss = tensor_mse_loss(output, target);
    tensor_backward(loss, NULL, false, false);
    optimizer_step(optimizer);
}
```

## API Summary

| Feature              | C-ML API                                                               |
| -------------------- | ---------------------------------------------------------------------- |
| Linear layer         | `nn_linear(10, 20, DTYPE_FLOAT32, DEVICE_CPU, true)`                   |
| ReLU activation      | `nn_relu(false)`                                                       |
| Dropout layer        | `nn_dropout(0.5, false)`                                               |
| Conv2d layer         | `nn_conv2d(3, 16, 3, 1, 1, 1, true, DTYPE_FLOAT32, DEVICE_CPU)`        |
| BatchNorm2d layer    | `nn_batchnorm2d(16, 1e-5, 0.1, true, true, DTYPE_FLOAT32, DEVICE_CPU)` |
| LayerNorm layer      | `nn_layernorm(64, 1e-5, true, DTYPE_FLOAT32, DEVICE_CPU)`              |
| MaxPool2d layer      | `nn_maxpool2d(2, 2, 0, 1, false)`                                      |
| AvgPool2d layer      | `nn_avgpool2d(2, 2, 0, false, true)`                                   |
| Sequential container | `nn_sequential()` + `sequential_add()`                                 |

## Implementation Status

### Fully Implemented

- Linear layer
- Activation layers (ReLU, Sigmoid, Tanh, LeakyReLU, GELU, Softmax, LogSoftmax)
- Dropout layer
- Conv2d layer
- BatchNorm2d layer
- MaxPool2d / AvgPool2d layers
- LayerNorm layer
- Sequential container

### Implementation Notes

- All layers support automatic differentiation via autograd
- All layers support training/evaluation mode switching
- All layers properly manage their own parameters
- Sequential container automatically collects parameters from submodules

## Notes

1. **Memory Management**: All layers allocate their own parameters. Call `module_free()` to clean up.
1. **Autograd**: All layers support automatic differentiation through the autograd system.
1. **Training Mode**: Use `module_set_training()` to switch between training and evaluation modes.
1. **Parameter Access**: Parameters can be accessed via layer-specific getter functions or through `module_get_parameter()`.

## Future Enhancements

- Additional layers: GroupNorm, Conv1d, Conv3d
- More activation functions: ELU, SELU, Swish, Mish
- Container layers: ModuleList, ModuleDict
- Performance optimizations: im2col for convolutions, SIMD operations
