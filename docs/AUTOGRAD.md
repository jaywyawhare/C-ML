# C-ML Autograd System

## Table of Contents

1. [Overview](#overview)
1. [Features](#features)
1. [Quick Start](#quick-start)
1. [API Reference](#api-reference)
1. [Usage Examples](#usage-examples)
1. [Best Practices](#best-practices)
1. [Technical Details](#technical-details)
1. [Implementation Details](#implementation-details)

## Overview

C-ML implements a comprehensive automatic differentiation (autograd) system. This system enables efficient computation of gradients for tensor operations, making it easy to train neural networks and optimize machine learning models.

The autograd system builds computation graphs dynamically during the forward pass and automatically computes gradients during the backward pass, enabling efficient gradient-based optimization for machine learning models.

## Features

### Core Features

- **Dynamic Computation Graphs**: Graphs are built on-the-fly during the forward pass, allowing flexible model architectures
- **Automatic Gradient Calculation**: Tracks all operations on tensors and automatically computes gradients during backward pass
- **Gradient Accumulation**: Supports accumulating gradients across multiple backward passes
- **Higher-Order Derivatives**: Can compute gradients of gradients (with `create_graph=true`)
- **No-Gradient Mode**: Disable gradient tracking for inference (`autograd_no_grad_enter()`)
- **Anomaly Detection**: Optional detection of NaN/Inf in gradients for debugging

### Supported Operations

#### Binary Operations

- Addition (`tensor_add`)
- Subtraction (`tensor_sub`)
- Multiplication (`tensor_mul`)
- Division (`tensor_div`)
- Power (`tensor_pow`)

#### Unary Operations

- Negation (`tensor_neg`)
- Exponential (`tensor_exp`)
- Logarithm (`tensor_log`)
- Square root (`tensor_sqrt`)
- Trigonometric functions (`tensor_sin`, `tensor_cos`, `tensor_tan`)
- Hyperbolic tangent (`tensor_tanh`)

#### Activation Functions

- ReLU (`tensor_relu`)
- Sigmoid (`tensor_sigmoid`)
- Leaky ReLU (`tensor_leaky_relu`)

#### Reduction Operations

- Sum (`tensor_sum`)
- Mean (`tensor_mean`)

#### Loss Functions

- Mean Squared Error (`tensor_mse_loss`)

## Quick Start

### Basic Usage

```c
#include "autograd/autograd.h"
#include "tensor/tensor.h"

// Initialize autograd engine
autograd_init();

// Create tensors
int shape[] = {1};
Tensor *x = tensor_ones(shape, 1, DTYPE_FLOAT32, DEVICE_CPU);
Tensor *y = tensor_ones(shape, 1, DTYPE_FLOAT32, DEVICE_CPU);

tensor_set_float(x, 0, 3.0f);
tensor_set_float(y, 0, 4.0f);

// Enable gradient tracking
x->requires_grad = true;
y->requires_grad = true;

// Forward pass: z = x * y
Tensor *z = tensor_mul(x, y);

// Backward pass: compute gradients
tensor_backward(z, NULL, false, false);

// Access gradients
printf("dz/dx = %.2f\n", tensor_get_float(x->grad, 0)); // Should be 4.0
printf("dz/dy = %.2f\n", tensor_get_float(y->grad, 0)); // Should be 3.0

// Cleanup
tensor_free(x);
tensor_free(y);
tensor_free(z);
autograd_shutdown();
```

## API Reference

### Initialization and Shutdown

#### `autograd_init()`

Initializes the global autograd engine. Must be called before using any autograd functionality.

#### `autograd_shutdown()`

Shuts down the autograd engine and frees resources.

#### `autograd_get_engine()`

Returns pointer to the global autograd engine.

### Gradient Mode Control

#### `autograd_set_grad_mode(bool enabled)`

Enable or disable gradient tracking globally.

#### `autograd_is_grad_enabled()`

Check if gradient tracking is currently enabled.

#### `autograd_no_grad_enter()`

Enter no-gradient mode (disables gradient tracking). Useful for inference.

```c
autograd_no_grad_enter();
// Operations here won't build computation graph
Tensor *result = tensor_mul(a, b);  // No grad_fn created
autograd_no_grad_exit();
```

#### `autograd_no_grad_exit()`

Exit no-gradient mode and re-enable gradient tracking.

### Backward Pass

#### `tensor_backward(Tensor *tensor, Tensor *gradient, bool retain_graph, bool create_graph)`

Computes gradients of the tensor with respect to all leaf tensors.

**Parameters:**

- `tensor`: The output tensor to compute gradients from
- `gradient`: Optional gradient to use (if NULL, uses ones for scalar tensors)
- `retain_graph`: If true, keeps the computation graph after backward (for multiple backward passes)
- `create_graph`: If true, creates a new graph for computing higher-order derivatives

```c
// Simple backward
tensor_backward(loss, NULL, false, false);

// Retain graph for multiple backward passes
tensor_backward(loss, NULL, true, false);

// Enable higher-order gradients
tensor_backward(loss, NULL, false, true);
```

### Tensor Gradient Management

#### `tensor_requires_grad(Tensor *t)`

Check if a tensor requires gradients.

#### `tensor_set_requires_grad(Tensor *t, bool requires_grad)`

Set whether a tensor should track gradients.

#### `tensor_is_leaf(Tensor *t)`

Check if a tensor is a leaf node (created by user, not by an operation).

#### `tensor_zero_grad(Tensor *t)`

Zero out the gradients of a tensor.

```c
// Zero gradients before backward pass
tensor_zero_grad(parameter);
// Compute gradients
tensor_backward(loss, NULL, false, false);
```

#### `tensor_detach(Tensor *t)`

Create a new tensor that shares data with the input but doesn't require gradients.

```c
Tensor *x = tensor_ones(shape, 1, DTYPE_FLOAT32, DEVICE_CPU);
x->requires_grad = true;

Tensor *y = tensor_mul(x, x);
Tensor *y_detached = tensor_detach(y);  // No grad_fn, doesn't track gradients
```

#### `tensor_accumulate_grad(Tensor *tensor, Tensor *new_grad)`

Accumulate gradients into a tensor's gradient buffer.

### Context Management

The autograd context is used to save tensors and values needed for the backward pass.

#### `autograd_context_create()`

Create a new autograd context.

#### `autograd_context_free(AutogradContext *ctx)`

Free an autograd context and its resources.

#### `autograd_context_save_for_backward(AutogradContext *ctx, Tensor **tensors, int num_tensors)`

Save tensors for use in the backward pass.

#### `autograd_context_get_saved_tensor(AutogradContext *ctx, int index)`

Retrieve a saved tensor from the context.

### Function (Operation Node) Management

#### `autograd_function_create(OpType op_type, const char *name)`

Create a new function node in the computation graph.

#### `autograd_function_set_backward(Function *fn, BackwardFn backward_fn)`

Set the backward function for a computation node.

#### `autograd_function_set_inputs(Function *fn, Tensor **inputs, int num_inputs)`

Set the input tensors for a function node.

### Advanced Features

#### `autograd_set_anomaly_detection(bool enabled)`

Enable detection of NaN/Inf in gradients for debugging.

```c
autograd_set_anomaly_detection(true);
tensor_backward(loss, NULL, false, false);
// Will log errors if NaN or Inf detected in gradients
```

#### `autograd_print_graph(Tensor *tensor)`

Print the computation graph for debugging.

```c
autograd_print_graph(output);
```

## Architecture

### Computation Graph

The autograd system builds a Directed Acyclic Graph (DAG) during the forward pass:

```
     x (leaf)    y (leaf)
        \         /
         \       /
          \     /
           \   /
            \ /
         mul_fn
             |
             z
```

Each tensor stores:

- `grad_fn`: Pointer to the function that created it
- `grad`: The accumulated gradient
- `requires_grad`: Whether to track gradients

Each function stores:

- `op_type`: The type of operation
- `inputs`: Parent tensors
- `ctx`: Context with saved values for backward
- `backward_fn`: Function to compute gradients

### Backward Pass

The backward pass:

1. Starts from the output tensor
1. Builds a topological ordering of all operations
1. Traverses in reverse topological order
1. Calls each operation's backward function
1. Accumulates gradients into leaf tensors

```c
// Forward pass builds the graph
Tensor *z = tensor_mul(tensor_add(x, y), w);

// Backward pass traverses in reverse
tensor_backward(z, NULL, false, false);
// Computes: dz/dw, dz/dx, dz/dy
```

### Memory Management

- Tensors use reference counting for automatic memory management
- Gradients are accumulated (not replaced) for flexibility
- Use `tensor_zero_grad()` to clear gradients between optimization steps
- Set `retain_graph=false` to free computation graph after backward

## Examples

### Example 1: Simple Gradient

```c
// f(x, y) = x^2 + y^2
int shape[] = {1};
Tensor *x = tensor_ones(shape, 1, DTYPE_FLOAT32, DEVICE_CPU);
Tensor *y = tensor_ones(shape, 1, DTYPE_FLOAT32, DEVICE_CPU);

tensor_set_float(x, 0, 3.0f);
tensor_set_float(y, 0, 4.0f);

x->requires_grad = true;
y->requires_grad = true;

Tensor *exp = tensor_ones(shape, 1, DTYPE_FLOAT32, DEVICE_CPU);
tensor_set_float(exp, 0, 2.0f);

Tensor *x2 = tensor_pow(x, exp);
Tensor *y2 = tensor_pow(y, exp);
Tensor *z = tensor_add(x2, y2);

tensor_backward(z, NULL, false, false);

// df/dx = 2x = 6, df/dy = 2y = 8
printf("df/dx = %.1f\n", tensor_get_float(x->grad, 0));
printf("df/dy = %.1f\n", tensor_get_float(y->grad, 0));
```

### Example 2: Neural Network

```c
// Simple layer: y = sigmoid(w*x + b)
Tensor *w = tensor_ones(shape, 1, DTYPE_FLOAT32, DEVICE_CPU);
Tensor *b = tensor_ones(shape, 1, DTYPE_FLOAT32, DEVICE_CPU);
Tensor *x = tensor_ones(shape, 1, DTYPE_FLOAT32, DEVICE_CPU);

tensor_set_float(w, 0, 0.5f);
tensor_set_float(b, 0, 0.1f);
tensor_set_float(x, 0, 2.0f);

w->requires_grad = true;
b->requires_grad = true;

// Forward
Tensor *linear = tensor_add(tensor_mul(w, x), b);
Tensor *y = tensor_sigmoid(linear);

// Backward
tensor_backward(y, NULL, false, false);

// Gradients
printf("dy/dw = %.4f\n", tensor_get_float(w->grad, 0));
printf("dy/db = %.4f\n", tensor_get_float(b->grad, 0));
```

### Example 3: Training Loop

```c
// Initialize
autograd_init();

// Create model parameters
Tensor *weights = /* ... */;
weights->requires_grad = true;

// Training loop
for (int epoch = 0; epoch < 100; epoch++) {
    // Zero gradients
    tensor_zero_grad(weights);

    // Forward pass
    Tensor *prediction = /* model(input) */;
    Tensor *loss = tensor_mse_loss(prediction, target);

    // Backward pass
    tensor_backward(loss, NULL, false, false);

    // Update weights (simple SGD)
    float lr = 0.01f;
    for (size_t i = 0; i < weights->numel; i++) {
        float w = tensor_get_float(weights, i);
        float grad = tensor_get_float(weights->grad, i);
        tensor_set_float(weights, i, w - lr * grad);
    }

    // Log progress
    if (epoch % 10 == 0) {
        printf("Epoch %d: Loss = %.4f\n", epoch, tensor_get_float(loss, 0));
    }

    tensor_free(prediction);
    tensor_free(loss);
}
```

## Best Practices

1. **Always call `autograd_init()` before using autograd features**
1. **Zero gradients before each backward pass in training loops**
1. **Use `autograd_no_grad_enter()` for inference to save memory**
1. **Set `requires_grad=true` only for trainable parameters**
1. **Free tensors when done to prevent memory leaks**
1. **Use `retain_graph=false` (default) unless you need multiple backward passes**
1. **Enable anomaly detection during debugging**

## Technical Details

C-ML's autograd system provides the following features:

| Feature                     | Status          |
| --------------------------- | --------------- |
| Dynamic computation graphs  | Fully supported |
| Automatic differentiation   | Fully supported |
| `requires_grad` flag        | Fully supported |
| `backward()` method         | Fully supported |
| `zero_grad()` functionality | Fully supported |
| `no_grad()` context         | Fully supported |
| `detach()` operation        | Fully supported |
| Gradient accumulation       | Fully supported |
| Higher-order gradients      | Fully supported |
| Custom autograd functions   | Basic support   |
| Broadcasting                | Limited support |
| GPU support                 | Planned         |

## Implementation Details

### Backward Functions

Each operation implements a backward function with the signature:

```c
void op_backward(Function *fn, Tensor *grad_output);
```

Example: Multiplication backward

```c
void mul_backward(Function *fn, Tensor *grad_output) {
    Tensor *a = fn->ctx->saved_tensors[0];
    Tensor *b = fn->ctx->saved_tensors[1];

    // da/dx = b * grad_output
    if (fn->needs_input_grad[0]) {
        Tensor *grad_a = tensor_mul(b, grad_output);
        tensor_accumulate_grad(fn->inputs[0], grad_a);
        tensor_free(grad_a);
    }

    // db/dy = a * grad_output
    if (fn->needs_input_grad[1]) {
        Tensor *grad_b = tensor_mul(a, grad_output);
        tensor_accumulate_grad(fn->inputs[1], grad_b);
        tensor_free(grad_b);
    }
}
```

### Topological Sorting

The backward pass uses topological sorting to ensure operations are executed in the correct order:

1. Build graph from output tensor
1. Assign depth to each node (distance from output)
1. Sort nodes by depth (descending)
1. Execute backward functions in sorted order

This ensures gradients flow correctly through the computation graph.

## Troubleshooting

### Common Issues

**Problem**: Gradients are NULL after backward

- **Solution**: Ensure `requires_grad=true` on input tensors

**Problem**: Memory leaks

- **Solution**: Always free tensors and use `tensor_zero_grad()` between iterations

**Problem**: Incorrect gradients

- **Solution**: Enable anomaly detection to check for NaN/Inf

**Problem**: Slow backward pass

- **Solution**: Use `no_grad` mode for inference operations

## Future Enhancements

- [ ] Full broadcasting support
- [ ] Custom autograd functions API
- [ ] Sparse tensor gradients
- [ ] GPU/CUDA support
- [ ] Hook system (pre/post backward hooks)
- [ ] Gradient checkpointing for memory efficiency
- [ ] JIT compilation of backward functions
- [ ] Parallel backward pass

## Contributing

To add a new operation with gradient support:

1. Add operation type to `OpType` enum
1. Implement forward function
1. Implement backward function
1. Register backward function with operation
1. Add tests for forward and backward passes

Example skeleton:

```c
// Forward
Tensor *tensor_my_op(Tensor *input) {
    // Compute result
    Tensor *result = /* ... */;

    // Set up autograd
    Function *fn = autograd_function_create(OP_MY_OP, "MyOp");
    Tensor *saved[] = {input};
    autograd_context_save_for_backward(fn->ctx, saved, 1);
    autograd_function_set_backward(fn, my_op_backward);
    Tensor *inputs[] = {input};
    result = create_output_with_grad_fn(result, fn, inputs, 1);

    return result;
}

// Backward
void my_op_backward(Function *fn, Tensor *grad_output) {
    Tensor *input = fn->ctx->saved_tensors[0];

    if (fn->needs_input_grad[0]) {
        // Compute gradient
        Tensor *grad_input = /* d(output)/d(input) * grad_output */;
        tensor_accumulate_grad(fn->inputs[0], grad_input);
        tensor_free(grad_input);
    }
}
```

## License

See LICENSE.md for details.
