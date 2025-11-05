# Training Neural Networks with C-ML

This guide explains how to train neural networks using C-ML. C-ML follows a flexible training pattern where users write their own training loops, giving full control over the training process.

## Table of Contents

1. [Overview](#overview)
1. [Training Pattern](#training-pattern)
1. [Model Definition](#model-definition)
1. [Parameter Collection](#parameter-collection)
1. [Optimizer Creation](#optimizer-creation)
1. [Loss Functions](#loss-functions)
1. [Training Loop](#training-loop)
1. [Learning Rate Scheduling and Early Stopping](#learning-rate-scheduling-and-early-stopping)
1. [Complete Example](#complete-example)
1. [Best Practices](#best-practices)

## Overview

C-ML's training pattern provides:

- **Flexibility**: Full control over training loops
- **Modularity**: Separate model, optimizer, and loss function definitions
- **Automatic Differentiation**: Automatic gradient computation via autograd
- **Memory Management**: Proper cleanup of tensors and modules

The training process consists of:

1. Model definition using `Module` and layers
1. Parameter collection from the model
1. Optimizer creation with parameters
1. Loss function selection
1. Custom training loop with forward/backward passes

## Training Pattern

### Basic Training Flow

```c
// 1. Initialize library
cml_init();

// 2. Create model
Module *model = create_model();

// 3. Collect parameters
Parameter **params;
int num_params;
module_collect_parameters(model, &params, &num_params, true);

// 4. Create optimizer
Optimizer *optimizer = optim_adam(params, num_params, 0.001f, ...);

// 5. Training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    for (int batch = 0; batch < num_batches; batch++) {
        // Zero gradients
        optimizer_zero_grad(optimizer);

        // Forward pass
        Tensor *outputs = module_forward(model, inputs);

        // Compute loss
        Tensor *loss = tensor_mse_loss(outputs, targets);

        // Backward pass
        tensor_backward(loss, NULL, false, false);

        // Update parameters
        optimizer_step(optimizer);

        // Cleanup
        tensor_free(loss);
        tensor_free(outputs);
    }
}

// 6. Cleanup
optimizer_free(optimizer);
CM_FREE(params);
module_free(model);
cml_cleanup();
```

## Model Definition

Models are created using the `Module` base class and various layers:

```c
#include "cml.h"

// Create a sequential model
Sequential *model = nn_sequential();

// Add layers
sequential_add(model, (Module*)nn_linear(784, 128, DTYPE_FLOAT32, DEVICE_CPU, true));
sequential_add(model, (Module*)nn_relu(false));
sequential_add(model, (Module*)nn_linear(128, 64, DTYPE_FLOAT32, DEVICE_CPU, true));
sequential_add(model, (Module*)nn_relu(false));
sequential_add(model, (Module*)nn_linear(64, 10, DTYPE_FLOAT32, DEVICE_CPU, true));
sequential_add(model, (Module*)nn_softmax(1));

// Set training mode
module_set_training((Module*)model, true);

// Print model summary
summary((Module*)model);
```

### Custom Models

You can also create custom models by inheriting from `Module`:

```c
typedef struct {
    Module base;
    Module *layer1;
    Module *layer2;
} MyModel;

Module *create_my_model(void) {
    MyModel *model = CM_MALLOC(sizeof(MyModel));
    module_init((Module*)model, "MyModel", NULL);

    model->layer1 = (Module*)nn_linear(10, 20, DTYPE_FLOAT32, DEVICE_CPU, true);
    model->layer2 = (Module*)nn_linear(20, 1, DTYPE_FLOAT32, DEVICE_CPU, true);

    return (Module*)model;
}

static Tensor *my_model_forward(Module *module, Tensor *input) {
    MyModel *model = (MyModel*)module;
    Tensor *x = module_forward(model->layer1, input);
    x = module_forward((Module*)nn_relu(false), x);
    x = module_forward(model->layer2, x);
    return x;
}
```

## Parameter Collection

After creating a model, collect all trainable parameters:

```c
Parameter **params = NULL;
int num_params = 0;

// Collect all parameters recursively
if (module_collect_parameters(model, &params, &num_params, true) != 0) {
    LOG_ERROR("Failed to collect parameters");
    return;
}

printf("Model has %d parameters\n", num_params);

// Don't forget to free the params array when done
// CM_FREE(params);
```

The `recursive` parameter controls whether to collect parameters from submodules:

- `true`: Collect from all submodules (recommended for Sequential models)
- `false`: Collect only from the top-level module

## Optimizer Creation

C-ML provides two optimizers: SGD and Adam.

### SGD Optimizer

```c
Optimizer *optimizer = optim_sgd(
    params,           // Parameter array
    num_params,       // Number of parameters
    0.01f,            // Learning rate
    0.9f,             // Momentum (0.0 = no momentum)
    0.0001f           // Weight decay (L2 regularization)
);
```

### Adam Optimizer

```c
Optimizer *optimizer = optim_adam(
    params,           // Parameter array
    num_params,       // Number of parameters
    0.001f,           // Learning rate
    0.0f,             // Weight decay
    0.9f,             // Beta1 (first moment decay)
    0.999f,           // Beta2 (second moment decay)
    1e-8f             // Epsilon (numerical stability)
);
```

### Optimizer Usage

```c
// Zero gradients before backward pass
optimizer_zero_grad(optimizer);

// After backward pass, update parameters
optimizer_step(optimizer);

// Get optimizer name
printf("Using optimizer: %s\n", optimizer_get_name(optimizer));

// Cleanup
optimizer_free(optimizer);
```

## Loss Functions

C-ML provides several loss functions:

### Mean Squared Error (MSE)

```c
Tensor *loss = tensor_mse_loss(outputs, targets);
```

### Mean Absolute Error (MAE)

```c
Tensor *loss = tensor_mae_loss(outputs, targets);
```

### Binary Cross Entropy

```c
Tensor *loss = tensor_bce_loss(outputs, targets);
```

### Cross Entropy

```c
Tensor *loss = tensor_cross_entropy_loss(outputs, targets);
```

All loss functions:

- Support automatic differentiation
- Return a scalar tensor
- Can be used in computation graphs

## Training Loop

A typical training loop includes:

1. **Forward Pass**: Compute model outputs
1. **Loss Calculation**: Compute loss between outputs and targets
1. **Backward Pass**: Compute gradients
1. **Parameter Update**: Update model parameters using optimizer

```c
for (int epoch = 0; epoch < num_epochs; epoch++) {
    float epoch_loss = 0.0f;
    int num_batches = 0;

    for (int batch = 0; batch < num_batches; batch++) {
        // Zero gradients
        optimizer_zero_grad(optimizer);

        // Forward pass
        Tensor *outputs = module_forward(model, inputs);
        if (!outputs) {
            LOG_ERROR("Forward pass failed");
            continue;
        }

        // Compute loss
        Tensor *loss = tensor_mse_loss(outputs, targets);
        if (!loss) {
            LOG_ERROR("Loss computation failed");
            tensor_free(outputs);
            continue;
        }

        // Backward pass
        tensor_backward(loss, NULL, false, false);

        // Get loss value for logging
        float *loss_data = (float*)tensor_data_ptr(loss);
        if (loss_data) {
            epoch_loss += loss_data[0];
        }

        // Update parameters
        optimizer_step(optimizer);

        // Cleanup
        tensor_free(loss);
        tensor_free(outputs);
    }

    epoch_loss /= num_batches;
    printf("Epoch %d/%d - Loss: %.6f\n", epoch + 1, num_epochs, epoch_loss);
}
```

## Learning Rate Scheduling and Early Stopping

C-ML exposes simple hooks to adjust learning rates at runtime. You can implement schedulers in your loop and perform early stopping with a patience counter.

Example: step LR every N epochs and stop if no improvement for P epochs.

```c
int num_epochs = 100;
float best = INFINITY;
int patience = 10, no_improve = 0;
int step_size = 20;    // decay every 20 epochs
float gamma = 0.5f;    // lr *= 0.5

for (int epoch = 0; epoch < num_epochs; epoch++) {
    // ... training over batches, accumulate epoch_loss ...

    // Early stopping
    if (epoch_loss < best - 1e-5f) {
        best = epoch_loss;
        no_improve = 0;
    } else if (++no_improve >= patience) {
        printf("Early stopping at epoch %d (best %.6f)\n", epoch + 1, best);
        break;
    }

    // Step LR scheduler
    if ((epoch + 1) % step_size == 0) {
        float current_lr = optimizer_get_group_lr(optimizer, 0);
        float new_lr = current_lr * gamma;
        optimizer_set_lr(optimizer, new_lr);
        printf("  LR -> %.6f\n", new_lr);
    }
}
```

See `examples/training_loop_example.c` for a complete usage pattern.

## Complete Example

Here's a complete example training a model on the XOR dataset:

```c
#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    // Initialize library
    cml_init();

    // Create XOR dataset
    float X_data[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float y_data[4][1] = {{0}, {1}, {1}, {0}};

    int X_shape[] = {4, 2};
    int y_shape[] = {4, 1};

    Tensor *X = tensor_from_array(X_data, X_shape, 2, DTYPE_FLOAT32, DEVICE_CPU);
    Tensor *y = tensor_from_array(y_data, y_shape, 2, DTYPE_FLOAT32, DEVICE_CPU);

    // Create model
    Sequential *model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_linear(4, 1, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_sigmoid());

    // Print model summary
    summary((Module*)model);

    // Collect parameters
    Parameter **params;
    int num_params;
    module_collect_parameters((Module*)model, &params, &num_params, true);

    // Create optimizer
    Optimizer *optimizer = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);

    // Training loop
    int num_epochs = 1000;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        optimizer_zero_grad(optimizer);

        Tensor *outputs = module_forward((Module*)model, X);
        Tensor *loss = tensor_mse_loss(outputs, y);
        tensor_backward(loss, NULL, false, false);
        optimizer_step(optimizer);

        if ((epoch + 1) % 100 == 0) {
            float *loss_data = (float*)tensor_data_ptr(loss);
            printf("Epoch %d - Loss: %.6f\n", epoch + 1, loss_data ? loss_data[0] : 0.0f);
        }

        tensor_free(loss);
        tensor_free(outputs);
    }

    // Cleanup
    optimizer_free(optimizer);
    CM_FREE(params);
    module_free((Module*)model);
    tensor_free(y);
    tensor_free(X);
    cml_cleanup();

    return 0;
}
```

## Best Practices

### 1. Memory Management

- Always free tensors after use: `tensor_free(tensor)`
- Free modules when done: `module_free((Module*)model)`
- Free optimizer: `optimizer_free(optimizer)`
- Free parameter arrays: `CM_FREE(params)`
- Call `cml_cleanup()` at the end

### 2. Gradient Management

- Always call `optimizer_zero_grad()` before backward pass
- Use `tensor_backward()` with `retain_graph=false` unless you need multiple backward passes
- Check for gradient computation: `tensor_has_grad(tensor)`

### 3. Training Mode

- Set training mode: `module_set_training(model, true)`
- Set evaluation mode: `module_set_training(model, false)`
- Some layers (e.g., Dropout, BatchNorm) behave differently in training vs evaluation

### 4. Error Handling

- Check return values from functions
- Use `LOG_ERROR()` for error logging
- Validate tensor shapes before operations

### 5. Performance

- Reuse tensors when possible
- Avoid unnecessary tensor allocations
- Use appropriate data types (float32 vs float64)
- Consider batch size for memory efficiency

### 6. Debugging

- Use `summary()` to inspect model structure
- Check parameter counts: `module_get_total_parameters(model)`
- Enable logging for debugging: `set_log_level(LOG_LEVEL_DEBUG)`
- Use anomaly detection: `autograd_set_anomaly_mode(true)`

## API Reference

### Key Functions

- **`module_collect_parameters()`**: Collect all parameters from a module
- **`module_get_parameters()`**: Get parameters from a module (non-recursive)
- **`optim_sgd()`**: Create SGD optimizer
- **`optim_adam()`**: Create Adam optimizer
- **`optimizer_zero_grad()`**: Zero all gradients
- **`optimizer_step()`**: Update parameters using gradients
- **`tensor_mse_loss()`**: Mean squared error loss
- **`tensor_backward()`**: Compute gradients
- **`module_forward()`**: Forward pass through model
- **`summary()`**: Print model summary

## See Also

- [Autograd System](AUTOGRAD.md) - Understanding automatic differentiation
- [Neural Network Layers](NN_LAYERS.md) - Available layers and usage
- [Examples](../examples/) - Complete example programs
