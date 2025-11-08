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

## Training Metrics

C-ML includes built-in training metrics tracking that **automatically records training progress** without any manual tracking code. Metrics are captured by default when you use `cml_init()` and `cml_cleanup()`.

### Automatic Metrics Capture

Metrics are automatically captured during training - no manual tracking needed:

```c
#include "cml.h"
#include "Core/cleanup.h"

int main(void) {
    CleanupContext *cleanup = cleanup_context_create();
    cml_init(); // Automatically initializes global metrics tracking

    // Create model
    Sequential *model = nn_sequential();
    // ... add layers ...
    cleanup_register_model(cleanup, (Module*)model);
    training_metrics_register_model((Module*)model); // Register for architecture export

    // Create optimizer
    Parameter **params;
    int num_params;
    module_collect_parameters((Module*)model, &params, &num_params, true);
    cleanup_register_params(cleanup, params);

    Optimizer *optimizer = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    cleanup_register_optimizer(cleanup, optimizer);

    // Set expected epochs for UI
    training_metrics_set_expected_epochs(100);

    // Training loop - metrics are automatically captured!
    for (int epoch = 0; epoch < 100; epoch++) {
        optimizer_zero_grad(optimizer); // Automatically detects new epoch

        Tensor *outputs = module_forward((Module*)model, X);
        Tensor *loss = tensor_mse_loss(outputs, y);
        tensor_backward(loss, NULL, false, false); // Automatically captures loss
        optimizer_step(optimizer); // Automatically captures LR and gradient norm

        // Capture training accuracy
        float accuracy = calculate_accuracy(outputs, y);
        training_metrics_auto_capture_train_accuracy(accuracy);

        tensor_free(loss);
        tensor_free(outputs);
    }

    // Metrics are automatically exported to training.json on cml_cleanup()

cleanup:
    cleanup_context_free(cleanup);
    cml_cleanup(); // Automatically exports final metrics
    return 0;
}
```

### Automatic Validation and Test Evaluation

Use `training_metrics_evaluate_dataset()` to automatically evaluate and record metrics:

```c
// Split dataset
Dataset *full_dataset = dataset_from_arrays(X_all, y_all, num_samples, input_size, output_size);
Dataset *train_dataset, *val_dataset, *test_dataset;
dataset_split_three(full_dataset, 0.7f, 0.15f, 0.15f,
                    &train_dataset, &val_dataset, &test_dataset);

// Training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    // ... training on train_dataset ...

    // Automatically evaluate on validation set and record metrics
    training_metrics_evaluate_dataset((Module*)model, val_dataset,
                                      tensor_mse_loss, true);
}

// Final evaluation on test set
training_metrics_evaluate_dataset((Module*)model, test_dataset,
                                  tensor_mse_loss, false);
```

### Early Stopping Support

C-ML tracks early stopping status automatically:

```c
int num_epochs = 100;
int patience = 15;
float best_loss = INFINITY;
int no_improve_epochs = 0;

training_metrics_set_expected_epochs(num_epochs);

for (int epoch = 0; epoch < num_epochs; epoch++) {
    // ... training ...

    // Early stopping logic
    if (epoch_loss < best_loss - 1e-5f) {
        best_loss = epoch_loss;
        no_improve_epochs = 0;
    } else {
        no_improve_epochs++;
        if (no_improve_epochs >= patience) {
            training_metrics_mark_early_stop(epoch); // Mark early stopping
            break;
        }
    }
}
```

The UI will display:

- Early stopping status badge
- Actual vs expected epochs
- Visual indicators on charts

### Learning Rate Scheduling

Track LR scheduler information for visualization:

```c
TrainingMetrics *metrics = training_metrics_get_global();
if (metrics) {
    training_metrics_set_learning_rate(metrics, initial_lr, "StepLR");
    char params_buf[128];
    snprintf(params_buf, sizeof(params_buf), "step_size=30,gamma=0.5");
    training_metrics_set_lr_schedule_params(metrics, params_buf);
}

// In training loop, adjust LR as needed
if ((epoch + 1) % lr_step_size == 0) {
    float current_lr = optimizer_get_group_lr(optimizer, 0);
    float new_lr = current_lr * lr_gamma;
    optimizer_set_lr(optimizer, new_lr);
}
```

The UI will display:

- Current learning rate
- Scheduler type (e.g., "StepLR", "Constant")
- Scheduler parameters (e.g., "step_size=30,gamma=0.5")

### Automatic JSON Export

Metrics are automatically exported to `training.json`:

- **Real-time updates**: JSON is updated after each loss capture and optimizer step (when `VIZ=1` or `CML_VIZ=1`)
- **Final export**: Complete metrics exported on `cml_cleanup()`
- **Location**: `viz-ui/public/training.json` (for UI visualization)
- **Trigger**: Set `VIZ=1` environment variable when running your program to enable automatic export

The exported JSON includes:

- Training, validation, and test losses/accuracies per epoch
- Epoch times and total training time
- Learning rates and gradient norms per epoch
- Loss reduction rate and stability metrics
- Model architecture summary
- Early stopping status (if applicable)
- LR scheduler information

### Metrics API

```c
// Global metrics (automatically initialized)
TrainingMetrics *training_metrics_get_global(void);

// Register model for architecture export
void training_metrics_register_model(Module *model);

// Set expected epochs for UI
void training_metrics_set_expected_epochs(size_t num_epochs);

// Capture training accuracy
void training_metrics_auto_capture_train_accuracy(float train_accuracy);

// Evaluate dataset and record metrics
int training_metrics_evaluate_dataset(Module *model, Dataset *dataset,
                                      Tensor *(*loss_fn)(Tensor*, Tensor*),
                                      bool is_validation);

// Early stopping
void training_metrics_mark_early_stop(size_t actual_epochs);

// Learning rate scheduling
void training_metrics_set_learning_rate(TrainingMetrics *metrics,
                                         float lr, const char *schedule);
void training_metrics_set_lr_schedule_params(TrainingMetrics *metrics,
                                              const char *params);
```

## Learning Rate Scheduling and Early Stopping

C-ML provides hooks to adjust learning rates at runtime and track early stopping. The metrics system automatically captures LR changes and early stopping status for visualization.

### Visualization with VIZ=1

To enable automatic graph and metrics export during training, set the `VIZ=1` environment variable:

```bash
# Automatic visualization launch
VIZ=1 ./build/main
VIZ=1 ./build/examples/test
```

This will:

1. Automatically launch the visualization UI before your program runs
1. Start FastAPI server (port 8001) and React frontend (port 5173)
1. Run your program with `CML_VIZ=1` set (enables automatic export)
1. Export graph and metrics to JSON files during training
1. Open browser to `http://localhost:5173` for real-time visualization

Alternatively, you can manually launch the visualization:

```bash
python scripts/viz.py <executable> [args...]
```

### Complete Example with Early Stopping and LR Scheduling

```c
#include "cml.h"
#include "Core/cleanup.h"

int main(void) {
    CleanupContext *cleanup = cleanup_context_create();
    cml_init();

    // ... create model and optimizer ...

    int num_epochs = 2000;
    int patience = 15;
    float improvement_tol = 1e-5f;
    int lr_step_size = 30;
    float lr_gamma = 0.5f;
    float initial_lr = 0.01f;

    float best_loss = INFINITY;
    int no_improve_epochs = 0;

    // Set expected epochs and LR scheduler info
    training_metrics_set_expected_epochs(num_epochs);
    TrainingMetrics *metrics = training_metrics_get_global();
    if (metrics) {
        training_metrics_set_learning_rate(metrics, initial_lr, "StepLR");
        char params_buf[128];
        snprintf(params_buf, sizeof(params_buf), "step_size=%d,gamma=%.2f",
                 lr_step_size, lr_gamma);
        training_metrics_set_lr_schedule_params(metrics, params_buf);
    }

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // ... training loop ...

        // Early stopping
        if (epoch_loss < best_loss - improvement_tol) {
            best_loss = epoch_loss;
            no_improve_epochs = 0;
        } else {
            no_improve_epochs++;
            if (no_improve_epochs >= patience) {
                printf("Early stopping at epoch %d\n", epoch + 1);
                training_metrics_mark_early_stop(epoch);
                break;
            }
        }

        // Learning rate scheduling (StepLR)
        if ((epoch + 1) % lr_step_size == 0) {
            float current_lr = optimizer_get_group_lr(optimizer, 0);
            float new_lr = current_lr * lr_gamma;
            optimizer_set_lr(optimizer, new_lr);
            printf("  [Epoch %d] LR decayed: %.6f -> %.6f\n",
                   epoch + 1, current_lr, new_lr);
        }
    }

cleanup:
    cleanup_context_free(cleanup);
    cml_cleanup();
    return 0;
}
```

See `examples/early_stopping_lr_scheduler.c` for a complete working example.

## Complete Example

Here's a complete example training a model on the XOR dataset using automatic metrics and centralized cleanup:

```c
#include "cml.h"
#include "Core/cleanup.h"
#include <stdio.h>

int main(void) {
    CleanupContext* cleanup = cleanup_context_create();
    if (!cleanup) return 1;

    cml_init();

    Sequential* model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_linear(4, 1, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_sigmoid());

    cleanup_register_model(cleanup, (Module*)model);
    training_metrics_register_model((Module*)model);

    summary((Module*)model);

    Parameter** params;
    int num_params;
    module_collect_parameters((Module*)model, &params, &num_params, true);
    cleanup_register_params(cleanup, params);

    Optimizer* optimizer = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    cleanup_register_optimizer(cleanup, optimizer);

    int X_shape[] = {4, 2};
    int y_shape[] = {4, 1};
    Tensor* X = tensor_empty(X_shape, 2, DTYPE_FLOAT32, DEVICE_CPU);
    Tensor* y = tensor_empty(y_shape, 2, DTYPE_FLOAT32, DEVICE_CPU);
    cleanup_register_tensor(cleanup, X);
    cleanup_register_tensor(cleanup, y);

    float* X_data = (float*)tensor_data_ptr(X);
    float* y_data = (float*)tensor_data_ptr(y);
    X_data[0] = 0.0f; X_data[1] = 0.0f; y_data[0] = 0.0f;
    X_data[2] = 0.0f; X_data[3] = 1.0f; y_data[1] = 1.0f;
    X_data[4] = 1.0f; X_data[5] = 0.0f; y_data[2] = 1.0f;
    X_data[6] = 1.0f; X_data[7] = 1.0f; y_data[3] = 0.0f;

    training_metrics_set_expected_epochs(1000);

    for (int epoch = 0; epoch < 1000; epoch++) {
        optimizer_zero_grad(optimizer);
        Tensor* outputs = module_forward((Module*)model, X);
        Tensor* loss = tensor_mse_loss(outputs, y);
        tensor_backward(loss, NULL, false, false);
        optimizer_step(optimizer);

        if ((epoch + 1) % 100 == 0) {
            float* loss_data = (float*)tensor_data_ptr(loss);
            printf("Epoch %d - Loss: %.6f\n", epoch + 1, loss_data ? loss_data[0] : 0.0f);
        }

        tensor_free(loss);
        tensor_free(outputs);
    }

cleanup:
    cleanup_context_free(cleanup);
    cml_cleanup();
    return 0;
}
```

For more advanced examples, see:

- `main.c` - Simple XOR example
- `examples/test.c` - Train/val/test splits with automatic metrics
- `examples/early_stopping_lr_scheduler.c` - Early stopping and LR scheduling

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
