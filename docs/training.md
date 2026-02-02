# Training API

Complete guide to training neural networks with C-ML.

## Table of Contents

- [Basic Training Loop](#basic-training-loop)
- [Optimizers](#optimizers)
- [Loss Functions](#loss-functions)
- [Data Loading](#data-loading)
- [Training Mode](#training-mode)
- [Metrics](#metrics)
- [Cleanup Context](#cleanup-context)
- [Best Practices](#best-practices)

## Basic Training Loop

```c
#include "cml.h"

cml_init();
cml_seed(42);

Sequential* model = nn_sequential();
model = sequential_add_chain(model, (Module*)nn_linear(784, 128, dtype, device, true));
model = sequential_add_chain(model, (Module*)nn_relu(false));
model = sequential_add_chain(model, (Module*)nn_linear(128, 10, dtype, device, true));

Optimizer* optimizer = optim_adam_for_model((Module*)model, lr, weight_decay, beta1, beta2, eps);

Dataset* dataset = dataset_xor();
Tensor* inputs = dataset->X;
Tensor* targets = dataset->y;

for (int epoch = 0; epoch < num_epochs; epoch++) {
    optimizer_zero_grad(optimizer);
    Tensor* outputs = module_forward((Module*)model, inputs);
    Tensor* loss = tensor_mse_loss(outputs, targets);
    tensor_backward(loss, NULL, false, false);
    optimizer_step(optimizer);
    tensor_free(loss);
    tensor_free(outputs);
}

cml_cleanup();
```

## Optimizers

```c
// Automatic parameter collection
Optimizer* optimizer = optim_adam_for_model((Module*)model, lr, weight_decay, beta1, beta2, eps);
Optimizer* sgd = optim_sgd_for_model((Module*)model, lr, momentum, weight_decay);

// Manual
Parameter** params;
int num_params;
module_collect_parameters(model, &params, &num_params, true);
Optimizer* optimizer = optim_adam(params, num_params, lr, weight_decay, beta1, beta2, eps);
Optimizer* sgd = optim_sgd(params, num_params, lr, momentum, weight_decay);

optimizer_zero_grad(optimizer);
optimizer_step(optimizer);
optimizer_free(optimizer);
```

## Loss Functions

```c
Tensor* loss = tensor_mse_loss(outputs, targets);
Tensor* loss = tensor_mae_loss(outputs, targets);
Tensor* loss = tensor_bce_loss(outputs, targets);
Tensor* loss = tensor_cross_entropy_loss(outputs, targets);
```

## Data Loading

```c
Dataset* dataset = dataset_from_arrays(X, y, num_samples, input_size, output_size);
Dataset* xor_data = dataset_xor();
Dataset* random_data = dataset_random_classification(num_samples, input_size, output_size);

DataLoader* loader = dataloader_create(dataset, batch_size, shuffle);
while (dataloader_has_next(loader)) {
    Batch* batch = dataloader_next_batch(loader);
    // Use batch->X and batch->y
    batch_free(batch);
}
dataloader_reset(loader);

Dataset* train, *val, *test;
dataset_split_three(full_dataset, train_ratio, val_ratio, &train, &val, &test);
```

## Training Mode

```c
module_set_training((Module*)model, true);   // Training mode
module_set_training((Module*)model, false);   // Evaluation mode
```

## Metrics

```c
// Set expected epochs for UI
training_metrics_set_expected_epochs(num_epochs);

// Capture training accuracy
training_metrics_auto_capture_train_accuracy(accuracy);

// Evaluate dataset and record metrics
training_metrics_evaluate_dataset((Module*)model, dataset, loss_fn, is_validation);

// Early stopping
training_metrics_mark_early_stop(actual_epochs);

// Learning rate scheduling info
TrainingMetrics* metrics = training_metrics_get_global();
training_metrics_set_learning_rate(metrics, lr, "StepLR");
training_metrics_set_lr_schedule_params(metrics, "step_size=30,gamma=0.5");

// Register model for architecture export
training_metrics_register_model((Module*)model);
```

## Cleanup Context

```c
// Centralized resource management
CleanupContext* cleanup = cleanup_context_create();

// Register resources
cleanup_register_model(cleanup, (Module*)model);
cleanup_register_optimizer(cleanup, optimizer);
cleanup_register_tensor(cleanup, tensor);
cleanup_register_params(cleanup, params);
cleanup_register_dataset(cleanup, dataset);
cleanup_register_memory(cleanup, ptr);

// Free all registered resources
cleanup_context_free(cleanup);

// Auto-register with library
cml_register_cleanup_context(cleanup);  // Auto-freed on cml_cleanup()
```

## Best Practices

### Memory Management

- Always free tensors after use: `tensor_free(tensor)`
- Use `CleanupContext` for centralized resource management
- Free modules when done: `module_free((Module*)model)`
- Free optimizer: `optimizer_free(optimizer)`

### Gradient Management

- Always call `cml_optim_zero_grad()` before backward pass
- Use `cml_backward()` with `retain_graph=false` unless you need multiple backward passes
- Check for gradient computation: `tensor_has_grad(tensor)`

### Training Mode

- Set training mode: `cml_nn_module_set_training((Module*)model, true)`
- Set evaluation mode: `cml_nn_module_eval((Module*)model)`
- Some layers (Dropout, BatchNorm) behave differently in training vs evaluation

### Error Handling

- Check return values from functions
- Use `CML_HAS_ERRORS()` after operations
- Validate tensor shapes before operations

### Performance

- Reuse tensors when possible
- Avoid unnecessary tensor allocations
- Use appropriate data types (float32 vs float64)
- Consider batch size for memory efficiency
- Use GPU when available for large computations
