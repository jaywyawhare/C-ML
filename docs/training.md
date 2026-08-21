# Training API

Complete guide to training neural networks with C-ML.

## Table of Contents

- [Basic Training Loop](#basic-training-loop)
- [Optimizers](#optimizers)
- [LR Schedulers](#lr-schedulers)
- [Loss Functions](#loss-functions)
- [Dataset Hub](#dataset-hub)
- [Training Mode](#training-mode)
- [Metrics](#metrics)
- [Cleanup Context](#cleanup-context)
- [Best Practices](#best-practices)

## Basic Training Loop

```c
#include "cml.h"

cml_init();
cml_seed(42);

DeviceType device = cml_get_default_device();
DType dtype = cml_get_default_dtype();

Sequential* model = cml_nn_sequential();
model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(4, 16, dtype, device, true));
model = cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(16, 3, dtype, device, true));

Optimizer* optimizer = cml_optim_adam_for_model((Module*)model, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);

Dataset* ds = cml_dataset_load("iris");
dataset_normalize(ds, "minmax");
Dataset *train, *test;
dataset_split(ds, 0.8f, &train, &test);

cml_nn_module_set_training((Module*)model, true);
for (int epoch = 0; epoch < 100; epoch++) {
    cml_optim_zero_grad(optimizer);
    Tensor* outputs = cml_nn_module_forward((Module*)model, train->X);
    Tensor* loss = cml_nn_mse_loss(outputs, train->y);
    cml_backward(loss, NULL, false, false);
    cml_optim_step(optimizer);

    if ((epoch + 1) % 10 == 0) {
        float* loss_data = (float*)tensor_data_ptr(loss);
        printf("Epoch %d: Loss = %.6f\n", epoch + 1, loss_data ? loss_data[0] : 0.0f);
    }

    tensor_free(loss);
    tensor_free(outputs);
}

optimizer_free(optimizer);
module_free((Module*)model);
dataset_free(train);
dataset_free(test);
dataset_free(ds);
cml_cleanup();
```

## Optimizers

### Available Optimizers

| Optimizer | Constructor        | Key Parameters                      |
| --------- | ------------------ | ----------------------------------- |
| SGD       | `optim_sgd()`      | lr, momentum, weight_decay          |
| Adam      | `optim_adam()`     | lr, weight_decay, beta1, beta2, eps |
| AdamW     | `optim_adamw()`    | lr, weight_decay, beta1, beta2, eps |
| RMSprop   | `optim_rmsprop()`  | lr, weight_decay, alpha, eps        |
| Adagrad   | `optim_adagrad()`  | lr, weight_decay, eps               |
| AdaDelta  | `optim_adadelta()` | rho, weight_decay, eps              |

### Model-based Constructors (Recommended)

Automatically collect parameters from the model:

```c
Optimizer* opt = cml_optim_adam_for_model((Module*)model, 0.001f, 0.0f, 0.9f, 0.999f, 1e-8f);
Optimizer* sgd = cml_optim_sgd_for_model((Module*)model, 0.01f, 0.9f, 0.0f);
```

### Manual Parameter Collection

```c
Parameter** params;
int num_params;
module_collect_parameters((Module*)model, &params, &num_params, true);
Optimizer* opt = optim_adam(params, num_params, 0.001f, 0.0f, 0.9f, 0.999f, 1e-8f);
```

### Optimizer Operations

```c
cml_optim_zero_grad(optimizer);
cml_optim_step(optimizer);
optimizer_set_lr(optimizer, 0.01f);
optimizer_free(optimizer);
```

## LR Schedulers

Learning rate schedulers adjust the learning rate during training.

### Available Schedulers

```c
#include "optim/lr_scheduler.h"

LRScheduler* sched = lr_scheduler_step(optimizer, 30, 0.1f);
LRScheduler* sched = lr_scheduler_exponential(optimizer, 0.95f);
LRScheduler* sched = lr_scheduler_cosine_annealing(optimizer, 100, 1e-6f);
LRScheduler* sched = lr_scheduler_reduce_on_plateau(optimizer, 0.1f, 10, 1e-4f, true);

int milestones[] = {30, 60, 90};
LRScheduler* sched = lr_scheduler_multi_step(optimizer, milestones, 3, 0.1f);
```

### Usage

```c
for (int epoch = 0; epoch < num_epochs; epoch++) {
    // ... training loop ...

    lr_scheduler_step_epoch(sched);
    lr_scheduler_step_metric(sched, val_loss);  // for ReduceOnPlateau

    float current_lr = lr_scheduler_get_lr(sched);
}

lr_scheduler_free(sched);
```

See `examples/tutorials/lr_scheduler.c` for a comparison of all schedulers.

## Loss Functions

```c
Tensor* loss = cml_nn_mse_loss(outputs, targets);           // Mean Squared Error
Tensor* loss = cml_nn_mae_loss(outputs, targets);           // Mean Absolute Error
Tensor* loss = cml_nn_bce_loss(outputs, targets);           // Binary Cross Entropy
Tensor* loss = cml_nn_cross_entropy_loss(outputs, targets); // Cross Entropy
Tensor* loss = cml_nn_huber_loss(outputs, targets, 1.0f);   // Huber Loss
Tensor* loss = cml_nn_kl_div_loss(outputs, targets);        // KL Divergence
```

Additional losses (from `autograd/loss_functions.h`):

```c
Tensor* loss = tensor_hinge_loss(outputs, targets);
Tensor* loss = tensor_focal_loss(outputs, targets, alpha, gamma);
Tensor* loss = tensor_smooth_l1_loss(outputs, targets, beta);
```

## Dataset Hub

Load datasets with one line:

```c
Dataset* ds = cml_dataset_load("iris");
Dataset* ds = cml_dataset_load("boston");
Dataset* ds = cml_dataset_load("mnist");
Dataset* ds = cml_dataset_from_csv("data.csv", -1);

dataset_normalize(ds, "minmax");

Dataset *train, *test;
dataset_split(ds, 0.8f, &train, &test);
```

See [docs/datasets.md](datasets.md) for the full list of supported datasets.

## Training Mode

```c
cml_nn_module_set_training((Module*)model, true);
cml_nn_module_eval((Module*)model);
```

Layers that behave differently in training vs evaluation:

- **Dropout**: Drops units during training, passes through during eval
- **BatchNorm2d**: Uses batch statistics during training, running statistics during eval

## Metrics

```c
training_metrics_set_expected_epochs(num_epochs);
training_metrics_auto_capture_train_accuracy(accuracy);
training_metrics_evaluate_dataset((Module*)model, dataset, loss_fn, is_validation);
training_metrics_mark_early_stop(actual_epochs);

TrainingMetrics* metrics = training_metrics_get_global();
training_metrics_set_learning_rate(metrics, lr, "StepLR");
training_metrics_register_model((Module*)model);
```

### Advancing the epoch counter

`training_metrics_set_expected_epochs()` switches metrics into **manual epoch
control**. Nothing then advances the epoch index on its own, so a hand-written
loop must close each epoch:

```c
training_metrics_set_expected_epochs(num_epochs);   /* -> manual epoch control */

for (int epoch = 0; epoch < num_epochs; epoch++) {
    /* ... forward / backward / step ... */
    training_metrics_complete_epoch();              /* required, once per epoch */
}
```

Omit it and every epoch overwrites slot 0, so the dashboard shows a single flat
point no matter how long training ran. If you break out early (early stopping),
close the epoch before breaking.

Loops that do *not* call `set_expected_epochs()` use automatic epoch detection,
which advances on `zero_grad`; there, call `training_metrics_complete_epoch()`
once at the end of training so the final epoch is not missed.

### Weight and gradient distributions

Under `VIZ=1`, each epoch also records the *shape* of the weight and gradient
populations, not just a single gradient-norm scalar:

```c
typedef struct {
    float min, p25, p50, p75, max;
    float mean, std;
    float frac_zero;   /* share of exactly-zero entries: dead ReLUs, pruned weights */
} DistributionSummary;
```

A norm averages away the failure modes that matter — a few exploding channels, a
layer that stopped moving, units that died. The bands show them directly.
Percentiles come from a fixed-bin histogram over `[min, max]` rather than a sort,
which would mean ordering every parameter in the model once per epoch.

Capture is automatic (it hangs off the optimizer step, where gradients are still
attached) and exported to `training.json` as `grad_distribution` /
`weight_distribution`. To summarise a parameter set yourself:

```c
training_metrics_record_distributions(metrics, epoch, (void**)params, num_params);
```

Validation curves (`epoch_validation_losses` / `epoch_validation_accuracies`) are
exported alongside the training curves, so train-vs-validation divergence is
visible in the dashboard.

## Cleanup Context

```c
CleanupContext* cleanup = cleanup_context_create();

cleanup_register_model(cleanup, (Module*)model);
cleanup_register_optimizer(cleanup, optimizer);
cleanup_register_tensor(cleanup, tensor);
cleanup_register_dataset(cleanup, dataset);

cml_register_cleanup_context(cleanup);
```

## Best Practices

### Memory Management

- Free tensors after each iteration: `tensor_free(loss); tensor_free(outputs);`
- Call `cml_reset_ir_context()` after each batch to prevent IR node accumulation
- **Free that batch's tensors *before* the reset, not after.** The reset reclaims
  the execution pool their storage came from, so freeing afterwards touches memory
  the pool has already taken back.
- **Do not hold a tensor across `cml_reset_ir_context()`.** A tensor materialised
  through the graph has its storage in that pool, so the reset frees it and any
  pointer you kept is dangling. Keep long-lived data (a dataset, say) in plain
  host memory and build tensors from it per iteration:

```c
/* Dataset owned by the caller — unaffected by graph resets. */
float* inputs = malloc((size_t)n * dim * sizeof(float));

for (int step = 0; step < steps; step++) {
    Tensor* batch = tensor_zeros_2d(batch_size, dim);
    memcpy(tensor_data_ptr(batch), inputs + offset, batch_bytes);
    /* ... forward / backward / step ... */
    tensor_free(batch);          /* before the reset */
    cml_reset_ir_context();
}
```
- Use `CleanupContext` for centralized resource management
- Don't manually free a cleanup context registered with `cml_register_cleanup_context()`

### Gradient Management

- Always call `cml_optim_zero_grad()` before backward pass
- Use `cml_backward()` with `retain_graph=false` unless you need multiple backward passes

### Training Mode

- Set training mode before training: `cml_nn_module_set_training((Module*)model, true)`
- Set eval mode before inference: `cml_nn_module_eval((Module*)model)`

### Performance

- Use appropriate data types (`DTYPE_FLOAT32` for most cases)
- Process data in batches for better throughput
- Use GPU when available for large computations
