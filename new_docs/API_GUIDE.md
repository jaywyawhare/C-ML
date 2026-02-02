# CML API Reference Guide

Complete reference for all CML APIs.

## Table of Contents

1. [Initialization & Cleanup](#initialization--cleanup)
2. [Tensor API](#tensor-api)
3. [Autograd API](#autograd-api)
4. [Neural Network API](#neural-network-api)
5. [Optimizer API](#optimizer-api)
6. [Loss Functions](#loss-functions)
7. [Device & Memory](#device--memory)
8. [Utilities](#utilities)

## Initialization & Cleanup

### `void cml_init(void)`

Initialize the CML library. Must be called before using any CML functions.

```c
cml_init();
```

**Returns:** void

**Side effects:**
- Initializes MLIR context
- Sets up memory pools
- Initializes device detection

### `void cml_cleanup(void)`

Clean up CML resources. Should be called at program end.

```c
cml_cleanup();
```

**Returns:** void

**Side effects:**
- Frees all allocated memory
- Releases MLIR context
- Cleans up device resources

### `void cml_seed(unsigned int seed)`

Set random seed for reproducible results.

```c
cml_seed(42);  // Reproducible random numbers
```

**Parameters:**
- `seed`: Random seed value

**Returns:** void

---

## Tensor API

### Tensor Creation

#### `Tensor* tensor_zeros(int* shape, int ndim, CleanupContext* ctx)`

Create a tensor filled with zeros.

```c
int shape[] = {10, 20};
Tensor* t = tensor_zeros(shape, 2, NULL);
```

**Parameters:**
- `shape`: Array of dimension sizes
- `ndim`: Number of dimensions
- `ctx`: Optional cleanup context (NULL for manual management)

**Returns:** Newly allocated tensor
**Memory:** Must be freed with `tensor_free()` unless using cleanup context

#### `Tensor* tensor_ones(int* shape, int ndim, CleanupContext* ctx)`

Create a tensor filled with ones.

```c
Tensor* t = tensor_ones((int[]){5, 5}, 2, NULL);
```

#### `Tensor* tensor_randn(int* shape, int ndim, CleanupContext* ctx)`

Create a tensor with random values from standard normal distribution.

```c
Tensor* t = tensor_randn((int[]){100, 100}, 2, NULL);  // N(0, 1)
```

#### `Tensor* tensor_rand(int* shape, int ndim, CleanupContext* ctx)`

Create a tensor with random values from uniform distribution [0, 1).

```c
Tensor* t = tensor_rand((int[]){50, 50}, 2, NULL);
```

#### `Tensor* tensor_full(int* shape, int ndim, float value, CleanupContext* ctx)`

Create a tensor filled with a specific value.

```c
Tensor* t = tensor_full((int[]){3, 3}, 2, 5.0f, NULL);  // 3x3 matrix of 5s
```

#### `Tensor* tensor_clone(Tensor* src, CleanupContext* ctx)`

Create a copy of a tensor.

```c
Tensor* copy = tensor_clone(original, NULL);
```

### Tensor Access & Manipulation

#### `float* tensor_data_ptr(Tensor* t)`

Get pointer to tensor data.

```c
Tensor* t = tensor_randn((int[]){10}, 1, NULL);
float* data = tensor_data_ptr(t);
printf("First element: %f\n", data[0]);
```

**Returns:** Pointer to underlying float array

#### `int tensor_size(Tensor* t)`

Get total number of elements in tensor.

```c
Tensor* t = tensor_randn((int[]){100, 50}, 2, NULL);
int size = tensor_size(t);  // Returns 5000
```

**Returns:** Number of elements (product of all dimensions)

#### `Tensor* tensor_reshape(Tensor* src, int* new_shape, int new_ndim)`

Reshape tensor (zero-copy if possible).

```c
Tensor* t = tensor_randn((int[]){10, 10}, 2, NULL);
Tensor* reshaped = tensor_reshape(t, (int[]){100}, 1);  // 2D → 1D
```

**Parameters:**
- `src`: Source tensor
- `new_shape`: New shape dimensions
- `new_ndim`: Number of new dimensions

**Returns:** Reshaped tensor (may share data with original)

#### `Tensor* tensor_transpose(Tensor* src, int axis1, int axis2)`

Transpose tensor along two axes.

```c
Tensor* t = tensor_randn((int[]){10, 20}, 2, NULL);
Tensor* transposed = tensor_transpose(t, 0, 1);  // Swap dimensions
```

**Parameters:**
- `src`: Source tensor
- `axis1`: First axis to swap
- `axis2`: Second axis to swap

**Returns:** Transposed tensor

#### `Tensor* tensor_slice(Tensor* src, int start, int end)`

Slice tensor along first dimension.

```c
Tensor* t = tensor_randn((int[]){100, 50}, 2, NULL);
Tensor* slice = tensor_slice(t, 10, 20);  // Elements 10-19
```

### Tensor Arithmetic

#### `Tensor* tensor_add(Tensor* a, Tensor* b)`

Element-wise addition.

```c
Tensor* result = tensor_add(a, b);
```

**Supports broadcasting:** Shapes automatically aligned

#### `Tensor* tensor_subtract(Tensor* a, Tensor* b)`

Element-wise subtraction.

```c
Tensor* result = tensor_subtract(a, b);
```

#### `Tensor* tensor_multiply(Tensor* a, Tensor* b)`

Element-wise multiplication.

```c
Tensor* result = tensor_multiply(a, b);
```

#### `Tensor* tensor_divide(Tensor* a, Tensor* b)`

Element-wise division.

```c
Tensor* result = tensor_divide(a, b);
```

#### `Tensor* tensor_power(Tensor* a, float exponent)`

Element-wise power operation.

```c
Tensor* squared = tensor_power(a, 2.0f);
```

#### `Tensor* tensor_matmul(Tensor* a, Tensor* b)`

Matrix multiplication.

```c
Tensor* a = tensor_randn((int[]){10, 20}, 2, NULL);
Tensor* b = tensor_randn((int[]){20, 15}, 2, NULL);
Tensor* result = tensor_matmul(a, b);  // 10x15
```

**Parameters:**
- `a`: First matrix (shape: [..., m, k])
- `b`: Second matrix (shape: [..., k, n])

**Returns:** Product (shape: [..., m, n])

### Tensor Activation Functions

#### `Tensor* tensor_relu(Tensor* t)`

ReLU activation: max(x, 0)

```c
Tensor* activated = tensor_relu(x);
```

#### `Tensor* tensor_sigmoid(Tensor* t)`

Sigmoid activation: 1 / (1 + exp(-x))

```c
Tensor* activated = tensor_sigmoid(x);
```

#### `Tensor* tensor_tanh(Tensor* t)`

Hyperbolic tangent activation.

```c
Tensor* activated = tensor_tanh(x);
```

#### `Tensor* tensor_softmax(Tensor* t, int dim)`

Softmax along specified dimension.

```c
Tensor* probs = tensor_softmax(logits, 1);  // dim 1 for batch axis
```

**Parameters:**
- `t`: Input tensor
- `dim`: Dimension to apply softmax

### Tensor Reduction

#### `Tensor* tensor_sum(Tensor* t)`

Sum all elements.

```c
Tensor* total = tensor_sum(t);  // Scalar result
```

#### `Tensor* tensor_mean(Tensor* t)`

Mean of all elements.

```c
Tensor* avg = tensor_mean(t);  // Scalar result
```

#### `Tensor* tensor_max(Tensor* t)`

Maximum element.

```c
Tensor* maximum = tensor_max(t);  // Scalar result
```

#### `Tensor* tensor_min(Tensor* t)`

Minimum element.

```c
Tensor* minimum = tensor_min(t);  // Scalar result
```

### Tensor Utility

#### `void tensor_free(Tensor* t)`

Free tensor memory.

```c
tensor_free(t);
t = NULL;
```

**Parameters:**
- `t`: Tensor to free

**Returns:** void

#### `void tensor_print(Tensor* t)`

Print tensor to stdout.

```c
tensor_print(t);
```

Useful for debugging.

---

## Autograd API

### `Tensor* cml_backward(Tensor* loss, Tensor* grad, bool clear_grads, bool retain_graph)`

Compute gradients via backpropagation.

```c
// Simple backward pass
cml_backward(loss, NULL, false, false);

// Keep gradients for multiple backward passes
cml_backward(loss1, NULL, false, true);
cml_backward(loss2, NULL, false, false);

// Clear gradients before backward
cml_backward(loss, NULL, true, false);
```

**Parameters:**
- `loss`: Loss tensor (usually scalar)
- `grad`: Gradient w.r.t. loss (NULL for scalar loss with grad=1)
- `clear_grads`: Clear existing gradients first
- `retain_graph`: Keep computation graph for multiple backwards

**Returns:** void

### `Tensor* cml_get_grad(Tensor* t)`

Get computed gradient for a tensor.

```c
Tensor* x = tensor_randn((int[]){10, 10}, 2, NULL);
// ... compute loss ...
cml_backward(loss, NULL, false, false);

Tensor* grad_x = cml_get_grad(x);
printf("Gradient shape: [%d, %d]\n", grad_x->shape[0], grad_x->shape[1]);
```

**Returns:** Gradient tensor or NULL if no gradient computed

### `void cml_enable_grad(Tensor* t)`

Enable gradient computation for a tensor.

```c
cml_enable_grad(x);
```

**Note:** Gradients are enabled by default for parameters

### `void cml_disable_grad(Tensor* t)`

Disable gradient computation for a tensor.

```c
cml_disable_grad(x);  // No gradient tracking
```

### Gradient Checkpointing

#### `void cml_checkpoint_save(Tensor* t)`

Save tensor for gradient checkpointing.

```c
// In forward pass, save intermediate results
cml_checkpoint_save(x1);
cml_checkpoint_save(x2);
```

#### `void cml_checkpoint_restore(Tensor* t)`

Restore tensor from checkpoint.

```c
// In backward pass, restore saved values
cml_checkpoint_restore(x1);
```

---

## Neural Network API

### Sequential Containers

#### `Sequential* cml_nn_sequential(void)`

Create an empty sequential container.

```c
Sequential* model = cml_nn_sequential();
```

**Returns:** New sequential model

#### `Sequential* cml_nn_sequential_add(Sequential* seq, Module* layer)`

Add a layer to sequential model.

```c
Sequential* model = cml_nn_sequential();
model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(784, 128, dtype, device, true));
model = cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
```

**Parameters:**
- `seq`: Sequential container
- `layer`: Layer module to add

**Returns:** Updated sequential model

### Linear Layer

#### `Linear* cml_nn_linear(int in_features, int out_features, DType dtype, DeviceType device, bool bias)`

Create a fully connected layer.

```c
Linear* fc = cml_nn_linear(784, 128, DTYPE_FLOAT32, DEVICE_CPU, true);
```

**Parameters:**
- `in_features`: Number of input features
- `out_features`: Number of output features
- `dtype`: Data type (DTYPE_FLOAT32, DTYPE_FLOAT64)
- `device`: Device type (DEVICE_CPU, DEVICE_CUDA, DEVICE_METAL, DEVICE_ROCM)
- `bias`: Whether to include bias term

**Returns:** Linear layer module

### Activation Layers

#### `ReLU* cml_nn_relu(bool inplace)`

Create ReLU activation.

```c
ReLU* relu = cml_nn_relu(false);
```

**Parameters:**
- `inplace`: Whether to modify input in-place

#### `Sigmoid* cml_nn_sigmoid(bool inplace)`

Create Sigmoid activation.

```c
Sigmoid* sigmoid = cml_nn_sigmoid(false);
```

#### `Tanh* cml_nn_tanh(bool inplace)`

Create Tanh activation.

```c
Tanh* tanh = cml_nn_tanh(false);
```

#### `Softmax* cml_nn_softmax(int dim, bool inplace)`

Create Softmax activation.

```c
Softmax* softmax = cml_nn_softmax(1, false);  // dim 1 for logits
```

#### `GELU* cml_nn_gelu(bool inplace)`

Create GELU activation.

```c
GELU* gelu = cml_nn_gelu(false);
```

### Regularization Layers

#### `Dropout* cml_nn_dropout(float dropout_rate, bool inplace)`

Create Dropout layer.

```c
Dropout* dropout = cml_nn_dropout(0.5f, false);  // 50% dropout
```

**Parameters:**
- `dropout_rate`: Probability of dropping (0.0 to 1.0)
- `inplace`: Whether to modify input in-place

### Normalization Layers

#### `BatchNorm2d* cml_nn_batchnorm2d(int num_features, DType dtype, DeviceType device, float momentum, float epsilon)`

Create 2D batch normalization.

```c
BatchNorm2d* bn = cml_nn_batchnorm2d(64, DTYPE_FLOAT32, DEVICE_CPU, 0.1f, 1e-5f);
```

#### `LayerNorm* cml_nn_layernorm(int* normalized_shape, int ndim, DType dtype, DeviceType device)`

Create layer normalization.

```c
LayerNorm* ln = cml_nn_layernorm((int[]){256}, 1, DTYPE_FLOAT32, DEVICE_CPU);
```

### Convolutional Layers

#### `Conv2d* cml_nn_conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, DType dtype, DeviceType device)`

Create 2D convolution layer.

```c
Conv2d* conv = cml_nn_conv2d(
    3,              // in_channels (RGB)
    64,             // out_channels
    3,              // kernel_size (3x3)
    1,              // stride
    1,              // padding
    DTYPE_FLOAT32,
    DEVICE_CPU
);
```

### Pooling Layers

#### `MaxPool2d* cml_nn_maxpool2d(int kernel_size, int stride, int padding)`

Create max pooling layer.

```c
MaxPool2d* pool = cml_nn_maxpool2d(2, 2, 0);  // 2x2 pooling, stride 2
```

#### `AvgPool2d* cml_nn_avgpool2d(int kernel_size, int stride, int padding)`

Create average pooling layer.

```c
AvgPool2d* pool = cml_nn_avgpool2d(2, 2, 0);
```

### Module Operations

#### `Tensor* cml_nn_module_forward(Module* module, Tensor* input)`

Forward pass through module.

```c
Tensor* output = cml_nn_module_forward((Module*)model, input);
```

**Parameters:**
- `module`: Module/layer to execute
- `input`: Input tensor

**Returns:** Output tensor

#### `void cml_nn_module_set_training(Module* module, bool training)`

Set training mode.

```c
cml_nn_module_set_training((Module*)model, true);   // Training
cml_nn_module_set_training((Module*)model, false);  // Inference
```

**Note:** Affects Dropout and BatchNorm behavior

#### `void module_free(Module* module)`

Free module memory.

```c
module_free((Module*)model);
```

---

## Optimizer API

### `Optimizer* cml_optim_adam_for_model(Module* module, float lr, float weight_decay, float beta1, float beta2, float epsilon)`

Create Adam optimizer for model parameters.

```c
Optimizer* opt = cml_optim_adam_for_model(
    (Module*)model,
    0.001f,   // learning rate
    0.0001f,  // weight decay (L2 regularization)
    0.9f,     // beta1 (momentum for first moment)
    0.999f,   // beta2 (momentum for second moment)
    1e-8f     // epsilon (numerical stability)
);
```

### `Optimizer* cml_optim_sgd_for_model(Module* module, float lr, float momentum, float weight_decay)`

Create SGD optimizer.

```c
Optimizer* opt = cml_optim_sgd_for_model(
    (Module*)model,
    0.01f,   // learning rate
    0.9f,    // momentum
    0.0f     // weight decay
);
```

### `Optimizer* cml_optim_rmsprop_for_model(Module* module, float lr, float alpha, float epsilon, float weight_decay)`

Create RMSprop optimizer.

```c
Optimizer* opt = cml_optim_rmsprop_for_model(
    (Module*)model,
    0.001f,  // learning rate
    0.99f,   // alpha (decay rate)
    1e-8f,   // epsilon
    0.0f     // weight decay
);
```

### `Optimizer* cml_optim_adagrad_for_model(Module* module, float lr, float epsilon)`

Create AdaGrad optimizer.

```c
Optimizer* opt = cml_optim_adagrad_for_model(
    (Module*)model,
    0.01f,   // learning rate
    1e-10f   // epsilon
);
```

### `void cml_optim_step(Optimizer* opt)`

Update parameters using computed gradients.

```c
cml_backward(loss, NULL, false, false);
cml_optim_step(optimizer);
```

### `void cml_optim_zero_grad(Optimizer* opt)`

Clear all parameter gradients.

```c
cml_optim_zero_grad(optimizer);
```

Should be called before computing new gradients.

### `void cml_optim_set_lr(Optimizer* opt, float lr)`

Update learning rate.

```c
cml_optim_set_lr(optimizer, 0.0001f);  // Reduce LR during training
```

### `void optimizer_free(Optimizer* opt)`

Free optimizer memory.

```c
optimizer_free(optimizer);
```

---

## Loss Functions

### `Tensor* cml_nn_mse_loss(Tensor* predictions, Tensor* targets)`

Mean squared error loss (regression).

```c
Tensor* loss = cml_nn_mse_loss(outputs, targets);
```

**Formula:** MSE = mean((predictions - targets)²)

### `Tensor* cml_nn_mae_loss(Tensor* predictions, Tensor* targets)`

Mean absolute error loss (regression).

```c
Tensor* loss = cml_nn_mae_loss(outputs, targets);
```

**Formula:** MAE = mean(|predictions - targets|)

### `Tensor* cml_nn_cross_entropy_loss(Tensor* logits, Tensor* labels)`

Cross entropy loss with softmax (classification).

```c
Tensor* loss = cml_nn_cross_entropy_loss(logits, labels);
```

**Note:**
- `logits`: Raw network outputs (shape: [batch, num_classes])
- `labels`: Class indices (shape: [batch])
- Applies softmax internally

### `Tensor* cml_nn_bce_loss(Tensor* predictions, Tensor* targets)`

Binary cross entropy loss.

```c
Tensor* loss = cml_nn_bce_loss(predictions, targets);
```

**Note:**
- Expects predictions in [0, 1] range (apply sigmoid first)
- Targets should be 0 or 1

### `Tensor* cml_nn_huber_loss(Tensor* predictions, Tensor* targets, float delta)`

Huber loss (robust to outliers).

```c
Tensor* loss = cml_nn_huber_loss(predictions, targets, 1.0f);
```

**Parameters:**
- `predictions`: Predicted values
- `targets`: Target values
- `delta`: Transition point between L2 and L1

### `Tensor* cml_nn_kl_divergence(Tensor* p_logits, Tensor* q_logits)`

KL divergence loss.

```c
Tensor* loss = cml_nn_kl_divergence(pred_logits, target_logits);
```

---

## Device & Memory

### Device Management

#### `DeviceType cml_get_default_device(void)`

Get default device.

```c
DeviceType device = cml_get_default_device();
```

**Returns:**
- `DEVICE_CPU`
- `DEVICE_CUDA` (if available)
- `DEVICE_METAL` (if macOS)
- `DEVICE_ROCM` (if available)

#### `void cml_set_device(DeviceType device)`

Set default device for new tensors.

```c
cml_set_device(DEVICE_CUDA);
```

**Parameters:**
- `DEVICE_CPU`: CPU execution
- `DEVICE_CUDA`: NVIDIA GPU
- `DEVICE_METAL`: Apple GPU (macOS)
- `DEVICE_ROCM`: AMD GPU

#### `bool cml_is_device_available(DeviceType device)`

Check if device is available.

```c
if (cml_is_device_available(DEVICE_CUDA)) {
    cml_set_device(DEVICE_CUDA);
}
```

**Returns:** true if device is available

#### `Tensor* tensor_to_device(Tensor* src, DeviceType device)`

Transfer tensor to device.

```c
Tensor* x_gpu = tensor_to_device(x_cpu, DEVICE_CUDA);
```

### Memory Management

#### `CleanupContext cleanup_create(void)`

Create automatic cleanup context.

```c
CleanupContext ctx = cleanup_create();
Tensor* x = tensor_randn((int[]){100, 100}, 2, &ctx);
Tensor* y = tensor_randn((int[]){100, 100}, 2, &ctx);

// All tensors freed automatically
cleanup_invoke(&ctx);
```

#### `void cleanup_invoke(CleanupContext* ctx)`

Clean up all resources in context.

```c
cleanup_invoke(&ctx);
```

### Data Types

```c
typedef enum {
    DTYPE_FLOAT32,   // 32-bit float
    DTYPE_FLOAT64,   // 64-bit double
    DTYPE_INT32,     // 32-bit integer
    DTYPE_INT64      // 64-bit integer
} DType;
```

**Default:** `DTYPE_FLOAT32`

#### `DType cml_get_default_dtype(void)`

Get default data type.

```c
DType dtype = cml_get_default_dtype();
```

#### `void cml_set_default_dtype(DType dtype)`

Set default data type.

```c
cml_set_default_dtype(DTYPE_FLOAT64);
```

---

## Utilities

### Datasets

#### `Dataset* dataset_xor(void)`

Get XOR dataset for testing.

```c
Dataset* dataset = dataset_xor();
Tensor* X = dataset->X;
Tensor* y = dataset->y;
```

**Returns:** XOR dataset with 4 samples

#### `void dataset_free(Dataset* dataset)`

Free dataset memory.

```c
dataset_free(dataset);
```

### Logging & Error Handling

#### `void cml_set_log_level(int level)`

Set logging verbosity.

```c
cml_set_log_level(4);  // DEBUG level
```

**Levels:**
- 0: SILENT
- 1: ERROR
- 2: WARN
- 3: INFO
- 4: DEBUG

#### `bool cml_has_error(void)`

Check if an error occurred.

```c
if (cml_has_error()) {
    printf("Error occurred\n");
}
```

#### `const char* cml_get_error(void)`

Get error message.

```c
const char* msg = cml_get_error();
printf("Error: %s\n", msg);
```

#### `void cml_clear_error(void)`

Clear error state.

```c
cml_clear_error();
```

#### `void cml_set_error_handler(void (*handler)(const char*))`

Set custom error handler.

```c
void my_error_handler(const char* msg) {
    fprintf(stderr, "Custom error: %s\n", msg);
}

cml_set_error_handler(my_error_handler);
```

### Metrics & Profiling

#### `TrainingMetrics* create_metrics(void)`

Create metrics tracker.

```c
TrainingMetrics* metrics = create_metrics();
```

#### `void metrics_add_loss(TrainingMetrics* metrics, float loss)`

Record loss value.

```c
metrics_add_loss(metrics, loss_value);
```

#### `void metrics_free(TrainingMetrics* metrics)`

Free metrics.

```c
metrics_free(metrics);
```

---

## Complete Example

```c
#include "cml.h"
#include <stdio.h>

int main(void) {
    // Initialization
    cml_init();
    cml_seed(42);

    // Setup
    DeviceType device = cml_get_default_device();
    DType dtype = cml_get_default_dtype();

    // Build model
    Sequential* model = cml_nn_sequential();
    model = cml_nn_sequential_add(model,
        (Module*)cml_nn_linear(10, 20, dtype, device, true));
    model = cml_nn_sequential_add(model,
        (Module*)cml_nn_relu(false));
    model = cml_nn_sequential_add(model,
        (Module*)cml_nn_linear(20, 1, dtype, device, true));

    // Create optimizer
    Optimizer* optimizer = cml_optim_adam_for_model(
        (Module*)model, 0.001f, 0.0f, 0.9f, 0.999f, 1e-8f);

    // Create data
    Tensor* X = tensor_randn((int[]){32, 10}, 2, NULL);
    Tensor* y = tensor_randn((int[]){32, 1}, 2, NULL);

    // Training loop
    cml_nn_module_set_training((Module*)model, true);
    for (int epoch = 0; epoch < 10; epoch++) {
        cml_optim_zero_grad(optimizer);

        Tensor* pred = cml_nn_module_forward((Module*)model, X);
        Tensor* loss = cml_nn_mse_loss(pred, y);

        cml_backward(loss, NULL, false, false);
        cml_optim_step(optimizer);

        float* loss_val = (float*)tensor_data_ptr(loss);
        printf("Epoch %d: Loss = %.4f\n", epoch, loss_val[0]);

        tensor_free(loss);
        tensor_free(pred);
    }

    // Cleanup
    tensor_free(X);
    tensor_free(y);
    optimizer_free(optimizer);
    module_free((Module*)model);
    cml_cleanup();

    return 0;
}
```

---

## See Also

- [Quick Start](QUICK_START.md) - Get started quickly
- [Architecture](ARCHITECTURE.md) - Internal design
- [Compilation](COMPILATION.md) - Build instructions
- [Running](RUNNING.md) - How to run programs
