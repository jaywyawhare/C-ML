# API Reference

Complete API documentation for C-ML library.

## Table of Contents

- [Initialization](#initialization)
- [Tensor Operations](#tensor-operations)
- [Neural Network Layers](#neural-network-layers)
- [Optimizers](#optimizers)
- [Training](#training)
- [UOps](#uops-micro-operations)
- [Context and Memory Management](#context-and-memory-management)
- [Error Handling](#error-handling)
- [Built-in Datasets](#built-in-datasets)
- [Device Types](#device-types)
- [Data Types](#data-types)

## Initialization

```c
#include "cml.h"

int main() {
    cml_init();
    cml_seed(42);
    // ...
    cml_cleanup();
    return 0;
}
```

### Configuration

```c
// Set defaults
cml_set_default_device(DEVICE_CUDA);
cml_set_default_dtype(DTYPE_FLOAT32);
DeviceType device = cml_get_default_device();
DType dtype = cml_get_default_dtype();

// Library info
int major, minor, patch;
const char* version;
cml_get_version(&major, &minor, &patch, &version);
const char* build_info = cml_get_build_info();

// Check initialization
bool initialized = cml_is_initialized();
int init_count = cml_get_init_count();
```

## Tensor Operations

### Creating Tensors

```c
// Using TensorConfig (recommended)
TensorConfig config = tensor_config_with_dtype_device(DTYPE_FLOAT32, DEVICE_CPU);
int shape[] = {2, 3};
Tensor* t = tensor_empty(shape, 2, &config);
Tensor* zeros = tensor_zeros(shape, 2, &config);
Tensor* ones = tensor_ones(shape, 2, &config);

// From data
float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
Tensor* from_data = tensor_from_data(data, shape, 2, &config);

// Convenience 2D functions
Tensor* t2d = cml_zeros_2d(10, 20);
Tensor* ones_2d = cml_ones_2d(10, 20);
Tensor* empty_2d = cml_empty_2d(10, 20);
Tensor* from_array = cml_tensor_2d(data, 10, 20);

// High-level API (cml_* prefix)
Tensor* t = cml_empty(shape, 2, &config);
Tensor* zeros = cml_zeros(shape, 2, &config);
Tensor* ones = cml_ones(shape, 2, &config);
```

### Elementwise Operations

```c
// Basic arithmetic
Tensor* sum = cml_add(a, b);      // or tensor_add(a, b)
Tensor* diff = cml_sub(a, b);     // or tensor_sub(a, b)
Tensor* prod = cml_mul(a, b);     // or tensor_mul(a, b)
Tensor* quot = cml_div(a, b);     // or tensor_div(a, b)
Tensor* power = cml_pow(a, b);    // or tensor_pow(a, b)

// Unary operations
Tensor* exp_result = cml_exp(a);      // or tensor_exp(a)
Tensor* log_result = cml_log(a);      // or tensor_log(a)
Tensor* sqrt_result = cml_sqrt(a);    // or tensor_sqrt(a)
Tensor* neg_result = tensor_neg(a);

// Trigonometric
Tensor* sin_result = cml_sin(a);      // or tensor_sin(a)
Tensor* cos_result = cml_cos(a);      // or tensor_cos(a)
Tensor* tan_result = cml_tan(a);      // or tensor_tan(a)
```

### Activation Functions

```c
Tensor* relu_out = cml_relu(x);        // or tensor_relu(x)
Tensor* sigmoid_out = cml_sigmoid(x);  // or tensor_sigmoid(x)
Tensor* tanh_out = cml_tanh(x);       // or tensor_tanh(x)
Tensor* softmax_out = cml_softmax(x, dim);  // or tensor_softmax(x, dim)
```

### Matrix Operations

```c
Tensor* matmul_result = cml_matmul(a, b);           // or tensor_matmul(a, b)
Tensor* transpose_result = cml_transpose(a, 0, 1);  // or tensor_transpose(a, 0, 1)
```

### Reduction Operations

```c
Tensor* sum_all = cml_sum(t, -1, false);    // Sum all elements
Tensor* mean_dim = cml_mean(t, 0, true);    // Mean along dimension 0
Tensor* max_dim = cml_max(t, 0, true);      // Max along dimension 0
Tensor* min_dim = cml_min(t, 0, true);      // Min along dimension 0
```

### View Operations

```c
Tensor* reshaped = cml_reshape(t, new_shape, new_ndim);  // or tensor_reshape(t, new_shape, new_ndim)
Tensor* permuted = tensor_permute(t, perm, ndim);
Tensor* sliced = tensor_slice(t, start, end, step, ndim);
Tensor* cloned = cml_clone(t);              // Deep copy
Tensor* detached = cml_detach(t);           // Detach from graph
Tensor* concat_result = cml_concat(tensors, num_tensors, dim);
Tensor* stack_result = cml_stack(tensors, num_tensors, dim);
```

## Neural Network Layers

### Sequential Model

```c
// Create sequential model
Sequential* model = cml_nn_sequential();  // or nn_sequential()

// Add layers (fluent API)
model = sequential_add_chain(model,
    (Module*)nn_linear(784, 128, dtype, device, true),
    (Module*)nn_relu(false),
    (Module*)nn_linear(128, 64, dtype, device, true),
    NULL
);

// Or using sequential_add
sequential_add(model, (Module*)nn_linear(784, 128, dtype, device, true));
sequential_add(model, (Module*)nn_relu(false));

// Forward pass
Tensor* output = cml_nn_module_forward((Module*)model, input);  // or module_forward((Module*)model, input)

// Model summary
cml_summary((Module*)model);

// Training mode
cml_nn_module_set_training((Module*)model, true);   // or module_set_training((Module*)model, true)
cml_nn_module_eval((Module*)model);                // or module_set_training((Module*)model, false)
bool is_training = cml_nn_module_is_training((Module*)model);
```

### Available Layers

```c
// Linear (Fully Connected)
Linear* linear = cml_nn_linear(in_features, out_features, dtype, device, bias);
// or nn_linear(in_features, out_features, dtype, device, bias)

// Convolutional
Conv2d* conv = cml_nn_conv2d(in_channels, out_channels, kernel_size,
                             stride, padding, dilation, bias, dtype, device);
// or nn_conv2d(in_channels, out_channels, kernel_size, stride, padding,
//              dilation, groups, bias, dtype, device)

// Activation Layers
ReLU* relu = cml_nn_relu(inplace);              // or nn_relu(inplace)
Sigmoid* sigmoid = cml_nn_sigmoid();            // or nn_sigmoid()
Tanh* tanh = cml_nn_tanh();                     // or nn_tanh()
LeakyReLU* leaky = cml_nn_leaky_relu(0.01f, false);  // or nn_leaky_relu(negative_slope, inplace)
GELU* gelu = nn_gelu(inplace);

// Normalization
BatchNorm2d* bn = cml_nn_batchnorm2d(num_features, eps, momentum,
                                     affine, track_running_stats, dtype, device);
// or nn_batchnorm2d(num_features, eps, momentum, affine, track_running_stats, dtype, device)

LayerNorm* ln = cml_nn_layernorm(normalized_shape, eps, affine, dtype, device);
// or nn_layernorm(normalized_shape, eps, affine, dtype, device)

// Pooling
MaxPool2d* maxpool = cml_nn_maxpool2d(kernel_size, stride, padding, dilation, ceil_mode);
// or nn_maxpool2d(kernel_size, stride, padding, dilation, ceil_mode)

AvgPool2d* avgpool = cml_nn_avgpool2d(kernel_size, stride, padding, ceil_mode, count_include_pad);
// or nn_avgpool2d(kernel_size, stride, padding, ceil_mode, count_include_pad)

// Dropout
Dropout* dropout = cml_nn_dropout(probability, inplace);  // or nn_dropout(probability, inplace)
```

## Optimizers

### Automatic Parameter Collection (Recommended)

```c
// Adam optimizer
Optimizer* optimizer = cml_optim_adam_for_model((Module*)model,
    lr, weight_decay, beta1, beta2, eps);
// or optim_adam_for_model((Module*)model, lr, weight_decay, beta1, beta2, eps)

// SGD optimizer
Optimizer* sgd = cml_optim_sgd_for_model((Module*)model, lr, momentum, weight_decay);
// or optim_sgd_for_model((Module*)model, lr, momentum, weight_decay)
```

### Manual Parameter Collection

```c
// Collect parameters
Parameter** params;
int num_params;
module_collect_parameters((Module*)model, &params, &num_params, true);

// Create optimizer
Optimizer* optimizer = cml_optim_adam(params, num_params, lr, weight_decay, beta1, beta2, eps);
// or optim_adam(params, num_params, lr, weight_decay, beta1, beta2, eps)

Optimizer* sgd = cml_optim_sgd(params, num_params, lr, momentum, weight_decay);
// or optim_sgd(params, num_params, lr, momentum, weight_decay)

// Additional optimizers
Optimizer* rmsprop = cml_optim_rmsprop(params, num_params, lr, weight_decay, alpha, eps);
Optimizer* adagrad = cml_optim_adagrad(params, num_params, lr, weight_decay, eps);
```

### Optimizer Usage

```c
// Zero gradients
cml_optim_zero_grad(optimizer);  // or optimizer_zero_grad(optimizer)

// Compute loss and backward
Tensor* loss = cml_nn_mse_loss(output, target);
cml_backward(loss, NULL, false, false);  // or tensor_backward(loss, NULL, false, false)

// Update parameters
cml_optim_step(optimizer);  // or optimizer_step(optimizer)

// Cleanup
optimizer_free(optimizer);
```

## Training

### Loss Functions

```c
Tensor* loss = cml_nn_mse_loss(outputs, targets);        // Mean Squared Error
Tensor* loss = cml_nn_mae_loss(outputs, targets);        // Mean Absolute Error
Tensor* loss = cml_nn_bce_loss(outputs, targets);        // Binary Cross Entropy
Tensor* loss = cml_nn_cross_entropy_loss(outputs, targets);  // Cross Entropy
Tensor* loss = cml_nn_huber_loss(outputs, targets, delta);    // Huber Loss
Tensor* loss = cml_nn_kl_div_loss(outputs, targets);     // KL Divergence
```

### Manual Training Loop

```c
// Set training mode
cml_nn_module_set_training((Module*)model, true);

for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Zero gradients
    cml_optim_zero_grad(optimizer);

    // Forward pass
    Tensor* outputs = cml_nn_module_forward((Module*)model, inputs);

    // Compute loss
    Tensor* loss = cml_nn_mse_loss(outputs, targets);

    // Backward pass
    cml_backward(loss, NULL, false, false);

    // Update parameters
    cml_optim_step(optimizer);

    // Cleanup
    tensor_free(loss);
    tensor_free(outputs);
}

// Set evaluation mode
cml_nn_module_eval((Module*)model);
```

### Training with DataLoader

```c
DataLoader* loader = dataloader_create(dataset, batch_size, shuffle);

for (int epoch = 0; epoch < num_epochs; epoch++) {
    dataloader_reset(loader);
    while (dataloader_has_next(loader)) {
        Batch* batch = dataloader_next_batch(loader);

        cml_optim_zero_grad(optimizer);
        Tensor* outputs = cml_nn_module_forward((Module*)model, batch->X);
        Tensor* loss = cml_nn_mse_loss(outputs, batch->y);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(optimizer);

        batch_free(batch);
        tensor_free(loss);
        tensor_free(outputs);
    }
}
```

## UOps (Micro-Operations)

```c
// Elementwise
Tensor* result = uop_add(a, b);
Tensor* result = uop_mul(a, b);
Tensor* result = uop_max(a, b);

// Reductions
ReduceParams params = {.dims = &dim, .num_dims = 1, .keepdim = false};
Tensor* reduced = uop_mean(tensor, &params);

// Movement
ReshapeParams reshape = {.new_shape = new_shape, .new_ndim = 2};
Tensor* reshaped = uop_reshape(tensor, &reshape);

// Special
Tensor* matmul_result = uop_matmul(a, b);
Conv2DParams conv_params = {...};
Tensor* conv_result = uop_conv2d(input, weight, bias, &conv_params);
```

## Context and Memory Management

```c
CMLContextParams params = {
    .mem_size = 1024 * 1024 * 100,
    .mem_buffer = NULL,
    .no_alloc = false
};
CMLContext_t ctx = cml_context_new(params);
Tensor* t1 = cml_context_alloc_tensor(ctx, shape1, ndim1, DTYPE_FLOAT32, DEVICE_CPU);
cml_context_free(ctx);
```

## Error Handling

```c
Tensor* t = tensor_empty(shape, ndim, NULL);
if (!t && CML_HAS_ERRORS()) {
    const char* msg = CML_LAST_ERROR();
    int code = CML_LAST_ERROR_CODE();
    error_stack_print_all();
}

// Checked constructors
Tensor* t = CML_CHECK(tensor_empty(shape, ndim, NULL), "Failed to create tensor");
Tensor* t = CML_CHECK_AUTO(tensor_empty(shape, ndim, NULL));

// Global error handler
void my_error_handler(int code, const char* msg, void* ctx) {
    fprintf(stderr, "Error %d: %s\n", code, msg);
}
cml_set_error_handler(my_error_handler);
```

## Built-in Datasets

```c
// Create datasets
Dataset* xor_data = dataset_xor();
Dataset* random_data = dataset_random_classification(num_samples, input_size, output_size);
Dataset* dataset = dataset_from_arrays(X, y, num_samples, input_size, output_size);

// Split datasets
Dataset* train, *val, *test;
dataset_split_three(full_dataset, train_ratio, val_ratio, &train, &val, &test);

Dataset* train, *val;
dataset_split(full_dataset, train_ratio, &train, &val);

// Normalize datasets
dataset_normalize(dataset, "zscore");   // Z-score normalization
dataset_normalize(dataset, "minmax");    // Min-max normalization

// Custom normalization
float mean[] = {0.5f, 0.5f, 0.5f};
float std[] = {0.25f, 0.25f, 0.25f};
transform_normalize(dataset, mean, std, 3);

// Copy dataset
Dataset* copy = dataset_copy(dataset);
```

## Device Types

```c
// Available devices
DEVICE_CPU      // CPU (always available)
DEVICE_CUDA     // NVIDIA GPU
DEVICE_METAL    // Apple GPU (macOS/iOS)
DEVICE_ROCM     // AMD GPU
DEVICE_AUTO     // Auto-detect best available

// Device utilities
DeviceType device = cml_get_default_device();
cml_set_default_device(DEVICE_CUDA);
bool cuda_available = device_cuda_available();
bool metal_available = device_metal_available();
bool rocm_available = device_rocm_available();
```

## Data Types

```c
// Available dtypes
DTYPE_FLOAT32   // 32-bit float (default)
DTYPE_FLOAT64   // 64-bit float
DTYPE_INT32     // 32-bit integer
DTYPE_INT64     // 64-bit integer
DTYPE_BOOL      // Boolean

// Dtype utilities
DType dtype = cml_get_default_dtype();
cml_set_default_dtype(DTYPE_FLOAT32);
size_t size = cml_dtype_size(DTYPE_FLOAT32);
DType promoted = cml_promote_dtype(DTYPE_FLOAT32, DTYPE_INT32);
```

## Library Information

```c
// Version information
int major, minor, patch;
const char* version_string;
cml_get_version(&major, &minor, &patch, &version_string);

// Build information
const char* build_info = cml_get_build_info();
printf("Build info: %s\n", build_info);

// Initialization status
bool initialized = cml_is_initialized();
int init_count = cml_get_init_count();
```

## See Also

- [Autograd System](autograd.md) - Automatic differentiation
- [Neural Network Layers](nn_layers.md) - Layer reference
- [Training Guide](training.md) - Training API
- [Graph Mode](graph_mode.md) - Lazy execution
