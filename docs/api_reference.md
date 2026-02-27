# API Reference

Complete API reference for the C-ML library.

## Table of Contents

- [Initialization](#initialization)
- [Tensor Creation](#tensor-creation)
- [Tensor Operations](#tensor-operations)
- [Neural Network Layers](#neural-network-layers)
- [Optimizers](#optimizers)
- [LR Schedulers](#lr-schedulers)
- [Loss Functions](#loss-functions)
- [Autograd](#autograd)
- [Dataset Hub](#dataset-hub)
- [Model Zoo](#model-zoo)
- [Model I/O](#model-io)
- [Device Management](#device-management)
- [Memory Management](#memory-management)
- [Kernel Cache](#kernel-cache)
- [Error Handling](#error-handling)

______________________________________________________________________

## Initialization

```c
#include "cml.h"

int  cml_init(void);               // Initialize library (call first)
int  cml_cleanup(void);            // Cleanup (auto-called via atexit)
int  cml_force_cleanup(void);      // Force cleanup ignoring refcount
bool cml_is_initialized(void);     // Check if library is initialized

void cml_get_version(int* major, int* minor, int* patch,
                     const char** version_string);
const char* cml_get_build_info(void);

void cml_seed(int seed);           // Set global random seed
```

### Typical Usage

```c
int main(void) {
    cml_init();
    cml_seed(42);

    // ... your code ...

    cml_cleanup();  // Also registered with atexit
    return 0;
}
```

`cml_init()` must be called before any other library function. `cml_cleanup()` is automatically registered via `atexit`, so explicit calls are optional but recommended for clarity. If the library was initialized multiple times (nested init), `cml_cleanup()` decrements a refcount; use `cml_force_cleanup()` to tear down regardless.

______________________________________________________________________

## Tensor Creation

### Generic Constructors

All generic constructors accept an optional `TensorConfig*` for dtype and device settings. Pass `NULL` for defaults (float32, CPU).

```c
Tensor* cml_empty(int* shape, int ndim, const TensorConfig* config);
Tensor* cml_zeros(int* shape, int ndim, const TensorConfig* config);
Tensor* cml_ones(int* shape, int ndim, const TensorConfig* config);
Tensor* cml_full(int* shape, int ndim, const TensorConfig* config, float value);
Tensor* cml_tensor(void* data, int* shape, int ndim, const TensorConfig* config);
```

| Function     | Description                                  |
| ------------ | -------------------------------------------- |
| `cml_empty`  | Allocate uninitialized tensor                |
| `cml_zeros`  | Allocate tensor filled with zeros            |
| `cml_ones`   | Allocate tensor filled with ones             |
| `cml_full`   | Allocate tensor filled with a constant value |
| `cml_tensor` | Create tensor from existing data buffer      |

### 2D Shortcuts

Convenience functions that default to float32 on CPU.

```c
Tensor* cml_zeros_2d(int rows, int cols);
Tensor* cml_ones_2d(int rows, int cols);
Tensor* cml_empty_2d(int rows, int cols);
Tensor* cml_tensor_2d(const float* data, int rows, int cols);
```

### 1D Shortcuts

```c
Tensor* cml_zeros_1d(int size);
Tensor* cml_ones_1d(int size);
Tensor* cml_empty_1d(int size);
Tensor* cml_tensor_1d(const float* data, int size);
```

### Random Initialization

```c
Tensor* tensor_randn(int* shape, int ndim, const TensorConfig* config);
```

Fills tensor with samples from a standard normal distribution (mean=0, std=1).

### Example

```c
int shape[] = {2, 3};
TensorConfig cfg = tensor_config_with_dtype_device(DTYPE_FLOAT32, DEVICE_CPU);

Tensor* a = cml_zeros(shape, 2, &cfg);
Tensor* b = cml_ones(shape, 2, NULL);   // NULL = default config
Tensor* c = cml_full(shape, 2, NULL, 3.14f);

// 2D shortcut
Tensor* d = cml_zeros_2d(4, 5);

// From data
float data[] = {1, 2, 3, 4, 5, 6};
Tensor* e = cml_tensor_2d(data, 2, 3);
```

______________________________________________________________________

## Tensor Operations

**Important:** Tensor operations return lazy/IR-based tensors. The computation is not executed immediately. Call `tensor_data_ptr()` to materialize the result and access the underlying data.

### Element-wise Arithmetic

```c
Tensor* cml_add(Tensor* a, Tensor* b);   // a + b
Tensor* cml_sub(Tensor* a, Tensor* b);   // a - b
Tensor* cml_mul(Tensor* a, Tensor* b);   // a * b
Tensor* cml_div(Tensor* a, Tensor* b);   // a / b
Tensor* cml_pow(Tensor* a, Tensor* b);   // a ^ b
```

### Unary Math Functions

```c
Tensor* cml_exp(Tensor* a);              // e^a
Tensor* cml_log(Tensor* a);              // ln(a)
Tensor* cml_sqrt(Tensor* a);             // sqrt(a)
Tensor* cml_sin(Tensor* a);
Tensor* cml_cos(Tensor* a);
Tensor* cml_tan(Tensor* a);
```

### Activation Functions

```c
Tensor* cml_relu(Tensor* a);             // max(0, a)
Tensor* cml_sigmoid(Tensor* a);          // 1 / (1 + e^(-a))
Tensor* cml_tanh(Tensor* a);             // tanh(a)
Tensor* cml_softmax(Tensor* a, int dim); // softmax along dim
```

### Reduction Operations

```c
Tensor* cml_sum(Tensor* a, int dim, bool keepdim);
Tensor* cml_mean(Tensor* a, int dim, bool keepdim);
Tensor* cml_max(Tensor* a, int dim, bool keepdim);
Tensor* cml_min(Tensor* a, int dim, bool keepdim);
```

Pass `dim = -1` to reduce over all dimensions.

### Linear Algebra

```c
Tensor* cml_matmul(Tensor* a, Tensor* b);           // Matrix multiply
Tensor* cml_transpose(Tensor* a, int dim1, int dim2); // Swap two dims
```

### Shape Manipulation

```c
Tensor* cml_reshape(Tensor* a, int* new_shape, int new_ndim);
Tensor* cml_clone(Tensor* a);           // Deep copy
Tensor* cml_detach(Tensor* a);          // Detach from computation graph
Tensor* cml_concat(Tensor** tensors, int num_tensors, int dim);
Tensor* cml_stack(Tensor** tensors, int num_tensors, int dim);
```

### Data Access and Cleanup

```c
float* tensor_data_ptr(Tensor* t);  // Materialize lazy tensor, return data pointer
void   tensor_free(Tensor* t);      // Free tensor and associated resources
```

### Example

```c
Tensor* a = cml_ones_2d(3, 4);
Tensor* b = cml_ones_2d(3, 4);

Tensor* c = cml_add(a, b);           // Lazy -- not yet computed
float* data = tensor_data_ptr(c);    // Materializes the computation
printf("c[0] = %f\n", data[0]);      // 2.0

Tensor* d = cml_matmul(a, cml_transpose(b, 0, 1)); // [3,4] x [4,3] = [3,3]

tensor_free(d);
tensor_free(c);
tensor_free(b);
tensor_free(a);
```

______________________________________________________________________

## Neural Network Layers

For detailed layer documentation, see [nn_layers.md](nn_layers.md).

### Sequential Container

```c
Sequential* cml_nn_sequential(void);
void cml_nn_sequential_add(Sequential* seq, Module* layer);
```

### Layer Constructors

**Dense / Linear**

```c
Linear* cml_nn_linear(int in_features, int out_features,
                       DType dtype, DeviceType device, bool bias);
```

**Convolutional**

```c
Conv1d* cml_nn_conv1d(int in_ch, int out_ch, int kernel_size,
                       int stride, int padding, int dilation,
                       bool bias, DType dtype, DeviceType device);
Conv2d* cml_nn_conv2d(int in_ch, int out_ch, int kernel_size,
                       int stride, int padding, int dilation,
                       bool bias, DType dtype, DeviceType device);
Conv3d* cml_nn_conv3d(int in_ch, int out_ch, int kernel_size,
                       int stride, int padding, int dilation,
                       bool bias, DType dtype, DeviceType device);
```

**Recurrent**

```c
RNNCell*  cml_nn_rnn_cell(int input_size, int hidden_size, ...);
LSTMCell* cml_nn_lstm_cell(int input_size, int hidden_size, ...);
GRUCell*  cml_nn_gru_cell(int input_size, int hidden_size, ...);
```

**Transformer**

```c
MultiHeadAttention*       cml_nn_multihead_attention(int embed_dim, int num_heads, ...);
TransformerEncoderLayer*  cml_nn_transformer_encoder_layer(int d_model, int nhead, ...);
```

**Embedding**

```c
Embedding* cml_nn_embedding(int num_embeddings, int embedding_dim, ...);
```

**Normalization**

```c
BatchNorm2d* cml_nn_batchnorm2d(int num_features, float eps, float momentum,
                                 bool affine, bool track_running_stats,
                                 DType dtype, DeviceType device);
LayerNorm*   cml_nn_layernorm(int normalized_shape, float eps, bool affine,
                               DType dtype, DeviceType device);
GroupNorm*   cml_nn_groupnorm(int num_groups, int num_channels, float eps,
                               bool affine, DType dtype, DeviceType device);
```

**Pooling**

```c
MaxPool2d* cml_nn_maxpool2d(int kernel_size, int stride, int padding,
                             int dilation, bool ceil_mode);
AvgPool2d* cml_nn_avgpool2d(int kernel_size, int stride, int padding,
                             bool ceil_mode, bool count_include_pad);
```

**Activation Layers**

```c
ReLU*      cml_nn_relu(bool inplace);
Sigmoid*   cml_nn_sigmoid(void);
Tanh*      cml_nn_tanh(void);
LeakyReLU* cml_nn_leaky_relu(float negative_slope, bool inplace);
Dropout*   cml_nn_dropout(float p, bool inplace);
```

**Container Layers**

```c
ModuleList* cml_nn_module_list(void);
ModuleDict* cml_nn_module_dict(void);
```

### Module API

Every layer inherits from `Module`. The following functions work on any module type:

```c
Tensor* cml_nn_module_forward(Module* module, Tensor* input);
void    cml_nn_module_set_training(Module* module, bool training);
void    cml_nn_module_eval(Module* module);    // Shorthand: set training=false
void    cml_nn_module_train(Module* module);   // Shorthand: set training=true
void    cml_summary(Module* module);           // Print layer summary
void    module_free(Module* module);           // Free module and parameters
```

### Example

```c
Sequential* model = cml_nn_sequential();
cml_nn_sequential_add(model, (Module*)cml_nn_linear(784, 128, DTYPE_FLOAT32, DEVICE_CPU, true));
cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
cml_nn_sequential_add(model, (Module*)cml_nn_dropout(0.2f, false));
cml_nn_sequential_add(model, (Module*)cml_nn_linear(128, 10, DTYPE_FLOAT32, DEVICE_CPU, true));

cml_summary((Module*)model);

Tensor* output = cml_nn_module_forward((Module*)model, input);
```

______________________________________________________________________

## Optimizers

### Direct Constructors

These take explicit parameter arrays collected from modules.

```c
Optimizer* optim_sgd(Parameter** params, int n, float lr,
                     float momentum, float weight_decay);

Optimizer* optim_adam(Parameter** params, int n, float lr,
                      float wd, float beta1, float beta2, float eps);

Optimizer* optim_adamw(Parameter** params, int n, float lr,
                        float wd, float beta1, float beta2, float eps);

Optimizer* optim_rmsprop(Parameter** params, int n, float lr,
                          float wd, float alpha, float eps);

Optimizer* optim_adagrad(Parameter** params, int n, float lr,
                          float wd, float eps);

Optimizer* optim_adadelta(Parameter** params, int n, float rho,
                           float wd, float eps);
```

### Model-based Constructors (Recommended)

These automatically collect parameters from a module:

```c
Optimizer* cml_optim_adam_for_model(Module* model, float lr, float wd,
                                    float beta1, float beta2, float eps);

Optimizer* cml_optim_sgd_for_model(Module* model, float lr,
                                    float momentum, float wd);
```

### Optimizer Operations

```c
void cml_optim_zero_grad(Optimizer* opt);    // Zero all parameter gradients
void cml_optim_step(Optimizer* opt);         // Apply one optimization step
void optimizer_set_lr(Optimizer* opt, float lr); // Manually set learning rate
void optimizer_free(Optimizer* opt);         // Free optimizer state
```

### Example

```c
Optimizer* opt = cml_optim_adam_for_model((Module*)model,
    /*lr=*/1e-3f, /*wd=*/1e-4f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*eps=*/1e-8f);

for (int epoch = 0; epoch < 100; epoch++) {
    cml_optim_zero_grad(opt);

    Tensor* out  = cml_nn_module_forward((Module*)model, x);
    Tensor* loss = cml_nn_mse_loss(out, y);

    cml_backward(loss, NULL, false, false);
    cml_optim_step(opt);

    tensor_free(loss);
    tensor_free(out);
}

optimizer_free(opt);
```

______________________________________________________________________

## LR Schedulers

Learning rate schedulers adjust the optimizer learning rate over time.

### Constructors

```c
LRScheduler* lr_scheduler_step(Optimizer* opt, int step_size, float gamma);
LRScheduler* lr_scheduler_exponential(Optimizer* opt, float gamma);
LRScheduler* lr_scheduler_cosine_annealing(Optimizer* opt, int T_max, float eta_min);
LRScheduler* lr_scheduler_reduce_on_plateau(Optimizer* opt, float factor,
                                             int patience, float threshold,
                                             bool mode_min);
LRScheduler* lr_scheduler_multi_step(Optimizer* opt, int* milestones,
                                      int num_milestones, float gamma);
```

| Scheduler           | Description                                            |
| ------------------- | ------------------------------------------------------ |
| `step`              | Multiply LR by `gamma` every `step_size` epochs        |
| `exponential`       | Multiply LR by `gamma` every epoch                     |
| `cosine_annealing`  | Cosine decay from initial LR to `eta_min` over `T_max` |
| `reduce_on_plateau` | Reduce LR by `factor` if metric stalls for `patience`  |
| `multi_step`        | Multiply LR by `gamma` at each milestone epoch         |

### Operations

```c
void  lr_scheduler_step_epoch(LRScheduler* scheduler);            // Call after each epoch
void  lr_scheduler_step_metric(LRScheduler* scheduler, float metric); // For ReduceOnPlateau
float lr_scheduler_get_lr(LRScheduler* scheduler);                // Get current LR
void  lr_scheduler_free(LRScheduler* scheduler);
```

### Example

```c
LRScheduler* sched = lr_scheduler_cosine_annealing(opt, /*T_max=*/50, /*eta_min=*/1e-6f);

for (int epoch = 0; epoch < 50; epoch++) {
    // ... training loop ...
    lr_scheduler_step_epoch(sched);
    printf("LR: %f\n", lr_scheduler_get_lr(sched));
}

lr_scheduler_free(sched);
```

______________________________________________________________________

## Loss Functions

### Standard Losses

```c
Tensor* cml_nn_mse_loss(Tensor* input, Tensor* target);           // Mean Squared Error
Tensor* cml_nn_mae_loss(Tensor* input, Tensor* target);           // Mean Absolute Error
Tensor* cml_nn_bce_loss(Tensor* input, Tensor* target);           // Binary Cross-Entropy
Tensor* cml_nn_cross_entropy_loss(Tensor* input, Tensor* target); // Cross-Entropy
Tensor* cml_nn_huber_loss(Tensor* input, Tensor* target, float delta); // Huber Loss
Tensor* cml_nn_kl_div_loss(Tensor* input, Tensor* target);        // KL Divergence
```

### Additional Losses (from loss_functions.h)

```c
Tensor* tensor_hinge_loss(Tensor* input, Tensor* target);
Tensor* tensor_focal_loss(Tensor* input, Tensor* target, float alpha, float gamma);
Tensor* tensor_smooth_l1_loss(Tensor* input, Tensor* target, float beta);
```

All loss functions return a scalar tensor. Call `cml_backward()` on the result to compute gradients.

______________________________________________________________________

## Autograd

C-ML provides automatic differentiation through a tape-based autograd engine.

```c
void cml_backward(Tensor* tensor, Tensor* gradient,
                  bool retain_graph, bool create_graph);
void cml_zero_grad(Tensor* tensor);    // Zero gradient for a single tensor

void cml_no_grad(void);               // Disable gradient tracking globally
void cml_enable_grad(void);           // Re-enable gradient tracking
bool cml_is_grad_enabled(void);       // Check if grad is enabled

bool cml_requires_grad(Tensor* t);
void cml_set_requires_grad(Tensor* t, bool requires_grad);
bool cml_is_leaf(Tensor* t);          // True if tensor was not produced by an op
```

### Parameters

- **`gradient`**: External gradient to seed backpropagation. Pass `NULL` for scalar tensors (defaults to 1.0).
- **`retain_graph`**: If `true`, the computation graph is preserved after backward (needed for multiple backward passes).
- **`create_graph`**: If `true`, gradients of gradients can be computed (higher-order derivatives).

### Example

```c
Tensor* x = cml_ones_2d(3, 3);
cml_set_requires_grad(x, true);

Tensor* y = cml_mul(x, x);       // y = x^2
Tensor* z = cml_mean(y, -1, false); // z = mean(x^2)

cml_backward(z, NULL, false, false);
// x->grad now contains dz/dx = 2x/9

tensor_free(z);
tensor_free(y);
tensor_free(x);
```

______________________________________________________________________

## Dataset Hub

Load common datasets with a single function call. Datasets are downloaded and cached automatically.

### Loading Datasets

```c
Dataset* cml_dataset_load(const char* name);
```

**Supported datasets:**

| Name              | Description                   | Samples | Features | Classes |
| ----------------- | ----------------------------- | ------- | -------- | ------- |
| `"iris"`          | Iris flower classification    | 150     | 4        | 3       |
| `"wine"`          | Wine recognition              | 178     | 13       | 3       |
| `"breast_cancer"` | Breast cancer diagnosis       | 569     | 30       | 2       |
| `"boston"`        | Boston housing regression     | 506     | 13       | --      |
| `"mnist"`         | Handwritten digits            | 70000   | 784      | 10      |
| `"fashion_mnist"` | Fashion article images        | 70000   | 784      | 10      |
| `"cifar10"`       | Tiny color images             | 60000   | 3072     | 10      |
| `"airline"`       | Airline passenger forecasting | varies  | varies   | --      |
| `"digits"`        | Smaller handwritten digits    | 1797    | 64       | 10      |

### From CSV

```c
Dataset* cml_dataset_from_csv(const char* path, int target_col);
```

`target_col` specifies which column is the label. Use `-1` for the last column.

### Dataset Operations

```c
void dataset_normalize(Dataset* ds, const char* method);  // "minmax" or "zscore"
void dataset_split(Dataset* ds, float ratio, Dataset** train, Dataset** test);
void dataset_free(Dataset* ds);
```

### Example

```c
Dataset* ds = cml_dataset_load("iris");
dataset_normalize(ds, "zscore");

Dataset* train;
Dataset* test;
dataset_split(ds, 0.8f, &train, &test);

// Use train->X, train->y, train->num_samples, etc.

dataset_free(train);
dataset_free(test);
dataset_free(ds);
```

______________________________________________________________________

## Model Zoo

Pre-configured model architectures for common tasks.

### Configuration

```c
CMLZooConfig cml_zoo_default_config(void);
```

Returns a default configuration struct that can be customized before passing to a model constructor.

### Generic Constructor

```c
Module* cml_zoo_create(CMLZooModel model, const CMLZooConfig* config);
```

### Named Constructors

```c
Module* cml_zoo_mlp_mnist(const CMLZooConfig* config);
Module* cml_zoo_resnet18(const CMLZooConfig* config);
Module* cml_zoo_resnet34(const CMLZooConfig* config);
Module* cml_zoo_resnet50(const CMLZooConfig* config);
Module* cml_zoo_vgg11(const CMLZooConfig* config);
Module* cml_zoo_vgg16(const CMLZooConfig* config);
Module* cml_zoo_gpt2_small(const CMLZooConfig* config);
Module* cml_zoo_bert_tiny(const CMLZooConfig* config);
```

### Example

```c
CMLZooConfig cfg = cml_zoo_default_config();
cfg.num_classes = 10;

Module* model = cml_zoo_resnet18(&cfg);
cml_summary(model);

Tensor* out = cml_nn_module_forward(model, input);
module_free(model);
```

______________________________________________________________________

## Model I/O

Save and restore model weights, optimizer state, and training progress.

### Save / Load Weights

```c
int model_save(Module* model, const char* filepath);
int model_load(Module* model, const char* filepath);
```

Returns `0` on success, non-zero on error.

### Checkpointing

```c
int model_save_checkpoint(Module* model, Optimizer* opt,
                          int epoch, float loss, const char* filepath);

int model_load_checkpoint(Module* model, Optimizer* opt,
                          int* epoch, float* loss, const char* filepath);
```

Checkpoints include model weights, optimizer state, current epoch, and loss value -- everything needed to resume training.

### Example

```c
// Save checkpoint
model_save_checkpoint(model, opt, epoch, loss_val, "checkpoint.cml");

// Resume training
int resume_epoch;
float resume_loss;
model_load_checkpoint(model, opt, &resume_epoch, &resume_loss, "checkpoint.cml");
printf("Resuming from epoch %d (loss=%.4f)\n", resume_epoch, resume_loss);
```

______________________________________________________________________

## Device Management

```c
// Available device types
DEVICE_CPU      // CPU (always available)
DEVICE_CUDA     // NVIDIA GPU
DEVICE_METAL    // Apple GPU (macOS/iOS)
DEVICE_ROCM     // AMD GPU
DEVICE_AUTO     // Auto-detect best available
```

Device selection is specified through `TensorConfig` when creating tensors, or through the `DType`/`DeviceType` arguments in layer constructors.

______________________________________________________________________

## Memory Management

### IR Context Reset

```c
void cml_reset_ir_context(void);
```

Frees all IR nodes accumulated during lazy evaluation. Call this after each training batch to prevent memory growth.

### Cleanup Context

```c
CleanupContext* cleanup_context_create(void);
void cml_register_cleanup_context(CleanupContext* ctx);
```

A cleanup context tracks allocated resources and frees them on program exit. `cleanup_context_create()` automatically registers with the global atexit handler via `cml_register_cleanup_context` -- do not call `cleanup_context_free()` manually if atexit will handle it (double-free risk).

### Resource Tracking

```c
void cml_track_module(Module* module);
void cml_track_optimizer(Optimizer* opt);
void cml_track_dataset(Dataset* ds);
```

Register resources with the global cleanup system so they are freed automatically at shutdown.

### Example

```c
cml_init();

Module* model = /* ... */;
Optimizer* opt = /* ... */;
Dataset* ds = /* ... */;

cml_track_module(model);
cml_track_optimizer(opt);
cml_track_dataset(ds);

for (int epoch = 0; epoch < 100; epoch++) {
    for (int batch = 0; batch < num_batches; batch++) {
        // ... forward, backward, step ...
        cml_reset_ir_context();  // Free IR nodes each batch
    }
}

cml_cleanup();  // Frees tracked resources
```

______________________________________________________________________

## Kernel Cache

The kernel cache stores compiled computation kernels to avoid redundant compilation.

```c
void   cml_kernel_cache_clear(void);
void   cml_kernel_cache_stats(size_t* hits, size_t* misses,
                               size_t* count, size_t* memory);
double cml_kernel_cache_hit_rate(void);
void   cml_kernel_cache_print_stats(void);
```

| Function                       | Description                                    |
| ------------------------------ | ---------------------------------------------- |
| `cml_kernel_cache_clear`       | Evict all cached kernels                       |
| `cml_kernel_cache_stats`       | Get hit/miss counts, entry count, memory usage |
| `cml_kernel_cache_hit_rate`    | Returns hit rate as a value in \[0.0, 1.0\]    |
| `cml_kernel_cache_print_stats` | Print formatted cache statistics to stdout     |

______________________________________________________________________

## Error Handling

### Global Error Handler

```c
typedef void (*CMLGlobalErrorHandler)(int code, const char* msg, void* ctx);

void cml_set_error_handler(CMLGlobalErrorHandler handler);
CMLGlobalErrorHandler cml_get_error_handler(void);
```

### Example

```c
void my_handler(int code, const char* msg, void* ctx) {
    fprintf(stderr, "[CML Error %d] %s\n", code, msg);
}

cml_set_error_handler(my_handler);
```

When set, the global error handler is invoked on any library error, giving you the opportunity to log, abort, or recover.

______________________________________________________________________

## See Also

- [Neural Network Layers](nn_layers.md) -- Detailed layer reference
- [Autograd System](autograd.md) -- Automatic differentiation internals
- [Graph Mode](graph_mode.md) -- Lazy execution and IR
- [Getting Started](getting_started.md) -- Build, install, first program
- [Datasets](datasets.md) -- Dataset hub and loading
