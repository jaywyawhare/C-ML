# C-ML TODO and Future Implementations

This document lists planned features, improvements, and enhancements for the C-ML library.

## Table of Contents

1. [High Priority](#high-priority)
1. [Medium Priority](#medium-priority)
1. [Low Priority](#low-priority)
1. [Performance Optimizations](#performance-optimizations)
1. [Testing and Quality](#testing-and-quality)

## High Priority

### 1. Additional Neural Network Layers

#### Conv1d - 1D Convolution

- **Status**: Not implemented
- **Priority**: High
- **File**: `src/nn/layers/conv1d.c`, `include/nn/layers/conv1d.h`
- **Description**: 1D convolution for time series data, audio processing, and sequence models
- **Implementation Notes**:
  - Similar to Conv2d but operates on 1D tensors
  - Input shape: `[batch, in_channels, length]`
  - Output shape: `[batch, out_channels, out_length]`
  - Support stride, padding, dilation
  - Use direct convolution loops (similar to Conv2d implementation)
- **Related Files**: `src/nn/layers/conv2d.c` (reference implementation)
- **Benefits**: Essential for time series models, RNNs, and audio processing

#### Conv3d - 3D Convolution

- **Status**: Not implemented
- **Priority**: High
- **File**: `src/nn/layers/conv3d.c`, `include/nn/layers/conv3d.h`
- **Description**: 3D convolution for video processing, medical imaging, and volumetric data
- **Implementation Notes**:
  - Extends Conv2d to 3D spatial dimensions
  - Input shape: `[batch, in_channels, depth, height, width]`
  - Output shape: `[batch, out_channels, out_depth, out_height, out_width]`
  - Support stride, padding, dilation in all 3 dimensions
- **Related Files**: `src/nn/layers/conv2d.c` (reference implementation)
- **Benefits**: Required for video analysis, 3D medical imaging, and volumetric CNNs

#### LayerNorm - Layer Normalization

- **Status**: ✅ **IMPLEMENTED**
- **Priority**: High
- **File**: `src/nn/layers/layernorm.c`, `include/nn/layers/layernorm.h`
- **Description**: Layer normalization for transformer architectures and modern NLP models
- **Implementation**:
  - Normalizes across features (last dimension)
  - Formula: `(x - mean) / sqrt(var + eps) * gamma + beta`
  - Similar to BatchNorm2d but normalizes across different dimensions
  - No running statistics needed (stateless normalization)
  - Learnable parameters: `gamma` (scale) and `beta` (shift)
- **Benefits**: Essential for transformers, BERT, GPT models, and modern NLP architectures

#### GroupNorm - Group Normalization

- **Status**: Not implemented
- **Priority**: Medium-High
- **File**: `src/nn/layers/groupnorm.c`, `include/nn/layers/groupnorm.h`
- **Description**: Group normalization alternative to BatchNorm, useful for small batch sizes
- **Implementation Notes**:
  - Divides channels into groups and normalizes within each group
  - Works well with batch size 1 (unlike BatchNorm)
  - Formula similar to BatchNorm but groups channels
  - Learnable parameters: `gamma` and `beta` per group
- **Related Files**: `src/nn/layers/batchnorm2d.c` (reference implementation)
- **Benefits**: Better than BatchNorm for small batches, used in modern architectures

#### ModuleList - Dynamic Module List

- **Status**: Not implemented
- **Priority**: Medium-High
- **File**: `src/nn/layers/modulelist.c`, `include/nn/layers/modulelist.h`
- **Description**: Container for dynamically adding/removing modules
- **Implementation Notes**:
  - Similar to Sequential but allows indexing and modification
  - Maintains array of Module pointers
  - Supports `append()`, `insert()`, `remove()`, `get()`
  - Collects parameters from all submodules
- **Related Files**: `src/nn/layers/sequential.c` (reference implementation)
- **Benefits**: Flexible model construction, dynamic architectures

#### ModuleDict - Named Module Dictionary

- **Status**: Not implemented
- **Priority**: Medium-High
- **File**: `src/nn/layers/moduledict.c`, `include/nn/layers/moduledict.h`
- **Description**: Container for named modules (key-value pairs)
- **Implementation Notes**:
  - Hash table or array of (name, Module\*) pairs
  - Supports `add()`, `get()`, `remove()`, `keys()`
  - Collects parameters from all submodules
- **Related Files**: `src/nn/layers/sequential.c` (reference implementation)
- **Benefits**: Named access to modules, better organization

### 2. Optimizer Improvements

#### SGD Momentum Buffer

- **Status**: TODO in code (line 339 in `src/optim.c`)
- **Priority**: High
- **File**: `src/optim.c`, `include/optim/optimizer.h`
- **Description**: Implement proper momentum buffer for SGD optimizer
- **Current Implementation**: Simple SGD without momentum buffer
- **Implementation Notes**:
  ```c
  // Current (line 339):
  // TODO: Implement momentum buffer
  // For now, simple SGD without momentum buffer

  // Need to:
  // 1. Store velocity buffer per parameter
  // 2. Update: v = momentum * v - lr * grad
  // 3. Update: param = param + v
  ```
- **Benefits**: Faster convergence, smoother optimization

#### RMSprop Optimizer

- **Status**: IMPLEMENTED
- **Priority**: High
- **File**: `src/optim.c`, `include/optim/optimizer.h`
- **Description**: Root Mean Square Propagation optimizer
- **Implementation**:
  - Added `RMSpropState` structure with `square_avg` tensor
  - Update rule: `square_avg = alpha * square_avg + (1 - alpha) * grad^2`
  - Parameter update: `param = param - lr * grad / sqrt(square_avg + eps)`
  - Parameters: `lr`, `alpha` (decay rate, default 0.99), `eps` (numerical stability)
  - Proper state initialization and cleanup
- **Function**: `optim_rmsprop(parameters, num_parameters, lr, weight_decay, alpha, epsilon)`
- **Benefits**: Good for non-stationary objectives, alternative to Adam

#### Adagrad Optimizer

- **Status**: IMPLEMENTED
- **Priority**: High
- **File**: `src/optim.c`, `include/optim/optimizer.h`
- **Description**: Adaptive Gradient optimizer with per-parameter learning rates
- **Implementation**:
  - Added `AdagradState` structure with `sum_sq_grad` tensor
  - Accumulates squared gradients: `sum_sq_grad += grad^2`
  - Update: `param = param - lr * grad / sqrt(sum_sq_grad + eps)`
  - No momentum, adaptive learning rates
  - Proper state initialization and cleanup
- **Function**: `optim_adagrad(parameters, num_parameters, lr, weight_decay, epsilon)`
- **Benefits**: Automatic learning rate adaptation, good for sparse gradients

#### AdamW Optimizer

- **Status**: Not implemented
- **Priority**: High
- **File**: `src/optim.c`, `include/optim/optimizer.h`
- **Description**: Adam with decoupled weight decay (fixes weight decay in Adam)
- **Implementation Notes**:
  - Similar to Adam but weight decay is applied separately
  - Update: `param = param - lr * (m_hat / sqrt(v_hat) + weight_decay * param)`
  - Better than Adam for regularization
- **Related Files**: `src/optim.c` (Adam implementation as reference)
- **Benefits**: Better generalization than Adam, preferred in modern training

#### AdaDelta Optimizer

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/optim.c`, `include/optim/optimizer.h`
- **Description**: Adaptive learning rate optimizer without manual learning rate
- **Implementation Notes**:
  - Maintains moving averages of gradients and parameter updates
  - No learning rate parameter needed
  - Update uses accumulated updates instead of learning rate
- **Benefits**: No need to tune learning rate, adaptive

#### Learning Rate Schedulers

- **Status**: Not implemented
- **Priority**: High
- **File**: `src/optim/lr_scheduler.c`, `include/optim/lr_scheduler.h`
- **Description**: Various learning rate scheduling strategies
- **Schedulers to Implement**:
  - **StepLR**: Decay by factor every N epochs
  - **ExponentialLR**: Exponential decay: `lr = lr * gamma^epoch`
  - **CosineAnnealingLR**: Cosine annealing schedule
  - **ReduceLROnPlateau**: Reduce LR when metric plateaus
  - **MultiStepLR**: Decay at specific milestones
  - **OneCycleLR**: One cycle policy (popularized by fastai)
- **Implementation Notes**:
  - Create scheduler interface that takes optimizer
  - Call `scheduler_step()` after each epoch or batch
  - Update optimizer learning rates
- **Benefits**: Better convergence, automatic learning rate tuning

### 3. Loss Functions

#### KL Divergence Loss

- **Status**: ✅ **IMPLEMENTED**
- **Priority**: High
- **File**: `src/autograd/loss_functions.c`, `include/autograd/loss_functions.h`
- **Description**: Kullback-Leibler divergence for probability distributions
- **Formula**: `KL(P||Q) = sum(P * log(P / Q))`
- **Use Cases**: Variational autoencoders, knowledge distillation, regularization
- **Implementation**: Forward pass computes KL divergence, backward computes gradients

#### Hinge Loss

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/autograd/loss_functions.c`, `include/autograd/loss_functions.h`
- **Description**: Maximum margin classifier loss
- **Formula**: `max(0, 1 - target * output)`
- **Use Cases**: Support vector machines, binary classification
- **Implementation**: Element-wise max operation with backward pass

#### Huber Loss

- **Status**: ✅ **IMPLEMENTED**
- **Priority**: Medium
- **File**: `src/autograd/loss_functions.c`, `include/autograd/loss_functions.h`
- **Description**: Robust loss function, less sensitive to outliers than MSE
- **Formula**:
  - If `|x| < delta`: `0.5 * x^2`
  - Else: `delta * |x| - 0.5 * delta^2`
- **Use Cases**: Robust regression, outlier handling
- **Parameters**: `delta` (threshold parameter)

#### Focal Loss

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/autograd/loss_functions.c`, `include/autograd/loss_functions.h`
- **Description**: Addresses class imbalance in classification
- **Formula**: `-alpha * (1 - p)^gamma * log(p)`
- **Use Cases**: Object detection, imbalanced datasets
- **Parameters**: `alpha` (weighting factor), `gamma` (focusing parameter)

#### Smooth L1 Loss

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/autograd/loss_functions.c`, `include/autograd/loss_functions.h`
- **Description**: Smooth version of L1 loss, used in object detection
- **Formula**:
  - If `|x| < beta`: `0.5 * x^2 / beta`
  - Else: `|x| - 0.5 * beta`
- **Use Cases**: Faster R-CNN, object detection
- **Related**: Similar to Huber loss but different parameterization

### 4. Backward Pass Verification

#### Complete Backward Pass Audit

- **Status**: Needs verification
- **Priority**: High
- **Files**: `src/autograd/forward_ops.c`, `src/autograd/backward_ops.c`
- **Description**: Verify all forward operations have corresponding backward passes
- **Tasks**:
  1. List all operations in `forward_ops.c`
  1. Check if each has a backward function in `backward_ops.c`
  1. Verify gradient correctness
  1. Test backward passes with numerical gradients
- **Operations to Check**:
  - All binary operations (add, sub, mul, div, pow)
  - All unary operations (exp, log, sqrt, sin, cos, tan)
  - All activation functions (relu, sigmoid, tanh)
  - All reduction operations (sum, mean, max, min)
  - Matrix operations (matmul, transpose)
  - Loss functions (mse, mae, bce, cross_entropy)
- **Benefits**: Ensure all operations support training

### 5. Model Persistence

#### Model Saving and Loading

- **Status**: Not implemented
- **Priority**: High
- **File**: `src/nn/model_io.c`, `include/nn/model_io.h`
- **Description**: Save and load model weights and architectures
- **Features to Implement**:
  - **Save Model State**: Save all parameters to file
    ```c
    int model_save(Module *model, const char *filepath);
    ```
  - **Load Model State**: Load parameters from file
    ```c
    int model_load(Module *model, const char *filepath);
    ```
  - **Save Full Model**: Save architecture + parameters
    ```c
    int model_save_full(Module *model, const char *filepath);
    ```
  - **Load Full Model**: Reconstruct model from saved file
    ```c
    Module *model_load_full(const char *filepath);
    ```
  - **Checkpoint Support**: Save checkpoints during training
    ```c
    int model_save_checkpoint(Module *model, Optimizer *optimizer,
                              int epoch, float loss, const char *filepath);
    ```
  - **Resume Training**: Load checkpoint and continue training
    ```c
    int model_load_checkpoint(Module *model, Optimizer *optimizer,
                              int *epoch, float *loss, const char *filepath);
    ```
- **File Format**: JSON or binary format for parameters
- **Benefits**: Model persistence, checkpointing, model sharing

## Medium Priority

### 6. Activation Functions

#### ELU (Exponential Linear Unit)

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/nn/layers/activations.c`, `include/nn/layers/activations.h`
- **Description**: ELU activation function with smooth negative values
- **Formula**: `x if x > 0 else alpha * (exp(x) - 1)`
- **Implementation**: Similar to ReLU but with exponential for negative values
- **Benefits**: Better than ReLU for negative inputs, prevents dead neurons

#### SELU (Scaled Exponential Linear Unit)

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/nn/layers/activations.c`, `include/nn/layers/activations.h`
- **Description**: Self-normalizing activation function
- **Formula**: `lambda * (x if x > 0 else alpha * (exp(x) - 1))`
- **Parameters**: `lambda = 1.0507`, `alpha = 1.6733` (fixed values)
- **Benefits**: Self-normalizing networks, no need for BatchNorm

#### Swish/SiLU (Sigmoid Linear Unit)

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/nn/layers/activations.c`, `include/nn/layers/activations.h`
- **Description**: Smooth, non-monotonic activation
- **Formula**: `x * sigmoid(x)`
- **Implementation**: Use existing sigmoid implementation
- **Benefits**: Better performance than ReLU in many cases

#### Mish

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/nn/layers/activations.c`, `include/nn/layers/activations.h`
- **Description**: Self-regularized activation function
- **Formula**: `x * tanh(softplus(x))` where `softplus(x) = log(1 + exp(x))`
- **Benefits**: Smooth, non-monotonic, better than Swish in some cases

#### Hard Swish

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/nn/layers/activations.c`, `include/nn/layers/activations.h`
- **Description**: Hard version of Swish, faster computation
- **Formula**: `x * relu6(x + 3) / 6` where `relu6(x) = min(max(x, 0), 6)`
- **Benefits**: Faster than Swish, used in mobile architectures

### 7. Data Loading Improvements

#### Multi-threaded Data Loading

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/Core/dataset.c`, `include/Core/dataset.h`
- **Description**: Parallel data loading using threads
- **Implementation Notes**:
  - Use pthreads or similar threading library
  - Prefetch batches in background threads
  - Queue system for loading batches ahead of time
- **Benefits**: Faster training, better GPU utilization

#### Data Augmentation Utilities

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/Core/data_augmentation.c`, `include/Core/data_augmentation.h`
- **Description**: Image and data augmentation functions
- **Augmentations to Implement**:
  - Random crop, random flip (horizontal/vertical)
  - Random rotation, random brightness/contrast
  - Color jitter, normalization
  - Mixup, CutMix (advanced augmentations)
- **Benefits**: Better generalization, data efficiency

#### Better Batch Sampling

- **Status**: Basic implementation exists
- **Priority**: Medium
- **File**: `src/Core/dataset.c`
- **Improvements**:
  - Weighted sampling
  - Stratified sampling
  - Random sampling with seed control
  - Sequential sampling
- **Benefits**: Better training dynamics

#### Prefetching

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/Core/dataset.c`
- **Description**: Preload next batch while current batch is being processed
- **Implementation**: Background thread or async loading
- **Benefits**: Eliminate data loading bottlenecks

### 8. Tensor Operations

#### Concatenate

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/tensor/ops.c`, `include/tensor/ops.h`
- **Description**: Concatenate tensors along specified dimension
- **Function**: `Tensor *tensor_concat(Tensor **tensors, int num_tensors, int dim)`
- **Use Cases**: Combining feature maps, joining layers

#### Stack

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/tensor/ops.c`, `include/tensor/ops.h`
- **Description**: Stack tensors along new dimension
- **Function**: `Tensor *tensor_stack(Tensor **tensors, int num_tensors, int dim)`
- **Use Cases**: Batch creation, stacking outputs

#### Split

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/tensor/ops.c`, `include/tensor/ops.h`
- **Description**: Split tensor into multiple tensors along dimension
- **Function**: `Tensor **tensor_split(Tensor *tensor, int num_splits, int dim)`
- **Use Cases**: Unpacking, splitting outputs

#### Gather

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/tensor/ops.c`, `include/tensor/ops.h`
- **Description**: Gather values from tensor using indices
- **Function**: `Tensor *tensor_gather(Tensor *input, Tensor *indices, int dim)`
- **Use Cases**: Indexing, embedding lookups

#### Scatter

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/tensor/ops.c`, `include/tensor/ops.h`
- **Description**: Scatter values into tensor at specified indices
- **Function**: `Tensor *tensor_scatter(Tensor *input, Tensor *indices, Tensor *src, int dim)`
- **Use Cases**: Sparse updates, embedding updates

#### Advanced Reductions

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/tensor/ops.c`, `include/tensor/ops.h`
- **Operations**:
  - **Standard Deviation**: `Tensor *tensor_std(Tensor *tensor, int dim, bool unbiased)`
  - **Variance**: `Tensor *tensor_var(Tensor *tensor, int dim, bool unbiased)`
  - **Argmax**: `Tensor *tensor_argmax(Tensor *tensor, int dim)`
  - **Argmin**: `Tensor *tensor_argmin(Tensor *tensor, int dim)`
- **Use Cases**: Statistics, finding max/min indices

### 9. Memory Management

#### Memory Pool Allocator

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/Core/memory_management.c`, `include/Core/memory_management.h`
- **Description**: Pre-allocated memory pools for tensor operations
- **Benefits**: Faster allocations, reduced fragmentation, better cache locality

#### Tensor Reuse

- **Status**: Partial support
- **Priority**: Medium
- **File**: `src/tensor/tensor.c`
- **Description**: Reuse tensor memory for intermediate computations
- **Implementation**: Tensor pool for common sizes
- **Benefits**: Reduced memory allocations, better performance

#### Gradient Checkpointing

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/autograd/autograd.c`, `include/autograd/autograd.h`
- **Description**: Trade computation for memory in backward pass
- **Implementation**: Recompute activations during backward instead of storing
- **Benefits**: Enable training of larger models with limited memory

#### Memory Profiling

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/Core/memory_management.c`
- **Description**: Track memory usage, identify leaks, profile allocations
- **Features**: Memory usage statistics, leak detection, allocation tracking
- **Benefits**: Better debugging, memory optimization

### 10. Broadcasting

#### Complete NumPy-style Broadcasting

- **Status**: Limited support
- **Priority**: Medium
- **Files**: `src/tensor/ops.c`, `src/autograd/backward_ops.c`
- **Description**: Full broadcasting support for all operations
- **Current Limitations**:
  - Some operations may not handle broadcasting correctly
  - Backward passes may not handle broadcasting gradients
- **Implementation Tasks**:
  1. Broadcast shapes before operations
  1. Handle gradient broadcasting in backward passes
  1. Support broadcasting for all binary operations
  1. Add broadcasting tests
- **Broadcasting Rules**:
  - Align dimensions from right
  - Dimensions must be equal or one of them is 1
  - Missing dimensions are treated as 1
- **Benefits**: More flexible operations, NumPy compatibility

## Low Priority

### 11. GPU Support

#### CUDA Implementation

- **Status**: Not implemented
- **Priority**: Low (but high impact)
- **Files**: `src/cuda/` (new directory)
- **Description**: GPU acceleration using CUDA
- **Components**:
  - CUDA tensor operations
  - GPU-accelerated autograd
  - Memory management for GPU
  - CUDA kernels for common operations
- **Benefits**: Significant speedup for large models

#### OpenCL Support

- **Status**: Not implemented
- **Priority**: Low
- **Description**: Cross-platform GPU support
- **Benefits**: Works on AMD, Intel, and other GPUs

### 12. Advanced Features

#### JIT Compilation

- **Status**: Not implemented
- **Priority**: Low
- **Description**: Just-In-Time compilation for operations
- **Benefits**: Faster execution for repeated operations

#### Parallel Backward Pass

- **Status**: Not implemented
- **Priority**: Low
- **Description**: Parallel execution of backward operations
- **Benefits**: Faster backward pass for large models

#### Double Backward (Grad of Grad)

- **Status**: Basic support exists
- **Priority**: Low
- **Description**: Complete support for higher-order derivatives
- **Benefits**: Advanced optimization techniques, meta-learning

## Performance Optimizations

### 13. Operation Optimizations

#### SIMD Optimizations

- **Status**: Not implemented
- **Priority**: Medium-High
- **Files**: `src/tensor/ops.c`
- **Description**: Use SIMD instructions for element-wise operations
- **Operations**: Add, Mul, ReLU, Sigmoid, Tanh, etc.
- **Benefits**: 2-4x speedup for element-wise operations

#### BLAS Integration

- **Status**: Not implemented
- **Priority**: Medium
- **Files**: `src/tensor/ops.c`
- **Description**: Use optimized BLAS libraries for matrix operations
- **Operations**: Matrix multiplication, dot product
- **Libraries**: OpenBLAS, Intel MKL, ATLAS
- **Benefits**: Significantly faster matrix operations

#### im2col for Convolutions

- **Status**: Not implemented
- **Priority**: Medium
- **File**: `src/nn/layers/conv2d.c`
- **Description**: Use im2col transformation for faster convolutions
- **Benefits**: Faster convolutions, especially for large kernels

### 14. Memory Optimizations

#### In-Place Operations

- **Status**: Partial support
- **Priority**: Medium
- **Description**: Expand in-place operation support for memory efficiency
- **Operations**: Add, Mul, ReLU (in-place versions)
- **Benefits**: Reduced memory usage

## Testing and Quality

### 15. Unit Tests

#### Comprehensive Test Suite

- **Status**: Basic tests may exist
- **Priority**: High
- **Files**: `test/` directory
- **Coverage Needed**:
  - Tensor operations (all operations)
  - Autograd operations (forward and backward)
  - Neural network layers (all layers)
  - Optimizers (SGD, Adam, and future optimizers)
  - Loss functions (all loss functions)
- **Testing Framework**: Create or use existing C testing framework
- **Benefits**: Ensure correctness, prevent regressions

#### Integration Tests

- **Status**: Not implemented
- **Priority**: Medium
- **Description**: End-to-end training tests
- **Tests**:
  - Complete training loop
  - Model checkpointing and resuming
  - Parameter saving and loading
- **Benefits**: Verify system integration

### 16. Code Quality

#### Static Analysis

- **Status**: Partial
- **Priority**: Medium
- **Tools**: cppcheck, clang-tidy, clang-analyzer, PVS-Studio
- **Benefits**: Catch bugs early, improve code quality

#### Code Coverage

- **Status**: Not implemented
- **Priority**: Medium
- **Description**: Measure and improve test coverage
- **Tools**: gcov, lcov
- **Goal**: 80%+ coverage for critical paths
- **Benefits**: Identify untested code

### 17. Benchmarking

#### Performance Benchmarks

- **Status**: Not implemented
- **Priority**: Medium
- **Description**: Benchmark common operations and training loops
- **Benchmarks**:
  - Operation speed (forward and backward)
  - Training loop performance
  - Memory usage
  - Comparison with other libraries
- **Benefits**: Track performance, identify bottlenecks

## Implementation Status Summary

### Completed

- Core autograd system
- Basic neural network layers (Linear, Conv2d, BatchNorm2d, Pooling, Activations)
- SGD and Adam optimizers (SGD needs momentum buffer)
- Basic loss functions (MSE, MAE, BCE, Cross Entropy)
- Sequential container
- Basic tensor operations

### In Progress / Needs Work

- SGD momentum buffer (TODO in code)
- Complete backward passes verification
- Broadcasting improvements
- Memory optimizations

### Planned (High Priority)

- Additional layers (Conv1d, Conv3d, LayerNorm, GroupNorm, ModuleList, ModuleDict)
- More optimizers (RMSprop, Adagrad, AdamW, AdaDelta)
- Learning rate schedulers
- Additional loss functions (KL Divergence, Hinge, Huber, Focal Loss)
- Model persistence (save/load, checkpoints)

### Planned (Medium Priority)

- Activation functions (ELU, SELU, Swish, Mish, Hard Swish)
- Data loading improvements (multi-threading, augmentation)
- Tensor operations (Concatenate, Stack, Split, Gather, Scatter)
- Memory management (pools, profiling, checkpointing)
- Broadcasting (complete NumPy-style)

### Planned (Low Priority)

- GPU support (CUDA, OpenCL)
- Advanced features (JIT, parallel backward, double backward)

## Contribution Guidelines

When implementing new features:

1. **Follow existing code style** - Match indentation, naming conventions
1. **Add comprehensive documentation** - Doxygen-style comments
1. **Include unit tests** - Test forward and backward passes
1. **Update relevant documentation** - Update docs/TODO_IMPLEMENTATIONS.md
1. **Add examples if applicable** - Show usage in examples/
1. **Ensure memory safety** - Proper allocation and deallocation
1. **Integrate with autograd** - Ensure backward passes work correctly

## Priority Justification

**High Priority** items are:

- Essential for common use cases
- Frequently requested features
- Blocking other improvements
- Core functionality gaps

**Medium Priority** items are:

- Useful enhancements
- Performance improvements
- Quality of life features
- Nice-to-have improvements

**Low Priority** items are:

- Advanced use cases
- Long-term goals
- Major infrastructure changes

## Notes

- This list is continuously updated as features are implemented
- Priorities may change based on user feedback and requirements
- Some items may be combined or split as implementation progresses
- Consider contributing to high-priority items first
- Check existing implementations for reference patterns
