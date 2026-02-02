# CML Architecture

This document describes the internal architecture of the C-ML library.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Core Components](#core-components)
3. [Data Structures](#data-structures)
4. [Execution Flow](#execution-flow)
5. [Memory Management](#memory-management)
6. [MLIR Integration](#mlir-integration)
7. [Computation Graph](#computation-graph)
8. [Device Management](#device-management)

## High-Level Architecture

```
┌─────────────────────────────────────────────────┐
│           User Application Code                  │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────┐
│        CML High-Level API (cml.h)               │
│  - Neural Networks (NN)                         │
│  - Optimizers                                   │
│  - Loss Functions                               │
│  - Training Utilities                           │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────┐
│        Autograd & Tensor Operations             │
│  - Automatic Differentiation                    │
│  - Tensor Operations                            │
│  - Graph Building                               │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────┐
│    IR Graph & Optimization Layer                │
│  - IR generation from operations                │
│  - Graph optimization                           │
│  - Kernel caching                               │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────┐
│    MLIR Backend & Execution                     │
│  - MLIR conversion                              │
│  - JIT compilation                              │
│  - Multi-backend support                        │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────┐
│    Hardware Backends                            │
│  - CPU (LLVM)                                   │
│  - CUDA (NVIDIA)                                │
│  - Metal (Apple)                                │
│  - ROCm (AMD)                                   │
│  - Vulkan                                       │
└─────────────────────────────────────────────────┘
```

## Core Components

### 1. Tensor Module (`tensor/`)

The foundation for all data structures.

**Files:**
- `tensor.c/h` - Core tensor implementation
- `tensor_views.c/h` - Zero-copy tensor slicing
- `tensor_manipulation.c/h` - Reshape, transpose, etc.

**Key Structures:**
```c
typedef struct {
    float* data;           // Tensor data
    int* shape;           // Shape (dimensions)
    int ndim;             // Number of dimensions
    int* strides;         // Memory strides
    size_t size;          // Total elements
    DType dtype;          // Data type
    DeviceType device;    // Device location
    // ... metadata
} Tensor;
```

**Responsibilities:**
- Tensor creation and destruction
- Memory allocation and management
- Shape and stride computations
- Zero-copy views for efficient slicing

### 2. Autograd Module (`autograd/`)

Automatic differentiation system.

**Files:**
- `autograd.c/h` - Core autograd implementation
- `forward_ops.c/h` - Forward pass operations
- `loss_functions.c/h` - Loss computations
- `checkpointing.c/h` - Memory-efficient gradients

**Key Concepts:**
```c
// Computation graph node
typedef struct {
    Tensor* (*forward)(Tensor* inputs);
    void (*backward)(Tensor* grad_output, ...);
    Tensor** inputs;
    Tensor* output;
    // ... metadata
} ComputationNode;
```

**Responsibilities:**
- Track computation graphs
- Compute gradients via backward pass
- Support gradient checkpointing
- Manage gradient buffers

### 3. Neural Network Module (`nn/`)

High-level layer implementations.

**File Structure:**
```
nn/
├── layers/
│   ├── linear.c/h         - Fully connected layers
│   ├── conv2d.c/h         - 2D convolutions
│   ├── batchnorm2d.c/h    - Batch normalization
│   ├── layernorm.c/h      - Layer normalization
│   ├── dropout.c/h        - Dropout regularization
│   ├── pooling.c/h        - Pooling operations
│   ├── activations.c/h    - Activation functions
│   └── sequential.c/h     - Sequential container
└── nn.c/h                 - Module interface
```

**Layer Base Structure:**
```c
typedef struct {
    Tensor* (*forward)(void* layer, Tensor* input);
    void (*backward)(void* layer, Tensor* grad_output);
    void (*free)(void* layer);
    DType dtype;
    DeviceType device;
    bool training;
} Module;
```

**Responsibilities:**
- Provide trainable layers
- Parameter management
- Training/inference modes
- Modular composition (Sequential)

### 4. Optimizer Module (`optim/`)

Parameter update algorithms.

**Available Optimizers:**
```c
typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_ADAM,
    OPTIMIZER_RMSPROP,
    OPTIMIZER_ADAGRAD
} OptimizerType;
```

**State per parameter:**
```c
typedef struct {
    Tensor* velocity;      // Momentum (SGD)
    Tensor* m, *v;        // First/second moments (Adam)
    Tensor* accumulator;   // Accumulated gradients
    float learning_rate;
    // ... hyperparameters
} OptimizerState;
```

**Responsibilities:**
- Parameter updates
- Momentum and adaptive learning rates
- Learning rate scheduling
- Weight decay

### 5. Operations Module (`ops/`)

Tensor operation implementations.

**File Structure:**
```
ops/
├── uops.c/h               - Unit operations
├── simd_utils.c/h         - SIMD utilities
├── simd_math.c/h          - SIMD math functions
├── ir/
│   ├── ir.c/h             - IR representation
│   ├── context.c/h        - IR context
│   ├── execution.c/h      - IR execution
│   ├── backward.c/h       - Backward pass
│   ├── optimization.c/h   - Graph optimization
│   ├── graph_cache.c/h    - Kernel caching
│   └── export.c/h         - Graph export
└── mlir/
    ├── mlir_context.c/h   - MLIR environment
    ├── mlir_convert.c/h   - Tensor → MLIR
    ├── mlir_execute.c/h   - MLIR execution
    ├── mlir_fusion.c/h    - Operation fusion
    └── backends/          - Device backends
```

**IR Graph Structure:**
```c
typedef struct {
    IRNode** nodes;        // Computation nodes
    int num_nodes;
    Tensor** inputs;
    Tensor** outputs;
    // ... optimization info
} IRGraph;

typedef struct {
    IROpType type;         // Operation type
    IRNode** inputs;       // Input nodes
    Tensor* output;
    void* kernel;          // Compiled kernel
} IRNode;
```

**Responsibilities:**
- Low-level tensor operations
- IR generation from operations
- MLIR conversion
- Kernel caching

### 6. Backend Module (`backend/`)

Hardware abstraction layer.

**Files:**
- `backend.c/h` - Backend interface
- `device.c/h` - Device management
- `backend_buffer.c/h` - Memory buffers
- `profiling.c/h` - Performance profiling
- `blas.c/h` - Linear algebra routines

**Device Interface:**
```c
typedef struct {
    DeviceType type;
    void* (*malloc)(size_t size);
    void (*free)(void* ptr);
    void (*memcpy)(void* dst, const void* src, size_t size);
    Tensor* (*execute_kernel)(void* kernel, Tensor** inputs);
    // ... device operations
} Backend;
```

**Responsibilities:**
- Abstract hardware differences
- Device memory management
- Operation dispatch
- Performance profiling

### 7. Memory Management Module (`alloc/`)

Efficient memory handling.

**Files:**
- `memory_management.c/h` - Core memory API
- `memory_pools.c/h` - Memory pooling
- `graph_allocator.c/h` - Graph-aware allocation

**Memory Pool Structure:**
```c
typedef struct {
    void** free_blocks;    // Available memory
    int num_free;
    void** allocated;      // Allocated blocks
    int num_allocated;
    size_t block_size;
} MemoryPool;
```

**Responsibilities:**
- Allocate/deallocate memory
- Memory pooling for efficiency
- Prevent fragmentation
- Automatic garbage collection

### 8. Core Module (`core/`)

Utility and infrastructure functions.

**Files:**
- `logging.c/h` - Logging system
- `error_stack.c/h` - Error tracking
- `cleanup.c/h` - Resource cleanup (RAII)
- `dataset.c/h` - Dataset utilities
- `augmentation.c/h` - Data augmentation
- `computation_graph.c/h` - Graph management
- `training_metrics.c/h` - Metric tracking
- `serialization.c/h` - Save/load models

**Responsibilities:**
- Logging and debugging
- Error handling
- Resource management
- Data loading
- Metric tracking

## Data Structures

### Tensor

```c
typedef struct {
    // Data and shape
    float* data;
    int* shape;
    int ndim;
    int* strides;
    size_t size;

    // Type information
    DType dtype;
    DeviceType device;

    // Autograd tracking
    bool requires_grad;
    struct ComputationNode* grad_fn;
    Tensor* grad;

    // Reference counting
    int ref_count;

    // Memory ownership
    bool owns_data;
    void* allocator;
} Tensor;
```

### Module

```c
typedef struct {
    // Function pointers
    Tensor* (*forward)(void* module, Tensor* input);
    void (*backward)(void* module, Tensor* grad);
    void (*free)(void* module);

    // Module parameters
    Tensor** parameters;
    int num_parameters;

    // Device and dtype
    DType dtype;
    DeviceType device;
    bool training;

    // Implementation-specific data
    void* impl;
} Module;
```

### ComputationGraph

```c
typedef struct {
    // Nodes and edges
    ComputationNode** nodes;
    int num_nodes;
    Tensor** inputs;
    Tensor** outputs;

    // Execution info
    IRGraph* ir;
    void* mlir_module;

    // Caching
    KernelCache* kernel_cache;

    // Optimization
    bool optimized;
    OptimizationStats stats;
} ComputationGraph;
```

## Execution Flow

### Forward Pass

```
1. User calls operation (e.g., tensor_matmul)
   ↓
2. Operation creates/updates computation graph
   ↓
3. IR generation from operation
   ↓
4. MLIR conversion (or execute in IR)
   ↓
5. JIT compilation (if needed)
   ↓
6. Kernel execution on backend
   ↓
7. Return result tensor
```

### Backward Pass

```
1. User calls cml_backward(loss_tensor)
   ↓
2. Traverse computation graph in reverse
   ↓
3. For each node:
   - Compute gradient w.r.t. inputs
   - Accumulate into input.grad
   ↓
4. Parameters now have gradients
```

### Training Loop

```
1. Initialize model and optimizer
   ↓
2. For each epoch:
   ↓
   3. For each batch:
      ↓
      4. Forward pass: output = model(input)
      ↓
      5. Compute loss: loss = criterion(output, target)
      ↓
      6. Backward pass: cml_backward(loss)
      ↓
      7. Update: optimizer.step()
      ↓
      8. Clear gradients: optimizer.zero_grad()
```

## Memory Management

### Reference Counting

Every tensor tracks reference count:

```c
Tensor* x = tensor_randn(...);  // ref_count = 1
Tensor* y = x;                   // ref_count = 2
tensor_free(x);                  // ref_count = 1
tensor_free(y);                  // ref_count = 0, actually freed
```

### Memory Pools

Efficient allocation for tensors:

```
Pool 1: 1MB blocks (small tensors)
Pool 2: 10MB blocks (medium tensors)
Pool 3: 100MB blocks (large tensors)
```

Benefits:
- Reduces fragmentation
- Faster allocation/deallocation
- Enables memory reuse

### Automatic Cleanup (RAII)

Using cleanup contexts:

```c
CleanupContext ctx = cleanup_create();
Tensor* x = tensor_create(..., &ctx);
Tensor* y = tensor_create(..., &ctx);
// ... use tensors ...
cleanup_invoke(&ctx);  // All freed automatically
```

## MLIR Integration

### Conversion Pipeline

```
Operation (e.g., tensor_matmul)
    ↓
Build IR Graph (uops)
    ↓
IR Optimization
    - Dead code elimination
    - Constant folding
    - Operation fusion
    ↓
MLIR Dialect Conversion
- Lower to linalg dialect
- Apply MLIR passes
    ↓
MLIR Optimization
- Fusion passes
- Vectorization
- Parallelization
    ↓
LLVM Lowering
    ↓
JIT Compilation
    ↓
Execution Engine
    ↓
Result Tensor
```

### Kernel Caching

Avoid recompilation:

```
IR Graph Hash → Cached Compiled Kernel?
├─ Yes: Execute directly
└─ No: Compile and cache
```

## Computation Graph

### Graph Building

Tensors track their computation history:

```c
Tensor* a = tensor_randn(...);
Tensor* b = tensor_randn(...);
Tensor* c = tensor_add(a, b);      // c.grad_fn = AddNode(a, b)
Tensor* d = tensor_multiply(c, 2); // d.grad_fn = MulNode(c, 2)
Tensor* loss = tensor_sum(d);      // loss.grad_fn = SumNode(d)

// Graph structure:
// a  b
//  \ /
//   c
//   |
//   d
//   |
// loss
```

### Graph Optimization

Before execution, optimize the graph:

```
Original Graph:
  a + b + c + d + e

After Fusion:
  fused_add_5(a, b, c, d, e)

Speedup: 5x (5 kernels → 1 kernel)
```

### Dynamic Graphs

Graphs can be different for each forward pass:

```c
if (training) {
    x = tensor_dropout(x, 0.5);  // Training branch
} else {
    // Inference branch
}

cml_backward(loss);  // Only computes gradients for used operations
```

## Device Management

### Device Abstraction

```c
// Uniform API across devices
Tensor* x = tensor_randn((int[]){1000, 1000}, 2, NULL);
cml_set_device(DEVICE_CUDA);
Tensor* y = tensor_matmul(x, y);  // Executes on CUDA

// Device transfer
Tensor* x_cpu = tensor_to_device(x, DEVICE_CPU);
```

### Multi-Device Execution

Distribute computation across devices:

```c
Tensor* x = tensor_to_device(x_orig, DEVICE_CUDA);
Tensor* y = tensor_to_device(y_orig, DEVICE_ROCM);

// Operations automatically handle device transfers
Tensor* z = tensor_add(x, y);  // x moved to ROCm device
```

## Performance Optimizations

### 1. Operation Fusion

Combine multiple operations into single kernel:

```
Before: a = x + y; b = a * z;    // 2 kernel launches
After:  b = (x + y) * z;         // 1 kernel launch
```

### 2. Constant Folding

Pre-compute constant expressions:

```
Before: loss = weight * input + bias (weight is constant)
After:  const_term = precomputed_value; loss = const_term + bias
```

### 3. Dead Code Elimination

Remove unused computations:

```
Before: z = x * y; unused = x + y; return z
After:  return x * y
```

### 4. Kernel Caching

Avoid recompilation of identical patterns.

### 5. SIMD Vectorization

Automatically use SIMD instructions (SSE, AVX, AVX-512).

## Design Decisions

1. **Pure C Implementation**: Ensures portability and ease of embedding
2. **MLIR Backend**: Leverages industry-standard infrastructure for optimization
3. **Dynamic Graphs**: Enables flexible architectures without compilation
4. **Reference Counting**: Simple, efficient memory management
5. **Layered Architecture**: Clear separation of concerns
6. **Zero-Copy Views**: Efficient tensor slicing without data duplication

## Next Steps

- [API Guide](API_GUIDE.md) - Learn the public API
- [Quick Start](QUICK_START.md) - Build your first program
- [Running Programs](RUNNING.md) - Execute examples
