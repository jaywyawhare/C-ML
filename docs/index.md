<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="dark-mode.svg">
    <source media="(prefers-color-scheme: light)" srcset="light-mode.svg">
    <img alt="C-ML" src="light-mode.svg" height="96">
  </picture>
</p>

# C-ML Documentation

**Version 0.0.3** | [GitHub](https://github.com/jaywyawhare/C-ML) | [Issues](https://github.com/jaywyawhare/C-ML/issues)


## Overview

C-ML is a production-ready machine learning library written in pure C, providing automatic differentiation, neural network layers, optimizers, and training utilities. This documentation covers all aspects of the library from basic usage to advanced features.

## Quick Navigation

### Getting Started

- **[Getting Started](getting_started.md)** - Quick start guide and installation

### Core Documentation

- **[API Reference](api_reference.md)** - Complete API documentation with examples
- **[Autograd](autograd.md)** - Automatic differentiation system
- **[Neural Network Layers](nn_layers.md)** - All 28 layers: linear, conv, recurrent, normalization, pooling, activations, utility
- **[Training](training.md)** - Training API and best practices
- **[Graph Mode](graph_mode.md)** - Lazy execution and optimization

### Advanced Features

- **[Advanced NN & LLM](advanced_nn.md)** - LoRA, QLoRA, Flash Attention, Paged Attention, LLaMA, Serving, Speculative Decoding
- **[Compiler Pipeline](compiler_pipeline.md)** - IR scheduling, linearization, fused codegen, AOT, JIT, Z3 verification
- **[GPU Backends](gpu_backends.md)** - CUDA, ROCm, Vulkan, WebGPU, Metal, OpenCL, Adreno, Hexagon
- **[Distributed Training](distributed.md)** - DDP, Pipeline Parallel, Tensor Parallel, collective ops
- **[Model I/O](model_io.md)** - GGUF, SafeTensors, ONNX, PyTorch .pth, quantization
- **[Memory Management](memory_management.md)** - TLSF allocator, graph allocator, memory pools, timeline planner

### Reference

- **[Datasets](datasets.md)** - Built-in datasets and data loading
- **[Optimizations](optimizations.md)** - IR optimizations and SIMD
- **[Linearization](linearization.md)** - IR linearization details
- **[BEAM Search](beam_search.md)** - GPU kernel auto-tuning
- **[Speculative Decoding](speculative_decoding.md)** - Draft-verify generation
- **[Kernel Studio](kernel_studio.md)** - Visualization and debugging
- **[Benchmarks](benchmarks.md)** - Performance benchmarks
- **[Miscellaneous](miscellaneous.md)** - Sparse tensors, SIMD, Winograd, augmentation, symbolic, CMake integration
- **[External Dependencies](EXTERNAL_DEPENDENCIES.md)** - Build dependencies

## Quick Start

### Installation

```bash
git clone https://github.com/jaywyawhare/C-ML.git
cd C-ML
make clean && make
```

### Basic Usage

```c
#include "cml.h"

int main() {
    cml_init();
    cml_seed(42);

    DeviceType device = cml_get_default_device();
    DType dtype = cml_get_default_dtype();

    Sequential* model = cml_nn_sequential();
    model = sequential_add_chain(model,
        (Module*)cml_nn_linear(784, 128, dtype, device, true),
        (Module*)cml_nn_relu(false),
        (Module*)cml_nn_linear(128, 10, dtype, device, true),
        NULL
    );

    Optimizer* optimizer = cml_optim_adam_for_model((Module*)model,
        0.001f, 0.0f, 0.9f, 0.999f, 1e-8f);

    Dataset* dataset = dataset_xor();
    Tensor* inputs = dataset->X;
    Tensor* targets = dataset->y;

    cml_nn_module_set_training((Module*)model, true);
    for (int epoch = 0; epoch < 100; epoch++) {
        cml_optim_zero_grad(optimizer);
        Tensor* outputs = cml_nn_module_forward((Module*)model, inputs);
        Tensor* loss = cml_nn_mse_loss(outputs, targets);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(optimizer);

        tensor_free(loss);
        tensor_free(outputs);
    }

    optimizer_free(optimizer);
    module_free((Module*)model);
    dataset_free(dataset);
    cml_cleanup();
    return 0;
}
```

## Optimization Stack

```
User Code  (C or Python API)
     |
     v
IR Graph   pattern matching, DCE, operator fusion, constant folding, Z3 verification
     |
     v
Compiler   schedule -> linearize -> fused codegen -> kernel cache
           backends: C / PTX / SPIR-V / WGSL / MSL
           modes: AOT, LLVM JIT, TinyJIT replay
     |
     v
Runtime    SIMD (SSE/AVX/AVX-512/NEON), BLAS, threading, memory pools
     |
     v
Hardware   CPU, CUDA, ROCm, Vulkan, WebGPU, Metal, OpenCL
```

## Examples

See the `examples/` directory for 29 complete working examples:

### Tutorials
- `hello_cml.c` - First program
- `simple_xor.c` - XOR classification
- `tensor_ops.c` - Tensor operations
- `linear_regression.c` - Linear regression
- `logistic_regression.c` - Logistic regression
- `mlp_classifier.c` - MLP classifier
- `autoencoder.c` - Autoencoder
- `conv_net.c` - Convolutional network
- `rnn_sequence.c` - RNN sequence modeling
- `lstm_timeseries.c` - LSTM time series
- `gru_classifier.c` - GRU classifier
- `embedding.c` - Embedding layers
- `gan.c` - Generative adversarial network
- `multi_task.c` - Multi-task learning
- `transformer.c` - Transformer model
- `lr_scheduler.c` - Learning rate scheduling
- `activations.c` - All activation functions

### Demos
- `autograd_example.c` - Gradient computation
- `auto_capture_example.c` - Graph capture
- `comprehensive_fusion_example.c` - Kernel fusion
- `dead_code_example.c` - Dead code elimination
- `early_stopping_lr_scheduler.c` - Early stopping
- `export_graph.c` - IR graph export
- `mnist_example.c` - MNIST digit recognition
- `print_kernels.c` - Kernel visualization
- `training_loop_example.c` - Full training pipeline
- `llama_inference.c` - LLaMA model inference

### Benchmarks
- `bench_forward.c` - Forward pass benchmarks
- `bench_gemm.c` - Matrix multiply benchmarks

## Platform Support

| Platform                    | Status          | Device Support  |
| --------------------------- | --------------- | --------------- |
| Linux (x86_64)              | Fully Supported | CPU, CUDA, ROCm |
| macOS (Intel/Apple Silicon) | Fully Supported | CPU, Metal      |
| Windows (MSVC/MinGW)        | Supported       | CPU             |

## Performance Tips

1. **Use appropriate data types**: Prefer `DTYPE_FLOAT32` unless precision requires `DTYPE_FLOAT64`
1. **Batch operations**: Process data in batches for better performance
1. **Memory management**: Use `CleanupContext` for automatic resource management
1. **Device selection**: Use GPU when available for large computations
1. **Graph mode**: Use lazy execution for inference workloads

## Error Handling

```c
Tensor* t = tensor_empty(shape, ndim, NULL);
if (!t && CML_HAS_ERRORS()) {
    const char* msg = CML_LAST_ERROR();
    int code = CML_LAST_ERROR_CODE();
    error_stack_print_all();
}

// Or use checked constructors
Tensor* t = CML_CHECK(tensor_empty(shape, ndim, NULL), "Failed to create tensor");
```

## Contributing

We welcome contributions! Please see the main [README](../README.md) for contributing guidelines.

## License

MIT License - see [LICENSE](../LICENSE) file for details.


**Need help?** Open an [issue](https://github.com/jaywyawhare/C-ML/issues) or check the [API Reference](api_reference.md).
