<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/dark-mode.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/light-mode.svg">
    <img alt="C-ML" src="docs/light-mode.svg" height="96">
  </picture>
</p>

<p align="center">Machine learning in pure C. No frameworks. No runtime. Just code.</p>

<p align="center">
  <a href="https://github.com/jaywyawhare/C-ML/releases"><img src="https://img.shields.io/badge/version-0.0.2-blue.svg" alt="Version"></a>
  <img src="https://img.shields.io/badge/C11-compatible-blue.svg" alt="C11">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg" alt="Platform">
</p>

---

C-ML is a machine learning library written from scratch in C11 with zero required dependencies beyond a C compiler and libm. It covers the full stack: tensor operations, automatic differentiation, 28 neural network layers, optimizers, loss functions, a compiler pipeline with operator fusion and codegen to six GPU backends, distributed training, LLM inference (LoRA, paged attention, speculative decoding), and model I/O for GGUF/ONNX/SafeTensors/PyTorch formats. Everything is lazy-evaluated through an IR graph that gets optimized, linearized, and compiled to native code (PTX, SPIR-V, MSL, WGSL, or plain C) before hitting hardware. Python bindings via CFFI are included.

## Build

```bash
git clone https://github.com/jaywyawhare/C-ML.git
cd C-ML && mkdir -p build && cd build
cmake -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON ..
make -j$(nproc)
ctest --output-on-failure   # 56 tests
```

## Usage

```c
#include "cml.h"

int main(void) {
    cml_init();

    Dataset* ds = cml_dataset_load("iris");
    dataset_normalize(ds, "minmax");
    Dataset *train, *test;
    dataset_split(ds, 0.8f, &train, &test);

    Sequential* model = cml_nn_sequential();
    DeviceType dev = cml_get_default_device();
    DType dt = cml_get_default_dtype();
    model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(4, 16, dt, dev, true));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(16, 3, dt, dev, true));

    Optimizer* opt = cml_optim_adam_for_model((Module*)model, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    cml_nn_module_set_training((Module*)model, true);

    for (int epoch = 0; epoch < 100; epoch++) {
        cml_optim_zero_grad(opt);
        Tensor* out = cml_nn_module_forward((Module*)model, train->X);
        Tensor* loss = cml_nn_mse_loss(out, train->y);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(opt);
        tensor_free(loss);
        tensor_free(out);
    }

    cml_cleanup();
}
```

```bash
gcc -std=c11 -O2 example.c -I./include -L./build/lib -lcml_static -lm -ldl -o example
```

## What's in the box

**Layers** -- 28 total. Linear, Conv1d/2d/3d, ConvTranspose1d/2d/3d, RNN, LSTM, GRU, Transformer, Embedding, BatchNorm, LayerNorm, RMSNorm, InstanceNorm, GroupNorm, Pooling, Dropout, PReLU, PixelShuffle, Upsample. Sequential/ModuleList/ModuleDict containers.

**LLM** -- LoRA/QLoRA fine-tuning, Flash Attention, grouped-query attention, paged KV cache, RoPE, mixture of experts, speculative decoding, continuous batching server. LLaMA 7B/13B/70B inference from GGUF.

**Training** -- 9 optimizers (Adam, AdamW, SGD, RMSprop, Adagrad, AdaDelta). 6 LR schedulers. 13 loss functions. Gradient checkpointing. DDP with bucketed all-reduce, GPipe pipeline parallelism, Megatron tensor parallelism over NCCL/MPI/Gloo.

**Compiler** -- Lazy IR graph with DCE, operator fusion (11+ patterns), pattern matching, Z3 formal verification. Schedule, linearize, fused codegen to C/PTX/SPIR-V/WGSL/MSL. AOT compilation, LLVM JIT, TinyJIT replay, kernel caching.

**GPU** -- CUDA (PTX codegen + userspace NV driver via RM ioctl), ROCm (KFD + AQL dispatch), Vulkan (SPIR-V), Metal (MSL), WebGPU (WGSL), OpenCL. Hardware command queue abstraction. Automatic dispatch with CPU fallback.

**Runtime** -- SIMD (SSE/AVX/AVX-512/NEON, runtime detected), BLAS (MKL/OpenBLAS/Accelerate, dynamically loaded), thread pool, TLSF allocator, memory pools, graph allocator with timeline planning.

**I/O** -- GGUF, SafeTensors, ONNX, PyTorch .pth loading. Int8/NF4 quantization. Architecture export to JSON.

**Data** -- 9 built-in datasets with auto-download and caching. CSV parser. Augmentation pipeline.

**Python** -- CFFI bindings with NumPy integration, operator overloading, context managers.

## Stack

```
user code       C or Python
     |
     v
cml.h           public API, 28 layers, optimizers, losses, LLM ops, serving
     |
     v
autograd        dynamic graphs, checkpointing, distributed training
     |
     v
tensor ops      broadcasting, SIMD, BLAS
     |
     v
compiler        IR -> schedule -> linearize -> codegen -> kernel cache
     |
     v
drivers         HCQ, NV (RM ioctl), AM (KFD), memory (TLSF, pools)
     |
     v
hardware        CUDA, ROCm, Vulkan, Metal, WebGPU, OpenCL, CPU
```

## Docs

| | |
|---|---|
| [Getting Started](docs/getting_started.md) | [API Reference](docs/api_reference.md) |
| [NN Layers](docs/nn_layers.md) | [Advanced NN & LLM](docs/advanced_nn.md) |
| [Training](docs/training.md) | [Compiler Pipeline](docs/compiler_pipeline.md) |
| [GPU Backends](docs/gpu_backends.md) | [Distributed Training](docs/distributed.md) |
| [Model I/O](docs/model_io.md) | [Memory Management](docs/memory_management.md) |
| [Datasets](docs/datasets.md) | [Optimizations](docs/optimizations.md) |
| [Autograd](docs/autograd.md) | [Graph Mode](docs/graph_mode.md) |
| [BEAM Search](docs/beam_search.md) | [Speculative Decoding](docs/speculative_decoding.md) |
| [Kernel Studio](docs/kernel_studio.md) | [Benchmarks](docs/benchmarks.md) |
| [External Deps](docs/EXTERNAL_DEPENDENCIES.md) | [Python Setup](python/INSTALLATION.md) |

## License

[DBaJ-NC-CFL](LICENCE.md)
