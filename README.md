<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/dark-mode.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/light-mode.svg">
    <img alt="C-ML" src="docs/light-mode.svg" height="96">
  </picture>
</p>

<p align="center">Machine learning in pure C. No frameworks. No runtime. Just code.</p>

<p align="center">
  <a href="https://github.com/jaywyawhare/C-ML/releases"><img src="https://img.shields.io/badge/version-0.0.3-blue.svg" alt="Version"></a>
  <img src="https://img.shields.io/badge/C11-compatible-blue.svg" alt="C11">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg" alt="Platform">
</p>

---

C-ML is a from-scratch C11 ML library. Tensors, autograd, 28 layers, optimizers, a compiler that fuses ops and emits GPU kernels (PTX/SPIR-V/WGSL/MSL), distributed training, LLM inference, and model I/O for GGUF/ONNX/SafeTensors. Zero required dependencies.

```bash
git clone https://github.com/jaywyawhare/C-ML.git
cd C-ML && mkdir -p build && cd build
cmake -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON .. && make -j$(nproc)
```

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

---

| | |
|:--|:--|
| **Layers** | 28: Linear, Conv1d/2d/3d, ConvTranspose, RNN/LSTM/GRU, Transformer, BatchNorm, LayerNorm, RMSNorm, Pooling, Dropout, PixelShuffle, ... |
| **LLM** | LoRA/QLoRA, Flash Attention, GQA, paged KV cache, RoPE, MoE, speculative decoding, LLaMA 7B-70B |
| **Training** | 9 optimizers, 13 losses, 6 LR schedulers, gradient checkpointing, DDP, pipeline & tensor parallel |
| **Compiler** | IR fusion (11 patterns), linearization, codegen to C/PTX/SPIR-V/WGSL/MSL, AOT, JIT, kernel cache |
| **GPU** | CUDA, ROCm, Vulkan, Metal, WebGPU, OpenCL — userspace drivers for NV (RM ioctl) and AMD (KFD) |
| **Runtime** | SIMD (SSE/AVX/AVX-512/NEON), BLAS, TLSF allocator, memory pools, thread pool |
| **I/O** | GGUF, SafeTensors, ONNX, PyTorch .pth, int8/NF4 quantization |
| **Python** | CFFI bindings, NumPy integration, operator overloading |

---

```
user code       C or Python
     |
cml.h           28 layers, optimizers, losses, LLM ops, serving
     |
autograd        dynamic graphs, checkpointing, distributed
     |
tensor ops      broadcasting, SIMD, BLAS
     |
compiler        IR -> schedule -> linearize -> codegen -> cache
     |
drivers         HCQ, NV (RM ioctl), AM (KFD), TLSF, pools
     |
hardware        CUDA, ROCm, Vulkan, Metal, WebGPU, OpenCL, CPU
```

---

<details>
<summary><b>Docs</b></summary>

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

</details>

[DBaJ-NC-CFL](LICENCE.md)
