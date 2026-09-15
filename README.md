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

## Build

```bash
git clone https://github.com/jaywyawhare/C-ML.git
cd C-ML && mkdir -p build && cd build
cmake -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON .. && make -j$(nproc)
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

---

## Features

| | |
|:--|:--|
| **Layers** | 28: Linear, Conv1d/2d/3d, ConvTranspose, RNN/LSTM/GRU, Transformer, BatchNorm, LayerNorm, RMSNorm, Pooling, Dropout, PixelShuffle, ... |
| **LLM** | LoRA/QLoRA, Flash Attention, GQA, paged KV cache, RoPE, MoE, speculative decoding, LLaMA 7B-70B |
| **Training** | 11 optimizers, 13 losses, 8 LR schedulers, gradient checkpointing, DDP, pipeline & tensor parallel |
| **Compiler** | IR fusion (5 patterns), linearization, codegen to C/PTX/SPIR-V/WGSL/MSL, AOT, JIT, kernel cache |
| **GPU** | CUDA, ROCm, Vulkan, Metal, WebGPU, OpenCL — userspace drivers for NV (RM ioctl) and AMD (KFD) (see maturity below) |
| **Runtime** | LLVM JIT by default (shape-specialized SIMD; interpreter fallback), multi-dtype compute (f32/f64/f16/bf16/int), BLAS, TLSF allocator, memory pools, thread pool |
| **I/O** | GGUF, SafeTensors, ONNX, PyTorch .pth, int8/int4/NF4 quantization |
| **Python** | CFFI bindings, NumPy integration, operator overloading |

### Backend maturity

GPU paths are exercised by CI only through the CPU reference backend, mocks,
and emulators — driver-call correctness needs the real hardware. Status per
backend as of 2026-08:

| Backend | Maturity | Notes |
|---|---|---|
| CPU (SIMD/BLAS/LLVM JIT) | stable | default; full test suite runs on it |
| HCQ queue/signal/pipeline | stable | uniform ops dispatch incl. CPU reference (`test_hcq_cpu`, `test_hcq_registry`) |
| CUDA / ROCm / OpenCL / Vulkan / WebGPU adapters | build-verified | compile everywhere, dlopen at runtime, fail cleanly without hardware; driver bodies unvalidated on-GPU |
| NV (RM ioctl) / AM (KFD) drivers | experimental | full mock-based tests (`nv_mock`, `am_mock`); real ioctl paths need hardware |
| Adreno / Hexagon / Thunder / USB | placeholder | identity/generic kernels or stubs; not for production |
| Multi-GPU (peer copy, device placement) | simulated only | `DEVICE_SIM_GPU` device simulator |

---

## Architecture

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
hardware        CUDA, ROCm, Vulkan, Metal, WebGPU, OpenCL, Adreno, Hexagon, CPU
```

---

## Testing & coverage

156 C test programs (plus Python pytest) run in CI. The suite includes
cross-dtype conformance (77 ops x 8 dtypes against f32 references), a VJP
sweep that checks every eager-backward rule against central finite
differences, an API contract sweep (bad arguments on every public surface),
and HCQ/quantization/matmul exactness tests.

Branch coverage is measured with union semantics — a branch counts as covered
when ANY test binary takes it:

```sh
cmake -S . -B build-coverage -DCMAKE_C_FLAGS="--coverage -O0"
cmake --build build-coverage -j
cmake --build build-coverage --target coverage   # runs ctest serially + report
```

Serial execution is required: all binaries update one shared set of `.gcda`
counters. Current baseline: ~58% line / ~38% branch outcomes; the largest
remaining gaps are the hardware-gated `gpu/*` backends and the f32 SIMD fast
paths.

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
| [IR Graph Management](docs/ir_graph_management.md) | [Linearization](docs/linearization.md) |
| [Datasets](docs/datasets.md) | [Optimizations](docs/optimizations.md) |
| [Autograd](docs/autograd.md) | [Graph Mode](docs/graph_mode.md) |
| [BEAM Search](docs/beam_search.md) | [Speculative Decoding](docs/speculative_decoding.md) |
| [Kernel Studio](docs/kernel_studio.md) | [Kernel Studio Quick Ref](docs/kernel_studio_quickref.md) |
| [Benchmarks](docs/benchmarks.md) | [Examples](docs/examples.md) |
| [Miscellaneous](docs/miscellaneous.md) | [External Deps](docs/EXTERNAL_DEPENDENCIES.md) |
| [Python Bindings](docs/python_installation.md) | [Python (repo)](python/INSTALLATION.md) |
| [License](docs/license.md) | [Documentation index](docs/index.md) |

</details>

[DBaJ-NC-CFL](LICENSE.md)
