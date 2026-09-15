# C-ML Quickstart

C-ML is a machine-learning framework written in C (~110K LOC) with PyTorch-style
Python bindings. This tutorial covers the essentials in ~10 minutes.

## Installation

```bash
# from a checkout (builds the C library via CMake automatically)
pip install ./python
```

Requires: CMake ≥ 3.16, a C11 compiler, LLVM (for the JIT backend), Python 3.9+.

## Tensors

```python
import cml
import numpy as np

cml.init()                       # one-time library setup

t = cml.Tensor(np.random.rand(3, 4).astype(np.float32))
z = cml.zeros([2, 3])            # also: ones, randn, arange, eye, full ...
```

`Tensor.numpy()` copies data out; every op allocates a fresh tensor
(out-of-place, like PyTorch).

## Operations

```python
a = cml.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
b = cml.ones([2, 3])

s   = a + b                      # elementwise add
mm  = a.matmul(a.transpose(0, 1))          # matmul / @-style
sm  = cml.einsum("ij,jk->ik", a, a.transpose(0, 1))
sel = cml.where(a > 2.0, a, -a)
sh  = a.reshape(3, 2).roll(1, axis=0)
```

The numpy-API breadth matrix (`python/tests/test_numpy_matrix.py`) documents
exactly which operations agree with numpy and where conventions differ.

## Autograd and training

```python
import cml.nn as cnn
import cml.losses as closses
import cml.optim as coptim

model = cnn.Sequential(
    cnn.Linear(4, 8),
    cnn.ReLU(),
    cnn.Linear(8, 2),
)

opt = coptim.SGD(model, lr=0.05)

for step in range(100):
    pred = model(cml.Tensor(X))            # X: float32 [batch, 4]
    loss = closses.mse_loss(pred, cml.Tensor(Y))
    loss.backward()
    opt.step()
    opt.zero_grad()                        # graph resets inside step()
```

Gradients land on `param.grad`; `cml.reset_graph()` fully clears the global
graph between models. Training parity against PyTorch is pinned by
`python/tests/test_torch_parity.py` — MLP **and** CNN loss curves match to
~1e-06 per step.

## Where to go next

| Topic | Entry point |
|---|---|
| Full op coverage vs numpy | `python/tests/test_numpy_matrix.py` |
| Torch parity harness | `python/tests/test_torch_parity.py` |
| C API | `include/cml.h` |
| Backend selection (CPU/OpenCL/Vulkan/CUDA/ROCm) | README backend table |
| Quantized inference (int8/int4/NF4) | `tests/test_quant_matmul.c` |
| Architecture notes & audit history | `docs/audits/AUDIT.md` |

## Environment knobs

| Variable | Effect |
|---|---|
| `CML_THREADS=n` | worker threads for elementwise kernels (default: all cores; `1` disables) |
| `GRAD_MODE=eager` | eager backward instead of graph-level autodiff |
| `CML_STRICT_GRAD=1` | fail loudly when an op has no gradient rule |
| `BACKEND=opencl/vulkan/...` | force an execution backend |
| `CML_AUTOTUNE=1` | auto-tune GEMM tile sizes (cached in ~/.cml) |
