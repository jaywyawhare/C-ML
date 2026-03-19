# CML Python Bindings

CFFI bindings for the C-ML library. PyTorch-style API backed by a C IR/JIT execution engine.

## Install

```bash
cd C-ML && mkdir -p build && cd build && cmake .. && make -j$(nproc)
cd ../python && pip install cffi && python3 cml/build_cffi.py && pip install -e .
```

Verify: `python3 -c "import cml; cml.init(); print('ok'); cml.cleanup()"`

## Usage

```python
import cml
import cml.nn as nn
import cml.optim as optim

cml.init()
cml.seed(42)

model = nn.Sequential()
model.add(nn.Linear(784, 128))
model.add(nn.ReLU())
model.add(nn.Dropout(0.2))
model.add(nn.Linear(128, 10))

optimizer = optim.Adam(model, lr=0.001)

X = cml.randn([32, 784])
y = cml.zeros([32, 10])

model.set_training(True)
for epoch in range(10):
    optimizer.zero_grad()
    loss = cml.cross_entropy_loss(model(X), y)
    cml.backward(loss)
    optimizer.step()

cml.cleanup()
```

## Module Hierarchy

```python
import cml                     # Tensor, creation ops, init/cleanup, autograd
import cml.nn                  # Linear, Conv2d, LSTM, Transformer, Sequential, ...
import cml.optim               # Adam, SGD, AdamW, StepLR, CosineAnnealingLR, ...
import cml.losses              # mse_loss, bce_loss, nll_loss, triplet_margin_loss, ...
import cml.distributed         # multi-GPU: init, allreduce, barrier
import cml.data                # Dataset, DataLoader, train_test_split
import cml.functional          # EarlyStopping, MetricsTracker, TrainingContext
```

The top-level `cml` namespace exposes what you need 90% of the time. Everything else lives in submodules.

## API at a Glance

### Tensors

```python
x = cml.zeros([10, 20])
x = cml.randn([10, 20])
x = cml.arange(0, 10)
x = cml.eye(5)
x = cml.Tensor.from_numpy(np_array)

z = x + y                      # arithmetic: +, -, *, /, @, **, //, %
z = x.relu()                   # activations: relu, sigmoid, tanh, softmax
z = x.sum(dim=0)               # reductions: sum, mean, max, min, var, std, argmax
z = x.reshape([5, 4])          # shape ops: reshape, transpose, squeeze, unsqueeze, flip
z = x.clamp(0, 1)              # math: exp, log, sqrt, sin, cos, sign, floor, ceil, erf
arr = x.numpy()                # numpy interop
```

### Layers

```python
nn.Linear(in_f, out_f)         nn.Conv1d(c_in, c_out, k)     nn.Embedding(vocab, dim)
nn.Conv2d(c_in, c_out, k)      nn.Conv3d(c_in, c_out, k)     nn.ConvTranspose1d(...)
nn.BatchNorm1d(n)               nn.BatchNorm2d(n)              nn.LayerNorm(n)
nn.GroupNorm(groups, channels)  nn.InstanceNorm2d(n)           nn.Dropout(p)
nn.ReLU()                       nn.LeakyReLU(0.01)             nn.PReLU()
nn.MaxPool2d(k)                 nn.AvgPool2d(k)                nn.AdaptiveAvgPool2d(h, w)
nn.RNN(inp, hid)                nn.LSTM(inp, hid)              nn.GRU(inp, hid)
nn.MultiHeadAttention(d, heads) nn.TransformerEncoder(...)      nn.Flatten()
nn.Upsample(scale)              nn.PixelShuffle(r)             nn.Identity()
```

### Optimizers & Schedulers

```python
optim.Adam(model, lr=1e-3)     optim.AdamW(model, lr=1e-3)    optim.SGD(model, lr=0.01)
optim.RMSprop(model)            optim.NAdam(model)              optim.LAMB(model)
optim.LARS(model)               optim.Muon(model)              optim.Adadelta(model)

optim.StepLR(opt, step_size=10)           optim.CosineAnnealingLR(opt, T_max=100)
optim.ReduceOnPlateau(opt, patience=5)    optim.OneCycleLR(opt, max_lr=0.01, total_steps=1000)
```

### Losses

```python
cml.mse_loss(pred, target)                cml.bce_loss(pred, target)
cml.cross_entropy_loss(logits, labels)    cml.losses.nll_loss(log_probs, targets)
cml.losses.huber_loss(pred, target)       cml.losses.kl_div_loss(p, q)
cml.losses.triplet_margin_loss(a, p, n)   cml.losses.cosine_embedding_loss(x1, x2, t)
```

### Autograd

```python
loss.backward()                 # or: cml.backward(loss)
param.grad                      # access gradients directly
with cml.no_grad():             # disable gradient tracking
    output = model(x)
```

### NumPy Interop

```python
tensor = cml.Tensor.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
arr = tensor.numpy()
arr = np.array(tensor)          # __array__ protocol
```

### Device Selection

```python
if cml.is_device_available(cml.DEVICE_CUDA):
    cml.set_device(cml.DEVICE_CUDA)
```

Supports: `DEVICE_CPU`, `DEVICE_CUDA`, `DEVICE_METAL`, `DEVICE_ROCM`.

## Troubleshooting

**`ImportError: CML bindings not found`** -- Run `python3 cml/build_cffi.py` in the `python/` dir.

**`OSError: cannot find libcml.so`** -- Set `LD_LIBRARY_PATH=../../build/lib:$LD_LIBRARY_PATH`.

**Version mismatch** -- Rebuild both: `cd .. && make clean && make && cd python && python3 setup.py build_ext --inplace`.

## Requirements

- Python 3.8+, CFFI, C-ML built with CMake, LLVM 18+ (optional, for JIT)

## License

DBaJ-GPL -- Use it, modify it, share it. Don't be a jerk about it.
