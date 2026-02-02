# CML Python Bindings

Full Python support for C-ML via CFFI (C Foreign Function Interface).

## Overview

The Python bindings provide a complete Pythonic interface to CML's high-performance machine learning library. Use CML's MLIR-powered execution engine directly from Python!

## Key Features

- **Full API Coverage**: All CML operations available in Python
- **High Performance**: Uses C-ML's MLIR JIT compilation
- **Pythonic API**: Natural Python interface to C code
- **Automatic Memory Management**: Reference counting prevents leaks
- **GPU Support**: CUDA, Metal, ROCm backends available
- **Easy Installation**: Standard Python package

## Quick Start

### Installation

```bash
# Build C library first
cd ..
make

# Install Python bindings
cd python
pip install cffi
python setup.py install

# Verify
python -c "import cml; cml.init(); print('OK'); cml.cleanup()"
```

### Hello World

```python
import cml

cml.init()

# Create tensors
x = cml.randn([10, 20])
y = cml.zeros([10, 20])

# Operations
z = x + y
result = z.sum()

cml.cleanup()
```

### Train a Neural Network

```python
import cml

cml.init()
cml.seed(42)

# Build model
model = cml.Sequential()
model.add(cml.Linear(784, 128))
model.add(cml.ReLU())
model.add(cml.Dropout(0.2))
model.add(cml.Linear(128, 10))

# Create optimizer
optimizer = cml.Adam(model, lr=0.001)

# Data
X = cml.randn([32, 784])
y = cml.zeros([32, 10])

# Training
model.set_training(True)
for epoch in range(10):
    optimizer.zero_grad()
    output = model(X)
    loss = cml.cross_entropy_loss(output, y)
    cml.backward(loss)
    optimizer.step()

cml.cleanup()
```

## API Reference

### Core Functions

| Function | Purpose |
|----------|---------|
| `cml.init()` | Initialize CML |
| `cml.cleanup()` | Clean up resources |
| `cml.seed(42)` | Set random seed |
| `cml.set_device(device)` | Set execution device |
| `cml.set_dtype(dtype)` | Set default data type |

### Tensor Creation

| Function | Purpose |
|----------|---------|
| `cml.zeros(shape)` | Zero tensor |
| `cml.ones(shape)` | Ones tensor |
| `cml.randn(shape)` | Random normal N(0,1) |
| `cml.rand(shape)` | Random uniform U(0,1) |
| `cml.full(shape, value)` | Filled tensor |
| `cml.clone(tensor)` | Clone tensor |

### Tensor Operations

**Arithmetic:**
```python
z = x + y        # Addition
z = x - y        # Subtraction
z = x * y        # Element-wise multiply
z = x / y        # Element-wise divide
z = x @ y        # Matrix multiply
```

**Activation:**
```python
y = x.relu()     # ReLU
y = x.sigmoid()  # Sigmoid
y = x.tanh()     # Tanh
y = x.softmax(dim=1)  # Softmax
```

**Reduction:**
```python
s = x.sum()      # Sum
m = x.mean()     # Mean
```

### Neural Network Layers

| Layer | Purpose |
|-------|---------|
| `cml.Linear(in, out)` | Fully connected |
| `cml.Conv2d(...)` | 2D Convolution |
| `cml.BatchNorm2d(features)` | Batch normalization |
| `cml.Dropout(p)` | Dropout regularization |
| `cml.ReLU()` | ReLU activation |
| `cml.Sigmoid()` | Sigmoid activation |
| `cml.MaxPool2d(k)` | Max pooling |
| `cml.AvgPool2d(k)` | Average pooling |

### Loss Functions

| Loss | Purpose |
|------|---------|
| `cml.mse_loss(pred, target)` | Mean squared error |
| `cml.mae_loss(pred, target)` | Mean absolute error |
| `cml.cross_entropy_loss(logits, labels)` | Classification |
| `cml.bce_loss(pred, target)` | Binary classification |
| `cml.huber_loss(pred, target)` | Robust regression |
| `cml.kl_divergence(p, q)` | Distribution divergence |

### Optimizers

| Optimizer | Purpose |
|-----------|---------|
| `cml.Adam(model, lr)` | Adaptive momentum |
| `cml.SGD(model, lr)` | Stochastic gradient descent |
| `cml.RMSprop(model, lr)` | Root mean square prop |
| `cml.AdaGrad(model, lr)` | Adaptive gradient |

### Autograd

```python
# Forward pass
output = model(x)
loss = cml.mse_loss(output, target)

# Backward pass
cml.backward(loss)

# Access gradients
grad = cml.get_grad(parameter)
```

## Examples

The `python/examples/` directory includes:

1. **01_hello_cml.py** - Simple tensor operations
2. **02_neural_network.py** - Training loop example
3. **03_classification.py** - Multi-class classification
4. **04_convolution.py** - Convolutional networks

Run examples:

```bash
cd python/examples
python 01_hello_cml.py
python 02_neural_network.py
```

## Files Structure

```
python/
├── cml/                    # Main package
│   ├── __init__.py        # Package initialization
│   ├── _cml_cffi.py       # CFFI declarations
│   ├── build_cffi.py      # Build script
│   ├── core.py            # Tensor class
│   ├── tensor_ops.py      # Tensor creation
│   ├── autograd.py        # Automatic differentiation
│   ├── nn.py              # Neural network layers
│   ├── losses.py          # Loss functions
│   ├── optim.py           # Optimizers
│   └── utils.py           # Utilities
├── examples/              # Example scripts
│   ├── 01_hello_cml.py
│   ├── 02_neural_network.py
│   ├── 03_classification.py
│   └── 04_convolution.py
├── tests/                 # Unit tests
├── setup.py               # Installation script
├── README.md              # Full documentation
├── INSTALLATION.md        # Installation guide
└── CFFI_DEVELOPMENT.md    # Developer guide
```

## Installation Paths

### Quick Install

```bash
cd python
pip install cffi
python setup.py install
```

### Development Mode

```bash
cd python
pip install -e .
```

### From Source

```bash
# Ensure C library is built
cd ..
make

# Build Python bindings
cd python
python cml/build_cffi.py
```

### Virtual Environment (Recommended)

```bash
# Create environment
python3 -m venv cml_env
source cml_env/bin/activate

# Install
cd python
pip install cffi
python setup.py install
```

## Device Support

### CPU (Default)

```python
import cml
cml.init()
cml.set_device(cml.DEVICE_CPU)
```

### NVIDIA GPU

```python
if cml.is_device_available(cml.DEVICE_CUDA):
    cml.set_device(cml.DEVICE_CUDA)
```

### Apple Metal

```python
if cml.is_device_available(cml.DEVICE_METAL):
    cml.set_device(cml.DEVICE_METAL)
```

## Troubleshooting

### Import Error

```python
ImportError: CML bindings not found
```

**Solution:**

```bash
cd python
python cml/build_cffi.py
```

### Library Not Found

```
OSError: cannot find libcml.so
```

**Solution:**

```bash
export LD_LIBRARY_PATH=../../build/lib:$LD_LIBRARY_PATH
python script.py
```

### Version Mismatch

Rebuild both C library and Python bindings:

```bash
cd ..
make clean && make
cd python
python cml/build_cffi.py
```

## Documentation

- **README.md**: Full API documentation and usage
- **INSTALLATION.md**: Detailed installation instructions
- **CFFI_DEVELOPMENT.md**: Guide for extending bindings
- **examples/**: Working example scripts
- **../new_docs/**: CML C library documentation

## Architecture

```
Python Code
    ↓
CML Python Package (cml/)
    ├─ Wrapper Classes
    ├─ Memory Management
    └─ API Conversion
    ↓
CFFI Library (_cml_lib.so)
    ├─ Function Declarations
    └─ Type Mappings
    ↓
CML C Library
    ├─ Tensor Operations
    ├─ Neural Networks
    ├─ Optimizers
    └─ MLIR Backend
    ↓
Hardware
    ├─ CPU
    ├─ CUDA
    ├─ Metal
    └─ ROCm
```

## Performance Tips

1. **Batch Operations**: Process multiple samples at once
2. **Device Selection**: Use GPU for large computations
3. **Data Types**: Use float32 unless precision is critical
4. **Memory**: Python garbage collection handles cleanup

## Common Patterns

### Training Loop

```python
model.set_training(True)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = loss_fn(output, y)
    cml.backward(loss)
    optimizer.step()
```

### Inference

```python
model.set_training(False)
with cml.no_grad():  # Optional: disable grad tracking
    output = model(X_test)
```

### Learning Rate Decay

```python
for epoch in range(num_epochs):
    if (epoch + 1) % 10 == 0:
        new_lr = old_lr * 0.1
        optimizer.set_lr(new_lr)
    # training...
```

## Extensions

The bindings can be extended by:

1. Adding CFFI declarations to `_cml_cffi.py`
2. Creating wrapper classes in appropriate modules
3. Exporting in `__init__.py`
4. See `CFFI_DEVELOPMENT.md` for details

## Requirements

- Python 3.8+
- CFFI 1.14.0+
- C11 compiler
- MLIR 18.x+
- C-ML library

## License

MIT License - Same as C-ML

## Support

- **Issues**: GitHub Issues page
- **Examples**: See `examples/` directory
- **Documentation**: See README.md and INSTALLATION.md
- **Development**: See CFFI_DEVELOPMENT.md

## Next Steps

1. **Install**: Follow INSTALLATION.md
2. **Learn**: Read README.md and examples
3. **Develop**: Check CFFI_DEVELOPMENT.md to extend
4. **Deploy**: Package with your application

## Version

- **CML Python Bindings**: v0.0.2
- **CML C Library**: v0.0.2
- **Python Support**: 3.8+
- **Release Date**: January 2026
