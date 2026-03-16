# CML Python Bindings

Python bindings for C-ML (C Machine Learning Library) using CFFI.

This enables you to use the high-performance C-ML library directly from Python with a Pythonic API.

## Installation

```bash
# Clone the repository
git clone https://github.com/jaywyawhare/C-ML.git
cd C-ML

# Build everything (C library + UI + Python bindings)
cd python
python build.py

# Install locally
pip install -e .
```

## Features

- **High Performance**: Uses C-ML's MLIR-based execution engine
- **Pythonic API**: Natural Python interface to C operations
- **Automatic Memory Management**: Reference counting handles cleanup with proper tracking
- **Full Feature Support**: All CML operations available in Python (63 CFFI tests passing)
- **Lazy Evaluation**: PyTorch-style lazy tensor operations for optimal performance
- **NumPy Integration**: Seamless conversion with `tensor.numpy()` and `Tensor.from_numpy()`
- **Type Hints**: Full IDE support with `.pyi` stub files
- **Context Managers**: RAII-style `init_context()`, `no_grad()`, `enable_grad()`
- **Visualization Dashboard**: Interactive graphs, training metrics, and kernel studio
- **Easy Installation**: Standard Python package installation

## Requirements

- Python 3.8+
- CFFI: `pip install cffi`
- C-ML library must be built first

## Installation

### Step 1: Build C-ML

Build the C library first from the project root:

```bash
cd ..  # Go to C-ML root
make   # or: mkdir build && cd build && cmake .. && make
```

### Step 2: Install Python Bindings

```bash
cd python
pip install cffi
python setup.py build_ext
python setup.py install

# Or for development:
pip install -e .
```

### Step 3: Verify Installation

```bash
python3 -c "import cml; cml.init(); print('CML Python bindings OK'); cml.cleanup()"
```

## Quick Start

### Hello CML

```python
import cml

cml.init()

# Create tensors
x = cml.randn([10, 20])
y = cml.zeros([10, 20])

# Tensor operations
z = x + y
result = z.sum()

cml.cleanup()
```

### Neural Network

```python
import cml

cml.init()
cml.seed(42)

# Create model
model = cml.Sequential()
model.add(cml.Linear(784, 128))
model.add(cml.ReLU())
model.add(cml.Dropout(0.2))
model.add(cml.Linear(128, 10))

# Create optimizer
optimizer = cml.Adam(model, lr=0.001)

# Create data
X = cml.randn([32, 784])
y = cml.zeros([32, 10])

# Training loop
model.set_training(True)
for epoch in range(10):
    optimizer.zero_grad()

    output = model(X)
    loss = cml.cross_entropy_loss(output, y)

    cml.backward(loss)
    optimizer.step()

    print(f"Epoch {epoch + 1}: Loss computed")

cml.cleanup()
```

## API Reference

### Core Functions

```python
cml.init()                          # Initialize CML
cml.cleanup()                       # Clean up resources
cml.seed(42)                        # Set random seed

# Device management
cml.set_device(cml.DEVICE_CUDA)     # Use GPU
cml.get_device()                    # Get current device

# Data type management
cml.set_dtype(cml.DTYPE_FLOAT64)    # Use double precision
```

### Tensor Creation

```python
x = cml.zeros([10, 20])             # Zero tensor
x = cml.ones([10, 20])              # Ones tensor
x = cml.randn([10, 20])             # Random normal
x = cml.rand([10, 20])              # Random uniform
x = cml.full([10, 20], 5.0)         # Filled with value
```

### Tensor Operations

```python
# Arithmetic
z = x + y
z = x - y
z = x * y
z = x / y
z = x @ y                           # Matrix multiplication

# Activation functions
y = x.relu()
y = x.sigmoid()
y = x.tanh()
y = x.softmax(dim=1)

# Reduction
s = x.sum()
m = x.mean()
```

### Neural Networks

```python
# Layers
cml.Linear(in_features, out_features)
cml.Conv2d(in_channels, out_channels, kernel_size)
cml.BatchNorm2d(num_features)
cml.Dropout(p=0.5)

# Activations
cml.ReLU()
cml.Sigmoid()
cml.Tanh()
cml.Softmax(dim=1)

# Pooling
cml.MaxPool2d(kernel_size=2)
cml.AvgPool2d(kernel_size=2)

# Container
model = cml.Sequential()
model.add(cml.Linear(10, 20))
model.add(cml.ReLU())
```

### Loss Functions

```python
cml.mse_loss(predictions, targets)
cml.mae_loss(predictions, targets)
cml.cross_entropy_loss(logits, labels)
cml.bce_loss(predictions, targets)
cml.huber_loss(predictions, targets)
cml.kl_divergence(p, q)
```

### Optimizers

```python
optimizer = cml.Adam(model, lr=0.001)
optimizer = cml.SGD(model, lr=0.01, momentum=0.9)
optimizer = cml.RMSprop(model, lr=0.001)
optimizer = cml.AdaGrad(model, lr=0.01)

optimizer.step()                    # Update parameters
optimizer.zero_grad()               # Clear gradients
```

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

## Visualization Dashboard

Launch an interactive dashboard to visualize your models:

```bash
# CLI
cml-viz

# Or from Python
from cml.viz import launch
launch(port=8001)
```

**Features:**
- **Ops Topology**: Interactive computational graph with dead code (red) and fused kernels (green)
- **Training Metrics**: Real-time loss/accuracy curves with early stopping detection
- **Kernel Studio**: Generated code for CUDA, Metal, OpenCL, WebGPU, SIMD backends

The dashboard reads `graph.json`, `training.json`, and `kernels.json` from the current directory.

## Examples

See `examples/` directory:

- `01_hello_cml.py` - Simple tensor operations
- `02_neural_network.py` - Training a neural network
- `03_classification.py` - Classification example
- `04_convolution.py` - Convolutional network example

Run examples:

```bash
python examples/01_hello_cml.py
python examples/02_neural_network.py
python examples/03_classification.py
python examples/04_convolution.py
```

## Device Selection

### CPU (Default)

```python
cml.set_device(cml.DEVICE_CPU)
```

### NVIDIA GPU

```python
if cml.is_device_available(cml.DEVICE_CUDA):
    cml.set_device(cml.DEVICE_CUDA)
```

### Apple Metal (macOS)

```python
if cml.is_device_available(cml.DEVICE_METAL):
    cml.set_device(cml.DEVICE_METAL)
```

### AMD ROCm

```python
if cml.is_device_available(cml.DEVICE_ROCM):
    cml.set_device(cml.DEVICE_ROCM)
```

## Troubleshooting

### CFFI Library Not Found

```
ImportError: CML bindings not found.
```

Solution:

```bash
# Build CFFI bindings
python cml/build_cffi.py

# Or reinstall
pip install -e .
```

### CML Library Not Found

```
OSError: cannot find libcml.so
```

Solution:

```bash
# Set library path
export LD_LIBRARY_PATH=../../build/lib:$LD_LIBRARY_PATH

# Then run your script
python your_script.py
```

### Version Mismatch

Ensure Python and C libraries are compatible:

```bash
# Rebuild both
cd ..
make clean && make

cd python
python setup.py build_ext --inplace
```

## Development

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

**Comprehensive CFFI Test Suite (63 tests):**

```bash
cd python
export LD_LIBRARY_PATH=../build/lib:$LD_LIBRARY_PATH
python3 test_cffi_complete.py
```

The test suite covers:
- Initialization and cleanup
- Device/dtype management
- Tensor creation and operations
- Neural network layers (Linear, ReLU, Sigmoid, etc.)
- Loss functions (MSE, MAE, BCE, Cross-Entropy, Huber, KL-Div)
- Optimizers (Adam, SGD)
- End-to-end training loops
- Memory management (no double-free issues)

**Unit Tests with pytest:**

```bash
pytest tests/
```

### Build Documentation

```bash
# See docs/ directory for CML documentation
```

## Building from Source

### Prerequisites

- Python 3.8+
- C11 compiler
- MLIR 18.x
- CMake 3.16+ or Make

### Build Steps

```bash
# Build C library
cd ..
make

# Build Python bindings
cd python
python setup.py build_ext --inplace

# Install
pip install -e .
```

## API Compatibility

The Python API mirrors the C API as closely as possible for consistency.

For functions not yet wrapped in Python, you can use CFFI directly:

```python
from cml._cml_lib import ffi, lib

# Call C function directly
tensor = lib.tensor_randn(shape_array, ndim, ffi.NULL)
```

## Performance Considerations

- **Tensor Creation**: Fast, uses C allocation
- **Operations**: Very fast, MLIR JIT compiled
- **GPU**: Use GPU-enabled build for significant speedup
- **Memory**: Automatic reference counting prevents leaks

## Known Limitations

1. **Shape access**: Tensor shape must be tracked separately in Python
2. **Advanced MLIR features**: Some MLIR optimizations may not be exposed
3. **GPU memory transfer**: Explicit device transfer may be needed

## Memory Management

The CFFI bindings use proper reference counting and resource tracking:

- **Tensor tracking**: Tensors are automatically managed with ref counting
- **Module tracking**: Neural network modules are tracked and cleaned up at exit
- **Optimizer tracking**: Optimizers are tracked and properly untracked when freed
- **IR context management**: Lazy evaluation IR is properly reset during cleanup

This ensures no double-free issues or memory leaks when using the Python bindings.

## Lazy Evaluation

CML uses PyTorch-style lazy evaluation where operations build a computation graph:

```python
# Operations are lazy - they build a graph
y = model(x)           # Creates IR graph nodes
loss = cml.mse_loss(y, target)  # More nodes added

# Execution happens on data access
cml.backward(loss)     # Triggers execution of forward + backward
```

This enables:
- **Operation fusion**: Multiple operations can be fused into optimized kernels
- **Memory optimization**: Intermediate buffers can be reused
- **Backend flexibility**: Same code works on CPU, CUDA, Metal, etc.

## NumPy Integration

Seamlessly convert between CML tensors and NumPy arrays:

```python
import cml
import numpy as np

cml.init()

# From NumPy to CML
arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
tensor = cml.Tensor.from_numpy(arr)
print(tensor.shape)  # (2, 2)

# From CML to NumPy (using np.array() protocol)
result = np.array(tensor)
print(result)  # [[1. 2.] [3. 4.]]

# Or explicitly
arr_copy = tensor.numpy()

cml.cleanup()
```

## Context Managers

Use context managers for clean resource management:

```python
import cml

# Automatic init/cleanup
with cml.init_context():
    x = cml.randn([10, 20])
    y = cml.zeros([10, 20])
    z = x + y
# Automatic cleanup here

# Disable gradients for inference
cml.init()
model = cml.Sequential()
# ... build model ...

with cml.no_grad():
    # No gradients tracked - faster inference
    output = model(x)

# Re-enable gradients for training
with cml.enable_grad():
    output = model(x)
    loss = cml.mse_loss(output, target)
    cml.backward(loss)

cml.cleanup()
```

## PyTorch-like API

CML provides a familiar PyTorch-like interface:

```python
import cml

cml.init()

# Tensor properties
x = cml.randn([3, 4])
print(x.shape)       # (3, 4)
print(x.ndim)        # 2
print(x.numel)       # 12
print(x.dtype)       # 0 (DTYPE_FLOAT32)

# Scalar tensors
scalar = cml.Tensor.from_numpy(np.array([3.14]))
print(scalar.item()) # 3.14

# In-place requires_grad
x.requires_grad_(True)
print(x.requires_grad)  # True

# Reshape operations
y = x.view(2, 6)
y = x.flatten()
y = x.unsqueeze(0)   # Add batch dimension
y = y.squeeze(0)     # Remove batch dimension

# Indexing (with negative indices)
val = x[0, -1]       # Last element of first row
x[1, 2] = 5.0        # Set value

cml.cleanup()
```

## Type Hints

CML includes full type stub files (`.pyi`) for IDE support:

```python
import cml

# IDE will show proper type hints
tensor: cml.Tensor = cml.randn([10, 20])  # Autocomplete works
result: cml.Tensor = tensor.relu()        # Return types known
arr: np.ndarray = tensor.numpy()          # Conversion types
```

## Contributing

Contributions welcome! Areas to improve:

- Shape tracking and manipulation
- NumPy array conversion
- Additional loss functions
- Performance optimizations

## License

DBaJ-GPL (Don't Be a Jerk General Public License) - See LICENSE file

TL;DR: Use it, modify it, share it. Just don't be a jerk about it.

## Support

- **Documentation**: https://jaywyawhare.github.io/C-ML
- **Examples**: See `examples/` folder
- **Issues**: https://github.com/jaywyawhare/C-ML/issues

## Changelog

### v0.0.4

- **NumPy array integration**:
  - `Tensor.from_numpy()` class method
  - `tensor.numpy()` with proper shape handling
  - `__array__` protocol for `np.array(tensor)`
- **PyTorch-like Tensor API**:
  - `.item()` for scalar tensors
  - `.requires_grad_()` in-place setter
  - Negative indexing with `tensor[-1]`
  - `__repr__` and `__str__` for printing
  - `.view()`, `.flatten()`, `.squeeze()`, `.unsqueeze()`
  - `.contiguous()` method
  - Direct property access: `.shape`, `.ndim`, `.numel`, `.dtype`, `.device`
- **Context managers**:
  - `cml.init_context()` for RAII-style init/cleanup
  - `cml.no_grad()` to disable gradient tracking
  - `cml.enable_grad()` to re-enable gradients
  - `cml.set_grad_enabled(mode)` for explicit control
- **Type hints and IDE support**:
  - Full `.pyi` stub file with type annotations
  - IDE autocompletion and type checking
- **API improvements**:
  - `is_grad_enabled()` function
  - `is_device_available()` function
  - Better error messages

### v0.0.3

- **63 CFFI tests passing** - Comprehensive test coverage
- **Fixed double-free issues** - Proper optimizer and module tracking
- **Lazy evaluation** - All operations now build IR graphs
- **UOP_GATHER implementation** - Fully lazy cross-entropy loss
- **Memory management improvements**:
  - Added `cml_untrack_optimizer` for proper cleanup
  - IR context properly detaches tensors before cleanup
  - Module free now calls specialized destructors

### v0.0.3

- Initial Python bindings
- CFFI implementation
- Basic API coverage
- Four example scripts

## References

- C-ML Documentation: `../docs/`
- CFFI Documentation: https://cffi.readthedocs.io/
