# CFFI Development Guide

Guide for developers extending CML's CFFI bindings.

## Architecture

```
CML C Library (include/, src/)
    ↓
CFFI Layer
    ├─ _cml_cffi.py      (C function declarations)
    ├─ _cml_lib.so       (Compiled CFFI library)
    └─ build_cffi.py     (Build script)
    ↓
Python Wrapper Layer
    ├─ core.py           (Tensor, initialization)
    ├─ tensor_ops.py     (Tensor creation)
    ├─ autograd.py       (Backprop)
    ├─ nn.py             (Neural network layers)
    ├─ losses.py         (Loss functions)
    ├─ optim.py          (Optimizers)
    └─ utils.py          (Utilities)
    ↓
User Code
    └─ example.py        (Python applications)
```

## How CFFI Works

1. **Declarations** (`_cml_cffi.py`): Declare C functions and types
2. **Build** (`build_cffi.py`): Generate Python/C interface
3. **Wrapper** (e.g., `core.py`): Create Pythonic API
4. **Use**: Call from Python code

## Adding New Functions

### Step 1: Declare in `_cml_cffi.py`

Add the C function declaration in the `ffi.cdef()` block:

```python
# In _cml_cffi.py, ffi.cdef("""
Tensor* my_new_function(Tensor* input, float param);
# """)
```

### Step 2: Create Python Wrapper

Add wrapper in appropriate module (e.g., `tensor_ops.py`):

```python
# In tensor_ops.py
def my_new_function(tensor, param=0.5):
    """My new function description.

    Args:
        tensor: Input tensor
        param: Parameter value

    Returns:
        Result tensor

    Example:
        >>> result = my_new_function(x, param=0.3)
    """
    result = lib.my_new_function(tensor._tensor, float(param))
    return Tensor(result)
```

### Step 3: Export in `__init__.py`

Add to imports and `__all__`:

```python
# In __init__.py
from cml.tensor_ops import my_new_function

__all__ = [
    # ... existing exports ...
    "my_new_function",
]
```

### Step 4: Rebuild

```bash
python3 build_cffi.py
# or
python3 setup.py build_ext --inplace
```

### Step 5: Test

```python
import cml
cml.init()
result = cml.my_new_function(cml.randn([10, 10]))
cml.cleanup()
```

## Adding New Layer Types

### Step 1: Declare Layer in `_cml_cffi.py`

```python
# Add to ffi.cdef()
typedef struct MyLayer MyLayer;
MyLayer* cml_nn_my_layer(int param1, int param2, ...);
```

### Step 2: Create Python Wrapper in `nn.py`

```python
class MyLayer(Module):
    """My custom layer.

    Description of what layer does.

    Example:
        >>> layer = MyLayer(param1=10)
        >>> output = layer(input)
    """

    def __init__(self, param1, param2, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        """Initialize MyLayer.

        Args:
            param1: First parameter
            param2: Second parameter
            dtype: Data type
            device: Device type
        """
        layer = lib.cml_nn_my_layer(param1, param2, dtype, device)
        super().__init__(layer)
        self.param1 = param1
        self.param2 = param2
```

### Step 3: Export in `__init__.py`

```python
from cml.nn import MyLayer

__all__ = [
    # ... existing ...
    "MyLayer",
]
```

## Adding New Loss Functions

### Step 1: Declare in `_cml_cffi.py`

```python
# Add to ffi.cdef()
Tensor* cml_nn_my_loss(Tensor* predictions, Tensor* targets, float param);
```

### Step 2: Create Wrapper in `losses.py`

```python
def my_loss(predictions, targets, param=1.0):
    """My custom loss function.

    Description of loss function.

    Args:
        predictions: Predicted values
        targets: Target values
        param: Loss parameter

    Returns:
        Loss tensor

    Example:
        >>> loss = my_loss(output, target)
    """
    result = lib.cml_nn_my_loss(
        predictions._tensor,
        targets._tensor,
        float(param)
    )
    return Tensor(result)
```

### Step 3: Export in `__init__.py`

```python
from cml.losses import my_loss

__all__ = [
    # ... existing ...
    "my_loss",
]
```

## Adding New Optimizers

### Step 1: Declare in `_cml_cffi.py`

```python
# Add to ffi.cdef()
Optimizer* cml_optim_my_optimizer_for_model(
    Module* module, float lr, float param1, float param2
);
```

### Step 2: Create Wrapper in `optim.py`

```python
class MyOptimizer(Optimizer):
    """My custom optimizer.

    Description of optimizer.

    Example:
        >>> opt = MyOptimizer(model, lr=0.01)
    """

    def __init__(self, model, lr=0.01, param1=0.9, param2=0.999):
        """Initialize MyOptimizer.

        Args:
            model: Neural network module
            lr: Learning rate
            param1: First hyperparameter
            param2: Second hyperparameter
        """
        optimizer = lib.cml_optim_my_optimizer_for_model(
            model._module, lr, param1, param2
        )
        super().__init__(optimizer)
```

## Debugging Tips

### Enable Verbose Build

```bash
python3 build_cffi.py  # Shows compilation output
```

### Check Generated Code

Generated code is in `.so` file. To see what was generated:

```bash
python3 -c "from cml._cml_lib import ffi; print(dir(ffi))"
```

### Use Python Debugger

```python
import pdb
import cml

cml.init()
pdb.set_trace()
x = cml.randn([10, 10])
cml.cleanup()
```

### Check Memory Leaks

```bash
# Using valgrind (Linux)
valgrind --leak-check=full python3 script.py

# Using LeakTracer (macOS)
leaktrace python3 script.py
```

### Verify C Bindings

```python
from cml._cml_lib import lib, ffi

# Check if function exists
if hasattr(lib, 'tensor_randn'):
    print("Function found")

# Check types
print(ffi.typeof('Tensor'))
```

## Performance Optimization

### Minimize Python Overhead

Batch operations where possible:

```python
# Slow: Many function calls
for i in range(1000):
    x = cml.randn([10, 10])
    y = x + 1

# Better: Batch operations
x = cml.randn([1000, 10, 10])
y = x + 1
```

### Use Appropriate Data Types

```python
# Faster on most hardware
cml.set_dtype(cml.DTYPE_FLOAT32)

# More precise but slower
cml.set_dtype(cml.DTYPE_FLOAT64)
```

### Enable GPU if Available

```python
if cml.is_device_available(cml.DEVICE_CUDA):
    cml.set_device(cml.DEVICE_CUDA)
```

## Testing

### Unit Tests

Create tests in `tests/`:

```python
# tests/test_tensor_ops.py
import cml

def test_randn():
    """Test tensor_randn creation."""
    cml.init()
    x = cml.randn([10, 20])
    assert x.size == 200
    cml.cleanup()

def test_operations():
    """Test tensor operations."""
    cml.init()
    x = cml.randn([10, 10])
    y = cml.zeros([10, 10])
    z = x + y
    cml.cleanup()
```

### Run Tests

```bash
pytest tests/
pytest tests/test_tensor_ops.py -v
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def my_function(arg1, arg2):
    """Brief description.

    Longer description if needed. Can span multiple lines.
    Explain purpose, behavior, and important notes.

    Args:
        arg1: Description of first argument
        arg2: Description of second argument

    Returns:
        Description of return value

    Raises:
        ValueError: When something is wrong

    Example:
        >>> result = my_function(10, 20)
        >>> print(result)
        30
    """
    pass
```

### Update README

Update `README.md` when adding public API:

```markdown
### New Feature

Brief description.

```python
result = cml.new_feature(arg1, arg2)
```

See `examples/` for full examples.
```

## Common Issues

### Issue: "ffi.new() failed: expected int, got float"

**Cause**: Type mismatch in CFFI call

**Fix**: Cast to correct type:

```python
# Wrong
shape_array = [10, 20]

# Right
shape_array = ffi.new("int[]", [10, 20])
```

### Issue: Segmentation fault

**Cause**: Memory management error

**Fix**:
- Check tensor is not freed prematurely
- Ensure all pointers are valid
- Use valgrind to debug

### Issue: Double-free at process exit

**Cause**: Resource freed manually and then also freed during cleanup

**Fix**: CML now tracks resources properly:
- Optimizers: `cml_untrack_optimizer()` is called automatically in `optimizer_free()`
- Modules: `cml_untrack_module()` is called automatically in `module_free()`
- IR Context: Reset at cleanup to detach all tensors

### Issue: "cffi library not found"

**Cause**: Build not run or failed

**Fix**:

```bash
python3 build_cffi.py  # Check for errors
```

## Architecture Decisions

### Why CFFI?

- **Simple**: Minimal overhead
- **Compatible**: Works with older Python versions
- **Flexible**: Can add custom C code if needed
- **Portable**: Works across platforms

### Why Wrapper Classes?

- **Pythonic**: Natural Python API
- **Automatic cleanup**: Reference counting
- **Type safety**: More maintainable code
- **Documentation**: docstrings for IDE help

### Why Separate Modules?

- **Organization**: Logical grouping
- **Maintainability**: Easier to navigate
- **Scalability**: Can grow independently
- **Usability**: Import only what you need

## Resource Tracking System

CML tracks resources to ensure proper cleanup at exit:

### Tracked Resources

| Resource | Track Function | Untrack Function | Auto-cleanup |
|----------|----------------|------------------|--------------|
| Modules | `cml_track_module()` | `cml_untrack_module()` | Yes |
| Optimizers | `cml_track_optimizer()` | `cml_untrack_optimizer()` | Yes |
| Datasets | `cml_track_dataset()` | N/A | Yes |

### How It Works

1. When a resource is created, it's added to a tracking list
2. When explicitly freed, it's removed from the tracking list
3. At process exit, `cml_auto_cleanup()` frees remaining tracked resources
4. IR context is reset first to detach all lazy tensors

### Best Practices

```python
# Option 1: Let auto-cleanup handle everything
cml.init()
model = cml.Sequential()
# ... use model ...
# At exit, model is automatically freed

# Option 2: Explicit cleanup (also works)
cml.init()
model = cml.Sequential()
optimizer = cml.Adam(model, lr=0.001)
# ... use them ...
lib.optimizer_free(optimizer._optimizer)  # Automatically untracked
# model will be freed at exit
```

## Future Improvements

- [x] NumPy array integration (buffer protocol) - **v0.0.4**
  - `Tensor.numpy()` with proper shape handling
  - `Tensor.from_numpy()` class method
  - `__array__` protocol for seamless `np.array(tensor)`
- [x] torch.Tensor-like behavior - **v0.0.4**
  - `.item()` for scalar tensors
  - `.requires_grad_()` in-place setter
  - Negative indexing support
  - `__repr__` and `__str__` methods
  - `.view()`, `.flatten()`, `.squeeze()`, `.unsqueeze()`
  - `.contiguous()` method
- [x] Context managers for automatic cleanup - **v0.0.4**
  - `cml.init_context()` for RAII-style init/cleanup
  - `cml.no_grad()` to disable gradients
  - `cml.enable_grad()` to re-enable gradients
  - `cml.set_grad_enabled(mode)` for explicit control
- [x] Type hints and stubs for IDE support - **v0.0.4**
  - Full `.pyi` stub file with type annotations
  - IDE autocompletion and type checking support
- [x] Lazy evaluation optimization
- [ ] Distributed tensor support
- [x] Comprehensive test suite (63 tests)

## References

- CFFI Documentation: https://cffi.readthedocs.io/
- Python Bindings Guide: https://docs.python.org/3/extending/
- Memory Management: Python C API docs
- Performance: https://cffi.readthedocs.io/en/latest/cdef.html

## Getting Help

- Check existing code for examples
- Review CFFI documentation
- Test with simple cases first
- Use Python debugger for issues
- File issues with complete examples
