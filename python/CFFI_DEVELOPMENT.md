# CFFI Development Guide

How to extend the Python bindings.

## Architecture

```
include/*.h  →  _cml_cffi.py (cdef)  →  _cml_lib.so  →  Python wrappers  →  User code
```

| File | Role |
|------|------|
| `_cml_cffi.py` | C function/type declarations (CFFI cdef) |
| `build_cffi.py` | Compiles `_cml_lib.so` from cdef + C headers |
| `core.py` | `Tensor` class, init/cleanup, context managers |
| `nn.py` | Layer wrappers (`Linear`, `Conv2d`, etc.) |
| `optim.py` | Optimizer + LR scheduler wrappers |
| `losses.py` | Loss function wrappers |
| `tensor_ops.py` | Module-level creation functions (`zeros`, `randn`) |
| `autograd.py` | `backward`, `get_grad`, gradient control |

## Adding a New C Function

1. Add the declaration to `_cml_cffi.py` inside `ffi.cdef("""...""")`
2. Add the Python wrapper in the appropriate module
3. Rebuild: `python3 cml/build_cffi.py`

Example — wrapping `cml_nn_my_layer`:

```python
# _cml_cffi.py
MyLayer* cml_nn_my_layer(int param1, int param2, DType dtype, DeviceType device);

# nn.py
class MyLayer(Module):
    def __init__(self, param1, param2, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        super().__init__(lib.cml_nn_my_layer(param1, param2, dtype, device))
```

## Adding a New Optimizer

Only `Adam` and `SGD` have `_for_model` convenience constructors in C. All others require collecting parameters first:

```python
class MyOpt(Optimizer):
    def __init__(self, model, lr=0.01):
        params, num = _collect_parameters(model._module)
        super().__init__(lib.cml_optim_my_opt(params, num, lr))
```

See `optim.py` for `_collect_parameters` implementation.

## Debugging

```bash
python3 -c "from cml._cml_lib import lib; print(dir(lib))"   # list available C functions
export LD_LIBRARY_PATH=../build/lib:$LD_LIBRARY_PATH          # fix library path
python3 build_cffi.py                                          # rebuild bindings
```

## Rules

- Every C function in `cml.h` should have a declaration in `_cml_cffi.py`
- Python wrappers should match C signatures exactly (don't invent parameters)
- One `Tensor` class in `core.py` — no subclasses
- Submodules own their domain: `nn.py` for layers, `optim.py` for optimizers, etc.
- Top-level `cml.*` only re-exports the most common functions
