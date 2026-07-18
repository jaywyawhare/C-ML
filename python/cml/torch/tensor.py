"""Tensor helpers via torch_c C API."""

from __future__ import annotations
from typing import Sequence, Optional
from cml._cml_lib import ffi, lib
from cml.core import Tensor, DTYPE_FLOAT32, DEVICE_CPU


class TensorOptions:
    """PyTorch-style tensor construction options for the torch_c API.

    Attributes:
        dtype: Element type (e.g. ``DTYPE_FLOAT32``).
        device: Target device (e.g. ``DEVICE_CPU``).
        requires_grad: Whether created tensors should track gradients.
    """

    def __init__(self, dtype=DTYPE_FLOAT32, device=DEVICE_CPU, requires_grad=False):
        """Initialize options with dtype, device, and autograd flag."""
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad

    def _to_c_ptr(self):
        """Build a stable CFFI ``TorchTensorOptions`` pointer for native calls."""
        opts = ffi.new("TorchTensorOptions*")
        base = lib.torch_options()
        base = lib.torch_options_dtype(base, self.dtype)
        base = lib.torch_options_device(base, self.device)
        base = lib.torch_options_requires_grad(base, self.requires_grad)
        opts[0] = base
        return opts


def options(dtype=DTYPE_FLOAT32, device=DEVICE_CPU, requires_grad=False) -> TensorOptions:
    """Create a ``TensorOptions`` bundle for tensor factory functions.

    Args:
        dtype: Element type for new tensors.
        device: Device placement for new tensors.
        requires_grad: Whether tensors should require gradients.

    Returns:
        A ``TensorOptions`` instance passed to ``zeros``, ``ones``, etc.
    """
    return TensorOptions(dtype=dtype, device=device, requires_grad=requires_grad)


def _make_tensor(c_tensor) -> Tensor:
    """Wrap a native tensor pointer or raise if creation failed."""
    if c_tensor == ffi.NULL:
        raise RuntimeError("torch_c tensor creation failed")
    return Tensor(c_tensor)


def zeros(shape: Sequence[int], opts: Optional[TensorOptions] = None) -> Tensor:
    """Create a tensor filled with zeros.

    Args:
        shape: Dimensions of the output tensor.
        opts: Optional construction options; defaults to float32 CPU.

    Returns:
        A new ``Tensor`` with all elements set to zero.
    """
    opts = opts or TensorOptions()
    c_shape = ffi.new("int[]", list(shape))
    t = lib.torch_zeros(c_shape, len(shape), opts._to_c_ptr())
    return _make_tensor(t)


def ones(shape: Sequence[int], opts: Optional[TensorOptions] = None) -> Tensor:
    """Create a tensor filled with ones.

    Args:
        shape: Dimensions of the output tensor.
        opts: Optional construction options; defaults to float32 CPU.

    Returns:
        A new ``Tensor`` with all elements set to one.
    """
    opts = opts or TensorOptions()
    c_shape = ffi.new("int[]", list(shape))
    t = lib.torch_ones(c_shape, len(shape), opts._to_c_ptr())
    return _make_tensor(t)


def randn(shape: Sequence[int], opts: Optional[TensorOptions] = None) -> Tensor:
    """Create a tensor with elements drawn from a standard normal distribution.

    Args:
        shape: Dimensions of the output tensor.
        opts: Optional construction options; defaults to float32 CPU.

    Returns:
        A new ``Tensor`` with pseudo-random normal values.
    """
    opts = opts or TensorOptions()
    c_shape = ffi.new("int[]", list(shape))
    t = lib.torch_randn(c_shape, len(shape), opts._to_c_ptr())
    return _make_tensor(t)


def _shape_factory(c_fn):
    def factory(shape: Sequence[int], opts: Optional[TensorOptions] = None) -> Tensor:
        opts = opts or TensorOptions()
        c_shape = ffi.new("int[]", list(shape))
        return _make_tensor(c_fn(c_shape, len(shape), opts._to_c_ptr()))
    return factory


empty = _shape_factory(lib.torch_empty)
rand = _shape_factory(lib.torch_rand)


def full(shape: Sequence[int], value: float, opts: Optional[TensorOptions] = None) -> Tensor:
    """Create a tensor filled with `value`."""
    opts = opts or TensorOptions()
    c_shape = ffi.new("int[]", list(shape))
    return _make_tensor(lib.torch_full(c_shape, len(shape), opts._to_c_ptr(), float(value)))


def eye(n: int, opts: Optional[TensorOptions] = None) -> Tensor:
    """Create an n x n identity matrix."""
    opts = opts or TensorOptions()
    return _make_tensor(lib.torch_eye(n, opts._to_c_ptr()))


def arange(start: float, end: float, step: float = 1.0,
           opts: Optional[TensorOptions] = None) -> Tensor:
    """Create a 1-D tensor of values [start, end) with the given step."""
    opts = opts or TensorOptions()
    return _make_tensor(lib.torch_arange(start, end, step, opts._to_c_ptr()))


def linspace(start: float, end: float, steps: int,
             opts: Optional[TensorOptions] = None) -> Tensor:
    """Create a 1-D tensor of `steps` evenly spaced values in [start, end]."""
    opts = opts or TensorOptions()
    return _make_tensor(lib.torch_linspace(start, end, steps, opts._to_c_ptr()))


def zeros_like(t: Tensor) -> Tensor:
    return _make_tensor(lib.torch_zeros_like(t._tensor))


def ones_like(t: Tensor) -> Tensor:
    return _make_tensor(lib.torch_ones_like(t._tensor))


def randn_like(t: Tensor) -> Tensor:
    return _make_tensor(lib.torch_randn_like(t._tensor))


def _binary(c_fn):
    def op(a: Tensor, b: Tensor) -> Tensor:
        return _make_tensor(c_fn(a._tensor, b._tensor))
    return op


def _unary(c_fn):
    def op(a: Tensor) -> Tensor:
        return _make_tensor(c_fn(a._tensor))
    return op


add = _binary(lib.torch_add)
sub = _binary(lib.torch_sub)
mul = _binary(lib.torch_mul)
div = _binary(lib.torch_div)
matmul = _binary(lib.torch_matmul)
pow = _binary(lib.torch_pow)
relu = _unary(lib.torch_relu)
sigmoid = _unary(lib.torch_sigmoid)
tanh = _unary(lib.torch_tanh)
gelu = _unary(lib.torch_gelu)
clone = _unary(lib.torch_clone)
detach = _unary(lib.torch_detach)
contiguous = _unary(lib.torch_contiguous)


def softmax(a: Tensor, dim: int = -1) -> Tensor:
    return _make_tensor(lib.torch_softmax(a._tensor, dim))


def _reduction(c_fn):
    def op(a: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
        return _make_tensor(c_fn(a._tensor, dim, keepdim))
    return op


sum = _reduction(lib.torch_sum)
mean = _reduction(lib.torch_mean)
max = _reduction(lib.torch_max)
min = _reduction(lib.torch_min)


def reshape(a: Tensor, shape: Sequence[int]) -> Tensor:
    c_shape = ffi.new("int[]", list(shape))
    return _make_tensor(lib.torch_reshape(a._tensor, c_shape, len(shape)))


def transpose(a: Tensor, dim0: int = 0, dim1: int = 1) -> Tensor:
    return _make_tensor(lib.torch_transpose(a._tensor, dim0, dim1))


def squeeze(a: Tensor, dim: int = -1) -> Tensor:
    return _make_tensor(lib.torch_squeeze(a._tensor, dim))


def unsqueeze(a: Tensor, dim: int) -> Tensor:
    return _make_tensor(lib.torch_unsqueeze(a._tensor, dim))


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    arr = ffi.new("Tensor*[]", [t._tensor for t in tensors])
    return _make_tensor(lib.torch_cat(arr, len(tensors), dim))


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    arr = ffi.new("Tensor*[]", [t._tensor for t in tensors])
    return _make_tensor(lib.torch_stack(arr, len(tensors), dim))
