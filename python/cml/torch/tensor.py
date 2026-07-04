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
