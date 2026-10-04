"""Tensor creation functions (module-level convenience wrappers)."""

from cml._cml_lib import ffi, lib
from cml.core import Tensor, _make_config


def zeros(shape, **kw):
    """Zero-filled tensor of the given shape (``torch.zeros``)."""
    return Tensor.zeros(shape, **kw)


def ones(shape, **kw):
    """One-filled tensor of the given shape (``torch.ones``)."""
    return Tensor.ones(shape, **kw)


def randn(shape, **kw):
    """Tensor drawn from the standard normal distribution (``torch.randn``)."""
    return Tensor.randn(shape, **kw)


def rand(shape, **kw):
    """Tensor drawn from the uniform ``[0, 1)`` distribution (``torch.rand``)."""
    return Tensor.rand(shape, **kw)


def full(shape, value, **kw):
    """Tensor of the given shape filled with ``value`` (``torch.full``)."""
    return Tensor.full(shape, value, **kw)


def empty(shape, **kw):
    """Uninitialized tensor of the given shape (``torch.empty``)."""
    return Tensor.empty(shape, **kw)


def clone(tensor):
    """Return a deep copy of ``tensor`` (``torch.clone``)."""
    return Tensor(lib.cml_clone(tensor._tensor))


def arange(start, end=None, step=1.0, dtype=None, device=None):
    """Evenly spaced values over ``[start, end)`` by ``step`` (``torch.arange``)."""
    return Tensor.arange(start, end, step, dtype=dtype, device=device)


def linspace(start, end, steps, dtype=None, device=None):
    """``steps`` evenly spaced values over the closed ``[start, end]`` (``torch.linspace``)."""
    return Tensor.linspace(start, end, steps, dtype=dtype, device=device)


def eye(n, dtype=None, device=None):
    """``n``-by-``n`` identity matrix (``torch.eye``)."""
    return Tensor.eye(n, dtype=dtype, device=device)


def randint(low, high, shape, dtype=None, device=None):
    """Tensor of random integers in ``[low, high)`` (``torch.randint``)."""
    if isinstance(shape, int):
        shape = [shape]
    shape_array = ffi.new("int[]", shape)
    config = _make_config(dtype, device)
    return Tensor(lib.cml_randint(int(low), int(high), shape_array, len(shape), config))


def randperm(n, dtype=None, device=None):
    """Random permutation of the integers ``0..n-1`` (``torch.randperm``)."""
    config = _make_config(dtype, device)
    return Tensor(lib.cml_randperm(int(n), config))


def manual_seed(seed):
    """Seed the global RNG for reproducible draws (``torch.manual_seed``)."""
    lib.cml_manual_seed(int(seed))


def zeros_like(tensor):
    """Zero-filled tensor matching ``tensor``'s shape/dtype (``torch.zeros_like``)."""
    return Tensor(lib.cml_zeros_like(tensor._tensor))


def ones_like(tensor):
    """One-filled tensor matching ``tensor``'s shape/dtype (``torch.ones_like``)."""
    return Tensor(lib.cml_ones_like(tensor._tensor))


def rand_like(tensor):
    """Uniform ``[0, 1)`` tensor matching ``tensor``'s shape/dtype (``torch.rand_like``)."""
    return Tensor(lib.cml_rand_like(tensor._tensor))


def randn_like(tensor):
    """Standard-normal tensor matching ``tensor``'s shape/dtype (``torch.randn_like``)."""
    return Tensor(lib.cml_randn_like(tensor._tensor))


def full_like(tensor, value):
    """Tensor matching ``tensor``'s shape/dtype filled with ``value`` (``torch.full_like``)."""
    return Tensor(lib.cml_full_like(tensor._tensor, float(value)))
