"""Tensor creation functions (module-level convenience wrappers)."""

from cml._cml_lib import ffi, lib
from cml.core import Tensor, _make_config


def zeros(shape, **kw): return Tensor.zeros(shape, **kw)
def ones(shape, **kw): return Tensor.ones(shape, **kw)
def randn(shape, **kw): return Tensor.randn(shape, **kw)
def rand(shape, **kw): return Tensor.rand(shape, **kw)
def full(shape, value, **kw): return Tensor.full(shape, value, **kw)
def empty(shape, **kw): return Tensor.empty(shape, **kw)


def clone(tensor):
    return Tensor(lib.cml_clone(tensor._tensor))


def arange(start, end=None, step=1.0, dtype=None, device=None):
    return Tensor.arange(start, end, step, dtype=dtype, device=device)


def linspace(start, end, steps, dtype=None, device=None):
    return Tensor.linspace(start, end, steps, dtype=dtype, device=device)


def eye(n, dtype=None, device=None):
    return Tensor.eye(n, dtype=dtype, device=device)


def randint(low, high, shape, dtype=None, device=None):
    if isinstance(shape, int):
        shape = [shape]
    shape_array = ffi.new("int[]", shape)
    config = _make_config(dtype, device)
    return Tensor(lib.cml_randint(int(low), int(high), shape_array, len(shape), config))


def randperm(n, dtype=None, device=None):
    config = _make_config(dtype, device)
    return Tensor(lib.cml_randperm(int(n), config))


def manual_seed(seed):
    lib.cml_manual_seed(int(seed))


def zeros_like(tensor): return Tensor(lib.cml_zeros_like(tensor._tensor))
def ones_like(tensor): return Tensor(lib.cml_ones_like(tensor._tensor))
def rand_like(tensor): return Tensor(lib.cml_rand_like(tensor._tensor))
def randn_like(tensor): return Tensor(lib.cml_randn_like(tensor._tensor))


def full_like(tensor, value):
    return Tensor(lib.cml_full_like(tensor._tensor, float(value)))
