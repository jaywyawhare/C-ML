"""Tensor creation functions."""

from cml._cml_lib import ffi, lib
from cml.core import Tensor


def zeros(shape):
    if isinstance(shape, int):
        shape = [shape]
    shape_array = ffi.new("int[]", shape)
    tensor = lib.tensor_zeros(shape_array, len(shape), ffi.NULL)
    return Tensor(tensor)


def ones(shape):
    if isinstance(shape, int):
        shape = [shape]
    shape_array = ffi.new("int[]", shape)
    tensor = lib.tensor_ones(shape_array, len(shape), ffi.NULL)
    return Tensor(tensor)


def randn(shape):
    if isinstance(shape, int):
        shape = [shape]
    shape_array = ffi.new("int[]", shape)
    tensor = lib.tensor_randn(shape_array, len(shape), ffi.NULL)
    return Tensor(tensor)


def rand(shape):
    if isinstance(shape, int):
        shape = [shape]
    shape_array = ffi.new("int[]", shape)
    tensor = lib.tensor_rand(shape_array, len(shape), ffi.NULL)
    return Tensor(tensor)


def full(shape, value):
    if isinstance(shape, int):
        shape = [shape]
    shape_array = ffi.new("int[]", shape)
    tensor = lib.tensor_full(shape_array, len(shape), float(value), ffi.NULL)
    return Tensor(tensor)


def clone(tensor):
    result = lib.tensor_clone(tensor._tensor, ffi.NULL)
    return Tensor(result)
