"""
Tensor creation and operation functions.
"""

from cml._cml_lib import ffi, lib
from cml.core import Tensor


def zeros(shape):
    """Create tensor filled with zeros.

    Args:
        shape: Tensor shape (int or list/tuple)

    Returns:
        Zero-filled tensor

    Example:
        >>> x = zeros([10, 20])
        >>> y = zeros((5, 5, 5))
    """
    if isinstance(shape, int):
        shape = [shape]
    shape_array = ffi.new("int[]", shape)
    tensor = lib.tensor_zeros(shape_array, len(shape), ffi.NULL)
    return Tensor(tensor)


def ones(shape):
    """Create tensor filled with ones.

    Args:
        shape: Tensor shape

    Returns:
        Ones-filled tensor

    Example:
        >>> x = ones([10, 20])
    """
    if isinstance(shape, int):
        shape = [shape]
    shape_array = ffi.new("int[]", shape)
    tensor = lib.tensor_ones(shape_array, len(shape), ffi.NULL)
    return Tensor(tensor)


def randn(shape):
    """Create tensor with random values from standard normal distribution N(0, 1).

    Args:
        shape: Tensor shape

    Returns:
        Random tensor

    Example:
        >>> x = randn([100, 100])  # 100x100 random matrix
        >>> y = randn([10, 20, 30])
    """
    if isinstance(shape, int):
        shape = [shape]
    shape_array = ffi.new("int[]", shape)
    tensor = lib.tensor_randn(shape_array, len(shape), ffi.NULL)
    return Tensor(tensor)


def rand(shape):
    """Create tensor with random values from uniform distribution U(0, 1).

    Args:
        shape: Tensor shape

    Returns:
        Random tensor

    Example:
        >>> x = rand([10, 10])
    """
    if isinstance(shape, int):
        shape = [shape]
    shape_array = ffi.new("int[]", shape)
    tensor = lib.tensor_rand(shape_array, len(shape), ffi.NULL)
    return Tensor(tensor)


def full(shape, value):
    """Create tensor filled with a specific value.

    Args:
        shape: Tensor shape
        value: Fill value

    Returns:
        Filled tensor

    Example:
        >>> x = full([3, 3], 5.0)  # 3x3 matrix of 5s
    """
    if isinstance(shape, int):
        shape = [shape]
    shape_array = ffi.new("int[]", shape)
    tensor = lib.tensor_full(shape_array, len(shape), float(value), ffi.NULL)
    return Tensor(tensor)


def clone(tensor):
    """Create a copy of a tensor.

    Args:
        tensor: Tensor to copy

    Returns:
        Cloned tensor

    Example:
        >>> x = randn([10, 10])
        >>> y = clone(x)  # Independent copy
    """
    result = lib.tensor_clone(tensor._tensor, ffi.NULL)
    return Tensor(result)
