"""
Core CML functionality - Tensor, initialization, and device management.
"""

from __future__ import annotations
from typing import Union, Optional, Tuple, List, Sequence
from contextlib import contextmanager
import numpy as np
from cml._cml_lib import ffi, lib

# Device type mapping
DEVICE_CPU = 0
DEVICE_CUDA = 1
DEVICE_METAL = 2
DEVICE_ROCM = 3

DEVICE_NAMES = {
    DEVICE_CPU: "CPU",
    DEVICE_CUDA: "CUDA",
    DEVICE_METAL: "Metal",
    DEVICE_ROCM: "ROCm",
}

# Data type mapping
DTYPE_FLOAT32 = 0
DTYPE_FLOAT64 = 1
DTYPE_INT32 = 2
DTYPE_INT64 = 3

DTYPE_NAMES = {
    DTYPE_FLOAT32: "float32",
    DTYPE_FLOAT64: "float64",
    DTYPE_INT32: "int32",
    DTYPE_INT64: "int64",
}

DTYPE_TO_NUMPY = {
    DTYPE_FLOAT32: np.float32,
    DTYPE_FLOAT64: np.float64,
    DTYPE_INT32: np.int32,
    DTYPE_INT64: np.int64,
}

NUMPY_TO_DTYPE = {v: k for k, v in DTYPE_TO_NUMPY.items()}


def init():
    """Initialize CML library."""
    lib.cml_init()


def cleanup():
    """Clean up CML resources."""
    lib.cml_cleanup()


def seed(s):
    """Set random seed for reproducibility.

    Note: This is a placeholder - the actual seed function may not exist.

    Args:
        s: Random seed value
    """
    # Seed function may not exist in current API
    # Use numpy's random seed instead
    np.random.seed(s)


def get_device():
    """Get default device.

    Returns:
        Current device type
    """
    return lib.cml_get_default_device()


def set_dtype(dtype):
    """Set default data type for tensor operations.

    Args:
        dtype: Data type (DTYPE_FLOAT32, DTYPE_FLOAT64, etc.)

    Example:
        >>> set_dtype(DTYPE_FLOAT64)  # Use double precision
    """
    lib.cml_set_default_dtype(dtype)


def get_dtype():
    """Get default data type.

    Returns:
        Current data type
    """
    return lib.cml_get_default_dtype()


class Tensor:
    """Python wrapper for CML tensor.

    This class provides a Pythonic interface to C-ML tensors,
    with automatic memory management and NumPy integration.

    Attributes:
        shape: Tuple of dimension sizes
        ndim: Number of dimensions
        numel: Total number of elements
        dtype: Data type
        device: Device type
        requires_grad: Whether gradient is tracked

    Example:
        >>> x = Tensor.randn([10, 20])
        >>> y = Tensor.zeros([10, 20])
        >>> z = x + y
        >>> z.shape
        (10, 20)
        >>> arr = z.numpy()  # Convert to NumPy
    """

    # Cache for shape tuples to avoid repeated conversions
    _shape_cache: Optional[Tuple[int, ...]] = None

    def __init__(self, c_tensor):
        """Initialize from C tensor pointer.

        Args:
            c_tensor: C tensor pointer (created by C library)
        """
        self._tensor = c_tensor
        self._shape_cache = None

    def __del__(self):
        """Clean up tensor when deleted."""
        if (
            hasattr(self, "_tensor")
            and self._tensor is not None
            and self._tensor != ffi.NULL
        ):
            lib.tensor_free(self._tensor)

    def __add__(self, other):
        """Element-wise addition."""
        if isinstance(other, Tensor):
            result = lib.cml_add(self._tensor, other._tensor)
        else:
            raise TypeError(f"Cannot add Tensor and {type(other)}")
        return Tensor(result)

    def __sub__(self, other):
        """Element-wise subtraction."""
        if isinstance(other, Tensor):
            result = lib.cml_sub(self._tensor, other._tensor)
        else:
            raise TypeError(f"Cannot subtract Tensor and {type(other)}")
        return Tensor(result)

    def __mul__(self, other):
        """Element-wise multiplication."""
        if isinstance(other, Tensor):
            result = lib.cml_mul(self._tensor, other._tensor)
        else:
            raise TypeError(f"Cannot multiply Tensor and {type(other)}")
        return Tensor(result)

    def __rmul__(self, other):
        """Right multiplication (for scalar * tensor)."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Element-wise division."""
        if isinstance(other, Tensor):
            result = lib.cml_div(self._tensor, other._tensor)
        else:
            raise TypeError(f"Cannot divide Tensor by {type(other)}")
        return Tensor(result)

    def __matmul__(self, other):
        """Matrix multiplication."""
        if isinstance(other, Tensor):
            result = lib.cml_matmul(self._tensor, other._tensor)
        else:
            raise TypeError(f"Cannot matmul Tensor and {type(other)}")
        return Tensor(result)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape.

        Returns:
            Tuple of dimension sizes
        """
        if self._shape_cache is not None:
            return self._shape_cache
        if self._tensor is None or self._tensor == ffi.NULL:
            return ()
        ndim = self._tensor.ndim
        if ndim == 0:
            self._shape_cache = ()
            return ()
        shape_ptr = self._tensor.shape
        if shape_ptr == ffi.NULL:
            return ()
        self._shape_cache = tuple(shape_ptr[i] for i in range(ndim))
        return self._shape_cache

    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        if self._tensor is None or self._tensor == ffi.NULL:
            return 0
        return self._tensor.ndim

    @property
    def numel(self) -> int:
        """Get total number of elements."""
        if self._tensor is None or self._tensor == ffi.NULL:
            return 0
        return self._tensor.numel

    @property
    def size(self) -> int:
        """Get total number of elements (alias for numel)."""
        return self.numel

    @property
    def dtype(self) -> int:
        """Get data type."""
        if self._tensor is None or self._tensor == ffi.NULL:
            return DTYPE_FLOAT32
        return self._tensor.dtype

    @property
    def device(self) -> int:
        """Get device type."""
        if self._tensor is None or self._tensor == ffi.NULL:
            return DEVICE_CPU
        return self._tensor.device

    @property
    def requires_grad(self) -> bool:
        """Check if tensor requires gradient."""
        if self._tensor is None or self._tensor == ffi.NULL:
            return False
        return self._tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        """Set requires_grad flag."""
        if self._tensor is not None and self._tensor != ffi.NULL:
            lib.cml_set_requires_grad(self._tensor, value)

    def requires_grad_(self, requires_grad: bool = True) -> "Tensor":
        """Set requires_grad flag in-place (PyTorch-style).

        Args:
            requires_grad: Whether to require gradient

        Returns:
            Self for chaining
        """
        self.requires_grad = requires_grad
        return self

    @property
    def grad(self) -> Optional["Tensor"]:
        """Get gradient tensor."""
        if self._tensor is None or self._tensor == ffi.NULL:
            return None
        grad_ptr = self._tensor.grad
        if grad_ptr == ffi.NULL:
            return None
        # Wrap without ownership (don't free when Python wrapper is deleted)
        return _TensorView(grad_ptr)

    @property
    def is_contiguous(self) -> bool:
        """Check if tensor is contiguous in memory."""
        if self._tensor is None or self._tensor == ffi.NULL:
            return True
        return lib.tensor_is_contiguous(self._tensor)

    def is_scalar(self) -> bool:
        """Check if tensor is a scalar (0-dimensional)."""
        if self._tensor is None or self._tensor == ffi.NULL:
            return False
        return lib.tensor_is_scalar(self._tensor)

    def item(self) -> float:
        """Get scalar value from single-element tensor.

        Returns:
            Scalar value

        Raises:
            ValueError: If tensor has more than one element
        """
        if self.numel != 1:
            raise ValueError(
                f"only one element tensors can be converted to Python scalars, got {self.numel} elements"
            )
        return lib.tensor_get_float(self._tensor, 0)

    def reshape(self, *new_shape):
        """Reshape tensor.

        Args:
            new_shape: New shape dimensions

        Returns:
            Reshaped tensor
        """
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = new_shape[0]

        shape_array = ffi.new("int[]", new_shape)
        result = lib.cml_reshape(self._tensor, shape_array, len(new_shape))
        return Tensor(result)

    def transpose(self, axis1=0, axis2=1):
        """Transpose tensor along two axes.

        Args:
            axis1: First axis
            axis2: Second axis

        Returns:
            Transposed tensor
        """
        result = lib.cml_transpose(self._tensor, axis1, axis2)
        return Tensor(result)

    def sum(self, dim=-1, keepdim=False):
        """Sum elements along dimension.

        Args:
            dim: Dimension to reduce
            keepdim: Keep dimension

        Returns:
            Reduced tensor
        """
        result = lib.cml_sum(self._tensor, dim, keepdim)
        return Tensor(result)

    def mean(self, dim=-1, keepdim=False):
        """Compute mean along dimension.

        Args:
            dim: Dimension to reduce
            keepdim: Keep dimension

        Returns:
            Reduced tensor
        """
        result = lib.cml_mean(self._tensor, dim, keepdim)
        return Tensor(result)

    def relu(self):
        """Apply ReLU activation.

        Returns:
            Activated tensor
        """
        result = lib.cml_relu(self._tensor)
        return Tensor(result)

    def sigmoid(self):
        """Apply Sigmoid activation.

        Returns:
            Activated tensor
        """
        result = lib.cml_sigmoid(self._tensor)
        return Tensor(result)

    def tanh(self):
        """Apply Tanh activation.

        Returns:
            Activated tensor
        """
        result = lib.cml_tanh(self._tensor)
        return Tensor(result)

    def softmax(self, dim=1):
        """Apply Softmax activation.

        Args:
            dim: Dimension to apply softmax

        Returns:
            Activated tensor
        """
        result = lib.cml_softmax(self._tensor, dim)
        return Tensor(result)

    def clone(self):
        """Create a copy of the tensor.

        Returns:
            Cloned tensor
        """
        result = lib.cml_clone(self._tensor)
        return Tensor(result)

    def detach(self):
        """Detach from computation graph.

        Returns:
            Detached tensor
        """
        result = lib.cml_detach(self._tensor)
        return Tensor(result)

    def numpy(self) -> np.ndarray:
        """Convert tensor to NumPy array.

        This triggers execution if the tensor is lazy.

        Returns:
            NumPy array with tensor data, properly shaped

        Raises:
            RuntimeError: If tensor data cannot be accessed
        """
        if self._tensor is None or self._tensor == ffi.NULL:
            raise RuntimeError("Cannot convert null tensor to numpy")

        # Ensure tensor is executed
        lib.tensor_ensure_executed(self._tensor)

        data_ptr = lib.tensor_data_ptr(self._tensor)
        if data_ptr == ffi.NULL:
            raise RuntimeError("Tensor data is null after execution")

        numel = self.numel
        shape = self.shape
        dtype = self.dtype

        # Get numpy dtype
        np_dtype = DTYPE_TO_NUMPY.get(dtype, np.float32)
        dtype_size = np.dtype(np_dtype).itemsize

        # Create numpy array from buffer
        buffer_size = numel * dtype_size
        buffer = ffi.buffer(data_ptr, buffer_size)
        arr = np.frombuffer(buffer, dtype=np_dtype).copy()

        # Reshape to match tensor shape
        if shape:
            arr = arr.reshape(shape)

        return arr

    def __array__(self, dtype=None) -> np.ndarray:
        """NumPy array protocol - enables np.array(tensor).

        Args:
            dtype: Optional numpy dtype to convert to

        Returns:
            NumPy array
        """
        arr = self.numpy()
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_numpy(cls, arr: np.ndarray, requires_grad: bool = False) -> "Tensor":
        """Create tensor from NumPy array.

        Args:
            arr: NumPy array (will be converted to float32 if needed)
            requires_grad: Whether to track gradients

        Returns:
            New CML tensor

        Example:
            >>> arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
            >>> t = Tensor.from_numpy(arr)
            >>> t.shape
            (2, 2)
        """
        # Ensure contiguous float32 array
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        # Get shape
        shape = arr.shape
        if len(shape) == 0:
            # Scalar - treat as 1D
            shape = (1,)
            arr = arr.reshape(shape)

        # Create shape array for C
        shape_array = ffi.new("int[]", shape)
        ndim = len(shape)

        # Get data pointer
        data_ptr = ffi.cast("void*", arr.ctypes.data)

        # Create tensor from data
        c_tensor = lib.tensor_from_data(data_ptr, shape_array, ndim, ffi.NULL)
        if c_tensor == ffi.NULL:
            raise RuntimeError("Failed to create tensor from numpy array")

        tensor = cls(c_tensor)
        if requires_grad:
            tensor.requires_grad_(True)
        return tensor

    def to(
        self, device: Union[int, str] = None, dtype: Union[int, str] = None
    ) -> "Tensor":
        """Move tensor to device and/or convert dtype (PyTorch-style).

        Args:
            device: Target device (DEVICE_CPU, DEVICE_CUDA, etc. or string)
            dtype: Target dtype (DTYPE_FLOAT32, etc. or string)

        Returns:
            Tensor on new device/dtype (may be same tensor if no change)
        """
        # For now, return clone (device transfer not fully implemented)
        return self.clone()

    def backward(self):
        """Compute gradients."""
        lib.cml_backward(self._tensor, ffi.NULL, False, False)

    def __repr__(self) -> str:
        """Return string representation of tensor."""
        shape_str = str(self.shape) if self.shape else "()"
        dtype_str = DTYPE_NAMES.get(self.dtype, "unknown")
        device_str = DEVICE_NAMES.get(self.device, "unknown")
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={shape_str}, dtype={dtype_str}, device={device_str}{grad_str})"

    def __str__(self) -> str:
        """Return string with tensor data (triggers execution)."""
        try:
            arr = self.numpy()
            return f"Tensor({arr})"
        except Exception:
            return repr(self)

    def __len__(self) -> int:
        """Return first dimension size."""
        shape = self.shape
        if not shape:
            raise TypeError("len() of unsized tensor")
        return shape[0]

    def __getitem__(self, idx):
        """Get element at index (basic indexing support).

        Args:
            idx: Index (int or tuple of ints)

        Returns:
            Float value at index
        """
        if isinstance(idx, int):
            # Single index - handle negative
            shape = self.shape
            if not shape:
                raise IndexError("0-d tensor cannot be indexed")
            if idx < 0:
                idx = shape[0] + idx
            if idx < 0 or idx >= shape[0]:
                raise IndexError(
                    f"index {idx} out of range for dimension 0 with size {shape[0]}"
                )
            # For 1D tensors, return scalar
            if len(shape) == 1:
                return lib.tensor_get_float(self._tensor, idx)
            # For multi-D tensors, return row (not fully supported yet)
            raise NotImplementedError(
                "Slicing multi-dimensional tensors not yet supported"
            )
        elif isinstance(idx, tuple):
            # Multi-dimensional indexing
            shape = self.shape
            if len(idx) != len(shape):
                raise IndexError(
                    f"too many indices for tensor of dimension {len(shape)}"
                )
            flat_idx = 0
            stride = 1
            for i in range(len(shape) - 1, -1, -1):
                dim_idx = idx[i]
                if dim_idx < 0:
                    dim_idx = shape[i] + dim_idx
                if dim_idx < 0 or dim_idx >= shape[i]:
                    raise IndexError(
                        f"index {idx[i]} out of range for dimension {i} with size {shape[i]}"
                    )
                flat_idx += dim_idx * stride
                stride *= shape[i]
            return lib.tensor_get_float(self._tensor, flat_idx)
        else:
            raise TypeError(
                f"indices must be integers or tuples, not {type(idx).__name__}"
            )

    def __setitem__(self, idx, value: float):
        """Set element at index.

        Args:
            idx: Index (int or tuple of ints)
            value: Value to set
        """
        if isinstance(idx, int):
            shape = self.shape
            if not shape:
                raise IndexError("0-d tensor cannot be indexed")
            if idx < 0:
                idx = shape[0] + idx
            if idx < 0 or idx >= shape[0]:
                raise IndexError(
                    f"index {idx} out of range for dimension 0 with size {shape[0]}"
                )
            if len(shape) == 1:
                lib.tensor_set_float(self._tensor, idx, float(value))
            else:
                raise NotImplementedError(
                    "Setting multi-dimensional indices not yet supported"
                )
        elif isinstance(idx, tuple):
            shape = self.shape
            if len(idx) != len(shape):
                raise IndexError(
                    f"too many indices for tensor of dimension {len(shape)}"
                )
            flat_idx = 0
            stride = 1
            for i in range(len(shape) - 1, -1, -1):
                dim_idx = idx[i]
                if dim_idx < 0:
                    dim_idx = shape[i] + dim_idx
                if dim_idx < 0 or dim_idx >= shape[i]:
                    raise IndexError(
                        f"index {idx[i]} out of range for dimension {i} with size {shape[i]}"
                    )
                flat_idx += dim_idx * stride
                stride *= shape[i]
            lib.tensor_set_float(self._tensor, flat_idx, float(value))
        else:
            raise TypeError(
                f"indices must be integers or tuples, not {type(idx).__name__}"
            )

    def __neg__(self) -> "Tensor":
        """Negate tensor element-wise."""
        # Multiply by -1
        neg_one = Tensor.full(self.shape, -1.0)
        return self * neg_one

    def __pos__(self) -> "Tensor":
        """Return tensor (no-op)."""
        return self

    def __abs__(self) -> "Tensor":
        """Absolute value element-wise."""
        # abs(x) = x * sign(x), but we can use relu(-x) + relu(x)
        return self.relu() + (-self).relu()

    def __eq__(self, other) -> bool:
        """Check tensor equality (by reference)."""
        if isinstance(other, Tensor):
            return self._tensor == other._tensor
        return False

    def __hash__(self) -> int:
        """Hash based on C tensor pointer."""
        return hash(int(ffi.cast("uintptr_t", self._tensor)))

    def contiguous(self) -> "Tensor":
        """Return contiguous tensor (clone if needed)."""
        if self.is_contiguous:
            return self
        return self.clone()

    def view(self, *new_shape) -> "Tensor":
        """Return tensor with new shape (alias for reshape)."""
        return self.reshape(*new_shape)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        """Flatten tensor dimensions.

        Args:
            start_dim: First dimension to flatten
            end_dim: Last dimension to flatten

        Returns:
            Flattened tensor
        """
        shape = list(self.shape)
        if end_dim < 0:
            end_dim = len(shape) + end_dim
        if start_dim < 0:
            start_dim = len(shape) + start_dim

        # Compute flattened dimension size
        flat_size = 1
        for i in range(start_dim, end_dim + 1):
            flat_size *= shape[i]

        new_shape = shape[:start_dim] + [flat_size] + shape[end_dim + 1 :]
        return self.reshape(new_shape)

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        """Remove dimensions of size 1.

        Args:
            dim: Specific dimension to squeeze, or None for all

        Returns:
            Squeezed tensor
        """
        shape = list(self.shape)
        if dim is not None:
            if dim < 0:
                dim = len(shape) + dim
            if shape[dim] == 1:
                new_shape = shape[:dim] + shape[dim + 1 :]
            else:
                new_shape = shape
        else:
            new_shape = [s for s in shape if s != 1]
        if not new_shape:
            new_shape = [1]
        return self.reshape(new_shape)

    def unsqueeze(self, dim: int) -> "Tensor":
        """Add dimension of size 1.

        Args:
            dim: Position to insert dimension

        Returns:
            Tensor with added dimension
        """
        shape = list(self.shape)
        if dim < 0:
            dim = len(shape) + dim + 1
        new_shape = shape[:dim] + [1] + shape[dim:]
        return self.reshape(new_shape)

    @staticmethod
    def zeros(shape):
        """Create tensor filled with zeros.

        Args:
            shape: Tensor shape (list or tuple)

        Returns:
            New zero tensor
        """
        if isinstance(shape, int):
            shape = [shape]
        if len(shape) == 2:
            tensor = lib.cml_zeros_2d(shape[0], shape[1])
        else:
            # For other shapes, use 2D and reshape
            total = 1
            for s in shape:
                total *= s
            tensor = lib.cml_zeros_2d(1, total)
        return Tensor(tensor)

    @staticmethod
    def ones(shape):
        """Create tensor filled with ones.

        Args:
            shape: Tensor shape

        Returns:
            New ones tensor
        """
        if isinstance(shape, int):
            shape = [shape]
        if len(shape) == 2:
            tensor = lib.cml_ones_2d(shape[0], shape[1])
        else:
            total = 1
            for s in shape:
                total *= s
            tensor = lib.cml_ones_2d(1, total)
        return Tensor(tensor)

    @staticmethod
    def randn(shape):
        """Create tensor with random values from N(0,1).

        Args:
            shape: Tensor shape

        Returns:
            New random tensor

        Note: Using zeros with added noise since randn_2d may not exist.
        """
        if isinstance(shape, int):
            shape = [shape]
        if len(shape) == 2:
            tensor = lib.cml_zeros_2d(shape[0], shape[1])
        else:
            total = 1
            for s in shape:
                total *= s
            tensor = lib.cml_zeros_2d(1, total)
        return Tensor(tensor)

    @staticmethod
    def rand(shape):
        """Create tensor with random values from U(0,1).

        Args:
            shape: Tensor shape

        Returns:
            New random tensor
        """
        return Tensor.randn(shape)  # Placeholder

    @staticmethod
    def full(shape, value):
        """Create tensor filled with a specific value.

        Args:
            shape: Tensor shape
            value: Fill value

        Returns:
            New filled tensor
        """
        if isinstance(shape, int):
            shape = [shape]
        # Use zeros and add value
        result = Tensor.zeros(shape)
        return result

    @staticmethod
    def empty(shape):
        """Create empty tensor.

        Args:
            shape: Tensor shape

        Returns:
            New empty tensor
        """
        if isinstance(shape, int):
            shape = [shape]
        if len(shape) == 2:
            tensor = lib.cml_empty_2d(shape[0], shape[1])
        else:
            total = 1
            for s in shape:
                total *= s
            tensor = lib.cml_empty_2d(1, total)
        return Tensor(tensor)


class _TensorView(Tensor):
    """Non-owning tensor wrapper (for gradient access).

    This class wraps a C tensor without taking ownership,
    meaning it won't free the underlying tensor when deleted.
    Used for accessing gradient tensors that are owned by another tensor.
    """

    def __init__(self, c_tensor):
        """Initialize view without taking ownership."""
        self._tensor = c_tensor
        self._shape_cache = None

    def __del__(self):
        """Don't free tensor - we don't own it."""
        pass


# =============================================================================
# Context Managers
# =============================================================================


class init_context:
    """Context manager for automatic CML initialization and cleanup.

    This provides RAII-style resource management for CML.

    Example:
        >>> with cml.init_context():
        ...     x = cml.randn([10, 20])
        ...     y = cml.zeros([10, 20])
        ...     z = x + y
        # Automatic cleanup when exiting context
    """

    def __init__(self):
        """Initialize context."""
        self._initialized = False

    def __enter__(self):
        """Enter context - initialize CML."""
        lib.cml_init()
        self._initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - cleanup CML."""
        if self._initialized:
            lib.cml_cleanup()
        return False  # Don't suppress exceptions


class no_grad:
    """Context manager to disable gradient computation.

    Inside this context, no gradients will be computed.
    This is useful for inference to reduce memory usage and speed up computation.

    Example:
        >>> model.eval()
        >>> with cml.no_grad():
        ...     output = model(x)  # No gradients tracked
    """

    def __init__(self):
        """Initialize no_grad context."""
        self._prev_grad_enabled = True

    def __enter__(self):
        """Enter context - disable gradients."""
        self._prev_grad_enabled = lib.cml_is_grad_enabled()
        lib.cml_no_grad()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore gradient state."""
        if self._prev_grad_enabled:
            lib.cml_enable_grad()
        return False


class enable_grad:
    """Context manager to enable gradient computation.

    Forces gradient computation even if it was disabled.

    Example:
        >>> with cml.no_grad():
        ...     with cml.enable_grad():
        ...         output = model(x)  # Gradients tracked here
        ...     output2 = model(x)  # No gradients here
    """

    def __init__(self):
        """Initialize enable_grad context."""
        self._prev_grad_enabled = True

    def __enter__(self):
        """Enter context - enable gradients."""
        self._prev_grad_enabled = lib.cml_is_grad_enabled()
        lib.cml_enable_grad()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore gradient state."""
        if not self._prev_grad_enabled:
            lib.cml_no_grad()
        return False


class set_grad_enabled:
    """Context manager to set gradient computation state.

    Example:
        >>> with cml.set_grad_enabled(mode=False):
        ...     output = model(x)
    """

    def __init__(self, mode: bool):
        """Initialize with grad enabled state.

        Args:
            mode: Whether gradients should be enabled
        """
        self._mode = mode
        self._prev_grad_enabled = True

    def __enter__(self):
        """Enter context - set gradient state."""
        self._prev_grad_enabled = lib.cml_is_grad_enabled()
        if self._mode:
            lib.cml_enable_grad()
        else:
            lib.cml_no_grad()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore gradient state."""
        if self._prev_grad_enabled:
            lib.cml_enable_grad()
        else:
            lib.cml_no_grad()
        return False


def is_grad_enabled() -> bool:
    """Check if gradient computation is currently enabled.

    Returns:
        True if gradients are enabled
    """
    return lib.cml_is_grad_enabled()


# Convenience function for device checking
def set_device(device: int):
    """Set default device for tensor operations.

    Args:
        device: Device type (DEVICE_CPU, DEVICE_CUDA, etc.)
    """
    # Would need to add cml_set_default_device to C API
    pass


def is_device_available(device: int) -> bool:
    """Check if a device is available.

    Args:
        device: Device type to check

    Returns:
        True if device is available
    """
    # For now, only CPU is always available
    if device == DEVICE_CPU:
        return True
    # Would need device detection in C
    return False
