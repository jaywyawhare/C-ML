"""
Core CML functionality - Tensor, initialization, and device management.
"""

from __future__ import annotations
from typing import Union, Optional, Tuple
import numpy as np
from cml._cml_lib import ffi, lib

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
    lib.cml_init()


def cleanup():
    lib.cml_cleanup()


def seed(s):
    lib.cml_seed(int(s))
    np.random.seed(s)


def get_device():
    return lib.cml_get_default_device()


def set_dtype(dtype):
    lib.cml_set_default_dtype(dtype)


def get_dtype():
    return lib.cml_get_default_dtype()


class Tensor:
    """Python wrapper for CML tensor with automatic memory management and NumPy integration."""

    _shape_cache: Optional[Tuple[int, ...]] = None

    def __init__(self, c_tensor):
        self._tensor = c_tensor
        self._shape_cache = None

    def __del__(self):
        if (
            hasattr(self, "_tensor")
            and self._tensor is not None
            and self._tensor != ffi.NULL
        ):
            lib.tensor_free(self._tensor)

    def __add__(self, other):
        if isinstance(other, Tensor):
            result = lib.cml_add(self._tensor, other._tensor)
        else:
            raise TypeError(f"Cannot add Tensor and {type(other)}")
        return Tensor(result)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            result = lib.cml_sub(self._tensor, other._tensor)
        else:
            raise TypeError(f"Cannot subtract Tensor and {type(other)}")
        return Tensor(result)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            result = lib.cml_mul(self._tensor, other._tensor)
        else:
            raise TypeError(f"Cannot multiply Tensor and {type(other)}")
        return Tensor(result)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            result = lib.cml_div(self._tensor, other._tensor)
        else:
            raise TypeError(f"Cannot divide Tensor by {type(other)}")
        return Tensor(result)

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            result = lib.cml_matmul(self._tensor, other._tensor)
        else:
            raise TypeError(f"Cannot matmul Tensor and {type(other)}")
        return Tensor(result)

    @property
    def shape(self) -> Tuple[int, ...]:
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
        if self._tensor is None or self._tensor == ffi.NULL:
            return 0
        return self._tensor.ndim

    @property
    def numel(self) -> int:
        if self._tensor is None or self._tensor == ffi.NULL:
            return 0
        return self._tensor.numel

    @property
    def size(self) -> int:
        return self.numel

    @property
    def dtype(self) -> int:
        if self._tensor is None or self._tensor == ffi.NULL:
            return DTYPE_FLOAT32
        return self._tensor.dtype

    @property
    def device(self) -> int:
        if self._tensor is None or self._tensor == ffi.NULL:
            return DEVICE_CPU
        return self._tensor.device

    @property
    def requires_grad(self) -> bool:
        if self._tensor is None or self._tensor == ffi.NULL:
            return False
        return self._tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        if self._tensor is not None and self._tensor != ffi.NULL:
            lib.cml_set_requires_grad(self._tensor, value)

    def requires_grad_(self, requires_grad: bool = True) -> "Tensor":
        self.requires_grad = requires_grad
        return self

    @property
    def grad(self) -> Optional["Tensor"]:
        if self._tensor is None or self._tensor == ffi.NULL:
            return None
        grad_ptr = self._tensor.grad
        if grad_ptr == ffi.NULL:
            return None
        return _TensorView(grad_ptr)

    @property
    def is_contiguous(self) -> bool:
        if self._tensor is None or self._tensor == ffi.NULL:
            return True
        return lib.tensor_is_contiguous(self._tensor)

    def is_scalar(self) -> bool:
        if self._tensor is None or self._tensor == ffi.NULL:
            return False
        return lib.tensor_is_scalar(self._tensor)

    def item(self) -> float:
        if self.numel != 1:
            raise ValueError(
                f"only one element tensors can be converted to Python scalars, got {self.numel} elements"
            )
        return lib.tensor_get_float(self._tensor, 0)

    def reshape(self, *new_shape):
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = new_shape[0]

        shape_array = ffi.new("int[]", new_shape)
        result = lib.cml_reshape(self._tensor, shape_array, len(new_shape))
        return Tensor(result)

    def transpose(self, axis1=0, axis2=1):
        result = lib.cml_transpose(self._tensor, axis1, axis2)
        return Tensor(result)

    def sum(self, dim=-1, keepdim=False):
        result = lib.cml_sum(self._tensor, dim, keepdim)
        return Tensor(result)

    def mean(self, dim=-1, keepdim=False):
        result = lib.cml_mean(self._tensor, dim, keepdim)
        return Tensor(result)

    def relu(self):
        result = lib.cml_relu(self._tensor)
        return Tensor(result)

    def sigmoid(self):
        result = lib.cml_sigmoid(self._tensor)
        return Tensor(result)

    def tanh(self):
        result = lib.cml_tanh(self._tensor)
        return Tensor(result)

    def softmax(self, dim=1):
        result = lib.cml_softmax(self._tensor, dim)
        return Tensor(result)

    def clone(self):
        result = lib.cml_clone(self._tensor)
        return Tensor(result)

    def detach(self):
        result = lib.cml_detach(self._tensor)
        return Tensor(result)

    def numpy(self) -> np.ndarray:
        """Triggers execution if the tensor is lazy."""
        if self._tensor is None or self._tensor == ffi.NULL:
            raise RuntimeError("Cannot convert null tensor to numpy")

        lib.tensor_ensure_executed(self._tensor)

        data_ptr = lib.tensor_data_ptr(self._tensor)
        if data_ptr == ffi.NULL:
            raise RuntimeError("Tensor data is null after execution")

        numel = self.numel
        shape = self.shape
        np_dtype = DTYPE_TO_NUMPY.get(self.dtype, np.float32)
        dtype_size = np.dtype(np_dtype).itemsize

        buffer = ffi.buffer(data_ptr, numel * dtype_size)
        arr = np.frombuffer(buffer, dtype=np_dtype).copy()

        if shape:
            arr = arr.reshape(shape)

        return arr

    def __array__(self, dtype=None) -> np.ndarray:
        arr = self.numpy()
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_numpy(cls, arr: np.ndarray, requires_grad: bool = False) -> "Tensor":
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        shape = arr.shape
        if len(shape) == 0:
            shape = (1,)
            arr = arr.reshape(shape)

        shape_array = ffi.new("int[]", shape)
        ndim = len(shape)
        data_ptr = ffi.cast("void*", arr.ctypes.data)

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
        return self.clone()

    def backward(self):
        lib.cml_backward(self._tensor, ffi.NULL, False, False)

    def __repr__(self) -> str:
        shape_str = str(self.shape) if self.shape else "()"
        dtype_str = DTYPE_NAMES.get(self.dtype, "unknown")
        device_str = DEVICE_NAMES.get(self.device, "unknown")
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={shape_str}, dtype={dtype_str}, device={device_str}{grad_str})"

    def __str__(self) -> str:
        try:
            arr = self.numpy()
            return f"Tensor({arr})"
        except Exception:
            return repr(self)

    def __len__(self) -> int:
        shape = self.shape
        if not shape:
            raise TypeError("len() of unsized tensor")
        return shape[0]

    def __getitem__(self, idx):
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
                return lib.tensor_get_float(self._tensor, idx)
            row_size = 1
            for d in range(1, len(shape)):
                row_size *= shape[d]
            flat_start = idx * row_size
            arr = self.numpy()[idx]
            return Tensor.from_numpy(arr)
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
            return lib.tensor_get_float(self._tensor, flat_idx)
        else:
            raise TypeError(
                f"indices must be integers or tuples, not {type(idx).__name__}"
            )

    def __setitem__(self, idx, value: float):
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
                row_size = 1
                for d in range(1, len(shape)):
                    row_size *= shape[d]
                flat_start = idx * row_size
                if isinstance(value, (int, float)):
                    for j in range(row_size):
                        lib.tensor_set_float(self._tensor, flat_start + j, float(value))
                elif hasattr(value, '_tensor'):
                    lib.tensor_ensure_executed(value._tensor)
                    for j in range(row_size):
                        v = lib.tensor_get_float(value._tensor, j)
                        lib.tensor_set_float(self._tensor, flat_start + j, v)
                else:
                    raise TypeError(f"Cannot set tensor elements from {type(value)}")
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
        neg_one = Tensor.full(self.shape, -1.0)
        return self * neg_one

    def __pos__(self) -> "Tensor":
        return self

    def __abs__(self) -> "Tensor":
        return self.relu() + (-self).relu()

    def __eq__(self, other) -> bool:
        if isinstance(other, Tensor):
            return self._tensor == other._tensor
        return False

    def __hash__(self) -> int:
        return hash(int(ffi.cast("uintptr_t", self._tensor)))

    def contiguous(self) -> "Tensor":
        if self.is_contiguous:
            return self
        return self.clone()

    def view(self, *new_shape) -> "Tensor":
        return self.reshape(*new_shape)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        shape = list(self.shape)
        if end_dim < 0:
            end_dim = len(shape) + end_dim
        if start_dim < 0:
            start_dim = len(shape) + start_dim

        flat_size = 1
        for i in range(start_dim, end_dim + 1):
            flat_size *= shape[i]

        new_shape = shape[:start_dim] + [flat_size] + shape[end_dim + 1 :]
        return self.reshape(new_shape)

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
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
        shape = list(self.shape)
        if dim < 0:
            dim = len(shape) + dim + 1
        new_shape = shape[:dim] + [1] + shape[dim:]
        return self.reshape(new_shape)

    @staticmethod
    def zeros(shape):
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
    def ones(shape):
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
        if isinstance(shape, int):
            shape = [shape]
        arr = np.random.rand(*shape).astype(np.float32)
        return Tensor.from_numpy(arr)

    @staticmethod
    def full(shape, value):
        if isinstance(shape, int):
            shape = [shape]
        arr = np.full(shape, float(value), dtype=np.float32)
        return Tensor.from_numpy(arr)

    @staticmethod
    def empty(shape):
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
    """Non-owning wrapper that won't free the underlying C tensor on deletion."""

    def __init__(self, c_tensor):
        self._tensor = c_tensor
        self._shape_cache = None

    def __del__(self):
        pass


class init_context:
    """RAII-style context manager for CML initialization and cleanup."""

    def __init__(self):
        self._initialized = False

    def __enter__(self):
        lib.cml_init()
        self._initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._initialized:
            lib.cml_cleanup()
        return False


class no_grad:
    """Context manager to disable gradient computation."""

    def __init__(self):
        self._prev_grad_enabled = True

    def __enter__(self):
        self._prev_grad_enabled = lib.cml_is_grad_enabled()
        lib.cml_no_grad()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prev_grad_enabled:
            lib.cml_enable_grad()
        return False


class enable_grad:
    """Context manager to force-enable gradient computation."""

    def __init__(self):
        self._prev_grad_enabled = True

    def __enter__(self):
        self._prev_grad_enabled = lib.cml_is_grad_enabled()
        lib.cml_enable_grad()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._prev_grad_enabled:
            lib.cml_no_grad()
        return False


class set_grad_enabled:
    """Context manager to explicitly set gradient computation on or off."""

    def __init__(self, mode: bool):
        self._mode = mode
        self._prev_grad_enabled = True

    def __enter__(self):
        self._prev_grad_enabled = lib.cml_is_grad_enabled()
        if self._mode:
            lib.cml_enable_grad()
        else:
            lib.cml_no_grad()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prev_grad_enabled:
            lib.cml_enable_grad()
        else:
            lib.cml_no_grad()
        return False


def is_grad_enabled() -> bool:
    return lib.cml_is_grad_enabled()


def set_device(device: int):
    try:
        lib.cml_set_default_device(device)
    except (AttributeError, Exception):
        if device != DEVICE_CPU:
            import warnings
            warnings.warn(f"Device {DEVICE_NAMES.get(device, device)} not available, using CPU")


def is_device_available(device: int) -> bool:
    if device == DEVICE_CPU:
        return True
    try:
        return bool(lib.cml_is_device_available(device))
    except (AttributeError, Exception):
        return False
