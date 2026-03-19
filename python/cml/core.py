"""Core CML functionality."""

from __future__ import annotations
from typing import Union, Optional, Tuple
import operator
import numpy as np
from cml._cml_lib import ffi, lib

DEVICE_CPU = 0
DEVICE_CUDA = 1
DEVICE_METAL = 2
DEVICE_ROCM = 3
DEVICE_SIM_GPU = 4
DEVICE_AUTO = 5

DEVICE_NAMES = {
    DEVICE_CPU: "CPU",
    DEVICE_CUDA: "CUDA",
    DEVICE_METAL: "Metal",
    DEVICE_ROCM: "ROCm",
    DEVICE_SIM_GPU: "SimGPU",
    DEVICE_AUTO: "Auto",
}

DTYPE_FLOAT32 = 0
DTYPE_FLOAT64 = 1
DTYPE_INT32 = 2
DTYPE_INT64 = 3
DTYPE_BOOL = 4
DTYPE_FLOAT16 = 5
DTYPE_BFLOAT16 = 6
DTYPE_INT8 = 7
DTYPE_UINT8 = 8

DTYPE_NAMES = {
    DTYPE_FLOAT32: "float32",
    DTYPE_FLOAT64: "float64",
    DTYPE_INT32: "int32",
    DTYPE_INT64: "int64",
    DTYPE_BOOL: "bool",
    DTYPE_FLOAT16: "float16",
    DTYPE_BFLOAT16: "bfloat16",
    DTYPE_INT8: "int8",
    DTYPE_UINT8: "uint8",
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


def set_device(device: int):
    try:
        lib.cml_set_default_device(device)
    except (AttributeError, Exception):
        if device != DEVICE_CPU:
            import warnings
            warnings.warn(f"Device {DEVICE_NAMES.get(device, device)} not available, using CPU")


def get_dtype():
    return lib.cml_get_default_dtype()


def set_dtype(dtype):
    lib.cml_set_default_dtype(dtype)


def _make_config(dtype=None, device=None):
    config = ffi.new("TensorConfig*")
    if dtype is not None:
        config.dtype = dtype
        config.has_dtype = True
    else:
        config.dtype = lib.cml_get_default_dtype()
        config.has_dtype = True
    if device is not None:
        config.device = device
        config.has_device = True
    else:
        config.has_device = False
    return config


def _coerce_shape(shape):
    if isinstance(shape, int):
        shape = [shape]
    return list(shape)


def _create(lib_fn, shape, dtype, device, *extra_args):
    shape = _coerce_shape(shape)
    shape_array = ffi.new("int[]", shape)
    config = _make_config(dtype, device)
    args = [shape_array, len(shape), config] + list(extra_args)
    return Tensor(lib_fn(*args))


def _cmp(self, other, np_op):
    if isinstance(other, (int, float)):
        return Tensor.from_numpy(np_op(self.numpy(), other).astype(np.float32))
    if isinstance(other, Tensor):
        return Tensor.from_numpy(np_op(self.numpy(), other.numpy()).astype(np.float32))
    return NotImplemented


class Tensor:
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
            return Tensor(lib.cml_add(self._tensor, other._tensor))
        raise TypeError(f"Cannot add Tensor and {type(other)}")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(lib.cml_sub(self._tensor, other._tensor))
        raise TypeError(f"Cannot subtract Tensor and {type(other)}")

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(lib.cml_sub(other._tensor, self._tensor))
        raise TypeError(f"Cannot subtract {type(other)} and Tensor")

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(lib.cml_mul(self._tensor, other._tensor))
        raise TypeError(f"Cannot multiply Tensor and {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(lib.cml_div(self._tensor, other._tensor))
        raise TypeError(f"Cannot divide Tensor by {type(other)}")

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(lib.cml_matmul(self._tensor, other._tensor))
        raise TypeError(f"Cannot matmul Tensor and {type(other)}")

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
        return Tensor(lib.cml_reshape(self._tensor, shape_array, len(new_shape)))

    def view(self, *new_shape) -> "Tensor":
        return self.reshape(*new_shape)

    def transpose(self, dim0=0, dim1=1):
        return Tensor(lib.cml_transpose(self._tensor, dim0, dim1))

    @property
    def T(self) -> "Tensor":
        return self.transpose(0, 1)

    def sum(self, dim=-1, keepdim=False):
        return Tensor(lib.cml_sum(self._tensor, dim, keepdim))

    def mean(self, dim=-1, keepdim=False):
        return Tensor(lib.cml_mean(self._tensor, dim, keepdim))

    def max(self, dim=-1, keepdim=False):
        return Tensor(lib.cml_max(self._tensor, dim, keepdim))

    def min(self, dim=-1, keepdim=False):
        return Tensor(lib.cml_min(self._tensor, dim, keepdim))

    def prod(self, dim=-1, keepdim=False):
        return Tensor(lib.cml_prod(self._tensor, dim, keepdim))

    def argmax(self, dim=-1):
        return Tensor(lib.cml_argmax(self._tensor, dim))

    def argmin(self, dim=-1):
        return Tensor(lib.cml_argmin(self._tensor, dim))

    def var(self, dim=-1, unbiased=True, keepdim=False):
        return Tensor(lib.cml_var(self._tensor, dim, unbiased, keepdim))

    def std(self, dim=-1, unbiased=True, keepdim=False):
        return Tensor(lib.cml_std(self._tensor, dim, unbiased, keepdim))

    def softmax(self, dim=1):
        return Tensor(lib.cml_softmax(self._tensor, dim))

    def pow(self, other):
        if isinstance(other, Tensor):
            return Tensor(lib.cml_pow(self._tensor, other._tensor))
        raise TypeError(f"pow requires a Tensor exponent, got {type(other)}")

    def clamp(self, min_val, max_val):
        return Tensor(lib.cml_clamp(self._tensor, float(min_val), float(max_val)))

    def clone(self):
        return Tensor(lib.cml_clone(self._tensor))

    def detach(self):
        return Tensor(lib.cml_detach(self._tensor))

    def contiguous(self) -> "Tensor":
        if self.is_contiguous:
            return self
        return Tensor(lib.cml_contiguous(self._tensor))

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        if dim is None:
            dim = -1
        return Tensor(lib.cml_squeeze(self._tensor, dim))

    def unsqueeze(self, dim: int) -> "Tensor":
        return Tensor(lib.cml_unsqueeze(self._tensor, dim))

    def flip(self, dim: int) -> "Tensor":
        return Tensor(lib.cml_flip(self._tensor, dim))

    def sort(self, dim: int = -1, descending: bool = False) -> "Tensor":
        return Tensor(lib.cml_sort(self._tensor, dim, descending))

    def cast(self, dtype: int) -> "Tensor":
        return Tensor(lib.cml_cast(self._tensor, dtype))

    def dot(self, other: "Tensor") -> "Tensor":
        return Tensor(lib.cml_dot(self._tensor, other._tensor))

    def matmul(self, other: "Tensor") -> "Tensor":
        return Tensor(lib.cml_matmul(self._tensor, other._tensor))

    def relu(self): return Tensor(lib.cml_relu(self._tensor))
    def sigmoid(self): return Tensor(lib.cml_sigmoid(self._tensor))
    def tanh(self): return Tensor(lib.cml_tanh(self._tensor))
    def exp(self): return Tensor(lib.cml_exp(self._tensor))
    def log(self): return Tensor(lib.cml_log(self._tensor))
    def sqrt(self): return Tensor(lib.cml_sqrt(self._tensor))
    def sin(self): return Tensor(lib.cml_sin(self._tensor))
    def cos(self): return Tensor(lib.cml_cos(self._tensor))
    def log2(self) -> "Tensor": return Tensor(lib.cml_log2(self._tensor))
    def tan(self) -> "Tensor": return Tensor(lib.cml_tan(self._tensor))
    def asin(self) -> "Tensor": return Tensor(lib.cml_asin(self._tensor))
    def acos(self) -> "Tensor": return Tensor(lib.cml_acos(self._tensor))
    def atan(self) -> "Tensor": return Tensor(lib.cml_atan(self._tensor))
    def rsqrt(self) -> "Tensor": return Tensor(lib.cml_rsqrt(self._tensor))
    def erf(self) -> "Tensor": return Tensor(lib.cml_erf(self._tensor))
    def exp2(self) -> "Tensor": return Tensor(lib.cml_exp2(self._tensor))
    def sign(self) -> "Tensor": return Tensor(lib.cml_sign(self._tensor))
    def ceil(self) -> "Tensor": return Tensor(lib.cml_ceil(self._tensor))
    def floor(self) -> "Tensor": return Tensor(lib.cml_floor(self._tensor))
    def square(self) -> "Tensor": return Tensor(lib.cml_square(self._tensor))
    def round(self, decimals: int = 0) -> "Tensor": return Tensor(lib.cml_round(self._tensor))

    def log10(self) -> "Tensor":
        return Tensor.from_numpy(np.log10(self.numpy()))

    def reciprocal(self) -> "Tensor":
        ones = Tensor.full(self.shape or [self.size], 1.0)
        return Tensor(lib.cml_div(ones._tensor, self._tensor))

    def numpy(self) -> np.ndarray:
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
        data_ptr = ffi.cast("void*", arr.ctypes.data)
        c_tensor = lib.tensor_from_data(data_ptr, shape_array, len(shape), ffi.NULL)
        if c_tensor == ffi.NULL:
            raise RuntimeError("Failed to create tensor from numpy array")

        tensor = cls(c_tensor)
        if requires_grad:
            tensor.requires_grad_(True)
        return tensor

    def to(self, device: Union[int, str] = None, dtype: Union[int, str] = None) -> "Tensor":
        if dtype is not None:
            return self.cast(dtype)
        return self.clone()

    def backward(self, gradient=None, retain_graph=False, create_graph=False):
        grad_ptr = gradient._tensor if gradient is not None else ffi.NULL
        lib.cml_backward(self._tensor, grad_ptr, retain_graph, create_graph)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        shape = list(self.shape)
        if end_dim < 0:
            end_dim = len(shape) + end_dim
        if start_dim < 0:
            start_dim = len(shape) + start_dim
        flat_size = 1
        for i in range(start_dim, end_dim + 1):
            flat_size *= shape[i]
        new_shape = shape[:start_dim] + [flat_size] + shape[end_dim + 1:]
        return self.reshape(new_shape)

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

    def __lt__(self, other): return _cmp(self, other, operator.lt)
    def __gt__(self, other): return _cmp(self, other, operator.gt)
    def __le__(self, other): return _cmp(self, other, operator.le)
    def __ge__(self, other): return _cmp(self, other, operator.ge)
    def __ne__(self, other): return _cmp(self, other, operator.ne)

    def __pow__(self, power: Union[int, float]) -> "Tensor":
        power_tensor = Tensor.full(self.shape or [self.size], float(power))
        return Tensor(lib.cml_pow(self._tensor, power_tensor._tensor))

    def __mod__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            other_t = Tensor.full(self.shape or [self.size], float(other))
            return Tensor(lib.cml_mod(self._tensor, other_t._tensor))
        if isinstance(other, Tensor):
            return Tensor(lib.cml_mod(self._tensor, other._tensor))
        raise TypeError(f"Cannot compute modulo of Tensor and {type(other)}")

    def __floordiv__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            other_t = Tensor.full(self.shape or [self.size], float(other))
            result = lib.cml_div(self._tensor, other_t._tensor)
            return Tensor(lib.cml_floor(result))
        if isinstance(other, Tensor):
            result = lib.cml_div(self._tensor, other._tensor)
            return Tensor(lib.cml_floor(result))
        raise TypeError("Floor division with tensors not supported")

    def __repr__(self) -> str:
        shape_str = str(self.shape) if self.shape else "()"
        dtype_str = DTYPE_NAMES.get(self.dtype, "unknown")
        device_str = DEVICE_NAMES.get(self.device, "unknown")
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={shape_str}, dtype={dtype_str}, device={device_str}{grad_str})"

    def __str__(self) -> str:
        try:
            return f"Tensor({self.numpy()})"
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
            return Tensor.from_numpy(self.numpy()[idx])
        elif isinstance(idx, tuple):
            shape = self.shape
            if len(idx) != len(shape):
                raise IndexError(f"too many indices for tensor of dimension {len(shape)}")
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
            raise TypeError(f"indices must be integers or tuples, not {type(idx).__name__}")

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
                raise IndexError(f"too many indices for tensor of dimension {len(shape)}")
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
            raise TypeError(f"indices must be integers or tuples, not {type(idx).__name__}")

    @staticmethod
    def zeros(shape, dtype=None, device=None):
        return _create(lib.cml_zeros, shape, dtype, device)

    @staticmethod
    def ones(shape, dtype=None, device=None):
        return _create(lib.cml_ones, shape, dtype, device)

    @staticmethod
    def randn(shape, dtype=None, device=None):
        return _create(lib.cml_randn, shape, dtype, device)

    @staticmethod
    def rand(shape, dtype=None, device=None):
        return _create(lib.cml_rand, shape, dtype, device)

    @staticmethod
    def full(shape, value, dtype=None, device=None):
        return _create(lib.cml_full, shape, dtype, device, float(value))

    @staticmethod
    def empty(shape, dtype=None, device=None):
        return _create(lib.cml_empty, shape, dtype, device)

    @staticmethod
    def logspace(start: float, end: float, steps: int = 50) -> "Tensor":
        return Tensor.from_numpy(np.logspace(start, end, steps, dtype=np.float32))

    @staticmethod
    def stack(tensors: list, dim: int = 0) -> "Tensor":
        if not tensors:
            raise ValueError("Need at least one tensor to stack")
        c_tensors = ffi.new("Tensor*[]", [t._tensor for t in tensors])
        return Tensor(lib.cml_stack(c_tensors, len(tensors), dim))

    @staticmethod
    def cat(tensors: list, dim: int = 0) -> "Tensor":
        if not tensors:
            raise ValueError("Need at least one tensor")
        c_tensors = ffi.new("Tensor*[]", [t._tensor for t in tensors])
        return Tensor(lib.cml_concat(c_tensors, len(tensors), dim))

    @staticmethod
    def eye(n: int, m: Optional[int] = None, dtype=None, device=None) -> "Tensor":
        config = _make_config(dtype, device)
        return Tensor(lib.cml_eye(n, config))

    @staticmethod
    def arange(start: float, end: float = None, step: float = 1.0, dtype=None, device=None) -> "Tensor":
        if end is None:
            end = float(start)
            start = 0.0
        config = _make_config(dtype, device)
        return Tensor(lib.cml_arange(float(start), float(end), float(step), config))

    @staticmethod
    def linspace(start: float, end: float, steps: int = 100, dtype=None, device=None) -> "Tensor":
        config = _make_config(dtype, device)
        return Tensor(lib.cml_linspace(float(start), float(end), int(steps), config))


class _TensorView(Tensor):
    """Non-owning view; does not free the underlying C tensor."""

    def __init__(self, c_tensor):
        self._tensor = c_tensor
        self._shape_cache = None

    def __del__(self):
        pass


class init_context:
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


class set_grad_enabled:
    def __init__(self, mode: bool):
        self._mode = mode
        self._prev = True

    def __enter__(self):
        self._prev = lib.cml_is_grad_enabled()
        if self._mode:
            lib.cml_enable_grad()
        else:
            lib.cml_no_grad()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prev:
            lib.cml_enable_grad()
        else:
            lib.cml_no_grad()
        return False


class no_grad(set_grad_enabled):
    def __init__(self):
        super().__init__(False)


class enable_grad(set_grad_enabled):
    def __init__(self):
        super().__init__(True)


def is_grad_enabled() -> bool:
    return lib.cml_is_grad_enabled()


def is_device_available(device: int) -> bool:
    if device == DEVICE_CPU:
        return True
    try:
        if device == DEVICE_CUDA:
            return bool(lib.device_cuda_available())
        elif device == DEVICE_METAL:
            return bool(lib.device_metal_available())
        elif device == DEVICE_ROCM:
            return bool(lib.device_rocm_available())
    except (AttributeError, Exception):
        pass
    return False
