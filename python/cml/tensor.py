"""Enhanced Tensor class with convenience methods and operators."""

from typing import Tuple, Optional, Union, List
import numpy as np
from cml._cml_lib import ffi, lib
from cml.core import Tensor as BaseTensor


class Tensor(BaseTensor):

    def __repr__(self) -> str:
        size = self.size
        return f"Tensor(size={size})"

    def __str__(self) -> str:
        return f"Tensor({self.size} elements)"

    def __pow__(self, power: Union[int, float]) -> "Tensor":
        return Tensor(lib.tensor_power(self._tensor, float(power)))

    def __mod__(self, other: Union[int, float]) -> "Tensor":
        if isinstance(other, (int, float)):
            # x % y = x - floor(x / y) * y
            return self - (self / Tensor.full(self.shape or [self.size], float(other))) * Tensor.full(self.shape or [self.size], float(other))
        if isinstance(other, Tensor):
            return self - (self / other) * other
        raise TypeError(f"Cannot compute modulo of Tensor and {type(other)}")

    def __floordiv__(self, other: Union[int, float]) -> "Tensor":
        if isinstance(other, (int, float)):
            return self / other
        raise TypeError("Floor division with tensors not supported")

    def __neg__(self) -> "Tensor":
        return Tensor(lib.tensor_multiply(self._tensor, -1.0))

    def __abs__(self) -> "Tensor":
        arr = np.abs(self.numpy())
        return Tensor.from_numpy(arr.astype(np.float32))

    def __lt__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            result = (self.numpy() < other).astype(np.float32)
            return Tensor.from_numpy(result)
        if isinstance(other, Tensor):
            result = (self.numpy() < other.numpy()).astype(np.float32)
            return Tensor.from_numpy(result)
        return NotImplemented

    def __gt__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            result = (self.numpy() > other).astype(np.float32)
            return Tensor.from_numpy(result)
        if isinstance(other, Tensor):
            result = (self.numpy() > other.numpy()).astype(np.float32)
            return Tensor.from_numpy(result)
        return NotImplemented

    def __le__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            result = (self.numpy() <= other).astype(np.float32)
            return Tensor.from_numpy(result)
        if isinstance(other, Tensor):
            result = (self.numpy() <= other.numpy()).astype(np.float32)
            return Tensor.from_numpy(result)
        return NotImplemented

    def __ge__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            result = (self.numpy() >= other).astype(np.float32)
            return Tensor.from_numpy(result)
        if isinstance(other, Tensor):
            result = (self.numpy() >= other.numpy()).astype(np.float32)
            return Tensor.from_numpy(result)
        return NotImplemented

    def __eq__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            result = (self.numpy() == other).astype(np.float32)
            return Tensor.from_numpy(result)
        if isinstance(other, Tensor):
            result = (self.numpy() == other.numpy()).astype(np.float32)
            return Tensor.from_numpy(result)
        return NotImplemented

    def __ne__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            result = (self.numpy() != other).astype(np.float32)
            return Tensor.from_numpy(result)
        if isinstance(other, Tensor):
            result = (self.numpy() != other.numpy()).astype(np.float32)
            return Tensor.from_numpy(result)
        return NotImplemented

    def __hash__(self):
        return hash(int(ffi.cast("uintptr_t", self._tensor)))

    @property
    def T(self) -> "Tensor":
        return self.transpose(0, 1)

    @property
    def shape(self) -> Optional[Tuple]:
        return super().shape

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        return super().squeeze(dim)

    def unsqueeze(self, dim: int) -> "Tensor":
        return super().unsqueeze(dim)

    def expand(self, *shape: int) -> "Tensor":
        return self.reshape(shape)

    def max(self, dim: Optional[int] = None) -> "Tensor":
        result = lib.tensor_max(self._tensor)
        return Tensor(result)

    def min(self, dim: Optional[int] = None) -> "Tensor":
        result = lib.tensor_min(self._tensor)
        return Tensor(result)

    def std(self) -> "Tensor":
        result = np.std(self.numpy())
        return Tensor.from_numpy(np.array([result], dtype=np.float32))

    def var(self) -> "Tensor":
        result = np.var(self.numpy())
        return Tensor.from_numpy(np.array([result], dtype=np.float32))

    def clamp(self, min_val: float = 0, max_val: float = 1) -> "Tensor":
        result = np.clip(self.numpy(), min_val, max_val)
        return Tensor.from_numpy(result)

    def clip(self, min_val: float, max_val: float) -> "Tensor":
        return self.clamp(min_val, max_val)

    def abs(self) -> "Tensor":
        return self.__abs__()

    def neg(self) -> "Tensor":
        return self.__neg__()

    def sign(self) -> "Tensor":
        result = np.sign(self.numpy())
        return Tensor.from_numpy(result)

    def round(self, decimals: int = 0) -> "Tensor":
        result = np.round(self.numpy(), decimals)
        return Tensor.from_numpy(result)

    def ceil(self) -> "Tensor":
        result = np.ceil(self.numpy())
        return Tensor.from_numpy(result)

    def floor(self) -> "Tensor":
        result = np.floor(self.numpy())
        return Tensor.from_numpy(result)

    def sqrt(self) -> "Tensor":
        return self.power(0.5)

    def square(self) -> "Tensor":
        return self.power(2)

    def exp(self) -> "Tensor":
        return Tensor(lib.cml_exp(self._tensor))

    def log(self) -> "Tensor":
        return Tensor(lib.cml_log(self._tensor))

    def log10(self) -> "Tensor":
        result = np.log10(self.numpy())
        return Tensor.from_numpy(result)

    def sin(self) -> "Tensor":
        return Tensor(lib.cml_sin(self._tensor))

    def cos(self) -> "Tensor":
        return Tensor(lib.cml_cos(self._tensor))

    def tan(self) -> "Tensor":
        return Tensor(lib.cml_tan(self._tensor))

    def apply(self, fn) -> "Tensor":
        return fn(self)

    def chain(self, *fns) -> "Tensor":
        result = self
        for fn in fns:
            result = fn(result)
        return result

    def reciprocal(self) -> "Tensor":
        one = lib.tensor_full(ffi.NULL, 0, 1.0, ffi.NULL)
        result = lib.tensor_divide(one, self._tensor)
        lib.tensor_free(one)
        return Tensor(result)

    def normalized(self) -> "Tensor":
        arr = self.numpy()
        min_val = arr.min()
        max_val = arr.max()
        denom = max_val - min_val
        if denom == 0:
            return Tensor.from_numpy(np.zeros_like(arr))
        result = (arr - min_val) / denom
        return Tensor.from_numpy(result)

    def standardized(self) -> "Tensor":
        arr = self.numpy()
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return Tensor.from_numpy(np.zeros_like(arr))
        result = (arr - mean) / std
        return Tensor.from_numpy(result)

    @property
    def ndim(self) -> int:
        return super().ndim

    @property
    def numel(self) -> int:
        return self.size

    def view(self, *shape: int) -> "Tensor":
        return self.reshape(shape)

    def as_numpy(self) -> Optional[np.ndarray]:
        return self.numpy()

    def to_numpy(self) -> Optional[np.ndarray]:
        return self.numpy()

    def detach(self) -> "Tensor":
        return self.clone()

    def requires_grad_(self, requires_grad: bool = True) -> "Tensor":
        if requires_grad:
            lib.cml_enable_grad(self._tensor)
        else:
            lib.cml_disable_grad(self._tensor)
        return self

    @staticmethod
    def eye(n: int, m: Optional[int] = None) -> "Tensor":
        """Create an identity matrix."""
        if m is None:
            m = n
        result = np.eye(n, m, dtype=np.float32)
        return Tensor.from_numpy(result)

    @staticmethod
    def arange(start: float, end: float, step: float = 1.0) -> "Tensor":
        """Create a 1D tensor with evenly spaced values."""
        result = np.arange(start, end, step, dtype=np.float32)
        return Tensor.from_numpy(result)

    @staticmethod
    def linspace(start: float, end: float, steps: int = 100) -> "Tensor":
        """Create a 1D tensor with linearly spaced values."""
        result = np.linspace(start, end, steps, dtype=np.float32)
        return Tensor.from_numpy(result)

    @staticmethod
    def logspace(start: float, end: float, steps: int = 50) -> "Tensor":
        """Create a 1D tensor with logarithmically spaced values."""
        result = np.logspace(start, end, steps, dtype=np.float32)
        return Tensor.from_numpy(result)

    @staticmethod
    def stack(tensors: list, dim: int = 0) -> "Tensor":
        """Stack tensors along a new dimension."""
        if not tensors:
            raise ValueError("Need at least one tensor to stack")
        result = np.stack([t.numpy() for t in tensors], axis=dim)
        return Tensor.from_numpy(result)

    @staticmethod
    def cat(tensors: list, dim: int = 0) -> "Tensor":
        """Concatenate tensors along an existing dimension."""
        if not tensors:
            raise ValueError("Need at least one tensor")
        result = np.concatenate([t.numpy() for t in tensors], axis=dim)
        return Tensor.from_numpy(result)
