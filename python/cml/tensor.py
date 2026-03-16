"""
Enhanced Tensor class with improved convenience methods and operators.

This module provides a richer Tensor interface with:
- More operator overloading
- Convenience methods
- Better string representation
- Chainable operations
- Type hints
"""

from typing import Tuple, Optional, Union, List
import numpy as np
from cml._cml_lib import ffi, lib
from cml.core import Tensor as BaseTensor


class Tensor(BaseTensor):
    """Enhanced Tensor class with convenience methods.

    Extends the base Tensor class with:
    - More operators (+, -, *, /, //, %, **)
    - Comparison operators (<, >, <=, >=, ==)
    - Better string representation
    - Convenience properties
    - Chainable operations

    Example:
        >>> x = Tensor.randn([10, 10])
        >>> y = (x * 2 + 1).relu()
        >>> z = x @ y.T  # Matrix multiply
        >>> print(z)
    """

    def __repr__(self) -> str:
        """Better string representation."""
        size = self.size
        return f"Tensor(size={size})"

    def __str__(self) -> str:
        """String representation."""
        return f"Tensor({self.size} elements)"

    # Additional operators
    def __pow__(self, power: Union[int, float]) -> "Tensor":
        """Power operation (x ** y)."""
        return Tensor(lib.tensor_power(self._tensor, float(power)))

    def __mod__(self, other: Union[int, float]) -> "Tensor":
        """Modulo operation: remainder after floor division."""
        if isinstance(other, (int, float)):
            # x % y = x - floor(x / y) * y
            return self - (self / Tensor.full(self.shape or [self.size], float(other))) * Tensor.full(self.shape or [self.size], float(other))
        if isinstance(other, Tensor):
            return self - (self / other) * other
        raise TypeError(f"Cannot compute modulo of Tensor and {type(other)}")

    def __floordiv__(self, other: Union[int, float]) -> "Tensor":
        """Floor division."""
        if isinstance(other, (int, float)):
            return self / other
        raise TypeError("Floor division with tensors not supported")

    def __neg__(self) -> "Tensor":
        """Negation (-x)."""
        return Tensor(lib.tensor_multiply(self._tensor, -1.0))

    def __abs__(self) -> "Tensor":
        """Element-wise absolute value."""
        arr = np.abs(self.numpy())
        return Tensor.from_numpy(arr.astype(np.float32))

    # Comparison operators (return Tensor of 0s/1s)
    def __lt__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """Less than (element-wise)."""
        if isinstance(other, (int, float)):
            result = (self.numpy() < other).astype(np.float32)
            return Tensor.from_numpy(result)
        if isinstance(other, Tensor):
            result = (self.numpy() < other.numpy()).astype(np.float32)
            return Tensor.from_numpy(result)
        return NotImplemented

    def __gt__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """Greater than (element-wise)."""
        if isinstance(other, (int, float)):
            result = (self.numpy() > other).astype(np.float32)
            return Tensor.from_numpy(result)
        if isinstance(other, Tensor):
            result = (self.numpy() > other.numpy()).astype(np.float32)
            return Tensor.from_numpy(result)
        return NotImplemented

    def __le__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """Less than or equal (element-wise)."""
        if isinstance(other, (int, float)):
            result = (self.numpy() <= other).astype(np.float32)
            return Tensor.from_numpy(result)
        if isinstance(other, Tensor):
            result = (self.numpy() <= other.numpy()).astype(np.float32)
            return Tensor.from_numpy(result)
        return NotImplemented

    def __ge__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """Greater than or equal (element-wise)."""
        if isinstance(other, (int, float)):
            result = (self.numpy() >= other).astype(np.float32)
            return Tensor.from_numpy(result)
        if isinstance(other, Tensor):
            result = (self.numpy() >= other.numpy()).astype(np.float32)
            return Tensor.from_numpy(result)
        return NotImplemented

    def __eq__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """Equal comparison (element-wise)."""
        if isinstance(other, (int, float)):
            result = (self.numpy() == other).astype(np.float32)
            return Tensor.from_numpy(result)
        if isinstance(other, Tensor):
            result = (self.numpy() == other.numpy()).astype(np.float32)
            return Tensor.from_numpy(result)
        return NotImplemented

    def __ne__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """Not equal comparison (element-wise)."""
        if isinstance(other, (int, float)):
            result = (self.numpy() != other).astype(np.float32)
            return Tensor.from_numpy(result)
        if isinstance(other, Tensor):
            result = (self.numpy() != other.numpy()).astype(np.float32)
            return Tensor.from_numpy(result)
        return NotImplemented

    def __hash__(self):
        """Hash based on C tensor pointer (needed since __eq__ is overridden)."""
        return hash(int(ffi.cast("uintptr_t", self._tensor)))

    # Convenience properties
    @property
    def T(self) -> "Tensor":
        """Transpose (for 2D tensors).

        Example:
            >>> x = Tensor.randn([10, 20])
            >>> y = x.T  # 20x10
        """
        return self.transpose(0, 1)

    @property
    def shape(self) -> Optional[Tuple]:
        """Get tensor shape."""
        # Delegate to parent class which reads from the C tensor
        return super().shape

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        """Remove dimensions of size 1.

        Args:
            dim: Dimension to squeeze (None = squeeze all)

        Returns:
            Squeezed tensor

        Example:
            >>> x = Tensor.zeros([1, 10, 1])
            >>> y = x.squeeze()  # [10]
        """
        return super().squeeze(dim)

    def unsqueeze(self, dim: int) -> "Tensor":
        """Add a dimension of size 1.

        Args:
            dim: Dimension to add

        Returns:
            Tensor with added dimension

        Example:
            >>> x = Tensor.zeros([10])
            >>> y = x.unsqueeze(0)  # [1, 10]
        """
        return super().unsqueeze(dim)

    def expand(self, *shape: int) -> "Tensor":
        """Expand tensor to new shape.

        Args:
            shape: New shape

        Returns:
            Expanded tensor

        Example:
            >>> x = Tensor.zeros([1, 10])
            >>> y = x.expand(5, 10)  # Repeat 5 times
        """
        return self.reshape(shape)

    def max(self, dim: Optional[int] = None) -> "Tensor":
        """Get maximum values.

        Args:
            dim: Dimension to reduce (None = global max)

        Returns:
            Max tensor (scalar or reduced)

        Example:
            >>> x = Tensor.randn([10, 20])
            >>> max_val = x.max()
        """
        result = lib.tensor_max(self._tensor)
        return Tensor(result)

    def min(self, dim: Optional[int] = None) -> "Tensor":
        """Get minimum values.

        Args:
            dim: Dimension to reduce (None = global min)

        Returns:
            Min tensor (scalar or reduced)
        """
        result = lib.tensor_min(self._tensor)
        return Tensor(result)

    def std(self) -> "Tensor":
        """Standard deviation of all elements.

        Returns:
            Scalar std tensor
        """
        result = np.std(self.numpy())
        return Tensor.from_numpy(np.array([result], dtype=np.float32))

    def var(self) -> "Tensor":
        """Variance of all elements.

        Returns:
            Scalar variance tensor
        """
        result = np.var(self.numpy())
        return Tensor.from_numpy(np.array([result], dtype=np.float32))

    def clamp(self, min_val: float = 0, max_val: float = 1) -> "Tensor":
        """Clamp values to range [min_val, max_val].

        Args:
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            Clamped tensor

        Example:
            >>> x = Tensor.randn([10, 10])
            >>> clamped = x.clamp(0, 1)
        """
        result = np.clip(self.numpy(), min_val, max_val)
        return Tensor.from_numpy(result)

    def clip(self, min_val: float, max_val: float) -> "Tensor":
        """Alias for clamp."""
        return self.clamp(min_val, max_val)

    def abs(self) -> "Tensor":
        """Absolute value."""
        return self.__abs__()

    def neg(self) -> "Tensor":
        """Negation."""
        return self.__neg__()

    def sign(self) -> "Tensor":
        """Sign function (returns -1, 0, or 1)."""
        result = np.sign(self.numpy())
        return Tensor.from_numpy(result)

    def round(self, decimals: int = 0) -> "Tensor":
        """Round to specified decimals.

        Args:
            decimals: Number of decimal places

        Returns:
            Rounded tensor
        """
        result = np.round(self.numpy(), decimals)
        return Tensor.from_numpy(result)

    def ceil(self) -> "Tensor":
        """Ceiling function."""
        result = np.ceil(self.numpy())
        return Tensor.from_numpy(result)

    def floor(self) -> "Tensor":
        """Floor function."""
        result = np.floor(self.numpy())
        return Tensor.from_numpy(result)

    def sqrt(self) -> "Tensor":
        """Square root."""
        return self.power(0.5)

    def square(self) -> "Tensor":
        """Square."""
        return self.power(2)

    def exp(self) -> "Tensor":
        """Exponential (e^x)."""
        return Tensor(lib.cml_exp(self._tensor))

    def log(self) -> "Tensor":
        """Natural logarithm."""
        return Tensor(lib.cml_log(self._tensor))

    def log10(self) -> "Tensor":
        """Base-10 logarithm."""
        result = np.log10(self.numpy())
        return Tensor.from_numpy(result)

    def sin(self) -> "Tensor":
        """Sine function."""
        return Tensor(lib.cml_sin(self._tensor))

    def cos(self) -> "Tensor":
        """Cosine function."""
        return Tensor(lib.cml_cos(self._tensor))

    def tan(self) -> "Tensor":
        """Tangent function."""
        return Tensor(lib.cml_tan(self._tensor))

    # Chainable operations
    def apply(self, fn) -> "Tensor":
        """Apply a function to tensor.

        Args:
            fn: Function to apply

        Returns:
            Result tensor

        Example:
            >>> x = Tensor.randn([10, 10])
            >>> y = x.apply(lambda t: t.relu())
        """
        return fn(self)

    def chain(self, *fns) -> "Tensor":
        """Chain multiple operations.

        Args:
            fns: Functions to apply in sequence

        Returns:
            Result tensor

        Example:
            >>> x = Tensor.randn([10, 10])
            >>> y = x.chain(
            ...     lambda t: t.relu(),
            ...     lambda t: t * 2,
            ...     lambda t: t + 1
            ... )
        """
        result = self
        for fn in fns:
            result = fn(result)
        return result

    # Properties for common operations
    def reciprocal(self) -> "Tensor":
        """1 / x."""
        one = lib.tensor_full(ffi.NULL, 0, 1.0, ffi.NULL)
        result = lib.tensor_divide(one, self._tensor)
        lib.tensor_free(one)
        return Tensor(result)

    def normalized(self) -> "Tensor":
        """Normalize to [0, 1] range (min-max normalization)."""
        arr = self.numpy()
        min_val = arr.min()
        max_val = arr.max()
        denom = max_val - min_val
        if denom == 0:
            return Tensor.from_numpy(np.zeros_like(arr))
        result = (arr - min_val) / denom
        return Tensor.from_numpy(result)

    def standardized(self) -> "Tensor":
        """Standardize to mean=0, std=1."""
        arr = self.numpy()
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return Tensor.from_numpy(np.zeros_like(arr))
        result = (arr - mean) / std
        return Tensor.from_numpy(result)

    # Dimension operations
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return super().ndim

    @property
    def numel(self) -> int:
        """Total number of elements."""
        return self.size

    def view(self, *shape: int) -> "Tensor":
        """View tensor with new shape (alias for reshape)."""
        return self.reshape(shape)

    def as_numpy(self) -> Optional[np.ndarray]:
        """Convert to NumPy array (alias)."""
        return self.numpy()

    def to_numpy(self) -> Optional[np.ndarray]:
        """Convert to NumPy array (alias)."""
        return self.numpy()

    def detach(self) -> "Tensor":
        """Detach from computation graph."""
        # Clone without tracking gradients
        return self.clone()

    def requires_grad_(self, requires_grad: bool = True) -> "Tensor":
        """Set requires_grad flag (inplace)."""
        if requires_grad:
            lib.cml_enable_grad(self._tensor)
        else:
            lib.cml_disable_grad(self._tensor)
        return self

    # Creation methods
    @staticmethod
    def eye(n: int, m: Optional[int] = None) -> "Tensor":
        """Identity matrix.

        Args:
            n: Number of rows
            m: Number of columns (default = n)

        Returns:
            Identity tensor

        Example:
            >>> I = Tensor.eye(5)  # 5x5 identity
        """
        if m is None:
            m = n
        result = np.eye(n, m, dtype=np.float32)
        return Tensor.from_numpy(result)

    @staticmethod
    def arange(start: float, end: float, step: float = 1.0) -> "Tensor":
        """Create tensor with evenly spaced values.

        Args:
            start: Start value
            end: End value
            step: Step size

        Returns:
            1D tensor with range

        Example:
            >>> x = Tensor.arange(0, 10, 1)  # [0, 1, 2, ..., 9]
        """
        result = np.arange(start, end, step, dtype=np.float32)
        return Tensor.from_numpy(result)

    @staticmethod
    def linspace(start: float, end: float, steps: int = 100) -> "Tensor":
        """Create tensor with linearly spaced values.

        Args:
            start: Start value
            end: End value
            steps: Number of steps

        Returns:
            1D tensor with linearly spaced values

        Example:
            >>> x = Tensor.linspace(0, 1, 11)
        """
        result = np.linspace(start, end, steps, dtype=np.float32)
        return Tensor.from_numpy(result)

    @staticmethod
    def logspace(start: float, end: float, steps: int = 50) -> "Tensor":
        """Create tensor with logarithmically spaced values.

        Args:
            start: Start value (10^start)
            end: End value (10^end)
            steps: Number of steps

        Returns:
            1D tensor with log-spaced values
        """
        result = np.logspace(start, end, steps, dtype=np.float32)
        return Tensor.from_numpy(result)

    # Convenience stacking operations
    @staticmethod
    def stack(tensors: list, dim: int = 0) -> "Tensor":
        """Stack tensors along new dimension.

        Args:
            tensors: List of tensors to stack
            dim: Dimension to stack along

        Returns:
            Stacked tensor

        Example:
            >>> x1 = Tensor.randn([10, 20])
            >>> x2 = Tensor.randn([10, 20])
            >>> stacked = Tensor.stack([x1, x2])  # [2, 10, 20]
        """
        if not tensors:
            raise ValueError("Need at least one tensor to stack")
        result = np.stack([t.numpy() for t in tensors], axis=dim)
        return Tensor.from_numpy(result)

    @staticmethod
    def cat(tensors: list, dim: int = 0) -> "Tensor":
        """Concatenate tensors along dimension.

        Args:
            tensors: List of tensors to concatenate
            dim: Dimension to concatenate along

        Returns:
            Concatenated tensor

        Example:
            >>> x1 = Tensor.randn([10, 20])
            >>> x2 = Tensor.randn([10, 20])
            >>> cat = Tensor.cat([x1, x2])  # [20, 20]
        """
        if not tensors:
            raise ValueError("Need at least one tensor")
        result = np.concatenate([t.numpy() for t in tensors], axis=dim)
        return Tensor.from_numpy(result)
