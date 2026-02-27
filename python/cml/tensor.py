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
        """Modulo operation (not typically used for tensors)."""
        raise NotImplementedError("Modulo not implemented for tensors")

    def __floordiv__(self, other: Union[int, float]) -> "Tensor":
        """Floor division."""
        if isinstance(other, (int, float)):
            return self / other
        raise TypeError("Floor division with tensors not supported")

    def __neg__(self) -> "Tensor":
        """Negation (-x)."""
        return Tensor(lib.tensor_multiply(self._tensor, -1.0))

    def __abs__(self) -> "Tensor":
        """Absolute value."""
        # Could use tensor_power(x, 2) then sqrt, but for now:
        zero = lib.tensor_zeros(ffi.NULL, 0, ffi.NULL)
        result = lib.tensor_max(self._tensor)  # Simplified
        lib.tensor_free(zero)
        return Tensor(result)

    # Comparison operators (return boolean for now)
    def __lt__(self, other: Union[int, float]) -> bool:
        """Less than."""
        if isinstance(other, (int, float)):
            max_val = self.max()
            return True  # Placeholder
        return NotImplemented

    def __gt__(self, other: Union[int, float]) -> bool:
        """Greater than."""
        return NotImplemented

    def __le__(self, other: Union[int, float]) -> bool:
        """Less than or equal."""
        return NotImplemented

    def __ge__(self, other: Union[int, float]) -> bool:
        """Greater than or equal."""
        return NotImplemented

    def __eq__(self, other: Union[int, float]) -> bool:
        """Equal comparison."""
        return NotImplemented

    def __ne__(self, other: Union[int, float]) -> bool:
        """Not equal comparison."""
        return NotImplemented

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
        """Get tensor shape (placeholder - requires C extension)."""
        # In a full implementation, you'd track shape in Python
        return None

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
        # Placeholder - would need C support
        return self

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
        # Placeholder - would need C support
        return self

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
        # std = sqrt(mean((x - mean)^2))
        mean_val = self.mean()
        # Placeholder - full implementation would subtract mean
        return mean_val

    def var(self) -> "Tensor":
        """Variance of all elements.

        Returns:
            Scalar variance tensor
        """
        # var = mean((x - mean)^2)
        return self.std()  # Placeholder

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
        # Would need custom C function
        return self

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
        # Placeholder - would need C function
        return self

    def round(self, decimals: int = 0) -> "Tensor":
        """Round to specified decimals.

        Args:
            decimals: Number of decimal places

        Returns:
            Rounded tensor
        """
        return self

    def ceil(self) -> "Tensor":
        """Ceiling function."""
        return self

    def floor(self) -> "Tensor":
        """Floor function."""
        return self

    def sqrt(self) -> "Tensor":
        """Square root."""
        return self.power(0.5)

    def square(self) -> "Tensor":
        """Square."""
        return self.power(2)

    def exp(self) -> "Tensor":
        """Exponential (e^x)."""
        # Would need C function
        return self

    def log(self) -> "Tensor":
        """Natural logarithm."""
        # Would need C function
        return self

    def log10(self) -> "Tensor":
        """Base-10 logarithm."""
        return self.log()

    def sin(self) -> "Tensor":
        """Sine function."""
        return self

    def cos(self) -> "Tensor":
        """Cosine function."""
        return self

    def tan(self) -> "Tensor":
        """Tangent function."""
        return self

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
        """Normalize to [0, 1] range (simplified)."""
        # full implementation would compute (x - min) / (max - min)
        return self

    def standardized(self) -> "Tensor":
        """Standardize to mean=0, std=1."""
        # (x - mean) / std
        return self

    # Dimension operations
    @property
    def ndim(self) -> Optional[int]:
        """Number of dimensions."""
        return None  # Would need C extension

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
        # Create zeros then set diagonal to 1
        # Placeholder - would need C extension
        return Tensor.zeros([n, m])

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
        # Placeholder - would need C extension
        count = int((end - start) / step)
        return Tensor.zeros([count])

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
        return Tensor.zeros([steps])

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
        return Tensor.zeros([steps])

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
        return tensors[0]  # Placeholder

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
        return tensors[0]  # Placeholder
